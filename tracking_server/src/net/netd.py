"""lwir_netd: the small daemon behind offline / direct-cable access. Stdlib only, no root needed, one instance per board.

    python3 -B -m src.net.netd            (from tracking_server/; started by scripts/run_real_gui.sh and watched by keepalive.sh)

Jobs (each one is independent, a failure of one never stops the others):
  1. supervisor : every 5 s internet check (+ Tailscale state) -> /dev/shm/lwir_net_state.json  (mode online | lan-only)
  2. link watch : carrier and addresses of the direct port; a re-plugged cable clears the captive-portal client table
  3. portal     : HTTP responder for the OS connectivity checks on <direct address>:8091 (port 80 is redirected here by nft)
  4. beacon     : UDP broadcast every 2 s on the direct port, answers LWIR_DISCOVER
  5. mDNS       : orangepi.local / <hostname>.local on the direct port
The web page itself is served by the tracking server (src/web/direct_listeners.py).
"""
import fcntl
import ipaddress
import os
import signal
import socket
import sys
import time

from . import discovery, portal, supervisor
from .netutil import arp_table, carrier, iface_addrs, parse_specs

DEFAULT_ADDRS = '10.42.0.1/24,169.254.42.1/16,192.168.50.2/24,192.168.77.2/24'


class Config:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.iface = env.get('LWIR_DIRECT_IFACE', 'enP4p65s0')
        self.specs = parse_specs(env.get('LWIR_LIVE_EXTRA_BINDS', DEFAULT_ADDRS) or DEFAULT_ADDRS)
        self.portal_addr = env.get('LWIR_PORTAL_ADDR', '10.42.0.1')
        self.portal_port = int(env.get('LWIR_PORTAL_PORT', '8091'))
        self.page_port = int(env.get('LWIR_LIVE_PORT', '8090'))
        self.beacon_port = int(env.get('LWIR_BEACON_PORT', str(discovery.BEACON_PORT)))
        self.state_path = env.get('LWIR_NET_STATE', '/dev/shm/lwir_net_state.json')
        self.lock_path = env.get('LWIR_NETD_LOCK', os.path.expanduser('~/.lwir_netd.lock'))
        self.silence_s = float(env.get('LWIR_PORTAL_SILENCE_S', '600'))
        self.apple_silence = env.get('LWIR_PORTAL_APPLE_SILENCE', '0') == '1'
        self.check_s = float(env.get('LWIR_NET_CHECK_S', '5'))
        hostname = socket.gethostname().split('.')[0]
        self.hostname = hostname
        names = env.get('LWIR_MDNS_NAMES', 'orangepi,%s' % hostname)
        self.names = [n for n in names.replace(' ', '').split(',') if n]
        self.nets = [net for _, net in self.specs]
        self.start_mdns = env.get('LWIR_MDNS', '1') != '0'
        self.portal_off_file = env.get('LWIR_PORTAL_OFF_FILE', os.path.expanduser('~/.lwir_portal_off'))
        self.portal_mode_off = env.get('LWIR_PORTAL_MODE', 'on').lower() == 'off'
        self.tailscale_cache = env.get('LWIR_TAILSCALE_CACHE', os.path.expanduser('~/.lwir_tailscale_ip'))


def log(message):
    print('%s %s' % (time.strftime('%F %T'), message), flush=True)


class Netd:
    def __init__(self, cfg, sup=None):
        self.cfg = cfg
        self.sup = sup or supervisor.Supervisor()
        self.table = portal.ClientTable(cfg.silence_s)
        self.portal = self.discover = self.mdns = None
        self.link = None
        self.stop = False
        self.started = time.monotonic()
        self.last_beacon = self.last_check = self.last_arp = self.last_tailscale = 0.0

    # ---- services (each start is retried by the loop until it works)
    def ensure_services(self):
        cfg = self.cfg
        if self.portal is None:
            try:
                self.portal = portal.start_portal(cfg.portal_addr, cfg.portal_port, self.table, cfg.page_port,
                                                 allowed_net=cfg.nets, apple_silence=cfg.apple_silence, log=log, is_off=self.portal_is_off)
                log('portal listening on %s:%d' % (cfg.portal_addr, cfg.portal_port))
            except OSError as exc:
                self.portal = None
                log('portal not started: %s' % exc)
        if self.discover is None:
            try:
                self.discover = discovery.DiscoverResponder(self.payload_for, cfg.beacon_port)
                self.discover.start()
                log('discover responder on udp/%d' % cfg.beacon_port)
            except OSError as exc:
                self.discover = None
                log('discover responder not started: %s' % exc)
        if self.mdns is None and cfg.start_mdns:
            try:
                self.mdns = discovery.MdnsResponder(cfg.names, cfg.iface, cfg.nets, log=log)
                self.mdns.start()
                log('mdns responder for %s' % ', '.join(sorted(self.mdns.names)))
            except OSError as exc:
                self.mdns = None
                log('mdns responder not started: %s' % exc)
        if self.mdns is not None:
            self.mdns.join()

    def portal_is_off(self):
        return self.cfg.portal_mode_off or os.path.exists(self.cfg.portal_off_file)

    def payload_for(self, address):
        mode = self.sup.mode
        return discovery.beacon_payload(self.cfg.hostname, address, self.cfg.page_port, '/', mode)

    # ---- link watch
    def watch_link(self):
        now = carrier(self.cfg.iface)
        if now != self.link:
            log('link %s: carrier %s -> %s' % (self.cfg.iface, self.link, now))
            if now:   # cable (re)plugged: a new session, show the portal again
                self.table.reset()
                if self.mdns is not None:
                    self.mdns.rejoin()
            self.link = now

    def direct_interfaces(self):
        wanted = {ipaddress.ip_address(a) for a, _ in self.cfg.specs}
        return [i for i in iface_addrs(self.cfg.iface) if i.ip in wanted]

    def state(self):
        addrs = [str(i.ip) for i in iface_addrs(self.cfg.iface)]
        state = self.sup.state()
        state.update(direct=dict(iface=self.cfg.iface, carrier=self.link, addresses=addrs),
                     clients=len(self.table.snapshot()), updated=round(time.time()), up=round(time.monotonic() - self.started),
                     pid=os.getpid(), version=1)
        return state

    def cache_tailscale_address(self):
        """Remember the board's Tailscale address: the server binds it (IP_FREEBIND) even when it starts while Tailscale is away."""
        found = [str(i.ip) for i in iface_addrs('tailscale0')]
        if not found:
            return
        try:
            with open(self.cfg.tailscale_cache) as handle:
                if handle.read().strip() == found[0]:
                    return
        except OSError:
            pass
        try:
            with open(self.cfg.tailscale_cache, 'w') as handle:
                handle.write(found[0] + '\n')
        except OSError as exc:
            log('tailscale address not cached: %s' % exc)

    # ---- main loop
    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        self.ensure_services()
        self.watch_link()
        if now - self.last_arp >= 2.0:
            self.last_arp = now
            arp = {ip: mac for ip, mac in arp_table().items() if any(ipaddress.ip_address(ip) in n for n in self.cfg.nets)}
            for ip in self.table.neighbours(arp):
                log('client %s changed its MAC address: portal shown again' % ip)
        if now - self.last_beacon >= 2.0:
            self.last_beacon = now
            discovery.send_beacons(self.direct_interfaces(), self.payload_for, self.cfg.beacon_port)
        if now - self.last_check >= self.cfg.check_s:
            self.last_check = now
            check_ts = now - self.last_tailscale >= 15.0   # `tailscale status` is a process: every 15 s is enough
            if check_ts:
                self.last_tailscale = now
            changed = self.sup.tick(check_tailscale=check_ts)
            if check_ts:
                self.cache_tailscale_address()
            if changed:
                log('network mode: %s (internet=%s tailscale=%s)' % (self.sup.mode, self.sup.internet, self.sup.tailscale_state))
            try:
                supervisor.write_state(self.cfg.state_path, self.state())
            except OSError as exc:
                log('state file not written: %s' % exc)

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop', True))
        log('lwir_netd start pid=%d iface=%s addrs=%s' % (os.getpid(), self.cfg.iface, ','.join(a for a, _ in self.cfg.specs)))
        while not self.stop:
            try:
                self.tick()
            except Exception as exc:   # a bug in one job must not end the daemon
                log('tick error: %r' % (exc,))
            time.sleep(1.0)
        log('lwir_netd stop')
        try:
            os.unlink(self.cfg.state_path)
        except OSError:
            pass


def single_instance(path):
    handle = open(path, 'w')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None
    return handle


def main(argv=None):
    cfg = Config()
    if '--print-config' in (argv or sys.argv[1:]):
        print(dict(iface=cfg.iface, specs=[a for a, _ in cfg.specs], names=cfg.names, portal=(cfg.portal_addr, cfg.portal_port)))
        return 0
    lock = single_instance(cfg.lock_path)
    if lock is None:
        return 0   # another lwir_netd runs
    Netd(cfg).run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
