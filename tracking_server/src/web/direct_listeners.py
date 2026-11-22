"""Extra web listeners for offline use (2026-11-22): the direct cable port and its fallbacks, plus the PC connect download.

Stdlib only; no camera, NPU, serial or Tailscale access. The same handler as the main page serves them.

* Every listener is bound with IP_FREEBIND, so starting never fails while the cable is out or the address is not assigned yet,
  and it begins to answer by itself when NetworkManager assigns the address (also after a cable is re-plugged).
* A listener only serves clients inside its own network (10.42.0.1 -> 10.42.0.0/24, ...): a packet that somehow arrives from
  elsewhere is dropped. Nothing listens on 0.0.0.0. Tailscale stays the primary listener (unchanged).
* GET /netinfo (any client): {mode, direct_urls, client_local, pc_download}; GET /pc/<file> (LAN clients only): the PC
  connect program (pc_connect/lwir_connect.bat|ps1|sh and a zip of them).
"""
import http.server
import io
import ipaddress
import json
import os
import socket
import threading
import time
import zipfile
from pathlib import Path

from ..net.netutil import address_assigned, is_lan_address, is_local_client, make_freebind_server, parse_specs

DEFAULT_SPECS = '10.42.0.1/24,169.254.42.1/16,192.168.50.2/24,192.168.77.2/24'
HERE = Path(__file__).resolve().parents[2]            # tracking_server/
PC_DIR = HERE / 'pc_connect'
PC_FILES = ('lwir_connect.bat', 'lwir_connect.ps1', 'lwir_connect.sh')
PC_TYPES = {'.bat': 'application/octet-stream', '.ps1': 'application/octet-stream', '.sh': 'text/x-shellscript', '.zip': 'application/zip'}
STATE_PATH = os.environ.get('LWIR_NET_STATE', '/dev/shm/lwir_net_state.json')
STATE_MAX_AGE_S = 30

TAILSCALE_NET = ipaddress.ip_network('100.64.0.0/10')
TAILSCALE_CACHE = os.environ.get('LWIR_TAILSCALE_CACHE', os.path.expanduser('~/.lwir_tailscale_ip'))

_listeners = []


def cached_tailscale_address(path=None):
    """The board's last known Tailscale address (written by lwir_netd), or None."""
    try:
        with open(path or TAILSCALE_CACHE) as handle:
            address = ipaddress.ip_address(handle.read().strip())
    except (OSError, ValueError):
        return None
    return str(address) if address.version == 4 and address in TAILSCALE_NET else None


def start(handler, port, specs_text=None, skip=(), log=print):
    """Open one listener per address in specs_text ('addr/prefix,...'; '' or 'off' = none). Returns the started addresses."""
    stop()
    text = DEFAULT_SPECS if specs_text is None else specs_text
    if text.strip().lower() in ('', 'off', 'none'):
        return []
    try:
        specs = parse_specs(text)
    except ValueError as exc:
        log('LIVE no direct listeners (%s)' % exc)
        return []
    tailscale = cached_tailscale_address()
    if tailscale:   # the server may have started while Tailscale was away (no internet): bind its address anyway, it answers when it returns
        specs.append((tailscale, TAILSCALE_NET))
    server_base = http.server.ThreadingHTTPServer
    started = []
    for address, net in specs:
        if address in skip:
            continue
        try:
            server = make_freebind_server(server_base, net)((address, port), handler)
            thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=0.5), daemon=True, name='direct-%s' % address)
            thread.start()
        except (OSError, RuntimeError) as exc:
            log('LIVE no listener on %s:%d (%s)' % (address, port, exc))
            continue
        _listeners.append(dict(address=address, port=port, server=server, thread=thread))
        started.append(address)
    if started:
        log('LIVE direct listeners (serve their own network only): %s' % ', '.join('%s:%d' % (a, port) for a in started))
    return started


def stop():
    while _listeners:
        item = _listeners.pop()
        try:
            item['server'].shutdown()
            item['server'].server_close()
        except OSError:
            pass


def assigned_urls():
    """URLs of the listeners whose address this host owns right now (the PC can reach those)."""
    return ['http://%s:%d/' % (item['address'], item['port']) for item in _listeners
            if is_lan_address(item['address']) and item['thread'].is_alive() and address_assigned(item['address'])]


def read_state(path=None):
    try:
        with open(path or STATE_PATH) as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) and time.time() - state.get('updated', 0) < STATE_MAX_AGE_S else {}


def pc_files_present():
    return all((PC_DIR / name).is_file() for name in PC_FILES[:2])


def netinfo(client_ip, state=None):
    state = read_state() if state is None else state
    local = is_local_client(client_ip)
    return dict(mode=state.get('mode'), internet=state.get('internet'), tailscale=state.get('tailscale'),
                direct_urls=assigned_urls(), client_local=local, pc_download=bool(local and pc_files_present()),
                name=socket.gethostname().split('.')[0])


def netinfo_body(client_ip):
    return json.dumps(netinfo(client_ip)).encode()


def pc_download(path, client_ip):
    """(status, content type, file name, body). Only LAN clients; only the files named above."""
    name = path.rsplit('/', 1)[-1]
    if not is_local_client(client_ip):
        return 404, None, None, b''
    if name == 'lwir_connect.zip':
        if not pc_files_present():
            return 404, None, None, b''
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for member in PC_FILES:
                source = PC_DIR / member
                if source.is_file():
                    info = zipfile.ZipInfo(member, date_time=time.gmtime(source.stat().st_mtime)[:6])
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (0o755 if member.endswith('.sh') else 0o644) << 16
                    archive.writestr(info, source.read_bytes())
        return 200, PC_TYPES['.zip'], name, buffer.getvalue()
    if name in PC_FILES and (PC_DIR / name).is_file():
        return 200, PC_TYPES[os.path.splitext(name)[1]], name, (PC_DIR / name).read_bytes()
    return 404, None, None, b''
