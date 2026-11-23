"""Small Linux network helpers shared by the web listeners and the lwir_netd daemon (stdlib only)."""
import http.server
import ipaddress
import re
import socket
import socketserver
import subprocess

IP_FREEBIND = 15   # Linux IP_FREEBIND: bind() to an address that is not (yet) assigned to this host

# The only networks a direct-cable / offline listener may serve: private ranges and link-local.
# Same ranges as page_handler.lan_ipv4 (explicit, not ipaddress's broader is_private).
LAN_NETWORKS = tuple(ipaddress.ip_network(n) for n in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '169.254.0.0/16'))


def is_lan_address(value):
    """True for an RFC1918 or link-local IPv4 address (not Tailscale 100.64/10, not public)."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.version == 4 and any(address in net for net in LAN_NETWORKS)


def is_local_client(value):
    """LAN address or loopback: a client that can use the PC connect program (not a Tailscale or public client)."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.is_loopback or is_lan_address(value)


def parse_specs(text):
    """'10.42.0.1/24,169.254.42.1/16' -> [(address, IPv4Network)]. Raises ValueError on anything outside LAN_NETWORKS."""
    specs = []
    for item in re.split(r'[,\s]+', (text or '').strip()):
        if not item:
            continue
        iface = ipaddress.ip_interface(item)
        if iface.version != 4 or not is_lan_address(str(iface.ip)):
            raise ValueError('listener address must be RFC1918 or link-local IPv4: %s' % item)
        if str(iface.ip) not in [a for a, _ in specs]:
            specs.append((str(iface.ip), iface.network))
    return specs


def make_freebind_server(base, allowed_net=None):
    """HTTP server class bound with IP_FREEBIND (bind never fails while the cable is out) that only serves allowed_net.

    base is http.server.ThreadingHTTPServer or a compatible class. server_bind skips getfqdn(): the reverse lookup waits
    for DNS timeouts when the internet is gone.
    """
    class FreebindServer(base):
        daemon_threads = True
        allow_reuse_address = True

        def server_bind(self):
            try:
                self.socket.setsockopt(socket.SOL_IP, IP_FREEBIND, 1)
            except OSError:
                pass
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = self.server_address[:2]

        def verify_request(self, request, client_address):
            if allowed_net is None:
                return True
            nets = tuple(allowed_net) if isinstance(allowed_net, (list, tuple, set)) else (allowed_net,)
            try:
                return any(ipaddress.ip_address(client_address[0]) in net for net in nets)
            except ValueError:
                return False
    return FreebindServer


def make_default_server():
    return make_freebind_server(http.server.ThreadingHTTPServer)


def address_assigned(address):
    """True when this host owns the address right now (a plain UDP bind; sends nothing)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind((address, 0))
        return True
    except OSError:
        return False


def run(cmd, timeout=3.0):
    """stdout of a command, '' on any failure."""
    try:
        done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout, check=False)
        return done.stdout.decode('utf-8', 'replace')
    except (OSError, subprocess.SubprocessError):
        return ''


def iface_addrs(iface, ip_command='ip'):
    """IPv4 addresses (IPv4Interface list) on one interface, from `ip -4 -o addr show dev IFACE`."""
    found = []
    for line in run([ip_command, '-4', '-o', 'addr', 'show', 'dev', iface]).splitlines():
        parts = line.split()
        if 'inet' in parts:
            try:
                found.append(ipaddress.ip_interface(parts[parts.index('inet') + 1]))
            except (ValueError, IndexError):
                pass
    return found


def carrier(iface):
    """True / False from /sys/class/net/IFACE/carrier, None when the interface does not exist."""
    try:
        with open('/sys/class/net/%s/carrier' % iface) as handle:
            return handle.read().strip() == '1'
    except OSError:
        # reading carrier of an interface that is administratively down raises EINVAL: no link
        try:
            with open('/sys/class/net/%s/operstate' % iface):
                return False
        except OSError:
            return None


def arp_table(path='/proc/net/arp'):
    """{ip: mac} of complete neighbour entries."""
    table = {}
    try:
        with open(path) as handle:
            next(handle, None)
            for line in handle:
                parts = line.split()
                if len(parts) >= 6 and parts[2] != '0x0' and parts[3] != '00:00:00:00:00:00':
                    table[parts[0]] = parts[3].lower()
    except OSError:
        pass
    return table


def source_for(destination):
    """Local address the kernel would use to reach destination (UDP connect sends nothing), or None."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((destination, 9))
            return probe.getsockname()[0]
    except OSError:
        return None


def tcp_ok(host, port, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
