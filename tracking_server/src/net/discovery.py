"""Discovery of the board by a PC that has no address to type: UDP beacon, UDP discover reply, and a tiny mDNS responder.

Beacon  : every 2 s a JSON datagram {"lwir": 1, "name", "addr", "port", "path", ...} to the broadcast address of the direct
          interface's networks (UDP BEACON_PORT).
Discover: a PC may send the text 'LWIR_DISCOVER' (broadcast or unicast, UDP BEACON_PORT); the board answers the sender
          with the same JSON (a PC firewall lets the answer to its own request in; it often blocks unsolicited beacons).
mDNS    : answers A (and a negative AAAA) for orangepi.local and <hostname>.local on the direct interface so that
          http://orangepi.local:8090/ works from Windows 10/11 and macOS without installing avahi.
"""
import errno
import ipaddress
import json
import socket
import struct
import threading

from .netutil import is_local_client, source_for

BEACON_PORT = 47890
DISCOVER_TEXT = b'LWIR_DISCOVER'
MDNS_GROUP, MDNS_PORT = '224.0.0.251', 5353


def beacon_payload(name, addr, port=8090, path='/', mode=None):
    return json.dumps(dict(lwir=1, name=name, addr=addr, port=port, path=path, mode=mode), sort_keys=True).encode()


def broadcast_targets(interfaces):
    """[(local address, broadcast address)] for IPv4Interface objects (a /32 or /31 has no broadcast: skipped)."""
    pairs = []
    for iface in interfaces:
        if iface.network.prefixlen >= 31:
            continue
        pairs.append((str(iface.ip), str(iface.network.broadcast_address)))
    return pairs


def send_beacons(interfaces, payload_for, port=BEACON_PORT, sender=None):
    """One round of beacons. payload_for(addr) -> bytes. Returns the number of datagrams sent."""
    sent = 0
    for local, broadcast in broadcast_targets(interfaces):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.bind((local, 0))
                sock.sendto(payload_for(local), (broadcast, port))
                sent += 1
        except OSError:
            pass
    return sent


class DiscoverResponder(threading.Thread):
    """Answers LWIR_DISCOVER with the beacon JSON, from the local address that reaches the sender."""

    def __init__(self, payload_for, port=BEACON_PORT, bind='0.0.0.0', log=None):
        super().__init__(daemon=True, name='discover')
        self.payload_for, self.port, self.log = payload_for, port, log
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind, port))
        self.sock.settimeout(1.0)
        self.stop = False
        self.answered = 0

    def run(self):
        while not self.stop:
            try:
                data, peer = self.sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                return
            if data.strip() != DISCOVER_TEXT or not is_local_client(peer[0]):
                continue   # only LAN askers get an answer (not a Tailscale or public address)
            local = source_for(peer[0])
            if local is None:
                continue
            try:
                # answer from a socket bound to the right local address so the PC sees the board's own address as the source
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as out:
                    out.bind((local, 0))
                    out.sendto(self.payload_for(local), peer)
                self.answered += 1
            except OSError:
                pass

    def close(self):
        self.stop = True
        self.sock.close()


# ---------------------------------------------------------------- mDNS (RFC 6762, answers only)

def _read_name(data, pos, depth=0):
    labels = []
    while True:
        if pos >= len(data) or depth > 8:
            raise ValueError('bad name')
        length = data[pos]
        if length == 0:
            return '.'.join(labels), pos + 1
        if length & 0xC0 == 0xC0:
            if pos + 1 >= len(data):
                raise ValueError('bad pointer')
            target = ((length & 0x3F) << 8) | data[pos + 1]
            name, _ = _read_name(data, target, depth + 1)
            if name:
                labels.append(name)
            return '.'.join(labels), pos + 2
        pos += 1
        if pos + length > len(data):
            raise ValueError('bad label')
        labels.append(data[pos:pos + length].decode('ascii', 'replace'))
        pos += length


def _encode_name(name):
    out = b''
    for label in name.strip('.').split('.'):
        raw = label.encode('ascii', 'replace')
        out += bytes([len(raw)]) + raw
    return out + b'\x00'


def parse_query(data):
    """(id, flags, [(name lowercase, type, class)]) of a DNS message, or None when it is not a plain query."""
    if len(data) < 12:
        return None
    qid, flags, qdcount = struct.unpack('!HHH', data[:6])
    if flags & 0x8000:
        return None   # a response, not a query
    pos, questions = 12, []
    try:
        for _ in range(min(qdcount, 16)):
            name, pos = _read_name(data, pos)
            if pos + 4 > len(data):
                return None
            qtype, qclass = struct.unpack('!HH', data[pos:pos + 4])
            pos += 4
            questions.append((name.lower(), qtype, qclass))
    except ValueError:
        return None
    return qid, flags, questions


def build_answer(qid, questions, names, address, legacy_unicast):
    """DNS response for the questions we know (A, ANY: the address; AAAA: negative NSEC). None when nothing is for us.

    legacy_unicast (query came from a port other than 5353, e.g. the Windows resolver): copy id and questions, TTL 10 s,
    no cache-flush bit. Otherwise a multicast answer with id 0 and cache-flush.
    """
    answers, echoed = [], []
    ttl = 10 if legacy_unicast else 120
    cls_a = 1 if legacy_unicast else 0x8001
    for name, qtype, qclass in questions:
        if name not in names:
            continue
        encoded = _encode_name(name)
        echoed.append(encoded + struct.pack('!HH', qtype, qclass))
        if qtype in (1, 255):
            answers.append(encoded + struct.pack('!HHIH', 1, cls_a, ttl, 4) + socket.inet_aton(address))
        elif qtype == 28:   # AAAA: say plainly that only an A record exists
            bitmap = b'\x00\x01\x40'   # window 0, 1 byte, bit for type 1 (A)
            rdata = encoded + bitmap
            answers.append(encoded + struct.pack('!HHIH', 47, cls_a, ttl, len(rdata)) + rdata)
    if not answers:
        return None
    header = struct.pack('!HHHHHH', qid if legacy_unicast else 0, 0x8400, len(echoed) if legacy_unicast else 0, len(answers), 0, 0)
    return header + (b''.join(echoed) if legacy_unicast else b'') + b''.join(answers)


class MdnsResponder(threading.Thread):
    """Answers .local queries on the direct interface. The address in the answer is the one that reaches the asker."""

    def __init__(self, names, iface, allowed_nets, log=None, port=MDNS_PORT, group=MDNS_GROUP):
        super().__init__(daemon=True, name='mdns')
        self.names = {n.lower().rstrip('.') + ('' if n.lower().rstrip('.').endswith('.local') else '.local') for n in names}
        self.iface, self.allowed, self.log, self.group, self.port = iface, allowed_nets, log, group, port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        self.sock.bind(('', port))
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
        self.sock.settimeout(1.0)
        self.joined = False
        self.stop = False
        self.answered = 0

    def join(self):
        """Join the mDNS group on the direct interface (idempotent; retry on every call until it works)."""
        if self.joined:
            return True
        try:
            index = socket.if_nametoindex(self.iface)
            # struct ip_mreqn: multicast group, local address (any), interface index
            mreq = socket.inet_aton(self.group) + b'\x00\x00\x00\x00' + struct.pack('@i', index)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            self.joined = True
        except OSError as exc:
            self.joined = exc.errno == errno.EADDRINUSE   # already a member (the kernel kept it over a link flap)
        return self.joined

    def rejoin(self):
        """After the cable was re-plugged the kernel may have dropped the membership: join again."""
        self.joined = False
        return self.join()

    def run(self):
        while not self.stop:
            try:
                data, peer = self.sock.recvfrom(1500)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                address = ipaddress.ip_address(peer[0])
                if not self.allowed or any(address in net for net in self.allowed):
                    self.handle(data, peer)
            except (OSError, ValueError):
                pass

    def handle(self, data, peer):
        parsed = parse_query(data)
        if parsed is None:
            return
        qid, _, questions = parsed
        if not any(q[0] in self.names for q in questions):
            return
        local = source_for(peer[0])
        if local is None:
            return
        legacy = peer[1] != self.port
        answer = build_answer(qid, questions, self.names, local, legacy)
        if answer is None:
            return
        unicast = legacy or any(q[2] & 0x8000 for q in questions)   # QU bit: the asker wants a unicast reply
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as out:
            out.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
            out.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local))
            out.bind((local, 0))
            out.sendto(answer, peer if unicast else (self.group, self.port))
        self.answered += 1

    def close(self):
        self.stop = True
        self.sock.close()
