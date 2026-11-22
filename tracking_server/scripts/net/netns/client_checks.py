"""Protocol checks run on the client side of the namespace test (the PC): beacon, discover, mDNS, DNS rcode.

    client_checks.py beacon               wait up to 4 s for the board's UDP beacon, print its JSON
    client_checks.py discover [ADDR]      send LWIR_DISCOVER (broadcast, or unicast to ADDR), print the answer
    client_checks.py mdns NAME            legacy-unicast mDNS question for NAME, print the A address and whether AAAA gets a negative answer
"""
import json
import socket
import struct
import sys
import time

PORT = 47890


def beacon():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', PORT))
    sock.settimeout(4)
    data, peer = sock.recvfrom(512)
    message = json.loads(data.decode())
    print(json.dumps(dict(message, _from=peer[0])))
    return 0 if message.get('lwir') == 1 else 1


def discover(target=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(1)
    end = time.time() + 5
    while time.time() < end:
        sock.sendto(b'LWIR_DISCOVER', (target or '255.255.255.255', PORT))
        try:
            data, peer = sock.recvfrom(512)
        except socket.timeout:
            continue
        message = json.loads(data.decode())
        print(json.dumps(dict(message, _from=peer[0])))
        return 0 if message.get('lwir') == 1 else 1
    return 2


def mdns(name):
    labels = b''.join(bytes([len(p)]) + p.encode() for p in name.split('.')) + b'\x00'
    result = {}
    for qtype, tag in ((1, 'A'), (28, 'AAAA')):
        query = struct.pack('!HHHHHH', 0x4321, 0, 1, 0, 0, 0) + labels + struct.pack('!HH', qtype, 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', 0))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
        sock.settimeout(2)
        sock.sendto(query, ('224.0.0.251', 5353))
        try:
            data, peer = sock.recvfrom(1500)
        except socket.timeout:
            result[tag] = None
            continue
        ancount = struct.unpack('!H', data[6:8])[0]
        # echoed question (header 12 + name + 4), then the first answer: name + type + class + ttl + rdlength + rdata
        pos = 12 + len(labels) + 4
        pos += len(labels)
        rtype, _, ttl, rdlength = struct.unpack('!HHIH', data[pos:pos + 10])
        rdata = data[pos + 10:pos + 10 + rdlength]
        result[tag] = dict(answers=ancount, type=rtype, ttl=ttl, id=hex(struct.unpack('!H', data[:2])[0]),
                           address=socket.inet_ntoa(rdata) if rtype == 1 else None, src=peer[0])
    print(json.dumps(result))
    a, aaaa = result.get('A'), result.get('AAAA')
    return 0 if a and a['type'] == 1 and aaaa and aaaa['type'] == 47 else 1


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'beacon':
        sys.exit(beacon())
    if command == 'discover':
        sys.exit(discover(sys.argv[2] if len(sys.argv) > 2 else None))
    if command == 'mdns':
        sys.exit(mdns(sys.argv[2]))
    sys.exit(64)
