"""Python port of the search logic in lwir_connect.ps1 (same order, same constants), so it can be run on Linux against the board.

    python3 check_find_pi.py [--fixed 10.42.0.1] [--mdns-name orangepi.local] [--linklocal 169.254.42.1]
                             [--steps fixed,mdns,beacon,linklocal] [--port 8090] [--beacon-seconds 5]
prints  FOUND <address> via <step>  and exits 0, or NOTFOUND and exits 2.
Keep this file and lwir_connect.ps1 in step: it is the test for the PowerShell logic that cannot run on the board.
"""
import argparse
import json
import socket
import struct
import subprocess
import sys
import time
import urllib.request

BEACON_PORT = 47890


def test_tcp(host, port, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_text(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as answer:
            return answer.read().decode('utf-8', 'replace') if answer.status == 200 else None
    except (OSError, ValueError):
        return None


def test_lwir(host, port):
    """The same check as Test-Lwir: our /netinfo, or /healthz of an older server."""
    if not test_tcp(host, port):
        return False
    info = get_text('http://%s:%d/netinfo' % (host, port))
    if info and '"client_local"' in info:
        return True
    health = get_text('http://%s:%d/healthz' % (host, port))
    return bool(health and '"up"' in health)


def local_ipv4():
    out = subprocess.run(['ip', '-4', '-o', 'addr', 'show'], stdout=subprocess.PIPE, check=False).stdout.decode()
    found = []
    for line in out.splitlines():
        parts = line.split()
        if 'inet' in parts and parts[1] != 'lo':
            found.append(parts[parts.index('inet') + 1].split('/')[0])
    return found


def mdns_query(name, timeout=3.0):
    """What the Windows resolver does for a .local name: a legacy unicast mDNS question from a random port, to 224.0.0.251:5353."""
    labels = b''.join(bytes([len(p)]) + p.encode() for p in name.strip('.').split('.')) + b'\x00'
    query = struct.pack('!HHHHHH', 0x1234, 0, 1, 0, 0, 0) + labels + struct.pack('!HH', 1, 1)
    deadline = time.time() + timeout
    for local in local_ipv4():
        if time.time() > deadline:
            break
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((local, 0))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
            sock.settimeout(1.0)
            sock.sendto(query, ('224.0.0.251', 5353))
            data, _ = sock.recvfrom(1500)
            if len(data) > 12 and data[:2] == b'\x12\x34':
                # the A record's rdata is the last 4 bytes of the answer we built (single A answer, no additional records)
                ancount = struct.unpack('!H', data[6:8])[0]
                if ancount >= 1 and data.rstrip()[-4:]:
                    return [socket.inet_ntoa(data[-4:])]
        except OSError:
            pass
        finally:
            sock.close()
    return []


def find_by_beacon(seconds):
    listener = None
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('', BEACON_PORT))
        listener.settimeout(0.2)
    except OSError:
        listener = None
    askers = []
    for ip in local_ipv4():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind((ip, 0))
            sock.settimeout(0.2)
            askers.append(sock)
        except OSError:
            pass
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            for sock in askers:
                try:
                    sock.sendto(b'LWIR_DISCOVER', ('255.255.255.255', BEACON_PORT))
                except OSError:
                    pass
            for sock in askers + ([listener] if listener else []):
                try:
                    data, _ = sock.recvfrom(512)
                    message = json.loads(data.decode())
                    if message.get('lwir') == 1 and message.get('addr'):
                        return message
                except (OSError, ValueError):
                    pass
    finally:
        for sock in askers + ([listener] if listener else []):
            sock.close()
    return None


def find_pi(args):
    steps = args.steps.split(',')
    if 'fixed' in steps and test_lwir(args.fixed, args.port):
        return args.fixed, 'fixed'
    if 'mdns' in steps:
        for ip in mdns_query(args.mdns_name):
            if test_lwir(ip, args.port):
                return ip, 'mdns'
    if 'beacon' in steps:
        beacon = find_by_beacon(args.beacon_seconds)
        if beacon and test_lwir(beacon['addr'], int(beacon.get('port') or args.port)):
            return beacon['addr'], 'beacon'
    if 'linklocal' in steps and any(a.startswith('169.254.') for a in local_ipv4()):
        if test_lwir(args.linklocal, args.port):
            return args.linklocal, 'linklocal'
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed', default='10.42.0.1')
    parser.add_argument('--mdns-name', default='orangepi.local')
    parser.add_argument('--linklocal', default='169.254.42.1')
    parser.add_argument('--steps', default='fixed,mdns,beacon,linklocal')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--beacon-seconds', type=int, default=5)
    args = parser.parse_args()
    address, step = find_pi(args)
    if address:
        print('FOUND %s via %s' % (address, step))
        return 0
    print('NOTFOUND')
    return 2


if __name__ == '__main__':
    sys.exit(main())
