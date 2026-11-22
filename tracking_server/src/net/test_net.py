"""Unit tests of the offline / direct-cable code. No root, no real network change; loopback only.
    python3 -m unittest src.net.test_net      (from tracking_server/)
"""
import http.client
import io
import ipaddress
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest
import zipfile
from unittest import mock

from . import discovery, portal, supervisor
from .netutil import is_lan_address, is_local_client, parse_specs

LINUX = sys.platform.startswith('linux')


class AddressRules(unittest.TestCase):
    def test_lan_and_local_client(self):
        for ok in ('10.42.0.1', '192.168.50.2', '172.16.0.1', '172.31.255.1', '169.254.42.1'):
            self.assertTrue(is_lan_address(ok), ok)
        for bad in ('100.64.1.2', '8.8.8.8', '172.32.0.1', '0.0.0.0', '224.0.0.1', '::1', 'x'):
            self.assertFalse(is_lan_address(bad), bad)
        self.assertTrue(is_local_client('127.0.0.1'))
        self.assertFalse(is_local_client('100.64.9.9'))   # a Tailscale client: no PC program (it has a working address already)

    def test_specs(self):
        specs = parse_specs('10.42.0.1/24, 169.254.42.1/16,10.42.0.1/24')
        self.assertEqual([a for a, _ in specs], ['10.42.0.1', '169.254.42.1'])
        self.assertEqual(str(specs[0][1]), '10.42.0.0/24')
        self.assertEqual(parse_specs(''), [])
        for bad in ('100.64.1.2/32', '0.0.0.0/0', '127.0.0.1/8', '8.8.8.8/24'):
            with self.assertRaises(ValueError, msg=bad):
                parse_specs(bad)


class SupervisorTests(unittest.TestCase):
    def make(self, results):
        it = iter(results)
        return supervisor.Supervisor(probe=lambda: next(it), tailscale=lambda: 'running', fail_limit=3, ok_limit=2, clock=lambda: 1000.0)

    def test_first_check_decides_at_once(self):
        s = self.make([False]); s.tick(); self.assertEqual(s.mode, 'lan-only')
        s = self.make([True]); s.tick(); self.assertEqual(s.mode, 'online')

    def test_hysteresis(self):
        s = self.make([True, False, False, True, False, False, False, True, False, True, True])
        s.tick(); self.assertEqual(s.mode, 'online')
        s.tick(); s.tick(); self.assertEqual(s.mode, 'online')      # two failures are not enough
        s.tick(); self.assertEqual(s.mode, 'online')                # a good one resets the count
        s.tick(); s.tick(); self.assertEqual(s.mode, 'online')
        self.assertTrue(s.tick()); self.assertEqual(s.mode, 'lan-only')   # third failure in a row
        s.tick(); self.assertEqual(s.mode, 'lan-only')              # one good check is not enough
        s.tick(); s.tick(); self.assertEqual(s.mode, 'lan-only')
        s.tick(); self.assertEqual(s.mode, 'online')

    def test_probe_internet_any_target(self):
        calls = []

        def connect(host, port, timeout):
            calls.append(host)
            return host == '8.8.8.8'
        self.assertTrue(supervisor.probe_internet(connect=connect))
        self.assertEqual(calls, ['1.1.1.1', '8.8.8.8'])
        self.assertFalse(supervisor.probe_internet(connect=lambda *a: False))

    def test_state_file_is_atomic_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'state.json')
            supervisor.write_state(path, dict(mode='lan-only', updated=1))
            self.assertEqual(json.load(open(path))['mode'], 'lan-only')
            self.assertEqual(os.listdir(tmp), ['state.json'])


class ClientTableTests(unittest.TestCase):
    def test_silence_window_and_reset(self):
        now = [0.0]
        table = portal.ClientTable(silence_s=600, clock=lambda: now[0])
        self.assertFalse(table.silenced('10.42.0.20'))
        table.mark_shown('10.42.0.20')
        now[0] = 599; self.assertTrue(table.silenced('10.42.0.20'))
        now[0] = 601; self.assertFalse(table.silenced('10.42.0.20'))
        table.mark_shown('10.42.0.20'); table.reset(); self.assertFalse(table.silenced('10.42.0.20'))

    def test_mac_change_shows_the_portal_again(self):
        table = portal.ClientTable()
        table.neighbours({'10.42.0.20': 'aa:aa:aa:aa:aa:01'})
        table.mark_shown('10.42.0.20')
        self.assertEqual(table.neighbours({'10.42.0.20': 'aa:aa:aa:aa:aa:01'}), [])
        self.assertTrue(table.silenced('10.42.0.20'))
        self.assertEqual(table.neighbours({'10.42.0.20': 'bb:bb:bb:bb:bb:02'}), ['10.42.0.20'])
        self.assertFalse(table.silenced('10.42.0.20'))

    def test_limit_forgets_the_oldest(self):
        now = [0.0]
        table = portal.ClientTable(clock=lambda: now[0], limit=2)
        for i, ip in enumerate(('1', '2', '3')):
            now[0] = i; table.mark_shown(ip)
        self.assertEqual(sorted(table.snapshot()), ['2', '3'])


class PortalHttp(unittest.TestCase):
    """The responder over a real loopback socket (the handler needs no special address)."""

    def setUp(self):
        self.now = [0.0]
        self.table = portal.ClientTable(silence_s=600, clock=lambda: self.now[0])
        handler = portal.make_handler(self.table, page_port=8090)
        import http.server
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def get(self, host, path, method='GET'):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        conn.request(method, path, headers={'Host': host})
        res = conn.getresponse()
        body = res.read()
        conn.close()
        return res.status, res.getheader('Location'), body

    def test_windows_flow(self):
        status, location, _ = self.get('www.msftconnecttest.com', '/connecttest.txt')
        self.assertEqual((status, location), (302, 'http://127.0.0.1:8090/'))
        self.assertEqual(self.get('www.msftconnecttest.com', '/connecttest.txt')[0], 302)   # probes alone never silence it
        self.assertEqual(self.get('www.msftconnecttest.com', '/redirect')[0], 302)          # the browser Windows opens
        status, _, body = self.get('www.msftconnecttest.com', '/connecttest.txt')
        self.assertEqual((status, body), (200, b'Microsoft Connect Test'))
        self.now[0] = 601
        self.assertEqual(self.get('www.msftconnecttest.com', '/connecttest.txt')[0], 302)

    def test_other_systems_after_a_visit(self):
        self.get('10.42.0.1', '/')   # a bare address typed in the browser is a visit
        self.assertEqual(self.get('connectivitycheck.gstatic.com', '/generate_204')[0], 204)
        self.assertEqual(self.get('www.msftncsi.com', '/ncsi.txt')[2], b'Microsoft NCSI')
        self.assertEqual(self.get('detectportal.firefox.com', '/success.txt')[2], b'success\n')
        self.assertEqual(self.get('nmcheck.gnome.org', '/check_network_status.txt')[2], b'NetworkManager is online\n')
        # Apple: never the success page (the Captive Network Assistant window would close by itself)
        self.assertEqual(self.get('captive.apple.com', '/hotspot-detect.html')[0], 302)

    def test_head_and_status_and_unknown_host(self):
        self.assertEqual(self.get('www.msftconnecttest.com', '/connecttest.txt', 'HEAD')[0], 302)
        _, _, body = self.get('x', '/lwir-status')
        self.assertEqual(json.loads(body), dict(lwir=1, shown=False))   # asking does not count as a visit
        self.get('whatever.example', '/x')
        self.assertEqual(json.loads(self.get('x', '/lwir-status')[2])['shown'], True)

    def test_known_check_bodies(self):
        self.assertEqual(portal.lookup_check('captive.apple.com', '/hotspot-detect.html')[1], portal.APPLE_SUCCESS)
        self.assertIsNone(portal.lookup_check('www.msftconnecttest.com', '/redirect'))
        self.assertIsNone(portal.lookup_check('', '/'))


class DnsAndMdns(unittest.TestCase):
    def query(self, name, qtype, qclass=1, qid=0x1234):
        labels = b''.join(bytes([len(p)]) + p.encode() for p in name.split('.')) + b'\x00'
        return struct.pack('!HHHHHH', qid, 0, 1, 0, 0, 0) + labels + struct.pack('!HH', qtype, qclass)

    def test_parse_and_answer_a(self):
        parsed = discovery.parse_query(self.query('OrangePi.local', 1))
        self.assertEqual(parsed[2], [('orangepi.local', 1, 1)])
        answer = discovery.build_answer(parsed[0], parsed[2], {'orangepi.local'}, '10.42.0.1', legacy_unicast=True)
        self.assertEqual(answer[:2], b'\x12\x34')
        self.assertEqual(answer[-4:], socket.inet_aton('10.42.0.1'))
        self.assertEqual(struct.unpack('!HHHH', answer[4:12]), (1, 1, 0, 0))   # qd 1, an 1
        multicast = discovery.build_answer(parsed[0], parsed[2], {'orangepi.local'}, '10.42.0.1', legacy_unicast=False)
        self.assertEqual(multicast[:2], b'\x00\x00')
        self.assertEqual(struct.unpack('!HHHH', multicast[4:12]), (0, 1, 0, 0))   # no question echoed, 1 answer

    def test_aaaa_gets_a_negative_answer_and_foreign_names_none(self):
        parsed = discovery.parse_query(self.query('orangepi.local', 28))
        answer = discovery.build_answer(parsed[0], parsed[2], {'orangepi.local'}, '10.42.0.1', True)
        self.assertIsNotNone(answer)
        self.assertEqual(answer[-3:], b'\x00\x01\x40')   # NSEC bitmap: only type 1 (A) exists for this name
        self.assertIn(struct.pack('!H', 47), answer)
        self.assertIsNone(discovery.build_answer(1, [('other.local', 1, 1)], {'orangepi.local'}, '10.42.0.1', True))

    def test_garbage_and_responses_are_ignored(self):
        self.assertIsNone(discovery.parse_query(b''))
        self.assertIsNone(discovery.parse_query(b'\x00' * 11))
        response = struct.pack('!HHHHHH', 1, 0x8400, 0, 0, 0, 0)
        self.assertIsNone(discovery.parse_query(response))
        loop = struct.pack('!HHHHHH', 1, 0, 1, 0, 0, 0) + b'\xc0\x0c' + struct.pack('!HH', 1, 1)   # pointer to itself
        self.assertIsNone(discovery.parse_query(loop))

    def test_beacon_targets_and_payload(self):
        nets = [ipaddress.ip_interface('10.42.0.1/24'), ipaddress.ip_interface('169.254.42.1/16'), ipaddress.ip_interface('10.0.0.1/32')]
        self.assertEqual(discovery.broadcast_targets(nets), [('10.42.0.1', '10.42.0.255'), ('169.254.42.1', '169.254.255.255')])
        payload = json.loads(discovery.beacon_payload('orangepi5plus', '10.42.0.1', 8090, '/', 'lan-only'))
        self.assertEqual((payload['lwir'], payload['addr'], payload['port'], payload['mode']), (1, '10.42.0.1', 8090, 'lan-only'))

    @unittest.skipUnless(LINUX, 'multicast socket options are Linux specific')
    def test_discover_responder_answers_on_loopback(self):
        responder = discovery.DiscoverResponder(lambda addr: discovery.beacon_payload('t', addr), port=0, bind='127.0.0.1')
        port = responder.sock.getsockname()[1]
        responder.start()
        self.addCleanup(responder.close)
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(2)
        self.addCleanup(client.close)
        client.sendto(b'something else', ('127.0.0.1', port))
        client.sendto(discovery.DISCOVER_TEXT, ('127.0.0.1', port))
        data, _ = client.recvfrom(512)
        self.assertEqual(json.loads(data)['addr'], '127.0.0.1')


@unittest.skipUnless(LINUX, 'IP_FREEBIND is Linux only')
class DirectListeners(unittest.TestCase):
    def setUp(self):
        from ..web import direct_listeners
        self.dl = direct_listeners
        self.addCleanup(direct_listeners.stop)
        patcher = mock.patch.object(direct_listeners, 'TAILSCALE_CACHE', '/nonexistent/lwir_tailscale_ip')   # never the real cache file of this board
        patcher.start()
        self.addCleanup(patcher.stop)

    def free_port(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            return sock.getsockname()[1]

    def test_starts_without_the_address_and_serves_only_its_network(self):
        import http.server

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.send_header('Content-Length', '2'); self.end_headers(); self.wfile.write(b'ok')

            def log_message(self, *a):
                pass
        port = self.free_port()
        self.assertEqual(self.dl.start(Handler, port, '10.42.0.1/24,100.64.1.2/32', log=lambda m: None), [])   # a Tailscale address makes the whole list invalid
        self.assertEqual(self.dl.start(Handler, port, '10.42.0.1/24,169.254.42.1/16', log=lambda m: None), ['10.42.0.1', '169.254.42.1'])
        self.assertEqual(self.dl.start(Handler, port, '10.42.0.1/24,169.254.42.1/16', skip=('10.42.0.1',), log=lambda m: None), ['169.254.42.1'])
        self.assertEqual(self.dl.start(Handler, port, 'off'), [])
        self.assertEqual(self.dl.start(Handler, port, '8.8.8.8/32', log=lambda m: None), [])
        self.assertEqual(self.dl.assigned_urls(), [])

    def test_wrong_client_network_is_dropped(self):
        from .netutil import make_freebind_server
        import http.server
        server_class = make_freebind_server(http.server.ThreadingHTTPServer, ipaddress.ip_network('10.42.0.0/24'))
        server = server_class(('127.0.0.1', 0), http.server.BaseHTTPRequestHandler)
        self.addCleanup(server.server_close)
        self.assertFalse(server.verify_request(None, ('127.0.0.1', 1)))
        self.assertTrue(server.verify_request(None, ('10.42.0.77', 1)))
        self.assertFalse(server.verify_request(None, ('100.64.9.9', 1)))

    def test_netinfo_and_downloads(self):
        state = dict(mode='lan-only', internet=False, tailscale='absent', updated=0)
        info = self.dl.netinfo('10.42.0.20', state)
        self.assertEqual((info['mode'], info['client_local']), ('lan-only', True))
        self.assertFalse(self.dl.netinfo('100.64.9.9', state)['pc_download'])
        self.assertEqual(self.dl.pc_download('/pc/lwir_connect.ps1', '100.64.9.9')[0], 404)   # not offered over Tailscale
        self.assertEqual(self.dl.pc_download('/pc/../src/app.py', '10.42.0.20')[0], 404)
        self.assertEqual(self.dl.pc_download('/pc/check_find_pi.py', '10.42.0.20')[0], 404)
        if self.dl.pc_files_present():
            status, ctype, name, body = self.dl.pc_download('/pc/lwir_connect.zip', '10.42.0.20')
            self.assertEqual((status, name), (200, 'lwir_connect.zip'))
            names = zipfile.ZipFile(io.BytesIO(body)).namelist()
            self.assertIn('lwir_connect.ps1', names)
            self.assertIn('lwir_connect.bat', names)

    def test_stale_state_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 's.json')
            json.dump(dict(mode='lan-only', updated=1), open(path, 'w'))
            self.assertEqual(self.dl.read_state(path), {})


class PortalSwitchAndCache(unittest.TestCase):
    def test_switch_makes_every_check_succeed(self):
        import http.server
        table = portal.ClientTable()
        off = [False]
        handler = portal.make_handler(table, 8090, is_off=lambda: off[0])
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def get(host, path):
            conn = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=3)
            conn.request('GET', path, headers={'Host': host})
            res = conn.getresponse()
            body = res.read()
            conn.close()
            return res.status, body
        self.assertEqual(get('www.msftconnecttest.com', '/connecttest.txt')[0], 302)
        off[0] = True   # touch ~/.lwir_portal_off
        self.assertEqual(get('www.msftconnecttest.com', '/connecttest.txt'), (200, b'Microsoft Connect Test'))
        self.assertEqual(get('captive.apple.com', '/hotspot-detect.html')[0], 200)   # Apple too: nothing pops up
        self.assertEqual(get('10.42.0.1', '/')[0], 302)                              # a typed address still lands on the page
        off[0] = False
        table.reset()
        self.assertEqual(get('www.msftconnecttest.com', '/connecttest.txt')[0], 302)

    @unittest.skipUnless(LINUX, 'IP_FREEBIND is Linux only')
    def test_cached_tailscale_address_becomes_a_listener(self):
        from ..web import direct_listeners as dl
        import http.server
        self.addCleanup(dl.stop)
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, 'ts')
            self.assertIsNone(dl.cached_tailscale_address(cache))
            open(cache, 'w').write('100.64.1.2\n')
            self.assertEqual(dl.cached_tailscale_address(cache), '100.64.1.2')
            open(cache, 'w').write('8.8.8.8\n')
            self.assertIsNone(dl.cached_tailscale_address(cache))   # only 100.64.0.0/10 counts
            open(cache, 'w').write('100.64.1.2\n')
            with mock.patch.object(dl, 'TAILSCALE_CACHE', cache):
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                started = dl.start(http.server.BaseHTTPRequestHandler, port, '10.42.0.1/24', log=lambda m: None)
                self.assertEqual(started, ['10.42.0.1', '100.64.1.2'])
                started = dl.start(http.server.BaseHTTPRequestHandler, port, '10.42.0.1/24', skip=('100.64.1.2',), log=lambda m: None)
                self.assertEqual(started, ['10.42.0.1'])   # the primary listener already has it
                self.assertTrue(all(url.startswith('http://10.') for url in dl.assigned_urls()))   # never lists the Tailscale address as a LAN url

    @unittest.skipUnless(LINUX, 'multicast socket options are Linux specific')
    def test_discover_ignores_non_lan_askers(self):
        with mock.patch.object(discovery, 'is_local_client', return_value=False):
            responder = discovery.DiscoverResponder(lambda addr: discovery.beacon_payload('t', addr), port=0, bind='127.0.0.1')
            port = responder.sock.getsockname()[1]
            responder.start()
            self.addCleanup(responder.close)
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.settimeout(1)
            self.addCleanup(client.close)
            client.sendto(discovery.DISCOVER_TEXT, ('127.0.0.1', port))
            with self.assertRaises(socket.timeout):
                client.recvfrom(512)


if __name__ == '__main__':
    unittest.main()
