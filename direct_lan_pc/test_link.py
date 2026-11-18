"""Host-only direct-LAN protocol and existing model-bank route tests."""
import http.server
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'direct_lan_pc'))

from tracking_server.src.common.state import APP
from tracking_server.src.direct_lan import DirectLink, call_existing, click_displayed, snapshot
from tracking_server.src.web import stream_sender
from link import Link


class DirectLanTests(unittest.TestCase):
    def setUp(self):
        self.old = vars(APP).copy()
        vars(APP).clear()
        self.current = {'tracker': 'confirm3'}
        APP.lv_http = http.server
        APP.rules_state = lambda: dict(self.current)
        def request(q):
            self.current['tracker'] = q['tracker'][0]
            return True, '적용됨'
        APP.rules_request = request
        APP.GB = None
        APP.LIVE = SimpleNamespace(jcond=threading.Condition(), snap_t=0,
                                   stopped=False, jpeg=b'fake-jpeg', seq=1,
                                   jpeg_frame_t=None)
        APP.LIVE_W, APP.LIVE_H = 640, 480

    def tearDown(self):
        if vars(APP).get('direct_link') is not None:
            APP.direct_link.close()
        vars(APP).clear()
        vars(APP).update(self.old)

    def test_existing_routes_in_memory(self):
        code, raw = call_existing('GET', '/rules')
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)['tracker'], 'confirm3')
        code, raw = call_existing('POST', '/rules/set?tracker=botsort')
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)['tracker'], 'botsort')
        self.assertEqual(self.current['tracker'], 'botsort')

    def test_no_web_page_route(self):
        code, _ = call_existing('GET', '/')
        self.assertEqual(code, 404)
        code, _ = call_existing('GET', '/stream')
        self.assertEqual(code, 404)

    def test_direct_socket_routes_and_binary_frame(self):
        APP.direct_link = DirectLink('127.0.0.1', 0, '127.0.0.1')
        port = APP.direct_link.listener.server_address[1]
        APP.direct_link.start()
        client = Link('127.0.0.1', port)
        self.assertEqual(client.get('/rules')['tracker'], 'confirm3')
        self.assertEqual(client.query('/rules/set', tracker='ocsort')['tracker'], 'ocsort')
        with client.session() as session:
            for _ in range(2):
                header, frame = client.snapshot(session=session)
                self.assertEqual(header['seq'], 1)
                self.assertEqual(header['token'], '1:0')
                self.assertEqual(frame, b'fake-jpeg')
        self.assertGreater(APP.LIVE.snap_t, 0)

    def test_click_uses_owner_from_displayed_frame(self):
        APP.direct_link = DirectLink('127.0.0.1', 0, '127.0.0.1')
        APP.GB = SimpleNamespace(mover_t={}, mover=None, state={})
        APP.gb_send = lambda _command: True
        owner = object()
        shown = time.perf_counter()
        APP.direct_link.remember('7:0', shown, ((5, 0, 0, 200, 200),), 9, {5: owner})
        with patch('tracking_server.src.direct_lan.live_tracks.click', return_value=(True, '')) as click:
            status, _ = click_displayed('7:0', 0.1, 0.1, '127.0.0.1')
        self.assertEqual(status, 200)
        self.assertIs(click.call_args.kwargs['expected_owner'], owner)
        self.assertEqual(click.call_args.kwargs['expected_ticket'], 9)
        with patch('tracking_server.src.direct_lan.live_tracks.click') as click:
            status, _ = click_displayed('8:0', 0.1, 0.1, '127.0.0.1')
        self.assertEqual(status, 409)
        click.assert_not_called()

    def test_picture_first_uses_displayed_overlay_even_if_picture_is_newer(self):
        APP.direct_link = DirectLink('127.0.0.1', 0, '127.0.0.1')
        shown = time.perf_counter()
        overlay_time = shown - 0.03
        key = stream_sender.frame_key(overlay_time)
        owner = object()
        APP.LIVE.jpeg_frame_t = shown
        APP.LIVE.overlay_seq = 1
        APP.LIVE.jpeg_people = ((3, 0, 0, 100, 100),)
        APP.LIVE.jpeg_tracks_ticket = 7
        APP.LIVE.overlay = json.dumps({'pf': 1, 'k': key, 'b': []}).encode()
        APP.LIVE.overlay_clicks = [(key, overlay_time, ((5, 0, 0, 100, 100),), 9, {5: owner})]
        with patch('tracking_server.src.direct_lan.picture_first.active', return_value=True):
            _, _, meta = snapshot()
            self.assertEqual(meta['overlay']['k'], key)
            self.assertNotEqual(meta['frame_key'], key)
            self.assertIs(APP.direct_link.recall(meta['token'])[3][5], owner)
            APP.LIVE.seq = 2
            APP.LIVE.overlay_seq = 2
            APP.LIVE.overlay = json.dumps({'pf': 1, 'k': key + 1, 'b': []}).encode()
            _, _, meta = snapshot()
        self.assertIsNone(APP.direct_link.recall(meta['token']))


if __name__ == '__main__':
    unittest.main()
