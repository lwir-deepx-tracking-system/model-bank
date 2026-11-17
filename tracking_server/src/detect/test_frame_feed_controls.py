from collections import deque
import http.server
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ..common.state import APP
from ..web import page_handler, system_buttons
from . import frame_feed


class ModeTests(unittest.TestCase):
    def test_saved_mode_overrides_environment_and_invalid_file_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'npu_feed.json'
            with patch.object(frame_feed, 'MODE_FILE', path), patch.dict(os.environ, LWIR_NPU_FEED='arrival'):
                self.assertEqual(frame_feed.load_mode(), 'arrival')
                frame_feed.save_mode('normal')
                self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'mode': 'normal'})
                self.assertEqual(frame_feed.load_mode(), 'normal')
                with self.assertRaises(ValueError):
                    frame_feed.save_mode('unknown')
                self.assertEqual(frame_feed.load_mode(), 'normal')
                path.write_text('{"mode": ["normal"]}', encoding='utf-8')
                with patch('builtins.print'):
                    self.assertEqual(frame_feed.load_mode(), 'arrival')

    def test_failed_replace_preserves_previous_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'npu_feed.json'
            with patch.object(frame_feed, 'MODE_FILE', path):
                frame_feed.save_mode('normal')
                with patch.object(frame_feed.os, 'replace', side_effect=PermissionError('read only')):
                    with self.assertRaises(PermissionError):
                        frame_feed.save_mode('arrival')
                self.assertEqual(frame_feed.load_mode(), 'normal')
                self.assertEqual(list(path.parent.iterdir()), [path])

    def test_ready_prefetch_selection(self):
        self.assertFalse(frame_feed.ready_prefetch_enabled('arrival', '1', True))
        self.assertFalse(frame_feed.ready_prefetch_enabled('arrival', '1', False))
        self.assertTrue(frame_feed.ready_prefetch_enabled('normal', '1', True))
        self.assertTrue(frame_feed.ready_prefetch_enabled('normal', '', True))
        self.assertFalse(frame_feed.ready_prefetch_enabled('normal', '0', True))
        with self.assertRaises(ValueError):
            frame_feed.ready_prefetch_enabled('normal', '1', False)


class FeedControlTests(unittest.TestCase):
    def setUp(self):
        self.live = SimpleNamespace(
            restart_supported=True, restart_pending=False, restart_stop=False, stopped=False,
            cam_hold=False, cam_lost=False, meas=None, replay=None, instance='test-instance',
            sys_lock=threading.Lock(), cam_lock=threading.Lock(), meas_lock=threading.Lock(),
            cond=threading.Condition(), arr=deque(), last=None,
        )
        app = patch.dict(vars(APP), dict(
            LIVE=self.live, GB=None, FEED=None, FEED_MODE='normal', FEED_READY=True,
            FEED_INCOMPATIBLE='', args=SimpleNamespace(live_wait=0), lv_http=http.server,
            MODEL=dict(req=None, busy=False, lock=threading.Lock()),
        ), clear=True)
        app.start()
        self.addCleanup(app.stop)
        replay = patch.object(frame_feed.camera_extra.replay_eval, 'EV', None)
        replay.start()
        self.addCleanup(replay.stop)

    def make_feed(self):
        feed = object.__new__(frame_feed.Feed)
        feed.hot_switch = True
        feed.mode = APP.FEED_MODE
        feed.pending = None
        feed.out = deque()
        feed.err = feed.reason = None
        feed.direct = feed.closed = feed.paused = feed.working = False
        feed.eng = [object(), object()]
        feed.busy = [False, False]
        feed.last_e = 1
        feed.st = dict(core=[0, 0], submitted=0, taken=0, order_bad=0)
        APP.FEED = feed
        return feed

    def test_status_distinguishes_startup_refusal_and_one_core_fallback(self):
        APP.FEED_MODE = 'arrival'
        APP.FEED_REASON = 'NPU 1 unavailable'
        refused = system_buttons.npu_feed_state()
        self.assertEqual(refused['requested'], 'arrival')
        self.assertFalse(refused['active'])
        self.assertFalse(refused['can_toggle'])
        self.assertEqual(refused['reason'], 'NPU 1 unavailable')
        feed = self.make_feed()
        feed.st.update(core=[2, 1], submitted=3, taken=2)
        active = system_buttons.npu_feed_state()
        self.assertTrue(active['active'])
        self.assertEqual(active['cores'], [2, 1])
        self.assertEqual((active['submitted'], active['taken'], active['order_errors']), (3, 2, 0))
        feed.eng.pop()
        feed.mode = 'normal'
        feed.reason = 'NPU 0 only'
        fallback = system_buttons.npu_feed_state()
        self.assertFalse(fallback['active'])
        self.assertEqual(fallback['mode'], 'normal')
        self.assertEqual(fallback['requested'], 'arrival')
        self.assertEqual(fallback['reason'], 'NPU 0 only')

    def test_conflicts_do_not_write_and_steady_replay_can_switch(self):
        feed = self.make_feed()
        self.live.replay = {'name': 'recording'}
        self.live.replay_busy = True
        with patch.object(frame_feed, 'save_mode') as save:
            self.assertFalse(system_buttons.npu_feed_state()['can_toggle'])
            self.assertFalse(system_buttons.request_npu_feed('arrival')[0])
            self.live.replay_busy = False
            ev = SimpleNamespace(position=lambda: {'seek_to': 10})
            with patch.object(frame_feed.camera_extra.replay_eval, 'EV', ev):
                self.assertFalse(system_buttons.request_npu_feed('arrival')[0])
            APP.MODEL['busy'] = True
            self.assertFalse(system_buttons.request_npu_feed('arrival')[0])
            APP.MODEL['busy'] = False
            self.assertTrue(system_buttons.request_npu_feed('arrival')[0])
            self.assertEqual(feed.pending, 'arrival')
            save.assert_not_called()

    def test_save_failure_keeps_mode_and_can_retry(self):
        feed = self.make_feed()
        self.assertTrue(system_buttons.request_npu_feed('arrival')[0])
        with patch.object(frame_feed, 'save_mode', side_effect=PermissionError), patch('builtins.print'):
            with self.live.cond:
                feed._apply_mode()
        status = system_buttons.npu_feed_state()
        self.assertEqual(status['mode'], 'normal')
        self.assertEqual(status['requested'], 'normal')
        self.assertFalse(status['pending'])
        self.assertTrue(status['reason'])
        self.assertTrue(status['can_toggle'])
        self.assertFalse(self.live.restart_pending)
        self.assertFalse(self.live.restart_stop)
        self.assertFalse(self.live.cam_hold)

    def test_mode_waits_for_every_old_job_before_switching(self):
        feed = self.make_feed()
        self.assertEqual(feed._free(), 0)
        self.assertTrue(system_buttons.request_npu_feed('arrival')[0])
        feed.out.append(object())
        feed.busy[0] = True
        with patch.object(frame_feed, 'save_mode') as save, patch.object(frame_feed.camera_extra, 'say'), self.live.cond:
            feed._apply_mode()
            save.assert_not_called()
            feed.out.popleft()
            feed._apply_mode()
            save.assert_not_called()
            feed.busy[0] = False
            feed._apply_mode()
            save.assert_called_once_with('arrival')
            self.assertEqual(feed._free(), 0)
            feed.last_e = 0
            self.assertEqual(feed._free(), 1)
            self.assertTrue(system_buttons.request_npu_feed('normal')[0])
            feed._apply_mode()
        self.assertEqual(feed.mode, 'normal')
        self.assertEqual(feed._free(), 0)
        self.assertEqual(len(feed.eng), 2)

    def test_route_requires_same_origin_and_valid_mode(self):
        self.make_feed()
        def post(path, headers):
            handler = object.__new__(page_handler.make_lv_Handler())
            handler.path = path
            handler.headers = headers
            handler.wfile = io.BytesIO()
            handler.send_response = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            handler.send_error = Mock()
            handler.do_POST()
            return handler

        headers = {'Host': 'localhost:8090', 'Origin': 'http://localhost:8090', 'X-Gimbal': '1'}
        path = '/system/npu-feed?mode=arrival'
        with patch.object(frame_feed, 'save_mode') as save:
            denied = post(path, dict(headers, Origin='http://another-host'))
            denied.send_error.assert_called_once_with(403)
            bad_mode = post(path + '&mode=normal', headers)
            bad_mode.send_response.assert_called_once_with(400)
            save.assert_not_called()
            with patch('builtins.print'):
                accepted = post(path, headers)
            accepted.send_response.assert_called_once_with(202)
            self.assertEqual(json.loads(accepted.wfile.getvalue())['instance'], 'test-instance')
            self.assertTrue(json.loads(accepted.wfile.getvalue())['ok'])
            self.assertFalse(self.live.restart_pending)
            self.assertFalse(self.live.restart_stop)
            self.assertEqual(APP.FEED.pending, 'arrival')
            save.assert_not_called()
            conflict = post('/system/npu-feed?mode=normal', headers)
            conflict.send_response.assert_called_once_with(409)
            save.assert_not_called()

    def test_idle_wait_applies_mode_then_can_restart_without_publishing(self):
        waiting = threading.Event()
        switched = threading.Event()

        class NotifyingCondition(threading.Condition):
            def wait(self, timeout=None):
                if APP.FEED.mode == 'arrival':
                    switched.set()
                waiting.set()
                return super().wait(timeout)

        self.live.cond = NotifyingCondition()
        feed = self.make_feed()
        result = []
        thread = threading.Thread(target=lambda: result.append(feed.read_frame(0)), daemon=True)
        with patch.object(frame_feed, 'save_mode') as save, patch.object(frame_feed.camera_extra, 'say'), patch('builtins.print'):
            thread.start()
            try:
                self.assertTrue(waiting.wait(1))
                self.assertTrue(system_buttons.request_npu_feed('arrival')[0])
                self.assertTrue(switched.wait(1))
                save.assert_called_once_with('arrival')
                self.assertEqual(system_buttons.npu_feed_state()['mode'], 'arrival')
            finally:
                with self.live.cond:
                    feed.out.append(object())
                    system_buttons.stop_for_restart()
                thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [(None, 0.0)])
        self.assertEqual(len(feed.out), 1)
        self.assertFalse(self.live.arr)


if __name__ == '__main__':
    unittest.main()
