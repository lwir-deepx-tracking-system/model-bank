"""Home submission and HTTP guards, with a fake device/server only."""
import http.client
import http.server
import json
from pathlib import Path
import subprocess
import tempfile
from threading import Thread
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ..common.state import APP
from ..web import page_handler


class HomeExecutorTests(unittest.TestCase):
    def test_executor_and_vendor_command(self):
        folder = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            binary = str(Path(temporary) / 'home')
            subprocess.run([
                'g++', '-std=c++17', '-pthread', '-Iv27/include', '-Iteam/include', '-Isim/stub',
                'test_v27_home.cpp', 'v27/src/control/gimbal_command_executor.cpp',
                'v27/src/control/auto_tracking_controller.cpp', 'team/src/control/pid_controller.cpp',
                'team/src/control/gimbal_device.cpp', 'team/src/common/logger.cpp',
                'team/src/pipeline/runtime_control.cpp', '-o', binary,
            ], cwd=folder, check=True)
            subprocess.run([binary], check=True, timeout=10)


class HomeHttpTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(APP.__dict__)
        APP.lv_http = http.server
        APP.GB = NS(state={}, mover='old', mover_t={}, home_control_supported=True)
        APP.LIVE = NS(last_frame_t=time.monotonic(), cam_fail_n=0)
        APP.gb_send = Mock(return_value=True)
        self.replay = patch.object(page_handler.replay_link, 'blocking', return_value=False)
        self.blocking = self.replay.start()
        self.position = patch.object(page_handler, 'live_gimbal_position', return_value={})
        self.position.start()
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), page_handler.make_lv_Handler())
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.finish)

    def finish(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.replay.stop(); self.position.stop()
        APP.__dict__.clear(); APP.__dict__.update(self.saved)

    def post(self, path='/gimbal/home', headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        try:
            connection.request('POST', path, headers=headers if headers is not None else {'X-Gimbal': '1'})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_home_submits_center_without_stop(self):
        self.assertEqual(self.post()[0], 200)
        self.assertEqual([c.args[0] for c in APP.gb_send.call_args_list], ['P', 'C'])
        self.assertIsNone(APP.GB.mover)

    def test_old_bridge_refused(self):
        APP.GB.home_control_supported = False
        status, body = self.post()
        self.assertEqual(status, 409)
        self.assertTrue(json.loads(body)['refused'])
        APP.gb_send.assert_not_called()

    def test_replay_refused(self):
        self.blocking.return_value = True
        self.assertEqual(self.post()[0], 409)
        APP.gb_send.assert_not_called()

    def test_restart_refused(self):
        APP.LIVE.restart_pending = True
        self.assertEqual(self.post()[0], 409)
        APP.gb_send.assert_not_called()

    def test_usb_recovery_refused(self):
        APP.GB.usb_hold = True
        self.assertEqual(self.post()[0], 409)
        APP.gb_send.assert_not_called()

    def test_missing_header_refused(self):
        self.assertEqual(self.post(headers={})[0], 403)
        APP.gb_send.assert_not_called()

    def test_stop_keeps_existing_command(self):
        self.assertEqual(self.post('/gimbal/stop')[0], 200)
        self.assertEqual([c.args[0] for c in APP.gb_send.call_args_list], ['P', 'H'])

    def test_closed_writer_is_not_success(self):
        APP.gb_send.return_value = False
        self.assertEqual(self.post()[0], 503)
