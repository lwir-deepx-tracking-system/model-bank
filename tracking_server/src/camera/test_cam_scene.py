"""11-15: the server's side of a picture without a scene (cam_pipe v2.13 SCENE_LOST / SCENE_BACK lines): cam_scene.py and its hooks. No camera, no board.

  - SCENE_LOST sets the page state (with or without the automatic engine reset), a later SCENE_LOST keeps the first time, SCENE_BACK clears it;
  - the camera log thread (camera_extra._tail) passes cam_pipe.log lines to it;
  - a new cam_pipe (lv_spawn, camera or replay) clears it;
  - /rules carries it as cam_scene;
  - the reset button treats it like a dead camera: on a port without a power switch only the replug notice (no new open), behind the hub the 5 V step.
Run from the tracking_server folder: python -m unittest src.camera.test_cam_scene
"""
import os
import shutil
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ..common.state import APP
from . import cam_frozen, cam_scene, camera_extra, camera_link
from .test_cam_frozen import LiveCase   # fixtures only (no tests of its own)

LOST_RESET = ('SCENE_LOST no scene in the picture for 20 s (centre_same 0.985: the middle of the picture is one value; seen 11-15 16:38 when the shutter stayed shut); '
              'one engine reset now (ResetMainBoard, close, 20 s, reopen with a calibration), the only one on this camera power-on (USB 3-1:3:9), at 16:38:31')
LOST_LEFT = ('SCENE_LOST no scene in the picture for 20 s (centre_same 0.985: the middle of the picture is one value; seen 11-15 16:38 when the shutter stayed shut); '
             'no automatic repair left (the one engine reset of this camera power-on was at 81234.5 s after boot): unplug the camera USB cable for 10 s and plug it back, at 16:39:20')
BACK = 'SCENE_BACK the picture has a scene again (centre_same 0.021 for 5 s), 80 s after SCENE_LOST, at 16:39:51'


class State(unittest.TestCase):
    def test_lost_with_reset_then_without_then_back(self):
        live_ = NS()
        self.assertIsNone(cam_scene.state(live_)); self.assertIsNone(cam_scene.state(None))
        self.assertTrue(cam_scene.on_line(live_, LOST_RESET))
        st_ = cam_scene.state(live_)
        self.assertEqual(st_['at'], '16:38:31'); self.assertTrue(st_['reset'])
        self.assertIn('스스로 한 번 다시 시작하는 중', st_['text']); self.assertIn('(16:38:31 부터)', st_['text']); self.assertEqual(st_['title'], cam_scene.TITLE)
        self.assertTrue(cam_scene.on_line(live_, LOST_LEFT))   # after the reset the picture still has no scene: no automatic repair left
        st_ = cam_scene.state(live_)
        self.assertEqual(st_['at'], '16:38:31'); self.assertFalse(st_['reset']); self.assertIn('이미 한 번 했습니다', st_['text']); self.assertIn('10초 뽑았다', st_['text'])
        self.assertFalse(cam_scene.on_line(live_, 'CAMFPS 29.9')); self.assertFalse(cam_scene.on_line(live_, 'SCENE_CHECK 600 checks in 600 s, centre_same largest 0.031'))
        self.assertIsNotNone(cam_scene.state(live_))
        self.assertTrue(cam_scene.on_line(live_, BACK)); self.assertIsNone(cam_scene.state(live_))
        self.assertFalse(cam_scene.on_line(live_, BACK))

    def test_line_without_a_clock(self):
        live_ = NS(); cam_scene.on_line(live_, 'SCENE_LOST something')
        self.assertRegex(cam_scene.state(live_)['at'], r'^\d\d:\d\d:\d\d$')


class Hooks(unittest.TestCase):
    def setUp(self):
        saved_ = APP.__dict__.copy(); self.addCleanup(lambda: (APP.__dict__.clear(), APP.__dict__.update(saved_)))
        self.d = tempfile.mkdtemp(prefix='cam_scene_'); self.addCleanup(shutil.rmtree, self.d, True)
        self.log = open(os.path.join(self.d, 'cam_pipe.log'), 'w'); self.addCleanup(self.log.close)
        APP.LIVE = NS(log=self.log)
        for k_, v_ in (('tail', None), ('tail_name', None), ('tail_buf', b'')):
            p_ = patch.dict(camera_extra._S, {k_: v_}); p_.start(); self.addCleanup(p_.stop)
        self.addCleanup(lambda: camera_extra._S['tail'] is not None and camera_extra._S['tail'].close())

    def test_camera_log_thread_passes_the_lines(self):
        self.log.write('CAMERA 640 480\nCAMFPS 29.0\n' + LOST_RESET + '\n'); self.log.flush()
        camera_extra._tail()
        self.assertTrue(APP.LIVE.cam_scene['reset'])
        self.log.write(BACK + '\n'); self.log.flush(); camera_extra._tail()
        self.assertIsNone(APP.LIVE.cam_scene)

    def test_lv_spawn_clears_it_for_camera_and_replay(self):
        for replay_ in (None, NS()):
            with self.subTest(replay=replay_ is not None):
                APP.LIVE = NS(replay=replay_, cam_scene=dict(at='16:38:31', reset=False, line=''))
                calls_ = []
                with patch.object(camera_link, 'invalidate_preview', lambda *a, **k: None), patch.object(camera_link.replay_link, 'spawn', lambda: calls_.append('replay')), \
                     patch.object(camera_link.cam_frozen, 'on_spawn', lambda live_: calls_.append('frozen')), patch.object(camera_link.camera_extra, 'wanted', lambda: True), \
                     patch.object(camera_link.camera_extra, 'spawn', lambda: calls_.append('camera')):
                    camera_link.lv_spawn()
                self.assertIsNone(APP.LIVE.cam_scene)
                self.assertEqual(calls_, ['replay'] if replay_ is not None else ['frozen', 'camera'])

    def test_rules_carry_it(self):
        from ..web import page_settings
        with open(page_settings.__file__, encoding='utf-8') as f_: self.assertIn('cam_scene=cam_scene.state(APP.LIVE)', f_.read())


class ResetButton(LiveCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.live.proc = NS(poll=lambda: None, terminate=lambda: self.calls.append('terminate'), wait=lambda t_: 0)
        self.live.last_frame_t = time.monotonic()   # frames come
        self.live.cam_scene = dict(at='16:38:31', reset=False, line='')
        APP.cam_usb_seen = lambda: True
        APP.cam_usb_gone = lambda: self.calls.append('gone-check') or False
        APP.cam_usb_power = lambda: (self.calls.append('power'), ('cut', 'USB 전원 껐다 켬, 카메라 다시 보임. '))[1]
        APP.lv_sp = NS(run=lambda *a, **k: self.calls.append(('run', a[0])))
        p_ = patch.object(camera_link.time, 'sleep', lambda s_: None); p_.start(); self.addCleanup(p_.stop)

    def spawn(self):
        super().spawn(); cam_scene.on_spawn(APP.LIVE)

    def press(self):
        camera_link.cam_reset_work()
        self.assertFalse(any(isinstance(c_, tuple) for c_ in self.calls), self.calls)
        self.assertNotIn('초기화 오류', self.live.cam_reset['text'])
        self.assertFalse(self.live.cam_hold)
        return self.live.cam_reset['text']

    def test_no_scene_on_a_root_port_answers_with_the_replug_notice_only(self):
        self.cam('3-1', 3, 6)
        text_ = self.press()
        self.assertEqual(text_, cam_scene.notice(False)); self.assertEqual(self.calls, []); self.assertEqual(self.spawns, [])

    def test_no_scene_behind_the_hub_cuts_the_power(self):
        self.cam('1-1.2', 1, 12)
        self.press()
        self.assertEqual(self.calls, ['terminate', 'power']); self.assertEqual(len(self.spawns), 1)
        self.assertIsNone(self.live.cam_scene)

    def test_scene_present_keeps_the_old_restart(self):
        self.cam('3-1', 3, 6); self.live.cam_scene = None
        self.press()
        self.assertEqual(self.calls, ['terminate']); self.assertEqual(len(self.spawns), 1)


if __name__ == '__main__':
    unittest.main()
