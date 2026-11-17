"""A paused (or finished and held) replay of a labelled recording keeps its picture and position.

2026-11-13: about 10 s after a pause the frame loop's input wait expired (hold packets are not queued for a labelled
recording), lv_cam_wait() replaced the paused frame with "WAITING FOR VIDEO" and "NO CAMERA FRAME" and cleared the replay
position, which locked the page controls. Reproduced here without a device; see camera_link.replay_holds_picture.

Run from the tracking_server folder: python3 -m unittest src.camera.test_replay_hold_picture
"""
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from . import camera_link, replay_eval, replay_link
from ..common.state import APP
from ..web import page_handler as page
from ..web.test_preview_pipeline import PublishFixture


class HeldReplayPicture(PublishFixture):
    def show_frame_37(self):
        """Replay on, async preview as on the device, frame 37 published. Returns (worker, its JPEG)."""
        worker = self.enable(True, async_replay=True)
        live = self.live
        live.replay = dict(name='fixture', fps=25.0, loop=False, bits=8, frames=4027, fin0=0, paused=False, preview='async')
        live.proc = NS(poll=lambda: None)   # replay program alive
        page.preview_worker.set_preview_producer(live, live.proc, before=time.perf_counter() - .1)
        live.frames_in = 38; live.replay_busy = False; live.replay_msg = None
        live.cam_fail_n = 0; live.cam_next_spawn = None; live.cam_lost = False; live.cam_hold = False
        live.proc_t0 = time.monotonic() - 5

        def frame_37(stamp, rows, *args, preview_receipt):
            preview_receipt.append(('run', 37, 1, stamp))
            return None, None

        for target, name, value in ((page.replay_link, 'top_text', lambda *a, **k: 'REPLAY fixture'),
                                    (replay_eval, 'EV', NS(run_id='run'))):
            patcher = patch.object(target, name, value); patcher.start(); self.addCleanup(patcher.stop)
        with patch.object(page.replay_eval, 'on_frame', side_effect=frame_37):
            self.publish()
            with live.jcond:
                self.assertTrue(live.jcond.wait_for(lambda: live.pic_n == 1, timeout=3))
        self.assertEqual(replay_link.state()['pos'], 37)
        return worker, live.jpeg

    def cam_wait(self):
        """One expiry of the frame loop's 10 s input wait; returns the pictures it published."""
        published = []
        commit = page.preview_worker.publish_jpeg_locked

        def record(live_, jpeg, frame_t, **kwargs):
            published.append(jpeg)
            return commit(live_, jpeg, frame_t, **kwargs)

        with patch.object(page.preview_worker, 'publish_jpeg_locked', side_effect=record), \
             patch.object(camera_link, 'publish_jpeg_locked', side_effect=record):
            camera_link.lv_cam_wait()
        return published

    def test_paused_replay_keeps_its_frame_and_position(self):
        worker, paused_jpeg = self.show_frame_37()
        live = self.live
        live.replay['paused'] = True
        for _ in range(3):                                   # three expiries = 30 s of pause on the device
            live.last_frame_t = time.monotonic()             # hold packets keep arriving, 5 a second
            self.assertEqual(self.cam_wait(), [])
        self.assertIs(live.jpeg, paused_jpeg)
        state = replay_link.state()
        self.assertEqual((state['on'], state['busy'], state['paused'], state['pos']), (True, False, True, 37))
        self.assertEqual(worker.stats(), dict(preview_dropped=0, preview_errors=0))

    def test_held_final_frame_is_kept_too(self):
        _, final_jpeg = self.show_frame_37()
        self.live.last_frame_t = time.monotonic()            # "REPLAY END: holding final frame" sends the same hold packets
        self.assertEqual(self.cam_wait(), [])
        self.assertIs(self.live.jpeg, final_jpeg)

    def test_silent_replay_program_still_gets_the_placeholder(self):
        _, paused_jpeg = self.show_frame_37()
        self.live.proc_t0 = time.monotonic() - 60
        self.live.last_frame_t = time.monotonic() - 6        # delivered after its start, then silent for 6 s: a real stall
        published = self.cam_wait()
        self.assertEqual(len(published), 2)                  # transition picture, then the no-frame picture
        self.assertIsNot(self.live.jpeg, paused_jpeg)
        self.assertIsNone(replay_link.state()['pos'])

    def test_replay_that_has_shown_nothing_keeps_the_loading_picture(self):
        worker = self.enable(True, async_replay=True)
        live = self.live
        live.replay = dict(name='fixture', fps=25.0, loop=False, bits=8, frames=4027, fin0=0, paused=False, preview='async')
        live.proc = NS(poll=lambda: None)
        page.preview_worker.set_preview_producer(live, live.proc, before=time.perf_counter() - .1)
        live.replay_busy = False; live.replay_msg = None; live.cam_fail_n = 0; live.cam_next_spawn = None
        live.cam_lost = False; live.cam_hold = False
        live.proc_t0 = time.monotonic() - 12; live.last_frame_t = live.proc_t0 - 1    # nothing from this program yet
        with patch.object(camera_link.replay_link, 'top_text', return_value='REPLAY fixture'):
            self.assertGreaterEqual(len(self.cam_wait()), 1)
        self.assertIsNone(getattr(live, 'jpeg_frame_t', None))
        del worker


class HoldRule(unittest.TestCase):
    def setUp(self):
        try: self.before = (APP.LIVE,)
        except NameError: self.before = ()
        now = time.monotonic()
        APP.LIVE = NS(replay=dict(name='fixture'), cam_lost=False, proc_t0=now - 30, last_frame_t=now - 0.2, jpeg_frame_t=123.0)
        self.now = now

    def tearDown(self):
        if self.before: APP.LIVE = self.before[0]
        else: delattr(APP, 'LIVE')

    def test_holding(self):
        self.assertTrue(camera_link.replay_holds_picture(self.now))
        APP.LIVE.last_frame_t = self.now - 3.5               # three quick seeks: 1.2 s of silence each
        self.assertTrue(camera_link.replay_holds_picture(self.now))

    def test_not_holding(self):
        cases = dict(replay=None, cam_lost=True, last_frame_t=self.now - 5.5, jpeg_frame_t=None, proc_t0=self.now)
        for name, value in cases.items():
            with self.subTest(changed=name):
                old = getattr(APP.LIVE, name)
                setattr(APP.LIVE, name, value)
                self.assertFalse(camera_link.replay_holds_picture(self.now))
                setattr(APP.LIVE, name, old)


if __name__ == '__main__':
    unittest.main()
