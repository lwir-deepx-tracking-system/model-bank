"""PC-only tests of the follow handoff (LWIR_FOLLOW_HANDOFF=1). Run from tracking_server: python -m unittest src.gimbal.test_follow_handoff"""
from collections import deque
from threading import Condition
from types import SimpleNamespace as NS
import os
import unittest
from unittest.mock import patch

from ..common.state import APP
from . import live_tracks, gimbal_link


class FollowHandoffTests(unittest.TestCase):
    def setUp(self):
        self.saved = APP.__dict__.copy()
        APP.tracker = object()
        APP.LIVE = NS(camera_bytetrack=None)
        APP.LIVE_W, APP.LIVE_H = 640, 480
        APP.CTL = dict(lv=None, lead_px=0, lead=False, lead_ms=150, lead_motion=None)
        APP.GB = NS(wcond=Condition(), wopen=True, wq=deque(), wframe=None, people=[], frame=0, gui=False, skipped=0,
                    track_control_supported=True, state=dict(id=0, auto=0))
        APP.gb_send_frame = gimbal_link.gb_send_frame
        APP.gb_send = gimbal_link.gb_send
        self.t = 1000.

    def tearDown(self):
        APP.__dict__.clear()
        APP.__dict__.update(self.saved)

    def frame(self, people, dt=1 / 30, acknowledge=False):
        self.t += dt
        with patch('time.perf_counter', return_value=self.t + .01):
            live_tracks.publish(people, None, self.t, None, live_tracks.current_source())
        if acknowledge:   # the bridge's answer to the first frame's boundary (as in test_live_tracks); a later one would end following
            live_tracks.halted(APP.GB.tracks_ticket)

    def follow(self, pid):
        APP.GB.state.update(id=pid, auto=1)
        APP.GB.selection_reset = False

    def selections(self):
        return [c for c in APP.GB.wq if isinstance(c, bytes) and c.startswith(b'A ')]

    def run_case(self, after, seconds=1.6):
        """id 7 followed and seen, then gone; `after` gives the boxes of the frames after that."""
        self.frame([(7, 100., 50., 300., 450., .9)], acknowledge=True)
        self.follow(7)
        for _ in range(5): self.frame([(7, 100., 50., 300., 450., .9)])
        for k in range(int(seconds * 30)): self.frame(after(k))

    def test_off_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('LWIR_FOLLOW_HANDOFF', None)
            self.run_case(lambda k: [(8, 95., 60., 295., 450., .8)])
        self.assertEqual(self.selections(), [])

    def test_a_box_that_stays_in_the_place_takes_over_once(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_HANDOFF': '1'}):
            self.run_case(lambda k: [(8, 95., 60., 295., 450., .8)])
        self.assertEqual(self.selections(), [b'A 8\n'])

    def test_not_before_the_old_id_has_been_missing_a_while(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_HANDOFF': '1'}):
            self.run_case(lambda k: [(8, 95., 60., 295., 450., .8)], seconds=.7)
        self.assertEqual(self.selections(), [])

    def test_the_old_id_coming_back_cancels_it(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_HANDOFF': '1'}):
            self.run_case(lambda k: [(8, 95., 60., 295., 450., .8)] + ([(7, 100., 50., 300., 450., .9)] if k % 20 == 19 else []))
        self.assertEqual(self.selections(), [])

    def test_a_box_somewhere_else_or_passing_through_is_not_taken(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_HANDOFF': '1'}):
            self.run_case(lambda k: [(8, 420., 60., 620., 450., .8)])
        self.assertEqual(self.selections(), [])
        self.tearDown(); self.setUp()
        with patch.dict(os.environ, {'LWIR_FOLLOW_HANDOFF': '1'}):   # crossing: 15 px a frame = 450 px/s
            self.run_case(lambda k: [(8, -100. + 15 * k, 60., 100. + 15 * k, 450., .8)])
        self.assertEqual(self.selections(), [])

    def test_two_boxes_changing_places_never_settle(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_HANDOFF': '1'}):
            self.run_case(lambda k: [(8 + (k // 6) % 2, 95., 60., 295., 450., .8)])
        self.assertEqual(self.selections(), [])

    def test_not_while_a_click_is_pending_or_the_keys_move_the_gimbal(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_HANDOFF': '1'}):
            APP.GB.mover = '1.2.3.4'
            self.run_case(lambda k: [(8, 95., 60., 295., 450., .8)])
        self.assertEqual(self.selections(), [])


if __name__ == '__main__':
    unittest.main()
