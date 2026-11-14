"""PC-only tests of the search re-catch (bridge option --search 1). Run from tracking_server: python -m unittest src.gimbal.test_search_recatch"""
from collections import deque
from threading import Condition
from types import SimpleNamespace as NS
import os
import unittest
from unittest.mock import patch

from ..common.state import APP
from . import live_tracks, gimbal_link


class SearchRecatchTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'LWIR_SEARCH_NEW_ID': '1'})
        self.env.start()
        self.saved = APP.__dict__.copy()
        APP.tracker = object()
        APP.LIVE = NS(camera_bytetrack=None)
        APP.LIVE_W, APP.LIVE_H = 640, 480
        APP.CTL = dict(lv=None, lead_px=0, lead=False, lead_ms=150, lead_motion=None, motion_hist=deque(maxlen=32))
        APP.GB = NS(wcond=Condition(), wopen=True, wq=deque(), wframe=None, people=[], frame=0, gui=False, skipped=0,
                    track_control_supported=True, state=dict(id=0, auto=0),
                    search_control_supported=True)
        APP.gb_send_frame = gimbal_link.gb_send_frame
        APP.gb_send = gimbal_link.gb_send
        self.t = 1000.
        self.pan = 0.   # camera pan in picture px: the scene moves the other way

    def tearDown(self):
        self.env.stop()
        APP.__dict__.clear()
        APP.__dict__.update(self.saved)

    def frame(self, people, dt=1 / 30, acknowledge=False, pan_step=0., motion=True):
        """people in SCENE coordinates; shown shifted by the camera pan. pan_step = camera turn this frame (px)."""
        t0 = self.t
        self.t += dt
        self.pan += pan_step
        if motion:
            APP.CTL['motion_hist'].append((t0, self.t, [[1., 0., -pan_step], [0., 1., 0.]]))
            APP.CTL['lead_motion'] = APP.CTL['motion_hist'][-1]
        shown = [(i, x1 - self.pan, y1, x2 - self.pan, y2, c) for i, x1, y1, x2, y2, c in people]
        with patch('time.perf_counter', return_value=self.t + .01):
            live_tracks.publish(shown, None, self.t, None, live_tracks.current_source())
        if acknowledge:
            live_tracks.halted(APP.GB.tracks_ticket)

    def selections(self):
        return [c for c in APP.GB.wq if isinstance(c, bytes) and c.startswith(b'A ')]

    def run_case(self, after, n=30, pan_step=-8., others=(), **kw):
        """id 7 followed and seen at scene x 300..380 (with `others`), then gone while the camera turns; `after(k)` = later boxes."""
        me = (7, 300., 100., 380., 325., .9)
        self.frame([me, *others], acknowledge=True)
        APP.GB.state.update(id=7, auto=1)
        APP.GB.selection_reset = False
        for _ in range(5):
            self.frame([me, *others])
        for k in range(n):
            self.frame(after(k), pan_step=pan_step, **kw)

    def test_off_without_the_bridge_option(self):
        APP.GB.search_control_supported = False   # the bridge did not say SEARCH_CONTROL 1
        self.run_case(lambda k: [(9, 300., 100., 380., 325., .8)])
        self.assertEqual(self.selections(), [])

    def test_off_by_default_even_with_search(self):   # the user's rule: a re-catch gives back the SAME id
        os.environ.pop('LWIR_SEARCH_NEW_ID')
        self.run_case(lambda k: [(9, 302., 102., 382., 327., .8)] if k >= 10 else [])
        self.assertEqual(self.selections(), [])

    def sent(self):
        return APP.GB.wframe.split(b'\n', 1)[0].decode()

    def test_vel_is_the_predictive_lead_velocity(self):
        me = lambda x: (7, x, 100., x + 80., 325., .9)
        self.frame([me(300.)], acknowledge=True)
        APP.GB.state.update(id=7, auto=1)
        APP.GB.selection_reset = False
        for k in range(1, 12):
            self.frame([me(300. + 6. * k)], pan_step=0.)   # 6 px a frame = 180 px/s, camera still
        vel = self.sent().split(' VEL ')[1].split()
        self.assertEqual(vel[0], '7')
        self.assertGreater(float(vel[1]), 100.)    # the 120 ms filter of predictive_lead is still rising after 11 frames
        self.assertLess(float(vel[1]), 181.)
        self.assertNotIn(' LEAD ', self.sent())
        APP.GB.search_control_supported = False      # an older bridge: never an unknown suffix
        self.frame([me(372.)])
        self.assertNotIn(' VEL ', self.sent())

    def test_a_new_id_where_the_person_must_be_is_taken_once(self):
        self.run_case(lambda k: [(9, 302., 102., 382., 327., .8)] if k >= 10 else [])
        self.assertEqual(self.selections(), [b'A 9\n'])

    def test_the_camera_motion_is_used(self):
        # the same box, but at the place it had in the PICTURE when the person was lost (the camera turned 240 px since)
        self.run_case(lambda k: [(9, 300. + self.pan, 100., 380. + self.pan, 325., .8)] if k >= 10 else [])
        self.assertEqual(self.selections(), [])

    def test_not_in_a_crowd(self):
        self.run_case(lambda k: [(9, 302., 102., 382., 327., .8), (10, 380., 100., 460., 325., .8)] if k >= 10 else [])
        self.assertEqual(self.selections(), [])

    def test_not_someone_who_was_already_there(self):
        other = (12, 520., 100., 600., 325., .9)
        self.run_case(lambda k: [(12, 305., 100., 385., 325., .9)] if k >= 10 else [other], others=(other,))
        self.assertEqual(self.selections(), [])

    def test_not_a_different_size(self):
        self.run_case(lambda k: [(9, 310., 200., 360., 320., .8)] if k >= 10 else [])
        self.assertEqual(self.selections(), [])

    def test_not_a_short_flash(self):
        self.run_case(lambda k: [(9, 302., 102., 382., 327., .8)] if k in (10, 11) else [])
        self.assertEqual(self.selections(), [])

    def test_unknown_camera_motion_stops_the_rule_for_this_loss(self):
        self.run_case(lambda k: [(9, 302., 102., 382., 327., .8)] if k >= 10 else [], motion=False)
        self.assertEqual(self.selections(), [])

    def test_the_same_id_back_needs_nothing(self):
        self.run_case(lambda k: [(7, 302., 102., 382., 327., .8)] if k >= 10 else [])
        self.assertEqual(self.selections(), [])

    def test_the_plain_handoff_is_off_while_searching(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_HANDOFF': '1'}):   # box left in the old PICTURE place: handoff would take it
            self.run_case(lambda k: [(9, 300. + self.pan, 100., 380. + self.pan, 325., .8)] if k >= 10 else [], n=60)
        self.assertEqual(self.selections(), [])
        self.assertEqual(getattr(APP.GB, 'handoff_n', 0), 0)


if __name__ == '__main__':
    unittest.main()
