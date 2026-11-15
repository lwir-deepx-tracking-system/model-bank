"""PC-only tests of the followed-person prediction (bridge --coast 1) and the wrong-person guard (LWIR_FOLLOW_GUARD=1).
Run from tracking_server: python -m unittest src.gimbal.test_follow_guard"""
from collections import deque
from threading import Condition
from types import SimpleNamespace as NS
import os
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from . import follow_guard as F
from . import live_tracks, gimbal_link

DT = 1 / 30


def person(i, cx, cy=200., h=240., w=90.):
    return (i, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, .9)


class Scene:
    """Scene coordinates; the picture shows them shifted by the camera pan. motion() = the GMC warps of the frame pairs."""

    def __init__(self, gmc=True):
        self.t, self.pan, self.hist, self.gmc = 100., 0., {}, gmc

    def step(self, pan_step=0.):
        t0 = self.t
        self.t = round(self.t + DT, 6)
        self.pan += pan_step
        if self.gmc:
            self.hist[t0] = (self.t, np.array([[1., 0., -pan_step], [0., 1., 0.]]))
        return self.t

    def motion(self, t0, t1):
        out, t = np.eye(3), t0
        while t < t1:
            if t not in self.hist: return None
            t, w = self.hist[t]
            out = np.vstack([w, [0, 0, 1]]) @ out
        return out[:2] if t == t1 else None

    def shown(self, people):
        return [(p[0], p[1] - self.pan, p[2], p[3] - self.pan, p[4], p[5]) for p in people]


class PredictionTests(unittest.TestCase):
    def walk(self, g, s, n, x0, v, pan_step=0., present=True):
        out = []
        for k in range(n):
            t = s.step(pan_step)
            x = x0 + v * (s.t - 100.)
            out.append((x, g.step(t, s.shown([person(7, x)]) if present else [], s.motion)))
        return out

    def test_the_prediction_moves_with_the_person_not_with_the_camera(self):
        s, g = Scene(), F.FollowGuard(7, guard=False)
        self.walk(g, s, 30, 200., 90., pan_step=4.)        # walking right at 90 px/s, the camera turns 120 px/s
        lost = self.walk(g, s, 15, 200., 90., pan_step=4., present=False)
        x, d = lost[-1]
        self.assertIsNotNone(d['pred'])
        self.assertAlmostEqual(d['pred'][0], x - s.pan, delta=6.)   # where the person is in the picture 0.5 s later
        self.assertFalse(d['mask']); self.assertEqual(d['reselect'], 0)

    def test_offered_for_a_bounded_time_only(self):
        s, g = Scene(), F.FollowGuard(7, guard=False)
        self.walk(g, s, 30, 200., 90.)
        lost = self.walk(g, s, int(F.COAST_S / DT) + 6, 200., 90., present=False)
        self.assertIsNotNone(lost[int(F.COAST_S / DT) - 3][1]['pred'])
        self.assertIsNone(lost[-1][1]['pred'])

    def test_no_camera_motion_no_prediction(self):
        s, g = Scene(gmc=False), F.FollowGuard(7, guard=False)
        self.walk(g, s, 20, 200., 90.)
        self.assertIsNone(self.walk(g, s, 5, 200., 90., present=False)[-1][1]['pred'])

    def test_no_prediction_while_seen(self):
        s, g = Scene(), F.FollowGuard(7, guard=True)
        self.assertTrue(all(d['pred'] is None for _, d in self.walk(g, s, 30, 200., 90.)))


class KeepInViewTests(unittest.TestCase):
    AIM = (320., 160.)

    def control(self, b, frac=.37):
        return b[0], b[1] - b[3] / 2 + frac * b[3]

    def test_inside_the_picture_no_motion(self):
        b = F.keep_in_view([(300., 220., 60., 160.), (360., 230., 60., 160.)], .2, self.AIM, .37)
        self.assertEqual(tuple(round(v, 6) for v in self.control(b)), self.AIM)

    def test_near_the_edge_the_smallest_shift(self):
        b = F.keep_in_view([(560., 220., 60., 160.)], .2, self.AIM, .37)
        r = (F.KIV_U0 + F.KIV_U1 * .2) * 160.
        self.assertAlmostEqual(self.control(b)[0], 320. + (560. + 30. + r) - 640. * (1 - F.KIV_MARGIN), delta=.01)
        self.assertAlmostEqual(self.control(b)[1], 160., delta=.01)

    def test_estimates_disagree_aim_between_them(self):
        b = F.keep_in_view([(40., 220., 60., 160.), (700., 220., 60., 160.)], 1., self.AIM, .37)
        self.assertAlmostEqual(self.control(b)[0], 370., delta=.01)   # the middle of the region (x 40..700 widened)

    def test_nothing_to_keep(self):
        self.assertIsNone(F.keep_in_view([None], .2, self.AIM, .37))


class GuardTests(unittest.TestCase):
    def cross(self, guard, swap, n_after=20):
        """7 (followed) walks right, 8 walks left; they overlap around x 320. swap: from the overlap on, the tracker has the
        ids the other way round. Returns the decisions."""
        s, g = Scene(), F.FollowGuard(7, guard=guard)
        out = []
        for k in range(60 + n_after):
            t = s.step(0.)
            a, b = 200. + 120. * (t - 100.), 440. - 120. * (t - 100.)
            ia, ib = (8, 7) if swap and abs(a - b) < 80 or swap and a > b else (7, 8)
            out.append(g.step(t, [person(ia, a), person(ib, b, h=236.)], s.motion))
            if out[-1]['reselect']:
                g.sel = out[-1]['reselect']   # live_tracks sends 'A id'; the bridge then follows it
        return out

    def test_a_swap_in_a_crossing_moves_following_back_to_the_person(self):
        d = self.cross(guard=True, swap=True)
        self.assertEqual([x['reselect'] for x in d if x['reselect']], [8])

    def test_a_crossing_without_swap_changes_nothing(self):
        d = self.cross(guard=True, swap=False)
        self.assertFalse(any(x['reselect'] or x['mask'] for x in d))

    def test_off_never_masks_or_reselects(self):
        d = self.cross(guard=False, swap=True)
        self.assertFalse(any(x['reselect'] or x['mask'] for x in d))

    def test_walking_side_by_side_is_not_hidden(self):
        # 7 and 8 walk together, partly overlapping (IoU ~0.15), for 4 s while the camera follows: the box keeps its trust
        s, g = Scene(), F.FollowGuard(7, guard=True)
        out = []
        for k in range(120):
            t = s.step(3.)
            x = 200. + 90. * (t - 100.)
            out.append(g.step(t, s.shown([person(7, x), person(8, x + 65.)]), s.motion))
        self.assertFalse(any(d['mask'] or d['reselect'] for d in out))

    def lose_then(self, later, n=40):
        """7 walks right and vanishes behind 8 (standing still); later(k, s) = boxes after the loss (scene coordinates)."""
        s, g = Scene(), F.FollowGuard(7, guard=True)
        occluder = person(8, 420., h=260., w=100.)
        for k in range(25):
            t = s.step(2.)
            g.step(t, s.shown([person(7, 200. + 90. * (t - 100.)), occluder]), s.motion)
        out = []
        for k in range(n):
            t = s.step(2.)
            out.append(g.step(t, s.shown(later(k, t)), s.motion))
            if out[-1]['reselect']: g.sel = out[-1]['reselect']
        return out

    def test_the_old_id_back_on_someone_else_is_hidden(self):
        d = self.lose_then(lambda k, t: [person(7, 600.), person(8, 420., h=260., w=100.)] if k >= 3 else [person(8, 420., h=260., w=100.)])
        self.assertTrue(all(x['mask'] for x in d[4:]))
        self.assertTrue(all(x['reselect'] == 0 for x in d))

    def test_a_new_id_at_the_prediction_is_taken(self):
        d = self.lose_then(lambda k, t: [person(9, 200. + 90. * (t - 100.)), person(8, 600., h=260., w=100.)] if k >= 10 else [])
        self.assertEqual([x['reselect'] for x in d if x['reselect']], [9])

    def test_the_old_id_back_at_the_prediction_is_followed(self):
        d = self.lose_then(lambda k, t: [person(7, 200. + 90. * (t - 100.))] if k >= 10 else [])
        self.assertFalse(any(x['mask'] or x['reselect'] for x in d))
        self.assertEqual(d[-1]['phase'], 'track')

    def test_someone_who_was_already_there_is_not_taken(self):
        d = self.lose_then(lambda k, t: [person(8, 200. + 90. * (t - 100.), h=245.)] if k >= 10 else [person(8, 420., h=260., w=100.)])
        self.assertTrue(all(x['reselect'] == 0 for x in d))


class GuardV2Tests(unittest.TestCase):
    FA, FB = np.r_[1., 0., 0.], np.r_[.3, .95, 0.]   # appearance of the clicked person / of the other one

    def cross(self, swap, n_after=25):
        s, g = Scene(), F.FollowGuardV2(7)
        out = []
        for k in range(60 + n_after):
            t = s.step(0.)
            a, b = 200. + 120. * (t - 100.), 440. - 120. * (t - 100.)
            ia, ib = (8, 7) if swap and (abs(a - b) < 80 or a > b) else (7, 8)
            feats = {ia: self.FA, ib: self.FB}
            close = abs(a - b) < 100   # no feature while they overlap (crops are not clean)
            g.app = lambda oid, f=feats, c=close: None if c else (f.get(oid), 0)
            out.append(g.step(t, [person(ia, a), person(ib, b, h=236.)], s.motion))
            if out[-1]['reselect']: g.sel = out[-1]['reselect']
        return out

    def test_swap_is_handed_back_by_appearance(self):
        d = self.cross(True)
        self.assertEqual([x['reselect'] for x in d if x['reselect']], [8])

    def test_crossing_without_swap_changes_nothing(self):
        d = self.cross(False)
        self.assertFalse(any(x['reselect'] or x['mask'] for x in d))

    def test_occluder_with_another_look_is_not_taken(self):
        s, g = Scene(), F.FollowGuardV2(7)
        feats = {7: self.FA, 8: self.FB}
        g.app = lambda oid: (feats.get(oid), 0) if oid in feats else None
        occ = person(8, 420., h=245., w=100.)
        out = []
        for k in range(70):
            t = s.step(0.)
            x = 200. + 90. * (t - 100.)
            people = [person(7, x), occ] if k < 25 else [occ] if k < 50 else [person(8, x, h=245.), person(9, 650.)]
            out.append(g.step(t, people, s.motion))
        self.assertTrue(all(d['reselect'] == 0 for d in out))

    def test_new_id_that_looks_like_the_person_is_taken(self):
        s, g = Scene(), F.FollowGuardV2(7)
        feats = {7: self.FA, 9: self.FA}
        g.app = lambda oid: (feats.get(oid), 0) if oid in feats else None
        out = []
        for k in range(50):
            t = s.step(0.)
            x = 200. + 90. * (t - 100.)
            out.append(g.step(t, [person(7, x)] if k < 25 else [] if k < 35 else [person(9, x)], s.motion))
            if out[-1]['reselect']: g.sel = out[-1]['reselect']
        self.assertEqual([d['reselect'] for d in out if d['reselect']], [9])


class PublishFixture(unittest.TestCase):
    def setUp(self):
        self.saved = APP.__dict__.copy()
        APP.tracker = object()
        APP.LIVE = NS(camera_bytetrack=None)
        APP.LIVE_W, APP.LIVE_H = 640, 480
        APP.CTL = dict(lv=None, lead_px=0, lead=False, lead_ms=150, lead_motion=None, motion_hist=deque(maxlen=32))
        APP.GB = NS(wcond=Condition(), wopen=True, wq=deque(), wframe=None, people=[], frame=0, gui=False, skipped=0,
                    track_control_supported=True, state=dict(id=0, auto=0), coast_control_supported=True)
        APP.gb_send_frame = gimbal_link.gb_send_frame
        self.t = 1000.

    def tearDown(self):
        APP.__dict__.clear()
        APP.__dict__.update(self.saved)

    def frame(self, people, acknowledge=False):
        t0 = self.t
        self.t += DT
        APP.CTL['motion_hist'].append((t0, self.t, [[1., 0., 0.], [0., 1., 0.]]))
        APP.CTL['lead_motion'] = APP.CTL['motion_hist'][-1]
        with patch('time.perf_counter', return_value=self.t + .01):
            live_tracks.publish(people, None, self.t, None, live_tracks.current_source())
        if acknowledge:
            live_tracks.halted(APP.GB.tracks_ticket)
        return APP.GB.wframe.split(b'\n', 1)[0].decode()

    def follow(self, n=10):
        self.frame([person(7, 300.)], acknowledge=True)
        APP.GB.state.update(id=7, auto=1)
        APP.GB.selection_reset = False
        for k in range(n):
            self.frame([person(7, 300. + 3. * k)])


class PublishTests(PublishFixture):
    def test_pred_only_while_missing_and_only_to_a_coast_bridge(self):
        self.follow()
        self.assertNotIn(' PRED ', self.frame([person(7, 330.)]))
        sent = self.frame([])
        self.assertIn(' PRED 7 ', sent)
        x, y, w, h = map(float, sent.split(' PRED 7 ')[1].split()[:4])
        self.assertAlmostEqual(x + w / 2, 333., delta=6.)
        APP.GB.coast_control_supported = False   # an older bridge refuses unknown suffixes
        self.assertNotIn(' PRED ', self.frame([]))

    def test_guard_hides_the_old_id_on_another_person(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_GUARD': '1'}):
            self.follow()
            for k in range(3): self.frame([])
            sent = self.frame([person(7, 600.)])   # the old id comes back far from where the person must be
            self.assertNotIn(' 7 ', sent.split(' PRED ')[0])
            self.assertIn(' PRED 7 ', sent)
            self.assertEqual(APP.GB.state['guard'][1], 1)

    def test_off_by_default(self):
        APP.GB.coast_control_supported = False
        self.follow()
        for k in range(3): self.frame([])
        sent = self.frame([person(7, 600.)])
        self.assertIn('7 555.0', sent)
        self.assertNotIn('guard', APP.GB.state)

    def test_prediction_log_only_changes_nothing_sent(self):
        APP.GB.coast_control_supported = False
        with patch.dict(os.environ, {'LWIR_FOLLOW_PREDICT_LOG': '1'}):
            self.follow()
            sent = self.frame([])
            self.assertNotIn(' PRED ', sent)
            self.assertEqual(APP.GB.state['guard'][3], 0)
            self.assertIsNotNone(APP.GB.state['guard'][2])
            self.assertIn('7 555.0', self.frame([person(7, 600.)]))   # never hidden


class CoastSwitchTests(PublishFixture):
    """11-15 live A/B: LWIR_COAST_SWITCH=<file> turns the coast on / off at run time without a bridge restart."""

    def setUp(self):
        super().setUp()
        import tempfile
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'coast_on')
        self.env = patch.dict(os.environ, {'LWIR_COAST_SWITCH': self.path, 'LWIR_FOLLOW_PREDICT_LOG': '1'})
        self.env.start()
        self.fast = patch.object(live_tracks, 'COAST_SWITCH_S', 0.)   # read the file on every frame
        self.fast.start()
        live_tracks.COAST_SWITCH.update(path=None, said=None)

    def tearDown(self):
        self.fast.stop()
        self.env.stop()
        self.dir.cleanup()
        live_tracks.COAST_SWITCH.update(path=None, said=None)
        super().tearDown()

    def switch(self, word=None):
        if word is None:
            if os.path.exists(self.path): os.remove(self.path)
        else:
            with open(self.path, 'w') as f: f.write(word + '\n')

    def run_lines(self, prepare):
        """the frame lines of a fixed follow / loss / return sequence, from a fresh fixture (prepare(self) sets it up)"""
        self.tearDown(); self.setUp()
        prepare(self)
        lines = [self.frame([person(7, 300.)], acknowledge=True)]
        APP.GB.state.update(id=7, auto=1)
        APP.GB.selection_reset = False
        for k in range(8): lines.append(self.frame([person(7, 300. + 3. * k)]))
        for k in range(6): lines.append(self.frame([]))
        for k in range(3): lines.append(self.frame([person(7, 340. + 3. * k)]))
        return lines

    def test_off_sends_exactly_what_a_bridge_without_coast_gets(self):
        def off_coast_bridge(t): t.switch(None)
        def old_bridge(t):   # the deployed server: no coast bridge, no switch, no prediction log
            APP.GB.coast_control_supported = False
            os.environ.pop('LWIR_COAST_SWITCH', None)
            os.environ.pop('LWIR_FOLLOW_PREDICT_LOG', None)
        off = self.run_lines(off_coast_bridge)
        self.assertEqual(APP.GB.state['coast_sw'], 'off')
        self.assertEqual(APP.GB.state['guard'][3], 0)   # the prediction is still logged, never sent
        old = self.run_lines(old_bridge)
        self.assertNotIn('coast_sw', APP.GB.state)
        self.assertNotIn('guard', APP.GB.state)
        self.assertTrue(all(' PRED ' not in line for line in off))
        self.assertEqual(off, old)

    def test_on_sends_pred_while_missing(self):
        self.switch('')
        self.follow()
        self.assertNotIn(' PRED ', self.frame([person(7, 330.)]))
        self.assertIn(' PRED 7 ', self.frame([]))
        self.assertEqual(APP.GB.state['coast_sw'], 'vel')
        self.assertEqual(APP.GB.state['guard'][3], 1)

    def test_toggle_keeps_the_selection_and_takes_effect_at_the_next_frame(self):
        self.switch('vel')
        self.follow()
        self.assertIn(' PRED 7 ', self.frame([]))
        self.switch(None)
        self.assertNotIn(' PRED ', self.frame([]))
        self.assertEqual((APP.GB.state['coast_sw'], APP.GB.state['guard'][3]), ('off', 0))
        self.switch('vel')
        self.assertIn(' PRED 7 ', self.frame([]))
        self.assertEqual(APP.GB.state['id'], 7)
        self.assertFalse([c for c in APP.GB.wq if c.startswith(b'A ')])   # no new selection was sent

    def test_the_word_in_the_file_picks_the_mode(self):
        self.switch('fused')
        self.follow()
        sent = self.frame([])
        self.assertEqual(APP.GB.state['coast_sw'], 'fused')
        x, y, w, h = map(float, sent.split(' PRED 7 ')[1].split()[:4])
        self.assertAlmostEqual(x + w / 2, 320., delta=1.)   # keep in view: the person is inside the picture, no motion
        self.switch('vel')
        x, y, w, h = map(float, self.frame([]).split(' PRED 7 ')[1].split()[:4])
        self.assertGreater(x + w / 2, 330.)   # the velocity prediction itself

    def test_without_the_variable_nothing_changes(self):
        os.environ.pop('LWIR_COAST_SWITCH', None)
        self.follow()
        self.assertIn(' PRED 7 ', self.frame([]))
        self.assertNotIn('coast_sw', APP.GB.state)
        self.assertEqual(live_tracks.coast_switch(), (True, None))


if __name__ == '__main__':
    unittest.main()
