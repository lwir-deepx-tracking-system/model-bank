"""PC-only tests of the gimbal search that follows the displayed prediction (bridge --seek 1): search_follow.py, the guard's
wide gate (follow_guard.FollowGuardV2.wide) and the SEEK suffix (live_tracks).
Run from tracking_server: python -m unittest src.gimbal.test_search_follow"""
import math
import os
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from . import follow_guard as F
from . import live_tracks
from . import search_follow as S
from .test_follow_guard import DT, PublishFixture, Scene, person

W, H = 640., 480.
AIM = (320., 240.)


class Camera:
    """A camera that turns toward the point it is told to look at, like the bridge with --seek 1: P control on the error,
    SPD capped, 31 px/s per SPD (measured 11-15), a direction change through a stop. Scene coordinates: the picture shows
    scene x - pan."""

    def __init__(self, cap=8, kp=30., px_per_spd=31., deadband=19.):
        self.cap, self.kp, self.k, self.dead = cap, kp, px_per_spd, deadband
        self.pan, self.speed, self.t = 0., 0., 100.
        self.hist = {}
        self.reversals = 0
        self.max_px_s = 0.

    def step(self, box):
        """one frame with the SEEK box `box` (or None) -> frame time"""
        want = 0.
        if box is not None:
            e = box[0] - AIM[0]
            if abs(e) > self.dead:
                want = math.copysign(min(self.cap, max(1., self.kp * abs(e) / (W / 2))), e) * self.k
        if want and self.speed and (want > 0) != (self.speed > 0):
            self.reversals += 1
            want = 0.   # through a stop
        self.speed = want
        self.max_px_s = max(self.max_px_s, abs(want))
        t0, step = self.t, self.speed * DT
        self.t = round(self.t + DT, 6)
        self.pan += step
        self.hist[t0] = (self.t, np.array([[1., 0., -step], [0., 1., 0.]]))
        return self.t

    def motion(self, t0, t1):
        out, t = np.eye(3), t0
        while t < t1:
            if t not in self.hist: return None
            t, w = self.hist[t]
            out = np.vstack([w, [0, 0, 1]]) @ out
        return out[:2] if t == t1 else None


def run_loss(seconds, *, x0=400., h=240., w=90., vx=0., near=(), cap=8, kal_frames=30, motion=True, hidden=False,
             can_reaccept=True, seek=None, cam=None):
    """The person stood at scene x0 and vanished at t = 100 (camera pan 0). -> (seek, camera, [(age, box, state)])"""
    cam = cam or Camera(cap)
    seek = seek or S.Seek(7)
    trust = (x0, 240., w, h)
    seek.seen(trust, [person(7, x0, 240., h, w)] + list(near))
    out, box = [], None
    for k in range(int(seconds / DT)):
        t = cam.step(box)
        age = t - 100.
        kal = (x0 + vx * min(age, 1.0) - cam.pan, 240., w, h) if k < kal_frames else None   # the tracker's memory, in the picture
        box, st = seek.step(t, age, trust, (vx, 0.), kal, None, cam.motion if motion else None, (W, H), AIM, .5,
                            hidden=hidden, can_reaccept=can_reaccept)
        out.append((age, box, dict(st), cam.pan))
    return seek, cam, out


class GeometryTests(unittest.TestCase):
    def test_the_circle_is_the_displayed_keepid_range(self):
        self.assertAlmostEqual(S.radius(160., 0.), 80.)                 # 0.5 heights at the loss
        self.assertAlmostEqual(S.radius(160., 1.0), 160. * 2.0)        # + 0.05 heights per lost frame
        self.assertAlmostEqual(S.radius(160., 5.0), 160. * 4.0)        # at most 4 heights

    def test_the_sweep_reaches_the_circles_edge_but_keeps_its_centre_in_view(self):
        self.assertEqual(S.amplitude(200., 90., W), 0.)                  # the circle fits in the picture: nothing to do
        self.assertAlmostEqual(S.amplitude(356., 90., W), 100.)         # 100 px of the circle are outside the picture
        self.assertAlmostEqual(S.amplitude(2000., 90., W), 320. - 64. - 45.)   # never so far that the predicted box leaves the picture


class PatternTests(unittest.TestCase):
    def test_nothing_while_the_prediction_is_trusted(self):
        _, cam, out = run_loss(0.75)
        self.assertTrue(all(b is None and s['phase'] == 'predict' for _, b, s, _ in out))
        self.assertEqual(cam.pan, 0.)

    def test_a_small_person_whose_circle_fits_in_the_picture_is_not_swept(self):
        _, cam, out = run_loss(5.0, h=60., w=24., x0=330.)               # radius at most 240 px
        self.assertTrue(all(b is None for _, b, _, _ in out))
        self.assertEqual({s['phase'] for _, _, s, _ in out}, {'predict', 'fits', 'wait'})
        self.assertEqual(cam.pan, 0.)

    def test_sweep_first_to_the_side_the_person_went_then_the_other_then_back(self):
        seek, cam, out = run_loss(5.0, x0=330., vx=60.)                  # walking right
        pans = [p for _, _, _, p in out]
        amax = 320. - 64. - 45.
        first = next(p for a, b, s, p in out if s['phase'] == 'sweep' and s['leg'] == 1)   # the pan when the first turn ended
        self.assertGreater(first, 200.)                                  # the first turn went right, to the circle's edge by then
        back = min(p for a, _, s, p in out if s['phase'] == 'sweep' and s['leg'] == 1)
        self.assertLess(back, first - 150.)                              # the second one came back across (the KeepID window ends it)
        self.assertLessEqual(max(abs(p) for p in pans), amax + 60. + 15.)   # bounded: the circle's centre (330 + 60 x 1 s) +- amplitude
        self.assertEqual([s['phase'] for a, _, s, _ in out if a < .79][-1], 'predict')
        self.assertTrue(all(s['phase'] == 'sweep' for a, _, s, _ in out if .85 < a < 2.95))
        end = out[-1]
        self.assertEqual(end[2]['phase'], 'wait')
        self.assertLessEqual(abs(end[2]['pan']), S.LEG_TOL)              # ended looking at the predicted place
        self.assertIsNone(end[1])
        self.assertEqual(seek.sweeps, 1)

    def test_the_predicted_place_never_leaves_the_picture(self):
        for x0, vx in ((330., 60.), (560., 150.), (80., -150.), (320., 0.)):
            _, cam, out = run_loss(5.0, x0=x0, vx=vx)
            inside = False   # (the coast is not played here: a prediction that starts outside is first brought in by the first turn)
            for age, box, st, pan in out:
                if st['phase'] in ('sweep', 'return'):
                    now = 64. - 12. < st['c'][0] < W - 64. + 12.   # the circle's centre, inside the picture with the margin
                    self.assertTrue(now or not inside, (x0, vx, age, st['c']))
                    inside = inside or now
            self.assertTrue(inside, (x0, vx))

    def test_speed_stays_at_the_cap_and_turns_go_through_a_stop(self):
        _, cam, out = run_loss(5.0, x0=330., vx=60.)
        self.assertAlmostEqual(cam.max_px_s, 8 * 31.)                    # 248 px/s of picture motion, 8 px per frame
        self.assertGreaterEqual(cam.reversals, 1)

    def test_standing_person_near_the_right_edge_is_searched_to_the_right_first(self):
        _, cam, out = run_loss(2.0, x0=560., vx=0.)
        self.assertGreater(out[-1][3], 150.)

    def test_the_point_sent_is_always_inside_the_picture_and_never_tilts(self):
        _, cam, out = run_loss(5.0, x0=600., vx=200., h=300., w=120.)
        for _, box, _, _ in out:
            if box is None: continue
            self.assertTrue(S.EDGE_PX <= box[0] <= W - S.EDGE_PX)
            self.assertAlmostEqual(box[1], AIM[1])                       # aim fraction 0.5: the box centre at the aim height

    def test_other_people_near_the_loss_mean_no_sweep(self):
        seek, cam, out = run_loss(5.0, x0=330., vx=60., near=[person(8, 330. + 1.5 * 240.)])   # 1.5 heights away
        self.assertTrue(all(b is None for _, b, _, _ in out))
        self.assertIn('crowd', {s['phase'] for _, _, s, _ in out})
        self.assertEqual((cam.pan, seek.skipped, seek.sweeps), (0., 1, 0))
        seek, cam, out = run_loss(3.0, x0=100., vx=-60., h=120., w=50., near=[person(8, 100. + 2.5 * 120.)])   # 2.5 heights away: alone enough
        self.assertEqual(seek.sweeps, 1)
        seek, cam, out = run_loss(3.0, x0=100., vx=-60., h=120., w=50.,
                                  near=[person(8, 500.), person(9, 560.), person(10, 620.)])   # three others in the picture
        self.assertEqual((seek.sweeps, cam.pan), (0, 0.))

    def test_a_hidden_number_or_one_that_could_not_be_taken_back_means_no_sweep(self):
        for kw in (dict(hidden=True), dict(can_reaccept=False)):
            seek, cam, out = run_loss(4.0, x0=330., vx=60., **kw)
            self.assertTrue(all(b is None for _, b, _, _ in out))
            self.assertIn('hold', {s['phase'] for _, _, s, _ in out})
            self.assertEqual(cam.pan, 0.)

    def test_without_camera_motion_there_is_no_search(self):
        seek, cam, out = run_loss(4.0, x0=330., vx=60., motion=False)
        self.assertTrue(all(b is None for _, b, _, _ in out))
        self.assertIn('blind', {s['phase'] for _, _, s, _ in out})

    def test_a_turn_that_never_arrives_is_given_up(self):
        class Stuck(Camera):   # at an end stop: commanded, but the picture does not move
            def step(self, box):
                t0 = self.t; self.t = round(self.t + DT, 6)
                self.hist[t0] = (self.t, np.array([[1., 0., 0.], [0., 1., 0.]]))
                return self.t
        with patch.object(S, 'SWEEP_END_S', 8.0):
            seek, cam, out = run_loss(7.0, x0=330., vx=60., cam=Stuck())
        legs = [s['leg'] for a, _, s, _ in out if s['phase'] == 'sweep']
        self.assertEqual(legs[0], 0)
        self.assertGreaterEqual(legs[-1], 2)                             # 2.5 s per turn at most

    def test_seen_again_ends_it_and_the_next_loss_starts_fresh(self):
        seek, cam, out = run_loss(2.0, x0=330., vx=60.)
        self.assertEqual(seek.phase, 'sweep')
        seek.seen((300., 240., 90., 240.), [person(7, 300.)])
        self.assertEqual((seek.phase, seek.c, seek.swept), ('idle', None, False))


class WideGateTests(unittest.TestCase):
    """follow_guard.FollowGuardV2.wide: the same number is taken back anywhere in the search circle when it clearly looks
    like the person; without the circle (or without a clear look) only inside the motion gate, as before."""
    FA, FB = np.r_[1., 0., 0.], np.r_[.3, .95, 0.]

    def back(self, *, wide, feat, x_back=820., frames_lost=45, tmpl=True):
        s, g = Scene(), F.FollowGuardV2(7)
        feats = {7: self.FA}
        g.app = lambda oid: (feats.get(oid), 0) if (tmpl and oid in feats) else None
        out = []
        for k in range(30):
            out.append(g.step(s.step(0.), [person(7, 300.)], s.motion))
        for k in range(frames_lost):
            out.append(g.step(s.step(0.), [], s.motion))
        feats[7] = feat
        g.app = lambda oid: (feats.get(oid), 0) if feats.get(oid) is not None else None
        for k in range(10):
            if wide: g.wide = (300., 200., 4 * 240.)
            out.append(g.step(s.step(0.), [person(7, x_back)], s.motion))
        return g, out[-10:]

    def test_far_from_the_prediction_it_is_hidden_without_the_circle(self):
        g, d = self.back(wide=False, feat=self.FA)
        self.assertTrue(all(x['mask'] for x in d))                       # 2.2 heights from the prediction: the old behaviour

    def test_inside_the_circle_with_the_persons_look_it_is_followed_again(self):
        g, d = self.back(wide=True, feat=self.FA)
        self.assertFalse(any(x['mask'] for x in d))
        self.assertEqual((g.phase, g.wide_reaccepts), ('track', 1))

    def test_another_look_stays_hidden(self):
        g, d = self.back(wide=True, feat=self.FB)                        # similarity 0.3
        self.assertTrue(all(x['mask'] for x in d))
        weak = unit = np.r_[.6, .8, 0.]                                   # similarity 0.6: enough for the motion gate, not for the circle
        g, d = self.back(wide=True, feat=weak)
        self.assertTrue(all(x['mask'] for x in d))

    def test_without_a_feature_it_stays_hidden_until_one_comes(self):
        s, g = Scene(), F.FollowGuardV2(7)
        feats = {7: self.FA}
        g.app = lambda oid: (feats.get(oid), 0) if feats.get(oid) is not None else None
        for k in range(30): g.step(s.step(0.), [person(7, 300.)], s.motion)
        for k in range(45): g.step(s.step(0.), [], s.motion)
        feats[7] = None
        masks = []
        for k in range(8):
            if k == 4: feats[7] = self.FA
            g.wide = (300., 200., 960.)
            masks.append(g.step(s.step(0.), [person(7, 820.)], s.motion)['mask'])
        self.assertEqual(masks, [True] * 4 + [False] * 4)

    def test_outside_the_circle_or_a_very_different_size_stays_hidden(self):
        g, d = self.back(wide=True, feat=self.FA, x_back=300. + 4 * 240. + 60.)
        self.assertTrue(all(x['mask'] for x in d))
        s, g = Scene(), F.FollowGuardV2(7)
        g.app = lambda oid: (self.FA, 0)
        for k in range(30): g.step(s.step(0.), [person(7, 300.)], s.motion)
        for k in range(45): g.step(s.step(0.), [], s.motion)
        g.wide = (300., 200., 960.)
        self.assertTrue(g.step(s.step(0.), [person(7, 820., h=100., w=40.)], s.motion)['mask'])   # 2.4 x smaller


class SeekPublishTests(PublishFixture):
    def setUp(self):
        super().setUp()
        APP.GB.seek_control_supported = True

    def lose(self, n, people=()):
        return [self.frame(list(people)) for _ in range(n)]

    def test_no_seek_suffix_for_a_bridge_without_seek(self):
        APP.GB.seek_control_supported = False
        self.follow()
        sent = self.lose(60)
        self.assertFalse(any(' SEEK ' in x for x in sent))
        self.assertNotIn('seek', APP.GB.state)

    def test_seek_follows_the_coast_and_only_one_of_them_is_sent(self):
        self.follow()
        sent = self.lose(60)
        self.assertIn(' PRED 7 ', sent[5])
        self.assertNotIn(' SEEK ', sent[5])
        both = [x for x in sent if ' PRED ' in x and ' SEEK ' in x]
        self.assertEqual(both, [])
        first = next(k for k, x in enumerate(sent) if ' SEEK 7 ' in x)
        self.assertIn(first, (23, 24))                                    # 0.8 s after the last trusted box
        x, y, w, h = map(float, sent[first].split(' SEEK 7 ')[1].split()[:4])
        self.assertGreater(x + w / 2, 330.)                               # the person walked right: the first turn is to the right
        self.assertAlmostEqual(y + h / 2, 240.)                           # no tilt
        self.assertEqual(APP.GB.state['seek'][0], S.PHASES['sweep'])

    def test_it_ends_in_the_frame_the_number_is_back(self):
        self.follow()
        self.lose(40)
        self.assertNotIn(' SEEK ', self.frame([person(7, 340.)]))
        self.assertNotIn('seek', APP.GB.state)

    def test_people_near_the_loss_no_seek(self):
        self.frame([person(7, 300.), person(8, 520.)], acknowledge=True)
        APP.GB.state.update(id=7, auto=1)
        APP.GB.selection_reset = False
        for k in range(10): self.frame([person(7, 300. + 3. * k), person(8, 520.)])
        sent = self.lose(60, [person(8, 520.)])
        self.assertFalse(any(' SEEK ' in x for x in sent))
        self.assertEqual(APP.GB.state['seek'][0], S.PHASES['crowd'])

    def test_guard_v1_cannot_take_the_number_back_far_away_so_no_seek(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_GUARD': '1'}):
            self.follow()
            sent = self.lose(60)
        self.assertFalse(any(' SEEK ' in x for x in sent))
        self.assertEqual(APP.GB.state['seek'][0], S.PHASES['hold'])

    def test_a_number_that_comes_back_far_away_is_followed_again_only_by_appearance(self):
        """guard v2 + --seek 1, through live_tracks: the number comes back 350 px from the predicted place inside the search
        circle. With the person's look it is sent to the bridge again (and the server says so); with another look it is hidden."""
        fa, fb = np.r_[1., 0., 0.], np.r_[.3, .95, 0.]
        for feat, taken in ((fa, True), (fb, False)):
            self.setUp()
            look = {7: fa}
            look_of = lambda oid: (look[oid], 0) if oid in look else None
            with patch.dict(os.environ, {'LWIR_FOLLOW_GUARD': '2'}), patch.object(live_tracks, '_appearance', look_of), \
                    patch.object(live_tracks, '_kalman_box', lambda sel: None), patch('builtins.print') as said:
                self.follow(20)
                self.lose(45)
                look[7] = feat
                sent = [self.frame([person(7, 700. - 20.)]) for _ in range(4)]
            rows = [x.split(' OBS ')[0].split(' SEEK ')[0].split(' PRED ')[0] for x in sent]
            self.assertEqual([' 7 ' in r for r in rows], [taken] * 4, (taken, sent))
            lines = ' '.join(str(c.args[0]) for c in said.call_args_list if c.args)
            self.assertEqual('followed again inside the search circle' in lines, taken)
            self.tearDown()
        self.setUp()

    def test_the_runtime_switch_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'seek_on')
            with patch.dict(os.environ, {'LWIR_SEEK_SWITCH': path}), patch.object(live_tracks, 'COAST_SWITCH_S', 0.0):
                self.follow()
                self.assertFalse(any(' SEEK ' in x for x in self.lose(40)))
                open(path, 'w').close()
                self.assertTrue(any(' SEEK ' in x for x in self.lose(10)))
        live_tracks.SEEK_SWITCH.update(path=None, checked=float('-inf'), on=True, said=None)


if __name__ == '__main__':
    unittest.main()
