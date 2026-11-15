"""11-15 checks of the box smoothing of the drawn / gimbal rows (box_refine.py), NPU-free.
Run from tracking_server/:  python -m unittest src.track.test_box_refine"""
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import box_refine as BR
from ..common.state import APP
from ..detect import model_scan


def rows_at(boxes):
    """[(id, cx, cy, w, h)] -> tracker rows (x1, y1, x2, y2, id, conf, class, det index)"""
    return [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, tid, 0.9, 0, i] for i, (tid, cx, cy, w, h) in enumerate(boxes)]


def gru_weights(rng, n=8, scale=0.3):
    return {'cell.weight_ih': rng.normal(0, scale, (3 * n, BR.NF)), 'cell.weight_hh': rng.normal(0, scale, (3 * n, n)), 'cell.bias_ih': rng.normal(0, scale, 3 * n),
            'cell.bias_hh': rng.normal(0, scale, 3 * n), 'head.weight': rng.normal(0, scale, (4, n)), 'head.bias': rng.normal(0, scale, 4)}


SIZE_T = dict(mc_s=1.0, beta_s=0.0, dcut=1.0, anchor='top', src='det')


class RefinerTests(unittest.TestCase):
    def test_euro_static_and_camera_pan(self):
        z = np.array([300., 250., 60., 120.])
        r = BR.Refiner('euro')
        for k in range(10): b = r.step([(1, z, None)], 0.04 * k, None)[1]
        np.testing.assert_allclose(b, z, atol=1e-9)   # a person who does not move stays where the detector puts it
        # camera pans 8 px per frame to the right (picture content moves +8 px); the person stands still in the world
        comp, raw = BR.Refiner('euro'), BR.Refiner('euro'); H = np.array([[1., 0, 8.], [0, 1., 0]])
        for k in range(15):
            zk = z + np.array([8. * k, 0, 0, 0])
            bc = comp.step([(1, zk, None)], 0.04 * k, H)[1]; bn = raw.step([(1, zk, None)], 0.04 * k, None)[1]
        np.testing.assert_allclose(bc, zk, atol=1e-6)     # with the camera warp: no lag
        self.assertGreater(zk[0] - bn[0], 1.0)              # without it the filter trails behind the pan

    def test_gru_zero_head_is_raw_and_features(self):
        rng = np.random.default_rng(0); w = gru_weights(rng); w['head.weight'][:] = 0; w['head.bias'][:] = 0
        m = BR.GRUNp(w); r = BR.Refiner('gru', m)
        for k in range(6):
            z = np.array([200. + 3 * k, 260., 50., 110.]); kp = np.c_[np.full(17, z[0]), np.full(17, z[1] - 20), np.linspace(0, 1, 17)]
            b = r.step([(4, z, kp)], 0.05 * k, np.eye(2, 3))[4]
            np.testing.assert_allclose(b, z, atol=1e-9)
        st = r.tr[4]; self.assertEqual(st['lost'], 0); self.assertEqual(st['h'].shape, (8,))
        # a missing frame: the id keeps its state, the Kalman filter predicts once, the next feature carries the gap
        r.step([], 0.30, np.eye(2, 3)); self.assertEqual(r.tr[4]['lost'], 1)
        st = dict(r.tr[4]); x = BR.features(st, np.array([221., 260., 50., 110.]), None, np.eye(2, 3))
        self.assertAlmostEqual(x[67], 1 / 15.); self.assertTrue(np.isfinite(x).all()); self.assertEqual(x.shape, (BR.NF,))

    def test_reset_on_jump_and_end(self):
        r = BR.Refiner('euro'); z = np.array([100., 200., 40., 100.])
        r.step([(2, z, None)], 0.0, None); r.step([(2, z + [1, 0, 0, 0], None)], 0.04, None)
        far = z + np.array([400., 0, 0, 0]); b = r.step([(2, far, None)], 0.08, None)[2]
        np.testing.assert_allclose(b, far)   # 4 heights away = another person: no smoothing from the old one
        for k in range(BR.LOST_MAX + 2): r.step([], 0.1 + 0.04 * k, None)
        self.assertNotIn(2, r.tr)            # an id gone for more than the tracker's buffer is forgotten

    def test_gap_rule_is_missing_frames_over_lost_max(self):
        """the one reset rule (server and the offline scoring use this class): up to LOST_MAX missing frames keep the state, one more resets it."""
        for miss, kept in ((BR.LOST_MAX, True), (BR.LOST_MAX + 1, False)):
            r = BR.Refiner('size', size=SIZE_T); z = np.array([100., 200., 40., 100.])
            for k in range(3): r.step([(5, z, None, z)], 0.04 * k, None)
            for k in range(miss): r.step([], 0.2 + 0.04 * k, None)
            r.step([(5, z, None, z)], 5.0, None)
            self.assertEqual(r.tr[5]['k'] == 3, kept, miss); self.assertEqual(r.tr[5]['new'], not kept)

    def test_jump_rule_moves_the_previous_box_with_the_camera(self):
        """gimbal slew: a far person 12 px tall while the picture moves 20 px per frame keeps its state (it was reset on 29 of 30 frames)."""
        H = np.array([[1., 0, 20.], [0, 1., 0]])
        for mode in ('euro', 'size', 'off'):
            r = BR.Refiner(mode, size=SIZE_T)
            for k in range(30):
                z = np.array([100. + 20 * k, 200., 5., 12.]); r.step([(1, z, None, z)], k / 25., H)
            self.assertEqual(r.tr[1]['k'], 29, mode)
        # 15 missed frames during a 5 px/frame pan, a 40 px person: the camera motion of the missing frames counts too
        H5 = np.array([[1., 0, 5.], [0, 1., 0]]); r = BR.Refiner('size', size=SIZE_T); z = np.array([100., 200., 20., 40.])
        r.step([(1, z, None, z)], 0.0, H5)
        for k in range(15): r.step([], 0.04 * (k + 1), H5)
        z2 = z + [5. * 16, 0, 0, 0]; r.step([(1, z2, None, z2)], 0.04 * 16, H5); self.assertEqual(r.tr[1]['k'], 1)
        r2 = BR.Refiner('gru', BR.GRUNp(gru_weights(np.random.default_rng(1))))
        for k in range(30):
            z = np.array([100. + 20 * k, 200., 5., 12.]); r2.step([(1, z, None)], k / 25., H)
        self.assertEqual(r2.tr[1]['k'], 29)

    def test_state_handover_when_the_id_changes(self):
        """KeepID relabels a visible person: the warm state moves to the new id (IoU >= 0.5 with the vanished id's last detection)."""
        rng = np.random.default_rng(3); m = BR.GRUNp(gru_weights(rng))
        for mode, mm in (('gru', m), ('size', None), ('euro', None)):
            r = BR.Refiner(mode, mm, size=SIZE_T); z = np.array([300., 250., 50., 120.])
            for k in range(8): r.step([(1, z + [k, 0, 0, 0], None, z + [k, 0, 0, 0])], 0.04 * k, None)
            st1 = r.tr[1]; z9 = z + [8, 0, 0, 0]
            r.step([(9, z9, None, z9)], 0.32, None)           # the same person, now id 9 (id 1 is gone this frame)
            self.assertIs(r.tr[9], st1, mode); self.assertNotIn(1, r.tr); self.assertEqual(r.tr[9]['k'], 8); self.assertEqual(r.handovers, 1)
            r.step([(1, z9 + [1, 0, 0, 0], None, z9 + [1, 0, 0, 0])], 0.36, None)   # and back to id 1 a frame later (KeepID gives the old id back)
            self.assertIs(r.tr[1], st1); self.assertEqual(r.tr[1]['k'], 9); self.assertEqual(r.handovers, 2)
        # two people: id 2 becomes id 3 at the same place -> its state; a new person elsewhere -> a fresh state, id 2's state waits
        a = np.array([100., 250., 50., 120.]); b = np.array([400., 250., 50., 120.])
        r = BR.Refiner('size', size=SIZE_T)
        for k in range(3): r.step([(1, a, None, a), (2, b, None, b)], 0.04 * k, None)
        st2 = r.tr[2]; r.step([(1, a, None, a), (3, b, None, b)], 0.12, None)
        self.assertIs(r.tr[3], st2); self.assertEqual(r.handovers, 1)
        r = BR.Refiner('size', size=SIZE_T)
        for k in range(3): r.step([(1, a, None, a), (2, b, None, b)], 0.04 * k, None)
        c = np.array([250., 250., 50., 120.]); r.step([(1, a, None, a), (3, c, None, c)], 0.12, None)   # a new person between them (no overlap with id 2)
        self.assertTrue(r.tr[3]['new']); self.assertEqual(r.handovers, 0); self.assertIn(2, r.tr)

    def test_blend_on_detector_change(self):
        r = BR.Refiner('off', blend_s=0.3); z1 = np.array([300., 200., 60., 120.]); z2 = np.array([300., 230., 60., 180.])   # the new detector's box is bigger
        r.step([(1, z1, None)], 0.0, None, 'near', True); r.step([(1, z1, None)], 0.1, None, 'near', True)
        b = r.step([(1, z2, None)], 0.2, None, 'far', False)[1]   # the switch is asked, but this frame still comes from the old model (resident switch)
        np.testing.assert_allclose(b, z2)                            # nothing to blend from the old frame
        r2 = BR.Refiner('off', blend_s=0.3)
        r2.step([(1, z1, None)], 0.0, None, 'near', True); r2.step([(1, z1, None)], 0.1, None, 'near', True)
        b0 = r2.step([(1, z2, None)], 0.2, None, 'far', True)[1]    # first frame of the new model: still the old box
        np.testing.assert_allclose(b0, z1, atol=1e-9)
        b1 = r2.step([(1, z2, None)], 0.35, None, 'far', True)[1]   # half way: half of the offset
        self.assertAlmostEqual(b1[3], math.sqrt(120. * 180.), places=6); self.assertAlmostEqual(b1[1], 215., places=6)
        b2 = r2.step([(1, z2, None)], 0.6, None, 'far', True)[1]    # after the window: the new box
        np.testing.assert_allclose(b2, z2)
        self.assertEqual(r2.blends, 1)

    def test_blend_is_per_person(self):
        """the person is missing on the first frame of the new model: their blend starts on their own next frame (was: 100 -> 180 in one frame)."""
        r = BR.Refiner('off', blend_s=0.3); near = np.array([300., 200., 50., 100.]); far = np.array([300., 240., 60., 180.]); other = np.array([100., 200., 40., 90.])
        for k in range(5): r.step([(1, near, None), (2, other, None)], 0.04 * k, None, 'near', True)
        r.step([(2, other, None)], 0.20, None, 'far', True)          # first frame of the far model: person 1 missing
        b = r.step([(1, far, None), (2, other, None)], 0.24, None, 'far', True)[1]
        np.testing.assert_allclose(b, near, atol=1e-9)               # still the old box, then the offset decays
        b2 = r.step([(1, far, None), (2, other, None)], 0.39, None, 'far', True)[1]
        self.assertAlmostEqual(b2[3], math.sqrt(100. * 180.), places=6)
        self.assertEqual(r.blends, 2)                                 # person 2 (seen on the switch frame) and person 1
        # an id gone longer than BLEND_RECENT frames at the switch starts from the new box
        r = BR.Refiner('off', blend_s=0.3)
        for k in range(5): r.step([(1, near, None)], 0.04 * k, None, 'near', True)
        for k in range(BR.BLEND_RECENT): r.step([], 0.2 + 0.04 * k, None, 'far', True)
        np.testing.assert_allclose(r.step([(1, far, None)], 0.4, None, 'far', True)[1], far); self.assertEqual(r.blends, 0)

    def test_blend_anchor_moves_with_the_box(self):
        """both detectors give the same box, moving 10 px per frame in the picture: no stop at the switch (was 10 px behind, sliding back over 0.3 s)."""
        r = BR.Refiner('off', blend_s=0.3)
        for k in range(10):
            z = np.array([100. + 10 * k, 200., 50., 120.]); b = r.step([(1, z, None)], 0.04 * k, None, 'near' if k < 5 else 'far', True)[1]
            np.testing.assert_allclose(b, z, atol=1e-9)
        self.assertEqual(r.blends, 1)

    def test_size_mode_position_is_the_tracker_row(self):
        rng = np.random.default_rng(5)
        for anchor in ('top', 'centre'):
            r = BR.Refiner('size', size=dict(SIZE_T, anchor=anchor))
            for k in range(40):
                zr = np.array([200. + 3 * k, 250. + rng.normal(0, 2), 50., 120. + rng.normal(0, 6)]); z = zr + np.r_[rng.normal(0, 2, 2), rng.normal(0, 3, 2)]
                b = r.step([(1, z, None, zr)], k / 25., None)[1]
                self.assertAlmostEqual(b[0], zr[0], places=9)
                if anchor == 'top': self.assertAlmostEqual(b[1] - b[3] / 2, zr[1] - zr[3] / 2, places=9)
                else: self.assertAlmostEqual(b[1], zr[1], places=9)
        # src 'trk': the size of the tracker row is filtered (a still person: exactly the tracker row)
        r = BR.Refiner('size', size=dict(SIZE_T, src='trk')); zr = np.array([200., 250., 50., 120.])
        for k in range(5): b = r.step([(1, zr + [0, 0, 9, 9], None, zr)], k / 25., None)[1]
        np.testing.assert_allclose(b, zr, atol=1e-9)

    def test_size_mode_spreads_a_detector_switch(self):
        """near -> far: the far detector's box is twice as tall (whole person vs head-to-hip). The state is kept across the switch, so the
        height moves smoothly from 100 to 200 px, the top edge stays, and the step per frame stays small; no blend needed."""
        r = BR.Refiner('size', size=SIZE_T); hs = []; tops = []
        for k in range(75):   # 3 s at 25 fps, switch at 1 s
            h = 100. if k < 25 else 200.; z = np.array([300., 100. + h / 2, 40., h])
            b = r.step([(1, z, None, z)], k / 25., None, 'near' if k < 25 else 'far', True)[1]; hs.append(b[3]); tops.append(b[1] - b[3] / 2)
        hs = np.array(hs); self.assertEqual(r.tr[1]['k'], 74); self.assertEqual(r.blends, 0)
        np.testing.assert_allclose(tops, 100., atol=1e-9)
        self.assertTrue((np.diff(hs[24:]) >= -1e-9).all())          # no overshoot back
        self.assertLess(np.diff(hs).max(), 25.)                       # largest step a quarter of the 100 px change (one frame of the old jump)
        self.assertGreater(hs[-1], 195.)                              # reached the new height within 2 s

    def test_size_mode_camera_zoom_is_not_filtered(self):
        """the camera zooms in 2 % per frame (warp scale), the person stands still in the world: the box grows with the picture at once."""
        s = 1.02; H = np.array([[s, 0, 320 * (1 - s)], [0, s, 256 * (1 - s)]]); r = BR.Refiner('size', size=SIZE_T); z = np.array([320., 256., 40., 100.])
        for k in range(20):
            b = r.step([(1, z, None, z)], k / 25., H if k else None)[1]
            np.testing.assert_allclose(b[2:], z[2:], rtol=1e-9); z = BR.warp_box(z, H)

    def test_size_mode_real_seconds(self):
        """dt is real time: the same 1 s of a growing box gives about the same output at 10 and 30 fps."""
        out = []
        for fps in (10, 30):
            r = BR.Refiner('size', size=SIZE_T)
            for k in range(fps + 1):
                t = k / fps; z = np.array([300., 250., 40., 100. if t < 0.5 else 150.]); b = r.step([(1, z, None, z)], t, None)[1]
            out.append(b[3])
        self.assertLess(abs(out[0] - out[1]), 0.06 * 50)


    # ---- 11-15 review round 2 (each test reproduced the problem before the fix)
    def test_no_blend_after_a_stopped_swap(self):
        """the old swap path stops frames for > 0.5 s: the previous output is where the person was before the stop, so no blend from it
        (was: the box -0.28 h behind on the first new frame after a 0.6 s stop, -0.58 h after 1.2 s)."""
        for mode in ('off', 'size'):
            for stop in (0.6, 1.2):
                r = BR.Refiner(mode, blend_s=0.3, size=SIZE_T); t = 0.0; k = 0
                while t < 1.0:
                    z = np.array([100. + 50. * t, 200., 40., 100.]); r.step([(1, z, None, z)], t, None, 'near', True); t += 0.04; k += 1
                t += stop; z = np.array([100. + 50. * t, 200., 40., 100.])
                b = r.step([(1, z, None, z)], t, None, 'far', True)[1]
                self.assertAlmostEqual(b[0], z[0], places=6, msg=(mode, stop)); self.assertEqual(r.blends, 0)
        # the instant (resident) switch, one frame interval: the blend still starts
        r = BR.Refiner('off', blend_s=0.3); z1 = np.array([300., 200., 60., 120.]); z2 = np.array([300., 230., 60., 180.])
        for k in range(3): r.step([(1, z1, None)], 0.04 * k, None, 'near', True)
        np.testing.assert_allclose(r.step([(1, z2, None)], 0.12, None, 'far', True)[1], z1, atol=1e-9); self.assertEqual(r.blends, 1)
        # a low frame rate (5 fps, 0.2 s per frame): one frame interval still blends
        r = BR.Refiner('off', blend_s=1.0)
        for k in range(3): r.step([(1, z1, None)], 0.2 * k, None, 'near', True)
        r.step([(1, z2, None)], 0.6, None, 'far', True); self.assertEqual(r.blends, 1)

    def test_blend_anchor_follows_a_camera_pan_that_starts_at_the_switch(self):
        """the gimbal pan starts on the switch frame (picture moves 20 px per frame), the person stands still in the world and both detectors
        give the same box: the output stays on the tracker row (was -20, -17.3, ... px behind)."""
        for mode in ('off', 'size'):
            r = BR.Refiner(mode, blend_s=0.3, size=SIZE_T); H = np.array([[1., 0, 20.], [0, 1., 0]]); z = np.array([300., 200., 40., 100.])
            for k in range(5): r.step([(1, z, None, z)], 0.04 * k, None, 'near', True)
            for k in range(5, 12):
                z = z + [20., 0, 0, 0]; b = r.step([(1, z, None, z)], 0.04 * k, H, 'far' if k >= 5 else 'near', True)[1]
                self.assertAlmostEqual(b[0], z[0], places=6, msg=(mode, k))
            self.assertEqual(r.blends, 1)
        # a steady pan plus the person walking 5 px per frame in the world: the carried output keeps both (warp + own motion)
        r = BR.Refiner('off', blend_s=0.3); z = np.array([300., 200., 40., 100.])
        for k in range(12):
            b = r.step([(1, z, None)], 0.04 * k, H if k else None, 'near' if k < 6 else 'far', True)[1]
            self.assertAlmostEqual(b[0], z[0], places=6, msg=k); z = z + [25., 0, 0, 0]

    def test_size_blend_keeps_the_tracker_position(self):
        """size mode + box_blend: the blend spreads only the size; the centre (or the aim point) stays the tracker row's
        (was up to 8 px off the tracker centre with blend 0.3)."""
        rng = np.random.default_rng(7)
        for anchor in ('centre', 'aim'):
            r = BR.Refiner('size', blend_s=0.3, size=dict(SIZE_T, anchor=anchor, src='trk'), aim_y=0.08); hs = []
            for k in range(40):
                h = (100. if k < 20 else 140.) + rng.normal(0, 3); zr = np.array([200. + 2 * k + rng.normal(0, 2), 250. + rng.normal(0, 2), 50., h])
                b = r.step([(5, zr, None, zr)], k / 25., None, 'near' if k < 20 else 'far', True)[5]; hs.append(b[3])
                if anchor == 'centre': self.assertAlmostEqual(b[1], zr[1], places=9)
                else: self.assertAlmostEqual(b[1] - b[3] / 2 + 0.08 * b[3], zr[1] - zr[3] / 2 + 0.08 * zr[3], places=9)
                self.assertAlmostEqual(b[0], zr[0], places=9)
            self.assertEqual(r.blends, 1)
            self.assertLess(np.diff(hs).max(), 15.)

    def test_size_aim_anchor_keeps_the_gimbal_aim_point(self):
        """anchor 'aim': top + aim_y x h equals the tracker row's for any aim_y, while the height is smoothed."""
        rng = np.random.default_rng(8)
        for ay in (0.08, 0.37, 0.5):
            r = BR.Refiner('size', size=dict(SIZE_T, anchor='aim', src='trk'), aim_y=ay); hb = []; hz = []
            for k in range(60):
                zr = np.array([200., 250. + rng.normal(0, 3), 50., 120. + rng.normal(0, 8)])
                b = r.step([(1, zr, None, zr)], k / 25., None)[1]; hb.append(b[3]); hz.append(zr[3])
                self.assertAlmostEqual(b[1] - b[3] / 2 + ay * b[3], zr[1] - zr[3] / 2 + ay * zr[3], places=9)
            self.assertLess(np.std(np.diff(hb)), 0.7 * np.std(np.diff(hz)))

    def test_blend_time_change_does_not_move_the_box_back(self):
        """the blend keeps the duration it started with (was: raising 0.3 -> 2.0 s at frame 3 moved the height 109.4 -> 102.0)."""
        r = BR.Refiner('off', blend_s=0.3); z1 = np.array([300., 200., 60., 100.]); z2 = np.array([300., 220., 60., 140.]); hs = []
        for k in range(3): r.step([(1, z1, None)], 0.04 * k, None, 'near', True)
        for k in range(3, 15):
            if k == 6: r.blend_s = 2.0
            hs.append(r.step([(1, z2, None)], 0.04 * k, None, 'far', True)[1][3])
        self.assertTrue((np.diff(hs) >= -1e-9).all(), hs); self.assertAlmostEqual(hs[-1], 140.)
        r.blend_s = 0.0; r.tr[1]['bl'] = (np.zeros(4), 0.5, 0.3)   # blend turned off: a running blend ends at once
        np.testing.assert_allclose(r.step([(1, z2, None)], 0.6, None, 'far', True)[1], z2)


class _Boom(Exception):
    pass


class ServerGlueTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(BR.BR); APP.args = SimpleNamespace(det='near_kpt'); APP.CTL = dict(lead_motion=None); APP.tracker = object()
        self.kpt = patch.object(model_scan, 'KPT_DETS', {'near_kpt'}); self.kpt.start()

    def tearDown(self):
        self.kpt.stop()
        BR.BR.clear(); BR.BR.update(self.saved)
        for k in ('args', 'CTL', 'tracker'): vars(APP).pop(k, None)

    def test_off_is_the_same_list(self):
        rows = rows_at([(1, 100., 200., 40., 100.)])
        self.assertIs(BR.apply(rows, None, 1.0), rows)

    def test_euro_rows_and_state(self):
        ok, txt, apply_ = BR.parse_request({'box_refine': ['euro'], 'box_blend': ['0.3']}); self.assertTrue(ok); apply_()
        self.assertEqual((BR.BR['mode'], BR.BR['blend_s']), ('euro', 0.3))
        dets = np.array([[80., 150., 120., 250., 0.9, 0.]]); out = None
        for k in range(5):
            rows = rows_at([(7, 100. + k, 200., 40., 100.)]); out = BR.apply(rows, dets + [k, 0, k, 0, 0, 0], 0.04 * k)
            self.assertIsNot(out, rows); self.assertEqual(len(out), 1); self.assertEqual(out[0][4:], rows[0][4:])
        st = BR.state(); self.assertEqual(st['mode'], 'euro'); self.assertEqual(st['people'], 1); self.assertGreater(st['frames'], 0); self.assertEqual(st['err'], '')
        APP.tracker = object(); BR.apply(rows_at([(7, 300., 200., 40., 100.)]), None, 1.0)   # a new tracker (ids start again): every state is dropped
        self.assertEqual(len(BR.BR['obj'].tr), 1); self.assertEqual(BR.BR['obj'].tr[7]['f'], 1)

    def test_reset_key_holds_the_tracker_object(self):
        """the key keeps the tracker itself (identity), so a replaced tracker cannot be mistaken for the old one by a reused address."""
        BR.parse_request({'box_refine': ['size']})[2]()
        t1 = APP.tracker; BR.apply(rows_at([(1, 100., 200., 40., 100.)]), None, 0.0); BR.apply(rows_at([(1, 100., 200., 40., 100.)]), None, 0.04)
        self.assertIs(BR.BR['reset_key'][0], t1); self.assertEqual(BR.BR['obj'].tr[1]['k'], 1)

        class Same:
            def __eq__(self, other): return True    # equal by value to any tracker: still a different tracker
            __hash__ = object.__hash__
        APP.tracker = Same(); BR.apply(rows_at([(1, 100., 200., 40., 100.)]), None, 0.08)
        self.assertEqual(BR.BR['obj'].tr[1]['k'], 0)

    @patch.dict(BR.SIZE, anchor='top', src='det')
    def test_size_mode_rows(self):
        BR.parse_request({'box_refine': ['size']})[2]()
        rows = rows_at([(3, 100., 200., 40., 100.)]); dets = np.array([[78., 140., 122., 262., 0.9, 0.]])   # the detection is taller than the tracker box
        out = BR.apply(rows, dets, 1.0)
        self.assertAlmostEqual(out[0][1], rows[0][1]); self.assertAlmostEqual((out[0][0] + out[0][2]) / 2, 100.)   # top edge and centre x of the tracker row
        self.assertAlmostEqual(out[0][3] - out[0][1], 122.)                                                         # the detection's height (first frame)
        self.assertEqual(BR.state()['mode_name'], '크기만 부드럽게')
        with patch.dict(BR.SIZE, anchor='centre', src='trk'):   # the val-A choice: the tracker row itself on the first frame
            BR.parse_request({'box_refine': ['euro']})[2](); BR.apply(rows, dets, 1.04); BR.parse_request({'box_refine': ['size']})[2]()
            np.testing.assert_allclose(BR.apply(rows, dets, 1.08)[0][:4], rows[0][:4], atol=1e-9)

    def test_blend_change_keeps_the_state(self):
        BR.parse_request({'box_refine': ['size']})[2]()
        for k in range(3): BR.apply(rows_at([(3, 100., 200., 40., 100.)]), None, 0.04 * k)
        obj = BR.BR['obj']; st = obj.tr[3]
        ok, _, ap = BR.parse_request({'box_blend': ['0.4']}); self.assertTrue(ok); ap()
        BR.apply(rows_at([(3, 100., 200., 40., 100.)]), None, 0.12)
        self.assertIs(BR.BR['obj'], obj); self.assertIs(obj.tr[3], st); self.assertEqual(obj.blend_s, 0.4); self.assertEqual(st['k'], 3)
        BR.parse_request({'box_refine': ['euro']})[2](); BR.apply(rows_at([(3, 100., 200., 40., 100.)]), None, 0.16)   # a new mode: a new state machine
        self.assertIsNot(BR.BR['obj'], obj); self.assertEqual(BR.BR['obj'].mode, 'euro')

    def test_blend_only_keeps_tracker_box(self):
        ok, _, ap = BR.parse_request({'box_refine': ['off'], 'box_blend': ['0.3']}); self.assertTrue(ok); ap()
        rows = rows_at([(3, 100., 200., 40., 100.)]); dets = np.array([[70., 140., 130., 260., 0.9, 0.]])   # the matched detection differs from the tracker's box
        for k in range(3): out = BR.apply(rows, dets, 1.0 + 0.04 * k)
        np.testing.assert_allclose(out[0][:4], rows[0][:4])   # no detector change: the tracker's box, untouched

    def test_parse_request_checks(self):
        self.assertFalse(BR.parse_request({'box_refine': ['smooth']})[0])
        self.assertFalse(BR.parse_request({'box_blend': ['5']})[0])
        self.assertFalse(BR.parse_request({'box_blend': ['x']})[0])
        BR.BR['dir'] = '/nonexistent_dir_for_test'; BR.BR.update(model=None, file=None, mt=None)
        self.assertFalse(BR.parse_request({'box_refine': ['gru']})[0])   # no weight file -> refused, nothing changes
        self.assertEqual(BR.BR['mode'], self.saved['mode'])
        self.assertTrue(BR.parse_request({'box_refine': ['size']})[0])

    def test_gru_needs_a_keypoint_detector(self):
        """no stale keypoints: a detector without keypoints never hands detect_frame.KP to gru, refuses gru, and pauses a running gru."""
        m = BR.GRUNp(gru_weights(np.random.default_rng(2)))
        with patch.object(BR, 'scan', return_value=m):
            BR.BR['model'] = m
            self.assertTrue(BR.parse_request({'box_refine': ['gru']})[0])
            APP.args.det = 's640'
            ok, txt, _ = BR.parse_request({'box_refine': ['gru']}); self.assertFalse(ok); self.assertIn('관절', txt)
            from ..detect import detect_frame
            with patch.dict(detect_frame.KP, cur=dict(boxes=np.array([[80., 150., 120., 250.]]), kpts=np.zeros((1, 17, 3)))):
                self.assertIsNone(BR._kp_cur())
                APP.args.det = 'near_kpt'; self.assertIsNotNone(BR._kp_cur()); APP.args.det = 's640'
            BR.BR['mode'] = 'gru'; rows = rows_at([(1, 100., 200., 40., 100.)])
            self.assertIs(BR.apply(rows, None, 1.0), rows); self.assertIn('관절', BR.BR['why'])
            self.assertFalse(next(x for x in BR.state()['modes'] if x['key'] == 'gru')['available'])

    def test_errors_in_a_row_only(self):
        BR.parse_request({'box_refine': ['size']})[2](); rows = rows_at([(1, 100., 200., 40., 100.)]); real = BR.Refiner.step
        with patch.object(BR.Refiner, 'step', side_effect=_Boom('x')):
            for k in range(BR.ERR_MAX - 1): self.assertIs(BR.apply(rows, None, 0.04 * k), rows)
        self.assertEqual(BR.BR['errors'], BR.ERR_MAX - 1)
        BR.apply(rows, None, 3.0); self.assertEqual(BR.BR['errors'], 0); self.assertEqual(BR.BR['mode'], 'size')   # a good frame: the count starts again
        with patch.object(BR.Refiner, 'step', side_effect=_Boom('x')):
            for k in range(BR.ERR_MAX - 1): BR.apply(rows, None, 4.0 + 0.04 * k)
        self.assertEqual(BR.BR['mode'], 'size'); self.assertEqual(BR.BR['errors_all'], 2 * (BR.ERR_MAX - 1))
        with patch.object(BR.Refiner, 'step', side_effect=_Boom('x')):
            for k in range(2): BR.apply(rows, None, 9.0 + 0.04 * k)
        self.assertEqual(BR.BR['mode'], 'off')   # more than ERR_MAX errors in a row
        self.assertIs(BR.Refiner.step, real)

    def test_off_drops_the_state(self):
        """size, then off for a long time, then size again: the people start from scratch (was: the 10-minute-old state reused as one frame)."""
        BR.parse_request({'box_refine': ['size']})[2]()
        for k in range(3): BR.apply(rows_at([(1, 100., 200., 40., 100.)]), None, 0.04 * k)
        BR.parse_request({'box_refine': ['off']})[2](); BR.apply(rows_at([(1, 160., 200., 40., 140.)]), None, 600.)
        self.assertNotIn('obj', BR.BR)
        BR.parse_request({'box_refine': ['size'], 'box_blend': ['0.3']})[2]()
        out = BR.apply(rows_at([(1, 160., 200., 40., 140.)]), None, 600.04)
        np.testing.assert_allclose(out[0][:4], rows_at([(1, 160., 200., 40., 140.)])[0][:4], atol=1e-9)
        self.assertEqual(BR.BR['obj'].tr[1]['k'], 0)

    def test_saved_gru_survives_startup_on_a_plain_detector(self):
        """startup: the command-line detector has no keypoints, the saved detector (asked for after the saved rules) has them: the saved gru
        is taken and waits (was refused, then erased from last_selection.json)."""
        m = BR.GRUNp(gru_weights(np.random.default_rng(2)))
        with patch.object(BR, 'scan', return_value=m):
            BR.BR['model'] = m; APP.args.det = 'w480'; APP.SEL = {'det': 'near_kpt', 'box_refine': 'gru'}
            try:
                ok = []
                for q in BR.restore_queries(APP.SEL):
                    o, _, ap = BR.parse_request(q); ok.append(o)
                    if o: ap()
                self.assertEqual(ok, [True]); self.assertEqual(BR.BR['mode'], 'gru')
                rows = rows_at([(1, 100., 200., 40., 100.)]); self.assertIs(BR.apply(rows, None, 1.0), rows)   # paused until the keypoint detector runs
                APP.args.det = 'near_kpt'; self.assertIsNot(BR.apply(rows, None, 1.04), rows)
                APP.SEL = {}; APP.args.det = 'w480'
                self.assertFalse(BR.parse_request({'box_refine': ['gru']})[0])   # no keypoint detector now or asked for: refused as before
            finally:
                vars(APP).pop('SEL', None)

    def test_status_text_is_cleared_by_a_new_choice(self):
        """the page showed '관절 없는 검출 모델: 학습 GRU 쉼' after the user picked 끔."""
        BR.BR.update(why='관절 없는 검출 모델: 학습 GRU 쉼 (그대로 둠)', ms=1.5)
        BR.parse_request({'box_refine': ['off']})[2](); self.assertEqual(BR.BR['why'], ''); self.assertEqual(BR.state()['ms'], 0.0)

    def test_auto_off_after_err_max_is_saved(self):
        saved = []; APP.sel_save = lambda **kw: saved.append(kw)
        try:
            BR.parse_request({'box_refine': ['size']})[2](); saved.clear(); rows = rows_at([(1, 100., 200., 40., 100.)])
            with patch.object(BR.Refiner, 'step', side_effect=_Boom('x')):
                for k in range(BR.ERR_MAX): BR.apply(rows, None, 0.04 * k)
            self.assertEqual(BR.BR['mode'], 'off'); self.assertEqual(saved, [dict(box_refine='off', box_blend=0.0)]); self.assertIn('오류', BR.BR['why'])
        finally:
            vars(APP).pop('sel_save', None)

    def test_aim_y_from_the_gimbal_options(self):
        self.assertEqual(BR._aim_y(), BR.AIM_Y)
        APP.args.gimbal_aim_y = 0.08; self.assertAlmostEqual(BR._aim_y(), 0.08)
        APP.GB = SimpleNamespace(proc=SimpleNamespace(args=['bridge', '--aim-y', '0.3']))
        try: self.assertAlmostEqual(BR._aim_y(), 0.3)   # the running bridge's own option wins
        finally: vars(APP).pop('GB', None)

    def test_startup_restores_the_blend_separately(self):
        """a saved gru choice that is refused at startup (no weight file / no keypoint detector) must not drop the saved blend time."""
        qs = BR.restore_queries({'box_refine': 'gru', 'box_blend': 0.3, 'det': 'x'})
        self.assertEqual(qs, [{'box_refine': ['gru']}, {'box_blend': ['0.3']}])
        BR.BR['dir'] = '/nonexistent_dir_for_test'; BR.BR.update(model=None, file=None, mt=None)
        done = []
        for q in qs:
            ok, _, ap = BR.parse_request(q); done.append(ok)
            if ok: ap()
        self.assertEqual(done, [False, True]); self.assertEqual(BR.BR['blend_s'], 0.3)


# ------------------------------------------------------------------------------------------------ gru2 (11-15): GRU box refiner version 2
TRACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'track')
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'box_refine_gru2_fixture.npz')
_G2 = {}


def g2_model():
    """the committed version-2 weights (models/track/box_refine_gru_v2_*.npz, name order last)."""
    if 'm' not in _G2:
        f = sorted(x for x in os.listdir(TRACK_DIR) if x.startswith(BR.V2_PREFIX) and x.endswith('.npz'))[-1]; _G2['path'] = os.path.join(TRACK_DIR, f)
        z = np.load(_G2['path'], allow_pickle=False); _G2['m'] = BR.G2Model({k: z[k] for k in z.files})
    return _G2['m']


def kp_of(z, vis=1.0):
    """17 keypoints inside the box (head near the top, hips near the bottom)."""
    y = np.linspace(-0.45, 0.45, 17); return np.c_[np.full(17, z[0]) + 0.2 * z[2] * np.cos(np.arange(17)), z[1] + y * z[3], np.full(17, vis)]


class Gru2Tests(unittest.TestCase):
    def test_parity_with_the_offline_engine_on_real_rows(self):
        """two val-B windows (thermal switch stream: 29 tracks with starts and gaps; dance: 17 people at once): the server mode gives the offline
        g2_lib 'out_f' (g2_server_parity.py wrote the fixture; the whole val-B and the device recordings were compared there: 0 px)."""
        import hashlib
        m = g2_model(); fx = np.load(FIXTURE, allow_pickle=False)
        self.assertEqual(hashlib.sha256(open(_G2['path'], 'rb').read()).hexdigest(), str(fx['weights_sha256']))   # the fixture belongs to these weights
        for c in ('a', 'b'):
            ids, f, z, kp, W, ref = fx[c + '_id'], fx[c + '_f'], fx[c + '_z'].astype(np.float64), fx[c + '_kp'].astype(np.float64), fx[c + '_W'].astype(np.float64), fx[c + '_out']
            fps = float(fx[c + '_fps']); worst = 0.; starts = 0
            with patch.object(BR, 'HANDOVER_IOU', 2.0):   # the offline engine has no hand-over
                r = BR.Refiner('gru2', m)
                for fr in range(len(W)):
                    j = np.flatnonzero(f == fr); out = r.step([(int(ids[q]), z[q], kp[q]) for q in j], fr / fps, W[fr] if fr > 0 else None)
                    for q in j:
                        worst = max(worst, float(np.abs(out[int(ids[q])] - ref[q]).max())); starts += r.tr[int(ids[q])]['new']
                        self.assertEqual(bool(r.tr[int(ids[q])]['new']), bool(fx[c + '_start'][q]))
            self.assertLess(worst, 1e-3, c); self.assertEqual(starts, int(fx[c + '_start'].sum())); self.assertGreater(len(f), 500)

    def test_start_of_track_and_large_boxes(self):
        m = g2_model(); r = BR.Refiner('gru2', m)
        z = np.array([300., 250., 16., 40.]); C = z + np.r_[m.conv_delta(z[None])[0, :2] * z[3], 0, 0]; C[2:] = z[2:] * np.exp(m.conv_delta(z[None])[0, 2:])
        b0 = r.step([(1, z, kp_of(z))], 0.0, None)[1]
        self.assertTrue(r.tr[1]['new']); self.assertEqual(r.tr[1]['k'], 0)
        self.assertLess(abs(b0[0] - C[0]), 0.4 * C[3] + 1e-9); self.assertLess(abs(math.log(b0[3] / C[3])), math.log(2.2) + 1e-9)   # the converted box + a bounded correction
        self.assertLess(b0[3], 0.8 * z[3])            # a small whole-person box (40 x 16) becomes a body box at once: no first-frame jump later
        hs = [b0[3]]
        for k in range(1, 8): hs.append(r.step([(1, z, kp_of(z))], 0.04 * k, None)[1][3])
        self.assertLess(np.abs(np.diff(hs)).max() / z[3], 0.05); self.assertEqual(r.tr[1]['k'], 7)   # version 1 moved 0.15+ heights on row 1 for 79 % of new tracks
        # a close person (the detector box already is the body box): the lookup must not shrink it (was x 0.64 before the 170 px rows were fixed)
        d = m.conv_delta(np.array([[320., 256., 190., 300.], [320., 256., 130., 200.], [320., 256., 250., 420.]]))
        self.assertLess(np.abs(d).max(), 0.01)
        r = BR.Refiner('gru2', m); zb = np.array([320., 260., 190., 300.])
        for k in range(10): b = r.step([(1, zb, kp_of(zb))], 0.04 * k, None)[1]
        self.assertGreater(b[3] / zb[3], 0.8); self.assertLess(b[3] / zb[3], 1.1)

    def test_handover_keeps_the_state(self):
        m = g2_model(); z = np.array([300., 250., 50., 120.]); ra = BR.Refiner('gru2', m); rb = BR.Refiner('gru2', m)
        for k in range(8):
            zk = z + [2 * k, 0, 0, 0]; ra.step([(1, zk, kp_of(zk))], 0.04 * k, None); rb.step([(1, zk, kp_of(zk))], 0.04 * k, None)
        st1 = ra.tr[1]; z9 = z + [16, 0, 0, 0]
        a = ra.step([(9, z9, kp_of(z9))], 0.32, None)[9]; b = rb.step([(1, z9, kp_of(z9))], 0.32, None)[1]   # the same person, now id 9
        self.assertIs(ra.tr[9], st1); self.assertNotIn(1, ra.tr); self.assertEqual(ra.tr[9]['k'], 8); self.assertEqual(ra.handovers, 1)
        self.assertIn(9, ra.eng.slot); self.assertNotIn(1, ra.eng.slot); np.testing.assert_allclose(a, b, atol=1e-12)   # exactly as if the id had not changed
        z10 = z9 + [2, 0, 0, 0]; a = ra.step([(1, z10, kp_of(z10))], 0.36, None)[1]; b = rb.step([(1, z10, kp_of(z10))], 0.36, None)[1]   # and back
        self.assertEqual(ra.tr[1]['k'], 9); self.assertEqual(ra.handovers, 2); np.testing.assert_allclose(a, b, atol=1e-12)
        # a new person elsewhere gets a fresh state; the vanished id's state waits
        c = np.array([100., 250., 50., 120.]); ra.step([(5, c, kp_of(c))], 0.40, None)
        self.assertTrue(ra.tr[5]['new']); self.assertEqual(ra.handovers, 2); self.assertIn(1, ra.tr); self.assertIn(1, ra.eng.slot)

    def test_detector_switch_keeps_the_state(self):
        """near <-> far: the state is kept (the training data had such switches), nothing blends with box_blend 0, and the box definition change of
        a small person (near: body box 30 px, far: whole person 60 px) arrives converted and spread over time, not as a doubled height in one frame."""
        m = g2_model(); ra = BR.Refiner('gru2', m); rb = BR.Refiner('gru2', m); z = np.array([300., 250., 60., 130.])
        for k in range(12):
            a = ra.step([(1, z, kp_of(z))], 0.04 * k, None, 'near' if k < 6 else 'far', True)[1]; b = rb.step([(1, z, kp_of(z))], 0.04 * k, None, 'near', True)[1]
            np.testing.assert_allclose(a, b, atol=1e-12)
        self.assertEqual(ra.tr[1]['k'], 11); self.assertEqual(ra.blends, 0)
        r = BR.Refiner('gru2', m); hs = []
        for k in range(40):   # body box 30 px (near), then the far detector's whole-person box 60 px tall and narrow
            zk = np.array([300., 250., 18., 30.]) if k < 20 else np.array([300., 265., 20., 60.]); hs.append(r.step([(1, zk, kp_of(zk))], 0.04 * k, None, 'near' if k < 20 else 'far', True)[1][3])
        self.assertEqual(r.tr[1]['k'], 39); self.assertLess(np.abs(np.diff(hs)).max(), 0.05 * 60.)   # the raw step is 30 px in one frame
        self.assertLess(max(hs), 0.75 * 60.)                                                           # and it stays a body box, not the whole-person box

    def test_camera_warp_gap_and_reset(self):
        m = g2_model(); H = np.array([[1., 0, 20.], [0, 1., 0]])
        r = BR.Refiner('gru2', m); r0 = BR.Refiner('gru2', m)
        for k in range(30):   # gimbal slew 20 px per frame, a far person 12 px tall: with the warp the state is kept, without it every frame is 'another person'
            z = np.array([20. + 20 * k, 200., 5., 12.]); r.step([(1, z, None)], k / 25., H); r0.step([(1, z, None)], k / 25., None)
        self.assertEqual(r.tr[1]['k'], 29); self.assertEqual(r0.tr[1]['k'], 0)
        H5 = np.array([[1., 0, 5.], [0, 1., 0]]); r = BR.Refiner('gru2', m); z = np.array([100., 200., 20., 40.])   # 15 missed frames in a 5 px/frame pan
        r.step([(1, z, None)], 0.0, H5)
        for k in range(15): r.step([], 0.04 * (k + 1), H5)
        z2 = z + [5. * 16, 0, 0, 0]; r.step([(1, z2, None)], 0.04 * 16, H5); self.assertEqual(r.tr[1]['k'], 1); self.assertFalse(r.tr[1]['new'])
        for miss, kept in ((BR.LOST_MAX, True), (BR.LOST_MAX + 1, False)):   # the one gap rule
            r = BR.Refiner('gru2', m)
            for k in range(3): r.step([(5, z, None)], 0.04 * k, None)
            for k in range(miss): r.step([], 0.2 + 0.04 * k, None)
            r.step([(5, z, None)], 5.0, None); self.assertEqual(r.tr[5]['k'] == 3, kept, miss); self.assertEqual(r.tr[5]['new'], not kept)
        r = BR.Refiner('gru2', m); r.step([(2, z, None)], 0.0, None); r.step([(2, z, None)], 0.04, None)   # 10 heights away = another person: from the converted box again
        far = z + [400., 0, 0, 0]; b = r.step([(2, far, None)], 0.08, None)[2]; f = BR.Refiner('gru2', m).step([(2, far, None)], 0.08, None)[2]
        np.testing.assert_allclose(b, f, atol=1e-12); self.assertTrue(r.tr[2]['new']); self.assertEqual(r.tr[2]['k'], 0)
        # a camera zoom (warp scale 1.02 per frame) is followed at once by the size filter: a person still in the world keeps a steady box / detection ratio
        s = 1.02; Hz = np.array([[s, 0, 320 * (1 - s)], [0, s, 256 * (1 - s)]]); r = BR.Refiner('gru2', m); z = np.array([320., 256., 60., 120.]); q = []
        for k in range(25):
            q.append(r.step([(1, z, kp_of(z))], k / 25., Hz if k else None)[1][3] / z[3]); z = BR.warp_box(z, Hz)
        self.assertLess(max(q[10:]) - min(q[10:]), 0.03)

    def test_no_keypoints_and_many_people(self):
        m = g2_model(); r = BR.Refiner('gru2', m); z = np.array([200., 250., 50., 110.])
        b = r.step([(1, z, None), (2, z + [200, 0, 0, 0], kp_of(z + [200, 0, 0, 0]))], 0.0, None)   # one person without keypoints next to one with
        self.assertTrue(np.isfinite(b[1]).all()); self.assertTrue(np.isfinite(b[2]).all())
        e = BR.G2Engine(m); o = e.step([1], z[None], np.zeros((1, 17, 3)), None, 0.0); np.testing.assert_allclose(b[1], o['out_f'][0], atol=1e-12)   # = zeros
        lo = BR.Refiner('gru2', m).step([(1, z, kp_of(z, 0.2))], 0.0, None)[1]; np.testing.assert_allclose(lo, b[1], atol=1e-12)   # all under the 0.5 visibility line = none
        # 30 people in one batch: every frame one matrix product, a few ms at most on any PC (measured: models/README)
        import time
        rng = np.random.default_rng(0); r = BR.Refiner('gru2', m); zs = np.c_[rng.uniform(40, 600, 30), rng.uniform(100, 400, 30), rng.uniform(15, 60, 30), rng.uniform(40, 150, 30)]; ts = []
        for k in range(120):
            zs[:, :2] += rng.normal(0, 1, (30, 2)); items = [(i, zs[i].copy(), kp_of(zs[i])) for i in range(30)]; t0 = time.perf_counter(); out = r.step(items, k / 25., np.eye(2, 3)); ts.append(time.perf_counter() - t0)
        self.assertEqual(len(out), 30); self.assertTrue(all(np.isfinite(v).all() for v in out.values())); self.assertLess(np.median(ts[20:]) * 1000, 25.0)
        self.assertEqual(r.eng.cap, 64); self.assertEqual(len(r.eng.slot), 30)
        for k in range(BR.LOST_MAX + 2): r.step([], 5.0 + 0.04 * k, None)   # everyone gone: every slot is freed
        self.assertEqual(len(r.eng.slot), 0); self.assertEqual(len(r.tr), 0); self.assertEqual(len(r.eng.free), 64)

    def test_no_model_is_the_tracker_row(self):
        r = BR.Refiner('gru2', None); z = np.array([200., 250., 50., 110.]); zr = z + [1, 1, 2, 2]
        np.testing.assert_allclose(r.step([(1, z, None, zr)], 0.0, None)[1], zr)


class Gru2GlueTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(BR.BR); APP.args = SimpleNamespace(det='near_kpt'); APP.CTL = dict(lead_motion=None); APP.tracker = object()
        self.kpt = patch.object(model_scan, 'KPT_DETS', {'near_kpt'}); self.kpt.start(); BR.BR['dir'] = TRACK_DIR
        self.now = patch.object(BR.time, 'time', return_value=4e9); self.now.start()   # a just-written weight file is not skipped as 'still being copied'

    def tearDown(self):
        self.kpt.stop(); self.now.stop()
        BR.BR.clear(); BR.BR.update(self.saved)
        for k in ('args', 'CTL', 'tracker', 'SEL'): vars(APP).pop(k, None)

    def test_both_weight_files_load_side_by_side(self):
        """the version-2 file sorts after the version-1 file: the version-1 loader must not pick it up (it would fail its shape check and switch gru off)."""
        m1 = BR.scan(); m2 = BR.scan2()
        self.assertIsNotNone(m1); self.assertIsNotNone(m2); self.assertFalse(BR.BR['file'].startswith(BR.V2_PREFIX)); self.assertTrue(BR.BR['file2'].startswith(BR.V2_PREFIX))
        self.assertEqual(m1.wih.shape[1], BR.NF); self.assertEqual(m2.wih.shape[1], BR.G2_NF); self.assertEqual(m2.size, (5.0, 0.0, 1.0)); self.assertEqual(m2.lost_max, BR.LOST_MAX); self.assertEqual(m2.jump_h, BR.JUMP_H)
        st = BR.state(); self.assertTrue(all(x['available'] for x in st['modes'])); self.assertEqual([x['key'] for x in st['modes']], list(BR.MODES)); self.assertEqual(st['gru2_file'], BR.BR['file2'])

    def test_default_off_is_the_same_list(self):
        BR.scan2(); self.assertEqual(self.saved['mode'], 'off'); rows = rows_at([(1, 100., 200., 40., 100.)])
        self.assertIs(BR.apply(rows, None, 1.0), rows); self.assertNotIn('obj', BR.BR)

    def test_select_run_and_reset(self):
        ok, txt, ap = BR.parse_request({'box_refine': ['gru2']}); self.assertTrue(ok, txt); ap(); self.assertEqual(BR.BR['mode'], 'gru2')
        from ..detect import detect_frame
        dets = np.array([[80., 150., 120., 250., 0.9, 0.]]); kp = kp_of(np.array([100., 200., 40., 100.]))[None]; out = None
        with patch.dict(detect_frame.KP, cur=dict(boxes=dets[:, :4].copy(), kpts=kp)):
            for k in range(5):
                rows = rows_at([(7, 100., 200., 40., 100.), (8, 300., 200., 40., 100.)]); rows.append([10., 10., 50., 40., 3, 0.9, 2, 5])   # two people and a car
                out = BR.apply(rows, dets, 0.04 * k)
                self.assertIsNot(out, rows); self.assertEqual(len(out), 3); self.assertEqual(out[2], rows[2]); self.assertEqual(out[0][4:], rows[0][4:])
        obj = BR.BR['obj']; self.assertEqual(obj.mode, 'gru2'); self.assertEqual(obj.tr[7]['k'], 4); self.assertEqual(len(obj.eng.slot), 2)
        st = BR.state(); self.assertEqual(st['mode_name'], BR.MODE_KO['gru2']); self.assertEqual(st['people'], 2); self.assertEqual(st['err'], ''); self.assertEqual(st['errors_all'], 0)
        e = BR.G2Engine(BR.BR['model2']); ref = None   # person 7: the matched detection + its keypoints went in (not the tracker row, not zeros)
        for k in range(5): ref = e.step([7], np.array([[100., 200., 40., 100.]]), kp, None, 0.04 * k)['out_f'][0]
        np.testing.assert_allclose([(out[0][0] + out[0][2]) / 2, (out[0][1] + out[0][3]) / 2, out[0][2] - out[0][0], out[0][3] - out[0][1]], ref, atol=1e-9)
        APP.tracker = object(); BR.apply(rows_at([(7, 100., 200., 40., 100.)]), None, 1.0)   # a new tracker: every state is dropped
        self.assertIsNot(BR.BR['obj'], obj); self.assertEqual(BR.BR['obj'].tr[7]['k'], 0); self.assertEqual(len(BR.BR['obj'].eng.slot), 1)
        BR.parse_request({'box_refine': ['off']})[2](); rows = rows_at([(7, 100., 200., 40., 100.)]); self.assertIs(BR.apply(rows, None, 2.0), rows); self.assertNotIn('obj', BR.BR)

    def test_needs_a_keypoint_detector_and_a_file(self):
        BR.scan2(); APP.args.det = 's640'
        ok, txt, _ = BR.parse_request({'box_refine': ['gru2']}); self.assertFalse(ok); self.assertIn('관절', txt)
        self.assertFalse(next(x for x in BR.state()['modes'] if x['key'] == 'gru2')['available'])
        BR.BR['mode'] = 'gru2'; rows = rows_at([(1, 100., 200., 40., 100.)])   # the detector changed to one without keypoints while gru2 runs: it rests
        self.assertIs(BR.apply(rows, None, 1.0), rows); self.assertIn('관절', BR.BR['why']); self.assertNotIn('obj', BR.BR)
        APP.args.det = 'near_kpt'; self.assertIsNot(BR.apply(rows, None, 1.04), rows)
        BR.BR['mode'] = 'off'; BR.BR['dir'] = '/nonexistent_dir_for_test'; BR.BR.update(model2=None, file2=None, mt2=None)
        ok, txt, _ = BR.parse_request({'box_refine': ['gru2']}); self.assertFalse(ok); self.assertIn('2판', txt); self.assertEqual(BR.BR['mode'], 'off')

    def test_saved_choice_comes_back_at_startup(self):
        """the saved gru2 is restored; with a plain command-line detector and a saved keypoint detector it is taken and waits."""
        saved = []; APP.sel_save = lambda **kw: saved.append(kw)
        try:
            BR.parse_request({'box_refine': ['gru2']})[2](); self.assertEqual(saved[-1], dict(box_refine='gru2', box_blend=0.0))
            BR.BR['mode'] = 'off'; APP.args.det = 'w480'; APP.SEL = {'det': 'near_kpt', 'box_refine': 'gru2', 'box_blend': 0.0}; done = []
            for q in BR.restore_queries(APP.SEL):
                o, _, ap = BR.parse_request(q); done.append(o)
                if o: ap()
            self.assertEqual(done, [True, True]); self.assertEqual(BR.BR['mode'], 'gru2')
            rows = rows_at([(1, 100., 200., 40., 100.)]); self.assertIs(BR.apply(rows, None, 1.0), rows)
            APP.args.det = 'near_kpt'; self.assertIsNot(BR.apply(rows, None, 1.04), rows)
        finally:
            vars(APP).pop('sel_save', None)


if __name__ == '__main__':
    unittest.main()
