"""11-15 checks of the box smoothing of the drawn / gimbal rows (box_refine.py), NPU-free. 11-16: only off / size / body are left.
Run from tracking_server/:  python -m unittest src.track.test_box_refine"""
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import box_refine as BR
from ..common.state import APP


def rows_at(boxes):
    """[(id, cx, cy, w, h)] -> tracker rows (x1, y1, x2, y2, id, conf, class, det index)"""
    return [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, tid, 0.9, 0, i] for i, (tid, cx, cy, w, h) in enumerate(boxes)]


SIZE_T = dict(mc_s=1.0, beta_s=0.0, dcut=1.0, anchor='top', src='det')


class RefinerTests(unittest.TestCase):
    def test_reset_on_jump_and_end(self):
        r = BR.Refiner('size', size=SIZE_T); z = np.array([100., 200., 40., 100.])
        r.step([(2, z, z)], 0.0, None); r.step([(2, z + [1, 0, 0, 9], z + [1, 0, 0, 9])], 0.04, None)
        far = z + np.array([400., 0, 0, 30.]); b = r.step([(2, far, far)], 0.08, None)[2]
        np.testing.assert_allclose(b, far)   # 4 heights away = another person: no smoothing from the old one
        self.assertTrue(r.tr[2]['new'])
        for k in range(BR.LOST_MAX + 2): r.step([], 0.1 + 0.04 * k, None)
        self.assertNotIn(2, r.tr)            # an id gone for more than the tracker's buffer is forgotten

    def test_gap_rule_is_missing_frames_over_lost_max(self):
        """the one reset rule (server and the offline scoring use this class): up to LOST_MAX missing frames keep the state, one more resets it."""
        for miss, kept in ((BR.LOST_MAX, True), (BR.LOST_MAX + 1, False)):
            r = BR.Refiner('size', size=SIZE_T); z = np.array([100., 200., 40., 100.])
            for k in range(3): r.step([(5, z, z)], 0.04 * k, None)
            for k in range(miss): r.step([], 0.2 + 0.04 * k, None)
            r.step([(5, z, z)], 5.0, None)
            self.assertEqual(r.tr[5]['k'] == 3, kept, miss); self.assertEqual(r.tr[5]['new'], not kept)

    def test_jump_rule_moves_the_previous_box_with_the_camera(self):
        """gimbal slew: a far person 12 px tall while the picture moves 20 px per frame keeps its state (it was reset on 29 of 30 frames)."""
        H = np.array([[1., 0, 20.], [0, 1., 0]])
        for mode in ('size', 'off'):
            r = BR.Refiner(mode, size=SIZE_T)
            for k in range(30):
                z = np.array([100. + 20 * k, 200., 5., 12.]); r.step([(1, z, z)], k / 25., H)
            self.assertEqual(r.tr[1]['k'], 29, mode)
        # 15 missed frames during a 5 px/frame pan, a 40 px person: the camera motion of the missing frames counts too
        H5 = np.array([[1., 0, 5.], [0, 1., 0]]); r = BR.Refiner('size', size=SIZE_T); z = np.array([100., 200., 20., 40.])
        r.step([(1, z, z)], 0.0, H5)
        for k in range(15): r.step([], 0.04 * (k + 1), H5)
        z2 = z + [5. * 16, 0, 0, 0]; r.step([(1, z2, z2)], 0.04 * 16, H5); self.assertEqual(r.tr[1]['k'], 1)

    def test_state_handover_when_the_id_changes(self):
        """KeepID relabels a visible person: the warm state moves to the new id (IoU >= 0.5 with the vanished id's last detection)."""
        for mode in ('size', 'off'):
            r = BR.Refiner(mode, size=SIZE_T); z = np.array([300., 250., 50., 120.])
            for k in range(8): r.step([(1, z + [k, 0, 0, 0], z + [k, 0, 0, 0])], 0.04 * k, None)
            st1 = r.tr[1]; z9 = z + [8, 0, 0, 0]
            r.step([(9, z9, z9)], 0.32, None)           # the same person, now id 9 (id 1 is gone this frame)
            self.assertIs(r.tr[9], st1, mode); self.assertNotIn(1, r.tr); self.assertEqual(r.tr[9]['k'], 8); self.assertEqual(r.handovers, 1)
            r.step([(1, z9 + [1, 0, 0, 0], z9 + [1, 0, 0, 0])], 0.36, None)   # and back to id 1 a frame later (KeepID gives the old id back)
            self.assertIs(r.tr[1], st1); self.assertEqual(r.tr[1]['k'], 9); self.assertEqual(r.handovers, 2)
        # two people: id 2 becomes id 3 at the same place -> its state; a new person elsewhere -> a fresh state, id 2's state waits
        a = np.array([100., 250., 50., 120.]); b = np.array([400., 250., 50., 120.])
        r = BR.Refiner('size', size=SIZE_T)
        for k in range(3): r.step([(1, a, a), (2, b, b)], 0.04 * k, None)
        st2 = r.tr[2]; r.step([(1, a, a), (3, b, b)], 0.12, None)
        self.assertIs(r.tr[3], st2); self.assertEqual(r.handovers, 1)
        r = BR.Refiner('size', size=SIZE_T)
        for k in range(3): r.step([(1, a, a), (2, b, b)], 0.04 * k, None)
        c = np.array([250., 250., 50., 120.]); r.step([(1, a, a), (3, c, c)], 0.12, None)   # a new person between them (no overlap with id 2)
        self.assertTrue(r.tr[3]['new']); self.assertEqual(r.handovers, 0); self.assertIn(2, r.tr)

    def test_blend_on_detector_change(self):
        r = BR.Refiner('off', blend_s=0.3); z1 = np.array([300., 200., 60., 120.]); z2 = np.array([300., 230., 60., 180.])   # the new detector's box is bigger
        r.step([(1, z1)], 0.0, None, 'near', True); r.step([(1, z1)], 0.1, None, 'near', True)
        b = r.step([(1, z2)], 0.2, None, 'far', False)[1]   # the switch is asked, but this frame still comes from the old model (resident switch)
        np.testing.assert_allclose(b, z2)                            # nothing to blend from the old frame
        r2 = BR.Refiner('off', blend_s=0.3)
        r2.step([(1, z1)], 0.0, None, 'near', True); r2.step([(1, z1)], 0.1, None, 'near', True)
        b0 = r2.step([(1, z2)], 0.2, None, 'far', True)[1]    # first frame of the new model: still the old box
        np.testing.assert_allclose(b0, z1, atol=1e-9)
        b1 = r2.step([(1, z2)], 0.35, None, 'far', True)[1]   # half way: half of the offset
        self.assertAlmostEqual(b1[3], math.sqrt(120. * 180.), places=6); self.assertAlmostEqual(b1[1], 215., places=6)
        b2 = r2.step([(1, z2)], 0.6, None, 'far', True)[1]    # after the window: the new box
        np.testing.assert_allclose(b2, z2)
        self.assertEqual(r2.blends, 1)

    def test_blend_is_per_person(self):
        """the person is missing on the first frame of the new model: their blend starts on their own next frame (was: 100 -> 180 in one frame)."""
        r = BR.Refiner('off', blend_s=0.3); near = np.array([300., 200., 50., 100.]); far = np.array([300., 240., 60., 180.]); other = np.array([100., 200., 40., 90.])
        for k in range(5): r.step([(1, near), (2, other)], 0.04 * k, None, 'near', True)
        r.step([(2, other)], 0.20, None, 'far', True)          # first frame of the far model: person 1 missing
        b = r.step([(1, far), (2, other)], 0.24, None, 'far', True)[1]
        np.testing.assert_allclose(b, near, atol=1e-9)               # still the old box, then the offset decays
        b2 = r.step([(1, far), (2, other)], 0.39, None, 'far', True)[1]
        self.assertAlmostEqual(b2[3], math.sqrt(100. * 180.), places=6)
        self.assertEqual(r.blends, 2)                                 # person 2 (seen on the switch frame) and person 1
        # an id gone longer than BLEND_RECENT frames at the switch starts from the new box
        r = BR.Refiner('off', blend_s=0.3)
        for k in range(5): r.step([(1, near)], 0.04 * k, None, 'near', True)
        for k in range(BR.BLEND_RECENT): r.step([], 0.2 + 0.04 * k, None, 'far', True)
        np.testing.assert_allclose(r.step([(1, far)], 0.4, None, 'far', True)[1], far); self.assertEqual(r.blends, 0)

    def test_blend_anchor_moves_with_the_box(self):
        """both detectors give the same box, moving 10 px per frame in the picture: no stop at the switch (was 10 px behind, sliding back over 0.3 s)."""
        r = BR.Refiner('off', blend_s=0.3)
        for k in range(10):
            z = np.array([100. + 10 * k, 200., 50., 120.]); b = r.step([(1, z)], 0.04 * k, None, 'near' if k < 5 else 'far', True)[1]
            np.testing.assert_allclose(b, z, atol=1e-9)
        self.assertEqual(r.blends, 1)

    def test_size_mode_position_is_the_tracker_row(self):
        rng = np.random.default_rng(5)
        for anchor in ('top', 'centre'):
            r = BR.Refiner('size', size=dict(SIZE_T, anchor=anchor))
            for k in range(40):
                zr = np.array([200. + 3 * k, 250. + rng.normal(0, 2), 50., 120. + rng.normal(0, 6)]); z = zr + np.r_[rng.normal(0, 2, 2), rng.normal(0, 3, 2)]
                b = r.step([(1, z, zr)], k / 25., None)[1]
                self.assertAlmostEqual(b[0], zr[0], places=9)
                if anchor == 'top': self.assertAlmostEqual(b[1] - b[3] / 2, zr[1] - zr[3] / 2, places=9)
                else: self.assertAlmostEqual(b[1], zr[1], places=9)
        # src 'trk': the size of the tracker row is filtered (a still person: exactly the tracker row)
        r = BR.Refiner('size', size=dict(SIZE_T, src='trk')); zr = np.array([200., 250., 50., 120.])
        for k in range(5): b = r.step([(1, zr + [0, 0, 9, 9], zr)], k / 25., None)[1]
        np.testing.assert_allclose(b, zr, atol=1e-9)

    def test_size_mode_spreads_a_detector_switch(self):
        """near -> far: the far detector's box is twice as tall (whole person vs head-to-hip). The state is kept across the switch, so the
        height moves smoothly from 100 to 200 px, the top edge stays, and the step per frame stays small; no blend needed."""
        r = BR.Refiner('size', size=SIZE_T); hs = []; tops = []
        for k in range(75):   # 3 s at 25 fps, switch at 1 s
            h = 100. if k < 25 else 200.; z = np.array([300., 100. + h / 2, 40., h])
            b = r.step([(1, z, z)], k / 25., None, 'near' if k < 25 else 'far', True)[1]; hs.append(b[3]); tops.append(b[1] - b[3] / 2)
        hs = np.array(hs); self.assertEqual(r.tr[1]['k'], 74); self.assertEqual(r.blends, 0)
        np.testing.assert_allclose(tops, 100., atol=1e-9)
        self.assertTrue((np.diff(hs[24:]) >= -1e-9).all())          # no overshoot back
        self.assertLess(np.diff(hs).max(), 25.)                       # largest step a quarter of the 100 px change (one frame of the old jump)
        self.assertGreater(hs[-1], 195.)                              # reached the new height within 2 s

    def test_size_mode_camera_zoom_is_not_filtered(self):
        """the camera zooms in 2 % per frame (warp scale), the person stands still in the world: the box grows with the picture at once."""
        s = 1.02; H = np.array([[s, 0, 320 * (1 - s)], [0, s, 256 * (1 - s)]]); r = BR.Refiner('size', size=SIZE_T); z = np.array([320., 256., 40., 100.])
        for k in range(20):
            b = r.step([(1, z, z)], k / 25., H if k else None)[1]
            np.testing.assert_allclose(b[2:], z[2:], rtol=1e-9); z = BR.warp_box(z, H)

    def test_size_mode_real_seconds(self):
        """dt is real time: the same 1 s of a growing box gives about the same output at 10 and 30 fps."""
        out = []
        for fps in (10, 30):
            r = BR.Refiner('size', size=SIZE_T)
            for k in range(fps + 1):
                t = k / fps; z = np.array([300., 250., 40., 100. if t < 0.5 else 150.]); b = r.step([(1, z, z)], t, None)[1]
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
                    z = np.array([100. + 50. * t, 200., 40., 100.]); r.step([(1, z, z)], t, None, 'near', True); t += 0.04; k += 1
                t += stop; z = np.array([100. + 50. * t, 200., 40., 100.])
                b = r.step([(1, z, z)], t, None, 'far', True)[1]
                self.assertAlmostEqual(b[0], z[0], places=6, msg=(mode, stop)); self.assertEqual(r.blends, 0)
        # the instant (resident) switch, one frame interval: the blend still starts
        r = BR.Refiner('off', blend_s=0.3); z1 = np.array([300., 200., 60., 120.]); z2 = np.array([300., 230., 60., 180.])
        for k in range(3): r.step([(1, z1)], 0.04 * k, None, 'near', True)
        np.testing.assert_allclose(r.step([(1, z2)], 0.12, None, 'far', True)[1], z1, atol=1e-9); self.assertEqual(r.blends, 1)
        # a low frame rate (5 fps, 0.2 s per frame): one frame interval still blends
        r = BR.Refiner('off', blend_s=1.0)
        for k in range(3): r.step([(1, z1)], 0.2 * k, None, 'near', True)
        r.step([(1, z2)], 0.6, None, 'far', True); self.assertEqual(r.blends, 1)

    def test_blend_anchor_follows_a_camera_pan_that_starts_at_the_switch(self):
        """the gimbal pan starts on the switch frame (picture moves 20 px per frame), the person stands still in the world and both detectors
        give the same box: the output stays on the tracker row (was -20, -17.3, ... px behind)."""
        for mode in ('off', 'size'):
            r = BR.Refiner(mode, blend_s=0.3, size=SIZE_T); H = np.array([[1., 0, 20.], [0, 1., 0]]); z = np.array([300., 200., 40., 100.])
            for k in range(5): r.step([(1, z, z)], 0.04 * k, None, 'near', True)
            for k in range(5, 12):
                z = z + [20., 0, 0, 0]; b = r.step([(1, z, z)], 0.04 * k, H, 'far' if k >= 5 else 'near', True)[1]
                self.assertAlmostEqual(b[0], z[0], places=6, msg=(mode, k))
            self.assertEqual(r.blends, 1)
        # a steady pan plus the person walking 5 px per frame in the world: the carried output keeps both (warp + own motion)
        r = BR.Refiner('off', blend_s=0.3); z = np.array([300., 200., 40., 100.])
        for k in range(12):
            b = r.step([(1, z)], 0.04 * k, H if k else None, 'near' if k < 6 else 'far', True)[1]
            self.assertAlmostEqual(b[0], z[0], places=6, msg=k); z = z + [25., 0, 0, 0]

    def test_size_blend_keeps_the_tracker_position(self):
        """size mode + box_blend: the blend spreads only the size; the centre (or the aim point) stays the tracker row's
        (was up to 8 px off the tracker centre with blend 0.3)."""
        rng = np.random.default_rng(7)
        for anchor in ('centre', 'aim'):
            r = BR.Refiner('size', blend_s=0.3, size=dict(SIZE_T, anchor=anchor, src='trk'), aim_y=0.08); hs = []
            for k in range(40):
                h = (100. if k < 20 else 140.) + rng.normal(0, 3); zr = np.array([200. + 2 * k + rng.normal(0, 2), 250. + rng.normal(0, 2), 50., h])
                b = r.step([(5, zr, zr)], k / 25., None, 'near' if k < 20 else 'far', True)[5]; hs.append(b[3])
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
                b = r.step([(1, zr, zr)], k / 25., None)[1]; hb.append(b[3]); hz.append(zr[3])
                self.assertAlmostEqual(b[1] - b[3] / 2 + ay * b[3], zr[1] - zr[3] / 2 + ay * zr[3], places=9)
            self.assertLess(np.std(np.diff(hb)), 0.7 * np.std(np.diff(hz)))

    def test_blend_time_change_does_not_move_the_box_back(self):
        """the blend keeps the duration it started with (was: raising 0.3 -> 2.0 s at frame 3 moved the height 109.4 -> 102.0)."""
        r = BR.Refiner('off', blend_s=0.3); z1 = np.array([300., 200., 60., 100.]); z2 = np.array([300., 220., 60., 140.]); hs = []
        for k in range(3): r.step([(1, z1)], 0.04 * k, None, 'near', True)
        for k in range(3, 15):
            if k == 6: r.blend_s = 2.0
            hs.append(r.step([(1, z2)], 0.04 * k, None, 'far', True)[1][3])
        self.assertTrue((np.diff(hs) >= -1e-9).all(), hs); self.assertAlmostEqual(hs[-1], 140.)
        r.blend_s = 0.0; r.tr[1]['bl'] = (np.zeros(4), 0.5, 0.3)   # blend turned off: a running blend ends at once
        np.testing.assert_allclose(r.step([(1, z2)], 0.6, None, 'far', True)[1], z2)


class _Boom(Exception):
    pass


TRACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'track')


class ServerGlueTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(BR.BR); APP.args = SimpleNamespace(det='s640'); APP.CTL = dict(lead_motion=None); APP.tracker = object(); BR.BR['dir'] = TRACK_DIR
        self.now = patch.object(BR.time, 'time', return_value=4e9); self.now.start()   # a just-written table file is not skipped as 'still being copied'

    def tearDown(self):
        self.now.stop()
        BR.BR.clear(); BR.BR.update(self.saved)
        for k in ('args', 'CTL', 'tracker', 'SEL', 'sel_save', 'GB'): vars(APP).pop(k, None)

    def test_off_is_the_same_list(self):
        rows = rows_at([(1, 100., 200., 40., 100.)])
        self.assertIs(BR.apply(rows, None, 1.0), rows); self.assertNotIn('obj', BR.BR)
        self.assertEqual(BR.MODES, ('off', 'size', 'body'))

    def test_size_rows_and_state(self):
        ok, txt, apply_ = BR.parse_request({'box_refine': ['size'], 'box_blend': ['0.3']}); self.assertTrue(ok); apply_()
        self.assertEqual((BR.BR['mode'], BR.BR['blend_s']), ('size', 0.3))
        dets = np.array([[80., 150., 120., 250., 0.9, 0.]]); out = None
        for k in range(5):
            rows = rows_at([(7, 100. + k, 200., 40., 100.)]); rows.append([10., 10., 50., 40., 3, 0.9, 2, 5]); out = BR.apply(rows, dets + [k, 0, k, 0, 0, 0], 0.04 * k)
            self.assertIsNot(out, rows); self.assertEqual(len(out), 2); self.assertEqual(out[0][4:], rows[0][4:]); self.assertEqual(out[1], rows[1])   # a car is untouched
        st = BR.state(); self.assertEqual(st['mode'], 'size'); self.assertEqual(st['people'], 1); self.assertGreater(st['frames'], 0); self.assertEqual(st['err'], '')
        self.assertEqual([x['key'] for x in st['modes']], list(BR.MODES))
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
        with patch.dict(BR.SIZE, anchor='centre', src='trk'):   # the tracker row itself on the first frame
            BR.parse_request({'box_refine': ['body']})[2](); BR.apply(rows, dets, 1.04); BR.parse_request({'box_refine': ['size']})[2]()
            np.testing.assert_allclose(BR.apply(rows, dets, 1.08)[0][:4], rows[0][:4], atol=1e-9)

    def test_blend_change_keeps_the_state(self):
        BR.parse_request({'box_refine': ['size']})[2]()
        for k in range(3): BR.apply(rows_at([(3, 100., 200., 40., 100.)]), None, 0.04 * k)
        obj = BR.BR['obj']; st = obj.tr[3]
        ok, _, ap = BR.parse_request({'box_blend': ['0.4']}); self.assertTrue(ok); ap()
        BR.apply(rows_at([(3, 100., 200., 40., 100.)]), None, 0.12)
        self.assertIs(BR.BR['obj'], obj); self.assertIs(obj.tr[3], st); self.assertEqual(obj.blend_s, 0.4); self.assertEqual(st['k'], 3)
        BR.parse_request({'box_refine': ['body']})[2](); BR.apply(rows_at([(3, 100., 200., 40., 100.)]), None, 0.16)   # a new mode: a new state machine
        self.assertIsNot(BR.BR['obj'], obj); self.assertEqual(BR.BR['obj'].mode, 'body')

    def test_blend_only_keeps_tracker_box(self):
        ok, _, ap = BR.parse_request({'box_refine': ['off'], 'box_blend': ['0.3']}); self.assertTrue(ok); ap()
        rows = rows_at([(3, 100., 200., 40., 100.)]); dets = np.array([[70., 140., 130., 260., 0.9, 0.]])   # the matched detection differs from the tracker's box
        for k in range(3): out = BR.apply(rows, dets, 1.0 + 0.04 * k)
        np.testing.assert_allclose(out[0][:4], rows[0][:4])   # no detector change: the tracker's box, untouched

    def test_parse_request_checks(self):
        for bad in ('smooth', 'euro', 'gru', 'gru2'):   # 11-16: euro, gru and gru2 were removed
            ok, txt, _ = BR.parse_request({'box_refine': [bad]}); self.assertFalse(ok, bad); self.assertIn('off, size, body', txt)
        self.assertFalse(BR.parse_request({'box_blend': ['5']})[0])
        self.assertFalse(BR.parse_request({'box_blend': ['x']})[0])
        self.assertEqual(BR.BR['mode'], self.saved['mode'])
        self.assertTrue(BR.parse_request({'box_refine': ['size']})[0])

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

    def test_status_text_is_cleared_by_a_new_choice(self):
        BR.BR.update(why='변환표 파일이 없어 그대로 둠', ms=1.5)
        BR.parse_request({'box_refine': ['off']})[2](); self.assertEqual(BR.BR['why'], ''); self.assertEqual(BR.state()['ms'], 0.0)

    def test_auto_off_after_err_max_is_saved(self):
        saved = []; APP.sel_save = lambda **kw: saved.append(kw)
        BR.parse_request({'box_refine': ['size']})[2](); saved.clear(); rows = rows_at([(1, 100., 200., 40., 100.)])
        with patch.object(BR.Refiner, 'step', side_effect=_Boom('x')):
            for k in range(BR.ERR_MAX): BR.apply(rows, None, 0.04 * k)
        self.assertEqual(BR.BR['mode'], 'off'); self.assertEqual(saved, [dict(box_refine='off', box_blend=0.0)]); self.assertIn('오류', BR.BR['why'])

    def test_aim_y_from_the_gimbal_options(self):
        self.assertEqual(BR._aim_y(), BR.AIM_Y)
        APP.args.gimbal_aim_y = 0.08; self.assertAlmostEqual(BR._aim_y(), 0.08)
        APP.GB = SimpleNamespace(proc=SimpleNamespace(args=['bridge', '--aim-y', '0.3']))
        self.assertAlmostEqual(BR._aim_y(), 0.3)   # the running bridge's own option wins

    def test_startup_restores_the_blend_separately(self):
        """a saved body choice that is refused at startup (no table file) must not drop the saved blend time."""
        qs = BR.restore_queries({'box_refine': 'body', 'box_blend': 0.3, 'det': 'x'})
        self.assertEqual(qs, [{'box_refine': ['body']}, {'box_blend': ['0.3']}])
        BR.BR['dir'] = '/nonexistent_dir_for_test'; BR.BR.update(table=None, tfile=None, tmt=None)
        done = []
        for q in qs:
            ok, _, ap = BR.parse_request(q); done.append(ok)
            if ok: ap()
        self.assertEqual(done, [False, True]); self.assertEqual(BR.BR['blend_s'], 0.3); self.assertEqual(BR.BR['mode'], 'off')

    def test_removed_saved_mode_comes_back_as_off(self):
        """11-16: last_selection.json may still say euro / gru / gru2 (removed): startup takes 'off', says so in one line and saves 'off'."""
        import contextlib
        import io
        for old in ('euro', 'gru', 'gru2', 'GRU2 ', 'nonsense'):
            saved = []; APP.sel_save = lambda **kw: saved.append(kw); BR.BR['mode'] = 'size'; buf = io.StringIO()
            with contextlib.redirect_stdout(buf): qs = BR.restore_queries({'box_refine': old, 'box_blend': 0.0})
            self.assertEqual(qs, [{'box_refine': ['off']}, {'box_blend': ['0.0']}], old)
            log = [x for x in buf.getvalue().splitlines() if x.startswith('BOX_REFINE')]; self.assertEqual(len(log), 1, old)
            self.assertIn('removed' if old.strip().lower() in BR.REMOVED_MODES else 'unknown', log[0])
            for q in qs:
                ok, _, ap = BR.parse_request(q); self.assertTrue(ok)
                with contextlib.redirect_stdout(io.StringIO()): ap()
            self.assertEqual(BR.BR['mode'], 'off'); self.assertEqual(saved[0]['box_refine'], 'off')
            rows = rows_at([(1, 100., 200., 40., 100.)]); self.assertIs(BR.apply(rows, None, 1.0), rows)
        self.assertEqual(BR.restore_queries({'box_refine': 'size'}), [{'box_refine': ['size']}])   # the kept modes come back unchanged
        self.assertEqual(BR.restore_queries({}), [])


# ------------------------------------------------------------------------------------------------ body (11-15): one body box for near and far, lookup + size filter
PERSON_T = (-0.02, 0.535, 0.331)   # whole-person -> body constant of the test table (the committed file has its own, measured values)


def table_file(body_euro=None, person=PERSON_T):
    """a table file content with a simple lookup: boxes up to 60 px tall and at most 0.45 wide / tall are whole-person boxes
    (head-to-hip box = top the same, height x 0.52, width x 0.8), everything else already is the body box (identity); + a person constant."""
    tab = np.zeros((4, 4, 4)); tab[:2, :2] = (0.0, -0.24, math.log(0.8), math.log(0.52))
    w = {'tab': tab, 'lh_cen': np.log([20., 60., 120., 280.]), 'ar_cen': np.array([0.3, 0.45, 0.6, 0.9])}
    if body_euro is not None: w['euro'] = np.array(body_euro, np.float64)
    if person is not None: w['person'] = np.array(person, np.float64)
    return w


def committed_table():
    f = sorted(x for x in os.listdir(TRACK_DIR) if x.startswith(BR.TABLE_PREFIX) and x.endswith('.npz'))[-1]
    z = np.load(os.path.join(TRACK_DIR, f), allow_pickle=False); return BR.BodyTable({k: z[k] for k in z.files})


def top_of(b): return b[1] - b[3] / 2


def aim_pt(b, ay): return np.array([b[0], b[1] - b[3] / 2 + ay * b[3]])


POSE = 'bank:yolo26m_pose_otp2cocomotllvip_20261115_bodybox_farc2_kpt_640x480'
PLAIN = 'bank:yolo26x_flir_nms_640x480'


class BodyTests(unittest.TestCase):
    def test_already_body_boxes_stay(self):
        """close and mid people (the detector box already is head-to-hip): the converted box is the tracker row; only its size is smoothed, around
        the gimbal aim point (default 'aim'), the top edge ('top') or the centre ('centre')."""
        m = BR.BodyTable(table_file()); rng = np.random.default_rng(1)
        for w, h in ((190., 300.), (75., 120.), (45., 70.)):
            r = BR.Refiner('body', m); zr = np.array([300., 250., w, h])
            for k in range(6): np.testing.assert_allclose(r.step([(1, zr, zr)], 0.04 * k, None)[1], zr, atol=1e-9)
            for an in ('aim', 'top', 'centre'):
                r = BR.Refiner('body', m, body=dict(anchor=an), aim_y=0.37); hb = []; hz = []
                for k in range(60):   # a noisy tracker row: the anchor point exactly the row's, the height steps smaller
                    g = 1 + rng.normal(0, 0.04); zr = np.array([300. + 2 * k + rng.normal(0, 1), 250. + rng.normal(0, 1), w * g, h * g]); b = r.step([(1, zr, zr)], k / 25., None)[1]
                    self.assertAlmostEqual(b[0], zr[0], places=9)
                    if an == 'aim': np.testing.assert_allclose(aim_pt(b, 0.37), aim_pt(zr, 0.37), atol=1e-9)
                    elif an == 'top': self.assertAlmostEqual(top_of(b), top_of(zr), places=9)
                    else: self.assertAlmostEqual(b[1], zr[1], places=9)
                    hb.append(b[3]); hz.append(zr[3])
                self.assertLess(np.std(np.diff(hb)), 0.6 * np.std(np.diff(hz)), an); self.assertLess(abs(np.mean(hb[20:]) / h - 1), 0.02)

    def test_aim_point_is_the_converted_rows_at_three_aim_y(self):
        """anchor 'aim': at aim_y 0.5 (the device today), 0.37 and 0.08 the drawn box's aim point is the converted (unfiltered) row's on every row,
        so its jitter and its lag equal the converted row's by construction; only the drawn height is smoother. Mixed people: small whole-person boxes
        (converted to body boxes), mid and close body boxes, one growing, with a camera pan."""
        m = BR.BodyTable(table_file()); H = np.array([[1., 0, 3.], [0, 1., 0]])
        for ay in (0.5, 0.37, 0.08):
            rng = np.random.default_rng(int(ay * 100)); rf = BR.Refiner('body', m, aim_y=ay); rc = BR.Refiner('body', m, aim_y=ay, body=dict(mc_s=float('inf')))
            A = {i: [] for i in range(4)}; Ac = {i: [] for i in range(4)}; Hf = {i: [] for i in range(4)}; Hc = {i: [] for i in range(4)}
            for k in range(80):
                rows = []
                for i, (w, h, cx) in enumerate(((16., 44., 100.), (45., 70., 250.), (120., 200., 420.), (40., 60. * (1.02 ** k), 560.))):
                    g = 1 + rng.normal(0, 0.05); rows.append((i, np.array([cx + 3. * k + rng.normal(0, 1.5), 250. + rng.normal(0, 1.5), w * g, h * g])))
                bf = rf.step([(i, zr, zr) for i, zr in rows], k / 25., H if k else None); bc = rc.step([(i, zr, zr) for i, zr in rows], k / 25., H if k else None)
                for i, _ in rows: A[i].append(aim_pt(bf[i], ay)); Ac[i].append(aim_pt(bc[i], ay)); Hf[i].append(bf[i][3]); Hc[i].append(bc[i][3])
            for i in range(4):
                a, c = np.array(A[i]), np.array(Ac[i]); np.testing.assert_allclose(a, c, atol=1e-9, err_msg=str((ay, i)))
                jit = lambda v: np.sqrt(np.mean(np.sum((v[2:] - 2 * v[1:-1] + v[:-2]) ** 2, 1)))
                self.assertAlmostEqual(jit(a), jit(c), places=9)                                   # aim jitter = the converted row's
                if i < 3: self.assertLess(np.std(np.diff(Hf[i])), 0.8 * np.std(np.diff(Hc[i])), (ay, i))   # the drawn height is smoother
            self.assertLess(abs(Hc[0][0] / 44. - 0.52), 0.06)   # the small whole-person box is drawn as a body box

    def test_small_whole_person_boxes_become_body_boxes(self):
        m = BR.BodyTable(table_file()); r = BR.Refiner('body', m); zr = np.array([300., 250., 16., 44.])   # a far person: the detector gives the whole person
        for k in range(5):
            b = r.step([(1, zr, zr)], 0.04 * k, None)[1]
            np.testing.assert_allclose(b, [300., 250. - 22. + 0.52 * 22., 16. * 0.8, 44. * 0.52], atol=1e-9)   # from the first frame on: head top kept, x 0.52 tall
        # the tracker row is converted, not the matched detection (the row is what the page draws today)
        z = zr + [3., 2., 10., 30.]; b2 = BR.Refiner('body', m).step([(1, z, zr)], 0.0, None)[1]; np.testing.assert_allclose(b2, b, atol=1e-9)
        # in between the table is interpolated: no step where the detector's box definition changes with the person's size
        hs = []
        for h in np.linspace(50., 130., 81): hs.append(BR.Refiner('body', m).step([(1, np.array([300., 250., 0.4 * h, h]))], 0.0, None)[1][3])
        self.assertLess(np.abs(np.diff(hs)).max(), 2.5); self.assertAlmostEqual(hs[0], 26.0, places=6); self.assertAlmostEqual(hs[-1], 130.0, places=6)

    def test_committed_table_identity_on_body_boxes(self):
        """the committed (repaired) file: boxes of the pose detectors' body-box shape (aspect 0.60-0.80; their median is 0.62) come out exactly unchanged
        (height ratio 1, no sideways shift; the first table drew close people 6 % smaller and 3 % to the right), small whole-person boxes become about
        half as tall with the head top kept, and the file carries the size filter chosen on val-A and the whole-person constant."""
        m = committed_table(); self.assertIsNotNone(m.body); self.assertIsNotNone(m.person)
        for h in (30., 50., 70., 100., 130., 170., 210., 250., 300., 360., 420.):
            for ar in (0.60, 0.62, 0.66, 0.75):
                zr = np.array([320., 256., ar * h, h]); d = m.conv_delta(zr[None], 'pose')[0]
                np.testing.assert_allclose(d, 0.0, atol=1e-12, err_msg=str((h, ar)))
                np.testing.assert_allclose(BR.Refiner('body', m).step([(1, zr, zr)], 0.0, None, POSE)[1], zr, atol=1e-9)
            if h >= 114.: np.testing.assert_allclose(m.conv_delta(np.array([[320., 256., 0.45 * h, h]]), 'pose')[0], 0.0, atol=1e-12)   # from the 114 px row up: identity whatever the shape
            d = m.conv_delta(np.array([[320., 256., 0.58 * h, h]]), 'pose')[0]; self.assertLess(abs(d[3]), 0.10 if h >= 100. else 0.17, h)   # aspect 0.58 sits on the shape boundary (knots 0.515 / 0.60): up to 16 % smaller under 100 px (0.1 % of the body boxes are that narrow)
        for w, h in ((14., 36.), (18., 48.), (22., 58.), (30., 75.)):
            zr = np.array([320., 256., w, h]); b = BR.Refiner('body', m).step([(1, zr, zr)], 0.0, None, POSE)[1]
            self.assertGreater(b[3] / h, 0.45, (w, h)); self.assertLess(b[3] / h, 0.6, (w, h)); self.assertLess(abs(top_of(b) - top_of(zr)) / h, 0.04, (w, h)); self.assertLess(abs(b[0] - zr[0]) / h, 0.015)

    def test_detector_family_choice(self):
        """the table belongs to our pose detectors ('bodybox' in the detector key); every other detector gives whole-person boxes at every size and
        gets the constant (top + top_off x h, height k_h x h, width k_w x h): a close person 370 px tall becomes a body box under a plain detector."""
        for k in (POSE, 'bank:yolo26m_pose_otp2cocomot_20261114_bodybox_occl_640x480', 'BANK:X_BODYBOX_Y', None): self.assertEqual(BR.det_family(k), 'pose', k)
        for k in (PLAIN, 'bank:yolo26m_flir_nms_640x480', 'w480', 's480', 'm640', 's6', 'ft1', 'bank:yolox_s_nms_640x640', 'bank:yolov8n_nms_640x640'): self.assertEqual(BR.det_family(k), 'person', k)
        m = BR.BodyTable(table_file()); zr = np.array([320., 300., 150., 370.])
        bp = BR.Refiner('body', m).step([(1, zr, zr)], 0.0, None, POSE)[1]; np.testing.assert_allclose(bp, zr, atol=1e-9)   # pose: aspect 0.41 at 370 px = identity in the test table
        bq = BR.Refiner('body', m).step([(1, zr, zr)], 0.0, None, PLAIN)[1]
        to, kh, kw = PERSON_T; self.assertAlmostEqual(bq[3], kh * 370.); self.assertAlmostEqual(bq[2], kw * 370.); self.assertAlmostEqual(top_of(bq), top_of(zr) + to * 370.); self.assertAlmostEqual(bq[0], 320.)
        mc = committed_table(); to, kh, kw = mc.person   # the committed constant: 0.53-0.54 of the person height from about the top, width tied to the height
        self.assertTrue(0.52 < kh < 0.55 and -0.04 < to < 0.01 and 0.30 < kw < 0.36, mc.person)
        b0 = BR.BodyTable(dict(tab=m.tab, lh_cen=m.lh_cen, ar_cen=m.ar_cen))   # a file without the constant (the first file): the table for every detector
        np.testing.assert_allclose(BR.Refiner('body', b0).step([(1, zr, zr)], 0.0, None, PLAIN)[1], zr, atol=1e-9)

    def _kalman_rows(self, h0, h1, k_sw, n, cx=320., top0=120., top1=120., w0=None, w1=None, gain=0.55):
        """a tracker row that follows a detector switch at frame k_sw like the server's Kalman filter (top, height and width move a share 'gain'
        per frame toward the new detector's box), with the detections of each frame"""
        w0 = w0 or 0.62 * h0; w1 = w1 or 0.62 * h1; rh, rw, rt = h0, w0, top0; out = []
        for k in range(n):
            dh, dw, dt = (h0, w0, top0) if k < k_sw else (h1, w1, top1)
            if k >= k_sw: rh += gain * (dh - rh); rw += gain * (dw - rw); rt += gain * (dt - rt)
            out.append((np.array([cx, dt + dh / 2, dw, dh]), np.array([cx, rt + rh / 2, rw, rh])))
        return out

    def test_switch_mid_track_pose_to_plain_and_back(self):
        """the page / auto_det / resident pair changes the detector while a close person is followed: pose (body box 200 px) -> plain YOLO
        (whole-person box 370 px, the tracker row needs a few frames to get there) -> pose again. The state survives, the drawn height and the aim
        point stay continuous (no dip to 0.53 x 200 and back; the aim point moves at most 3 % of the height in a frame where the row's own moves 23 %), and the box is
        the body box before and after."""
        m = BR.BodyTable(table_file())
        for ay in (0.5, 0.08):
            r = BR.Refiner('body', m, aim_y=ay, body=dict(mc_s=2.0, beta_s=1.0)); hs = []; aims = []
            tw = 120. - PERSON_T[0] * 370.   # the plain detector's whole-person box: its top lies 0.02 x 370 px below the body box's (the measured relation)
            seq = self._kalman_rows(200., 370., 25, 50, top1=tw, w0=124., w1=150.) + self._kalman_rows(370., 200., 1, 41, top0=tw, w0=150., w1=124.)[1:]
            for k, (z, zr) in enumerate(seq):
                key = POSE if k < 25 or k >= 50 else PLAIN
                b = r.step([(1, z.copy(), zr)], k / 25., None, key, True)[1]; hs.append(b[3]); aims.append(aim_pt(b, ay)[1])
            self.assertEqual(r.tr[1]['k'], len(seq) - 1); self.assertGreaterEqual(r.switches, 2)
            hs = np.array(hs); self.assertLess(np.abs(hs / 200. - 1).max(), 0.06, (ay, hs.min(), hs.max()))   # the body box all along (0.535 x 370 = 198)
            self.assertLess(np.abs(np.diff(aims)).max(), 0.03 * 200., ay)   # the row's own aim point steps 46 px (aim 0.5) on the first frame
            # without the transition (the first version) the same switch dips the converted height to about 0.6 x and jumps the aim point
            r0 = BR.Refiner('body', m, aim_y=ay, body=dict(mc_s=float('inf'), sw=False)); h0 = []
            for k, (z, zr) in enumerate(seq): h0.append(r0.step([(1, z.copy(), zr)], k / 25., None, POSE if k < 25 or k >= 50 else PLAIN, True)[1][3])
            self.assertLess(min(h0) / 200., 0.85)

    def test_switch_near_far_small_person(self):
        """near (body box 30 px) <-> far (whole-person box 58 px) for a small person, the row moving over a few frames: the drawn box stays the
        body box (the table alone maps the in-between rows to 0.6-0.8 of it), and a switch where both detectors agree starts no transition."""
        m = BR.BodyTable(table_file()); r = BR.Refiner('body', m, body=dict(mc_s=float('inf'))); hs = []
        seq = self._kalman_rows(30., 58., 20, 45, w0=18.6, w1=20.)
        for k, (z, zr) in enumerate(seq): hs.append(r.step([(1, z.copy(), zr)], k / 25., None, 'near_bodybox' if k < 20 else 'far_bodybox', True)[1][3])
        self.assertLess(np.abs(np.array(hs) / 30. - 1).max(), 0.08, (min(hs), max(hs)))
        r = BR.Refiner('body', m); z = np.array([300., 250., 120., 200.])
        for k in range(10): r.step([(1, z.copy(), z)], k / 25., None, 'near_bodybox' if k < 5 else 'far_bodybox', True)
        self.assertEqual(r.switches, 1); self.assertIsNone(r.tr[1]['tw'])   # same box from both: the transition ends at once (|log h| < SW_SAME)
        # an id missing on the first frames of the new detector (> BLEND_RECENT): no transition, its own conversion
        r = BR.Refiner('body', m)
        for k in range(5): r.step([(1, z.copy(), z)], k / 25., None, 'near_bodybox', True)
        for k in range(BR.BLEND_RECENT + 1): r.step([], (5 + k) / 25., None, 'far_bodybox', True)
        r.step([(1, z.copy(), z)], 0.5, None, 'far_bodybox', True); self.assertEqual(r.switches, 0)

    def test_new_track_and_fast_growth_are_not_drawn_small(self):
        """the size filter of the committed file: a new track starts at the converted size (ratio 1 on its first row); a person growing 50 % per
        second (walking toward the camera) is drawn at >= 0.9 of the converted height after the first rows (the first version: 0.75-0.9 for a second)."""
        m = committed_table(); rng = np.random.default_rng(4)
        for fps in (25., 9.):
            r = BR.Refiner('body', m); rc = BR.Refiner('body', m, body=dict(mc_s=float('inf'))); u = []
            for k in range(int(2 * fps)):
                h = 80. * math.exp(0.5 * k / fps) * (1 + rng.normal(0, 0.02)); zr = np.array([320., 120. + h / 2, 0.62 * h, h])
                b = r.step([(1, zr, zr)], k / fps, None, POSE)[1]; c = rc.step([(1, zr, zr)], k / fps, None, POSE)[1]; u.append(b[3] / c[3])
            self.assertAlmostEqual(u[0], 1.0, places=12); self.assertGreater(min(u[3:]), 0.90, (fps, min(u)))

    def test_handover_reset_and_detector_switch(self):
        m = BR.BodyTable(table_file()); z = np.array([300., 250., 50., 120.]); ra = BR.Refiner('body', m); rb = BR.Refiner('body', m); rng = np.random.default_rng(2)
        for k in range(8):
            zk = z + [2 * k, 0, 0, rng.normal(0, 5)]; ra.step([(1, zk, zk)], 0.04 * k, None); rb.step([(1, zk, zk)], 0.04 * k, None)
        st1 = ra.tr[1]; z9 = z + [16, 0, 0, 9.]
        a = ra.step([(9, z9, z9)], 0.32, None)[9]; b = rb.step([(1, z9, z9)], 0.32, None)[1]   # KeepID gave the person another number
        self.assertIs(ra.tr[9], st1); self.assertEqual(ra.tr[9]['k'], 8); self.assertEqual(ra.handovers, 1); np.testing.assert_allclose(a, b, atol=1e-12); self.assertNotAlmostEqual(a[3], z9[3], places=1)
        for miss, kept in ((BR.LOST_MAX, True), (BR.LOST_MAX + 1, False)):   # the gap rule of the other modes
            r = BR.Refiner('body', m)
            for k in range(3): r.step([(5, z, z)], 0.04 * k, None)
            for k in range(miss): r.step([], 0.2 + 0.04 * k, None)
            r.step([(5, z, z)], 5.0, None); self.assertEqual(r.tr[5]['k'] == 3, kept, miss)
        H = np.array([[1., 0, 20.], [0, 1., 0]]); r = BR.Refiner('body', m)   # warp-aware jump rule: a 12 px person in a 20 px / frame pan keeps the state
        for k in range(30):
            zk = np.array([100. + 20 * k, 200., 5., 12.]); r.step([(1, zk, zk)], k / 25., H)
        self.assertEqual(r.tr[1]['k'], 29)
        far = z + [400., 0, 0, 80.]; r = BR.Refiner('body', m); r.step([(2, z, z)], 0.0, None); r.step([(2, z, z)], 0.04, None)   # another person: from that box at once
        np.testing.assert_allclose(r.step([(2, far, far)], 0.08, None)[2], far, atol=1e-9); self.assertTrue(r.tr[2]['new'])
        # near -> far for a small person, the row jumping at once: the state is kept and the drawn box stays the body box (head top, height)
        r = BR.Refiner('body', m); hs = []; tops = []
        for k in range(60):
            zk = np.array([300., 213., 16., 26.]) if k < 30 else np.array([300., 225., 16., 50.])   # same head top (200), near box ar 0.62, far box ar 0.32
            b = r.step([(1, zk.copy(), zk)], k / 25., None, 'near_bodybox' if k < 30 else 'far_bodybox', True)[1]; hs.append(b[3]); tops.append(top_of(b))
        self.assertEqual(r.tr[1]['k'], 59); self.assertEqual(r.blends, 0); np.testing.assert_allclose(tops, 200., atol=1e-9); np.testing.assert_allclose(hs, 26., atol=1e-6)

    def test_camera_zoom_and_real_seconds(self):
        m = BR.BodyTable(table_file()); s = 1.02; H = np.array([[s, 0, 320 * (1 - s)], [0, s, 256 * (1 - s)]]); r = BR.Refiner('body', m); z = np.array([320., 256., 60., 120.])
        for k in range(20):   # the camera zooms 2 % per frame: the box grows with the picture at once (the camera scale is not filtered)
            b = r.step([(1, z, z)], k / 25., H if k else None)[1]; np.testing.assert_allclose(b, z, rtol=1e-9); z = BR.warp_box(z, H)
        out = []
        for fps in (10, 30):   # the same second of a growing box at 10 and 30 fps
            r = BR.Refiner('body', m)
            for k in range(fps + 1):
                t = k / fps; z = np.array([300., 250., 60., 100. if t < 0.5 else 150.]); b = r.step([(1, z, z)], t, None)[1]
            out.append(b[3])
        self.assertLess(abs(out[0] - out[1]), 0.06 * 50)

    def test_filter_values_from_the_file_or_the_default(self):
        self.assertEqual(BR.Refiner('body', BR.BodyTable(table_file())).body, BR.BODY)
        m = BR.BodyTable(table_file(body_euro=[2.0, 1.0, 1.0])); self.assertEqual(BR.Refiner('body', m).body, dict(BR.BODY, mc_s=2.0, beta_s=1.0, dcut=1.0))
        self.assertEqual(BR.Refiner('body', m, body=dict(mc_s=5.0)).body['mc_s'], 5.0)
        r = BR.Refiner('body', None); zr = np.array([200., 250., 20., 50.]); np.testing.assert_allclose(r.step([(1, zr + 1, zr)], 0.0, None)[1], zr)   # no table: the tracker row
        self.assertEqual(BR.BODY['anchor'], 'aim')


class BodyGlueTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(BR.BR); APP.args = SimpleNamespace(det='s640'); APP.CTL = dict(lead_motion=None); APP.tracker = object(); BR.BR['dir'] = TRACK_DIR
        self.now = patch.object(BR.time, 'time', return_value=4e9); self.now.start()

    def tearDown(self):
        self.now.stop()
        BR.BR.clear(); BR.BR.update(self.saved)
        for k in ('args', 'CTL', 'tracker', 'SEL', 'sel_save'): vars(APP).pop(k, None)

    def test_select_run_reset_and_restore(self):
        saved = []; APP.sel_save = lambda **kw: saved.append(kw)
        ok, txt, ap = BR.parse_request({'box_refine': ['body']}); self.assertTrue(ok, txt); ap(); self.assertEqual(saved[-1], dict(box_refine='body', box_blend=0.0))
        st = BR.state(); self.assertTrue(next(x for x in st['modes'] if x['key'] == 'body')['available'])
        self.assertEqual(st['mode_name'], BR.MODE_KO['body']); self.assertEqual(st['body_file'], BR.BR['tfile']); self.assertTrue(st['body_file'].startswith(BR.TABLE_PREFIX))
        APP.args.det = POSE   # our pose detector: the table
        dets = np.array([[60., 100., 140., 300., 0.9, 0.]]); out = None   # the matched detection differs from the tracker row: body converts the ROW
        for k in range(4):
            rows = rows_at([(7, 320., 256., 190., 300.), (8, 100., 200., 16., 40.)]); rows.append([10., 10., 50., 40., 3, 0.9, 2, 5]); out = BR.apply(rows, dets, 0.04 * k)
            self.assertIsNot(out, rows); self.assertEqual(out[2], rows[2]); self.assertEqual(out[0][4:], rows[0][4:])
        np.testing.assert_allclose(out[0][:4], rows[0][:4], atol=1e-6)                             # the close person (body-box shape): unchanged
        self.assertLess((out[1][3] - out[1][1]) / 40., 0.66); self.assertLess(abs(out[1][1] - rows[1][1]), 0.06 * 40)   # the far person: half as tall, head top kept
        st = BR.state(); self.assertEqual(st['people'], 2); self.assertEqual(st['err'], ''); self.assertEqual(st['body_family'], 'pose'); self.assertEqual(st['body_family_name'], BR.FAMILY_KO['pose'])
        for k, det in enumerate((PLAIN, POSE)):   # a detector change keeps the state machine and every person's state
            APP.args.det = det; obj = BR.BR['obj']; BR.apply(rows_at([(7, 320., 256., 190., 300.)]), None, 0.2 + 0.04 * k); self.assertIs(BR.BR['obj'], obj); self.assertGreaterEqual(obj.tr[7]['k'], 4)
        APP.args.det = 's640'; self.assertEqual(BR.state()['body_family'], 'person')   # an internal plain detector name: the whole-person constant
        o2 = BR.apply(rows_at([(9, 120., 300., 150., 370.)]), None, 0.3); to, kh, kw = BR.BR['table'].person   # far from id 7 (no hand-over)
        self.assertAlmostEqual(o2[0][3] - o2[0][1], kh * 370., places=4); self.assertAlmostEqual(o2[0][2] - o2[0][0], kw * 370., places=4)   # a new person: the constant at once
        APP.tracker = object(); BR.apply(rows_at([(7, 320., 256., 190., 300.)]), None, 0.3); self.assertEqual(BR.BR['obj'].tr[7]['k'], 0)   # a new tracker: from scratch
        BR.parse_request({'box_refine': ['off']})[2](); rows = rows_at([(7, 320., 256., 190., 300.)]); self.assertIs(BR.apply(rows, None, 1.0), rows); self.assertNotIn('obj', BR.BR)
        BR.BR['mode'] = 'off'; done = []   # startup restore of the saved choice
        for q in BR.restore_queries({'box_refine': 'body', 'box_blend': 0.0}):
            o, _, ap = BR.parse_request(q); done.append(o)
            if o: ap()
        self.assertEqual(done, [True, True]); self.assertEqual(BR.BR['mode'], 'body')

    def test_no_table_file(self):
        BR.BR['dir'] = '/nonexistent_dir_for_test'; BR.BR.update(table=None, tfile=None, tmt=None)
        ok, txt, _ = BR.parse_request({'box_refine': ['body']}); self.assertFalse(ok); self.assertIn('변환표', txt); self.assertEqual(BR.BR['mode'], 'off')
        self.assertFalse(next(x for x in BR.state()['modes'] if x['key'] == 'body')['available'])
        BR.BR['mode'] = 'body'; rows = rows_at([(1, 100., 200., 40., 100.)]); self.assertIs(BR.apply(rows, None, 1.0), rows); self.assertIn('변환표', BR.BR['why'])

    def test_a_later_file_is_taken_without_a_code_change(self):
        """the newest table file by name; a later file replaces it; an unreadable newest file is said so, not silently the older one."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            np.savez(os.path.join(d, BR.TABLE_PREFIX + '20261115.npz'), **table_file())
            BR.BR['dir'] = d; BR.BR.update(table=None, tfile=None, tmt=None)
            m2 = BR.scan_table(); self.assertEqual(BR.BR['tfile'], BR.TABLE_PREFIX + '20261115.npz'); self.assertIsNone(m2.body)
            BR.parse_request({'box_refine': ['body']})[2](); BR.apply(rows_at([(1, 100., 200., 16., 40.)]), None, 0.0); obj = BR.BR['obj']; self.assertIs(obj.table, m2)
            np.savez(os.path.join(d, BR.TABLE_PREFIX + '20261116.npz'), **table_file(body_euro=[2.0, 0.5, 1.0]))   # a later file arrives
            m3 = BR.scan_table(); self.assertIsNot(m3, m2); self.assertEqual(BR.BR['tfile'], BR.TABLE_PREFIX + '20261116.npz'); self.assertEqual(BR.state()['body']['mc_s'], 2.0)
            BR.apply(rows_at([(1, 100., 200., 16., 40.)]), None, 0.04); self.assertIsNot(BR.BR['obj'], obj); self.assertIs(BR.BR['obj'].table, m3); self.assertEqual(BR.BR['obj'].body['mc_s'], 2.0)
            np.savez(os.path.join(d, BR.TABLE_PREFIX + '20261199_broken.npz'), x=np.zeros(3)); self.assertIsNone(BR.scan_table()); self.assertIn('읽지 못함', BR.BR['tnote'])


if __name__ == '__main__':
    unittest.main()
