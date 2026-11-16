"""11-14 checks of the keypoint path: keypoint decode of NMS-kept boxes, keypoint position memory (zero output = plain Kalman path), path data for the overlay.
Run from tracking_server/:  python -m unittest src.track.test_position_kpt"""
import os
import unittest
from types import SimpleNamespace

import numpy as np

from ..detect import detect_frame as DF
from . import position_memory as PM
from ..common.shared_values import TRACKED


class KptDecodeTests(unittest.TestCase):
    def test_anchor_and_decode(self):
        h, w = 480, 640; gx, gy, gs = DF._kp_grid(h, w); n = len(gx)
        dec = np.zeros((1, 20, n), np.float32); kraw = np.zeros((1, 51, n), np.float32)
        a = 4800 + 20 * 40 + 7   # stride 16, row 20, col 7
        self.assertEqual((gs[a], gx[a], gy[a]), (16, 7.5, 20.5))
        dec[0, :4, a] = [200, 300, 50, 120]; dec[0, 4, a] = 0.9
        dec[0, :4, a + 1] = [205, 300, 50, 120]; dec[0, 4, a + 1] = 0.9   # same score, other box: must not be picked
        kraw[0, 0::3, a] = 1.0; kraw[0, 1::3, a] = -2.0; kraw[0, 2::3, a] = 3.0
        p_ = np.array([[175, 240, 225, 360, 0.9, 0]], np.float32)
        DF.attach_kpts(dec, kraw, p_, h, w, 16.0)
        k = DF.KP['cur']['kpts'][0]
        np.testing.assert_allclose(k[:, 0], (1.0 + 7.5) * 16, atol=1e-4)
        np.testing.assert_allclose(k[:, 1], (-2.0 + 20.5) * 16 + 16, atol=1e-4)
        np.testing.assert_allclose(k[:, 2], 1 / (1 + np.exp(-3.0)), atol=1e-6)
        np.testing.assert_allclose(DF.KP['cur']['boxes'][0], [175, 256, 225, 376])

    def test_no_person(self):
        DF.attach_kpts(np.zeros((1, 20, 6300), np.float32), np.zeros((1, 51, 6300), np.float32), np.zeros((0, 6), np.float32), 480, 640, 16.0)
        self.assertEqual(DF.KP['cur']['kpts'].shape, (0, 17, 3))


# COCO keypoints of a standing person relative to a head-to-hip box (centre, box height): nose at the top, hips at the bottom edge, knees / ankles below
KP_REL = np.array([[0, -0.40], [-0.03, -0.43], [0.03, -0.43], [-0.07, -0.42], [0.07, -0.42], [-0.17, -0.20], [0.17, -0.20], [-0.22, 0.05], [0.22, 0.05],
                   [-0.22, 0.28], [0.22, 0.28], [-0.10, 0.42], [0.10, 0.42], [-0.10, 0.80], [0.10, 0.80], [-0.10, 1.15], [0.10, 1.15]])


class ZeroCell:
    kpt = True
    def init(self): return np.zeros(3)
    def step(self, x, st):
        assert x.shape == (PM.KP_NF,) and np.isfinite(x).all(); return np.zeros(4), st


class ZeroCellFix(ZeroCell):
    warp_fix = True   # a file trained with lost step 1 = the warp of its own frame (npz opt_warp_fix = 1)


class FakeKF:
    warp = None
    def multi_predict(self, mean, cov):
        mean = mean.copy(); mean[:, :4] += mean[:, 4:8]; return mean, cov


class KptMemoryTests(unittest.TestCase):
    def setUp(self):
        PM.POSP.update(key='gru_kpt', model=ZeroCell(), gen=PM.POSP['gen'] + 1, err=0, used=0, show=True)
        DF.KP['cur'] = DF.KP['prev'] = None

    def tearDown(self):
        PM.POSP.update(key='kalman', model=None, gen=PM.POSP['gen'] + 1, err=0, used=0, show=False)

    def test_zero_output_is_plain_path_and_trail(self):
        t = SimpleNamespace(state=TRACKED, cls=0, track_id=3, pp_inn=np.zeros(4)); kf = FakeKF()
        mean = np.array([[100., 200., 40., 100., 2., 1., 0., 0.]]); cov = np.zeros((1, 8, 8))
        for _ in range(6):   # seen frames (keypoints of the previous frame's matching detection)
            b = mean[0, :4]; DF.KP['prev'] = dict(boxes=np.array([[b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2]], np.float32),
                                                   kpts=np.tile(np.array([[b[0], b[1] - 30, 0.9]], np.float32), (17, 1))[None])
            mean, cov = PM.pos_predict(kf, [t], mean, cov)
        self.assertEqual(t.pk['n'], 6); last = mean[0, :2].copy()
        t.state = 2   # lost
        mean, cov = PM.pos_predict(kf, [t], mean, cov)   # steps 1 and 2 of the plain path from the last post state
        self.assertEqual(t.pk['j'], 2); self.assertEqual(PM.POSP['err'], 0)
        np.testing.assert_allclose(mean[0, :2], t.pk['c'])   # zero correction = the plain Kalman path
        e = PM.POSP['boxes'][0]; self.assertEqual(e[0], 3); self.assertEqual(e[2]['src'], 'gru_kpt'); self.assertEqual(e[2]['n'], 1); self.assertIsNotNone(e[2]['last'])
        for _ in range(20): mean, cov = PM.pos_predict(kf, [t], mean, cov)
        self.assertEqual(t.pk['j'], PM.KP_STEPS)   # stops after 15 frames, then Kalman alone

    def _seen_then_lost(self, w_last_seen_call, w_first_lost_call):
        """6 seen frames, the 6th pos_predict call (still TRACKED, = lost step 1 if not matched) with camera warp w_last_seen_call, then the first
        LOST call (= lost step 2) with w_first_lost_call -> (post state, plain-path centre after step 2)"""
        t = SimpleNamespace(state=TRACKED, cls=0, track_id=5, pp_inn=np.zeros(4)); kf = FakeKF()
        mean = np.array([[100., 200., 40., 100., 2., 1., 0., 0.]]); cov = np.zeros((1, 8, 8))
        for k in range(6):
            b = mean[0, :4]; DF.KP['prev'] = dict(boxes=np.array([[b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2]], np.float32),
                                                   kpts=np.zeros((1, 17, 3), np.float32))
            kf.warp = w_last_seen_call if k == 5 else None
            mean, cov = PM.pos_predict(kf, [t], mean, cov)
        post = t.pk['post'].copy(); t.state = 2; kf.warp = w_first_lost_call
        PM.pos_predict(kf, [t], mean, cov)
        return post, t.pk['c'].copy()

    def test_lost_step_1_uses_the_camera_warp_of_its_own_frame(self):
        # 11-16: before, the first LOST call applied ITS warp to steps 1 and 2 and never the warp of step 1's frame (for files with opt_warp_fix)
        PM.POSP.update(model=ZeroCellFix())
        shift = np.array([[1., 0., 10.], [0., 1., 0.]])
        post, c = self._seen_then_lost(shift, None)   # the camera moved 10 px in step 1's frame, stood still in step 2's
        np.testing.assert_allclose(c, post[:2] + [10., 0.] + 2 * post[4:6])
        post, c = self._seen_then_lost(None, shift)   # still in step 1's frame, moved in step 2's: the shift once, not twice
        np.testing.assert_allclose(c, post[:2] + [10., 0.] + 2 * post[4:6])

    def test_files_without_the_warp_flag_keep_the_order_they_were_trained_with(self):
        # 11-16 review: the Mamba file and the older files were trained with the old order (fix_lib fixwarp=False: the warp of lost step 2 for
        # steps 1 and 2); the step-1 change must not reach them (the 11-16 server gave the Mamba file up to 10.8 px other boxes with a moving camera)
        shift = np.array([[1., 0., 10.], [0., 1., 0.]])
        post, c = self._seen_then_lost(shift, None)   # step 1's frame moved: the old order never applies it
        np.testing.assert_allclose(c, post[:2] + 2 * post[4:6])
        post, c = self._seen_then_lost(None, shift)   # step 2's frame moved: the old order applies it twice
        np.testing.assert_allclose(c, post[:2] + [20., 0.] + 2 * post[4:6])


class RecCell(ZeroCell):
    """zero output, keeps every input row"""
    def __init__(self, edges): self.edges = edges; self.xs = []
    def step(self, x, st): self.xs.append(x.copy()); return np.zeros(4), st


class EdgeFlagTests(unittest.TestCase):
    """11-16: a file with opt_no_edges = 1 gets the constant 5 (= far from every edge) for the 4 picture-edge inputs, seen and lost; others the distances"""
    def tearDown(self):
        PM.POSP.update(key='kalman', model=None, gen=PM.POSP['gen'] + 1, err=0, used=0, show=False)

    def _rows(self, cell):
        PM.POSP.update(key='gru_kpt', model=cell, gen=PM.POSP['gen'] + 1, err=0, used=0, show=False)
        t = SimpleNamespace(state=TRACKED, cls=0, track_id=9, pp_inn=np.zeros(4)); kf = FakeKF()
        mean = np.array([[320., 440., 20., 50., 0., 0., 0., 0.]]); cov = np.zeros((1, 8, 8))   # box bottom 31 px above the picture bottom (496)
        DF.KP['prev'] = dict(boxes=np.array([[310., 415., 330., 465.]], np.float32), kpts=np.zeros((1, 17, 3), np.float32))
        for _ in range(5): mean, cov = PM.pos_predict(kf, [t], mean, cov)
        t.state = 2; mean, cov = PM.pos_predict(kf, [t], mean, cov)
        return np.array(cell.xs)

    def test_no_edge_file_gets_constant_edges(self):
        x = self._rows(RecCell(False)); np.testing.assert_allclose(x[:, 12:16], PM.KP_CLIP)   # 5 seen rows (warm-up) + 2 lost steps
        x = self._rows(RecCell(True)); np.testing.assert_allclose(x[0, 15], 31. / 50.)
        z = {'cell.cell.weight_ih': np.zeros((6, PM.KP_NF)), 'cell.cell.weight_hh': np.zeros((6, 2)), 'cell.cell.bias_ih': np.zeros(6),
             'cell.cell.bias_hh': np.zeros(6), 'cell.head.weight': np.zeros((4, 2)), 'cell.head.bias': np.zeros(4)}   # export key names (torch PosNet.cell.*)
        self.assertTrue(PM.PosKpt(z).edges); self.assertFalse(PM.PosKpt(dict(z, opt_no_edges=np.array([1.], np.float32))).edges)

    def test_file_flags_and_the_shipped_files(self):
        # 11-16 review: opt_warp_fix (lost step 1 with its own frame's warp) and opt_box_mode_targets (no Kalman rule for big people without keypoints)
        z = {'cell.cell.weight_ih': np.zeros((6, PM.KP_NF)), 'cell.cell.weight_hh': np.zeros((6, 2)), 'cell.cell.bias_ih': np.zeros(6),
             'cell.cell.bias_hh': np.zeros(6), 'cell.head.weight': np.zeros((4, 2)), 'cell.head.bias': np.zeros(4)}
        m = PM.PosKpt(z); self.assertFalse(m.warp_fix); self.assertTrue(m.nokpt_kalman)
        m = PM.PosKpt(dict(z, opt_warp_fix=np.array([1.], np.float32), opt_box_mode_targets=np.array([1.], np.float32)))
        self.assertTrue(m.warp_fix); self.assertFalse(m.nokpt_kalman)
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'track')
        for f, warp_fix, edges in (('pos_gru_kpt_tmotdance_20261116.npz', True, False), ('pos_mamba_kpt_tmotdance_20261115.npz', False, True)):
            p = os.path.join(d, f)
            if not os.path.exists(p): continue
            m = PM.PosKpt(dict(np.load(p)))
            with self.subTest(f=f):   # the GRU was trained with fix_lib fixwarp=True and no edge inputs; the Mamba file with the old builder
                self.assertEqual((m.warp_fix, m.edges, m.nokpt_kalman), (warp_fix, edges, True))


class ShrinkCell(ZeroCell):
    """constant output: height x exp(log 1.5 x tanh(-1)) = 0.74x, centre unchanged; counts its lost steps"""
    def __init__(self, nokpt_kalman=True): self.nokpt_kalman = nokpt_kalman; self.lost_steps = 0
    def step(self, x, st):
        self.lost_steps += x[67] > 0; return np.array([0., 0., 0., -1.]), st


class NoKeypointKalmanTests(unittest.TestCase):
    """11-16 review: a lost person whose last seen box is >= 64 px with no visible keypoint gets the plain Kalman path (files without
    opt_box_mode_targets: their targets ask such a person for a head-to-hip box although the detector draws the whole body)"""
    def tearDown(self):
        PM.POSP.update(key='kalman', model=None, gen=PM.POSP['gen'] + 1, err=0, used=0, show=False); DF.KP['cur'] = DF.KP['prev'] = None

    def _lost_h(self, cell, h, n_vis):
        PM.POSP.update(key='gru_kpt', model=cell, gen=PM.POSP['gen'] + 1, err=0, used=0, show=False)
        t = SimpleNamespace(state=TRACKED, cls=0, track_id=4, pp_inn=np.zeros(4)); kf = FakeKF(); w = 0.4 * h
        mean = np.array([[320., 200., w, h, 0., 0., 0., 0.]]); cov = np.zeros((1, 8, 8))
        kp = np.zeros((1, 17, 3), np.float32); kp[0, :, 0] = 320.; kp[0, :, 1] = 200.; kp[0, :n_vis, 2] = 0.9
        DF.KP['prev'] = dict(boxes=np.array([[320. - w / 2, 200. - h / 2, 320. + w / 2, 200. + h / 2]], np.float32), kpts=kp)
        for _ in range(6): mean, cov = PM.pos_predict(kf, [t], mean, cov)
        t.state = 2
        for _ in range(5): mean, cov = PM.pos_predict(kf, [t], mean, cov)
        self.assertEqual(PM.POSP['err'], 0)
        return mean[0, 3] / h, cell.lost_steps

    def test_big_person_without_keypoints_gets_the_kalman_path(self):
        for h, n_vis, kalman in ((63., 0, False), (64., 0, True), (200., 0, True), (64., 1, False), (200., 17, False)):
            with self.subTest(h=h, n_vis=n_vis):
                r, n = self._lost_h(ShrinkCell(), h, n_vis)
                if kalman: self.assertEqual((r, n), (1.0, 0))   # plain path: the still person keeps its box, the cell never steps a lost frame
                else: self.assertLess(r, 0.8); self.assertEqual(n, 6)
        r, n = self._lost_h(ShrinkCell(nokpt_kalman=False), 200., 0); self.assertLess(r, 0.8)   # a file with opt_box_mode_targets: the cell's box


class ShippedGruNearBottomTests(unittest.TestCase):
    """11-16: the shipped GRU file must not shrink and lift a lost person because its box is near the picture bottom. The 11-15 file did
    (on our camera the lost box at step 2 was 0.85x and 0.16 box heights higher, 3x the Kalman error), mainly because its picture-edge inputs
    carried the public cameras' scene layout (holding them at 5 on the old file: TEST 0.116 vs Kalman 0.110); smaller parts: DanceTrack's
    letterbox measured as the device picture and targets made from gt boxes cut by the picture edge. The 11-16 file has no edge inputs."""
    def setUp(self):
        self.dir0 = PM.POSP['dir']; PM.POSP['dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'track')
        PM.POSP['avail'].pop('gru_kpt', None); PM.pos_scan(); a = PM.POSP['avail'].get('gru_kpt')
        if a is None or a['model'] is None: self.skipTest('no gru_kpt weight file')
        PM.POSP.update(key='gru_kpt', model=a['model'], gen=PM.POSP['gen'] + 1, err=0, used=0, show=False)
        DF.KP['cur'] = DF.KP['prev'] = None

    def tearDown(self):
        PM.POSP.update(key='kalman', model=None, gen=PM.POSP['gen'] + 1, err=0, used=0, show=False); PM.POSP['dir'] = self.dir0
        PM.POSP['avail'].pop('gru_kpt', None)

    def _lost_boxes(self, h, gap, vis=False):
        """a person standing still, box height h, box bottom gap px above the picture bottom (canvas row 496), keypoints not visible
        (as our camera's detector gives for most people) or all 17 visible on a head-to-hip box (vis): 20 seen frames, then 15 lost frames
        -> lost boxes of steps 2..15"""
        w = 0.4 * h; cy = PM.KP_ROW1 - gap - h / 2; box = np.array([320., cy, w, h])
        t = SimpleNamespace(state=TRACKED, cls=0, track_id=7, pp_inn=np.zeros(4)); kf = FakeKF()
        mean = np.r_[box, np.zeros(4)][None]; cov = np.zeros((1, 8, 8)); kp = np.zeros((1, 17, 3), np.float32)
        if vis: kp[0, :, 0] = 320. + KP_REL[:, 0] * h; kp[0, :, 1] = cy + KP_REL[:, 1] * h; kp[0, :, 2] = 0.9
        DF.KP['prev'] = dict(boxes=np.array([[320. - w / 2, cy - h / 2, 320. + w / 2, cy + h / 2]], np.float32), kpts=kp)
        for _ in range(20): mean, cov = PM.pos_predict(kf, [t], mean, cov)
        t.state = 2; out = []
        for _ in range(14): mean, cov = PM.pos_predict(kf, [t], mean, cov); out.append(mean[0, :4].copy())
        self.assertEqual(PM.POSP['err'], 0)
        return box, np.array(out)

    def test_lost_box_does_not_depend_on_the_distance_to_the_picture_bottom(self):
        # the same still person near the bottom and 250 px above it must get the same lost boxes (11-15 file: h 44 at 60 px 0.71x, at 250 px 0.97x)
        for h, gap, vis in ((30., 60., False), (44., 60., False), (64., 40., False), (120., 20., False), (64., 40., True), (120., 20., True)):
            box, q = self._lost_boxes(h, gap, vis); box2, q2 = self._lost_boxes(h, 250., vis)
            with self.subTest(h=h, gap=gap, vis=vis):
                self.assertLess(float(np.abs(q[:, 3] / box[3] - q2[:, 3] / box2[3]).max()), 0.03)
                self.assertLess(float(np.abs((q[:, 1] - box[1]) - (q2[:, 1] - box2[1])).max() / h), 0.03)

    def test_small_still_person_near_the_bottom_keeps_its_box(self):
        # people < 50 px: our camera's detector gives them almost no visible keypoint (11-16: 10 of 26,528 outdoor detections have one)
        # 64 px and more without a visible keypoint: test_big_still_person_without_keypoints_gets_the_kalman_box
        for h, gap in ((30., 60.), (44., 60.)):
            box, q = self._lost_boxes(h, gap)
            with self.subTest(h=h, gap=gap):
                self.assertLess(float(np.abs(q[:, 3] / box[3] - 1).max()), 0.10)        # height within 10 %
                self.assertLess(float(np.abs(q[:, 1] - box[1]).max() / box[3]), 0.10)   # centre within 0.1 box heights


    def test_big_still_person_without_keypoints_gets_the_kalman_box(self):
        # 11-16 review: the file itself draws a still 90-200 px person with no visible keypoint at 0.67-0.70x and lifts it (-0.18..-0.50 box
        # heights at step 15), near the bottom or not: its training targets ask for a head-to-hip box (one per-height lookup) although the detector
        # draws the whole body when it gives no keypoint. The server gives such a loss the plain Kalman path (position_memory.KP_NOKPT_H)
        for h, gap in ((64., 250.), (90., 250.), (120., 20.), (200., 250.)):
            used = PM.POSP['used']; box, q = self._lost_boxes(h, gap)
            with self.subTest(h=h, gap=gap):
                np.testing.assert_allclose(q, np.tile(box, (len(q), 1)), rtol=0, atol=1e-9)   # still person: the Kalman box = the last box
                self.assertEqual(PM.POSP['used'], used)                                          # the file never corrected it
        box, q = self._lost_boxes(120., 250., vis=True); self.assertGreater(PM.POSP['used'], 0)    # with keypoints the file still runs


if __name__ == '__main__':
    unittest.main()
