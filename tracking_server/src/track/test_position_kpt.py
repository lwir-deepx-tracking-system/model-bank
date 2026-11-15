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


class ZeroCell:
    kpt = True
    def init(self): return np.zeros(3)
    def step(self, x, st):
        assert x.shape == (PM.KP_NF,) and np.isfinite(x).all(); return np.zeros(4), st


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
        # 11-16: before, the first LOST call applied ITS warp to steps 1 and 2 and never the warp of step 1's frame
        shift = np.array([[1., 0., 10.], [0., 1., 0.]])
        post, c = self._seen_then_lost(shift, None)   # the camera moved 10 px in step 1's frame, stood still in step 2's
        np.testing.assert_allclose(c, post[:2] + [10., 0.] + 2 * post[4:6])
        post, c = self._seen_then_lost(None, shift)   # still in step 1's frame, moved in step 2's: the shift once, not twice
        np.testing.assert_allclose(c, post[:2] + [10., 0.] + 2 * post[4:6])


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


class ShippedGruNearBottomTests(unittest.TestCase):
    """11-16: the shipped GRU file must not shrink and lift a lost person because its box is near the picture bottom. The 11-15 file did
    (on our camera the lost box at step 2 was 0.85x and 0.16 box heights higher, 3x the Kalman error): it was trained with DanceTrack's
    letterbox measured as the device picture (no person closer than 60 px to the 'bottom') and with targets made from gt boxes cut by the
    picture edge (a smaller, higher box near the bottom)."""
    def setUp(self):
        self.dir0 = PM.POSP['dir']; PM.POSP['dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'track')
        PM.POSP['avail'].pop('gru_kpt', None); PM.pos_scan(); a = PM.POSP['avail'].get('gru_kpt')
        if a is None or a['model'] is None: self.skipTest('no gru_kpt weight file')
        PM.POSP.update(key='gru_kpt', model=a['model'], gen=PM.POSP['gen'] + 1, err=0, used=0, show=False)
        DF.KP['cur'] = DF.KP['prev'] = None

    def tearDown(self):
        PM.POSP.update(key='kalman', model=None, gen=PM.POSP['gen'] + 1, err=0, used=0, show=False); PM.POSP['dir'] = self.dir0
        PM.POSP['avail'].pop('gru_kpt', None)

    def _lost_boxes(self, h, gap):
        """a person standing still, box height h, box bottom gap px above the picture bottom (canvas row 496), keypoints not visible
        (as our camera's detector gives for most people): 20 seen frames, then 15 lost frames -> lost boxes of steps 2..15"""
        w = 0.4 * h; cy = PM.KP_ROW1 - gap - h / 2; box = np.array([320., cy, w, h])
        t = SimpleNamespace(state=TRACKED, cls=0, track_id=7, pp_inn=np.zeros(4)); kf = FakeKF()
        mean = np.r_[box, np.zeros(4)][None]; cov = np.zeros((1, 8, 8))
        DF.KP['prev'] = dict(boxes=np.array([[320. - w / 2, cy - h / 2, 320. + w / 2, cy + h / 2]], np.float32), kpts=np.zeros((1, 17, 3), np.float32))
        for _ in range(20): mean, cov = PM.pos_predict(kf, [t], mean, cov)
        t.state = 2; out = []
        for _ in range(14): mean, cov = PM.pos_predict(kf, [t], mean, cov); out.append(mean[0, :4].copy())
        self.assertEqual(PM.POSP['err'], 0)
        return box, np.array(out)

    def test_lost_box_does_not_depend_on_the_distance_to_the_picture_bottom(self):
        # the same still person near the bottom and 250 px above it must get the same lost boxes (11-15 file: h 44 at 60 px 0.71x, at 250 px 0.97x)
        for h, gap in ((30., 60.), (44., 60.), (64., 40.), (120., 20.)):
            box, q = self._lost_boxes(h, gap); box2, q2 = self._lost_boxes(h, 250.)
            with self.subTest(h=h, gap=gap):
                self.assertLess(float(np.abs(q[:, 3] / box[3] - q2[:, 3] / box2[3]).max()), 0.03)
                self.assertLess(float(np.abs((q[:, 1] - box[1]) - (q2[:, 1] - box2[1])).max() / h), 0.03)

    def test_small_still_person_near_the_bottom_keeps_its_box(self):
        # people < 50 px: our camera's detector gives them almost no visible keypoint (11-16: 10 of 26,528 outdoor detections have one)
        # NOT asserted (known, 11-16 file): a still person of 64-200 px with NO visible keypoint is drawn 0.68-0.70x and up to 0.5 h higher
        # when lost (public training data has such people almost only when occluded); see models/README.md, GRU position memory
        for h, gap in ((30., 60.), (44., 60.)):
            box, q = self._lost_boxes(h, gap)
            with self.subTest(h=h, gap=gap):
                self.assertLess(float(np.abs(q[:, 3] / box[3] - 1).max()), 0.10)        # height within 10 %
                self.assertLess(float(np.abs(q[:, 1] - box[1]).max() / box[3]), 0.10)   # centre within 0.1 box heights


if __name__ == '__main__':
    unittest.main()
