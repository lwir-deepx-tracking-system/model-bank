"""11-14 checks of the keypoint path: keypoint decode of NMS-kept boxes, keypoint position memory (zero output = plain Kalman path), path data for the overlay.
Run from tracking_server/:  python -m unittest src.track.test_position_kpt"""
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


if __name__ == '__main__':
    unittest.main()
