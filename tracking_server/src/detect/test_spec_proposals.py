"""Spatial-spectrum people proposals: feature shape, window skipping around reported people, background worker, drawing."""
import time
import unittest

import numpy as np

from src.detect import spec_proposals as sp


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    f = (rng.normal(20000, 30, (480, 640))).astype(np.float32)
    yy, xx = np.mgrid[0:480, 0:640]
    f += 900 * np.exp(-(((xx - 400) / 40.0) ** 2 + ((yy - 250) / 90.0) ** 2))   # smooth warm blob (person-like)
    f[60:120, 60:200] += 700   # sharp-edged warm rectangle (monitor-like)
    return f.astype(np.uint16)


class SpecProposalTest(unittest.TestCase):
    def test_features_shape_and_finite(self):
        F = sp.features(_scene(), [(0, 0, 64, 64), (100, 100, 196, 196), (300, 200, 460, 360)])
        self.assertEqual(F.shape, (3, 129))
        self.assertTrue(np.isfinite(F).all())

    def test_resize_block_mean_matches_reshape(self):
        c = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
        self.assertTrue(np.allclose(sp.resize64(c), c.reshape(64, 2, 64, 2).mean((1, 3))))

    def test_windows_cover_frame(self):
        w = sp.windows(480, 640, (64, 160), 32)
        self.assertTrue(all(0 <= x0 and 0 <= y0 and x1 <= 640 and y1 <= 480 and x1 - x0 == y1 - y0 for x0, y0, x1, y1 in w))

    def test_model_loads_and_skips_people(self):
        m = sp.Model()
        self.assertEqual(len(m.w), 129)
        m.thr = -1e9   # every window passes: the result shows which windows were considered
        out = m.propose(_scene(), [(320, 120, 480, 400)])
        self.assertTrue(0 < len(out) <= m.top)
        for x0, y0, x1, y1, s in out:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            self.assertFalse(320 <= cx <= 480 and 120 <= cy <= 400)

    def test_one_box_per_place(self):
        m = sp.Model()
        m.thr = -1e9; m.top = 10
        out = m.propose(_scene(), [])
        for i, a in enumerate(out):
            for b in out[:i]:
                self.assertLess(sp._inside(a[:4], b[:4]), 0.3)   # no window inside / mostly over another one
                self.assertFalse(sp._same_column(a[:4], b[:4]))   # no head window right above a body window
        self.assertTrue(sp._same_column((100, 40, 196, 136), (90, 140, 250, 300)))
        self.assertFalse(sp._same_column((100, 40, 196, 136), (300, 140, 460, 300)))
        self.assertEqual(sp._inside((10, 10, 74, 74), (0, 0, 160, 160)), 1.0)

    def test_worker_fills_prop(self):
        p = sp.Proposer()
        p.model.thr = -1e9
        runs = sp.PROP['runs']
        p.submit(1.0, _scene(), [])
        for _ in range(100):
            if sp.PROP['runs'] > runs:
                break
            time.sleep(0.05)
        self.assertGreater(sp.PROP['runs'], runs)
        self.assertEqual(sp.PROP['frame_t'], 1.0)
        self.assertTrue(sp.PROP['boxes'])
        self.assertEqual(set(sp.state()), {'show', 'ms', 'runs', 'n', 'errors', 'last_error'})

    def test_preview_draws_proposals(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('no OpenCV')
        from src.web import preview_worker as pw
        base = dict(frame_t=0.0, gray=np.zeros((480, 640), np.uint8), scale=1.0, boxes=(), gt_boxes=(), predictions=(),
                    top_text='', replay=False, gimbal_text='', gimbal_color=(0, 0, 0))
        a = pw.render_preview(cv2, pw.PreviewFrame(**base))
        b = pw.render_preview(cv2, pw.PreviewFrame(**base, proposals=(((100, 100), (200, 200), 2.0),)))
        self.assertFalse(np.array_equal(np.asarray(a), np.asarray(b)))


if __name__ == '__main__':
    unittest.main()
