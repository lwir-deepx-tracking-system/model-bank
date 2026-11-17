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

    def test_steady_and_mark(self):
        a, b = (100, 100, 196, 196, 4.0), (400, 50, 560, 210, 4.2)
        self.assertEqual(sp.steady([], [a]), [])   # seen once: not shown yet
        self.assertEqual(sp.steady([[a]], [(110, 104, 206, 200, 4.1)]), [(110, 104, 206, 200, 4.1)])   # again, moved a little
        self.assertEqual(sp.steady([[a], []], [b]), [])
        m = sp.mark([a, b], [(90, 60, 220, 400)])
        self.assertEqual([x[5] for x in m], [True, False])   # on the detected person / missed

    def test_worker_fills_prop(self):
        p = sp.Proposer()
        p.model.thr = -1e9
        runs = sp.PROP['runs']
        for t in (1.0, 2.0):   # a place needs two passes
            p.submit(t, _scene(), [])
            time.sleep(0.05)
        for _ in range(100):
            if sp.PROP['runs'] > runs + 1:
                break
            time.sleep(0.05)
        self.assertGreater(sp.PROP['runs'], runs)
        self.assertEqual(sp.PROP['frame_t'], 2.0)
        self.assertTrue(sp.PROP['boxes'])
        self.assertEqual(set(sp.state()), {'show', 'ms', 'runs', 'n', 'n_missed', 'missed_total', 'errors', 'last_error', 'log_on', 'log_frames', 'fuse', 'fuse_boosted', 'fuse_ms'})

    def test_weak_boxes(self):
        d = [(10, 10, 60, 120, 0.2), (300, 100, 400, 300, 0.4), (305, 105, 400, 300, 0.02)]
        self.assertEqual(sp.weak_boxes(d, [(300, 100, 400, 300)]), [(10, 10, 60, 120, 0.2)])
        m = sp.Model(); m.thr = -1e9
        r = m.rescore(_scene(), [(360, 160, 440, 340, 0.1), (0, 0, 10, 10, 0.3)])
        self.assertEqual(len(r), 1); self.assertEqual(r[0][:4], (360, 160, 440, 340))
        self.assertEqual(len(m.pick_weak(_scene(), [(360, 160, 440, 340, 0.1), (365, 165, 440, 340, 0.1)])), 1)   # one per place

    def test_fuse(self):
        m = sp.Model(); m.thr = -1e9; sp._FUSE_MODEL[0] = m
        arr = np.array([[360, 176, 440, 356, 0.12, 0], [10, 26, 20, 36, 0.12, 0], [100, 116, 200, 316, 0.6, 0], [300, 116, 400, 316, 0.12, 2]], np.float32)
        sp.FUSE['mode'] = 2
        out = sp.fuse(arr, _scene(), 0, 16.0)
        self.assertEqual([round(float(v), 2) for v in out[:, 4]], [0.26, 0.12, 0.6, 0.12])   # weak person boosted; tiny / strong / other class untouched
        self.assertEqual(arr[0, 4].round(2), 0.12)   # input not changed
        sp.FUSE['mode'] = 1   # current rule: a weak box on a strong detection is left alone, conf must be >= 0.10
        arr2 = np.array([[360, 176, 440, 356, 0.12, 0], [365, 180, 440, 356, 0.6, 0], [100, 116, 200, 316, 0.08, 0]], np.float32)
        self.assertEqual([round(float(v), 2) for v in sp.fuse(arr2, _scene(), 0, 16.0)[:, 4]], [0.12, 0.6, 0.08])
        arr3 = np.array([[360, 176, 440, 356, 0.12, 0]], np.float32)
        self.assertEqual([round(float(v), 2) for v in sp.fuse(arr3, _scene(), 0, 16.0)[:, 4]], [0.26])
        sp._FUSE_MODEL[0] = None

    def test_log(self):
        import os, json, tempfile
        sp.LOG['rows'] = []
        sp.log_frame(_scene(), [(1, 2, 30, 40, 0.12)], [(5, 6, 50, 60)], [(7, 8, 70, 80, 4.5, False)])
        self.assertEqual(len(sp.LOG['rows'][0]['sig']), 12 * 16)
        path = os.path.join(tempfile.mkdtemp(), 'log.json')
        self.assertEqual(sp.log_write(path), 1)
        row = json.load(open(path))['rows'][0]
        self.assertEqual(row['dets'], [[1.0, 2.0, 30.0, 40.0, 0.12]])
        self.assertEqual(row['props'], [[7.0, 8.0, 70.0, 80.0, 4.5, False]])
        self.assertEqual(sp.LOG['rows'], [])

    def test_preview_draws_proposals(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('no OpenCV')
        from src.web import preview_worker as pw
        base = dict(frame_t=0.0, gray=np.zeros((480, 640), np.uint8), scale=1.0, boxes=(), gt_boxes=(), predictions=(),
                    top_text='', replay=False, gimbal_text='', gimbal_color=(0, 0, 0))
        a = pw.render_preview(cv2, pw.PreviewFrame(**base))
        b = pw.render_preview(cv2, pw.PreviewFrame(**base, proposals=(((100, 100), (200, 200), 2.0, False), ((300, 100), (400, 200), 4.0, True))))
        self.assertFalse(np.array_equal(np.asarray(a), np.asarray(b)))


if __name__ == '__main__':
    unittest.main()
