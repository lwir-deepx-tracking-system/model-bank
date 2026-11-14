"""11-14 check of the KeepID growing search range (relabel_online.search_range, OnlineRelabel.pred_gate).
A person walks right at 5 px a frame (height 100), is lost for 40 frames, then a look-alike comes back either ahead on the walking line
(predicted place) or the same distance behind. The original gate (3.5 heights + 0.1 height a frame around the last seen place) takes both;
the predicted range takes only the one ahead. Run from tracking_server/: python -m unittest src.reid.test_relink_range"""
import unittest
import numpy as np
from . import relabel_online as RO


def run(pred_gate, back_x):
    rl = RO.OnlineRelabel(budget=False); rl.pred_gate = pred_gate; f = np.array([1.0, 0.0])
    embed = lambda ks: {k: f for k in ks}
    x = 100.0
    for t in range(10):   # seen: tracker id 1
        b = [x, 100, x + 40, 200]; rl.step([b + [1, 0.9, 0, 0]], {0: b}, embed); x += 5
    for t in range(40):   # lost
        rl.step([], {}, embed); x += 5
    b = [back_x, 100, back_x + 40, 200]   # comes back as tracker id 2
    out = rl.step([b + [2, 0.9, 0, 0]], {0: b}, embed)
    return int(out[0][4])


class RelinkRangeTests(unittest.TestCase):
    def test_range_moves_and_grows(self):
        cx, cy, r = RO.search_range([(0, 100., 50.), (5, 125., 50.)], 10, 100.)
        self.assertAlmostEqual(cx, 175.); self.assertAlmostEqual(cy, 50.); self.assertAlmostEqual(r, 100. * (0.5 + 0.05 * 10))
        _, _, r_max = RO.search_range([(0, 0., 0.)], 1000, 100.); self.assertAlmostEqual(r_max, 400.)

    def test_ahead_relinked_behind_not(self):
        last_x = 100 + 5 * 9; ahead = last_x + 5 * 41; behind = last_x - 5 * 41
        self.assertEqual(run(False, ahead), 1); self.assertEqual(run(False, behind), 1)   # original gate: both
        self.assertEqual(run(True, ahead), 1); self.assertEqual(run(True, behind), 2)     # predicted range: only ahead


if __name__ == '__main__':
    unittest.main()
