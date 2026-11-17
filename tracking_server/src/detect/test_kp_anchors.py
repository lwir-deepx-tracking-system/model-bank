"""The all-at-once keypoint anchor lookup gives the anchors of the one-row-at-a-time loop (2026-11-15)."""
import unittest

import numpy as np

from . import detect_frame as DF
from ..common.fast_path import FAST, STAT


def scene(rng, n=6300, people=60, ties=True):
    """Decoded candidates [1,20,N] with low scores everywhere and `people` anchors above the score limit, and the NMS rows copied from them."""
    d = np.zeros((1, 20, n), np.float32)
    d[0, 0] = rng.uniform(0, 640, n); d[0, 1] = rng.uniform(0, 480, n); d[0, 2] = rng.uniform(4, 200, n); d[0, 3] = rng.uniform(8, 400, n)
    d[0, 4] = rng.uniform(0, 0.0009, n).astype(np.float32)
    hot = rng.choice(n, people, replace=False); d[0, 4, hot] = rng.uniform(0.001, 0.95, people).astype(np.float32)
    if ties and people >= 4:   # equal scores on several anchors: the box corner decides; two of them are the same box too (the first anchor wins)
        d[0, 4, hot[1]] = d[0, 4, hot[0]]; d[0, 4, hot[3]] = d[0, 4, hot[2]]; d[0, :4, hot[3]] = d[0, :4, hot[2]]
    rows = np.zeros((people, 6), np.float32)
    rows[:, 0] = d[0, 0, hot] - d[0, 2, hot] * .5; rows[:, 1] = d[0, 1, hot] - d[0, 3, hot] * .5
    rows[:, 2] = d[0, 0, hot] + d[0, 2, hot] * .5; rows[:, 3] = d[0, 1, hot] + d[0, 3, hot] * .5; rows[:, 4] = d[0, 4, hot]
    return d, rows[rng.permutation(people)]


class KeypointAnchorTests(unittest.TestCase):
    def test_same_anchors_as_the_loop(self):
        rng = np.random.default_rng(7)
        for people in (1, 2, 5, 60, 110, 300):
            for _ in range(8):
                d, rows = scene(rng, people=people)
                np.testing.assert_array_equal(DF._kp_anchors(d[0], rows), DF._kp_anchors_loop(d[0], rows))

    def test_rows_without_a_matching_anchor_and_bad_numbers_go_through_the_loop(self):
        rng = np.random.default_rng(8); d, rows = scene(rng, people=40)
        rows[3, 4] += 0.01; rows[7, 4] = 0.99999   # scores no anchor has: the loop looks at every anchor
        rows[11, 0] = np.nan; rows[12, 1] = np.inf
        before = STAT['kpts_loop_rows']
        np.testing.assert_array_equal(DF._kp_anchors(d[0], rows), DF._kp_anchors_loop(d[0], rows))
        self.assertGreaterEqual(STAT['kpts_loop_rows'] - before, 4)
        d[0, 4, 5] = np.nan; d[0, 0, 9] = np.nan
        np.testing.assert_array_equal(DF._kp_anchors(d[0], rows), DF._kp_anchors_loop(d[0], rows))
        self.assertEqual(len(DF._kp_anchors(d[0], rows[:0])), 0)

    def test_attach_kpts_gives_the_same_keypoints_either_way(self):
        rng = np.random.default_rng(9); d, rows = scene(rng, people=90)
        rows[::7, 5] = 2   # other classes are left out
        kraw = rng.normal(0, 1, (1, 51, d.shape[2])).astype(np.float32); got = {}
        old = dict(FAST)
        try:
            for fast in (False, True):
                FAST['kpts'] = fast; DF.KP['cur'] = None
                DF.attach_kpts(d, kraw, rows, 480, 640, 16.0); got[fast] = DF.KP['cur']
            FAST['kpts_check'] = True; n0, d0 = STAT['kpts_checked'], STAT['kpts_differ']
            DF.attach_kpts(d, kraw, rows, 480, 640, 16.0)
            self.assertEqual((STAT['kpts_checked'] - n0, STAT['kpts_differ'] - d0), (1, 0))
        finally:
            FAST.update(old)
        self.assertEqual(len(got[True]['boxes']), int((rows[:, 5] == 0).sum()))
        np.testing.assert_array_equal(got[True]['boxes'], got[False]['boxes'])
        np.testing.assert_array_equal(got[True]['kpts'], got[False]['kpts'])


if __name__ == '__main__':
    unittest.main()
