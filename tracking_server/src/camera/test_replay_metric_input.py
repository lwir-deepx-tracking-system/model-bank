"""Replay scoring keeps the same detector candidates as before 11-15, with less work per frame.

Eval._metric_input runs in the frame loop for every frame of a labelled recording. It tested each detector candidate
with its own numpy call (the pose models give 100-300 candidates a frame: several ms a frame on the board). It now
tests all of them at once and reads the candidate rows as Python numbers in one step. The rule and every number must be
unchanged: checked here against the replaced code.

Run from the tracking_server folder: python3 -m unittest src.camera.test_replay_metric_input
"""
import random
import time
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from ..track import gt_score
from ..track.stabilizer import stab_people_real
from . import replay_eval


def reference(gt, tracks, kept, detections, sc):
    """Eval._metric_input as it was up to 11-15."""
    people, ignore = gt_score.split_gt(gt, True)
    targets = [(int(gid), *box) for gid, box in people]
    kept_tracks = [tracks[j] for j in kept]
    if detections is None:
        return targets, kept_tracks, None
    det = [(0, float(r[0]), float(r[1]) - 16, float(r[2]), float(r[3]) - 16, float(r[4]))
           for r in detections if int(r[5]) == 0]
    if sc is not None:
        det = stab_people_real(det, sc, float(APP.LIVE_W), float(APP.LIVE_H))
    valid_overlap = gt_score.iou_matrix([b for _, b in people], [d[1:5] for d in det])
    det_out = [(d[5], *d[1:5]) for j, d in enumerate(det)
               if (valid_overlap[:, j] >= gt_score.IOU_MIN).any()
               or not any(gt_score.ioa(d[1:5], b) >= gt_score.IOA_MIN for b in ignore)]
    return targets, kept_tracks, det_out


def frame(seed):
    """(gt rows, track rows, kept indexes, candidates as the detector gives them, picture shift or None)"""
    r = random.Random(seed)
    gt = []
    for i in range(r.choice([0, 1, 3, 9])):
        x, y = r.uniform(-10, 600), r.uniform(0, 380)
        gt.append((i + 1, x, y, x + r.uniform(8, 60), y + r.uniform(30, 120), r.choice([0, 0, 0, 1, 2])))   # 1 = ignore region, 2 = another class
    if seed % 3 == 0:
        gt.append((50, 0.0, 100.0, 12.0, 200.0, 0))                                                         # a sliver at the left edge: ignored
    tracks = [(100 + i, g[1] + 1, g[2] + 1, g[3] + 1, g[4] + 1) for i, g in enumerate(gt)]
    kept = [j for j in range(len(tracks)) if r.random() < 0.8]
    rows = []
    for g in gt:                                                                                            # candidates on labelled boxes
        for _ in range(r.randint(0, 2)):
            rows.append([g[1] + r.uniform(-6, 6), g[2] + 16 + r.uniform(-6, 6), g[3] + r.uniform(-6, 6), g[4] + 16 + r.uniform(-6, 6),
                         r.uniform(0.1, 0.9), r.choice([0, 0, 0, 1])])
        if g[5] == 1:                                                                                       # and inside ignore regions
            rows.append([g[1] + 2, g[2] + 18, g[3] - 2, g[4] + 14, r.uniform(0.1, 0.9), 0])
    for _ in range(r.choice([0, 5, 120, 280])):                                                             # clutter
        x, y = r.uniform(0, 600), r.uniform(16, 420)
        rows.append([x, y, x + r.uniform(5, 40), y + r.uniform(10, 80), r.uniform(0.001, 0.2), r.choice([0, 0, 0, 2])])
    r.shuffle(rows)
    detections = np.asarray(rows, np.float32).reshape(-1, 6)
    return gt, tracks, kept, detections, (None if seed % 4 else (r.uniform(-9, 9), r.uniform(-9, 9)))


class SameCandidates(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(vars(APP), LIVE_W=640, LIVE_H=480)
        patcher.start(); self.addCleanup(patcher.stop)
        self.ev = replay_eval.Eval.__new__(replay_eval.Eval)

    def test_same_rows_as_the_replaced_code(self):
        seen = 0
        for seed in range(60):
            gt, tracks, kept, detections, sc = frame(seed)
            with self.subTest(seed=seed):
                got = self.ev._metric_input(gt, tracks, kept, detections, sc)
                want = reference(gt, tracks, kept, detections, sc)
                self.assertEqual(got, want)
                self.assertEqual([type(v) for row in got[2] for v in row], [type(v) for row in want[2] for v in row])
                seen += len(want[2])
        self.assertGreater(seen, 3000)

    def test_dropped_candidates_are_the_ones_inside_an_ignore_region(self):
        gt = [(1, 100.0, 100.0, 140.0, 220.0, 0), (2, 300.0, 100.0, 380.0, 260.0, 1)]
        detections = np.array([[101, 117, 141, 237, 0.9, 0],      # on the person
                               [310, 130, 360, 250, 0.8, 0],      # inside the ignore region
                               [500, 116, 540, 216, 0.7, 0],      # elsewhere
                               [101, 117, 141, 237, 0.6, 1]], np.float32)   # another class
        targets, kept_tracks, candidates = self.ev._metric_input(gt, [], [], detections, None)
        self.assertEqual(targets, [(1, 100.0, 100.0, 140.0, 220.0)])
        self.assertEqual([round(c[0], 1) for c in candidates], [0.9, 0.7])
        self.assertEqual(self.ev._metric_input(gt, [], [], detections, None), reference(gt, [], [], detections, None))

    def test_lists_and_no_candidates(self):
        gt = [(1, 100.0, 100.0, 140.0, 220.0, 0)]
        rows = [[101.0, 117.0, 141.0, 237.0, 0.9, 0]]
        self.assertEqual(self.ev._metric_input(gt, [], [], rows, None), reference(gt, [], [], rows, None))
        self.assertEqual(self.ev._metric_input(gt, [], [], np.zeros((0, 6), np.float32), None), ([(1, 100.0, 100.0, 140.0, 220.0)], [], []))
        self.assertEqual(self.ev._metric_input(gt, [], [], None, None)[2], None)
        self.assertEqual(self.ev._metric_input([], [], [], rows, None), reference([], [], [], rows, None))

    def test_a_crowded_frame_is_cheap(self):
        r = random.Random(7)
        gt = [(i + 1, 60.0 * i + 20, 100.0, 60.0 * i + 55, 230.0, 0) for i in range(6)]
        rows = [[x, y, x + 30, y + 90, r.uniform(0.001, 0.2), 0] for x, y in ((r.uniform(0, 600), r.uniform(16, 400)) for _ in range(300))]
        detections = np.asarray(rows, np.float32)

        def seconds(work):
            best = float('inf')
            for _ in range(5):
                started = time.perf_counter()
                for _ in range(20):
                    work(gt, [], [], detections, None)
                best = min(best, time.perf_counter() - started)
            return best

        self.assertEqual(self.ev._metric_input(gt, [], [], detections, None), reference(gt, [], [], detections, None))
        self.assertLess(seconds(self.ev._metric_input), seconds(reference) * 0.7)


if __name__ == '__main__':
    unittest.main()
