"""11-20 merge_split / pre_capture (track/merge_split.py, layer after the tracker) and kpt_assoc (Python BoT-SORT). Not recommended (HANDOFF 11-20):
these tests only fix what the switches do. Rows are given as a tracker gives them: [x1, y1, x2, y2, id, conf, cls, k].
  off       switches off: rows come back unchanged, no state is kept;
  merge     A (raw 1) and B (raw 2) walk together, one box covers both, the tracker gives it to A's raw (the front person is B);
            the layer shows B's number on it (memory), holds A's, gives A's number to no third box, and at the split puts each number on
            the box whose look matches its memory;
  hold      a group older than HOLD_S releases the held number;
  judge     OnlineJudge.ms_hold: a held number is not handed to a birth;
  pre       pre_capture alone: an approaching pair whose numbers the tracker crossed is swapped back by the frozen memories;
  kpt       kpt_assoc off leaves the cost matrix as it was; on raises the cost of a pair whose torso keypoints disagree."""
import unittest
from pathlib import Path

import numpy as np

from . import merge_split as MS

rng = np.random.default_rng(7)


def _u(v): return v / np.linalg.norm(v)


F_A = _u(rng.normal(size=32)); F_B = _u(rng.normal(size=32) - 1.5 * F_A); F_C = _u(rng.normal(size=32))


def bx(cx, w=60.0, h=160.0, cy=250.0): return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class _Sw:
    def __init__(self, **kw): self.kw = kw
    def __enter__(self): self.old = dict(MS.SW); MS.SW.update(self.kw)
    def __exit__(self, *a): MS.SW.clear(); MS.SW.update(self.old)


def frame(layer, items, feats=None):
    """items: [(box, raw, feature)] -> output ids in item order"""
    dets = np.asarray([b + [0.9, 0] for b, _, _ in items], np.float32).reshape(-1, 6)
    rows = [b + [raw, 0.9, 0, k] for k, (b, raw, _) in enumerate(items)]
    fe = {k: f for k, (_, _, f) in enumerate(items) if f is not None} if feats is None else feats
    return [int(r[4]) for r in layer.step(rows, dets, fe, None, fps=30.0)]


class MergeSplitTest(unittest.TestCase):
    def test_off_rows_unchanged(self):
        self.assertFalse(MS.active())
        self.assertEqual(MS.SW, dict(pre_capture=False, merge_split=False))   # app.py runs the layer only when active()

    def _merge_scene(self, front_feat_on_box=True):
        L = MS.MergeSplit(); out = []
        for f in range(10): out.append(frame(L, [(bx(100 + 4 * f), 1, F_A), (bx(300 - 4 * f), 2, F_B)]))   # apart: clean memories
        for f in range(10, 16):   # one box over both; the tracker gives it to raw 1 (A), the person in front is B
            out.append(frame(L, [(bx(200, w=80), 1, F_B if front_feat_on_box else None)]))
            if f == 12: self.assertEqual(len(L.groups), 1)
        # split: B continues to the left, A to the right, but the tracker keeps raw 1 on the left box (B) and starts raw 3 on A
        for f in range(16, 22): out.append(frame(L, [(bx(200 - 12 * (f - 15)), 1, F_B), (bx(200 + 12 * (f - 15)), 3, F_A)]))
        return L, out

    def test_merge_holds_and_splits_by_memory(self):
        with _Sw(merge_split=True):
            L, out = self._merge_scene()
            self.assertEqual(out[12], [2])            # B's number on the one box (its look), A's held
            self.assertEqual(out[-1], [2, 1])         # left box B -> 2, right box A -> 1
            self.assertGreaterEqual(L.stats['split_app'], 1)

    def test_third_box_gets_fresh_number(self):
        with _Sw(merge_split=True):
            L = MS.MergeSplit()
            for f in range(10): frame(L, [(bx(100 + 4 * f), 1, F_A), (bx(300 - 4 * f), 2, F_B)])
            frame(L, [(bx(200, w=80), 1, F_B)]); frame(L, [(bx(200, w=80), 1, F_B)])
            self.assertEqual(len(L.groups), 1)
            ids = frame(L, [(bx(200, w=80), 1, F_B), (bx(560), 2, F_C)])   # the held number's raw re-found far away
            self.assertEqual(ids[0], 2); self.assertGreaterEqual(ids[1], MS.FRESH)

    def test_hold_limit_releases(self):
        with _Sw(merge_split=True):
            L = MS.MergeSplit()
            for f in range(10): frame(L, [(bx(100 + 4 * f), 1, F_A), (bx(300 - 4 * f), 2, F_B)])
            for f in range(int(MS.HOLD_S * 30) + 3): frame(L, [(bx(200, w=80), 1, F_B)])
            self.assertEqual(L.groups, []); self.assertEqual(L.stats['hold_expired'], 1)

    def test_pre_capture_swaps_crossed_numbers_back(self):
        with _Sw(pre_capture=True):
            L = MS.MergeSplit()
            for f in range(8): frame(L, [(bx(150 + 10 * f), 1, F_A), (bx(290 - 10 * f), 2, F_B)])   # approach, crops stored
            for f in range(4): frame(L, [(bx(225 + f), 1, F_A), (bx(230 - f), 2, F_B)])            # overlap: frozen
            ids = None
            for f in range(6): ids = frame(L, [(bx(130 - 10 * f), 1, F_B), (bx(330 + 10 * f), 2, F_A)])   # apart, tracker crossed them
            self.assertEqual(ids, [2, 1]); self.assertEqual(L.stats['pc_swaps'], 1)

    def test_judge_hold_blocks_relink(self):
        from ..reid import relink_online_judge as judge
        spec = Path(__file__).resolve().parents[2] / 'models' / 'track' / 'relink_judge_frozen.json'
        res = {}
        for hold in (None, {1}):
            J = judge.OnlineJudge(str(spec), crops=2, compact_ids=True, joint_candidates=True)
            for t in range(20):
                rows = [bx(200 + t) + [1, 0.9, 0, 0]]; P = {0: bx(200 + t)}
                J.step(rows, P, lambda ks: {k: F_A for k in ks})
            J.ms_hold = hold
            out = J.step([bx(222) + [7, 0.9, 0, 0]], {0: bx(222)}, lambda ks: {k: F_A for k in ks})
            res[hold is None] = out[0][4]
        self.assertEqual(res[True], 1)             # without the hold the birth takes number 1 back (same look, same place)
        self.assertNotEqual(res[False], 1)         # held: a new number


class KptAssocTest(unittest.TestCase):
    def test_off_and_on(self):
        from ..detect import detect_frame as DF
        from .botsort import BotSortTracker
        tr = BotSortTracker('bytetrack', n_init=2, class_gate=False)
        k = np.zeros((17, 3), np.float32); k[[5, 6, 11, 12], 2] = 1.0
        def kp(cx, flip=False):
            q = k.copy(); q[5, :2] = (cx - 20, 200); q[6, :2] = (cx + 20, 200); q[11, :2] = (cx - 15, 280); q[12, :2] = (cx + 15, 280)
            if flip: q[[5, 6, 11, 12], 1] += 60
            return q
        def step(arr, kps):
            DF.KP['prev'] = DF.KP['cur']; DF.KP['cur'] = dict(boxes=np.asarray(arr, np.float32)[:, :4], kpts=np.asarray(kps, np.float32))
            return tr.update(np.asarray(arr, np.float32), None, None)
        try:
            for f in range(3): step([bx(200) + [0.9, 0], bx(230) + [0.9, 0]], [kp(200), kp(230)])
            pool = [t for t in tr.tracked]
            from .tracker_common import Track
            dets = [Track(np.r_[200.0, 250.0, 60.0, 160.0], 0.9, 0, 0), Track(np.r_[230.0, 250.0, 60.0, 160.0], 0.9, 0, 1)]
            DF.KP['prev'] = DF.KP['cur']; DF.KP['cur'] = dict(boxes=np.asarray([bx(200), bx(230)], np.float32), kpts=np.asarray([kp(200, True), kp(230)], np.float32))
            tr.frame_id += 1   # as update() does before the first association
            d0 = tr._dists(pool, dets)
            self.assertFalse(getattr(tr, 'kpt_assoc', False))
            d1 = tr._kpt_cost(d0, pool, dets)
            self.assertTrue(np.all(d1 >= d0 - 1e-9)); self.assertTrue(np.any(d1 > d0 + 1e-6)); self.assertTrue(np.all(d1 <= max(tr.match, d0.max()) + 1e-9))
        finally:
            DF.KP['cur'] = None; DF.KP['prev'] = None


if __name__ == '__main__':
    unittest.main()
