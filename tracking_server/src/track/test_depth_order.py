"""11-20 depth_order (tracker_common.depth_order_blocks, BoT-SORT page switch, default off): when two seen person tracks overlap strongly
and both fit the same detection best, the one behind (bottom edge higher in the picture, else shorter) may not take it.

Synthetic scene on the real Python BoT-SORT (plain Kalman): A stands behind (shorter box, bottom higher), B stands half in front of A; then
only one box is detected, B's box drawn a little toward A, which A's prediction fits better: the old rule gives A's number to B's box while A
is still seen. With depth_order B (in front) keeps its own number and A keeps its number on the lost path.
Not covered (measured, HANDOFF 11-20): a frame later the hidden person's LOST track can still take the box when B's prediction runs on."""
import unittest

import numpy as np

from ..common import settings
from ..common.shared_values import LOST, TRACKED
from .botsort import BotSortTracker
from .position_memory import POSP
from .tracker_common import Track, depth_order_blocks


def box(cx, cy, w, h, score=0.9):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, 0]


def scene(depth_order, frames=12):
    tr = BotSortTracker('bytetrack', n_init=2, class_gate=False, radius=2.0); tr.depth_order = depth_order
    ids, rows = {}, []
    for f in range(frames):
        if f < 6: arr = [box(300.0, 230.0, 90.0, 180.0), box(330.0, 250.0, 100.0, 200.0)]   # A behind (bottom 320), B in front (bottom 350)
        else: arr = [box(305.0, 250.0, 100.0, 200.0)]                                         # only B's box, drawn toward A
        rows = tr.update(np.asarray(arr, np.float32), None, None)
        if f == 0: ids = {('A' if r[0] < 270 else 'B'): int(r[4]) for r in rows}
    return ids, rows, tr


def track(x1, y1, x2, y2, state=TRACKED, active=True):
    t = Track(np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]), 0.9, 0, 0)
    t.state = state; t.is_activated = active
    return t


class DepthOrderTests(unittest.TestCase):
    def setUp(self):
        self._posp = dict(POSP); POSP.update(key='kalman', model=None, show=False); self._edge = settings.EDGE_BIRTH; settings.EDGE_BIRTH = 0.0

    def tearDown(self):
        POSP.clear(); POSP.update(self._posp); settings.EDGE_BIRTH = self._edge

    def test_blocks_only_the_one_behind_on_the_shared_best_detection(self):
        pool = [track(255, 140, 345, 320), track(250, 150, 350, 350)]   # 0 behind (bottom 320), 1 in front (bottom 350)
        self.assertEqual(depth_order_blocks(np.array([[0.3, 0.95], [0.4, 0.6]]), pool, 0.9), [(0, 0)])   # detection 0 is the best of both
        self.assertEqual(depth_order_blocks(np.array([[0.3, 0.95], [0.4, 0.2]]), pool, 0.9), [])         # the front one fits detection 1 best
        self.assertEqual(depth_order_blocks(np.array([[0.3, 0.95], [0.95, 0.96]]), pool, 0.9), [])       # over the limit for the front one

    def test_no_action_without_overlap_same_depth_or_lost(self):
        cost = np.array([[0.3, 0.6], [0.3, 0.6]])
        self.assertEqual(depth_order_blocks(cost, [track(0, 0, 100, 200), track(300, 0, 400, 220)], 0.9), [])      # apart
        self.assertEqual(depth_order_blocks(cost, [track(0, 0, 100, 200), track(10, 1, 110, 201)], 0.9), [])       # same bottom and height
        self.assertEqual(depth_order_blocks(cost, [track(255, 140, 345, 320), track(250, 150, 350, 350, state=LOST)], 0.9), [])   # lost: not in the rule
        self.assertEqual(depth_order_blocks(cost, [track(255, 140, 345, 320), track(250, 150, 350, 350, active=False)], 0.9), [])

    def test_hidden_person_number_not_moved_onto_front_person(self):
        ids, rows, tr = scene(False)
        self.assertEqual([int(r[4]) for r in rows], [ids['A']], 'old rule: the box of B shows the hidden person A\'s number')
        ids, rows, tr = scene(True)
        self.assertEqual([int(r[4]) for r in rows], [ids['B']])
        self.assertIn(ids['A'], [t.track_id for t in tr.lost])
        self.assertGreater(tr.depth_order_acted, 0)

    def test_switch_off_counts_nothing(self):
        _, _, tr = scene(False)
        self.assertEqual(tr.depth_order_acted, 0)

    def test_same_rows_when_nobody_overlaps(self):
        out = []
        for on in (False, True):
            tr = BotSortTracker('bytetrack', n_init=2, class_gate=False, radius=2.0); tr.depth_order = on; got = []
            rng = np.random.default_rng(5)
            for f in range(40):
                arr = [box(60.0 + 130.0 * k + rng.normal(0, 2), 200.0 + 30 * (k % 2) + rng.normal(0, 2), 60.0, 120.0 + 20 * k, 0.8) for k in range(4)]
                got.append([[round(float(v), 6) for v in r] for r in tr.update(np.asarray(arr, np.float32), None, None)])
            out.append(got)
        self.assertEqual(out[0], out[1])


if __name__ == '__main__':
    unittest.main()
