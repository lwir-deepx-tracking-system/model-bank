"""11-20 cost_first (tracker_common._cost_first, BoT-SORT page switch): a pair is made only when it lowers the total cost.

Synthetic two-person scene on the real Python BoT-SORT (plain Kalman): A stands half behind B, A is hidden for 4 frames (its track is lost),
then a third box C appears on B's far side. B's track can reach C (cost 0.85 <= 0.9) and A's lost track can reach B's box (0.59): the old rule
makes both pairs, so A's number goes onto B and B's number onto C. Cost-first keeps B on B (0.15 + 0.45 + 0.45 < 0.85 + 0.59).
The 'old' assertions document the steal the switch removes."""
import unittest

import numpy as np

from ..common import settings
from .botsort import BotSortTracker
from .position_memory import POSP
from .tracker_common import linear_assignment


def b(cx, score, cy=250.0, w=100.0, h=200.0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, 0]


def scene(cost_first):
    tr = BotSortTracker('bytetrack', n_init=2, class_gate=False, radius=2.0); tr.cost_first = cost_first
    ids = {}
    for f in range(12):
        if f < 8: arr = [b(260.0, 0.9), b(300.0, 0.9)]          # A (x 260) half behind B (x 300)
        elif f < 11: arr = [b(300.0, 0.9)]                       # A hidden: only B
        else: arr = [b(297.0, 0.9), b(370.0, 0.85)]              # B moved 3 px; C appears on B's far side
        rows = tr.update(np.asarray(arr, np.float32), None, None)
        if f == 0: ids = {('A' if r[0] < 250 else 'B'): int(r[4]) for r in rows}
    return ids, {('B box', 'C box')[int(r[7])]: int(r[4]) for r in rows}, tr   # by detection row of the last frame


class CostFirstTests(unittest.TestCase):
    def setUp(self):
        self._posp = dict(POSP); POSP.update(key='kalman', model=None, show=False); self._edge = settings.EDGE_BIRTH; settings.EDGE_BIRTH = 0.0

    def tearDown(self):
        POSP.clear(); POSP.update(self._posp); settings.EDGE_BIRTH = self._edge

    def test_assignment_prefers_lower_total_cost(self):
        cost = np.array([[0.15, 0.85], [0.60, 1.0]])            # rows: B's track, A's lost track; columns: B's box, a third box
        self.assertEqual(sorted(linear_assignment(cost, 0.9)[0]), [(0, 1), (1, 0)])   # old: two pairs, A's number on B's box
        self.assertEqual(linear_assignment(cost, 0.9, cost_first=True)[0], [(0, 0)])
        self.assertEqual(linear_assignment(cost, 0.8, cost_first=True)[0], linear_assignment(cost, 0.8)[0])

    def test_same_result_when_no_pair_competes(self):
        rng = np.random.default_rng(3)
        for _ in range(300):
            n = int(rng.integers(1, 7)); c = np.full((n, n), 1.0); d = rng.uniform(0, 0.5, n); c[np.arange(n), rng.permutation(n)] = d
            self.assertEqual(sorted(linear_assignment(c, 0.9)[0]), sorted(linear_assignment(c, 0.9, cost_first=True)[0]))

    def test_hidden_person_number_not_moved_onto_front_person(self):
        ids, last, tr = scene(False)
        self.assertEqual(last['B box'], ids['A'], 'old rule: the box of B shows the hidden person A\'s number')
        self.assertEqual(last['C box'], ids['B'])
        ids, last, tr = scene(True)
        self.assertEqual(last['B box'], ids['B'])
        self.assertNotIn(ids['A'], last.values())
        self.assertGreater(tr.cost_first_changed, 0)

    def test_switch_off_counts_nothing(self):
        _, _, tr = scene(False)
        self.assertEqual(tr.cost_first_changed, 0)


if __name__ == '__main__':
    unittest.main()
