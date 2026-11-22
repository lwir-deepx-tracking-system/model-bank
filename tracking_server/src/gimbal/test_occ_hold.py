"""11-22: occlusion hold of the follow guard v2, a synthetic scene from the 12:41 event samples (id 44 box ~ (230,140,440,404), id 23 on it at IoU 0.65-0.75
for 8 s, id 44 absent from the rows for 1-2 frames about every second, features: the blended box looks like the occluder). No recording of that event exists."""
import os, unittest
import numpy as np
from . import follow_guard as fg

TARGET = np.array([1., 0., 0.]); OCC = np.array([0., 1., 0.])


def run(hold, secs=8.0, leave_at=None, occ_swaps_id=False):
    os.environ['LWIR_FOLLOW_OCC_HOLD'] = '1' if hold else '0'
    g = fg.FollowGuardV2(44); feats = {}; g.app = lambda oid: (feats.get(oid), 0)
    g.kal = None; motion = lambda a, b: np.array([[1., 0., 0.], [0., 1., 0.]])
    rng = np.random.RandomState(1); res = []; t = 0.0
    for i in range(int(secs * 30) + 60):
        t = 1.0 + i / 30.0
        covered = i >= 60 and (leave_at is None or t - 1.0 < leave_at)
        j = rng.randint(-6, 7)
        mine = [44, 230 + j, 140 + j, 440 + j, 404 + j]
        occ = [23, 235 + j, 145 + j, 445 + j, 410 + j] if covered else [23, 520, 140, 620, 400]
        people = [mine, occ]
        if covered and i % 30 in (10, 11): people = [occ]   # id 44 absent from the rows for 2 frames each second
        feats[44] = TARGET if not covered else (OCC * 0.9 + TARGET * 0.3)   # a blended box: nearer the occluder
        feats[23] = OCC
        d = g.step(t, people, motion); res.append((covered, d['mask'], d['reselect'], d['phase']))
    return g, res


class OccHold(unittest.TestCase):
    def tearDown(self): os.environ.pop('LWIR_FOLLOW_OCC_HOLD', None)

    def test_without_hold_the_covered_target_gets_masked(self):
        g, res = run(False)
        self.assertTrue(any(c and m for c, m, r, p in res))   # the guard hides it: the bridge sees no id (the 12:41 loss)

    def test_with_hold_it_is_never_masked_or_handed_to_the_occluder(self):
        g, res = run(True, secs=5.5)
        self.assertFalse(any(m for c, m, r, p in res)); self.assertFalse(any(r for c, m, r, p in res)); self.assertEqual(g.holds, 1)

    def test_hold_is_limited_and_released(self):
        g, res = run(True, secs=10.0)   # covered longer than HOLD_MAX_S: the normal rules take over again
        self.assertTrue(any(c and m for c, m, r, p in res[-120:]))
        g, res = run(True, secs=8.0, leave_at=4.0)   # the occluder leaves: back to track, not masked after
        self.assertFalse(any(m for c, m, r, p in res if not c)); self.assertEqual(res[-1][3], 'track')
        self.assertIsNone(g.hold_t0)

    def test_off_by_default(self):
        os.environ.pop('LWIR_FOLLOW_OCC_HOLD', None); self.assertFalse(fg.hold_enabled())


if __name__ == '__main__':
    unittest.main()
