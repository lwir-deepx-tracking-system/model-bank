"""11-20 feat_guard (relink_online_judge.GUARD): the front person's look must not be stored as the hidden person's look.

Synthetic two-person scenes fed to the real OnlineJudge with the frozen judge file (rows as a tracker would give them):
  slide      A stands still, B walks in front of A; A's row vanishes (hidden), then B's own row vanishes and A's raw number is put on B's box
             (the tracker's lost-track re-find on the front person);
  duplicate  A's raw number sits as a second box on B while B's own row is still shown;
  new birth  as 'slide', but A's number stays on B for longer than the judge's 90-frame window, then the tracker breaks and a new raw track is
             born on B: the judge must not hand A's number to B because A's memory was filled with B's look.
The 'old' assertions document what the code does with the switch off (the behaviour the guard removes)."""
import unittest
from pathlib import Path

import numpy as np

from . import relink_online_judge as judge

SPEC = Path(__file__).resolve().parents[2] / 'models' / 'track' / 'relink_judge_frozen.json'
rng = np.random.default_rng(20)
F_A = judge.unit(rng.normal(size=64)); F_B = judge.unit(rng.normal(size=64) - 2 * F_A)
H, W = 180.0, 70.0


def box(cx, cy=250.0):
    return [cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2]


def run(scene, guard, frames):
    """scene(t) -> list of (raw id, box, person 'A' | 'B'); one detection per listed row (k = row index). Returns the judge."""
    old = dict(judge.GUARD); judge.GUARD['feat_guard'] = guard
    try:
        J = judge.OnlineJudge(str(SPEC), crops=0, compact_ids=True)
        out = []
        for t in range(frames):
            rows, P, look = [], {}, {}
            for k, (raw, b, who) in enumerate(scene(t)):
                rows.append([*b, raw, 0.8, 0, k]); P[k] = list(b); look[k] = F_A if who == 'A' else F_B
            embed = lambda ks, look=look: {k: look[k] + 0.02 * rng.normal(size=64) for k in ks}
            res = J.step(rows, P, lambda ks, e=embed: {k: judge.unit(v) for k, v in e(ks).items()}, frame=t)
            out.append({int(r[4]): int(o[4]) for r, o in zip(rows, res)})
        J.out = out
        return J
    finally:
        judge.GUARD.update(old)


def gallery_cos(J, raw, f):
    g = judge.gallery(J.hist[raw], 'm1')
    return [float(o['m1'] @ f) for o in g]


def slide(t, hold_until=60):
    """raw 1 = A at x 300; raw 2 = B walks from x 520 to 330 (in front of A from t 20); A hidden from t 26; at t 30 B's raw 2 vanishes and
    raw 1 is on B's box until hold_until; after that a new raw 3 is born on B"""
    bx = max(330.0, 520.0 - 10.0 * t)
    rows = []
    if t < 26: rows.append((1, box(300.0), 'A'))
    if t < 30: rows.append((2, box(bx), 'B'))
    elif t < hold_until: rows.append((1, box(bx + 0.2 * (t - 30)), 'B'))
    else: rows.append((3, box(bx + 0.2 * (t - 30) + 30.0), 'B'))
    return rows


class FeatGuardTests(unittest.TestCase):
    def test_slide_old_code_stores_front_persons_look_under_hidden_number(self):
        J = run(slide, False, 60)
        self.assertTrue(any(c > 0.9 for c in gallery_cos(J, 1, F_B)), 'without the guard raw 1 (A) holds B crops')
        J = run(slide, True, 60)
        cs = gallery_cos(J, 1, F_B)
        self.assertTrue(cs and all(c < 0.5 for c in cs), cs)
        self.assertTrue(all(c > 0.9 for c in gallery_cos(J, 1, F_A)))
        self.assertGreater(J.guard_stats['feat_held_hidden'] + J.guard_stats['feat_held_disagree'], 0)

    def test_duplicate_box_on_front_person(self):
        def dup(t):
            rows = [(2, box(400.0), 'B')]
            if t < 20: rows.append((1, box(250.0), 'A'))
            elif t < 40: rows.append((1, box(404.0), 'B'))   # A hidden, its number a second box on B
            return rows
        J = run(dup, False, 40)
        self.assertTrue(any(c > 0.9 for c in gallery_cos(J, 1, F_B)))
        J = run(dup, True, 40)
        self.assertTrue(all(c < 0.5 for c in gallery_cos(J, 1, F_B)))
        self.assertGreater(J.guard_stats['feat_held_overlap'], 0)

    def test_new_track_born_on_front_person_does_not_take_hidden_number(self):
        frames = 140
        scene = lambda t: slide(t, hold_until=130)   # B's own raw 2 vanished at t 30: > 90 frames before the birth at t 130
        old = run(scene, False, frames); new = run(scene, True, frames)
        a_number = old.out[10][1]
        self.assertEqual(old.out[135][3], a_number, 'without the guard the birth on B takes A\'s number')
        self.assertNotEqual(new.out[135][3], a_number)

    def test_switch_off_runs_nothing(self):
        J = run(slide, False, 60)
        self.assertEqual(J.guard_stats['feat_seen'], 0)
        self.assertTrue(all('held' not in o for h in J.hist.values() for o in h))

    def test_agreement_check_ends_after_limit(self):
        """A passes behind B (overlap t 20-30), then A's look really changes (F_C, e.g. turned or cut at the edge): with no limit every later
        feature of A is held; with G_NEED_MAX the check ends and A's new look is stored again."""
        F_C = judge.unit(rng.normal(size=64) - 2 * F_B)   # nearer to B's look than A's pre-overlap memory, as the reviewer's train4 case
        def turn(t):
            bx = 600.0 if t < 15 or t > 35 else 320.0
            return [(1, box(300.0), 'A' if t < 36 else 'C'), (2, box(bx), 'B')]
        old_lim = judge.G_NEED_MAX
        try:
            def go(lim, frames=200):
                judge.G_NEED_MAX = lim
                old = dict(judge.GUARD); judge.GUARD['feat_guard'] = True
                try:
                    J = judge.OnlineJudge(str(SPEC), crops=0, compact_ids=True)
                    for t in range(frames):
                        rows, P, look = [], {}, {}
                        for k, (raw, b_, who) in enumerate(turn(t)):
                            rows.append([*b_, raw, 0.8, 0, k]); P[k] = list(b_); look[k] = dict(A=F_A, B=F_B, C=F_C)[who]
                        J.step(rows, P, lambda ks, look=look: {k: judge.unit(look[k] + 0.02 * rng.normal(size=64)) for k in ks}, frame=t)
                    return J
                finally:
                    judge.GUARD.update(old)
            J = go(None)
            self.assertFalse(any(float(o['m1'] @ F_C) > 0.9 for o in judge.gallery(J.hist[1], 'm1')), 'no limit: the new look is never stored')
            self.assertGreater(J.guard_state()['hold_now_frames'], 150)
            J = go(90)
            self.assertTrue(any(float(o['m1'] @ F_C) > 0.9 for o in judge.gallery(J.hist[1], 'm1')))
            self.assertEqual(J.guard_stats['feat_check_expired'], 1)
            self.assertLessEqual(J.guard_stats['hold_max_frames'], 15 + 90 + 10)
            self.assertEqual(J.guard_state()['held_now'], 0)
        finally:
            judge.G_NEED_MAX = old_lim

    def test_guard_state_is_pruned(self):
        """numbers that come and go (a new raw number every 20 frames, each overlapping a standing person) must not keep guard state forever"""
        def churn(t):
            raw = 10 + t // 20
            return [(1, box(300.0), 'A'), (raw, box(320.0), 'B')]
        old = dict(judge.GUARD); judge.GUARD['feat_guard'] = True
        try:
            J = judge.OnlineJudge(str(SPEC), crops=0, compact_ids=True)
            for t in range(3000):
                rows, P, look = [], {}, {}
                for k, (raw, b_, who) in enumerate(churn(t)):
                    rows.append([*b_, raw, 0.8, 0, k]); P[k] = list(b_); look[k] = F_A if who == 'A' else F_B
                J.step(rows, P, lambda ks, look=look: {k: judge.unit(look[k] + 0.02 * rng.normal(size=64)) for k in ks}, frame=t)
        finally:
            judge.GUARD.update(old)
        n = J.guard_state()['state_numbers']
        self.assertLess(n, 3000 // 20 // 2, n)   # 151 numbers were shown; without pruning all 151 keep state
        self.assertLessEqual(n, (judge.G_PRUNE + 30) // 20 + 2, n)   # numbers seen in the last G_PRUNE frames only
        self.assertEqual(set(J._g_mem) - {1}, {c for c in J._g_mem if c != 1 and J.t - J.hist[c][-1]['t'] <= judge.G_PRUNE + 30})


if __name__ == '__main__':
    unittest.main()
