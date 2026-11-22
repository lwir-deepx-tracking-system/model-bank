"""11-22: position trust of the follow guard v2 (LWIR_FOLLOW_POS_TRUST). Synthetic scenes built from the two real events of the board's recorder:
 A (12:41, id 44, a big person covered IoU 0.62-0.88 by id 23 for 3+ s, id 44 absent 1-2 frames about every second; see test_occ_hold.py),
 B (15:31, id 18, a small person, height about 150 px, another person on it only IoU 0.2-0.4; the tracker kept id 18, its box agreed with the guard's
   prediction to 0.05 heights, the appearance said 'looks like a person near it' and the guard masked it; the bridge let go after 10 s).
The features are made up (orthogonal unit vectors for the target T and the neighbour D, the covered box a blend near D): no recording of those events exists."""
import os
import unittest

import numpy as np

from . import follow_guard as fg
from . import follow_log
from . import test_occ_hold as occ
from types import SimpleNamespace as NS

T = np.array([1., 0., 0.]); D = np.array([0., 1., 0.]); X = np.array([0., 0., 1.])
BLEND = 0.3 * T + 0.9 * D          # the covered small crop: nearer the neighbour
MOTION = lambda a, b: np.array([[1., 0., 0.], [0., 1., 0.]])
FPS = 30.


def box(i, cx, cy, w, h):
    return [i, cx - w / 2., cy - h / 2., cx + w / 2., cy + h / 2., .9]


def make(on, sel=18):
    os.environ['LWIR_FOLLOW_POS_TRUST'] = '1' if on else '0'
    os.environ['LWIR_FOLLOW_OCC_HOLD'] = '0'
    g = fg.FollowGuardV2(sel); g.feats = {}
    g.app = lambda oid: (g.feats.get(oid), 0)
    g.kal = None
    return g


def drive(g, n, frame, t0=1.0):
    """frame(i, t) -> (people, {id: feature}); -> list of step() results"""
    res = []
    for i in range(n):
        t = t0 + i / FPS
        people, feats = frame(i, t)
        g.feats.clear(); g.feats.update(feats)
        res.append(g.step(t, people, MOTION))
    return res


def event_b(overlap_from=45, jump_at=None, gap_at=None, thief_at=None, size_glitch=()):
    """small target id 18 (60 x 150) walking 20 px/s; neighbour id 20 stands 110 px aside, then at frame overlap_from slides on it (IoU ~0.25)"""
    def frame(i, t):
        s = t - 1.0
        cx = 330. + 20. * s
        w, h = 60., 150.
        if i in size_glitch: w, h = w * 1.4, h * 1.4
        nb_x = cx + 110. if i < overlap_from else cx + 38.
        covered = i >= overlap_from
        feats = {20: D, 18: BLEND if covered else T}
        people = [box(18, cx, 230., w, h), box(20, nb_x, 232., 60., 160.)]
        if jump_at is not None and i >= jump_at:       # the id leaps onto the neighbour (an id swap seen from outside): 1.0 heights away
            people[0] = box(18, cx + 150., 230., w, h); feats[18] = D
        if gap_at is not None and gap_at <= i < gap_at + 12:   # the id vanishes for 12 frames (0.4 s) and comes back looking like the neighbour
            people = [people[1]]
        if gap_at is not None and i >= gap_at + 12: feats[18] = D
        if thief_at is not None and i >= thief_at:     # id 18 gone, a NEW id 99 stands exactly at the target's place, features like the neighbour
            people = [box(99, cx, 230., 60., 150.), people[1]]; feats[99] = D
        return people, feats
    return frame


def masks(res): return [r['mask'] for r in res]


class EventB(unittest.TestCase):
    def tearDown(self):
        for k in ('LWIR_FOLLOW_POS_TRUST', 'LWIR_FOLLOW_OCC_HOLD'): os.environ.pop(k, None)

    def test_scene_overlap_is_low(self):
        p, _ = event_b()(60, 3.0)
        o = fg.iou(p[0][1:5], p[1][1:5]); self.assertTrue(0.15 < o < 0.3, o)

    def test_without_the_rule_the_true_target_is_masked_until_the_bridge_gives_up(self):
        g = make(False); res = drive(g, int(12 * FPS), event_b())
        m = masks(res)
        self.assertTrue(any(m)); first = m.index(True)
        self.assertTrue(all(m[first:first + int(10 * FPS)]))   # sticky: still hidden after the bridge's 10 s reacquire timeout
        self.assertGreaterEqual(g.app_masks, 1); self.assertEqual(g.why, 'looks like a person near it')

    def test_with_the_rule_it_is_kept_all_the_time(self):
        g = make(True); res = drive(g, int(9 * FPS), event_b())
        self.assertFalse(any(masks(res))); self.assertFalse(any(r['reselect'] for r in res))
        self.assertGreater(g.pos_trust, 100); self.assertGreater(g.pos_keeps, 100)    # and the appearance did say 'distractor'
        self.assertEqual(g.app_masks, 0); self.assertEqual(res[-1]['phase'], 'risk')
        self.assertIn('position trust', g.why)

    def test_the_cap_ends_the_trust_of_one_overlap(self):
        g = make(True); res = drive(g, int((fg.POS_MAX_S + 4) * FPS), event_b())
        self.assertFalse(any(masks(res)[:int(fg.POS_MAX_S * FPS)]))
        self.assertTrue(any(masks(res)[-int(2 * FPS):]))    # then the appearance rules decide again

    def test_a_jump_still_masks(self):
        g = make(True); res = drive(g, int(6 * FPS), event_b(jump_at=int(3 * FPS)))
        m = masks(res)
        self.assertFalse(any(m[:int(3 * FPS)])); self.assertTrue(any(m[int(3 * FPS):int(3 * FPS) + 3]))   # within three frames
        self.assertTrue(m[-1])

    def test_an_id_that_comes_back_after_a_gap_is_judged_by_appearance(self):
        g = make(True); res = drive(g, int(6 * FPS), event_b(gap_at=int(3 * FPS)))
        self.assertTrue(masks(res)[-1])            # eligible only after a flicker, not after 0.4 s away: the old rules hide it

    def test_a_thief_with_a_new_id_at_the_same_place_is_not_followed(self):
        g = make(True); res = drive(g, int(6 * FPS), event_b(thief_at=int(3 * FPS)))
        self.assertFalse(any(r['reselect'] for r in res))
        self.assertEqual(g.sel, 18)
        self.assertFalse(any(r['pred'] is None for r in res[int(3 * FPS) + 1:int(3 * FPS) + 10]))   # it coasts on the prediction instead

    def test_one_odd_frame_does_not_mask_two_in_a_row_do(self):
        # a 1.4 x box (not agreeing with the prediction, still inside the old gate) while it looks like the neighbour
        g = make(True); res = drive(g, 90, event_b(size_glitch=(60,)))
        self.assertFalse(any(masks(res))); self.assertGreaterEqual(g.pos_waits, 1)
        g = make(True); res = drive(g, 90, event_b(size_glitch=(60, 61)))
        self.assertTrue(masks(res)[61]); self.assertFalse(masks(res)[60])
        g = make(False); res = drive(g, 90, event_b(size_glitch=(60,)))
        self.assertTrue(masks(res)[60])            # without the rule the single frame already hides it

    def test_unmask_at_once_when_position_agreement_returns(self):
        g = make(True); res = drive(g, 90, event_b(size_glitch=(60, 61)))     # masked at 61, the box is normal again from 62
        self.assertTrue(masks(res)[61]); self.assertFalse(any(masks(res)[64:]))
        self.assertGreaterEqual(g.pos_unmasks, 1)
        self.assertTrue(masks(res)[62])            # the first agreeing frame still waits for the second

    def test_default_off_and_the_recorder_has_the_counters(self):
        os.environ.pop('LWIR_FOLLOW_POS_TRUST', None); self.assertFalse(fg.pos_trust_enabled())
        g = make(True); drive(g, 90, event_b())
        gd = follow_log._guard(NS(fguard=g, state={'guard': [0, 0]}))
        for k in ('pos_trust', 'pos_keeps', 'pos_unmasks', 'pos_waits'): self.assertIn(k, gd)
        self.assertIn('position trust', gd['why'])


class Crossing(unittest.TestCase):
    """two people walking through each other with the tracker swapping the ids at the crossing (both boxes within 0.3 heights of the prediction at that
    moment): the position check alone cannot tell, the box's own motion can; the existing hand-back must still work with the rule on"""
    def tearDown(self):
        os.environ.pop('LWIR_FOLLOW_POS_TRUST', None)

    def test_swap_at_a_crossing_is_still_handed_back_with_the_rule_on(self):
        from .test_follow_guard import GuardV2Tests
        os.environ['LWIR_FOLLOW_POS_TRUST'] = '1'
        t = GuardV2Tests(); d = t.cross(True)
        self.assertEqual([x['reselect'] for x in d if x['reselect']], [8])
        d = t.cross(False)
        self.assertFalse(any(x['mask'] or x['reselect'] for x in d))


class EventA(unittest.TestCase):
    def tearDown(self):
        for k in ('LWIR_FOLLOW_POS_TRUST', 'LWIR_FOLLOW_OCC_HOLD'): os.environ.pop(k, None)

    def test_a_heavily_covered_target_with_flicker_is_kept_then_the_cap_applies(self):
        os.environ['LWIR_FOLLOW_POS_TRUST'] = '1'
        g, res = occ.run(False, secs=8.0); os.environ['LWIR_FOLLOW_POS_TRUST'] = '1'
        self.assertFalse(any(m for c, m, r, p in res)); self.assertFalse(any(r for c, m, r, p in res))
        self.assertGreater(g.pos_trust, 100)
        os.environ['LWIR_FOLLOW_POS_TRUST'] = '1'
        g, res = occ.run(False, secs=15.0)
        self.assertTrue(any(c and m for c, m, r, p in res[-90:]))
        os.environ['LWIR_FOLLOW_POS_TRUST'] = '0'
        g, res = occ.run(False, secs=8.0)
        self.assertTrue(any(c and m for c, m, r, p in res))     # and without it this is the old loss


if __name__ == '__main__':
    unittest.main()
