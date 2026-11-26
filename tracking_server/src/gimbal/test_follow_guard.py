"""PC-only tests of the followed-person prediction (bridge --coast 1) and the wrong-person guard (LWIR_FOLLOW_GUARD=1).
Run from tracking_server: python -m unittest src.gimbal.test_follow_guard"""
from collections import deque
from threading import Condition
from types import SimpleNamespace as NS
import os
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from . import follow_guard as F
from . import live_tracks, gimbal_link

DT = 1 / 30


def person(i, cx, cy=200., h=240., w=90.):
    return (i, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, .9)


class Scene:
    """Scene coordinates; the picture shows them shifted by the camera pan. motion() = the GMC warps of the frame pairs."""

    def __init__(self, gmc=True):
        self.t, self.pan, self.hist, self.gmc = 100., 0., {}, gmc

    def step(self, pan_step=0.):
        t0 = self.t
        self.t = round(self.t + DT, 6)
        self.pan += pan_step
        if self.gmc:
            self.hist[t0] = (self.t, np.array([[1., 0., -pan_step], [0., 1., 0.]]))
        return self.t

    def motion(self, t0, t1):
        out, t = np.eye(3), t0
        while t < t1:
            if t not in self.hist: return None
            t, w = self.hist[t]
            out = np.vstack([w, [0, 0, 1]]) @ out
        return out[:2] if t == t1 else None

    def shown(self, people):
        return [(p[0], p[1] - self.pan, p[2], p[3] - self.pan, p[4], p[5]) for p in people]


class ForegroundOcclusionTests(unittest.TestCase):
    def test_uncertain_id_cannot_resume_on_a_larger_box_after_short_gap(self):
        scene, guard = Scene(), F.FollowGuardV2(7)
        for _ in range(8):
            guard.step(scene.step(), [person(7, 260.)], scene.motion)
        guard.step(scene.step(), [person(7, 75.)], scene.motion)
        self.assertEqual(guard.phase, 'masked')
        for _ in range(9):
            guard.step(scene.step(), [], scene.motion)
        result = guard.step(scene.step(), [person(7, 285., h=354., w=133.), person(9, 210.)], scene.motion)
        self.assertTrue(result['mask'])
        self.assertTrue(result['foreground_blocked'])
        self.assertEqual(guard.trust[2:], (90., 240.))
        for _ in range(4):
            still_wrong = guard.step(scene.step(), [person(7, 285.), person(9, 205.)], scene.motion)
            self.assertTrue(still_wrong['mask'])
        self.assertEqual(guard.trust[0], 260.)

    def test_disputed_foreground_can_resume_after_three_new_appearance_samples(self):
        scene, guard = Scene(), F.FollowGuardV2(7)
        guard.allow_unlocalized_recovery = True
        vector = np.array([1., 0., 0.])
        guard.app = lambda _: (vector, 0, 10)
        for _ in range(8):
            guard.step(scene.step(), [person(7, 260.)], scene.motion)
        guard.step(scene.step(), [person(7, 75.)], scene.motion)
        for _ in range(9):
            guard.step(scene.step(), [], scene.motion)
        guard.app = lambda _: None
        self.assertTrue(guard.step(scene.step(), [person(7, 285., h=354., w=133.)], scene.motion)['mask'])
        for stamp in (11, 11, 12):
            guard.app = lambda _: (vector, 0, stamp)
            self.assertTrue(guard.step(scene.step(), [person(7, 268.)], scene.motion)['mask'])
        guard.app = lambda _: (vector, 0, 13)
        self.assertFalse(guard.step(scene.step(), [person(7, 268.)], scene.motion)['mask'])
        self.assertFalse(guard.foreground_identity_required)

    def _foreground_case(self):
        scene, guard = Scene(), F.FollowGuardV2(4)
        guard.allow_unlocalized_recovery = guard.continuous_control_owner = True
        feature = np.array([1., 0., 0.])
        guard.app = lambda ident: (feature, 0, 1)
        for _ in range(5):
            guard.step(scene.step(), [person(4, 398., 200., h=85., w=54.)], scene.motion)
        guard.app = lambda ident: (feature, F.APP_FRESH + 1, 1)
        grown = person(4, 378., 260., h=173., w=105.)
        for _ in range(7):
            decision = guard.step(scene.step(), [grown], scene.motion)
            self.assertTrue(decision['foreground_blocked'])
        return scene, guard, grown, feature

    def test_steady_selected_person_resumes_after_three_independent_features(self):
        scene, guard, grown, feature = self._foreground_case()
        other = person(5, 520., 260., h=173., w=94.)
        different = np.array([0., 1., 0.])
        stamp = 2
        guard.app = lambda ident: (feature if ident == 4 else different, 0, stamp)
        for _ in range(4):
            self.assertTrue(guard.step(scene.step(), [grown, other], scene.motion)['mask'])
        stamp += 1
        self.assertTrue(guard.step(scene.step(), [grown, other], scene.motion)['mask'])
        stamp += 1
        result = guard.step(scene.step(), [grown, other], scene.motion)
        self.assertFalse(result['mask'])
        self.assertEqual(result['reselect'], 0)
        self.assertEqual(guard.trust, F.cwh(grown))
        self.assertFalse(guard.foreground_blocked)

    def test_foreground_holds_when_same_number_does_not_prove_ownership(self):
        for failure in ('different_owner', 'similar_rival', 'unverified_rival', 'overlap', 'different_feature'):
            with self.subTest(failure=failure):
                scene, guard, grown, feature = self._foreground_case()
                different = np.array([0., 1., 0.])
                others = [person(5, 520., 260., h=173., w=94.)]
                rival = feature
                if failure == 'different_owner':
                    guard.continuous_control_owner = False
                    others = []
                elif failure == 'unverified_rival':
                    rival = None
                elif failure == 'overlap':
                    others = [person(5, 390., 260., h=170., w=97.)]
                    rival = different
                elif failure == 'different_feature':
                    feature = different
                    others = []
                guard.app = lambda ident: (feature if ident == 4 else rival, 0, stamp)
                for stamp in range(2, 8):
                    result = guard.step(scene.step(), [grown, *others], scene.motion)
                    self.assertTrue(result['foreground_blocked'])
                    self.assertTrue(result['mask'])
                    self.assertEqual(result['reselect'], 0)

    def test_same_id_large_foreground_cannot_steer_or_replace_trusted_box(self):
        scene, guard = Scene(), F.FollowGuardV2(7)
        target = person(7, 250., 200., h=120., w=65.)
        for _ in range(5):
            guard.step(scene.step(), [target], scene.motion)
        trust = guard.trust
        for x in (271., 285., 320., 345.):
            intruder = person(7, x, 205., h=172., w=104.)
            decision = guard.step(scene.step(), [intruder], scene.motion)
            self.assertTrue(decision['foreground_blocked'])
            self.assertTrue(decision['mask'])
            self.assertEqual(decision['pred'], trust)
            self.assertEqual(decision['reselect'], 0)
        self.assertTrue(guard.step(scene.step(), [], scene.motion)['mask'])
        self.assertTrue(guard.step(scene.step(), [target], scene.motion)['mask'])
        decision = guard.step(scene.step(), [target], scene.motion)
        self.assertFalse(decision['mask'])
        self.assertEqual(guard.sel, 7)

    def test_fast_continuous_approach_and_camera_pan_are_allowed(self):
        scene, guard = Scene(), F.FollowGuardV2(7)
        for i in range(9):
            t = scene.step(pan_step=6.)
            target = person(7, 230. + 8. * i, 210., h=100. + 10. * i, w=55. + 5. * i)
            decision = guard.step(t, scene.shown([target]), scene.motion)
            self.assertFalse(decision['mask'])
            self.assertEqual(guard.trust_t, t)

    def test_weak_corner_cannot_clear_disputed_target_by_position(self):
        scene, guard = Scene(), F.FollowGuardV2(211)
        guard.frame_size = (640, 480)
        target = (211, 420, 150, 653, 500, .71)
        for _ in range(8):
            self.assertFalse(guard.step(scene.step(), [target], scene.motion)['mask'])
        trust = guard.trust
        jump = (211, 400, 100, 800, 700, .40)
        self.assertTrue(guard.step(scene.step(), [jump], scene.motion)['foreground_blocked'])
        # The two clipped rows were logged during the loss of 211. A warm,
        # stationary anchor and a synthetic jump exercise the reaccept path.
        first = (211, 428, 156, 660, 502, .40)
        second = (211, 450, 174, 661, 491, .46)
        with patch.dict(os.environ, {'LWIR_FOLLOW_POS_TRUST': '1'}):
            for weak in (first, second, first, second):
                decision = guard.step(scene.step(), [weak], scene.motion)
                self.assertTrue(decision['mask'], 'a weak corner row regained steering')
                self.assertTrue(decision.get('foreground_blocked', False))
                self.assertEqual(guard.trust, trust)
            self.assertTrue(guard.step(scene.step(), [target], scene.motion)['mask'])
            self.assertFalse(guard.step(scene.step(), [target], scene.motion)['mask'])
            self.assertEqual(guard.sel, 211)

    def test_weak_corner_of_an_undisputed_person_keeps_steering(self):
        scene, guard = Scene(), F.FollowGuardV2(211)
        guard.frame_size = (640, 480)
        target = (211, 428, 156, 660, 502, .40)
        with patch.dict(os.environ, {'LWIR_FOLLOW_POS_TRUST': '1'}):
            for _ in range(5):
                decision = guard.step(scene.step(), [target], scene.motion)
                self.assertFalse(decision['mask'])
                self.assertEqual(guard.trust, F.cwh(target))

    def test_a_larger_passer_cannot_drag_the_last_position_via_position_trust(self):
        scene, guard = Scene(), F.FollowGuardV2(7)
        target = person(7, 280., 210., h=120., w=66.)
        foreground = person(9, 310., 212., h=183., w=101.)
        with patch.dict(os.environ, {'LWIR_FOLLOW_POS_TRUST': '1'}):
            for _ in range(8):
                guard.step(scene.step(), [target], scene.motion)
            trusted = guard.trust
            for x in (293., 298., 301.):
                suspect = person(7, x, 211., h=122., w=72.)
                result = guard.step(scene.step(), [suspect, foreground], scene.motion)
                self.assertTrue(result['mask'])
                self.assertEqual(guard.trust, trusted)
                self.assertEqual(result['pred'], trusted)
                self.assertFalse(result['reselect'])
            guard.step(scene.step(), [target], scene.motion)
            result = guard.step(scene.step(), [target], scene.motion)
            self.assertFalse(result['mask'])
            self.assertEqual(guard.pred, trusted)


class KeepIDWallResumeTests(unittest.TestCase):
    def lost(self):
        scene, guard = Scene(), F.FollowGuardV2(7, frame_size=(640, 480))
        feature = np.array([1., 0., 0.])
        guard.app = lambda _: (feature, 0, 1)
        for _ in range(5):
            guard.step(scene.step(), [person(7, 120., h=150.)], scene.motion)
        for _ in range(40):
            guard.step(scene.step(), [], scene.motion)
        self.assertEqual(guard.phase, 'masked')
        guard.keepid_resume = ((10, object()), scene.t)
        return scene, guard, feature

    def test_confirmed_keepid_returns_to_same_id_beyond_stale_prediction(self):
        scene, guard, feature = self.lost()
        target = person(7, 530., h=150.)
        self.assertFalse(guard.fits(F.cwh(target), scene.t))
        for stamp in (2, 2, 3):
            guard.app = lambda _, s=stamp: (feature, 0, s)
            self.assertTrue(guard.step(scene.step(), [target], scene.motion)['mask'])
        guard.app = lambda _: (feature, 0, 4)
        out = guard.step(scene.step(), [target], scene.motion)
        self.assertTrue(out['keepid_resumed'])
        self.assertFalse(out['mask'])
        self.assertEqual((out['phase'], out['reselect'], guard.trust[0]), ('track', 0, 530.))

    def test_same_number_without_keepid_transfer_does_not_bypass_position(self):
        scene, guard, feature = self.lost()
        guard.keepid_resume = None
        target = person(7, 530., h=150.)
        for stamp in (2, 3, 4, 5):
            guard.app = lambda _, s=stamp: (feature, 0, s)
            self.assertTrue(guard.step(scene.step(), [target], scene.motion)['mask'])

    def test_keepid_resume_rejects_overlap_and_ambiguous_appearance(self):
        for rival in (person(9, 540., h=150.), person(9, 300., h=150.)):
            with self.subTest(rival=rival):
                scene, guard, feature = self.lost()
                for stamp in (2, 3, 4, 5):
                    guard.app = lambda _, s=stamp: (feature, 0, s)
                    out = guard.step(scene.step(), [person(7, 530., h=150.), rival], scene.motion)
                    self.assertTrue(out['mask'])
                    self.assertFalse(out.get('keepid_resumed'))

    def test_keepid_transfer_must_match_old_and_new_owners_and_current_frame(self):
        namespace = {10: 7}
        old, new = (4, object()), (10, object())
        guard = NS(sel=7, trusted_control_owner=old, keepid_resume=None,
                   allow_unlocalized_recovery=True, phase='lost')
        bridge = NS(state={'lost': 1})
        event = dict(t=10, old_id=7, cand_tid=4, tracker_id=10)
        judge = NS(t=10, id_map=namespace, events=[event])
        with patch.dict(APP.__dict__, {'RULES': {'tracker':'botsort', 'relabel':True}}):
            for key, wrong in (('t', 9), ('old_id', 8), ('cand_tid', 5), ('tracker_id', 11)):
                with self.subTest(key=key), patch.dict(event, {key:wrong}):
                    live_tracks._update_keepid_resume(bridge, guard, new, namespace, judge, 100.)
                    self.assertIsNone(guard.keepid_resume)
            live_tracks._update_keepid_resume(bridge, guard, new, namespace, judge, 100.)
            self.assertEqual(guard.keepid_resume, (new, 100.))
            judge.t += 1
            live_tracks._update_keepid_resume(bridge, guard, new, namespace, judge, 101.)
            self.assertEqual(guard.keepid_resume, (new, 100.))
            live_tracks._update_keepid_resume(bridge, guard, new, namespace, judge, 102.)
            self.assertIsNone(guard.keepid_resume)


class SearchAppearanceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.scene = Scene()
        self.guard = F.FollowGuardV2(7)
        self.vector = np.array([1., 0., 0.])
        self.guard.app = lambda ident: (self.vector, 0, 10) if ident == 7 else None
        for _ in range(4):
            self.step([person(7, 230.)])
        for _ in range(30):
            self.step([])

    def step(self, rows):
        return self.guard.step(self.scene.step(), rows, self.scene.motion)

    def test_recycled_selected_id_with_unconfirmed_feature_cannot_end_search(self):
        self.guard.distract[14] = np.array([0., 1., 0.])
        self.guard.app = lambda ident: (self.guard.distract[14], 0, 11)
        for _ in range(4):
            out = self.step([person(7, 260.)])
            self.assertTrue(out['mask'])
            self.assertFalse(out['reselect'])
        self.assertEqual(self.guard.sel, 7)

    def test_wide_recovery_counts_three_distinct_appearance_observations(self):
        stamp = 11
        self.guard.app = lambda ident: (self.vector, 0, stamp)
        target = person(9, 440.)
        for _ in range(4):
            out = self.step([target])
            self.assertTrue(out['mask'])
            self.assertFalse(out['reselect'])
        stamp = 12
        self.assertTrue(self.step([target])['mask'])
        stamp = 13
        self.assertEqual(self.step([target])['reselect'], 9)

    def test_far_candidate_must_clear_a_higher_similarity_threshold(self):
        stamp = 11
        reasonable = np.array([.72, np.sqrt(1. - .72 ** 2), 0.])
        self.guard.app = lambda ident: (reasonable, 0, stamp)
        for stamp in range(11, 15):
            out = self.step([person(7, 480.)])
            self.assertTrue(out['mask'])
        self.assertEqual(self.guard.search_verify_n, 0)
        self.assertTrue(self.step([person(7, 255.)])['mask'])
        stamp += 1
        self.assertFalse(self.step([person(7, 255.)])['mask'])

    def test_two_similar_candidates_keep_the_gimbal_searching(self):
        stamp = 11
        self.guard.app = lambda ident: (self.vector, 0, stamp)
        for offset in range(3):
            stamp = 11 + offset
            out = self.step([person(7, 165.), person(9, 295.)])
            self.assertTrue(out['mask'])
            self.assertFalse(out['reselect'])

    def test_original_id_resumes_after_independent_evidence(self):
        stamp = 11
        self.guard.app = lambda ident: (self.vector, 0, stamp)
        self.assertTrue(self.step([person(7, 240.)])['mask'])
        stamp = 12
        self.assertFalse(self.step([person(7, 240.)])['mask'])
        self.assertEqual(self.guard.sel, 7)

    def test_search_log_distinguishes_missing_candidates_from_missing_features(self):
        from . import follow_log

        self.step([])
        self.assertEqual(self.guard.search_probe['last_reason'], 'no_person_rows')
        self.guard.app = lambda ident: None
        target = person(9, 240.)
        self.assertTrue(self.step([target])['mask'])
        self.assertEqual(self.guard.search_probe['last_reason'], 'no_fresh_features')
        self.assertEqual(self.guard.search_probe['people'], 1)
        stamp = 11
        self.guard.app = lambda ident: (self.vector, 0, stamp)
        self.assertTrue(self.step([target])['mask'])
        self.assertEqual(self.guard.search_probe['last_reason'], 'waiting_for_independent_features')
        stamp += 1
        self.assertEqual(self.step([target])['reselect'], 9)
        log = follow_log._guard(NS(fguard=self.guard, state={}))['search_probe']
        self.assertEqual(log['last_reason'], 'appearance_verified')
        self.assertEqual(log['candidate'], 9)
        self.assertEqual(log['outcomes']['no_fresh_features'], 1)

    def test_search_log_distinguishes_position_and_appearance_rejection(self):
        self.guard.app = lambda ident: (np.array([0., 1., 0.]), 0, 11)
        self.assertTrue(self.step([person(9, 240.)])['mask'])
        self.assertEqual(self.guard.search_probe['last_reason'], 'appearance_threshold_or_margin')
        self.guard.app = lambda ident: (self.vector, 0, 12)
        self.assertTrue(self.step([person(9, 240.), person(10, 242.)])['mask'])
        self.assertEqual(self.guard.search_probe['last_reason'], 'position_or_overlap_blocked')
        self.assertEqual(self.guard.search_probe['overlapping'], 2)

    def test_brief_missing_feature_does_not_reset_same_visible_target(self):
        target = person(7, 240.)
        self.guard.app = lambda ident: (self.vector, 0, 11)
        self.assertTrue(self.step([target])['mask'])
        self.guard.app = lambda ident: None
        self.assertTrue(self.step([target])['mask'])
        self.assertEqual(self.guard.search_verify_n, 1)
        self.guard.app = lambda ident: (self.vector, 0, 12)
        self.assertFalse(self.step([target])['mask'])

    def test_disappearing_or_competing_target_clears_the_partial_search_evidence(self):
        target = person(7, 240.)
        for interruption in ([], [target, person(9, 241.)]):
            with self.subTest(interruption=interruption):
                self.guard.app = lambda ident: (self.vector, 0, 11)
                self.assertTrue(self.step([target])['mask'])
                self.guard.app = lambda ident: None
                self.assertTrue(self.step(interruption)['mask'])
                self.assertEqual(self.guard.search_verify_n, 0)

    def test_search_evidence_expires_if_no_new_crop_is_received(self):
        target = person(7, 240.)
        self.guard.app = lambda ident: (self.vector, 0, 11)
        self.assertTrue(self.step([target])['mask'])
        for _ in range(8):
            self.assertTrue(self.step([target])['mask'])
        self.guard.app = lambda ident: (self.vector, 0, 12)
        self.assertTrue(self.step([target])['mask'])
        self.assertEqual(self.guard.search_verify_n, 1)

    def test_early_return_outside_motion_gate_requires_fresh_strong_evidence(self):
        scene, guard = Scene(), F.FollowGuardV2(7)
        guard.app = lambda ident: (self.vector, 0, 10)
        for _ in range(5):
            guard.step(scene.step(), [person(7, 230.)], scene.motion)
        for _ in range(4):
            guard.step(scene.step(), [], scene.motion)
        same = person(7, 405.)
        wrong = np.array([.53, np.sqrt(1. - .53 ** 2), 0.])
        guard.app = lambda ident: (wrong, 0, 11)
        self.assertTrue(guard.step(scene.step(), [same], scene.motion)['mask'])
        guard.app = lambda ident: (self.vector, 0, 12)
        result = guard.step(scene.step(), [same], scene.motion)
        self.assertTrue(result['mask'])
        self.assertFalse(result['seek_masked'])
        self.assertTrue(guard.step(scene.step(), [same], scene.motion)['mask'])
        guard.app = lambda ident: (self.vector, 0, 13)
        self.assertFalse(guard.step(scene.step(), [same], scene.motion)['mask'])
        self.assertEqual(guard.sel, 7)

    def _fast_return(self, others=()):
        scene, guard = Scene(), F.FollowGuardV2(7)
        guard.defer_retarget = True
        guard.allow_fast_return = True
        guard.app = lambda ident: (self.vector, 0, 10)
        for _ in range(5):
            guard.step(scene.step(), [person(7, 110.), *others], scene.motion)
        for _ in range(3):
            guard.step(scene.step(), [], scene.motion)
        return scene, guard

    def test_short_far_return_commits_only_a_fresh_three_sample_proposal(self):
        scene, guard = self._fast_return()
        returning = person(9, 470.)
        for stamp in (11, 11, 12):
            guard.app = lambda ident: (self.vector, 0, stamp)
            observed = guard.step(scene.step(), [returning], scene.motion)
            self.assertFalse(observed['reselect'])
            self.assertTrue(observed['mask'])
            self.assertEqual(guard.display_recovery_radius(scene.t, observed), F.SEARCH_FAST_RETURN_H)
            self.assertGreater(guard.dist(F.cwh(returning)), guard.recovery_radius(scene.t))
        guard.app = lambda ident: (self.vector, 0, 13)
        observed = guard.step(scene.step(), [returning], scene.motion)
        self.assertEqual(observed['reselect'], 9)
        self.assertEqual(guard.sel, 7)
        guard.accept_retarget(observed['retarget'])
        self.assertEqual(guard.sel, 9)

    def test_existing_search_circle_is_warped_before_the_next_candidate_check(self):
        scene, guard = self._fast_return()
        guard.set_wide((450., 200., 30.), scene.t)
        guard.step(scene.step(pan_step=90.), [], scene.motion)
        self.assertEqual(guard.wide, (360., 200., 30.))
        self.assertTrue(guard._wide_ok(F.cwh(person(7, 365.)), self.vector))
        self.assertFalse(guard._wide_ok(F.cwh(person(7, 450.)), self.vector))
        previous = scene.t
        new_time = scene.step()
        del scene.hist[previous]
        guard.step(new_time, [], scene.motion)
        self.assertIsNone(guard.wide)
        self.assertEqual(guard.display_recovery_radius(new_time, {}), 0.)

    def test_short_far_return_forgets_evidence_when_second_person_appears(self):
        scene, guard = self._fast_return()
        returning = person(9, 470.)
        for stamp in (11, 12):
            guard.app = lambda ident: (self.vector, 0, stamp)
            self.assertFalse(guard.step(scene.step(), [returning], scene.motion)['reselect'])
        guard.app = lambda ident: (self.vector, 0, 13)
        self.assertFalse(guard.step(scene.step(), [returning, person(10, 550.)], scene.motion)['reselect'])
        guard.app = lambda ident: (self.vector, 0, 14)
        self.assertFalse(guard.step(scene.step(), [returning], scene.motion)['reselect'])

    def test_short_far_return_rejects_preexisting_person_and_weak_match(self):
        returning = person(9, 470.)
        for others, vector in (((returning,), self.vector),
                               ((), np.array([.81, np.sqrt(1. - .81**2), 0.]))):
            with self.subTest(others=others):
                scene, guard = self._fast_return(others)
                for stamp in (11, 12, 13):
                    guard.app = lambda ident: (vector, 0, stamp)
                    self.assertFalse(guard.step(scene.step(), [returning], scene.motion)['reselect'])
                self.assertEqual(guard.sel, 7)

    def _continuous_target(self, prediction):
        # Model a stale trusted position while the raw owner has stayed visible.
        self.guard.allow_unlocalized_recovery = True
        self.guard.continuous_control_owner = True
        self.guard.pt_last_t = self.scene.t
        self.guard.pt_run = F.POS_RUN_N
        self.guard.pt_reborn = False
        self.guard.pred, self.guard.pred_t = prediction, self.scene.t
        return person(7, 235.)

    def test_continuous_owner_reanchors_with_three_fresh_features(self):
        for prediction in (None, (230., 710., 90., 240.)):
            with self.subTest(prediction=prediction):
                self.setUp()
                target = self._continuous_target(prediction)
                stamp = 11
                self.guard.app = lambda ident: (self.vector, 0, stamp)
                for _ in range(4):
                    self.assertTrue(self.step([target])['mask'])
                stamp = 12
                self.assertTrue(self.step([target])['mask'])
                stamp = 13
                result = self.step([target])
                self.assertFalse(result['mask'])
                self.assertEqual(result['reselect'], 0)
                self.assertEqual(self.guard.trust, F.cwh(target))

    def test_partial_overlap_requires_independent_identity_evidence(self):
        other_feature = np.array([0., 1., 0.])
        for case in ('partial', 'heavy', 'large_cover', 'similar_rival', 'missing_rival', 'new_owner'):
            with self.subTest(case=case):
                self.setUp()
                target = self._continuous_target((230., 200., 90., 240.))
                other = person(8, 305.)
                rival_feature = other_feature
                if case == 'heavy':
                    other = person(8, 270.)
                elif case == 'large_cover':
                    other = person(8, 235., 200., w=420., h=370.)
                elif case == 'similar_rival':
                    rival_feature = self.vector
                elif case == 'missing_rival':
                    rival_feature = None
                elif case == 'new_owner':
                    self.guard.continuous_control_owner = False
                for stamp in (11, 11, 12, 13):
                    self.guard.app = lambda ident: (self.vector if ident == 7 else rival_feature, 0, stamp)
                    out = self.step([target, other])
                    self.assertEqual(out['mask'], case != 'partial' or stamp < 13)
                    self.assertFalse(out['reselect'])
                if case == 'partial':
                    self.assertEqual(self.guard.phase, 'risk')
                    self.assertEqual(self.guard.trust, F.cwh(target))

    def test_unlocalized_target_stays_masked_without_independent_identity(self):
        for failure in ('disabled', 'owner_changed', 'second_person', 'moderate_feature', 'stale_feature', 'box_grew'):
            with self.subTest(failure=failure):
                self.setUp()
                target = self._continuous_target(None)
                people = [target]
                feature, age = self.vector, 0
                if failure == 'disabled':
                    self.guard.allow_unlocalized_recovery = False
                elif failure == 'owner_changed':
                    self.guard.continuous_control_owner = False
                elif failure == 'second_person':
                    people.append(person(9, 500.))
                elif failure == 'moderate_feature':
                    feature = np.array([.83, np.sqrt(1. - .83 ** 2), 0.])
                elif failure == 'stale_feature':
                    age = 3
                else:
                    people = [person(7, 235., w=160.)]
                self.guard.app = lambda ident: (feature, age, stamp)
                for stamp in range(11, 16):
                    result = self.step(people)
                    self.assertTrue(result['mask'])
                    self.assertEqual(result['reselect'], 0)


class PredictionTests(unittest.TestCase):
    def walk(self, g, s, n, x0, v, pan_step=0., present=True):
        out = []
        for k in range(n):
            t = s.step(pan_step)
            x = x0 + v * (s.t - 100.)
            out.append((x, g.step(t, s.shown([person(7, x)]) if present else [], s.motion)))
        return out

    def test_the_prediction_moves_with_the_person_not_with_the_camera(self):
        s, g = Scene(), F.FollowGuard(7, guard=False)
        self.walk(g, s, 30, 200., 90., pan_step=4.)        # walking right at 90 px/s, the camera turns 120 px/s
        lost = self.walk(g, s, 15, 200., 90., pan_step=4., present=False)
        x, d = lost[-1]
        self.assertIsNotNone(d['pred'])
        self.assertAlmostEqual(d['pred'][0], x - s.pan, delta=6.)   # where the person is in the picture 0.5 s later
        self.assertFalse(d['mask']); self.assertEqual(d['reselect'], 0)

    def test_offered_for_a_bounded_time_only(self):
        s, g = Scene(), F.FollowGuard(7, guard=False)
        self.walk(g, s, 30, 200., 90.)
        lost = self.walk(g, s, int(F.COAST_S / DT) + 6, 200., 90., present=False)
        self.assertIsNotNone(lost[int(F.COAST_S / DT) - 3][1]['pred'])
        self.assertIsNone(lost[-1][1]['pred'])

    def test_no_camera_motion_no_prediction(self):
        s, g = Scene(gmc=False), F.FollowGuard(7, guard=False)
        self.walk(g, s, 20, 200., 90.)
        self.assertIsNone(self.walk(g, s, 5, 200., 90., present=False)[-1][1]['pred'])

    def test_coast_distance_is_independent_of_camera_pan_and_reversal(self):
        for guard_class in (lambda: F.FollowGuard(7, guard=False), lambda: F.FollowGuardV2(7)):
            for distance in (149., 151.):
                for pan_steps in ((0., 0.), (180., -60.), (-180., 60.)):
                    with self.subTest(distance=distance, pan_steps=pan_steps):
                        s, g = Scene(), guard_class()
                        g.step(s.step(), [person(7, 320., h=100.)], s.motion)
                        g.vx = distance / (2 * DT)
                        for pan in pan_steps:
                            d = g.step(s.step(pan_step=pan), [], s.motion)
                        self.assertEqual(d['pred'] is not None, distance < 150.)
                        if d['pred'] is not None:
                            self.assertAlmostEqual(d['pred'][0], 320. + distance - s.pan, delta=.01)

    def test_large_pan_alone_does_not_cancel_coast(self):
        s, g = Scene(), F.FollowGuardV2(7)
        g.step(s.step(), [person(7, 320., h=100.)], s.motion)
        d = g.step(s.step(pan_step=180.), [], s.motion)
        self.assertAlmostEqual(d['pred'][0], 140.)

    def test_missing_camera_motion_does_not_reuse_velocity_after_a_new_observation(self):
        s, g = Scene(), F.FollowGuardV2(7)
        self.walk(g, s, 20, 200., 90.)
        self.assertGreater(g.vx, 50.)
        previous = s.t
        t = s.step()
        del s.hist[previous]
        g.step(t, [person(7, 300.)], s.motion)
        self.assertFalse(g.v_known)
        d = g.step(s.step(), [], s.motion)
        self.assertAlmostEqual(d['pred'][0], 300.)

    def test_coast_does_not_keep_old_direction_when_target_reverses(self):
        for reversals in (1, 2):
            with self.subTest(reversals=reversals):
                scene, guard = Scene(), F.FollowGuardV2(7)
                x = 190.
                for _ in range(24):
                    x += 4.
                    guard.step(scene.step(pan_step=2.), scene.shown([person(7, x, h=100.)]), scene.motion)
                for _ in range(reversals):
                    x -= 4.
                    guard.step(scene.step(pan_step=-5.), scene.shown([person(7, x, h=100.)]), scene.motion)
                result = guard.step(scene.step(pan_step=6.), [], scene.motion)
                expected = x - scene.pan - (4. if reversals == 2 else 0.)
                self.assertAlmostEqual(result['pred'][0], expected, delta=.01)

    def test_no_prediction_while_seen(self):
        s, g = Scene(), F.FollowGuard(7, guard=True)
        self.assertTrue(all(d['pred'] is None for _, d in self.walk(g, s, 30, 200., 90.)))


class KeepInViewTests(unittest.TestCase):
    AIM = (320., 160.)

    def control(self, b, frac=.37):
        return b[0], b[1] - b[3] / 2 + frac * b[3]

    def test_inside_the_picture_no_motion(self):
        b = F.keep_in_view([(300., 220., 60., 160.), (360., 230., 60., 160.)], .2, self.AIM, .37)
        self.assertEqual(tuple(round(v, 6) for v in self.control(b)), self.AIM)

    def test_near_the_edge_the_smallest_shift(self):
        b = F.keep_in_view([(560., 220., 60., 160.)], .2, self.AIM, .37)
        r = (F.KIV_U0 + F.KIV_U1 * .2) * 160.
        self.assertAlmostEqual(self.control(b)[0], 320. + (560. + 30. + r) - 640. * (1 - F.KIV_MARGIN), delta=.01)
        self.assertAlmostEqual(self.control(b)[1], 160., delta=.01)

    def test_estimates_disagree_aim_between_them(self):
        b = F.keep_in_view([(40., 220., 60., 160.), (700., 220., 60., 160.)], 1., self.AIM, .37)
        self.assertAlmostEqual(self.control(b)[0], 370., delta=.01)   # the middle of the region (x 40..700 widened)

    def test_nothing_to_keep(self):
        self.assertIsNone(F.keep_in_view([None], .2, self.AIM, .37))


class GuardTests(unittest.TestCase):
    def cross(self, guard, swap, n_after=20):
        """7 (followed) walks right, 8 walks left; they overlap around x 320. swap: from the overlap on, the tracker has the
        ids the other way round. Returns the decisions."""
        s, g = Scene(), F.FollowGuard(7, guard=guard)
        out = []
        for k in range(60 + n_after):
            t = s.step(0.)
            a, b = 200. + 120. * (t - 100.), 440. - 120. * (t - 100.)
            ia, ib = (8, 7) if swap and abs(a - b) < 80 or swap and a > b else (7, 8)
            out.append(g.step(t, [person(ia, a), person(ib, b, h=236.)], s.motion))
            if out[-1]['reselect']:
                g.sel = out[-1]['reselect']   # live_tracks sends 'A id'; the bridge then follows it
        return out

    def test_a_swap_in_a_crossing_moves_following_back_to_the_person(self):
        d = self.cross(guard=True, swap=True)
        self.assertEqual([x['reselect'] for x in d if x['reselect']], [8])

    def test_a_crossing_without_swap_changes_nothing(self):
        d = self.cross(guard=True, swap=False)
        self.assertFalse(any(x['reselect'] or x['mask'] for x in d))

    def test_off_never_masks_or_reselects(self):
        d = self.cross(guard=False, swap=True)
        self.assertFalse(any(x['reselect'] or x['mask'] for x in d))

    def test_walking_side_by_side_is_not_hidden(self):
        # 7 and 8 walk together, partly overlapping (IoU ~0.15), for 4 s while the camera follows: the box keeps its trust
        s, g = Scene(), F.FollowGuard(7, guard=True)
        out = []
        for k in range(120):
            t = s.step(3.)
            x = 200. + 90. * (t - 100.)
            out.append(g.step(t, s.shown([person(7, x), person(8, x + 65.)]), s.motion))
        self.assertFalse(any(d['mask'] or d['reselect'] for d in out))

    def lose_then(self, later, n=40):
        """7 walks right and vanishes behind 8 (standing still); later(k, s) = boxes after the loss (scene coordinates)."""
        s, g = Scene(), F.FollowGuard(7, guard=True)
        occluder = person(8, 420., h=260., w=100.)
        for k in range(25):
            t = s.step(2.)
            g.step(t, s.shown([person(7, 200. + 90. * (t - 100.)), occluder]), s.motion)
        out = []
        for k in range(n):
            t = s.step(2.)
            out.append(g.step(t, s.shown(later(k, t)), s.motion))
            if out[-1]['reselect']: g.sel = out[-1]['reselect']
        return out

    def test_the_old_id_back_on_someone_else_is_hidden(self):
        d = self.lose_then(lambda k, t: [person(7, 600.), person(8, 420., h=260., w=100.)] if k >= 3 else [person(8, 420., h=260., w=100.)])
        self.assertTrue(all(x['mask'] for x in d[4:]))
        self.assertTrue(all(x['reselect'] == 0 for x in d))

    def test_a_new_id_at_the_prediction_is_taken(self):
        d = self.lose_then(lambda k, t: [person(9, 200. + 90. * (t - 100.)), person(8, 600., h=260., w=100.)] if k >= 10 else [])
        self.assertEqual([x['reselect'] for x in d if x['reselect']], [9])

    def test_the_old_id_back_at_the_prediction_is_followed(self):
        d = self.lose_then(lambda k, t: [person(7, 200. + 90. * (t - 100.))] if k >= 10 else [])
        self.assertFalse(any(x['mask'] or x['reselect'] for x in d))
        self.assertEqual(d[-1]['phase'], 'track')

    def test_someone_who_was_already_there_is_not_taken(self):
        d = self.lose_then(lambda k, t: [person(8, 200. + 90. * (t - 100.), h=245.)] if k >= 10 else [person(8, 420., h=260., w=100.)])
        self.assertTrue(all(x['reselect'] == 0 for x in d))


class GuardV2Tests(unittest.TestCase):
    FA, FB = np.r_[1., 0., 0.], np.r_[.3, .95, 0.]   # appearance of the clicked person / of the other one

    def cross(self, swap, n_after=25):
        s, g = Scene(), F.FollowGuardV2(7)
        out = []
        for k in range(60 + n_after):
            t = s.step(0.)
            a, b = 200. + 120. * (t - 100.), 440. - 120. * (t - 100.)
            ia, ib = (8, 7) if swap and (abs(a - b) < 80 or a > b) else (7, 8)
            feats = {ia: self.FA, ib: self.FB}
            close = abs(a - b) < 100   # no feature while they overlap (crops are not clean)
            g.app = lambda oid, f=feats, c=close: None if c else (f.get(oid), 0)
            out.append(g.step(t, [person(ia, a), person(ib, b, h=236.)], s.motion))
            if out[-1]['reselect']: g.sel = out[-1]['reselect']
        return out

    def test_swap_is_handed_back_by_appearance(self):
        d = self.cross(True)
        self.assertEqual([x['reselect'] for x in d if x['reselect']], [8])

    def test_crossing_without_swap_changes_nothing(self):
        d = self.cross(False)
        self.assertFalse(any(x['reselect'] or x['mask'] for x in d))

    def test_occluder_with_another_look_is_not_taken(self):
        s, g = Scene(), F.FollowGuardV2(7)
        feats = {7: self.FA, 8: self.FB}
        g.app = lambda oid: (feats.get(oid), 0) if oid in feats else None
        occ = person(8, 420., h=245., w=100.)
        out = []
        for k in range(70):
            t = s.step(0.)
            x = 200. + 90. * (t - 100.)
            people = [person(7, x), occ] if k < 25 else [occ] if k < 50 else [person(8, x, h=245.), person(9, 650.)]
            out.append(g.step(t, people, s.motion))
        self.assertTrue(all(d['reselect'] == 0 for d in out))

    def test_new_id_that_looks_like_the_person_is_taken(self):
        s, g = Scene(), F.FollowGuardV2(7)
        feats = {7: self.FA, 9: self.FA}
        g.app = lambda oid: (feats.get(oid), 0) if oid in feats else None
        out = []
        for k in range(50):
            t = s.step(0.)
            x = 200. + 90. * (t - 100.)
            out.append(g.step(t, [person(7, x)] if k < 25 else [] if k < 35 else [person(9, x)], s.motion))
            if out[-1]['reselect']: g.sel = out[-1]['reselect']
        self.assertEqual([d['reselect'] for d in out if d['reselect']], [9])


class PublishFixture(unittest.TestCase):
    def setUp(self):
        self.saved = APP.__dict__.copy()
        APP.tracker = object()
        APP.relabeler = None
        APP.RULES = {}
        APP.LIVE = NS(camera_bytetrack=None)
        APP.LIVE_W, APP.LIVE_H = 640, 480
        APP.CTL = dict(lv=None, lead_px=0, lead=False, lead_ms=150, lead_motion=None, motion_hist=deque(maxlen=32))
        APP.GB = NS(wcond=Condition(), wopen=True, wq=deque(), wframe=None, people=[], frame=0, gui=False, skipped=0,
                    track_control_supported=True, state=dict(id=0, auto=0), coast_control_supported=True)
        APP.gb_send_frame = gimbal_link.gb_send_frame
        self.t = 1000.

    def tearDown(self):
        APP.__dict__.clear()
        APP.__dict__.update(self.saved)

    def frame(self, people, acknowledge=False, *, healthy_live=False, click_owners=None, click_namespace=None):
        t0 = self.t
        self.t += DT
        APP.CTL['motion_hist'].append((t0, self.t, [[1., 0., 0.], [0., 1., 0.]]))
        APP.CTL['lead_motion'] = APP.CTL['motion_hist'][-1]
        with patch('time.perf_counter', return_value=self.t + .01):
            live_tracks.publish(people, None, self.t, None, live_tracks.current_source(), healthy_live=healthy_live,
                                click_owners=click_owners, click_namespace=click_namespace)
        if acknowledge:
            live_tracks.halted(APP.GB.tracks_ticket)
        return APP.GB.wframe.split(b'\n', 1)[0].decode()

    def follow(self, n=10):
        self.frame([person(7, 300.)], acknowledge=True)
        APP.GB.state.update(id=7, auto=1)
        APP.GB.selection_reset = False
        for k in range(n):
            self.frame([person(7, 300. + 3. * k)])


class PublishTests(PublishFixture):
    def test_botsort_keepid_wall_return_reaches_bridge_under_same_number(self):
        feature = np.array([1., 0., 0.])
        namespace = {4: 7}
        judge = NS(id_map=namespace, t=1, events=[])
        APP.relabeler = judge
        APP.RULES.update(tracker='botsort', relabel=True)
        with patch.dict(os.environ, {'LWIR_FOLLOW_GUARD': '2'}), patch.object(live_tracks, '_appearance') as app:
            app.return_value = (feature, 0, 1)
            self.frame([person(7, 120., h=150.)], acknowledge=True, healthy_live=True,
                       click_owners={7:4}, click_namespace=namespace)
            APP.GB.state.update(id=7, auto=1)
            for _ in range(5):
                self.frame([person(7, 120., h=150.)], healthy_live=True, click_owners={7:4}, click_namespace=namespace)
            self.assertEqual(APP.GB.fguard.trusted_control_owner[0], 4)
            for _ in range(40):
                self.frame([], healthy_live=True, click_namespace=namespace)
            namespace.clear()
            namespace[10] = 7
            judge.t = 2
            judge.events.append(dict(t=2, old_id=7, cand_tid=4, tracker_id=10))
            for stamp in (2, 3, 4):
                judge.t = stamp
                app.return_value = (feature, 0, stamp)
                packet = self.frame([person(7, 530., h=150.)], healthy_live=True,
                                    click_owners={7:10}, click_namespace=namespace)
                if stamp < 4:
                    self.assertNotIn(' 7 485.0 ', packet)
            self.assertIn(' 7 485.0 ', packet)
            self.assertEqual(APP.GB.state['id'], 7)
            self.assertEqual(APP.GB.state['keepid_follow_resumes'], 1)

    def test_recovery_uses_the_live_appearance_namespace_for_every_tracker(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_GUARD': '2'}):
            self.follow()
            gb = APP.GB
            namespace = {4: 7}
            APP.relabeler = NS(id_map=namespace)
            for tracker in ('confirm3', 'rebind', 'sort', 'ocsort', 'botsort', 'deepocsort', 'hybridsort'):
                with self.subTest(tracker=tracker), gb.wcond:
                    APP.RULES['tracker'] = tracker
                    gb.click_namespace = namespace
                    live_tracks._guard_locked(gb, 7, self.t)
                    self.assertTrue(gb.fguard.allow_unlocalized_recovery)
                    gb.click_namespace = dict(namespace)
                    live_tracks._guard_locked(gb, 7, self.t)
                    self.assertFalse(gb.fguard.allow_unlocalized_recovery)

    def enable_reentry_tilt(self):
        self.frame([person(7, 300., cy=240., h=200.)], acknowledge=True)
        gb = APP.GB
        gb.state.update(id=7, auto=1)
        gb.proc = NS(args=['bridge', '--aim-y', '0.5', '--aim-screen-y', '0.5'])
        gb.lead_control_supported = True
        gb.control_epoch = 3
        APP.CTL['lead'] = True
        for _ in range(5):
            self.frame([person(7, 300., cy=240., h=200.)])

    def test_abrupt_drop_holds_tilt_before_the_box_disappears_and_resumes(self):
        self.enable_reentry_tilt()
        partial = person(7, 318., cy=318., h=124.)
        packet = self.frame([partial])
        self.assertTrue(APP.GB.state['reentry_tilt_hold'])
        self.assertEqual(APP.GB.people, [partial])
        self.assertIn(' 7 273.0 178.0 90.0 124.0 0.900', packet)
        self.assertEqual(float(packet.split(' LEAD 7 ')[1].split()[1]), 0.)
        for _ in range(5):
            self.frame([])
        self.frame([person(7, 318., cy=240., h=200.)])
        self.assertFalse(APP.GB.state['reentry_tilt_hold'])
        self.assertFalse(APP.GB.state.get('lost', 0))

    def test_loss_suppresses_a_downward_coast_and_seek(self):
        self.enable_reentry_tilt()

        def downward_targets(gb, *_args, **_kwargs):
            gb.fguard = NS(upper_coast_view=None)
            gb.seek_raw, gb.seek_out = (310., 259., 20., 20.), None
            return 0, (355., 400., 80., 100.)

        with patch.object(live_tracks, '_guard_locked', side_effect=downward_targets):
            packet = self.frame([])
        self.assertIn(' PRED 7 315.0 190.0 80.0 100.0', packet)
        self.assertNotIn(' SEEK ', packet)

    def test_clear_and_stable_lower_position_restores_tilt_after_short_hold(self):
        self.enable_reentry_tilt()
        low = person(7, 300., cy=315., h=200.)
        self.frame([low])
        self.assertTrue(APP.GB.state['reentry_tilt_hold'])
        for _ in range(14):
            packet = self.frame([low])
        self.assertFalse(APP.GB.state['reentry_tilt_hold'])
        self.assertIn(' 7 255.0 215.0 90.0 200.0 0.900', packet)

    def test_camera_motion_alone_does_not_trigger_the_drop_guard(self):
        self.enable_reentry_tilt()
        tilt = APP.GB.reentry_tilt
        with patch.object(live_tracks, '_motion', return_value=np.array([[1., 0., 0.], [0., 1., 60.]])):
            self.assertFalse(tilt.step(self.t + DT, (300., 300., 90., 200.), (0., 0.), live_tracks._motion))

    def test_pred_only_while_missing_and_only_to_a_coast_bridge(self):
        self.follow()
        self.assertNotIn(' PRED ', self.frame([person(7, 330.)]))
        sent = self.frame([])
        self.assertIn(' PRED 7 ', sent)
        x, y, w, h = map(float, sent.split(' PRED 7 ')[1].split()[:4])
        self.assertAlmostEqual(x + w / 2, 333., delta=6.)
        APP.GB.coast_control_supported = False   # an older bridge refuses unknown suffixes
        self.assertNotIn(' PRED ', self.frame([]))

    def test_guard_hides_the_old_id_on_another_person(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_GUARD': '1'}):
            self.follow()
            for k in range(3): self.frame([])
            sent = self.frame([person(7, 600.)])   # the old id comes back far from where the person must be
            self.assertNotIn(' 7 ', sent.split(' PRED ')[0])
            self.assertIn(' PRED 7 ', sent)
            self.assertEqual(APP.GB.state['guard'][1], 1)

    def test_off_by_default(self):
        APP.GB.coast_control_supported = False
        self.follow()
        for k in range(3): self.frame([])
        sent = self.frame([person(7, 600.)])
        self.assertIn('7 555.0', sent)
        self.assertNotIn('guard', APP.GB.state)

    def test_prediction_log_only_changes_nothing_sent(self):
        APP.GB.coast_control_supported = False
        with patch.dict(os.environ, {'LWIR_FOLLOW_PREDICT_LOG': '1'}):
            self.follow()
            sent = self.frame([])
            self.assertNotIn(' PRED ', sent)
            self.assertEqual(APP.GB.state['guard'][3], 0)
            self.assertIsNotNone(APP.GB.state['guard'][2])
            self.assertIn('7 555.0', self.frame([person(7, 600.)]))   # never hidden


class CoastSwitchTests(PublishFixture):
    """11-15 live A/B: LWIR_COAST_SWITCH=<file> turns the coast on / off at run time without a bridge restart."""

    def setUp(self):
        super().setUp()
        import tempfile
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'coast_on')
        self.env = patch.dict(os.environ, {'LWIR_COAST_SWITCH': self.path, 'LWIR_FOLLOW_PREDICT_LOG': '1'})
        self.env.start()
        self.fast = patch.object(live_tracks, 'COAST_SWITCH_S', 0.)   # read the file on every frame
        self.fast.start()
        live_tracks.COAST_SWITCH.update(path=None, said=None)

    def tearDown(self):
        self.fast.stop()
        self.env.stop()
        self.dir.cleanup()
        live_tracks.COAST_SWITCH.update(path=None, said=None)
        super().tearDown()

    def switch(self, word=None):
        if word is None:
            if os.path.exists(self.path): os.remove(self.path)
        else:
            with open(self.path, 'w') as f: f.write(word + '\n')

    def run_lines(self, prepare):
        """the frame lines of a fixed follow / loss / return sequence, from a fresh fixture (prepare(self) sets it up)"""
        self.tearDown(); self.setUp()
        prepare(self)
        lines = [self.frame([person(7, 300.)], acknowledge=True)]
        APP.GB.state.update(id=7, auto=1)
        APP.GB.selection_reset = False
        for k in range(8): lines.append(self.frame([person(7, 300. + 3. * k)]))
        for k in range(6): lines.append(self.frame([]))
        for k in range(3): lines.append(self.frame([person(7, 340. + 3. * k)]))
        return lines

    def test_off_sends_exactly_what_a_bridge_without_coast_gets(self):
        def off_coast_bridge(t): t.switch(None)
        def old_bridge(t):   # the deployed server: no coast bridge, no switch, no prediction log
            APP.GB.coast_control_supported = False
            os.environ.pop('LWIR_COAST_SWITCH', None)
            os.environ.pop('LWIR_FOLLOW_PREDICT_LOG', None)
        off = self.run_lines(off_coast_bridge)
        self.assertEqual(APP.GB.state['coast_sw'], 'off')
        self.assertEqual(APP.GB.state['guard'][3], 0)   # the prediction is still logged, never sent
        old = self.run_lines(old_bridge)
        self.assertNotIn('coast_sw', APP.GB.state)
        self.assertNotIn('guard', APP.GB.state)
        self.assertTrue(all(' PRED ' not in line for line in off))
        self.assertEqual(off, old)

    def test_on_sends_pred_while_missing(self):
        self.switch('')
        self.follow()
        self.assertNotIn(' PRED ', self.frame([person(7, 330.)]))
        self.assertIn(' PRED 7 ', self.frame([]))
        self.assertEqual(APP.GB.state['coast_sw'], 'vel')
        self.assertEqual(APP.GB.state['guard'][3], 1)

    def test_toggle_keeps_the_selection_and_takes_effect_at_the_next_frame(self):
        self.switch('vel')
        self.follow()
        self.assertIn(' PRED 7 ', self.frame([]))
        self.switch(None)
        self.assertNotIn(' PRED ', self.frame([]))
        self.assertEqual((APP.GB.state['coast_sw'], APP.GB.state['guard'][3]), ('off', 0))
        self.switch('vel')
        self.assertIn(' PRED 7 ', self.frame([]))
        self.assertEqual(APP.GB.state['id'], 7)
        self.assertFalse([c for c in APP.GB.wq if c.startswith(b'A ')])   # no new selection was sent

    def test_the_word_in_the_file_picks_the_mode(self):
        self.switch('fused')
        self.follow()
        sent = self.frame([])
        self.assertEqual(APP.GB.state['coast_sw'], 'fused')
        x, y, w, h = map(float, sent.split(' PRED 7 ')[1].split()[:4])
        self.assertAlmostEqual(x + w / 2, 320., delta=1.)   # keep in view: the person is inside the picture, no motion
        self.switch('vel')
        x, y, w, h = map(float, self.frame([]).split(' PRED 7 ')[1].split()[:4])
        self.assertGreater(x + w / 2, 330.)   # the velocity prediction itself

    def test_without_the_variable_nothing_changes(self):
        os.environ.pop('LWIR_COAST_SWITCH', None)
        self.follow()
        self.assertIn(' PRED 7 ', self.frame([]))
        self.assertNotIn('coast_sw', APP.GB.state)
        self.assertEqual(live_tracks.coast_switch(), (True, None))


if __name__ == '__main__':
    unittest.main()
