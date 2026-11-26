"""Exercise the bounded scan with deterministic camera-motion feedback."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from .bounded_seek import BoundedSeek
from .follow_guard import UpperCoastView
from .test_follow_guard import PublishFixture, person
from . import live_tracks
from ..common.state import APP

SIZE = (640., 480.)
AIM = (320., 240.)


def warp(dx=0., dy=0.):
    return np.array([[1., 0., dx], [0., 1., dy]])


class BoundedSeekTests(unittest.TestCase):
    def setUp(self):
        self.seek = BoundedSeek()
        self.view = UpperCoastView()
        self.view.anchor = 9.96, (280., 190., 50., 100.)
        self.t = 10.

    def push(self, transform=warp(), prediction=None):
        box = self.seek.step(self.t, self.view, lambda *_: transform, SIZE, AIM, .5, prediction=prediction)
        self.t += .04
        return None if box is None else (round(box[0] + box[2] / 2 - AIM[0], 1),
                                         round(box[1] + box[3] / 2 - AIM[1], 1))

    def at(self, elapsed, transform=warp(), prediction=None):
        while self.t + .0001 < 10. + elapsed:
            self.push(prediction=prediction)
        return self.push(transform, prediction)

    def test_stationary_subject_looks_left_then_right_during_the_wait(self):
        self.assertIsNone(self.at(.76))
        self.assertEqual(self.at(.84), (-38.4, 0))
        self.assertIsNone(self.at(1.52))
        self.assertEqual(self.at(1.68), (38.4, 0))
        self.assertIsNone(self.at(2.56))
        self.assertIsNone(self.at(2.72))
        self.assertIsNone(self.at(4.00))

    def test_confirmed_motion_and_fallback_prediction_choose_the_first_side(self):
        self.assertEqual(self.at(.84, prediction=(340., 190., 50., 100.)), (38.4, 0))
        self.seek, self.t = BoundedSeek(), 10.
        self.view.direction, self.view.direction_at = (0, -1), 9.96
        self.assertEqual(self.at(.84, prediction=(340., 190., 50., 100.)), (-38.4, 0))

    def test_image_displacement_stops_each_side_and_returns_to_the_anchor(self):
        self.assertEqual(self.at(.84), (-38.4, 0))
        self.assertIsNone(self.push(warp(dx=18.)))
        self.assertEqual(self.at(1.68), (38.4, 0))
        self.assertIsNone(self.push(warp(dx=-36.)))
        self.assertIsNone(self.at(2.56))
        self.assertEqual(self.at(2.68), (-38.4, 0))
        self.assertIsNone(self.push(warp(dx=16.)))
        self.assertTrue(self.seek.stopped)
    def test_steering_keeps_upper_edge_in_view_without_trusted_descent(self):
        self.view.anchor = (9.96, (280., 88., 50., 100.))
        self.assertEqual(self.at(.84), (0, -28.8))
        self.assertIsNone(self.at(1.68))

    def test_cancellation_or_unavailable_camera_motion_stops_the_loss(self):
        for transform in (None, warp(dx=float('nan'))):
            with self.subTest(transform=transform):
                self.seek, self.t = BoundedSeek(), 10.
                self.assertEqual(self.at(.84), (-38.4, 0))
                self.assertIsNone(self.push(transform))
                self.assertIsNone(self.at(1.68))
        self.seek, self.t = BoundedSeek(), 10.
        self.push()
        self.t += .28
        self.assertIsNone(self.push())
        self.assertTrue(self.seek.stopped)

    def test_no_trusted_anchor_stays_stopped_and_total_time_is_bounded(self):
        self.view.anchor = None
        self.assertIsNone(self.at(1.68))
        self.seek, self.t = BoundedSeek(), 10.
        self.view.anchor = (9.96, (280., 190., 50., 100.))
        self.at(.84)
        self.push(warp(dx=12.))
        self.assertEqual(self.at(3.40), (38.4, 0))
        self.assertIsNone(self.at(3.52))
        self.assertTrue(self.seek.stopped)

    def test_wide_pan_does_not_move_a_trusted_box_out_of_the_frame(self):
        self.seek = BoundedSeek(wide_pan=True)
        self.view.anchor = (9.96, (100., 190., 70., 100.))
        self.view.direction, self.view.direction_at = (0, 1), 9.96
        self.assertEqual(self.at(.84), (-38.4, 0.))
        commands = [self.push() for _ in range(145)]
        self.assertNotIn((38.4, 0.), commands)
        self.assertIn((0., -28.8), commands)
        self.assertIn((0., 28.8), commands)

    def test_other_axis_starts_during_a_short_loss(self):
        for first in ((0, 1), (1, -1)):
            with self.subTest(first=first):
                self.seek, self.t = BoundedSeek(wide_pan=True), 10.
                self.view.anchor = (9.96, (320., 240., 90., 130.))
                self.view.direction, self.view.direction_at = first, 9.96
                commands = [self.push() for _ in range(45)]
                first_command = next(cmd for cmd in commands if cmd is not None)
                self.assertGreater(first_command[first[0]] * first[1], 0)
                self.assertTrue(any(cmd and cmd[1 - first[0]] for cmd in commands))

    def test_recent_trusted_edge_refreshes_search_when_clean_view_has_expired(self):
        for box, first_command in (((284.5, 405.5, 141., 151.), (0., 28.8)),
                                   ((280., 58., 90., 120.), (0., -28.8)),
                                   ((605., 240., 74., 140.), (38.4, 0.))):
            with self.subTest(box=box):
                self.seek, self.t = BoundedSeek(wide_pan=True), 10.
                self.view = UpperCoastView()
                self.view.observe(9.55, (300., 314., 194., 324.), lambda *_: warp(), SIZE)
                original = self.view.anchor
                self.view.observe(9.96, box, lambda *_: warp(), SIZE)
                self.assertEqual(self.view.anchor, original)
                self.assertEqual(self.at(.84), first_command)

    def test_nearly_invisible_box_cannot_start_search(self):
        self.seek, self.view = BoundedSeek(wide_pan=True), UpperCoastView()
        self.view.observe(9.96, (320., 510., 90., 80.), lambda *_: warp(), SIZE)
        self.assertIsNone(self.at(1.8))
        self.assertEqual(self.seek.status, 'no_trusted_view')

    def test_search_covers_four_directions_and_returns_regardless_of_last_motion(self):
        for motion in (None, (0, -1), (0, 1), (1, -1), (1, 1)):
            with self.subTest(motion=motion):
                self.check_four_direction_scan(motion)

    def check_four_direction_scan(self, motion):
        self.seek = BoundedSeek(wide_pan=True)
        self.view.anchor = (9.96, (320., 240., 90., 130.))
        self.view.direction, self.view.direction_at = motion, 9.96
        commanded = applied = (0, 0)
        next_change = 0.
        position = np.zeros(2)
        path = [position.copy()]
        directions = set()
        first = None
        for index in range(198):
            elapsed = index / 30.
            if elapsed >= next_change:
                applied = commanded
            displacement = np.asarray(applied) * 62. / 30.
            position += displacement
            path.append(position.copy())
            look = self.seek.step(10. + elapsed, self.view,
                                  lambda *_: warp(dx=-displacement[0], dy=-displacement[1]),
                                  SIZE, AIM, .5)
            desired = ((0, 0) if look is None else
                       tuple(int(np.sign(value)) for value in (look[0] + 10. - AIM[0], look[1] + 10. - AIM[1])))
            self.assertLessEqual(sum(map(abs, desired)), 1)
            if desired != commanded:
                next_change = elapsed + (.108 if desired != (0, 0) else .064)
                commanded = desired
            if first is None and any(desired):
                first = desired
            directions.add(desired)
        self.assertEqual(directions, {(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)})
        if motion is not None:
            self.assertEqual(first[motion[0]], motion[1])
        offsets = np.asarray(path)
        self.assertTrue(np.all(offsets.min(axis=0) < -19.), offsets.min(axis=0))
        self.assertTrue(np.all(offsets.max(axis=0) > 19.), offsets.max(axis=0))
        self.assertTrue(np.all(np.abs(position) < 7.), position)
        self.assertTrue(self.seek.stopped)

    def test_search_checks_the_top_without_cropping_even_after_downward_motion(self):
        for motion in (None, (1, 1)):
            with self.subTest(motion=motion):
                self.seek, self.t = BoundedSeek(wide_pan=True), 10.
                self.view.anchor = (9.96, (280., 87., 60., 118.))
                self.view.direction, self.view.direction_at = motion, 9.96
                commands = [self.push() for _ in range(160)]
                self.assertEqual(next(command for command in commands if command), (0., -28.8))
                self.assertNotIn((0., 28.8), commands)
                self.assertIn((-38.4, 0.), commands)
                self.assertIn((38.4, 0.), commands)


class BoundedSeekTransportTests(PublishFixture):
    def test_extended_overlap_does_not_expire_the_trusted_search_view(self):
        target = person(7, 280., h=120., w=65.)
        bystander = person(8, 325., h=120., w=65.)
        feature = np.array([1., 0.])
        with (patch.dict(os.environ, {'LWIR_FOLLOW_GUARD': '2', 'LWIR_COAST_MODE': 'fused',
                                      'LWIR_COAST_UPPER_GUARD': '1', 'LWIR_COAST_BOUNDED_SEEK': '1'}),
              patch.object(live_tracks, '_appearance', return_value=(feature, 0, 1))):
            APP.GB.bounded_seek_supported = True
            APP.GB.observation_status_supported = True
            self.frame([target], acknowledge=True, healthy_live=True)
            APP.GB.state.update(id=7, auto=1)
            APP.GB.selection_reset = False
            for _ in range(12):
                self.frame([target], healthy_live=True)
            guard = APP.GB.fguard
            self.assertTrue(guard.tmpl)
            risk = dict(phase='risk', mask=False, pred=None, reselect=0, why='')
            with patch.object(guard, 'step', return_value=risk):
                for _ in range(40):
                    self.assertNotIn(' SEEK ', self.frame([target, bystander], healthy_live=True))
            self.assertGreater(self.t - guard.upper_coast_view.anchor[0], 1.)
            lost = dict(phase='masked', mask=True, pred=None, reselect=0, why='')
            with patch.object(guard, 'step', return_value=lost):
                lines = [self.frame([target, bystander], healthy_live=True) for _ in range(58)]
            xs = [float(row.split(' SEEK 7 ')[1].split()[0]) for row in lines if ' SEEK 7 ' in row]
            self.assertTrue(any(x < 305 for x in xs) and any(x > 315 for x in xs), xs)
            self.assertTrue(all(' SEEK 8 ' not in row for row in lines))

    def test_guarded_occlusion_scans_both_sides_without_following_the_intruder(self):
        target = person(7, 280., h=120., w=65.)
        intruder = person(7, 345., cy=205., h=172., w=104.)
        bystander = person(8, 390.)
        feature = np.array([1., 0.])
        stamp = 1
        with (patch.dict(os.environ, {'LWIR_FOLLOW_GUARD': '2', 'LWIR_COAST_MODE': 'fused',
                                      'LWIR_COAST_UPPER_GUARD': '1', 'LWIR_COAST_BOUNDED_SEEK': '1'}),
              patch.object(live_tracks, '_appearance', side_effect=lambda ident: (feature, 0, stamp))):
            APP.GB.bounded_seek_supported = True
            APP.GB.observation_status_supported = True
            self.frame([target], acknowledge=True, healthy_live=True)
            APP.GB.state.update(id=7, auto=1)
            APP.GB.selection_reset = False
            for _ in range(12):
                self.frame([target], healthy_live=True)
            lost = [self.frame([intruder, bystander], healthy_live=True) for _ in range(60)]
            xs = [float(row.split(' SEEK 7 ')[1].split()[0]) for row in lost if ' SEEK 7 ' in row]
            self.assertTrue(any(x < 305 for x in xs) and any(x > 315 for x in xs), xs)
            self.assertTrue(all(' SEEK 8 ' not in row for row in lost))
            self.assertEqual(APP.GB.state['id'], 7)
            for stamp in range(2, 8):
                resumed = self.frame([target], healthy_live=True)
            self.assertNotIn(' SEEK ', resumed)
            self.assertEqual(APP.GB.state['guard'][0], 0)

    def test_only_the_selected_id_gets_seek_packets_and_a_bystander_stops_the_scan(self):
        with patch.dict(os.environ, {'LWIR_FOLLOW_GUARD': '1', 'LWIR_COAST_MODE': 'fused',
                                     'LWIR_COAST_UPPER_GUARD': '1', 'LWIR_COAST_BOUNDED_SEEK': '1'}):
            APP.GB.bounded_seek_supported = True
            self.frame([person(7, 280.)], acknowledge=True)
            APP.GB.state.update(id=7, auto=1)
            APP.GB.selection_reset = False
            for _ in range(12):
                self.frame([person(7, 280.)])
            lost = [self.frame([]) for _ in range(58)]
            xs = [float(row.split(' SEEK 7 ')[1].split()[0]) for row in lost if ' SEEK 7 ' in row]
            self.assertTrue(any(x < 305 for x in xs) and any(x > 315 for x in xs), xs)
            self.assertNotIn(' SEEK ', self.frame([person(8, 530.)]))
            self.assertTrue(all(' SEEK ' not in self.frame([]) for _ in range(10)))
            self.assertEqual(APP.GB.state['id'], 7)
