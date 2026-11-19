"""Exercise the bounded scan with deterministic camera-motion feedback."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .bounded_seek import BoundedSeek
from .test_follow_guard import PublishFixture, person
from ..common.state import APP

SIZE = (640., 480.)
AIM = (320., 240.)


def warp(dx=0., dy=0.):
    return np.array([[1., 0., dx], [0., 1., dy]])


class BoundedSeekTests(unittest.TestCase):
    def setUp(self):
        self.seek = BoundedSeek()
        self.view = SimpleNamespace(anchor=(9.96, (280., 190., 50., 100.)),
                                    direction=None, direction_at=None)
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


class BoundedSeekTransportTests(PublishFixture):
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
