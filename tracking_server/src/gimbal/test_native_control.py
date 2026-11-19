import math
import random
import unittest
from unittest.mock import patch

import numpy as np

from . import native_control as control
from . import predictive_lead
from .reentry_tilt import ReentryTilt


@unittest.skipUnless(control.native is not None, 'native gimbal extension is not built')
class NativeControlTests(unittest.TestCase):
    def python_result(self, function, *args):
        with patch.object(control, 'native', None):
            return function(*args)

    def test_rows_and_full_protocol_match_with_and_without_tilt(self):
        rows = [(1, -.05, -.05, 25.15, 101.25, .1255),
                (2, 125.55, 40.05, 240.05, 333.35, .3755),
                (3, 675.05, 498.15, 705.35, 580.45, .6255)]
        for people in ([], rows, rows + [rows[1]]):
            for hidden in (0, 2):
                for correction in (None, (0., 0.), (3.125, -1.75)):
                    args = (people, hidden, correction, 2, 640., 480.)
                    result = control.prepare(*args)
                    self.assertEqual(result, self.python_result(control.prepare, *args))
                    for hold in (False, True):
                        real, _ = result
                        args = (real, 2, hold, .65, 183.)
                        got = control.observations(*args)
                        self.assertEqual(got, self.python_result(control.observations, *args))
                        for picture in (False, True):
                            suffixes = (' PRED 2 1.0 2.0 3.0 4.0', '', ' VEL 2 3.0 4.0', ' LEAD 2 2.0 3.0 4')
                            args = (31, 100010001, len(real), picture, 640, 480, got[0], True, not hold, suffixes)
                            self.assertEqual(control.header(*args), self.python_result(control.header, *args))

    def test_suffix_and_crowd_parity(self):
        for hold in (False, True):
            for xywh in (False, True):
                for coords in ((10., 32., 60., 90.), (50., 240., 91., 390.)):
                    args = ('SEEK', 2, coords, xywh, hold, .5, 240.)
                    self.assertEqual(control.target_suffix(*args), self.python_result(control.target_suffix, *args))
            for vx, vy in ((0., 0.), (0., 30.), (0., -20.), (15.25, 20.25), (math.nan, 2.)):
                args = (2, vx, vy, hold)
                self.assertEqual(control.velocity(*args), self.python_result(control.velocity, *args))
        people = [(4, 0, 0, 40, 60, .8), (5, 30, 0, 70, 60, .9), (6, 85, 0, 125, 60, .9)]
        from . import follow_guard, search_follow

        self.assertEqual(follow_guard.overlaps(people[0], people), [1.0, 1.0 / 7.0, 0.0])
        for ignored in ((), {5}):
            args = (people, 4, 20, 30, 31, ignored)
            self.assertEqual(search_follow.count_near(*args), self.python_result(search_follow.count_near, *args))

    def test_reentry_and_velocity_over_sequences(self):
        rng = random.Random(42)
        native = control.reentry(7)
        python = ReentryTilt(7)
        previous_native = previous_python = None

        def motion(before, after):
            if before < 1.06 and after > 1.22:
                return None
            return np.array([[1., 0., (after - before) * 2.], [0., 1., (after - before) * -3.]])

        for frame in range(420):
            stamp = 1. + frame / 30.
            correction = ((frame % 4) - 2., (frame % 6) - 3.)
            step = frame % 35
            cy = 120. + (35. if 10 <= step <= 28 else 0.) + rng.uniform(-.12, .12)
            sample = None if step in (12, 13) else (200., cy, 60., 100.)
            self.assertEqual(native.step(stamp, sample, correction, motion),
                             python.step(stamp, sample, correction, motion), frame)
            observed = None if sample is None else (7, 160. + step, cy - 50, 220. + step, cy + 50)
            warp = (stamp - 1. / 30., stamp, motion(stamp - 1. / 30., stamp))
            args = (7, observed, stamp, warp, 200)
            previous_native, lead_native = predictive_lead.estimate(previous_native, *args, correction=correction)
            why = predictive_lead.LAST['why']
            with patch.object(control, 'native', None):
                previous_python, lead_python = predictive_lead.estimate(previous_python, *args, correction=correction)
                self.assertEqual(why, predictive_lead.LAST['why'], frame)
            np.testing.assert_allclose(lead_native, lead_python, atol=1e-10, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
