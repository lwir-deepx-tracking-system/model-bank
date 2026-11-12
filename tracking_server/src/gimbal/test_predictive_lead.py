"""CPU tests for measured motion pairing and the lead filter."""

import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..track.camera_motion import GMC, gmc_job
from .predictive_lead import NO_LEAD, TIME_CONSTANT, estimate


IDENTITY = np.eye(2, 3)


def person(x, y):
    return (7, x - 10.0, y - 10.0, x + 10.0, y + 10.0, .9)


class LeadTests(unittest.TestCase):
    def test_image_reference_includes_camera_motion_without_changing_default(self):
        seed, _ = estimate(None, 7, person(100, 100), 1., None, 60)
        warp = np.array([[1., 0., -12.], [0., 1., 0.]])
        # Person moves +4 in the scene, camera shifts its image -12: net -8.
        observed = person(92, 100)
        scene, scene_lead = estimate(seed, 7, observed, 1.04, (1., 1.04, warp), 60)
        image, image_lead = estimate(seed, 7, observed, 1.04, (1., 1.04, warp), 60, space="image")
        self.assertGreater(scene.vx, 0)
        self.assertGreater(scene_lead[0], 0)
        self.assertLess(image.vx, 0)
        self.assertLess(image_lead[0], 0)
        self.assertEqual(estimate(seed, 7, observed, 1.04, None, 60, space="image")[1], NO_LEAD)

    def test_stationary_target_with_camera_motion_has_no_lead(self):
        # The affine is in padded canvas coordinates, while a person row is in the picture.
        warp = np.array([[.9, -.1, 20.], [.1, .9, -7.]])
        x, y = 170., 210.
        next_x, next_y = warp @ (x, y + 16., 1.)
        seed, _ = estimate(None, 7, person(x, y), 1., None, 150)
        sample, offset = estimate(seed, 7, person(next_x, next_y - 16.), 1.04, (1., 1.04, warp), 150)
        self.assertEqual(offset, NO_LEAD)
        self.assertAlmostEqual(sample.vx, 0.)
        self.assertAlmostEqual(sample.vy, 0.)

    def test_height_change_tracks_the_configured_anchor_not_the_box_center(self):
        before = (7, 100., 60., 140., 160., .9)
        taller = (7, 100., 56., 140., 206., .9)
        for space in ("scene", "image"):
            with self.subTest(space=space):
                seed, _ = estimate(None, 7, before, 1., None, 150, aim_y=.08, space=space)
                sample, offset = estimate(seed, 7, taller, 1.04, (1., 1.04, IDENTITY),
                                          150, aim_y=.08, space=space)
                self.assertEqual((seed.x, seed.y), (120., 68.))
                self.assertEqual((sample.x, sample.y, sample.vy), (120., 68., 0.))
                self.assertEqual(offset, NO_LEAD)

    def test_aim_change_reseeds_without_using_the_previous_velocity(self):
        row = (7, 100., 60., 140., 260., .9)
        seed, _ = estimate(None, 7, person(100, 100), 1., None, 150, aim_y=.08)
        moving, offset = estimate(seed, 7, row, 1.04, (1., 1.04, IDENTITY), 150, aim_y=.08)
        self.assertNotEqual(offset, NO_LEAD)
        changed, offset = estimate(moving, 7, row, 1.08, (1.04, 1.08, IDENTITY), 150, aim_y=.3)
        self.assertEqual((changed.y, changed.vx, changed.vy, offset), (120., 0., 0., NO_LEAD))
        _, offset = estimate(changed, 7, row, 1.12, (1.08, 1.12, IDENTITY), 150, aim_y=.3)
        self.assertEqual(offset, NO_LEAD)

    def test_invalid_anchor_observation_or_correction_drops_the_sample(self):
        row = person(100., 100.)
        seed, _ = estimate(None, 7, row, 1., None, 150, aim_y=.08)
        for observed, timestamp, aim_y, correction in (
            *((row, 1.04, value, None) for value in (None, float('nan'), float('inf'), -.01, 1.01)),
            ((7, 90., 90., 110., 90., .9), 1.04, .08, None),
            ((7, 90., 90., 110., float('inf'), .9), 1.04, .08, None),
            (row, float('nan'), .08, None),
            (row, 1.04, .08, (0., float('inf'))),
        ):
            with self.subTest(observed=observed, timestamp=timestamp, aim_y=aim_y, correction=correction):
                self.assertEqual(estimate(seed, 7, observed, timestamp, (1., 1.04, IDENTITY),
                                          150, aim_y=aim_y, correction=correction), (None, NO_LEAD))

    def test_invalid_motion_burst_gap_repeat_and_reversal_restart_filter(self):
        seed, _ = estimate(None, 7, person(100, 100), 1., None, 150)
        sample, lead = estimate(seed, 7, person(125, 100), 1.04, (1., 1.04, IDENTITY), 150)
        self.assertGreater(lead[0], 20.)
        for timestamp, pair in (
            (1.08, None),  # invalid or absent GMC
            (1.08, (1.03, 1.08, IDENTITY)),
            (1.08, (1.04, 1.09, IDENTITY)),
            (1.08, (1.04, 1.08, np.full((2, 3), np.nan))),
            (1.049, (1.04, 1.049, IDENTITY)),
            (1.29, (1.04, 1.29, IDENTITY)),
            (1.04, (1.04, 1.04, IDENTITY)),
            (1.00, (1.04, 1.00, IDENTITY)),
        ):
            with self.subTest(timestamp=timestamp, pair=pair):
                reseeded, offset = estimate(sample, 7, person(260, 210), timestamp, pair, 150)
                self.assertEqual(offset, NO_LEAD)
                self.assertEqual((reseeded.time, reseeded.vx, reseeded.vy), (timestamp, 0., 0.))
                resumed = timestamp + 0.04
                next_sample, next_lead = estimate(reseeded, 7, person(260, 210), resumed,
                                                  (timestamp, resumed, IDENTITY), 150)
                self.assertEqual((next_sample.vx, next_sample.vy, next_lead), (0., 0., NO_LEAD))

    def test_filter_depends_on_elapsed_seconds_and_bounds_lead_vector(self):
        for step in (.02, .05):
            sample, _ = estimate(None, 7, person(100, 100), 10., None, 150)
            for index in range(1, round(0.2 / step) + 1):
                timestamp = 10. + index * step
                sample, lead = estimate(sample, 7, person(100 + index * step * 200, 100), timestamp,
                                        (sample.time, timestamp, IDENTITY), 150)
            self.assertAlmostEqual(sample.vx, 200 * (1 - math.exp(-.2 / TIME_CONSTANT)))
            self.assertAlmostEqual(lead[0], sample.vx * .15)

        seed, _ = estimate(None, 7, person(50, 50), 100., None, 400)
        _, lead = estimate(seed, 7, person(300, 300), 100.01, (100., 100.01, IDENTITY), 400)
        self.assertAlmostEqual(math.hypot(*lead), 60.)
        self.assertGreater(lead[0], 0.)
        self.assertGreater(lead[1], 0.)


class MotionTests(unittest.TestCase):
    def test_fallback_identity_differs_from_successful_identity_and_pair_is_exact(self):
        a, b, c, d = (np.full((8, 8), n, np.uint8) for n in range(4))
        points = np.arange(12, dtype=np.float32).reshape(6, 1, 2)
        flags = np.ones((6, 1), np.uint8)
        affine_results = iter([None, IDENTITY.copy(), np.array([[1., 0., 5.], [0., 1., -3.]])])
        cv2 = SimpleNamespace(
            RANSAC=1, goodFeaturesToTrack=lambda *args, **kwargs: points,
            calcOpticalFlowPyrLK=lambda *args: (points, flags, None),
            estimateAffinePartial2D=lambda *args, **kwargs: (next(affine_results), None),
        )
        with patch.dict(sys.modules, {'cv2': cv2}):
            gmc = GMC(1)
            first, failed, valid = (gmc_job(gmc, frame)[0] for frame in (a, b, c))
            ordinary_matrix = gmc.apply(d)
        for motion in (first, failed, valid):
            np.testing.assert_array_equal(motion.warp, IDENTITY)
        self.assertFalse(first.valid)
        self.assertFalse(failed.valid)
        self.assertIsNone(failed.paired(a, b, 1., 1.04))
        self.assertTrue(valid.valid)
        self.assertEqual(valid.paired(b, c, 1.04, 1.08)[:2], (1.04, 1.08))
        self.assertIsNone(valid.paired(a, c, 1., 1.08))
        self.assertIsNone(valid.paired(b, d, 1.04, 1.08))
        self.assertIsNone(valid.paired(b, c, None, 1.08))
        self.assertEqual(gmc.n_fail, 1)
        np.testing.assert_array_equal(ordinary_matrix, [[1., 0., 5.], [0., 1., -3.]])


if __name__ == '__main__':
    unittest.main()
