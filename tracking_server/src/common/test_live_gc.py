"""Verify cyclic cleanup and bounded fallback independently of the camera."""

import gc
import unittest
import weakref
from unittest.mock import patch

from . import live_gc


MIB = live_gc._MIB


class LiveGCTest(unittest.TestCase):
    def setUp(self):
        self.original = gc.get_threshold()
        self.now = 100.0
        self.rss = 700 * MIB
        self.policy = live_gc.LiveGC()
        self.addCleanup(self.policy.close)
        rss = patch.object(live_gc, '_rss', side_effect=lambda: self.rss)
        clock = patch.object(live_gc.time, 'monotonic', side_effect=lambda: self.now)
        for helper in (rss, clock):
            helper.start()
            self.addCleanup(helper.stop)

    def test_moves_old_cycle_cleanup_to_source_boundary(self):
        class Cycle:
            pass

        self.policy.frame(100)
        self.assertEqual(gc.get_threshold(), self.original)
        self.policy.frame(300)
        self.assertTrue(gc.isenabled())
        self.assertEqual(gc.get_threshold()[:2], self.original[:2])
        node = Cycle()
        node.link = node
        ref = weakref.ref(node)
        gc.collect(1)
        del node
        gc.collect(1)
        self.assertIsNotNone(ref())
        full_before = gc.get_stats()[2]['collections']
        self.policy.between_sources()
        self.assertIsNone(ref())
        self.assertEqual(gc.get_stats()[2]['collections'], full_before + 1)
        self.assertEqual(self.policy.snapshot()['between_sources'], 1)
        self.assertTrue(self.policy.snapshot()['active'])

    def test_rss_guard_restores_default_and_rearms_at_source_boundary(self):
        self.policy.frame(300)
        self.now += 3
        self.rss += 64 * MIB
        self.policy.frame(400)
        self.assertEqual(gc.get_threshold(), self.original)
        self.assertEqual(self.policy.snapshot()['mode'], 'rss_growth')
        self.assertEqual(self.policy.snapshot()['fallbacks'], 1)
        self.policy.frame(500)
        self.assertEqual(self.policy.snapshot()['fallbacks'], 1)
        self.policy.between_sources()
        self.assertTrue(self.policy.snapshot()['active'])

    def test_time_guard(self):
        self.policy.frame(300)
        self.now += 600
        self.policy.frame(400)
        self.assertTrue(self.policy.snapshot()['active'])
        self.assertTrue(self.policy.snapshot()['maintenance_due'])
        self.now += 3
        self.rss += 64 * MIB
        self.policy.frame(520)
        self.assertFalse(self.policy.snapshot()['active'])
        self.assertEqual(self.policy.snapshot()['mode'], 'rss_growth')
        self.policy.between_sources()
        self.assertFalse(self.policy.snapshot()['maintenance_due'])

    def test_absolute_rss_guard_and_close(self):
        self.rss = 1024 * MIB
        self.policy.frame(300)
        self.assertEqual(self.policy.snapshot()['mode'], 'rss_limit')
        self.assertFalse(self.policy.snapshot()['active'])
        self.rss = 700 * MIB
        self.policy.between_sources()
        self.assertTrue(self.policy.snapshot()['active'])
        self.policy.close()
        self.assertEqual(gc.get_threshold(), self.original)
        self.policy.between_sources()
        self.assertEqual(self.policy.snapshot()['between_sources'], 1)

    def test_rss_probe_error(self):
        self.policy.frame(300)
        self.now += 3
        with patch.object(live_gc, '_rss', side_effect=OSError):
            self.policy.frame(400)
        self.assertTrue(gc.isenabled())
        self.assertEqual(gc.get_threshold(), self.original)
        self.assertEqual(self.policy.snapshot()['mode'], 'rss_unavailable')


if __name__ == '__main__':
    unittest.main()
