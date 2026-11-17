import unittest

from .pipeline_perf import PipelinePerf


class PipelinePerfTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.perf = PipelinePerf(lambda: self.now)

    def fps(self):
        data = self.perf.snapshot()
        return tuple(data[key] for key in ('camera_fps', 'det_fps', 'track_fps'))

    def test_stage_counts_share_one_completed_interval_and_zero_when_idle(self):
        self.perf.camera(100.9)
        self.perf.camera(100.95)  # received and then dropped; still counts as input
        self.perf.detected(100.9, 100.99)
        self.now = 100.99
        self.assertEqual(self.fps(), (None, None, None))
        self.perf.tracked(100.9, 101.0)  # crosses into the next interval
        self.now = 101.1
        self.assertEqual(self.fps(), (2, 1, 0))
        self.now = 102.1
        self.assertEqual(self.fps(), (0, 0, 1))
        self.now = 103.1
        self.assertEqual(self.fps(), (0, 0, 0))

    def test_p95_uses_frame_samples_and_expires_without_new_frames(self):
        self.perf.camera(100.05, 1)
        self.perf.camera(100.10)  # no camera clock for this frame
        self.perf.detected(100.05, 100.055)
        self.perf.detected(100.10, 100.115)
        self.perf.tracked(100.05, 100.075)
        self.perf.tracked(100.10, 100.140)
        self.perf.tracked(None, 100.15)
        self.perf.tracked(100.16, 100.15)  # invalid pairing
        self.now = 101.2
        data = self.perf.snapshot()
        self.assertEqual((data['camera_count'], data['det_count'], data['sample_count']), (1, 2, 2))
        self.assertEqual((data['camera_p95_ms'], data['det_p95_ms'], data['track_p95_ms']), (1, 14.5, 39.25))
        self.assertEqual(data['goal_ms'], 33)
        self.now = 161.2
        data = self.perf.snapshot()
        self.assertEqual((data['camera_count'], data['det_count'], data['sample_count']), (0, 0, 0))
        self.assertEqual((data['camera_p95_ms'], data['det_p95_ms'], data['track_p95_ms']), (None, None, None))
