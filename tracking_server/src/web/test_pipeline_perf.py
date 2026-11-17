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

    def test_five_completed_seconds_report_throughput_gaps_and_queue_drops(self):
        self.perf.camera(100.5, dropped=True)
        self.perf.tracked(100.5, 100.6)
        self.now = 101.2
        self.perf.dropped_input()
        self.perf.camera(102.9)
        self.perf.detected(102.9, 102.92)
        self.perf.tracked(102.9, 102.95)
        self.perf.camera(102.96, dropped=True)
        self.perf.tracked(102.96, 103.0)
        self.perf.camera(103.03)
        self.perf.tracked(103.03, 103.1)
        self.perf.camera(104.99)
        self.perf.tracked(104.99, 105.05)
        self.now = 104.99
        early = self.perf.snapshot()
        self.assertIsNone(early['track_fps_5s'])
        self.assertIsNone(early['queue_drops_5s'])

        self.now = 106.2
        data = self.perf.snapshot()
        self.assertEqual([data[k] for k in ('camera_fps_5s', 'det_fps_5s', 'track_fps_5s')], [.8, .2, .8])
        self.assertEqual(data['queue_drops_5s'], 2)
        self.assertEqual(data['output_interval_p95_ms'], 1765.0)

        self.now = 112.2
        idle = self.perf.snapshot()
        self.assertEqual(idle['queue_drops_5s'], 0)
        self.assertEqual(idle['track_fps_5s'], 0)
        self.assertIsNone(idle['output_interval_p95_ms'])

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

    def test_schedule_uses_real_offsets_and_includes_only_three_paired_frames(self):
        for received, detected, tracked, cores in (
            (100.010, 100.059, 100.061, (0,)),
            (100.043, 100.081, 100.095, (1,)),
            (100.077, 100.123, 100.126, (0, 1)),
            (100.109, 100.160, 100.165, ()),
        ):
            self.perf.camera(received, 2)
            self.perf.detected(received, detected, cores)
            self.perf.tracked(received, tracked)
        self.now = 100.2
        self.assertEqual(self.perf.snapshot()['frame_schedule_ms'], [
            [0.0, 38.0, 52.0], [34.0, 80.0, 83.0], [66.0, 117.0, 122.0],
        ])
        self.assertEqual(self.perf.snapshot()['frame_schedule_npu'], [[1], [0, 1], []])

    def test_schedule_never_joins_different_frames_or_reversed_stages(self):
        for received in (100.01, 100.04, 100.07, 100.10):
            self.perf.camera(received, 2)
        self.perf.detected(100.01, 100.05)  # no track output
        self.perf.tracked(100.04, 100.08)   # no detection for this arrival
        self.perf.detected(100.07, 100.12)
        self.perf.tracked(100.07, 100.11)   # track output before detection
        self.perf.detected(100.10, 100.13)
        self.perf.tracked(100.10, 100.15)
        self.perf.tracked(100.10, 100.16)   # repeated publication
        self.now = 100.2
        self.assertEqual(self.perf.snapshot()['frame_schedule_ms'], [[0.0, 30.0, 50.0]])

    def test_schedule_orders_an_out_of_order_completion_by_camera_receipt(self):
        for received, core in ((100.01, 1), (100.04, 0), (100.07, 1)):
            self.perf.camera(received, 2)
            self.perf.detected(received, received + .01, (core,))
        self.perf.tracked(100.07, 100.09)
        self.perf.tracked(100.01, 100.10)
        self.perf.tracked(100.04, 100.11)
        self.now = 100.15
        self.assertEqual(self.perf.snapshot()['frame_schedule_ms'], [
            [0.0, 10.0, 90.0], [30.0, 40.0, 100.0], [60.0, 70.0, 80.0],
        ])
        self.assertEqual(self.perf.snapshot()['frame_schedule_npu'], [[1], [0], [1]])

    def test_schedule_clears_for_replay_and_when_live_frames_stop(self):
        self.perf.camera(100.02, 1)
        self.perf.detected(100.02, 100.04)
        self.perf.tracked(100.02, 100.05)
        self.perf.camera(100.06, 1)
        self.now = 100.07
        self.assertTrue(self.perf.snapshot()['frame_schedule_ms'])
        self.perf.camera(100.08)  # replay packets lack a live camera stamp
        self.perf.detected(100.06, 100.09)  # in-flight live work cannot reappear
        self.perf.tracked(100.06, 100.11)
        self.perf.detected(100.08, 100.12)
        self.perf.tracked(100.08, 100.13)
        self.now = 100.15
        self.assertEqual(self.perf.snapshot()['frame_schedule_ms'], [])
        self.perf.camera(100.16, 1)
        self.perf.detected(100.16, 100.18)
        self.perf.tracked(100.16, 100.19)
        self.now = 100.20
        self.assertTrue(self.perf.snapshot()['frame_schedule_ms'])
        self.now = 103.0
        data = self.perf.snapshot()
        self.assertEqual(data['frame_schedule_ms'], [])
        self.assertIsNotNone(data['track_p95_ms'])
