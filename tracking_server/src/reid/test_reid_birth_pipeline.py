"""Exercise same-frame ReID slot reuse without a device or runtime SDK."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from . import reid_embed
from ..detect import npu_engine
from ..common import fast_path


class FakeRuntime:
    def __init__(self):
        self.events = []
        self.pending = {}
        self.fail_submit = None
        self.fail_wait = None
        self.invalid_output = None
        self.short_output = None
        self.inputs = []

    def run_async(self, data, *, user_arg, output_buffer):
        job = len(self.inputs)
        self.inputs.append(data.copy())
        self.events.append(('submit', job))
        if job == self.fail_submit:
            raise RuntimeError('submit failed')
        for pending_input, pending_output in self.pending.values():
            if np.shares_memory(data, pending_input) or np.shares_memory(output_buffer, pending_output):
                raise AssertionError('reused a busy slot')
        self.pending[job] = (data, output_buffer)
        return job

    def wait(self, job):
        self.events.append(('wait', job))
        data, output = self.pending[job]
        np.testing.assert_array_equal(data, self.inputs[job])
        if job == self.fail_wait:
            raise RuntimeError('wait failed')
        self.pending.pop(job)
        output[:] = [data[0, 0, 0, 0], data[0, -1, -1, 0], 1]
        if job == self.invalid_output:
            output[0] = np.nan
        return [output[:-1] if job == self.short_output else output]


class FakeSerial:
    W, H = 4, 3

    def __init__(self):
        self.b = FakeRuntime()
        self.output = np.empty(3, dtype=np.float32)

    def submit(self, data, frame):
        self.job = self.b.run_async(data, user_arg={'f': frame}, output_buffer=self.output)

    def wait(self):
        return self.b.wait(self.job)


class ReidBirthPipelineTests(unittest.TestCase):
    FAST_EMBED = False   # the two-at-a-time rule of 11-13 (fast_path 'embed' off)

    def setUp(self):
        self.gray = np.random.default_rng(13).integers(0, 256, (512, 640), dtype=np.uint8)
        self.boxes = {8: [-3, -2, 18, 14], 2: [630, 500, 650, 530], 6: [10, 20, 30, 40],
                      5: [11, 20, 12, 40], 9: [140.5, 150.5, 160.8, 170.9]}
        self.serial = FakeSerial()
        self.async_engine = npu_engine.AsyncEmbedder.__new__(npu_engine.AsyncEmbedder)
        self.async_engine.b = FakeRuntime()
        self.async_engine.n, self.async_engine.W, self.async_engine.H = 16, 4, 3
        self.async_engine.ins = [np.empty((1, 3, 4, 3), dtype=np.uint8) for _ in range(16)]
        self.async_engine.outs = [np.empty(3, dtype=np.float32) for _ in range(16)]
        self.async_engine.inflight, self.async_engine.quarantined = 0, False
        self.async_engine.sub_ms, self.async_engine.col_ms = [], []
        self.app = SimpleNamespace(cv2=cv2, args=SimpleNamespace(mock=True), rl_x=np.empty_like(self.async_engine.ins[0]),
                                   rl_eng=self.serial, rl_async=self.async_engine, rl_crop_ms=[], rl_npu=[],
                                   rl_embed=reid_embed.rl_embed)
        for module in (reid_embed, npu_engine):
            app_patch = patch.object(module, 'APP', self.app)
            app_patch.start()
            self.addCleanup(app_patch.stop)
        fast_patch = patch.dict(fast_path.FAST, {'embed': self.FAST_EMBED})
        fast_patch.start()
        self.addCleanup(fast_patch.stop)

    def embed(self, keys, *, enabled=True):
        with patch.dict(os.environ, {'LWIR_REID_BIRTH_PIPELINE': '1'} if enabled else {}, clear=True):
            return reid_embed.rl_embed(keys, self.gray, self.boxes)

    def test_keepid_reuses_only_exact_same_frame_pixels(self):
        tracked_boxes = {8: [-5.0, -3.0, 18.01, 14.8], 6: [9.9, 20., 30., 40.]}
        tracked_features = reid_embed.rl_embed([8, 6], self.gray, tracked_boxes)
        expected = reid_embed.rl_embed([8, 6, 2], self.gray, self.boxes)
        sync_before = len(self.serial.b.inputs)
        with patch.dict(fast_path.STAT, {'embed_reused': 0}):
            sync, submit = reid_embed.same_frame_keepid(self.gray, self.boxes, reid_embed.rl_embed,
                                                       self.async_engine.submit, tracked_boxes, tracked_features)
            sync_result = sync([6, 8])
            self.assertEqual(list(sync_result), [6, 8])
            self.assertEqual(len(self.serial.b.inputs)-sync_before, 1)
            collect = submit([8, 2, 6])
            self.assertEqual(self.async_engine.inflight, 2)
            delayed = collect()
            self.assertEqual(list(delayed), [8, 2, 6])
            self.assertEqual(self.async_engine.inflight, 0)
            self.assertEqual(fast_path.STAT['embed_reused'], 2)
        for output in (sync_result, delayed):
            for key, feature in output.items():
                np.testing.assert_array_equal(feature, expected[key])

    def test_botsort_collects_only_after_the_prediction_boundary(self):
        with patch.dict(os.environ, {'LWIR_REID_BIRTH_PIPELINE': '1'}, clear=True):
            batch = reid_embed.start_botsort_batch([8, 2], self.gray, self.boxes)
        events = self.async_engine.b.events
        self.assertEqual(events, [('submit', 0), ('submit', 1)])
        events.append(('predict', None))
        self.assertEqual(list(batch.collect()), [8, 2])
        batch.drain()
        self.assertEqual(events[2:], [('predict', None), ('wait', 0), ('wait', 1)])
        self.assertEqual(self.async_engine.inflight, 0)

    def test_botsort_early_exit_drains_an_uncollected_batch_once(self):
        with patch.dict(os.environ, {'LWIR_REID_BIRTH_PIPELINE': '1'}, clear=True):
            batch = reid_embed.start_botsort_batch([8, 2], self.gray, self.boxes)
        batch.drain()
        batch.drain()
        self.assertEqual(self.async_engine.b.events, [('submit', 0), ('submit', 1), ('wait', 0), ('wait', 1)])
        self.assertEqual(self.async_engine.inflight, 0)

    def test_bounded_batches_preserve_crops_order_embeddings_and_ownership(self):
        keys = [8, 2, 5, 6, 9, 8]
        expected = self.embed(keys, enabled=False)
        actual = self.embed(keys)
        self.assertEqual(list(actual), [8, 2, 6, 9])
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
            self.assertTrue(actual[key].flags.owndata)
        self.assertEqual(len(self.async_engine.b.inputs), len(self.serial.b.inputs))
        for actual_input, expected_input in zip(self.async_engine.b.inputs, self.serial.b.inputs):
            np.testing.assert_array_equal(actual_input, expected_input)
        self.assertEqual(self.async_engine.b.events,
                         [('submit', 0), ('submit', 1), ('wait', 0), ('wait', 1),
                          ('submit', 2), ('wait', 2),
                          ('submit', 3), ('submit', 4), ('wait', 3), ('wait', 4)])
        self.assertEqual(self.async_engine.inflight, 0)
        self.assertFalse(self.async_engine.b.pending)
        self.assertEqual(len(self.app.rl_crop_ms), 5)
        self.assertEqual(self.app.rl_npu, [])
        self.assertEqual((self.async_engine.sub_ms, self.async_engine.col_ms), ([], []))
        self.assertEqual([count for count, _ in self.async_engine.birth_pipeline_batches], [2, 1, 2])
        self.embed([2, 6])
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_busy_unavailable_and_single_crop_keep_serial_path(self):
        occupied = self.async_engine.b.run_async(self.async_engine.ins[0], user_arg=None,
                                                 output_buffer=self.async_engine.outs[0])
        self.async_engine.inflight = 1
        self.assertEqual(list(self.embed([8, 2])), [8, 2])
        self.assertEqual(self.async_engine.b.events, [('submit', occupied)])
        self.assertEqual(self.async_engine.inflight, 1)
        self.assertIn(occupied, self.async_engine.b.pending)
        self.app.rl_async = None
        self.assertEqual(list(self.embed([8, 2])), [8, 2])
        self.app.rl_async = self.async_engine
        self.async_engine.b.wait(occupied)
        self.async_engine.inflight = 0
        self.assertEqual(list(self.embed(iter([8]))), [8])
        self.assertFalse(hasattr(self.async_engine, 'birth_pipeline_batches'))
        self.assertEqual(len(self.app.rl_crop_ms), 5)

    def test_nonfinite_and_wrong_size_outputs_drain_remaining_jobs(self):
        for attribute in ('invalid_output', 'short_output'):
            with self.subTest(attribute=attribute):
                self.async_engine.b = runtime = FakeRuntime()
                setattr(runtime, attribute, 0)
                with self.assertRaisesRegex(AssertionError, 're-ID output'):
                    self.embed([8, 2, 6])
                self.assertEqual(runtime.events, [('submit', 0), ('submit', 1), ('wait', 0), ('wait', 1)])
                self.assertFalse(runtime.pending)
                self.assertEqual(self.async_engine.inflight, 0)
                self.assertEqual(list(self.embed([2, 6])), [2, 6])

    def test_submit_failure_drains_known_job(self):
        runtime = self.async_engine.b
        runtime.fail_submit = 1
        with self.assertRaisesRegex(RuntimeError, 'submit failed'):
            self.embed([8, 2, 6])
        self.assertEqual(runtime.events, [('submit', 0), ('submit', 1), ('wait', 0)])
        self.assertFalse(runtime.pending)
        self.assertEqual(self.async_engine.inflight, 0)
        self.assertTrue(self.async_engine.quarantined)
        with self.assertRaisesRegex(RuntimeError, 'quarantined'):
            self.async_engine.submit([8, 2], self.gray, self.boxes)
        self.assertEqual(runtime.events, [('submit', 0), ('submit', 1), ('wait', 0)])

    def test_failed_wait_leaves_slot_quarantined_and_drains_other_job(self):
        runtime = self.async_engine.b
        runtime.fail_wait = 0
        with self.assertRaisesRegex(RuntimeError, 'wait failed'):
            self.embed([8, 2, 6])
        expected_events = [('submit', 0), ('submit', 1), ('wait', 0), ('wait', 1)]
        self.assertEqual(runtime.events, expected_events)
        self.assertEqual(self.async_engine.inflight, 1)
        self.assertTrue(self.async_engine.quarantined)
        self.assertEqual(list(runtime.pending), [0])
        self.assertEqual(list(self.embed([8, 2])), [8, 2])
        with self.assertRaisesRegex(RuntimeError, 'quarantined'):
            self.async_engine.submit([8, 2], self.gray, self.boxes)
        self.assertEqual(runtime.events, expected_events)

    def test_refresh_failed_wait_quarantines_submission_and_existing_collect(self):
        runtime = self.async_engine.b
        collect = self.async_engine.submit([8, 2], self.gray, self.boxes)
        runtime.fail_wait = 0
        with self.assertRaisesRegex(RuntimeError, 'pending work'):
            self.async_engine.submit([6, 9], self.gray, self.boxes)
        with self.assertRaisesRegex(RuntimeError, 'wait failed'):
            collect()
        expected_events = [('submit', 0), ('submit', 1), ('wait', 0), ('wait', 1)]
        self.assertEqual(runtime.events, expected_events)
        self.assertEqual(list(runtime.pending), [0])
        self.assertTrue(self.async_engine.quarantined)
        with self.assertRaisesRegex(RuntimeError, 'quarantined'):
            collect()
        with self.assertRaisesRegex(RuntimeError, 'quarantined'):
            self.async_engine.submit([6, 9], self.gray, self.boxes)
        self.assertEqual(runtime.events, expected_events)

    def test_refresh_overflow_collects_two_at_a_time_before_deferred_jobs(self):
        runtime = self.async_engine.b
        keys = [8, 2] * 9
        with patch.dict(os.environ, {'LWIR_REID_BIRTH_PIPELINE': '1'}):
            collect = self.async_engine.submit(keys, self.gray, self.boxes)
        self.assertEqual(self.async_engine.inflight, 16)
        self.assertEqual(runtime.events[:6], [('submit', 0), ('submit', 1), ('wait', 0), ('wait', 1),
                                             ('submit', 2), ('submit', 3)])
        self.assertEqual(list(collect()), [8, 2])
        self.assertFalse(runtime.pending)
        self.assertEqual(self.async_engine.inflight, 0)
        self.assertEqual(len(self.async_engine.sub_ms), 1)
        self.assertEqual(len(self.async_engine.col_ms), 1)


class ReidFastEmbedTests(ReidBirthPipelineTests):
    """2026-11-15 fast_path 'embed': up to four crops together, also beside the refresh crops of the last frame."""
    FAST_EMBED = True

    test_bounded_batches_preserve_crops_order_embeddings_and_ownership = None
    test_nonfinite_and_wrong_size_outputs_drain_remaining_jobs = None
    test_submit_failure_drains_known_job = None
    test_failed_wait_leaves_slot_quarantined_and_drains_other_job = None
    test_refresh_overflow_collects_two_at_a_time_before_deferred_jobs = None

    def test_four_at_a_time_same_embeddings_as_one_by_one(self):
        keys = [8, 2, 5, 6, 9, 8]
        expected = self.embed(keys, enabled=False)
        actual = self.embed(keys)
        self.assertEqual(list(actual), [8, 2, 6, 9])
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
            self.assertTrue(actual[key].flags.owndata)
        for actual_input, expected_input in zip(self.async_engine.b.inputs, self.serial.b.inputs):
            np.testing.assert_array_equal(actual_input, expected_input)
        self.assertEqual(self.async_engine.b.events,
                         [('submit', 0), ('submit', 1), ('submit', 2), ('wait', 0), ('wait', 1), ('wait', 2),
                          ('submit', 3), ('submit', 4), ('wait', 3), ('wait', 4)])
        self.assertEqual(self.async_engine.inflight, 0)
        self.assertFalse(self.async_engine.b.pending)
        self.assertEqual([count for count, _ in self.async_engine.birth_pipeline_batches], [3, 2])

    def test_crops_go_together_beside_the_refresh_crops_of_the_last_frame(self):
        runtime = self.async_engine.b
        expected = self.embed([6, 9, 8], enabled=False)
        collect = self.async_engine.submit([8, 2], self.gray, self.boxes)   # refresh crops on slots 0 and 1, collected at the next frame
        self.assertEqual(self.async_engine.reserved, (2, 2))
        actual = self.embed([6, 9, 8])   # FakeRuntime refuses a busy slot
        self.assertEqual(runtime.events, [('submit', 0), ('submit', 1), ('submit', 2), ('submit', 3), ('submit', 4),
                                          ('wait', 2), ('wait', 3), ('wait', 4)])
        self.assertEqual(self.async_engine.inflight, 2)
        self.assertEqual(sorted(runtime.pending), [0, 1])
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        refreshed = collect()
        self.assertEqual(list(refreshed), [8, 2])
        np.testing.assert_array_equal(refreshed[8], expected[8])
        self.assertIsNone(self.async_engine.reserved)
        self.assertEqual(self.async_engine.inflight, 0)
        self.assertFalse(runtime.pending)
        self.assertEqual(list(self.embed([2, 6])), [2, 6])
        self.assertEqual(runtime.events[-4:], [('submit', 5), ('submit', 6), ('wait', 5), ('wait', 6)])

    def test_refresh_with_a_skipped_crop_keeps_its_slots_apart(self):
        collect = self.async_engine.submit([5, 8, 2], self.gray, self.boxes)   # box 5 is too narrow: slots 1 and 2 are used, 0 is not
        self.assertEqual(self.async_engine.reserved, (3, 2))
        self.assertEqual(list(self.embed([6, 9])), [6, 9])
        self.assertEqual(list(collect()), [8, 2])
        self.assertEqual(self.async_engine.inflight, 0)

    def test_unknown_jobs_in_flight_or_a_full_engine_keep_one_by_one(self):
        self.async_engine.b.run_async(self.async_engine.ins[0], user_arg=None, output_buffer=self.async_engine.outs[0])
        self.async_engine.inflight = 1; self.async_engine.reserved = (2, 2)   # the count does not match what submit wrote down
        self.assertEqual(list(self.embed([8, 2])), [8, 2])
        self.assertEqual(len(self.async_engine.b.events), 1)
        self.async_engine.reserved = (15, 1)   # one free slot only
        self.assertEqual(list(self.embed([8, 2])), [8, 2])
        self.assertEqual(len(self.async_engine.b.events), 1)

    def test_failed_crop_drains_the_others_and_leaves_the_refresh_alone(self):
        runtime = self.async_engine.b
        collect = self.async_engine.submit([8, 2], self.gray, self.boxes)
        runtime.invalid_output = 2
        with self.assertRaisesRegex(AssertionError, 're-ID output'):
            self.embed([6, 9, 8])
        self.assertEqual(runtime.events[-3:], [('wait', 2), ('wait', 3), ('wait', 4)])
        self.assertEqual(self.async_engine.inflight, 2)
        self.assertEqual(list(collect()), [8, 2])
        self.assertFalse(runtime.pending)


if __name__ == '__main__':
    unittest.main()
