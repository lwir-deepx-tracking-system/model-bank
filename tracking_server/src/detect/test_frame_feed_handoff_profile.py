"""Bounded split-feed diagnostics; no camera, DX-RT, or scheduler tuning."""
from collections import deque
from contextlib import ExitStack
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from . import frame_feed
from .tile_job_profile import TileJobProfile


class HandoffProfileTests(unittest.TestCase):
    def run_feed(self, active):
        profile = TileJobProfile(Path('.'), None)
        if active:
            profile.capture = {'started_unix_ns': 1, 'frames': []}
        else:
            profile.until_poll = 1000
        saved, written = [], threading.Event()
        consumer_id, consumer_waiting = threading.get_ident(), threading.Event()

        class Condition(threading.Condition):
            def wait(self, timeout=None):
                if threading.get_ident() == consumer_id:
                    consumer_waiting.set()
                return super().wait(timeout)

        def write(path, capture):
            saved.append(json.loads(json.dumps(capture)))
            written.set()

        class Engine:
            name = 'tiles'
            packed = np.empty(0, np.float32)

            def __init__(self):
                self.b = self
                self.profile = profile

            def run_async(self, image, *, user_arg, output_buffer):
                seq = user_arg['f']
                self.trace = profile.begin(seq)
                return seq

            def wait(self):
                seq = APP.pending.pop(self.name)
                if active and seq == 0 and not consumer_waiting.wait(2):
                    raise TimeoutError('consumer did not enter the condition wait')
                if self.trace is not None:
                    profile.completed = seq, self.trace
                return [np.array([seq], np.float32)]

        pixels = np.zeros((480, 640), np.uint8)
        live = SimpleNamespace(
            cond=Condition(), q=deque((1 + k * .03, pixels) for k in range(3)),
            stopped=False, cam_lost=False, prev_t=None, last=None,
            stabq=deque(), arr=deque(), rawq=deque(), q_drop=0,
        )
        tile = Engine()

        def submit(image, seq):
            APP.a.submit(image, seq)
            return seq

        with ExitStack() as stack:
            stack.enter_context(patch.dict(vars(APP), dict(
                LIVE=live, a=tile, FEED_MODE='tile', GB=None,
                args=SimpleNamespace(pipeline=True, gmc_thread=False, live_wait=0),
                pending={}, lv_to8=lambda raw: raw, FRAME_DY=[0.0],
            ), clear=True))
            stack.enter_context(patch.dict(frame_feed.STAB, on=False))
            stack.enter_context(patch.object(frame_feed, 'stab_off', return_value=None))
            stack.enter_context(patch.object(frame_feed.dual_tile, 'assets_reason', return_value=None))
            stack.enter_context(patch.object(frame_feed.dual_tile, 'selected', return_value=True))
            stack.enter_context(patch.object(frame_feed.dual_tile, 'Tone', return_value=lambda raw: raw))
            stack.enter_context(patch.object(frame_feed.dual_tile, 'Engine', return_value=tile))
            stack.enter_context(patch.object(frame_feed.camera_extra.replay_eval, 'lossless', return_value=True))
            stack.enter_context(patch.object(frame_feed.camera_extra.replay_eval, 'reset_first_input', return_value=False))
            stack.enter_context(patch.object(frame_feed.detect_frame, 'submit_tiles', side_effect=submit))
            stack.enter_context(patch('src.detect.tile_job_profile.SAMPLE_FRAMES', 3))
            stack.enter_context(patch.object(profile, '_write', side_effect=write))
            if not active:
                stack.enter_context(patch.object(frame_feed.time, 'thread_time_ns', side_effect=AssertionError('inactive trace clock')))
            feed = frame_feed.Feed(Engine())
            try:
                values = []
                for seq in range(3):
                    feed.read_frame(seq)
                    values.append(int(feed.wait()[0][0]))
                self.assertEqual(values, [0, 1, 2])
                self.assertEqual(feed.st['order_bad'], 0)
                self.assertEqual((feed.st['drop_wait'], live.q_drop), (0, 0))
                self.assertIsNone(feed.err)
                if active:
                    self.assertTrue(written.wait(2))
            finally:
                feed._stop_thread()
        return saved

    def test_profile_contains_last_consumed_frame_and_preserves_order(self):
        saved = self.run_feed(True)
        self.assertEqual(len(saved), 1)
        frames = saved[0]['frames']
        self.assertEqual([frame['frame'] for frame in frames], [0, 1, 2])
        self.assertIsNotNone(frames[0]['tracker_wait_start_ns'])
        self.assertLessEqual(frames[0]['tracker_read_start_ns'], frames[0]['tracker_wait_start_ns'])
        for frame in frames:
            self.assertLessEqual(frame['feed_copy_done_ns'], frame['feed_publish_ns'])
            self.assertLessEqual(frame['feed_publish_ns'], frame['tracker_take_ns'])
            self.assertLessEqual(frame['tracker_take_ns'], frame['tracker_collect_done_ns'])
            self.assertLessEqual(frame['tracker_take_cpu_ns'], frame['tracker_collect_cpu_ns'])
            self.assertFalse(frame['handoff_discarded'])
            if frame['tracker_wait_start_ns'] is not None:
                self.assertLessEqual(frame['tracker_wait_start_ns'], frame['tracker_wake_ns'])
                self.assertLessEqual(frame['tracker_wake_ns'], frame['tracker_take_ns'])
                self.assertLessEqual(frame['tracker_wait_cpu_ns'], frame['tracker_wake_cpu_ns'])

    def test_inactive_profile_adds_no_thread_cpu_clocks(self):
        self.assertEqual(self.run_feed(False), [])

    def test_capture_stays_bounded_until_last_handoff(self):
        profile = TileJobProfile(Path('.'), None)
        trace = {'frame': 0}
        profile.capture = {'started_unix_ns': 1, 'frames': []}
        profile.completed = 'job', trace
        with patch('src.detect.tile_job_profile.SAMPLE_FRAMES', 1), patch.object(profile, 'close') as close:
            self.assertIs(profile.finish_frame('job', 1, 2, 3, handoff=True), trace)
            self.assertIsNone(profile.begin(1))
            close.assert_not_called()
            profile.finish_handoff(trace, discarded=True)
            close.assert_called_once()
            self.assertTrue(trace['handoff_discarded'])


if __name__ == '__main__':
    unittest.main()
