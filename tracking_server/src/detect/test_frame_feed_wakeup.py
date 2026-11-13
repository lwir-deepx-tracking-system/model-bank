from collections import deque
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from . import frame_feed


class FeedWakeupTests(unittest.TestCase):
    def test_queued_frame_reaches_second_core_before_first_result(self):
        producer_parked = threading.Event()
        core0_waiting = threading.Event()
        core0_release = threading.Event()
        core1_submitted = threading.Event()

        class NotifyingCondition(threading.Condition):
            def wait(self, timeout=None):
                if threading.current_thread().name == 'npu-feed':
                    producer_parked.set()
                    # Test the notification itself, independent of the fallback poll.
                    timeout = None
                return super().wait(timeout)

        class Engine:
            def __init__(self, core):
                self.core = core
                self.name = str(core)
                self.b = self
                self.packed = np.empty(1, np.float32)

            def run_async(self, image, *, user_arg, output_buffer):
                if self.core == 1:
                    core1_submitted.set()
                return user_arg['f']

            def wait(self):
                job = APP.pending.pop(self.name)
                if self.core == 0:
                    core0_waiting.set()
                    if not core0_release.wait(5):
                        raise TimeoutError('core 0 was not released')
                return [np.array([job], np.float32)]

        def submit_tiles(image, seq):
            APP.a.submit(image, seq)
            return seq

        pixels = np.zeros((480, 640), dtype=np.uint8)
        live = SimpleNamespace(
            cond=NotifyingCondition(), q=deque([(1.0, pixels), (1.03, pixels)]),
            stopped=False, cam_lost=False, prev_t=None, last=None,
            stabq=deque(), arr=deque(), rawq=deque(),
        )
        result = []
        errors = []

        def consume():
            try:
                for seq in range(2):
                    feed.read_frame(seq)
                    result.append(int(feed.wait()[0][0]))
            except Exception as exc:
                errors.append(exc)

        with (patch.dict(vars(APP), dict(
                LIVE=live, a=Engine(0), FEED_MODE='arrival', GB=None,
                args=SimpleNamespace(pipeline=True, gmc_thread=False, live_wait=0),
                pending={}, lv_to8=lambda raw: raw, FRAME_DY=[0.0],
        ), clear=True), patch.dict(frame_feed.STAB, on=False),
              patch.object(frame_feed, 'stab_off', return_value=None),
              patch.object(frame_feed.camera_extra.replay_eval, 'lossless', return_value=True),
              patch.object(frame_feed.camera_extra.replay_eval, 'reset_first_input', return_value=False),
              patch.object(frame_feed.detect_frame, 'submit_tiles', side_effect=submit_tiles)):
            feed = frame_feed.Feed(Engine(1))
            consumer = threading.Thread(target=consume, daemon=True)
            try:
                self.assertTrue(producer_parked.wait(2), 'producer never waited on the full slot')
                with live.cond:
                    self.assertEqual(len(feed.out), 1)
                    self.assertEqual(len(live.q), 1)
                    self.assertEqual(feed.busy, [True, False])
                consumer.start()
                self.assertTrue(core0_waiting.wait(2), 'consumer never entered the first SDK wait')
                self.assertTrue(core1_submitted.wait(2), 'second core slept with a queued frame and a free slot')
            finally:
                core0_release.set()
                if consumer.ident is not None:
                    consumer.join(2)
                feed._stop_thread()
            self.assertFalse(consumer.is_alive())
            self.assertFalse(feed.th.is_alive())
            self.assertFalse(errors)
            self.assertIsNone(feed.err)
            self.assertEqual(result, [0, 1])
            self.assertEqual(feed.st['core'], [1, 1])
            self.assertEqual(feed.st['overlap'], 1)
            self.assertEqual(feed.st['order_bad'], 0)
