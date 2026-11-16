"""11-14: with the end-to-end model on, the arrival feed queues frames without the detector, the frame loop does not wait for a
result, and nothing stalls (device hang of 868d70c: the loop skipped wait(), the engine never became free, no frame came).
Run from tracking_server/: python -m unittest src.detect.test_frame_feed_e2e"""
from collections import deque
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from ..track import e2e_live
from . import frame_feed


class Engine:
    def __init__(self, core):
        self.core = core; self.name = str(core); self.b = self; self.packed = np.empty(1, np.float32); self.runs = 0

    def run_async(self, image, *, user_arg, output_buffer):
        self.runs += 1; return user_arg['f']

    def wait(self):
        return [np.array([APP.pending.pop(self.name)], np.float32)]


class FeedE2ETests(unittest.TestCase):
    def test_frames_flow_without_detector_then_with_it_again(self):
        pixels = np.zeros((480, 640), dtype=np.uint8)
        live = SimpleNamespace(cond=threading.Condition(), q=deque((1.0 + k * 0.04, pixels) for k in range(6)),
                               stopped=False, cam_lost=False, prev_t=None, last=None, stabq=deque(), arr=deque(), rawq=deque())
        e0, e1 = Engine(0), Engine(1)

        def submit_tiles(image, seq):
            APP.a.submit(image, seq); return seq

        own = [True]
        with (patch.dict(vars(APP), dict(LIVE=live, a=e0, FEED_MODE='arrival', GB=None,
                                         args=SimpleNamespace(pipeline=True, gmc_thread=False, live_wait=0),
                                         pending={}, lv_to8=lambda raw: raw, FRAME_DY=[0.0]), clear=True),
              patch.dict(frame_feed.STAB, on=False), patch.object(frame_feed, 'stab_off', return_value=None),
              patch.object(frame_feed.camera_extra.replay_eval, 'lossless', return_value=True),
              patch.object(frame_feed.camera_extra.replay_eval, 'reset_first_input', return_value=False),
              patch.object(frame_feed.detect_frame, 'submit_tiles', side_effect=submit_tiles),
              patch.object(e2e_live, 'skip_detector', side_effect=lambda: own[0])):
            feed = frame_feed.Feed(e1); got = []
            try:
                for k in range(3):   # end-to-end on: no detector, the loop does not call wait()
                    g, _ = feed.read_frame(k); self.assertIsNotNone(g)
                    self.assertFalse(feed.cur_submitted()); got.append(None); feed.cur = None
                self.assertEqual(e0.runs + e1.runs, 0); self.assertEqual(feed.busy, [False, False])
                own[0] = False
                with live.cond: live.cond.notify_all()
                for k in range(3, 6):   # off again: frames go to the detector (one queued frame may still be a skipped one)
                    g, _ = feed.read_frame(k); self.assertIsNotNone(g)
                    got.append(int(feed.wait()[0][0]) if feed.cur_submitted() else None)
                    if not feed.cur_submitted(): feed.cur = None
            finally:
                feed._stop_thread()
            self.assertIsNone(feed.err)
            self.assertGreaterEqual(e0.runs + e1.runs, 2)
            self.assertEqual(feed.st['order_bad'], 0)
            self.assertGreaterEqual(feed.st.get('e2e_skip', 0), 3)


if __name__ == '__main__':
    unittest.main()
