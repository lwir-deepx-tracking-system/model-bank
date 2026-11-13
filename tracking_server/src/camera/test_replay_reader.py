from collections import deque, OrderedDict
from contextlib import ExitStack
import io
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..common.state import APP
from . import camera_extra, replay_cam, replay_eval


class ObservedLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.attempted = threading.Event()

    def acquire(self, *args, **kwargs):
        if threading.current_thread().name == 'replay-reader-test':
            self.attempted.set()
        return self.lock.acquire(*args, **kwargs)

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *_):
        self.release()


class ObservedCondition(threading.Condition):
    def __init__(self):
        super().__init__()
        self.reader_waiting = threading.Event()

    def wait(self, timeout=None):
        if threading.current_thread().name == 'replay-reader-test':
            self.reader_waiting.set()
        return super().wait(timeout)


class ReplayReaderTests(unittest.TestCase):
    def setUp(self):
        self.live = SimpleNamespace(
            q=deque(maxlen=1), cond=ObservedCondition(), jcond=threading.Condition(),
            stopped=False, replay_src_t=None, replay={}, cam_lost=False,
            q_drop=0, frames_in=0, last_frame_t=0, rec=None,
            cam_stamp=dict(missed=0, seq=0, frames=0, age_bad=0), cam_stamp_recent=deque(),
        )
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(vars(APP), dict(LIVE=self.live, LIVE_W=1, LIVE_H=1)))
        stack.enter_context(patch.multiple(replay_cam, W=1, H=1, FRAME_BYTES=2))
        stack.enter_context(patch.object(replay_eval, '_RT', OrderedDict()))
        stack.enter_context(patch.dict(camera_extra._ACC))
        stack.enter_context(patch.dict(camera_extra._S))
        images = Path(stack.enter_context(TemporaryDirectory()))
        (images / '000017.png').touch()
        (images / '000023.png').touch()
        with patch.object(replay_eval, 'load_labels', return_value=dict(ok=True, gt={}, frames={17, 23})):
            self.ev = replay_eval.Eval(dict(name='fixture', fps=30, loop=False, dir=str(images), frames=3))
        self.assertTrue(self.ev.lab['ok'])
        self.errors = []

    def start_reader(self, frames):
        data = io.BytesIO(b''.join(
            replay_cam.header(seq, time.monotonic_ns(), 33333, flags=flags, slot=slot,
                              ring_id=(pass_index << 20) | png) + struct.pack('=H', png)
            for seq, (flags, slot, pass_index, png) in enumerate(frames, 1)
        ))
        producer = SimpleNamespace(stdout=data, lwir_eval=self.ev)
        camera_extra.install_producer(producer)

        def read():
            try:
                camera_extra._reader(producer, [None])
            except BaseException as error:
                self.errors.append(error)

        thread = threading.Thread(target=read, name='replay-reader-test', daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        return thread

    def check_reader(self, thread):
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.errors, [])

    def test_evaluation_tag_wait_leaves_input_queue_available(self):
        self.ev.lock = ObservedLock()
        with self.ev.lock:
            thread = self.start_reader([(2, 0, 0, 17)])
            self.assertTrue(self.ev.lock.attempted.wait(3))
            self.assertTrue(self.live.cond.acquire(timeout=0.5), 'waiting for the tag held the input queue lock')
            try:
                self.assertFalse(self.live.q, 'a frame reached the queue without its tag')
            finally:
                self.live.cond.release()
        self.check_reader(thread)
        stamp, pixels = self.live.q[0]
        self.assertEqual((self.ev.tags[stamp], pixels.item()), ((17, 1, 0), 17))
        self.assertTrue(self.live.cam_lost)

    def test_superseded_source_cannot_tag_or_enqueue_after_wait(self):
        self.ev.lock = ObservedLock()
        # A source switch clears the queue before it closes the old Eval and installs the next producer.
        with self.ev.lock:
            thread = self.start_reader([(6, 0, 0, 17), (2, 0, 0, 17)])
            self.assertTrue(self.ev.lock.attempted.wait(3))
            self.assertTrue(self.live.cond.acquire(timeout=0.5))
            try:
                self.live.replay_src_t = time.perf_counter()
                self.live.q.clear()
            finally:
                self.live.cond.release()
        self.check_reader(thread)
        self.assertFalse(self.ev.tags or self.ev.rep or replay_eval._RT or self.live.q)
        self.assertEqual((self.ev.n['received'], self.ev.n['repeats'], self.live.frames_in, self.live.last_frame_t), (0, 0, 0, 0))

    def test_full_queue_preserves_tags_order_seek_repeat_and_eof(self):
        thread = self.start_reader([(2, 0, 0, 17), (2, 1, 0, 23), (2, 0, 1, 17), (6, 0, 1, 17)])
        output = []
        for index, tag in enumerate(((17, 1, 0), (23, 1, 1))):
            self.assertTrue(self.live.cond.reader_waiting.wait(3))
            with self.live.cond:
                self.live.cond.reader_waiting.clear()
                self.assertEqual(len(self.live.q), 1)
                self.assertFalse(self.live.cam_lost)
                self.assertEqual(self.ev.n['received'], index + 1)
                stamp, pixels = self.live.q.popleft()
                output.append((stamp, pixels.item()))
                self.assertEqual(self.ev.tags[stamp], tag)
                if index == 1:
                    self.assertEqual(self.ev.want_seek(0, stamp), 1)
                self.live.cond.notify_all()
        self.check_reader(thread)
        self.assertTrue(self.live.cam_lost)
        self.assertEqual((self.live.frames_in, self.live.q_drop), (3, 0))
        stamp, pixels = self.live.q.popleft()
        output.append((stamp, pixels.item()))
        self.assertEqual([self.ev.tags[t] for t, _ in output], [(17, 1, 0), (23, 1, 1), (17, 2, 0)])
        self.assertEqual([pixel for _, pixel in output], [17, 23, 17])
        self.assertEqual(list(self.ev.rep.values()), [(17, 2, 0)])
        self.assertEqual(self.ev.n['repeats'], 1)
        self.ev.frame(stamp, [], None, 'post-seek')
        self.assertIsNone(self.ev.position()['seek_to'])
