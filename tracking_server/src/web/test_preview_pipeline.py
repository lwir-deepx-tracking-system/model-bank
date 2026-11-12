"""PC-only preview isolation tests: no detector, camera, serial or HTTP server."""
from collections import deque
from contextlib import ExitStack
from dataclasses import replace
from threading import Condition, Event, Lock
from types import SimpleNamespace as NS
import os
import io
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ..common.state import APP
from ..camera import camera_link
from . import page_handler as page
from .preview_worker import PreviewFrame, PreviewWorker, render_preview


def make_live():
    return NS(arr=deque(), raws=deque(), rawq=deque(), stabq=deque(),
              done_ts=deque(maxlen=40), meas=None, meas_lock=Lock(),
              perf_acc=[time.perf_counter(), 0, 0., 0., 0.], perf=deque(maxlen=900),
              dlock=Lock(), cam_lock=Lock(), jcond=Condition(), delays=deque(maxlen=900),
              viewers=1, pic_t=-1e9, pic_n=0, snap_t=-1e9, jpeg=b'previous',
              seq=0, jpeg_frame_t=None, cam_fps=25., t0=time.monotonic(),
              cam_lost=False, cam_hold=False, stopped=False, replay=None,
              restart_pending=False, restart_stop=False, camera_bytetrack=None)


def frame(stamp=None, value=75):
    gray = np.full((480, 640), value, np.uint8)
    gray.setflags(write=False)
    return PreviewFrame(time.perf_counter() if stamp is None else stamp, gray, 1.,
                        (((70, 90), (130, 270), (255, 100, 30), 7),),
                        ((65., 85., 135., 275., 0),), (), 'snapshot', False, None, (0, 0, 0))


class SlowCV:
    """Block only the encoder, after the render thread has entered it."""
    def __init__(self):
        self.entered, self.release = Event(), Event()

    def __getattr__(self, name):
        return getattr(cv2, name)

    def imencode(self, *args):
        self.entered.set()
        if not self.release.wait(3):
            raise TimeoutError('test did not release encoder')
        return cv2.imencode(*args)


class PreviewWorkerTests(unittest.TestCase):
    def worker(self, encoder=cv2):
        live = make_live()
        worker = PreviewWorker(live, encoder, transition_jpeg=b'transition placeholder')
        def close():
            if isinstance(encoder, SlowCV): encoder.release.set()
            worker.close()
            worker.join(3)
        self.addCleanup(close)
        return live, worker

    def wait_seq(self, live, n):
        with live.jcond:
            self.assertTrue(live.jcond.wait_for(lambda: live.seq >= n, timeout=3))

    def test_same_image_bytes_as_synchronous_renderer(self):
        live, worker = self.worker()
        sample = frame()
        expected = render_preview(cv2, sample)
        worker.submit(worker.token(sample.frame_t), sample)
        self.wait_seq(live, 1)
        self.assertEqual(live.jpeg, expected)
        self.assertEqual(live.jpeg_frame_t, sample.frame_t)
        self.assertEqual((live.pic_n, len(live.delays)), (1, 1))

    def test_pending_queue_keeps_only_latest_frame(self):
        encoder = SlowCV()
        live, worker = self.worker(encoder)
        first = frame()
        worker.submit(worker.token(first.frame_t), first)
        self.assertTrue(encoder.entered.wait(2))
        for i in range(30):
            latest = frame(value=100 + i)
            worker.submit(worker.token(latest.frame_t), latest)
        self.assertEqual(live.seq, 0)
        self.assertEqual(worker.stats()['preview_dropped'], 29)
        encoder.release.set()
        self.wait_seq(live, 2)
        self.assertEqual(live.jpeg, render_preview(cv2, latest))
        self.assertEqual(live.pic_n, 2)

    def test_invalidation_during_encode_cannot_replace_placeholder(self):
        encoder = SlowCV()
        live, worker = self.worker(encoder)
        sample = frame()
        token = worker.token(sample.frame_t)
        worker.submit(token, sample)
        self.assertTrue(encoder.entered.wait(2))
        worker.invalidate(before=time.perf_counter())
        with live.jcond:
            live.jpeg = b'camera resetting'
            live.jpeg_frame_t = None
        encoder.release.set()
        deadline = time.monotonic() + 3
        while worker.stats()['preview_dropped'] < 1 and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(worker.stats()['preview_dropped'], 1)
        self.assertEqual(live.jpeg, b'camera resetting')
        self.assertEqual((live.seq, live.pic_n, len(live.delays)), (1, 0, 0))
        self.assertIsNone(worker.token(sample.frame_t))
        current = frame(value=160)
        worker.submit(worker.token(current.frame_t), current)
        self.wait_seq(live, 2)
        self.assertEqual(live.jpeg, render_preview(cv2, current))

    def test_old_generation_cannot_submit_after_switch(self):
        live, worker = self.worker()
        old = frame()
        token = worker.token(old.frame_t)
        worker.invalidate(before=time.perf_counter())
        worker.submit(token, old)
        current = frame(value=205)
        worker.submit(worker.token(current.frame_t), current)
        self.wait_seq(live, 2)
        self.assertEqual(live.jpeg, render_preview(cv2, current))

    def test_failed_encode_does_not_increment_sequence_or_latency(self):
        encoder = SlowCV()
        live, worker = self.worker(encoder)
        sample = frame()
        with patch.object(encoder, 'imencode', return_value=(False, None)):
            worker.submit(worker.token(sample.frame_t), sample)
            deadline = time.monotonic() + 3
            while worker.stats()['preview_errors'] == 0 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(worker.stats()['preview_errors'], 1)
        self.assertEqual(live.jpeg, b'previous')
        self.assertEqual((live.seq, live.pic_n, len(live.delays)), (0, 0, 0))
        sample = frame(value=90)
        encoder.release.set()
        worker.submit(worker.token(sample.frame_t), sample)
        self.wait_seq(live, 1)

    def test_close_prevents_late_publication(self):
        encoder = SlowCV()
        live, worker = self.worker(encoder)
        sample = frame()
        worker.submit(worker.token(sample.frame_t), sample)
        self.assertTrue(encoder.entered.wait(2))
        worker.close()
        encoder.release.set()
        worker.join(3)
        self.assertEqual(live.seq, 0)
        self.assertIsNone(worker.token(time.perf_counter()))

    def test_invalidation_replaces_cached_image_immediately(self):
        live, worker = self.worker()
        live.jpeg_frame_t=time.perf_counter()
        live.jpeg_people=((7,10.,10.,40.,40.),)
        live.jpeg_tracks_ticket=1
        worker.invalidate(before=time.perf_counter())
        self.assertEqual(live.jpeg,b'transition placeholder')
        self.assertIsNone(live.jpeg_frame_t)
        self.assertIsNone(live.jpeg_people)
        self.assertEqual((live.seq,live.pic_n,len(live.delays)),(1,0,0))

    def test_replay_mode_rejects_delayed_live_image(self):
        encoder = SlowCV()
        live, worker = self.worker(encoder)
        sample = frame()
        worker.submit(worker.token(sample.frame_t), sample)
        self.assertTrue(encoder.entered.wait(2))
        live.replay = {'name': 'new replay'}
        worker.invalidate(before=time.perf_counter())
        encoder.release.set()
        deadline = time.monotonic() + 3
        while worker.stats()['preview_dropped'] < 1 and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(worker.stats()['preview_dropped'], 1)
        self.assertEqual(live.jpeg, b'transition placeholder')
        self.assertEqual((live.pic_n, len(live.delays)), (0, 0))
        self.assertIsNone(worker.token(time.perf_counter()))

    def test_close_clears_click_identity(self):
        live, worker = self.worker()
        live.jpeg_people = ((7, 10., 10., 40., 40.),)
        live.jpeg_tracks_ticket = 4
        live.jpeg_frame_t = time.perf_counter()
        worker.close()
        self.assertIsNone(live.jpeg_people)
        self.assertIsNone(live.jpeg_tracks_ticket)
        self.assertIsNone(live.jpeg_frame_t)


class PublishFixture(unittest.TestCase):
    def setUp(self):
        saved = APP.__dict__.copy()
        self.addCleanup(lambda: (APP.__dict__.clear(), APP.__dict__.update(saved)))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.live = make_live()
        APP.LIVE = self.live
        APP.LIVE_H, APP.LIVE_W = 480, 640
        APP.args = NS(live_scale=1., live_seconds=3600., live_wait=0, mamba=False)
        APP.cv2 = cv2
        APP.GB = NS(state=dict(id=0, auto=0, running=True, ready=True, error=0, lost=0, moving=0),
                    gui=False, wcond=Condition(), wopen=True, selection_reset=False,
                    people=[], people_t=None, tracks_ticket=3)
        APP.relabeler = None
        APP.rows, APP.preds, APP.tracks_out = [], [], []
        self.controls, self.evaluations = [], []
        def control(people, *args, **kwargs):
            self.controls.append(tuple(people))
            with APP.GB.wcond:
                APP.GB.people=list(people)
                APP.GB.people_t=args[1]
            return APP.GB.state['id']
        def evaluation(stamp, rows, *args):
            self.evaluations.append(np.array(rows, copy=True))
            return None, None
        for target, attr, value in ((page.live_tracks, 'publish', control),
                                    (page.live_tracks, 'current_source', lambda: ('fixture',)),
                                    (page.replay_eval, 'on_frame', evaluation),
                                    (page.replay_eval, 'is_repeat', lambda _: False),
                                    (page.replay_link, 'frame_blocked', lambda _: False)):
            self.stack.enter_context(patch.object(target, attr, value))
        self.stack.enter_context(patch.dict(page.POSP, show=False))

    def enable(self, enabled, encoder=cv2):
        APP.cv2 = encoder
        with patch.dict(os.environ, LWIR_ASYNC_PREVIEW='1' if enabled else '0'), \
             patch.object(page.preview_worker,'transition_jpeg',return_value=b'transition placeholder'):
            page.init_preview(self.live)
        worker = getattr(self.live, 'preview_worker', None)
        if worker is not None:
            def close():
                if isinstance(encoder, SlowCV): encoder.release.set()
                worker.close()
                worker.join(3)
            self.addCleanup(close)
        return worker

    def publish(self, rows=None, gray=None):
        gray = np.full((512, 640), 75, np.uint8) if gray is None else gray
        rows = np.array([[100, 116, 140, 276, 7, .9, 0]], np.float32) if rows is None else rows
        stamp = time.perf_counter() - .01
        raw = np.full((480, 640), 100 + len(self.controls), np.uint16)
        self.live.arr.append(stamp)
        self.live.raws.append(raw)
        self.live.rawq.append(raw)
        self.live.stabq.append(None)
        page.live_publish(gray, rows, dict(frame_total_ms=20., image_id=1),
                          detections=rows, tracker_detections=rows)
        return stamp

    def install_replay(self):
        self.live.replay = {'name': 'fixture'}
        self.live.proc = object()
        page.preview_worker.set_preview_producer(
            self.live, self.live.proc, before=time.perf_counter() - .1)


class PublishTests(PublishFixture):
    def test_snapshot_after_stop_does_not_return_cached_camera_image(self):
        APP.lv_http = NS(BaseHTTPRequestHandler=object)
        handler = page.make_lv_Handler()()
        handler.path = '/snap.jpg'
        handler.wfile = io.BytesIO()
        errors = []
        handler.send_error = errors.append
        self.live.stopped = True
        handler.do_GET()
        self.assertEqual(errors, [503])
        self.assertEqual(handler.wfile.getvalue(), b'')

    def test_replay_stays_synchronous_with_async_option_enabled(self):
        worker = self.enable(True)
        self.install_replay()
        initial_seq = self.live.seq
        with patch.object(page.replay_link, 'top_text', return_value='REPLAY fixture'), \
             patch.object(worker, 'submit', side_effect=AssertionError('replay must not queue')):
            self.publish()
        self.assertEqual((self.live.seq - initial_seq, len(self.controls), len(self.evaluations)), (1, 1, 1))
        self.assertIsNone(worker._thread)

    def test_replay_source_change_during_encoding_rejects_old_output(self):
        worker = self.enable(True)
        self.install_replay()
        original_encode = cv2.imencode
        def switch_source(*args):
            self.live.replay = {'name': 'new replay'}
            worker.invalidate(before=time.perf_counter())
            return original_encode(*args)
        with patch.object(page.replay_link, 'top_text', return_value='REPLAY fixture'), \
             patch.object(cv2, 'imencode', side_effect=switch_source):
            self.publish()
        self.assertEqual(self.live.jpeg, b'transition placeholder')
        self.assertEqual((self.live.pic_n, len(self.live.delays)), (0, 0))

    def test_first_replay_frame_displays_while_transition_cleanup_is_busy(self):
        for enabled in (False, True):
            with self.subTest(async_option=enabled):
                worker = self.enable(enabled)
                self.install_replay()
                self.live.cam_hold = self.live.replay_busy = True
                seq, pics = self.live.seq, self.live.pic_n
                with patch.object(page.replay_link, 'top_text', return_value='REPLAY fixture'):
                    self.publish()
                self.assertEqual((self.live.seq-seq, self.live.pic_n-pics), (1, 1))
                self.live.cam_hold = self.live.replay_busy = False

    def test_retired_replay_producer_cannot_publish_while_old_process_still_exists(self):
        for enabled in (False, True):
            with self.subTest(async_option=enabled):
                self.enable(enabled)
                self.install_replay()
                self.live.cam_hold = self.live.replay_busy = True
                page.preview_worker.set_preview_producer(self.live, None, before=time.perf_counter() - .1)
                pics = self.live.pic_n
                with patch.object(page.replay_link, 'top_text', return_value='REPLAY fixture'):
                    self.publish()
                self.assertEqual(self.live.pic_n, pics)
                self.live.cam_hold = self.live.replay_busy = False

    def test_default_synchronous_path_is_retained(self):
        self.assertIsNone(self.enable(False))
        self.publish()
        self.assertEqual((self.live.seq, len(self.controls), len(self.evaluations)), (1, 1, 1))

    def test_slow_preview_does_not_drop_or_block_control_or_gt_updates(self):
        encoder = SlowCV()
        worker = self.enable(True, encoder)
        self.publish()
        self.assertTrue(encoder.entered.wait(2))
        for i in range(1, 20):
            self.publish(rows=np.array([[100+i, 116, 140+i, 276, 7, .9, 0]], np.float32))
        self.assertEqual((len(self.controls), len(self.evaluations)), (20, 20))
        self.assertEqual(self.live.seq, 0)
        self.assertEqual(worker.stats()['preview_dropped'], 18)
        self.assertEqual([r[0][1] for r in self.controls], list(range(100, 120)))
        self.assertEqual([r[0, 0] for r in self.evaluations], list(range(100, 120)))
        encoder.release.set()
        with self.live.jcond:
            self.assertTrue(self.live.jcond.wait_for(lambda: self.live.seq == 2, timeout=3))

    def test_async_snapshot_owns_image_and_annotations(self):
        encoder = SlowCV()
        worker = self.enable(True, encoder)
        gray = np.full((512, 640), 75, np.uint8)
        rows = np.array([[100, 116, 140, 276, 7, .9, 0]], np.float32)
        self.publish()
        self.assertTrue(encoder.entered.wait(2))
        self.publish(rows, gray)
        gray[:] = 255
        rows[:] = 0
        encoder.release.set()
        with self.live.jcond:
            self.assertTrue(self.live.jcond.wait_for(lambda: self.live.seq == 2, timeout=3))
        image = cv2.imdecode(np.frombuffer(self.live.jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image[350, 350].tolist(), [75, 75, 75])
        self.assertGreater(int(image[100:104, 100:140].max()), 100)
        self.assertEqual(self.controls[-1][0][0], 7)

    def test_web_click_carries_the_published_image_identity(self):
        self.enable(True)
        self.live.jpeg_frame_t = time.perf_counter()
        self.live.jpeg_people = ((7, 100., 100., 140., 260.),)
        self.live.jpeg_tracks_ticket = 3
        with patch.object(page.live_tracks, 'click', return_value=(True,'')) as click:
            self.assertTrue(page._click_live(110., 130.)[0])
            click.assert_called_once_with(110., 130., expected_id=7, expected_ticket=3)

    def test_old_or_missing_preview_rejects_click_without_command(self):
        self.enable(True)
        self.live.jpeg_people = ((7, 100., 100., 140., 260.),)
        self.live.jpeg_tracks_ticket = 3
        for stamp in (None, time.perf_counter()-2):
            self.live.jpeg_frame_t = stamp
            with patch.object(page.live_tracks, 'click') as click:
                self.assertFalse(page._click_live(110.,130.)[0])
                click.assert_not_called()

    def test_live_identity_changed_since_jpeg_cannot_select_another_person(self):
        gb=APP.GB
        gb.wcond=Condition(); gb.wopen=True; gb.track_control_supported=True
        gb.selection_reset=False; gb.people_source=('fixture',); gb.people_t=time.perf_counter()
        gb.people=[(8,100.,100.,140.,260.,.9)]; gb.tracks_ticket=3
        gb.wq=deque(); gb.wframe=b'F fixture\n'
        with patch.object(page.live_tracks,'reset_lead'):
            self.assertFalse(page.live_tracks.click(110,130,expected_id=7,expected_ticket=3)[0])
            self.assertEqual(list(gb.wq),[])
            self.assertEqual(gb.wframe,b'F fixture\n')

    def test_new_source_ticket_rejects_even_matching_numeric_id(self):
        gb=APP.GB
        gb.wcond=Condition(); gb.wopen=True; gb.track_control_supported=True
        gb.selection_reset=False; gb.people_source=('fixture',); gb.people_t=time.perf_counter()
        gb.people=[(7,100.,100.,140.,260.,.9)]; gb.tracks_ticket=4
        gb.wq=deque(); gb.wframe=b'F fixture\n'
        self.assertFalse(page.live_tracks.click(110,130,expected_id=7,expected_ticket=3)[0])
        self.assertEqual(list(gb.wq),[])

    def test_missing_timestamp_is_not_a_zero_delay_measurement(self):
        self.enable(False)
        self.live.perf_acc[0]=time.perf_counter()-2
        gray=np.zeros((512,640),np.uint8)
        rows=np.empty((0,7),np.float32)
        page.live_publish(gray,rows,dict(frame_total_ms=20.,image_id=1),detections=rows,tracker_detections=rows)
        self.assertIsNone(self.live.perf[-1][3])
        self.assertIsNone(self.live.perf[-1][4])
        self.assertEqual(self.live.perf[-1][6],0)
        self.assertEqual(len(self.live.delays),0)

    def test_retired_camera_reader_cannot_enqueue_buffered_frame(self):
        self.live.cond=Condition(); self.live.q=deque(maxlen=3)
        self.live.q_drop=0; self.live.frames_in=0; self.live.rec=None
        self.live.proc=object()
        old=NS(stdout=io.BytesIO(bytes(640*480*2)))
        with patch.object(camera_link.camera_extra,'stamped',return_value=False):
            camera_link.lv_reader(old)
        self.assertEqual((len(self.live.q),self.live.frames_in,self.live.q_drop),(0,0,0))
        self.assertFalse(self.live.cam_lost)

    def test_current_camera_reader_keeps_identical_raw_pixels(self):
        self.live.cond=Condition(); self.live.q=deque(maxlen=3)
        self.live.q_drop=0; self.live.frames_in=0; self.live.rec=None
        raw=np.arange(640*480,dtype=np.uint16).reshape(480,640)
        current=NS(stdout=io.BytesIO(raw.tobytes()))
        self.live.proc=current
        with patch.object(camera_link.camera_extra,'stamped',return_value=False):
            camera_link.lv_reader(current)
        self.assertEqual((len(self.live.q),self.live.frames_in,self.live.q_drop),(1,1,0))
        self.assertTrue(np.array_equal(self.live.q[0][1],raw))


if __name__ == '__main__':
    unittest.main()
