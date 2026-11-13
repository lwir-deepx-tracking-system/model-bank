"""Draw a frame snapshot and, optionally, encode it outside the tracker loop."""
from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass
from threading import Event, Thread
import os
import time

# 11-13 (user, design review): the page shows mode, speed and gimbal state as chips, so the red / black text bars are no
# longer drawn into the picture. LWIR_PICTURE_BANNER=1 draws them again (for viewers of /stream without the page).
PICTURE_BANNER = os.environ.get('LWIR_PICTURE_BANNER') == '1'

import numpy as np


# Valid 8×8 gray JPEG for startup when OpenCV cannot encode the normal placeholder.
_FALLBACK_TRANSITION = b64decode(
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4n'
    'ICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAIAAgBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAAB//E'
    'ABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AD3//2Q=='
)


@dataclass(frozen=True)
class PreviewFrame:
    frame_t: float
    gray: np.ndarray
    scale: float
    boxes: tuple
    gt_boxes: tuple
    predictions: tuple
    top_text: str
    replay: bool
    gimbal_text: str | None
    gimbal_color: tuple
    click_people: tuple | None = None
    click_ticket: int | None = None
    replay_position: tuple | None = None


def transition_jpeg(cv2):
    """Build the source-transition image once when async preview is enabled."""
    try:
        image = np.zeros((480, 640), dtype=np.uint8)
        cv2.putText(image, 'WAITING FOR VIDEO', (167, 242), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 210, 2)
        ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok and buf is not None:
            return buf.tobytes() or _FALLBACK_TRANSITION
    except Exception:
        pass
    return _FALLBACK_TRANSITION


def _dash_rect(cv2, img, p1, p2, col, th, dash):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), col, th)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), col, th)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), col, th)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), col, th)


def render_preview(cv2, frame):
    sc = frame.scale
    img = cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
    if sc != 1:
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for p1, p2, col, tid in frame.boxes:
        cv2.rectangle(img, p1, p2, col, 2)
        cv2.putText(img, str(tid), (p1[0], max(p1[1] - 4, 12)), font, 0.55, col, 2)
    hh, ww = img.shape[:2]
    for x1, y1, x2, y2, kind in frame.gt_boxes:
        a = min(max(int(x1 * sc), 0), ww - 1), min(max(int(y1 * sc), 0), hh - 1)
        b = min(max(int(x2 * sc), 0), ww - 1), min(max(int(y2 * sc), 0), hh - 1)
        if a[0] < b[0] and a[1] < b[1]:
            _dash_rect(cv2, img, a, b, (255, 255, 255) if kind == 0 else (200, 90, 255), 1, 4)
    for p1, p2, col, tid in frame.predictions:
        _dash_rect(cv2, img, p1, p2, col, 2, 7)
        cv2.putText(img, '%d predict' % tid, (p1[0], max(p1[1] - 4, 12)), font, 0.55, col, 2)
    if PICTURE_BANNER:
        cv2.rectangle(img, (0, 0), (ww, 26), (0, 0, 150) if frame.replay else (0, 0, 0), -1)
        cv2.putText(img, frame.top_text, (8, 19), font, 0.55, (255, 255, 255), 1)
    if PICTURE_BANNER and frame.gimbal_text is not None:
        cv2.rectangle(img, (0, 26), (ww, 50), (0, 0, 0), -1)
        cv2.putText(img, frame.gimbal_text, (8, 44), font, 0.55, frame.gimbal_color, 1)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok and buf is not None else None


def publish_jpeg_locked(live, jpeg, frame_t, *, paired=True, click_people=None, click_ticket=None,
                        replay_position=None):
    """Commit one JPEG under jcond. Placeholders have no frame latency sample."""
    live.jpeg = jpeg
    live.jpeg_frame_t = frame_t
    live.jpeg_people = click_people
    live.jpeg_tracks_ticket = click_ticket
    live.jpeg_replay_position = replay_position
    live.seq += 1
    if frame_t is not None:
        live.pic_n += 1
        if paired:
            with live.dlock:
                ready = time.perf_counter()
                live.delays.append((ready, (ready - frame_t) * 1000.0))
    live.jcond.notify_all()


def replay_preview_source(live, frame_t):
    """Return the installed replay boundary for this frame, under jcond."""
    source = getattr(live, 'preview_replay_source', None)
    if source is None or live.stopped:
        return None
    producer, replay, installed_at = source
    if (getattr(live, 'proc', None) is producer and getattr(live, 'replay', None) is replay
            and frame_t >= installed_at):
        return source
    return None


class PreviewWorker:
    """One renderer and a replaceable pending slot, under the JPEG condition."""

    def __init__(self, live, cv2, *, transition_jpeg=None, async_live=True, async_replay=False):
        self._live = live
        self._cv2 = cv2
        self._cond = live.jcond
        self._wake = Event()
        self._pending = self._thread = None
        self._closed = False
        self._generation = 0
        self._before = float('-inf')
        self._transition_jpeg = bytes(transition_jpeg) if transition_jpeg else _FALLBACK_TRANSITION
        self._dropped = self._errors = 0
        self.async_live = async_live
        self.async_replay = async_replay
        self._seek = None
        self._seek_queued = False

    def _can_publish(self, frame_t, *, replay=False):
        live = self._live
        if (self._closed or live.stopped or frame_t < self._before
                or getattr(live, 'restart_pending', False) or getattr(live, 'restart_stop', False)):
            return False
        if replay:
            return replay_preview_source(live, frame_t) is not None
        return (getattr(live, 'replay', None) is None and not getattr(live, 'cam_hold', False)
                and not getattr(live, 'replay_busy', False))

    def token(self, frame_t, *, replay=False, source=None):
        with self._cond:
            if source is not None and getattr(self._live, 'replay', None) is not source:
                return None
            return self._generation if self._can_publish(frame_t, replay=replay) else None

    def submit(self, token, frame, *, requested_at=None):
        with self._cond:
            if frame.replay and not self.async_replay:
                self._dropped += 1
                return False
            if frame.replay:
                # A live preview may be superseded. Replay keeps every accepted job in order.
                self._cond.wait_for(lambda: self._pending is None or token != self._generation
                                    or not self._can_publish(frame.frame_t, replay=True))
            if (token != self._generation or not self._can_publish(frame.frame_t, replay=frame.replay)
                    or (frame.replay and self._seek is not None and not self._seek_queued
                        and not self._seek_matches(frame.replay_position))):
                self._dropped += 1
                return False
            if self._thread is None:
                thread = Thread(target=self._run, name='lwir-web-preview', daemon=True)
                try:
                    thread.start()
                except RuntimeError:
                    self._errors += 1
                    return False
                self._thread = thread
            if self._pending is not None:
                self._dropped += 1
            self._pending = token, frame
            if frame.replay and self._seek_matches(frame.replay_position):
                self._seek_queued = True
            if requested_at is not None:
                self._live.pic_t = requested_at
            self._wake.set()
            return True

    def begin_replay_seek(self, run_id, slot, before, previous_pass, *, replay):
        with self._cond:
            if replay_preview_source(self._live, before) is None or self._live.replay is not replay:
                return
            self._seek = run_id, slot, before, previous_pass
            self._invalidate_locked(before, keep_jpeg=True)

    def replay_state(self, run_id):
        with self._cond:
            position = getattr(self._live, 'jpeg_replay_position', None)
            current = position is not None and position[0] == run_id
            seek_to = self._seek[1] if self._seek is not None and self._seek[0] == run_id else None
            return dict(pos=position[1] if current else None, pass_=position[2] if current else None,
                        seek_to=seek_to)

    def replay_required(self, position, total):
        if position is None:
            return False
        with self._cond:
            return position[1] == total - 1 or self._seek_matches(position)

    def _seek_matches(self, position):
        return (self._seek is not None and position is not None
                and position[:2] == self._seek[:2] and position[3] >= self._seek[2]
                and position[2] > self._seek[3])

    def commit_inline(self, token, frame_t, jpeg, *, requested_at, replay):
        """Commit an inline render only while its source generation is current."""
        with self._cond:
            if token != self._generation or not self._can_publish(frame_t, replay=replay):
                self._dropped += 1
                return False
            self._live.pic_t = requested_at
            if not jpeg:
                self._errors += 1
                return False
            publish_jpeg_locked(self._live, jpeg, frame_t)
            return True

    def _invalidate_locked(self, before, *, keep_jpeg=False):
        self._generation += 1
        if before is not None:
            self._before = max(before, self._before)
        if self._pending is not None:
            self._dropped += 1
            self._pending = None
        self._live.pic_t = float('-inf')
        if not keep_jpeg:
            self._seek = None
        self._seek_queued = False
        if (not keep_jpeg and not self._closed and not self._live.stopped
                and self._live.jpeg is not self._transition_jpeg):
            publish_jpeg_locked(self._live, self._transition_jpeg, None)
        self._cond.notify_all()
        return self._generation

    def invalidate(self, *, before=None):
        with self._cond:
            return self._invalidate_locked(before)

    def close(self):
        with self._cond:
            self._closed = True
            self._live.jpeg = self._transition_jpeg
            self._live.jpeg_frame_t = None
            self._live.jpeg_people = None
            self._live.jpeg_tracks_ticket = None
            self._live.jpeg_replay_position = None
            self._seek = None
            self._seek_queued = False
            if self._pending is not None:
                self._dropped += 1
                self._pending = None
            self._wake.set()
            self._cond.notify_all()

    def publish_placeholder(self, token, jpeg):
        with self._cond:
            if token == self._generation and not self._closed and not self._live.stopped:
                publish_jpeg_locked(self._live, jpeg, None)

    def join(self, timeout):
        if self._thread is not None:
            self._thread.join(timeout)

    def stats(self):
        with self._cond:
            return dict(preview_dropped=self._dropped, preview_errors=self._errors)

    def _run(self):
        while True:
            self._wake.wait()
            with self._cond:
                if self._closed:
                    return
                self._wake.clear()
                if self._pending is None:
                    continue
                token, frame = self._pending
                self._pending = None
                self._cond.notify_all()
                if frame.replay and self._seek is not None and not self._seek_matches(frame.replay_position):
                    self._dropped += 1
                    continue
            try:
                jpeg = render_preview(self._cv2, frame)
            except Exception:
                jpeg = None
            with self._cond:
                if token != self._generation or not self._can_publish(frame.frame_t, replay=frame.replay):
                    self._dropped += 1
                elif not jpeg:
                    self._errors += 1
                    if frame.replay and self._seek_matches(frame.replay_position):
                        self._seek_queued = False
                elif frame.replay and self._seek is not None and not self._seek_matches(frame.replay_position):
                    self._dropped += 1
                else:
                    if frame.replay:
                        self._seek = None
                        self._seek_queued = False
                    publish_jpeg_locked(self._live, jpeg, frame.frame_t,
                                        click_people=frame.click_people, click_ticket=frame.click_ticket,
                                        replay_position=frame.replay_position)
            # Release the completed image while waiting on the next request.
            del frame, jpeg


def invalidate_preview(live, *, before=None):
    worker = getattr(live, 'preview_worker', None)
    if worker is not None:
        return worker.invalidate(before=before)


def set_preview_producer(live, process, *, before):
    """Install or retire a replay preview source without changing control holds."""
    with live.jcond:
        replay = getattr(live, 'replay', None)
        previous_source = getattr(live, 'preview_replay_source', None)
        live.preview_replay_source = (process, replay, before) if process is not None and replay is not None else None
        worker = getattr(live, 'preview_worker', None)
        if worker is not None:
            worker._invalidate_locked(before)
        elif replay is not None or previous_source is not None:
            live.pic_t = float('-inf')
