"""Draw a frame snapshot and, optionally, encode it outside the tracker loop."""
from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass
from threading import Event, Thread
import time

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
    cv2.rectangle(img, (0, 0), (ww, 26), (0, 0, 150) if frame.replay else (0, 0, 0), -1)
    cv2.putText(img, frame.top_text, (8, 19), font, 0.55, (255, 255, 255), 1)
    if frame.gimbal_text is not None:
        cv2.rectangle(img, (0, 26), (ww, 50), (0, 0, 0), -1)
        cv2.putText(img, frame.gimbal_text, (8, 44), font, 0.55, frame.gimbal_color, 1)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok and buf is not None else None


def publish_jpeg_locked(live, jpeg, frame_t, *, paired=True, click_people=None, click_ticket=None):
    """Commit one JPEG under jcond. Placeholders have no frame latency sample."""
    live.jpeg = jpeg
    live.jpeg_frame_t = frame_t
    live.jpeg_people = click_people
    live.jpeg_tracks_ticket = click_ticket
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

    def __init__(self, live, cv2, *, transition_jpeg=None):
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

    def _can_publish(self, frame_t, *, replay=False):
        live = self._live
        if (self._closed or live.stopped or frame_t < self._before
                or getattr(live, 'restart_pending', False) or getattr(live, 'restart_stop', False)):
            return False
        if replay:
            return replay_preview_source(live, frame_t) is not None
        return (getattr(live, 'replay', None) is None and not getattr(live, 'cam_hold', False)
                and not getattr(live, 'replay_busy', False))

    def token(self, frame_t, *, replay=False):
        with self._cond:
            return self._generation if self._can_publish(frame_t, replay=replay) else None

    def submit(self, token, frame, *, requested_at=None):
        with self._cond:
            if frame.replay or token != self._generation or not self._can_publish(frame.frame_t):
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
            if requested_at is not None:
                self._live.pic_t = requested_at
            self._wake.set()
            return True

    def commit_replay(self, token, frame_t, jpeg, *, requested_at):
        """Commit a JPEG rendered inline by the replay frame loop."""
        with self._cond:
            if token != self._generation or not self._can_publish(frame_t, replay=True):
                self._dropped += 1
                return False
            self._live.pic_t = requested_at
            if not jpeg:
                self._errors += 1
                return False
            publish_jpeg_locked(self._live, jpeg, frame_t)
            return True

    def _invalidate_locked(self, before):
        self._generation += 1
        if before is not None:
            self._before = max(before, self._before)
        if self._pending is not None:
            self._dropped += 1
            self._pending = None
        self._live.pic_t = float('-inf')
        if not self._closed and not self._live.stopped and self._live.jpeg is not self._transition_jpeg:
            publish_jpeg_locked(self._live, self._transition_jpeg, None)
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
            if self._pending is not None:
                self._dropped += 1
                self._pending = None
            self._wake.set()

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
            try:
                jpeg = render_preview(self._cv2, frame)
            except Exception:
                jpeg = None
            with self._cond:
                if token != self._generation or not self._can_publish(frame.frame_t):
                    self._dropped += 1
                elif not jpeg:
                    self._errors += 1
                else:
                    publish_jpeg_locked(self._live, jpeg, frame.frame_t,
                                        click_people=frame.click_people, click_ticket=frame.click_ticket)
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
