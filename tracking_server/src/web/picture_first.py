"""Picture first (11-15, user: total delay about one camera frame).

The page picture is made from the camera frame as soon as it arrives (in its own thread, before detection, tracking and
the gimbal), and the boxes, ids and the followed person go to the page separately (GET /overlay, server-sent events),
which draws them over the picture (page/js/picture_first.js). The picture leaves the server a few ms after the frame
arrived instead of after detection and tracking (about 57 ms on the device); the boxes lag the picture by about that.
A click still picks from the people of the newest boxes sent to the page (click_people / click_ticket, see
page_handler._click_live), which are the boxes the viewer sees.

Used only for the live camera: during a replay, with image stabilisation, or while people proposals or skin temperatures
are shown, the server draws the picture as before. Predicted boxes of lost people (pred_show) go in the overlay too.
Switch: LWIR_PICTURE_FIRST=1 at start, POST /rules/set?picture_first=1|0 live (not kept over a restart).
"""
import json
import os
import threading
import time

import numpy as np

from ..track.stabilizer import STAB
from ..detect import spec_proposals, skin_temp
from . import preview_worker

PF = dict(on=os.environ.get('LWIR_PICTURE_FIRST') == '1', pictures=0, overlays=0, errors=0, encode_ms=None)


def active(live):
    """True when the page picture comes from this module (and the boxes from the overlay)."""
    return bool(PF['on'] and live is not None and getattr(live, 'replay', None) is None and not STAB['on']
                and not spec_proposals.PROP['show'] and not skin_temp.TEMP['show'])


def set_on(on, live=None):
    PF['on'] = bool(on)
    if live is not None and not active(live):
        with live.jcond:   # pages clear the boxes they drew
            publish_overlay_locked(live, None)


def state(live):
    return dict(on=PF['on'], active=active(live), pictures=PF['pictures'], overlays=PF['overlays'],
                errors=PF['errors'], encode_ms=PF['encode_ms'])


def to8_view(live, raw, fixed=False):
    """The camera frame in 8 bits with the live mapping's current window, without moving it (the tracker's copy does)."""
    if fixed:
        return (raw >> 8).astype(np.uint8)
    lo, hi = getattr(live, 'lo', None), getattr(live, 'hi', None)
    if lo is None or hi is None or hi <= lo:
        return None
    return np.clip((raw.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


class PictureWorker:
    """Newest camera frame -> page picture, in one thread. offer() is called by the camera reader and never blocks."""

    def __init__(self, live, cv2, *, fixed_map=False):
        self.live, self.cv2, self.fixed = live, cv2, fixed_map
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.slot = None
        self.thread = None
        self.last = -1e9   # perf time of the last picture made

    def wanted(self):
        """Every frame while /stream has a viewer or for 3 s after /snap.jpg, else one picture a second (like the drawn picture)."""
        live, now = self.live, time.perf_counter()
        return active(live) and (live.viewers > 0 or now - getattr(live, 'snap_t', -1e9) < 3.0 or now - self.last >= 1.0)

    def offer(self, frame_t, raw):
        if not self.wanted():
            return False
        with self.lock:
            self.slot = (frame_t, raw)   # an older waiting frame is replaced: only the newest picture matters
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='lwir-picture-first', daemon=True)
                self.thread.start()
        self.wake.set()
        return True

    def _run(self):
        live = self.live
        while not live.stopped:
            if not self.wake.wait(1.0):
                continue
            self.wake.clear()
            with self.lock:
                item, self.slot = self.slot, None
            if item is None:
                continue
            frame_t, raw = item
            t0 = time.perf_counter()
            try:
                gray = to8_view(live, raw, self.fixed)
                if gray is None:
                    continue
                ok, buf = self.cv2.imencode('.jpg', gray, [self.cv2.IMWRITE_JPEG_QUALITY, preview_worker.JPEG_QUALITY,
                                                            self.cv2.IMWRITE_JPEG_OPTIMIZE, 1])   # optimised tables: about 12 % fewer bytes for under 1 ms
                jpeg = buf.tobytes() if ok and buf is not None else None
            except Exception:   # noqa: BLE001 (a bad frame must not stop the pictures)
                jpeg = None
            if not jpeg:
                PF['errors'] += 1
                continue
            ms = (time.perf_counter() - t0) * 1000.0
            PF['encode_ms'] = round(ms if PF['encode_ms'] is None else 0.9 * PF['encode_ms'] + 0.1 * ms, 2)
            with live.jcond:
                if not active(live) or live.stopped:
                    continue
                preview_worker.publish_jpeg_locked(live, jpeg, frame_t, click=False, drawn=gray)
                PF['pictures'] += 1
                self.last = time.perf_counter()


def overlay_message(frame_t, boxes, follow, now_perf, now_unix, width=640, height=480, predictions=()):
    """One overlay as JSON bytes: t = the boxes' camera frame time (unix s), b = [x1, y1, x2, y2, id, followed] in picture
    pixels, f = the followed id (0 = none), p = predicted boxes of lost people [x1, y1, x2, y2, id, label, last seen
    [cx, cy, w, h] or null, path [[x, y], ...], search radius, 'rgb(..)'] (page_handler._predictions at scale 1)."""
    p = [[a[0], a[1], b_[0], b_[1], int(tid), str(lab), list(last) if last else None, [list(q) for q in pts], int(rad),
          'rgb(%d,%d,%d)' % (col[2], col[1], col[0])] for a, b_, col, tid, lab, last, pts, rad in predictions]
    return json.dumps(dict(pf=1, t=round(now_unix - (now_perf - frame_t), 4), w=width, h=height, f=int(follow),
                           b=[[int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), int(tid), int(fol)]
                              for x1, y1, x2, y2, tid, fol in boxes], p=p), separators=(',', ':')).encode()


def publish_overlay_locked(live, message, *, people=None, ticket=None, frame_t=None):
    """Under live.jcond. message None = no boxes from this module (the page clears its drawing)."""
    live.overlay = message if message is not None else b'{"pf":0}'
    live.overlay_seq = getattr(live, 'overlay_seq', 0) + 1
    live.click_people, live.click_ticket, live.click_frame_t = people, ticket, frame_t
    if message is not None:
        PF['overlays'] += 1
    live.jcond.notify_all()


def serve_overlay(live, write, keepalive=5.0):
    """GET /overlay: every new overlay as a server-sent event (newest only), a comment line when nothing new for a while."""
    last = None
    while not live.stopped:
        with live.jcond:
            live.jcond.wait_for(lambda: getattr(live, 'overlay_seq', 0) != last or live.stopped, timeout=keepalive)
            if live.stopped:
                return
            seq = getattr(live, 'overlay_seq', 0)
            msg = getattr(live, 'overlay', None) if seq != last else None
            last = seq
        write(b'data: ' + msg + b'\n\n' if msg is not None else b': keepalive\n\n')
