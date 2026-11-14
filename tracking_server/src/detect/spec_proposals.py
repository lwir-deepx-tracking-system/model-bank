"""People proposals from the spatial frequency content of the picture (2026-11-14, research feature, display only).

Idea (close_range_20261113/fft_liveness): a window that holds a person has a different 2-D power spectrum than warm
things (monitors, laptops, seats): less fine detail (body heat spreads smoothly) and more oblique coarse structure.
A small logistic model on the spectrum (16 rings x 8 orientations, log shares) + the window's warmth was trained on four
indoor GT recordings; on recordings it had not seen, its best window pointed at a person in 83-100 % of the frames,
also where the warmest spot was a laptop.

Here: square windows slide over the 16-bit raw frame of every shown frame (11-14, user: show what the spectrum finds,
also where the detector found people); the best windows above the model's threshold (up to 'top', overlap < 0.3) are
proposals. propose(people=...) can still skip windows on given boxes; the server passes none. They are only drawn on the picture
(dashed-looking magenta box, 'cand' + score); nothing goes to the tracker or the gimbal.

Runs in its own thread with one slot (the newest frame replaces a waiting one), so a slow pass never holds the frame loop.
Switch: POST /rules/set?prop_show=1|0 (kept over a restart like the other display switches)."""
import json
import os
import threading
import time

import numpy as np

SZ = 64
_yy, _xx = np.mgrid[-SZ // 2:SZ // 2, -SZ // 2:SZ // 2]
_RAD = np.hypot(_yy, _xx)
_RB = np.minimum((_RAD / 2).astype(int), 15)
_ANG = ((np.arctan2(_yy, _xx) % np.pi) / np.pi * 8).astype(int) % 8
_BIN = np.where(_RAD >= 1, _RB * 8 + _ANG, -1).ravel()
_A = np.zeros((SZ * SZ, 128), np.float32)
_A[np.flatnonzero(_BIN >= 0), _BIN[_BIN >= 0]] = 1.0
_A_UNSHIFTED = _A.reshape(SZ, SZ, 128)[np.fft.fftshift(np.arange(SZ))][:, np.fft.fftshift(np.arange(SZ))].reshape(SZ * SZ, 128)   # same bins, FFT order (no fftshift)
try:
    import scipy.fft as _sfft   # 4-5x faster than numpy.fft on the Orange Pi for this size
except ImportError:
    _sfft = None
_HANN = np.outer(np.hanning(SZ), np.hanning(SZ)).astype(np.float32)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spec_proposals_model.json')

PROP = dict(show=False, boxes=[], boxes_t=0.0, frame_t=None, ms=None, runs=0, errors=0, last_error='')   # read by the page / preview


def resize64(c):
    """window -> 64x64 by block averaging when the size is a multiple of 64, else bilinear (OpenCV if present)."""
    h, w = c.shape
    if h == w and h % SZ == 0:
        k = h // SZ
        return c.reshape(SZ, k, SZ, k).mean((1, 3)).astype(np.float32)
    try:
        import cv2
        return cv2.resize(c.astype(np.float32), (SZ, SZ), interpolation=cv2.INTER_AREA)
    except ImportError:
        yi = (np.arange(SZ) + 0.5) * h / SZ - 0.5; xi = (np.arange(SZ) + 0.5) * w / SZ - 0.5
        y0 = np.clip(np.floor(yi).astype(int), 0, h - 2); x0 = np.clip(np.floor(xi).astype(int), 0, w - 2)
        wy = (yi - y0)[:, None]; wx = (xi - x0)[None, :]
        c = c.astype(np.float32)
        return ((1 - wy) * ((1 - wx) * c[y0][:, x0] + wx * c[y0][:, x0 + 1]) + wy * ((1 - wx) * c[y0 + 1][:, x0] + wx * c[y0 + 1][:, x0 + 1]))


def spectra(crops):
    """crops [n, 64, 64] -> [n, 128] log shares of the 2-D power spectrum in 16 rings x 8 orientations."""
    c = (crops - crops.mean((1, 2), keepdims=True)) * _HANN
    Z = _sfft.fft2(c.astype(np.float32), workers=1) if _sfft is not None else np.fft.fft2(c)
    P = (Z.real.astype(np.float32) ** 2 + Z.imag.astype(np.float32) ** 2)
    v = P.reshape(len(P), -1) @ _A_UNSHIFTED
    return np.log(v / (v.sum(1, keepdims=True) + 1e-12) + 1e-9)


def features(frame, boxes):
    """frame: 2-D array (16-bit raw or any linear picture); boxes [(x0, y0, x1, y1)] square windows ->
    [n, 129]: 128 spectrum bins + warmth (window mean minus frame median, over frame std)."""
    f = frame.astype(np.float32)
    med, sd = float(np.median(f)), float(f.std()) + 1e-6
    crops = np.stack([resize64(f[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes])
    warm = (crops.mean((1, 2)) - med) / sd
    return np.c_[spectra(crops), warm]


def windows(h, w, sizes, stride):
    return [(x0, y0, x0 + z, y0 + z) for z in sizes for y0 in range(0, h - z + 1, stride) for x0 in range(0, w - z + 1, stride)]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


class Model:
    def __init__(self, path=MODEL_PATH):
        with open(path, encoding='utf-8') as fh:
            m = json.load(fh)
        self.mu, self.sd = np.array(m['mu'], np.float32), np.array(m['sd'], np.float32)
        self.w, self.b, self.thr = np.array(m['w'], np.float32), float(m['b']), float(m['threshold'])
        self.sizes, self.stride, self.top = tuple(m['sizes']), int(m['stride']), int(m.get('top', 3))
        self.meta = {k: m[k] for k in ('trained_on', 'note', 'threshold_rule') if k in m}
        self.relative = m.get('decision') == 'within_frame_z'   # threshold on (s - median) / (1.4826 MAD) inside the frame

    def score(self, F):
        return ((F - self.mu) / self.sd) @ self.w + self.b

    def propose(self, frame, people=()):
        """frame [H, W]; people: boxes (x0, y0, x1, y1) already found (picture coordinates) -> [(x0, y0, x1, y1, score)]"""
        H, W = frame.shape
        wins = [b for b in windows(H, W, self.sizes, self.stride)
                if not any(_iou(b, p) > 0.2 or (p[0] <= (b[0] + b[2]) / 2 <= p[2] and p[1] <= (b[1] + b[3]) / 2 <= p[3]) for p in people)]
        if not wins:
            return []
        s = self.score(features(frame, wins))
        if self.relative:
            s = (s - np.median(s)) / (1.4826 * np.median(np.abs(s - np.median(s))) + 1e-9)
        out = []
        for i in np.argsort(-s):
            if s[i] < self.thr or len(out) >= self.top:
                break
            if all(_iou(wins[i], o[:4]) < 0.3 for o in out):
                out.append((*wins[i], float(s[i])))
        return out


class Proposer:
    """One background thread, one slot: submit() replaces a waiting frame; results land in PROP (picture coordinates)."""
    def __init__(self, model=None):
        self.model = model or Model()
        self._slot, self._ev, self._lock = None, threading.Event(), threading.Lock()
        self._th = threading.Thread(target=self._run, name='spec-proposals', daemon=True)
        self._th.start()

    def submit(self, frame_t, raw, people):
        with self._lock:
            self._slot = (frame_t, raw, list(people))
        self._ev.set()

    def _run(self):
        while True:
            self._ev.wait(); self._ev.clear()
            with self._lock:
                job, self._slot = self._slot, None
            if job is None:
                continue
            t0 = time.perf_counter()
            try:
                boxes = self.model.propose(job[1], job[2])
                PROP.update(boxes=boxes, boxes_t=time.monotonic(), frame_t=job[0], ms=round((time.perf_counter() - t0) * 1e3, 1), runs=PROP['runs'] + 1)
            except Exception as e:   # never let the display feature hurt the server
                PROP.update(errors=PROP['errors'] + 1, last_error=repr(e)[:200])


_PROPOSER = [None]


def submit(frame_t, raw, people):
    """called per published frame when PROP['show'] is on; starts the worker on first use"""
    if _PROPOSER[0] is None:
        try:
            _PROPOSER[0] = Proposer()
        except Exception as e:
            PROP.update(show=False, errors=PROP['errors'] + 1, last_error='model: ' + repr(e)[:180])
            return
    _PROPOSER[0].submit(frame_t, raw, people)


def state():
    return dict(show=PROP['show'], ms=PROP['ms'], runs=PROP['runs'], n=len(PROP['boxes']), errors=PROP['errors'], last_error=PROP['last_error'])


if __name__ == '__main__':   # timing: python -m src.detect.spec_proposals [frames]
    import sys
    m = Model()
    rng = np.random.default_rng(0)
    fr = (rng.random((480, 640)) * 3000 + 20000).astype(np.uint16)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = m.propose(fr, [(100, 100, 300, 400)]); ts.append((time.perf_counter() - t) * 1e3)
    print(json.dumps(dict(windows=len(windows(480, 640, m.sizes, m.stride)), sizes=m.sizes, stride=m.stride, ms=[round(x, 1) for x in ts], proposals=len(r))))
