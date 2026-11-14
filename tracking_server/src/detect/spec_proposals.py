"""People proposals from the spatial frequency content of the picture (2026-11-14, research feature, display only).

Idea (close_range_20261113/fft_liveness): a window that holds a person has a different 2-D power spectrum than warm
things (monitors, laptops, seats): less fine detail (body heat spreads smoothly) and more oblique coarse structure.
A small logistic model on the spectrum (16 rings x 8 orientations, log shares) + the window's warmth was trained on four
indoor GT recordings; on recordings it had not seen, its best window pointed at a person in 83-100 % of the frames,
also where the warmest spot was a laptop.

Here: square windows slide over the 16-bit raw frame of every shown frame (11-14, user: show what the spectrum finds,
also where the detector found people); the best windows above the model's threshold (up to 'top', overlap < 0.3) are
proposals; a window that lies mostly (>= 30 % of the smaller one) over a chosen one is dropped, so one person gets one box.
propose(people=...) can still skip windows on given boxes; the server passes none. They are only drawn on the picture
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

LOG = dict(on=False, rows=[])   # 11-14 research log (src/web/page_handler.py, /rules/set?prop_log=1|0)
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


SAT = 65000   # the camera's 16-bit values clip at 65280 (hot faces in a close, gain-raised picture)


def features(frame, boxes, extra=False):
    """frame: 2-D array (16-bit raw or any linear picture); boxes [(x0, y0, x1, y1)] square windows ->
    [n, 129]: 128 spectrum bins + warmth (window mean minus frame median, over frame std).
    extra (11-14): + 3 heat cues that survive the camera's automatic gain (the raw values are not temperatures: the
    same room reads ~15000 or ~35000 depending on the recording): window 90th / 99th percentile placed between the frame
    median (0) and the frame's 99.5th percentile (1), and the share of clipped pixels -> [n, 132]."""
    f = frame.astype(np.float32)
    med, sd = float(np.median(f)), float(f.std()) + 1e-6
    crops = np.stack([resize64(f[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes])
    warm = (crops.mean((1, 2)) - med) / sd
    if not extra:
        return np.c_[spectra(crops), warm]
    top = float(np.percentile(f, 99.5)) - med + 1e-6
    flat = crops.reshape(len(crops), -1)
    q = np.partition(flat, (int(0.90 * flat.shape[1]), int(0.99 * flat.shape[1])), axis=1)
    p90 = (q[:, int(0.90 * flat.shape[1])] - med) / top
    p99 = (q[:, int(0.99 * flat.shape[1])] - med) / top
    sat = (flat >= SAT).mean(1)
    return np.c_[spectra(crops), warm, p90, p99, sat]


def windows(h, w, sizes, stride):
    return [(x0, y0, x0 + z, y0 + z) for z in sizes for y0 in range(0, h - z + 1, stride) for x0 in range(0, w - z + 1, stride)]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def _inside(a, b):
    """overlap as a share of the smaller box (1.0 when one window lies inside the other)"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / m if m > 0 else 0.0


def _same_column(a, b):
    """a person is tall and the windows square: a head window just above a body window is the same person.
    x overlap >= 50 % of the narrower window and vertical gap <= 50 % of the smaller height. On the indoor GT recordings
    (in-sample, close_range_20261113/fft_liveness/spec_dup_eval.py) this cut people with 2+ boxes 64 -> 46 while people
    with a box went 228 -> 226."""
    ox = min(a[2], b[2]) - max(a[0], b[0]); w = min(a[2] - a[0], b[2] - b[0])
    gap = max(a[1], b[1]) - min(a[3], b[3]); h = min(a[3] - a[1], b[3] - b[1])
    return ox >= 0.5 * w and gap <= 0.5 * h


class Model:
    def __init__(self, path=MODEL_PATH):
        with open(path, encoding='utf-8') as fh:
            m = json.load(fh)
        self.mu, self.sd = np.array(m['mu'], np.float32), np.array(m['sd'], np.float32)
        self.w, self.b, self.thr = np.array(m['w'], np.float32), float(m['b']), float(m['threshold'])
        self.sizes, self.stride, self.top = tuple(m['sizes']), int(m['stride']), int(m.get('top', 3))
        self.meta = {k: m[k] for k in ('trained_on', 'note', 'threshold_rule') if k in m}
        self.extra = bool(m.get('extra_features'))   # 11-14: model trained with features(extra=True)
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
        s = self.score(features(frame, wins, self.extra))
        if self.relative:
            s = (s - np.median(s)) / (1.4826 * np.median(np.abs(s - np.median(s))) + 1e-9)
        out = []
        for i in np.argsort(-s):
            if s[i] < self.thr or len(out) >= self.top:
                break
            if all(_inside(wins[i], o[:4]) < 0.3 and not _same_column(wins[i], o[:4]) for o in out):   # 11-14 (user): one box per person (head inside / above the body)
                out.append((*wins[i], float(s[i])))
        return out


    def rescore(self, frame, boxes):
        """boxes (x0, y0, x1, y1, ...) -> [(x0, y0, x1, y1, spectral score)] scored on the central square of each box
        (as the person samples in training); boxes whose square is < 16 px are left out"""
        H, W = frame.shape
        keep, sqs = [], []
        for b in boxes:
            x0, y0, x1, y1 = max(0, int(round(b[0]))), max(0, int(round(b[1]))), min(W, int(round(b[2]))), min(H, int(round(b[3])))
            s = min(x1 - x0, y1 - y0); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if s >= 16:
                keep.append((x0, y0, x1, y1)); sqs.append((cx - s // 2, cy - s // 2, cx - s // 2 + s, cy - s // 2 + s))
        if not keep:
            return []
        return [(*k, float(v)) for k, v in zip(keep, self.score(features(frame, sqs, self.extra)))]

    def pick_weak(self, frame, weak):
        """the detector's weak boxes whose spectral score passes the threshold, best first, one per place, at most 'top'"""
        out = []
        for o in sorted(self.rescore(frame, weak), key=lambda r: -r[4]):
            if o[4] < self.thr or len(out) >= self.top:
                break
            if all(_inside(o[:4], p[:4]) < 0.3 and not _same_column(o[:4], p[:4]) for p in out):
                out.append(o)
        return out


WEAK_MIN, WEAK_IOU = 0.05, 0.3


def weak_boxes(dets, tracks):
    """person detections (x0, y0, x1, y1, score) the tracker did not use: score >= WEAK_MIN and IoU < WEAK_IOU with every tracker person"""
    return [d for d in dets if d[4] >= WEAK_MIN and all(_iou(d, t) < WEAK_IOU for t in tracks)][:60]


KEEP_N, KEEP_K = 3, 2   # 11-14: a place is shown when it was found in >= 2 of the last 3 passes (one pass ~0.2-0.3 s on the device)


def _same_place(a, b):
    """the same place in two passes ~0.25 s apart (people move a little)"""
    if _inside(a, b) >= 0.3 or _same_column(a, b):
        return True
    z = min(a[2] - a[0], b[2] - b[0])
    return abs((a[0] + a[2]) - (b[0] + b[2])) / 2 <= 0.75 * z and abs((a[1] + a[3]) - (b[1] + b[3])) / 2 <= 0.75 * z


def _covered(b, people):
    """the window is on a person the detector reported: its centre lies in a person box or it lies mostly inside one"""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return any((p[0] <= cx <= p[2] and p[1] <= cy <= p[3]) or _inside(b, p) >= 0.5 for p in people)


def steady(history, out):
    """out: this pass's [(x0, y0, x1, y1, score)]; history: earlier passes (newest last) -> the ones also found in
    enough earlier passes (KEEP_K counting this one)"""
    return [o for o in out if 1 + sum(any(_same_place(o[:4], h[:4]) for h in past) for past in history[-(KEEP_N - 1):]) >= KEEP_K]


def mark(out, people):
    """-> [(x0, y0, x1, y1, score, covered)]: covered = on a person the detector already reported"""
    return [(*o[:5], _covered(o[:4], people)) for o in out]


class Proposer:
    """One background thread, one slot: submit() replaces a waiting frame; results land in PROP (picture coordinates).
    11-14 (user: an aid that tells what the detector missed): with weak boxes (the server's way) only the detector's
    weak boxes are scored (weak_boxes, Model.pick_weak); without, the whole frame is searched (Model.propose). A place is
    kept only when found again in the next passes (steady), and each kept place says whether the tracker has a person
    there (people = the tracker's person boxes of the same frame, raw/real coordinates).
    Why weak boxes (close_range_20261113/fft_liveness/weak_det_spec_eval.py, device replay logs of 4 indoor GT
    recordings, model trained without the tested recording): among weak boxes, the spectral score told boxes on missed
    people from false ones far better than the detector score (pooled AUC 0.79 vs 0.47; alerts on missed people 57 % vs
    18 %), but per recording it was 0.88 / 0.44 / 0.71 and on one recording every alert was false."""
    def __init__(self, model=None):
        self.model = model or Model()
        self._slot, self._ev, self._lock = None, threading.Event(), threading.Lock()
        self._hist = []
        self._th = threading.Thread(target=self._run, name='spec-proposals', daemon=True)
        self._th.start()

    def submit(self, frame_t, raw, people, weak=None):
        with self._lock:
            self._slot = (frame_t, raw, list(people), None if weak is None else list(weak))
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
                out = self.model.propose(job[1]) if job[3] is None else self.model.pick_weak(job[1], job[3])
                boxes = mark(steady(self._hist, out), job[2])
                self._hist = (self._hist + [out])[-(KEEP_N - 1):]
                PROP.update(boxes=boxes, boxes_t=time.monotonic(), frame_t=job[0], ms=round((time.perf_counter() - t0) * 1e3, 1), runs=PROP['runs'] + 1,
                            missed=PROP.get('missed', 0) + sum(1 for b in boxes if not b[5]))
            except Exception as e:   # never let the display feature hurt the server
                PROP.update(errors=PROP['errors'] + 1, last_error=repr(e)[:200])


_PROPOSER = [None]


def submit(frame_t, raw, people, weak=None):
    """called per published frame when PROP['show'] is on; starts the worker on first use"""
    if _PROPOSER[0] is None:
        try:
            _PROPOSER[0] = Proposer()
        except Exception as e:
            PROP.update(show=False, errors=PROP['errors'] + 1, last_error='model: ' + repr(e)[:180])
            return
    _PROPOSER[0].submit(frame_t, raw, people, weak)


def log_frame(raw, dets, tracks, props=()):
    """raw: 16-bit frame (real coordinates); dets: [(x0, y0, x1, y1, score)] every person detection; tracks: [(x0, y0, x1, y1)];
    props: the proposals on screen at this frame (PROP['boxes']: x0, y0, x1, y1, score, covered), for end-to-end checks"""
    if len(LOG['rows']) < 20000:
        LOG['rows'].append(dict(sig=raw[::40, ::40].astype(int).ravel().tolist(), dets=[[round(float(v), 1) for v in d[:4]] + [round(float(d[4]), 4)] for d in dets if d[4] >= 0.05][:60],   # the detector keeps up to 300 boxes down to 0.001
                                tracks=[[round(float(v), 1) for v in t[:4]] for t in tracks],
                                props=[[round(float(v), 1) for v in p[:5]] + [bool(p[5])] for p in props]))


def log_write(path):
    rows, LOG['rows'] = LOG['rows'], []
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(dict(frames=len(rows), rows=rows), fh)
    return len(rows)


def state():
    return dict(show=PROP['show'], ms=PROP['ms'], runs=PROP['runs'], n=len(PROP['boxes']), n_missed=sum(1 for b in PROP['boxes'] if not b[5]),
                missed_total=PROP.get('missed', 0), errors=PROP['errors'], last_error=PROP['last_error'], log_on=LOG['on'], log_frames=len(LOG['rows']))


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
