"""ByteTrack association and the server's optional rebind mode."""

import ctypes as C
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np

from ..common.shared_values import LOST, PERSON_CLS, REMOVED, TRACKED
from .kalman import KF
from .position_memory import pos_predict
from .tracker_common import Track, edge_ok, iou_dist, linear_assignment


class ByteTrackTracker:
    """mode 'bytetrack': IoU only, buffer 30. mode 'rebind': + motion re-bind (radius 2, guard .3, buffer 60); appearance arrives
    asynchronously through attach_embedding() and may merge a new track into a lost one (late re-bind, sim >= self.sim)."""
    def __init__(self, mode, sim=0.5, radius=2.0, guard=0.3, buffer=None, app=False, late_window=10, app_first=False, gate=3.5, size_ratio=(0.5, 2.0),
                 lost_max_age=90, motion_fallback=True, vit_conf=0.25, vit_min_h=32.0, hybrid=True, sync_k=2, hybrid_fallback=False, n_init=2, class_gate=False):
        self.mode, self.sim, self.radius, self.guard, self.app, self.late_window = mode, sim, radius, guard, app, late_window
        # n_init (--n-init, default 2 = unchanged): matched frames incl. the birth frame before a new track is output (idsw_fix FixTracker F4);
        # <= 2 skips the extra code below entirely (a new track already needs its birth frame + 1 hit; frame-1 births are output at once)
        self.n_init = int(n_init)
        # class_gate (--class-gate, default False = unchanged): person / non-person pairs are forbidden in every association (_cg)
        self.class_gate = bool(class_gate)
        # hybrid: detections WITH a feature are re-associated by appearance only; detections WITHOUT one keep the motion fallback.
        # sync_k: max candidate crops embedded synchronously per frame through update(feat_fn=...)
        # hybrid_fallback (default off): a FEATURED detection left unmatched by the appearance pass may still take the motion fallback, but
        # ONLY against lost tracks WITHOUT a gallery feature (where both sides have a feature the appearance veto stays final)
        self.hybrid_fallback = hybrid_fallback; self.diag = None   # diag: optional list, filled with new-track records (analysis only)
        self.hybrid, self.sync_k = hybrid, sync_k; self.sync_frames = 0; self.sync_crops = 0; self.cand_frames = 0; self.cand_crops = 0
        # appearance-first re-association of LOST tracks (wide spatial gate, size ratio, lost age); leftovers -> motion fallback
        self.app_first, self.gate, self.size_ratio, self.lost_max_age, self.motion_fallback = app_first, gate, size_ratio, lost_max_age, motion_fallback
        self.vit_conf, self.vit_min_h = vit_conf, vit_min_h
        self.high, self.low, self.new, self.match = 0.25, 0.10, 0.25, 0.8
        self.buffer = buffer if buffer is not None else (30 if mode == 'bytetrack' else 60)
        self.kf = KF(); self.frame_id = 0; self.tracked = []; self.lost = []; self.next_id = 0
        self.rebinds_motion = 0; self.rebinds_app = 0; self.rebinds_fallback = 0; self.new_this_frame = []; self.lost_this_frame = []

    def _nid(self):
        self.next_id += 1; return self.next_id

    def _dists(self, tracks, dets):
        d = iou_dist(tracks, dets)
        if d.size: d = 1 - (1 - d) * np.asarray([x.score for x in dets])[None]
        return d

    def _cg(self, cost, A, B):
        """--class-gate only: cost 1.0 (= never matched) for every track / detection pair whose person / non-person group differs.
        Off: returns cost unchanged (same object), so the default path is byte-identical."""
        if not self.class_gate or cost.size == 0: return cost
        ga = np.asarray([t.cls == PERSON_CLS for t in A]); gb = np.asarray([d.cls == PERSON_CLS for d in B])
        cost[ga[:, None] != gb[None, :]] = 1.0
        return cost

    def _tentative(self, u):
        """--n-init N > 2 only (== idsw_fix FixTracker F4): called right after an UNCONFIRMED track u matched a detection (u.update set
        is_activated); hits counts matched frames incl. the birth frame (1 when absent); below N the track stays unconfirmed (not output)."""
        u.hits = getattr(u, 'hits', 1) + 1
        if u.hits < self.n_init: u.is_activated = False

    def update(self, arr, det_feats=None, feat_fn=None):
        """arr [n,6] x1,y1,x2,y2,conf,cls (frame coords); det_feats: optional {det_idx: L2 feature} available at association time
        (offline oracle); feat_fn(list of det_idx) -> {det_idx: feature}: synchronous embedding of the re-identification CANDIDATES
        (unmatched person detections h >= vit_min_h after the first association, at most sync_k) = the --vit-sync mode.
        Returns rows [x1,y1,x2,y2,id,conf,cls,det_idx]."""
        self.frame_id += 1; act, refind, lost, removed = [], [], [], []; self.new_this_frame, self.lost_this_frame = [], []
        wh = arr[:, 2:4] - arr[:, 0:2]; valid = (wh[:, 0] > 0) & (wh[:, 1] > 0)
        hi = np.flatnonzero(valid & (arr[:, 4] >= self.high)); lo = np.flatnonzero(valid & (arr[:, 4] > self.low) & (arr[:, 4] < self.high))
        def mk(i):
            t = Track(np.r_[(arr[i, :2] + arr[i, 2:4]) / 2, wh[i]], arr[i, 4], arr[i, 5], i)
            if det_feats is not None and int(i) in det_feats: t.det_feat = det_feats[int(i)]
            return t
        dets = [mk(i) for i in hi]; dets2 = [mk(i) for i in lo]
        unconfirmed = [t for t in self.tracked if not t.is_activated]; tracked = [t for t in self.tracked if t.is_activated]
        pool = tracked + [t for t in self.lost if t not in tracked]
        if pool:
            mean = np.asarray([t.mean for t in pool]); cov = np.asarray([t.cov for t in pool])
            for i, t in enumerate(pool):
                if t.state != TRACKED: mean[i, 6:8] = 0
            mean, cov = pos_predict(self.kf, pool, mean, cov)   # v32
            for i, t in enumerate(pool): t.mean, t.cov = mean[i], cov[i]
        m, ut, ud = linear_assignment(self._cg(self._dists(pool, dets), pool, dets), self.match)
        for i, j in m:
            t = pool[i]; was_lost = t.state == LOST; t.update(dets[j], self.frame_id); (refind if was_lost else act).append(t)
            if getattr(dets[j], 'det_feat', None) is not None: t.set_feat(dets[j].det_feat); t.feat_frame = self.frame_id
        if self.mode == 'rebind' and self.app and self.app_first:
            lost_f = [i for i in ut if pool[i].state == LOST and pool[i].feat is not None and self.frame_id - pool[i].end_frame <= self.lost_max_age]
            # candidates = unmatched person detections >= vit_min_h; synchronous embedding (feat_fn) only when a lost gallery exists
            cands = [j for j in ud if dets[j].cls == PERSON_CLS and dets[j].score >= self.vit_conf and dets[j].xywh0[3] >= self.vit_min_h]
            if cands: self.cand_frames += 1; self.cand_crops += len(cands)
            if feat_fn is not None and lost_f and cands:
                want = [j for j in cands if getattr(dets[j], 'det_feat', None) is None][:self.sync_k]
                if want:
                    got = feat_fn([dets[j].idx for j in want]) or {}
                    for j in want:
                        f = got.get(dets[j].idx)
                        if f is not None: dets[j].det_feat = f
                    self.sync_frames += 1; self.sync_crops += len(want)
            # appearance-FIRST: lost tracks with a gallery feature x unmatched detections that carry a feature
            cand_f = [j for j in ud if getattr(dets[j], 'det_feat', None) is not None]
            if lost_f and cand_f:
                cost = np.ones((len(lost_f), len(cand_f)), np.float32)
                for a, i in enumerate(lost_f):
                    t = pool[i]; tc, th = t.xywh[:2], max(float(t.xywh[3]), 1.0)
                    for b, j in enumerate(cand_f):
                        d = dets[j]; dh = max(float(d.xywh0[3]), 1.0); ratio = dh / th
                        if ratio < self.size_ratio[0] or ratio > self.size_ratio[1]: continue
                        if float(np.hypot(d.xywh0[0] - tc[0], d.xywh0[1] - tc[1])) > self.gate * max(th, dh): continue
                        s = float(t.feat @ d.det_feat)
                        if s >= self.sim: cost[a, b] = 1.0 - s
                mm, _, _ = linear_assignment(cost, 0.999); tk, dk = set(), set()
                for a, b in mm:
                    i, j = lost_f[a], cand_f[b]; pool[i].re_activate(dets[j], self.frame_id); pool[i].set_feat(dets[j].det_feat); refind.append(pool[i]); tk.add(i); dk.add(j); self.rebinds_app += 1
                ut = [i for i in ut if i not in tk]; ud = [j for j in ud if j not in dk]
        if self.mode == 'rebind' and (self.motion_fallback or not self.app):
            lost_i = [i for i in ut if pool[i].state == LOST]; cand = list(ud); feat_d = ()
            if self.app and self.app_first and self.hybrid and self.hybrid_fallback:
                # --hybrid-fallback: featured detections stay candidates, but may only pair with lost tracks that have NO gallery feature
                feat_d = {j for j in cand if getattr(dets[j], 'det_feat', None) is not None}
            elif self.app and self.app_first and self.hybrid:
                # hybrid: detections that HAVE a feature were decided by appearance only (no motion fallback for them)
                cand = [j for j in cand if getattr(dets[j], 'det_feat', None) is None]
            elif self.app and self.app_first and det_feats is None and feat_fn is None:
                # deferred variant (non-hybrid): crop-eligible person detections are NOT grabbed by motion now; they start a new track, get
                # embedded, and attach_embedding() decides (appearance first, then late motion fallback)
                cand = [j for j in cand if not (dets[j].cls == PERSON_CLS and dets[j].score >= self.vit_conf and dets[j].xywh0[3] >= self.vit_min_h)]
            if cand and self.guard > 0:
                occ = [t for t in pool if t.state == TRACKED] + [t for t in act if t not in pool]
                if occ:
                    iou = 1 - iou_dist(occ, [dets[j] for j in cand]); cand = [j for j, free in zip(cand, iou.max(0) <= self.guard) if free]
            if lost_i and cand:
                # == run_ab.py GatedBOTSORT._post_first_association: rebind_sim=S, rebind_fallback=True (app) / rebind_noapp (motion only)
                cost = np.ones((len(lost_i), len(cand)), np.float32)
                for a, i in enumerate(lost_i):
                    t = pool[i]; tc, th = t.xywh[:2], max(float(t.xywh[3]), 1.0)
                    for b, j in enumerate(cand):
                        d = dets[j]; rel = float(np.hypot(d.xywh0[0] - tc[0], d.xywh0[1] - tc[1])) / (self.radius * th)
                        if rel > 1.0: continue
                        if self.class_gate and (t.cls == PERSON_CLS) != (d.cls == PERSON_CLS): continue   # --class-gate
                        if feat_d and j in feat_d and t.feat is not None: continue              # appearance veto (both sides featured)
                        if not self.app: cost[a, b] = 0.5 * rel                                   # motion-only control (B)
                        elif t.feat is None or getattr(d, 'det_feat', None) is None: cost[a, b] = 0.8 + 0.19 * rel   # hybrid fallback (no appearance on a side)
                        else:
                            s = float(t.feat @ d.det_feat)
                            if s >= self.sim: cost[a, b] = 1.0 - s                                   # appearance pair beats every fallback pair
                mm, _, _ = linear_assignment(cost, 0.999); tk, dk = set(), set()
                for a, b in mm:
                    i, j = lost_i[a], cand[b]; pool[i].re_activate(dets[j], self.frame_id); refind.append(pool[i]); tk.add(i); dk.add(j)
                    if cost[a, b] >= 0.8 or not self.app: self.rebinds_motion += 1
                    else: self.rebinds_app += 1
                    if feat_d and j in feat_d: self.rebinds_fallback += 1
                ut = [i for i in ut if i not in tk]; ud = [j for j in ud if j not in dk]
        rt = [pool[i] for i in ut if pool[i].state == TRACKED]
        if rt and dets2:
            m2, ut2, _ = linear_assignment(self._cg(iou_dist(rt, dets2), rt, dets2), 0.5)
            for i, j in m2: rt[i].update(dets2[j], self.frame_id); act.append(rt[i])
        else: ut2 = list(range(len(rt)))
        for i in ut2:
            if rt[i].state != LOST: rt[i].state = LOST; lost.append(rt[i]); self.lost_this_frame.append(rt[i])
        dets = [dets[j] for j in ud]
        if unconfirmed:
            m3, uu, ud3 = linear_assignment(self._cg(self._dists(unconfirmed, dets), unconfirmed, dets), 0.7)
            for i, j in m3:
                unconfirmed[i].update(dets[j], self.frame_id); act.append(unconfirmed[i])
                if self.n_init > 2: self._tentative(unconfirmed[i])
            for i in uu: unconfirmed[i].state = REMOVED; removed.append(unconfirmed[i])
        else: ud3 = list(range(len(dets)))
        for j in ud3:
            if dets[j].score >= self.new and edge_ok(dets[j]):   # v24f
                dets[j].activate(self.kf, self.frame_id, self._nid()); act.append(dets[j]); self.new_this_frame.append(dets[j])
                if self.diag is not None: self._diag_new(dets[j])
                if getattr(dets[j], 'det_feat', None) is not None: dets[j].set_feat(dets[j].det_feat); dets[j].feat_frame = self.frame_id
        for t in self.lost:
            if self.frame_id - t.end_frame > self.buffer: t.state = REMOVED; removed.append(t)
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        for t in act + refind:
            if t not in self.tracked: self.tracked.append(t)
        self.lost = [t for t in self.lost if t.state == LOST and t not in self.tracked] + [t for t in lost if t not in self.tracked]
        self.lost = [t for t in self.lost if t.state != REMOVED]
        return [[*t.xyxy.tolist(), t.track_id, t.score, t.cls, t.idx] for t in self.tracked if t.is_activated]

    def _diag_new(self, d):
        """analysis only (self.diag is a list): why did this detection start a NEW track? one record per lost track in the pool"""
        f = getattr(d, 'det_feat', None); dh = max(float(d.xywh0[3]), 1.0); lost = []
        for l in self.lost:
            if l.state != LOST: continue
            lh = max(float(l.xywh[3]), 1.0); dist = float(np.hypot(d.xywh0[0] - l.xywh[0], d.xywh0[1] - l.xywh[1]))
            lost.append(dict(id=l.track_id, feat=l.feat is not None, age=self.frame_id - l.end_frame, ratio=dh / lh,
                             gate_rel=dist / max(lh, dh), motion_rel=dist / (self.radius * lh),
                             sim=(float(l.feat @ f) if (l.feat is not None and f is not None) else None)))
        self.diag.append(dict(frame=self.frame_id, new_id=d.track_id, h=dh, cls=d.cls, conf=d.score, det_feat=f is not None, lost=lost))

    def find(self, track_id):
        for t in self.tracked + self.lost:
            if t.track_id == track_id and t.state != REMOVED: return t
        return None

    def attach_embedding(self, track_id, frame_id, feat, kind):
        """Late arrival of a ViT embedding for the detection that track `track_id` owned at `frame_id`. Returns 'merged' id or None."""
        t = self.find(track_id)
        if t is None: return None
        t.set_feat(feat); t.feat_frame = frame_id
        # late appearance re-bind (the online stand-in for the PC rule's appearance pairs): a track that started recently from a
        # detection the motion fallback did NOT bind is merged into the LOST track with the best cosine >= sim within the radius
        if self.mode != 'rebind' or not self.app or kind != 'new' or self.frame_id - t.start_frame > self.late_window: return None
        best, bs = None, self.sim
        tc, th = t.xywh[:2], max(float(t.xywh[3]), 1.0)
        gate = self.gate if self.app_first else self.radius
        for l in self.lost:
            if l is t or l.feat is None or l.state != LOST: continue
            lh = max(float(l.xywh[3]), 1.0)
            if self.app_first and (self.frame_id - l.end_frame > self.lost_max_age or not (self.size_ratio[0] <= th / lh <= self.size_ratio[1])): continue
            if float(np.hypot(l.xywh[0] - tc[0], l.xywh[1] - tc[1])) > gate * max(th, lh): continue
            s = float(l.feat @ t.feat)
            if s >= bs: best, bs = l, s
        if best is None and self.app_first and self.motion_fallback and not self.hybrid:
            # late motion fallback for a deferred detection (non-hybrid variant): nearest LOST track within the motion radius
            bd = 1.0
            for l in self.lost:
                if l is t or l.state != LOST: continue
                rel = float(np.hypot(l.xywh[0] - tc[0], l.xywh[1] - tc[1])) / (self.radius * max(float(l.xywh[3]), 1.0))
                if rel <= bd: best, bd = l, rel
            if best is not None:
                old = t.track_id; t.track_id = best.track_id; best.state = REMOVED; self.lost = [l for l in self.lost if l is not best]; self.rebinds_motion += 1
                return (old, t.track_id, -1.0)
        if best is None: return None
        old = t.track_id; t.track_id = best.track_id; t.feat = 0.5 * (t.feat + best.feat); t.feat /= max(np.linalg.norm(t.feat), 1e-12)
        best.state = REMOVED; self.lost = [l for l in self.lost if l is not best]; self.rebinds_app += 1
        return (old, best.track_id, bs)


Tracker = ByteTrackTracker


def crop_requests(tracker, every, k, vit_conf, vit_min_h, has_prev):
    """ViT crop requests after tracker.update(): EVERY frame the new person tracks (re-bind candidates) and just-lost tracks without
    a gallery feature; on every-N frames also a round-robin refresh of active tracks. Returns at most k request dicts
    (kind, track_id, frame, idx, box in frame xyxy, src 'cur'|'prev')."""
    req = []
    for t in tracker.new_this_frame:
        if t.cls == PERSON_CLS and t.score >= vit_conf and (t.last_det[2][3] - t.last_det[2][1]) >= vit_min_h:
            req.append(dict(kind='new', track_id=t.track_id, frame=tracker.frame_id, idx=t.idx, box=t.last_det[2], src='cur'))
    for t in tracker.lost_this_frame:
        if t.cls == PERSON_CLS and t.feat is None and t.last_det is not None and t.last_det[0] == tracker.frame_id - 1 and has_prev:
            req.append(dict(kind='lost', track_id=t.track_id, frame=t.last_det[0], idx=t.last_det[1], box=t.last_det[2], src='prev'))
    if tracker.frame_id % every == 0:
        act = [t for t in tracker.tracked if t.is_activated and t.cls == PERSON_CLS and t.last_det[0] == tracker.frame_id and t.score >= vit_conf
               and (t.last_det[2][3] - t.last_det[2][1]) >= vit_min_h and t not in tracker.new_this_frame]
        act.sort(key=lambda t: t.feat_frame)
        for t in act: req.append(dict(kind='refresh', track_id=t.track_id, frame=tracker.frame_id, idx=t.idx, box=t.last_det[2], src='cur'))
    return req[:k]


class _Det(C.Structure):
    _fields_ = [(name, C.c_float) for name in ("x1", "y1", "x2", "y2", "score")] + [("cls", C.c_int)]


class _Track(C.Structure):
    _fields_ = [(name, C.c_float) for name in ("x1", "y1", "x2", "y2", "score")] + [
        (name, C.c_int) for name in ("id", "cls", "source_index")]


class NativeByteTrack:
    """Plain C++ ByteTrack; one stateful instance per input stream."""

    @staticmethod
    def bind(library):
        lib = C.CDLL(str(library))
        lib.lwirtk_abi.argtypes, lib.lwirtk_abi.restype = [], C.c_int
        if lib.lwirtk_abi() != 2:
            raise RuntimeError("Unsupported C++ ByteTrack ABI")
        lib.lwirtk_create.argtypes, lib.lwirtk_create.restype = [], C.c_void_p
        lib.lwirtk_close.argtypes, lib.lwirtk_close.restype = [C.c_void_p], None
        lib.lwirtk_reset.argtypes, lib.lwirtk_reset.restype = [C.c_void_p], C.c_int
        lib.lwirtk_step.argtypes = [C.c_void_p, C.POINTER(_Det), C.c_int, C.POINTER(_Track), C.c_int]
        lib.lwirtk_step.restype = C.c_int
        return lib

    def __init__(self, library):
        self._lib = lib = load_bytetrack_library(Path(library))
        self._handle = lib.lwirtk_create()
        if not self._handle:
            raise RuntimeError("Could not create C++ ByteTrack")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def _check_open(self):
        if not self._handle:
            raise RuntimeError("C++ ByteTrack is closed")

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.lwirtk_close(self._handle)
            self._handle = None

    def reset(self):
        self._check_open()
        if self._lib.lwirtk_reset(self._handle):
            raise RuntimeError("Could not reset C++ ByteTrack")

    def update(self, rows):
        self._check_open()
        rows = np.asarray(rows, dtype=np.float32)
        if rows.shape == (0,):
            rows = rows.reshape(0, 6)
        if (rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all()
                or np.any(rows[:, 4:] < 0) or np.any(rows[:, 4] > 1)
                or np.any(rows[:, 5] > 2147483520) or np.any(rows[:, 5] != np.trunc(rows[:, 5]))):
            raise ValueError("Expected finite [N,6] xyxy, score in [0,1], and nonnegative integer class IDs")
        n = len(rows)
        inputs = (_Det * n)(*(_Det(*row[:5], int(row[5])) for row in rows))
        output = (_Track * n)()
        size = self._lib.lwirtk_step(self._handle, inputs, n, output, n)
        if size < 0:
            self.close()  # after a native failure the frame may have partially advanced
            raise RuntimeError(f"C++ ByteTrack step failed ({size})")
        return [[r.x1, r.y1, r.x2, r.y2, r.id, r.score, r.cls, r.source_index]
                for r in output[:size]]


_BYTETRACK_LINKS = Path(__file__).resolve().parents[2] / 'models' / 'links'


@lru_cache(maxsize=8)
def _bind_bytetrack_library(path, signature):
    try:
        return NativeByteTrack.bind(path)
    except (AttributeError, OSError, RuntimeError):
        return None


def load_bytetrack_library(path):
    path = path.resolve(strict=True)
    stat = path.stat()
    lib = _bind_bytetrack_library(path, (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size))
    if lib is None:
        raise RuntimeError("C++ ByteTrack library has an incompatible ABI")
    return lib


def available_bytetrack_library(args):
    """Find the installed ABI without requiring a path setting or a server restart."""
    override = getattr(args, 'replay_bytetrack_library', '')
    choices = (Path(override),) if override else (
        _BYTETRACK_LINKS / 'bytetrack.so', _BYTETRACK_LINKS / 'team_bytetrack.so')
    for path in choices:
        try:
            if path.is_file():
                load_bytetrack_library(path)
                return path
        except (OSError, RuntimeError):
            continue
    return None


class CameraByteTrack:
    """Independent preview tracker. Its IDs never enter the control or ReID pipeline."""

    def __init__(self, args):
        self.args = args
        self.lock = Lock()
        self.requested = self.active = False
        self.error = None
        self.tracker = self.last_t = None

    def state(self):
        available = available_bytetrack_library(self.args) is not None
        with self.lock:
            return dict(available=available, requested=self.requested, active=self.active, error=self.error)

    def select(self, enabled):
        with self.lock:
            if enabled and available_bytetrack_library(self.args) is None:
                return False
            self.requested, self.error = enabled, None
            return True

    def blocks_selection(self):
        with self.lock:
            return self.requested or self.active

    def _close(self):
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
        self.last_t = None
        self.active = False

    def close(self):
        with self.lock:
            self._close()

    def step(self, rows, frame_t, camera):
        with self.lock:
            if not self.requested or not camera or rows is None:
                self._close()
                return None
            try:
                if self.last_t is not None and not 0 < frame_t - self.last_t <= 1.0:
                    self._close()
                if self.tracker is None:
                    library = available_bytetrack_library(self.args)
                    if library is None:
                        raise RuntimeError("C++ ByteTrack library is no longer available")
                    self.tracker = NativeByteTrack(library)
                tracked = self.tracker.update(rows)
                self.last_t = frame_t
                self.active = True
                return tracked
            except (OSError, RuntimeError, ValueError) as exc:
                self._close()
                self.requested = False
                self.error = str(exc)
                return None
