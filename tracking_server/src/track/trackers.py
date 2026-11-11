"""추적기: 검출 상자를 프레임 사이에서 같은 번호로 이어 붙인다 (Track, Tracker, BotSortTracker, AltTracker, LostFirstTracker)."""
import numpy as np
from ..common import settings
from ..common.shared_values import PERSON_CLS, TRACKED, LOST, REMOVED
from ..track.kalman import KF, gmc_warp
from ..track.position_memory import POSP, pos_inn, pos_predict
from ..track.stabilizer import STAB_M


# ------------------------------------------------------------------------------------------------------------- tracker
try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None


def linear_assignment(cost, thresh):
    if cost.size == 0: return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    if cost.shape == (1, 1):                                   # trivial case: no solver call
        return ([(0, 0)], [], []) if cost[0, 0] <= thresh else ([], [0], [0])
    c = np.where(cost > thresh, 1e6, cost)
    r, k = _lsa(c)
    matches = [(int(i), int(j)) for i, j in zip(r, k) if cost[i, j] <= thresh]
    ma = {i for i, _ in matches}; mb = {j for _, j in matches}
    return matches, [i for i in range(cost.shape[0]) if i not in ma], [j for j in range(cost.shape[1]) if j not in mb]
CANVAS_W, CANVAS_H, CANVAS_Y0 = 640.0, 512.0, 16.0   # detections live on the 640x512 canvas; camera rows are 16..496


def edge_ok(det):
    """False when the detection touches a frame edge and its score is below EDGE_BIRTH (v24f edge birth rule)."""
    if settings.EDGE_BIRTH <= 0 or det.score >= settings.EDGE_BIRTH: return True
    x, y, w, h = det.xywh0
    if STAB_M[0] is not None:   # stab: the real picture edge lies inside the canvas by the replicated border
        ml_, mt_, mr_, mb_ = STAB_M[0]
        return not (x - w / 2 <= settings.EDGE_PX + ml_ or x + w / 2 >= CANVAS_W - settings.EDGE_PX - mr_ or y - h / 2 <= CANVAS_Y0 + settings.EDGE_PX + mt_ or y + h / 2 >= CANVAS_Y0 + 480.0 - settings.EDGE_PX - mb_)
    return not (x - w / 2 <= settings.EDGE_PX or x + w / 2 >= CANVAS_W - settings.EDGE_PX or y - h / 2 <= CANVAS_Y0 + settings.EDGE_PX or y + h / 2 >= CANVAS_Y0 + 480.0 - settings.EDGE_PX)


class Track:
    def __init__(self, xywh, score, cls, idx):
        self.xywh0 = np.asarray(xywh, np.float64); self.score = float(score); self.cls = int(cls); self.idx = int(idx)
        self.feat = None; self.mean = None; self.cov = None; self.state = 0; self.is_activated = False; self.track_id = 0
        self.frame_id = 0; self.end_frame = 0; self.start_frame = 0; self.last_det = None; self.feat_frame = -1
    def set_feat(self, f):
        f = f / max(np.linalg.norm(f), 1e-12)
        self.feat = f if self.feat is None else 0.9 * self.feat + 0.1 * f
        self.feat /= max(np.linalg.norm(self.feat), 1e-12)
    @property
    def xywh(self): return self.xywh0 if self.mean is None else self.mean[:4]
    @property
    def xyxy(self):
        x, y, w, h = self.xywh; return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    def activate(self, kf, frame_id, next_id):
        self.track_id = next_id; self.kf = kf; self.mean, self.cov = kf.initiate(self.xywh0); self.state = TRACKED
        self.is_activated = frame_id == 1; self.frame_id = self.end_frame = self.start_frame = frame_id; self.last_det = (frame_id, self.idx, self.xyxy.copy())
    def _absorb(self, det, frame_id):
        if POSP['model'] is not None: self.pp_inn = pos_inn(det.xywh0, self.mean)   # v32: innovation against the prediction, input of the position predictor
        self.mean, self.cov = self.kf.update(self.mean, self.cov, det.xywh0); self.state = TRACKED; self.is_activated = True
        self.frame_id = self.end_frame = frame_id; self.score, self.cls, self.idx = det.score, det.cls, det.idx
        x, y, w, h = det.xywh0; self.last_det = (frame_id, det.idx, np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
    re_activate = _absorb
    update = _absorb


def xyxy_of(T):
    """Vectorised Track.xyxy for a list (same elementwise arithmetic as the property)."""
    m = np.asarray([t.xywh for t in T], np.float64)
    x, y, w, h = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def iou_dist(A, B):
    if not A or not B: return np.zeros((len(A), len(B)), np.float32)
    a = xyxy_of(A); b = xyxy_of(B)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return (1 - inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)).astype(np.float32)


class Tracker:
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


def iou_xyxy(a, b):
    """IoU matrix of [n,4] and [m,4] x1,y1,x2,y2 boxes."""
    a = np.asarray(a, np.float64).reshape(-1, 4); b = np.asarray(b, np.float64).reshape(-1, 4)
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih; ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None, :] - inter, 1e-9)


def grow_xyxy(a, s):
    """C-BIoU buffered box: every side moved outwards by s x (width / height)."""
    a = np.asarray(a, np.float64).reshape(-1, 4); w = (a[:, 2] - a[:, 0]) * s; h = (a[:, 3] - a[:, 1]) * s
    return np.stack([a[:, 0] - w, a[:, 1] - h, a[:, 2] + w, a[:, 3] + h], 1)


class BotSortTracker(Tracker):
    """v38: BoT-SORT (Aharon et al., 2022) association cost on top of the ByteTrack path: min(IoU-score distance, appearance distance).
    Appearance distance = 0.5 * cosine distance, counted only when cosine distance < APP_THR and IoU distance < PROX_THR
    (paper defaults .25 / .5; not measured on our camera). Appearance can only LOWER a cost between boxes that already overlap."""
    APP_THR = 0.25; PROX_THR = 0.5

    def _dists(self, tracks, dets):
        d_iou = iou_dist(tracks, dets); d = d_iou
        if d.size:
            d = 1 - (1 - d_iou) * np.asarray([x.score for x in dets])[None]
            for a, t in enumerate(tracks):
                if t.feat is None: continue
                for b, x in enumerate(dets):
                    f = getattr(x, 'det_feat', None)
                    if f is None or d_iou[a, b] >= self.PROX_THR: continue
                    dc = 1.0 - float(t.feat @ f)
                    if dc < self.APP_THR and 0.5 * dc < d[a, b]: d[a, b] = 0.5 * dc; self.app_used = getattr(self, 'app_used', 0) + 1
        return d


class AltTracker(Tracker):
    """v31 (user 11-10): extra trackers for the live page's list. NOT the default and NOT measured on our camera.
    kind 'ocsort' = OC-SORT (Cao et al., CVPR 2023): first association by IoU + direction consistency (OCM, delta t 3, weight 0.2, IoU floor 0.3),
      low-score pass (BYTE), then unmatched tracks' LAST OBSERVATION vs unmatched detections (OCR, IoU >= 0.3); a track found again after a gap
      gets its Kalman state rebuilt from the last observation along a straight virtual path (ORU). Stored observations follow the camera-motion warp.
    kind 'biou' = buffered IoU of C-BIoU (Yang et al., WACV 2023): boxes grown by 0.3, leftovers by 0.5, on the Kalman prediction (the paper's own
      motion model is not used); the rest is the ByteTrack cascade.
    Kept from Tracker: thresholds .25 / .10, class gate, unconfirmed stage + n_init, edge birth rule, buffer 30, the returned rows."""
    def __init__(self, kind, **kw):
        assert kind in ('ocsort', 'biou', 'sort'), kind
        super().__init__('bytetrack', **kw); self.kind = kind
        if kind == 'sort': self.low = self.high; self.buffer = 1   # v36 SORT (Bewley et al., ICIP 2016): no low-score pass, a missed track is dropped after 1 frame (max_age 1), IoU >= 0.3, 3 hits to show
        self.dt, self.vdc, self.iou_min = 3, 0.2, 0.3; self.b1, self.b2 = 0.3, 0.5

    @staticmethod
    def _box(d):
        x, y, w, h = d.xywh0; return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

    def _prev(self, obs):
        """observation delta t frames ago (else the nearest later one, else the newest) = OC-SORT k_previous_obs"""
        for dt in range(self.dt, 0, -1):
            for f, o in obs:
                if f == self.frame_id - dt: return o
        return obs[-1][1] if obs else None

    def _note(self, t, det):
        b = self._box(det); obs = getattr(t, 'obs', []); p = self._prev(obs); t.vdir = None
        if p is not None:
            v = (b[:2] + b[2:]) / 2 - (p[:2] + p[2:]) / 2; n = float(np.hypot(v[0], v[1]))
            if n > 1e-6: t.vdir = v / n
        t.obs = (obs + [(self.frame_id, b)])[-8:]; t.kf_obs = (t.mean.copy(), t.cov.copy())

    def _take(self, t, det):
        """t absorbs det; OC-SORT ORU first when the track was unobserved for more than one frame"""
        if self.kind == 'ocsort':
            gap = self.frame_id - t.end_frame
            if gap > 1 and getattr(t, 'obs', None):
                (m, c), b = t.kf_obs, t.obs[-1][1]; z0 = np.r_[(b[:2] + b[2:]) / 2, b[2:] - b[:2]]; z1 = det.xywh0
                for k in range(1, gap):
                    m, c = KF.multi_predict(self.kf, m[None], c[None]); m, c = self.kf.update(m[0], c[0], z0 + (z1 - z0) * k / gap)
                m, c = KF.multi_predict(self.kf, m[None], c[None]); t.mean, t.cov = m[0], c[0]
        t.update(det, self.frame_id)
        if self.kind == 'ocsort': self._note(t, det)

    def _ocm(self, T, D):
        if not T or not D: return np.zeros((len(T), len(D)), np.float32)
        db = np.asarray([self._box(d) for d in D]); sc = np.asarray([d.score for d in D]); iou = iou_xyxy(xyxy_of(T), db)
        dc = (db[:, :2] + db[:, 2:]) / 2; ang = np.zeros_like(iou)
        for a, t in enumerate(T):
            v = getattr(t, 'vdir', None); p = self._prev(getattr(t, 'obs', []))
            if v is None or p is None: continue
            u = dc - (p[:2] + p[2:]) / 2; n = np.hypot(u[:, 0], u[:, 1])
            ang[a] = (np.pi / 2 - np.arccos(np.clip((u @ v) / np.maximum(n, 1e-6), -1, 1))) / np.pi * (n > 1e-6)
        cost = 1 - (iou + self.vdc * ang * sc[None]); cost[iou < self.iou_min] = 1.0
        return cost.astype(np.float32)

    def _bdist(self, T, D, s):
        if not T or not D: return np.zeros((len(T), len(D)), np.float32)
        iou = iou_xyxy(grow_xyxy(xyxy_of(T), s), grow_xyxy([self._box(d) for d in D], s))
        return (1 - iou * np.asarray([d.score for d in D])[None]).astype(np.float32)

    def update(self, arr, det_feats=None, feat_fn=None):
        self.frame_id += 1; act, refind, lost, removed = [], [], [], []; self.new_this_frame, self.lost_this_frame = [], []
        oc = self.kind == 'ocsort'
        wh = arr[:, 2:4] - arr[:, 0:2]; valid = (wh[:, 0] > 0) & (wh[:, 1] > 0)
        hi = np.flatnonzero(valid & (arr[:, 4] >= self.high)); lo = np.flatnonzero(valid & (arr[:, 4] > self.low) & (arr[:, 4] < self.high))
        mk = lambda i: Track(np.r_[(arr[i, :2] + arr[i, 2:4]) / 2, wh[i]], arr[i, 4], arr[i, 5], i)
        dets = [mk(i) for i in hi]; dets2 = [mk(i) for i in lo]
        W = getattr(self.kf, 'warp', None)
        if oc and W is not None:   # stored observations and the saved Kalman state follow the camera motion (same warp the prediction gets)
            R, tr = W[:2, :2], W[:2, 2]
            for t in self.tracked + self.lost:
                if not getattr(t, 'obs', None): continue
                nb = []
                for f, b in t.obs:
                    c = R @ ((b[:2] + b[2:]) / 2) + tr; s = (b[2:] - b[:2]) / 2; nb.append((f, np.r_[c - s, c + s]))
                t.obs = nb; m, c = gmc_warp(t.kf_obs[0][None].copy(), t.kf_obs[1], W); t.kf_obs = (m[0], c)
        unconfirmed = [t for t in self.tracked if not t.is_activated]; tracked = [t for t in self.tracked if t.is_activated]
        pool = tracked + [t for t in self.lost if t not in tracked]
        if pool:
            mean = np.asarray([t.mean for t in pool]); cov = np.asarray([t.cov for t in pool])
            for i, t in enumerate(pool):
                if t.state != TRACKED: mean[i, 6:8] = 0
            mean, cov = pos_predict(self.kf, pool, mean, cov)   # v32
            for i, t in enumerate(pool): t.mean, t.cov = mean[i], cov[i]
        if oc: m, ut, ud = linear_assignment(self._cg(self._ocm(pool, dets), pool, dets), 0.95)
        elif self.kind == 'sort': m, ut, ud = linear_assignment(self._cg(iou_dist(pool, dets), pool, dets), 1 - self.iou_min)   # v36: plain IoU on the Kalman prediction
        else:
            m, ut, ud = linear_assignment(self._cg(self._bdist(pool, dets, self.b1), pool, dets), self.match)
            if ut and ud:
                P = [pool[i] for i in ut]; D = [dets[j] for j in ud]
                mb, ub, db = linear_assignment(self._cg(self._bdist(P, D, self.b2), P, D), 0.7)
                m = m + [(ut[a], ud[b]) for a, b in mb]; ut, ud = [ut[a] for a in ub], [ud[b] for b in db]
        for i, j in m:
            t = pool[i]; was_lost = t.state == LOST; self._take(t, dets[j]); (refind if was_lost else act).append(t)
        rt = [pool[i] for i in ut if pool[i].state == TRACKED]; done = set()
        if rt and dets2:
            m2, _, _ = linear_assignment(self._cg(iou_dist(rt, dets2), rt, dets2), 0.5)
            for i, j in m2: self._take(rt[i], dets2[j]); act.append(rt[i]); done.add(id(rt[i]))
        if oc and ud:   # OCR: last observation of every still unmatched track vs the unmatched high-score detections
            P = [pool[i] for i in ut if id(pool[i]) not in done and getattr(pool[i], 'obs', None)]
            if P:
                D = [dets[j] for j in ud]; cost = (1 - iou_xyxy([t.obs[-1][1] for t in P], [self._box(d) for d in D])).astype(np.float32)
                mr, _, dr = linear_assignment(self._cg(cost, P, D), 1 - self.iou_min)
                for a, b in mr:
                    t = P[a]; was_lost = t.state == LOST; self._take(t, D[b]); (refind if was_lost else act).append(t); done.add(id(t))
                ud = [ud[b] for b in dr]
        for t in rt:
            if id(t) not in done and t.state != LOST: t.state = LOST; lost.append(t); self.lost_this_frame.append(t)
        dets = [dets[j] for j in ud]
        if unconfirmed:
            m3, uu, ud3 = linear_assignment(self._cg(self._dists(unconfirmed, dets), unconfirmed, dets), 0.7)
            for i, j in m3:
                self._take(unconfirmed[i], dets[j]); act.append(unconfirmed[i])
                if self.n_init > 2: self._tentative(unconfirmed[i])
            for i in uu: unconfirmed[i].state = REMOVED; removed.append(unconfirmed[i])
        else: ud3 = list(range(len(dets)))
        for j in ud3:
            if dets[j].score >= self.new and edge_ok(dets[j]):
                dets[j].activate(self.kf, self.frame_id, self._nid()); act.append(dets[j]); self.new_this_frame.append(dets[j])
                if oc: self._note(dets[j], dets[j])
        for t in self.lost:
            if self.frame_id - t.end_frame > self.buffer: t.state = REMOVED; removed.append(t)
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        for t in act + refind:
            if t not in self.tracked: self.tracked.append(t)
        self.lost = [t for t in self.lost if t.state == LOST and t not in self.tracked] + [t for t in lost if t not in self.tracked]
        self.lost = [t for t in self.lost if t.state != REMOVED]
        return [[*t.xyxy.tolist(), t.track_id, t.score, t.cls, t.idx] for t in self.tracked if t.is_activated]


class LostFirstTracker(Tracker):
    """--vit-budget N --vit-priority lostfirst (default off; ported 2026-10-23 from tracker_ab/vb_tracker.py BudgetTracker with
    gallery G1 / association A1 / p1 'gated' / priority 1234 = the pre-registered choice G1_A1_g3.5_s0.6_N2, VITBUDGET_KO.md).
    Every frame at most N person crops are embedded SYNCHRONOUSLY (batch 1 on NPU_2, one after another), ALL priorities together:
      P1 re-identification candidates = person detections (conf >= vit_conf, h >= vit_min_h) left unmatched by the first IoU
         association that lie inside the appearance gate (centre distance <= gate x max(h), size ratio .5-2) of at least one LOST
         track WITH a gallery and lost age <= 90, nearest first; embedded inside update() through feat_fn, BEFORE the appearance pass
      P2 at-risk tracked persons matched this frame (box overlaps another detection box conf > .10, or within 8 px of the image
         border, or conf < .40), oldest gallery feature first
      P3 person tracks that became LOST this frame without a feature from the last 5 frames: crop of their previous-frame detection
         (feat_fn_prev on the previous frame)
      P4 round-robin refresh of the other tracked persons, oldest feature first
    P2-P4 use what P1 left of the budget, after the association (they only feed the galleries). Gallery = Track.set_feat EMA
    (0.9/0.1). Association = the current hybrid rule (appearance-first pass on lost tracks with a gallery, cos >= sim; featured
    detections get no motion fallback; the rest = motion re-bind radius 2 x h, guard .3, buffer 60). No asynchronous queue.
    feat_fn(det_idx list) / feat_fn_prev(det_idx list) -> {det_idx: L2 feature}; a missing entry = no crop (not counted)."""
    IMG_W, IMG_H = 640.0, 512.0

    def __init__(self, budget=2, sim=0.6, gate=3.5, vit_conf=0.25, vit_min_h=32.0, risk_border=8.0, risk_conf=0.40, p3_recent=5, app_age=90,
                 radius=2.0, n_init=2):
        # radius / n_init (--radius / --n-init, defaults 2.0 / 2 = unchanged): same meaning as in Tracker; not verified offline with the ViT
        super().__init__('rebind', sim=sim, radius=radius, app=budget > 0, app_first=True, gate=gate, lost_max_age=app_age, vit_conf=vit_conf,
                         vit_min_h=vit_min_h, hybrid=True, sync_k=budget, n_init=n_init)
        self.N, self.app_age = int(budget), int(app_age)
        self.risk_border, self.risk_conf, self.p3_recent = float(risk_border), float(risk_conf), int(p3_recent)
        self.use = dict(P1=0, P2=0, P3=0, P4=0); self.demand = dict(P1=0, P2=0, P3=0, P4=0)
        self.per_frame = []          # (crops used, P1 demand, P2 demand, P3 demand, P4 demand) per frame

    def _add(self, t, f, frame):
        if f is None: return
        t.set_feat(f); t.feat_frame = frame

    def _gate_ok(self, t, d):
        th = max(float(t.xywh[3]), 1.0); dh = max(float(d.xywh0[3]), 1.0); r = dh / th
        if r < self.size_ratio[0] or r > self.size_ratio[1]: return None
        rel = float(np.hypot(d.xywh0[0] - t.xywh[0], d.xywh0[1] - t.xywh[1])) / max(th, dh)
        return rel if rel <= self.gate else None

    def update(self, arr, det_feats=None, feat_fn=None, feat_fn_prev=None):
        """arr [n,6] x1,y1,x2,y2,conf,cls (frame coords); det_feats must be None (no oracle here). Returns rows [x1,y1,x2,y2,id,conf,cls,det_idx]."""
        assert det_feats is None
        self.frame_id += 1; fid = self.frame_id; act, refind, lost, removed = [], [], [], []; self.new_this_frame, self.lost_this_frame = [], []
        used = 0; dem = [0, 0, 0, 0]
        wh = arr[:, 2:4] - arr[:, 0:2]; valid = (wh[:, 0] > 0) & (wh[:, 1] > 0)
        hi = np.flatnonzero(valid & (arr[:, 4] >= self.high)); lo = np.flatnonzero(valid & (arr[:, 4] > self.low) & (arr[:, 4] < self.high))
        mk = lambda i: Track(np.r_[(arr[i, :2] + arr[i, 2:4]) / 2, wh[i]], arr[i, 4], arr[i, 5], i)
        dets = [mk(i) for i in hi]; dets2 = [mk(i) for i in lo]
        unconfirmed = [t for t in self.tracked if not t.is_activated]; tracked = [t for t in self.tracked if t.is_activated]
        pool = tracked + [t for t in self.lost if t not in tracked]
        if pool:
            mean = np.asarray([t.mean for t in pool]); cov = np.asarray([t.cov for t in pool])
            for i, t in enumerate(pool):
                if t.state != TRACKED: mean[i, 6:8] = 0
            mean, cov = pos_predict(self.kf, pool, mean, cov)   # v32
            for i, t in enumerate(pool): t.mean, t.cov = mean[i], cov[i]
        m, ut, ud = linear_assignment(self._dists(pool, dets), self.match)
        for i, j in m:
            t = pool[i]; was_lost = t.state == LOST; t.update(dets[j], fid); (refind if was_lost else act).append(t)
        if self.app:
            lost_f = [i for i in ut if pool[i].state == LOST and pool[i].feat is not None and fid - pool[i].end_frame <= self.app_age]
            cands = [j for j in ud if dets[j].cls == PERSON_CLS and dets[j].score >= self.vit_conf and dets[j].xywh0[3] >= self.vit_min_h]
            want = []
            if cands:                                                  # P1 'gated': inside the gate of >= 1 lost track with a gallery, nearest first
                sc = []
                for j in cands:
                    r = [x for x in (self._gate_ok(pool[i], dets[j]) for i in lost_f) if x is not None]
                    if r: sc.append((min(r), j))
                want = [j for _, j in sorted(sc)]
            dem[0] = len(want)
            self.cand_frames += bool(want); self.cand_crops += len(want)
            if want and feat_fn is not None:
                take = want[:self.N]; got = feat_fn([dets[j].idx for j in take]) or {}
                for j in take:
                    f = got.get(dets[j].idx)
                    if f is not None: dets[j].det_feat = f; used += 1; self.use['P1'] += 1
            if want: self.sync_frames += 1; self.sync_crops += min(len(want), self.N)
            cand_f = [j for j in ud if getattr(dets[j], 'det_feat', None) is not None]
            if lost_f and cand_f:                                      # appearance-first pass (A1, G1)
                cost = np.ones((len(lost_f), len(cand_f)), np.float32)
                for a, i in enumerate(lost_f):
                    for b, j in enumerate(cand_f):
                        if self._gate_ok(pool[i], dets[j]) is None: continue
                        s = float(pool[i].feat @ dets[j].det_feat)
                        if s >= self.sim: cost[a, b] = 1.0 - s
                mm, _, _ = linear_assignment(cost, 0.999); tk, dk = set(), set()
                for a, b in mm:
                    i, j = lost_f[a], cand_f[b]; pool[i].re_activate(dets[j], fid); self._add(pool[i], dets[j].det_feat, fid)
                    refind.append(pool[i]); tk.add(i); dk.add(j); self.rebinds_app += 1
                ut = [i for i in ut if i not in tk]; ud = [j for j in ud if j not in dk]
        # motion fallback (hybrid: featured detections were decided by appearance only)
        lost_i = [i for i in ut if pool[i].state == LOST]
        cand = list(ud)
        if self.app: cand = [j for j in cand if getattr(dets[j], 'det_feat', None) is None]
        if cand and self.guard > 0:
            occ = [t for t in pool if t.state == TRACKED] + [t for t in act if t not in pool]
            if occ:
                iou = 1 - iou_dist(occ, [dets[j] for j in cand]); cand = [j for j, free in zip(cand, iou.max(0) <= self.guard) if free]
        if lost_i and cand:
            cost = np.ones((len(lost_i), len(cand)), np.float32)
            for a, i in enumerate(lost_i):
                t = pool[i]; tc, th = t.xywh[:2], max(float(t.xywh[3]), 1.0)
                for b, j in enumerate(cand):
                    d = dets[j]; rel = float(np.hypot(d.xywh0[0] - tc[0], d.xywh0[1] - tc[1])) / (self.radius * th)
                    if rel > 1.0: continue
                    df = getattr(d, 'det_feat', None)
                    if not self.app: cost[a, b] = 0.5 * rel
                    elif t.feat is None or df is None: cost[a, b] = 0.8 + 0.19 * rel
                    else:
                        s = float(t.feat @ df)
                        if s >= self.sim: cost[a, b] = 1.0 - s
            mm, _, _ = linear_assignment(cost, 0.999); tk, dk = set(), set()
            for a, b in mm:
                i, j = lost_i[a], cand[b]; pool[i].re_activate(dets[j], fid); refind.append(pool[i]); tk.add(i); dk.add(j)
                df = getattr(dets[j], 'det_feat', None)
                if df is not None: self._add(pool[i], df, fid)
                if cost[a, b] >= 0.8 or not self.app: self.rebinds_motion += 1
                else: self.rebinds_app += 1
            ut = [i for i in ut if i not in tk]; ud = [j for j in ud if j not in dk]
        rt = [pool[i] for i in ut if pool[i].state == TRACKED]
        if rt and dets2:
            m2, ut2, _ = linear_assignment(iou_dist(rt, dets2), 0.5)
            for i, j in m2: rt[i].update(dets2[j], fid); act.append(rt[i])
        else: ut2 = list(range(len(rt)))
        for i in ut2:
            if rt[i].state != LOST: rt[i].state = LOST; lost.append(rt[i]); self.lost_this_frame.append(rt[i])
        dets = [dets[j] for j in ud]
        if unconfirmed:
            m3, uu, ud3 = linear_assignment(self._dists(unconfirmed, dets), 0.7)
            for i, j in m3:
                unconfirmed[i].update(dets[j], fid); act.append(unconfirmed[i])
                if self.n_init > 2: self._tentative(unconfirmed[i])
                df = getattr(dets[j], 'det_feat', None)
                if df is not None: self._add(unconfirmed[i], df, fid)
            for i in uu: unconfirmed[i].state = REMOVED; removed.append(unconfirmed[i])
        else: ud3 = list(range(len(dets)))
        for j in ud3:
            if dets[j].score >= self.new and edge_ok(dets[j]):   # v24f
                dets[j].activate(self.kf, fid, self._nid()); act.append(dets[j]); self.new_this_frame.append(dets[j])
                if self.diag is not None: self._diag_new(dets[j])
                df = getattr(dets[j], 'det_feat', None)
                if df is not None: self._add(dets[j], df, fid)
        for t in self.lost:
            if fid - t.end_frame > self.buffer: t.state = REMOVED; removed.append(t)
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        for t in act + refind:
            if t not in self.tracked: self.tracked.append(t)
        self.lost = [t for t in self.lost if t.state == LOST and t not in self.tracked] + [t for t in lost if t not in self.tracked]
        self.lost = [t for t in self.lost if t.state != REMOVED]
        # ---- P2..P4: gallery refresh with what P1 left of the budget (sync; P2/P4 = this frame's crops, P3 = the previous frame's)
        if self.app:
            h_ok = lambda t: t.score >= self.vit_conf and (t.last_det[2][3] - t.last_det[2][1]) >= self.vit_min_h
            cur = [t for t in self.tracked if t.cls == PERSON_CLS and t.last_det is not None and t.last_det[0] == fid and t.feat_frame != fid and h_ok(t)]
            risk = []
            if cur:
                ab = arr[valid & (arr[:, 4] > self.low), :4].astype(np.float64); own = np.flatnonzero(valid & (arr[:, 4] > self.low))
                for t in cur:
                    x1, y1, x2, y2 = t.last_det[2]
                    r = t.score < self.risk_conf or x1 <= self.risk_border or y1 <= self.risk_border or x2 >= self.IMG_W - self.risk_border or y2 >= self.IMG_H - self.risk_border
                    if not r and len(ab):
                        iw = np.minimum(ab[:, 2], x2) - np.maximum(ab[:, 0], x1); ih = np.minimum(ab[:, 3], y2) - np.maximum(ab[:, 1], y1)
                        ov = (iw > 0) & (ih > 0) & (own != t.idx); r = bool(ov.any())
                    if r: risk.append(t)
            risk.sort(key=lambda t: t.feat_frame)
            p3 = [t for t in self.lost_this_frame if t.cls == PERSON_CLS and t.last_det is not None and t.last_det[0] == fid - 1
                  and t.feat_frame < fid - self.p3_recent and h_ok(t)]
            rs = set(map(id, risk)); p4 = sorted([t for t in cur if id(t) not in rs], key=lambda t: t.feat_frame)
            dem[1], dem[2], dem[3] = len(risk), len(p3), len(p4)
            for key, lst, fn, fr in (('2', risk, feat_fn, fid), ('3', p3, feat_fn_prev, fid - 1), ('4', p4, feat_fn, fid)):
                if fn is None: continue
                for t in lst:
                    if used >= self.N: break
                    f = (fn([t.last_det[1]]) or {}).get(t.last_det[1])
                    if f is None: continue
                    self._add(t, f, fr); used += 1; self.use['P' + key] += 1
        for k, v in zip(('P1', 'P2', 'P3', 'P4'), dem): self.demand[k] += v
        self.per_frame.append((used,) + tuple(dem))
        return [[*t.xyxy.tolist(), t.track_id, t.score, t.cls, t.idx] for t in self.tracked if t.is_activated]


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
