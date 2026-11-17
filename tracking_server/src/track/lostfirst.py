"""Lost-first association with a synchronous per-frame appearance budget."""

import numpy as np

from ..common.shared_values import LOST, PERSON_CLS, REMOVED, TRACKED
from .bytetrack import ByteTrackTracker
from .position_memory import pos_predict
from .tracker_common import Track, edge_ok, iou_dist, linear_assignment


class LostFirstTracker(ByteTrackTracker):
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
