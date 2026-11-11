"""학습된 확인 판정기: 새로 생긴 추적을 화면에 내보낼지 작은 GRU 모델로 정한다 (GateTracker, NumpyGRUGate)."""
import json
from pathlib import Path
import numpy as np
import math
from ..common.shared_values import PERSON_CLS, TRACKED, LOST, REMOVED
from ..track.position_memory import pos_predict
from ..track.stabilizer import STAB_M
from ..track.trackers import linear_assignment, edge_ok, Track, iou_dist, Tracker


# ===================================================================================================================== v11 learned confirmation gate
GFEAT = ['matched', 'matched_low', 'conf', 'h', 'aspect', 'logh', 'speed', 'kf_iou', 'overlap', 'contrast', 'texture', 'hits', 'misses', 'cmiss', 'age', 'edge', 'lost_iou', 'lost_dist', 'valid']
GF = len(GFEAT); GK = 10; GWIN = GK + 1
GFRAME_W, GFRAME_H = 640.0, 512.0
# lost_iou / lost_dist: max IoU with, and min centre distance (/ h, clipped 5) to, the KF-predicted boxes of the LOST person tracks of the
# tracker (a new track near a lost one is likely the lost person re-appearing: confirming it early makes an ID switch)


def crop_stats(gray, box):
    """(contrast, texture) of a box on the gray frame: contrast = (mean inside - mean of a ring half a box wide) / (std ring + 1), texture = std inside"""
    if gray is None: return 0.0, 0.0
    H, W = gray.shape
    x1 = max(0, int(math.floor(box[0]))); y1 = max(0, int(math.floor(box[1]))); x2 = min(W, int(math.ceil(box[2]))); y2 = min(H, int(math.ceil(box[3])))
    if x2 <= x1 or y2 <= y1: return 0.0, 0.0
    ins = gray[y1:y2, x1:x2].astype(np.float32); n_in = ins.size; s_in = float(ins.sum()); q_in = float(np.square(ins).sum())
    m_in = s_in / n_in; v_in = max(q_in / n_in - m_in * m_in, 0.0)
    mx, my = max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)
    ox1, oy1, ox2, oy2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
    out = gray[oy1:oy2, ox1:ox2].astype(np.float32); n_out = out.size - n_in
    if n_out <= 0: return 0.0, math.sqrt(v_in) / 64.0
    s_out = float(out.sum()) - s_in; q_out = float(np.square(out).sum()) - q_in
    m_out = s_out / n_out; v_out = max(q_out / n_out - m_out * m_out, 0.0)
    return float(np.clip((m_in - m_out) / (math.sqrt(v_out) + 1.0), -8.0, 8.0)), math.sqrt(v_in) / 64.0


def box_iou_max(b, others):
    if not len(others): return 0.0
    o = np.asarray(others, np.float64).reshape(-1, 4)
    ix1 = np.maximum(b[0], o[:, 0]); iy1 = np.maximum(b[1], o[:, 1]); ix2 = np.minimum(b[2], o[:, 2]); iy2 = np.minimum(b[3], o[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1]); ao = (o[:, 2] - o[:, 0]) * (o[:, 3] - o[:, 1])
    return float((inter / np.maximum(a + ao - inter, 1e-9)).max())


class GateTracker(Tracker):
    def __init__(self, gate=None, thr_out=0.5, thr_low=0.0, tent_low=False, log=False, **kw):
        super().__init__('bytetrack', **kw)
        self.gate = gate; self.thr_out = float(thr_out); self.thr_low = float(thr_low); self.tent_low = bool(tent_low); self.log = bool(log)
        self.gray = None; self.rows_log = []        # log: per frame list of dicts (kind post/pre, track, box, window, s1)
        self.stats = dict(births=0, births_frame1=0, confirmed=0, confirm_at_hit={}, vetoed_tracks=0, low_blocked=0, low_eligible=0, tent_low_kept=0, tent_low_tried=0,
                          gate_calls=0, gate_rows=0)
        self.gate_time = 0.0

    # ---- per-track bookkeeping
    def _init_track(self, t):
        t.fh = []; t.hits_n = 0; t.miss_n = 0; t.cmiss = 0; t.birth = self.frame_id; t.prev_c = None; t.gconf = bool(t.is_activated); t.vetoed = False

    def _fvec(self, t, box, matched, low, conf, kf_iou, others, lost):
        h = max(float(box[3] - box[1]), 1.0); w = max(float(box[2] - box[0]), 1.0); c = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        sp = 0.0 if t.prev_c is None else math.hypot(c[0] - t.prev_c[0], c[1] - t.prev_c[1]) / h
        con, tex = crop_stats(self.gray, box)
        hits = t.hits_n + (1 if matched else 0); miss = t.miss_n + (0 if matched else 1); cm = 0 if matched else t.cmiss + 1
        edge = min(box[0], box[1] - 16.0, GFRAME_W - box[2], 496.0 - box[3]) / h
        if STAB_M[0] is not None: edge = max(min(box[0] - STAB_M[0][0], box[1] - 16.0 - STAB_M[0][1], GFRAME_W - STAB_M[0][2] - box[2], 496.0 - STAB_M[0][3] - box[3]) / h, min(edge, 0.0))   # stab: real picture edge
        lb = [b for i, b in lost if i != id(t)]; l_iou = box_iou_max(box, lb); l_dist = 5.0
        if lb:
            o = np.asarray(lb, np.float64).reshape(-1, 4); l_dist = min(5.0, float(np.hypot((o[:, 0] + o[:, 2]) / 2 - c[0], (o[:, 1] + o[:, 3]) / 2 - c[1]).min()) / h)
        return np.array([1.0 if matched else 0.0, 1.0 if low else 0.0, float(conf), h / 480.0, w / h, math.log(h), min(sp, 5.0), float(kf_iou), box_iou_max(box, others),
                         con, min(tex, 4.0), min(hits, 30) / 10.0, min(miss, 30) / 10.0, min(cm, 30) / 10.0, min(self.frame_id - t.birth, 60) / 30.0, float(np.clip(edge, -1.0, 2.0)),
                         l_iou, l_dist, 1.0], np.float32)

    def _window(self, t, cur):
        hist = t.fh[-GK:]; pad = GWIN - 1 - len(hist)
        rows = ([np.zeros(GF, np.float32)] * pad if pad > 0 else []) + list(hist) + [cur]
        return np.stack(rows, 0)

    def _commit(self, t, vec, matched):
        t.fh.append(vec)
        if len(t.fh) > 64: del t.fh[:-GWIN]
        if matched: t.hits_n += 1; t.cmiss = 0
        else: t.miss_n += 1; t.cmiss += 1
        b = t.xyxy; t.prev_c = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def _gate(self, wins):
        if not len(wins): return np.zeros(0)
        import time
        t0 = time.perf_counter(); p = np.asarray(self.gate(np.stack(wins, 0)), np.float64).reshape(-1); self.gate_time += time.perf_counter() - t0
        self.stats['gate_calls'] += 1; self.stats['gate_rows'] += len(wins); return p

    # ---- the base bytetrack update with the two gated decisions
    def update(self, arr, det_feats=None, feat_fn=None):
        self.frame_id += 1; act, refind, lost, removed = [], [], [], []; self.new_this_frame, self.lost_this_frame = [], []
        wh = arr[:, 2:4] - arr[:, 0:2]; valid = (wh[:, 0] > 0) & (wh[:, 1] > 0)
        hi = np.flatnonzero(valid & (arr[:, 4] >= self.high)); lo = np.flatnonzero(valid & (arr[:, 4] > self.low) & (arr[:, 4] < self.high))
        def mk(i):
            return Track(np.r_[(arr[i, :2] + arr[i, 2:4]) / 2, wh[i]], arr[i, 4], arr[i, 5], i)
        dets = [mk(i) for i in hi]; dets2 = [mk(i) for i in lo]
        unconfirmed = [t for t in self.tracked if not t.is_activated]; tracked = [t for t in self.tracked if t.is_activated]
        pool = tracked + [t for t in self.lost if t not in tracked]
        if pool:
            mean = np.asarray([t.mean for t in pool]); cov = np.asarray([t.cov for t in pool])
            for i, t in enumerate(pool):
                if t.state != TRACKED: mean[i, 6:8] = 0
            mean, cov = pos_predict(self.kf, pool, mean, cov)   # v32
            for i, t in enumerate(pool): t.mean, t.cov = mean[i], cov[i]
        pred = {id(t): t.xyxy.copy() for t in pool + unconfirmed}          # KF / warped box before any update this frame
        m, ut, ud = linear_assignment(self._cg(self._dists(pool, dets), pool, dets), self.match)
        for i, j in m:
            t = pool[i]; was_lost = t.state == LOST; t.update(dets[j], self.frame_id); (refind if was_lost else act).append(t)
        rt = [pool[i] for i in ut if pool[i].state == TRACKED]
        # ---- (b) low-score eligibility of confirmed tracked tracks (p_pre), and the pre rows
        s1_ids = {id(t) for t in act + refind}; s1_boxes = [t.xyxy for t in act + refind if t.cls == PERSON_CLS]
        lost_now = [(id(l), pred[id(l)]) for l in pool if l.state == LOST and l.cls == PERSON_CLS]      # lost at the start of the frame, still unmatched
        pre = {}
        if (self.gate is not None and self.thr_low > 0) or self.log:
            cand = [t for t in rt if t.cls == PERSON_CLS]
            for t in cand: pre[id(t)] = self._fvec(t, pred[id(t)], False, False, 0.0, 0.0, s1_boxes, lost_now)
        rt2 = rt
        if self.gate is not None and self.thr_low > 0 and pre:
            cand = [t for t in rt if t.cls == PERSON_CLS]; p = self._gate([self._window(t, pre[id(t)]) for t in cand])
            ok = set()
            for t, q in zip(cand, p):
                t.p_pre = float(q)
                if q >= self.thr_low: ok.add(id(t))
            self.stats['low_eligible'] += len(ok); self.stats['low_blocked'] += len(cand) - len(ok)
            rt2 = [t for t in rt if t.cls != PERSON_CLS or id(t) in ok]
        used2 = set()
        if rt2 and dets2:
            m2, ut2_, _ = linear_assignment(self._cg(iou_dist(rt2, dets2), rt2, dets2), 0.5)
            for i, j in m2: rt2[i].update(dets2[j], self.frame_id); act.append(rt2[i]); rt2[i].low_now = True; used2.add(j)
            matched2 = {id(rt2[i]) for i, _ in m2}
        else: matched2 = set()
        for t in rt:
            if id(t) not in matched2 and t.state != LOST: t.state = LOST; lost.append(t); self.lost_this_frame.append(t)
        dets = [dets[j] for j in ud]
        if unconfirmed:
            m3, uu, ud3 = linear_assignment(self._cg(self._dists(unconfirmed, dets), unconfirmed, dets), 0.7)
            for i, j in m3:
                unconfirmed[i].update(dets[j], self.frame_id); act.append(unconfirmed[i])
                if self.gate is None and self.n_init > 2: self._tentative(unconfirmed[i])
            # ---- tent_low: unmatched tentative person tracks may take a leftover low-score detection
            uu_keep = []
            if uu and (self.tent_low or self.log):
                tt = [unconfirmed[i] for i in uu if unconfirmed[i].cls == PERSON_CLS]
                for t in tt: pre[id(t)] = self._fvec(t, pred[id(t)], False, False, 0.0, 0.0, s1_boxes, lost_now)
                if self.tent_low and self.gate is not None and tt:
                    free = [j for j in range(len(dets2)) if j not in used2]
                    p = self._gate([self._window(t, pre[id(t)]) for t in tt]); self.stats['tent_low_tried'] += len(tt)
                    el = [t for t, q in zip(tt, p) if q >= self.thr_low]
                    if el and free:
                        d2 = [dets2[j] for j in free]
                        m4, _, _ = linear_assignment(self._cg(iou_dist(el, d2), el, d2), 0.5)
                        for a, b in m4:
                            t = el[a]; t.update(d2[b], self.frame_id); t.low_now = True; act.append(t); uu_keep.append(t); self.stats['tent_low_kept'] += 1
            for i in uu:
                if unconfirmed[i] not in uu_keep: unconfirmed[i].state = REMOVED; removed.append(unconfirmed[i])
        else: ud3 = list(range(len(dets)))
        for j in ud3:
            if dets[j].score >= self.new and edge_ok(dets[j]):   # v24f
                dets[j].activate(self.kf, self.frame_id, self._nid()); act.append(dets[j]); self.new_this_frame.append(dets[j]); self._init_track(dets[j])
                self.stats['births'] += 1; self.stats['births_frame1'] += int(dets[j].is_activated)
        for t in self.lost:
            if self.frame_id - t.end_frame > self.buffer: t.state = REMOVED; removed.append(t)
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        for t in act + refind:
            if t not in self.tracked: self.tracked.append(t)
        self.lost = [t for t in self.lost if t.state == LOST and t not in self.tracked] + [t for t in lost if t not in self.tracked]
        self.lost = [t for t in self.lost if t.state != REMOVED]
        # ---- per-frame features of every person track alive, then (a) the confirmation gate for tentative matched tracks
        if self.gate is not None or self.log:
            matched_t = [t for t in self.tracked if t.cls == PERSON_CLS and t.end_frame == self.frame_id]
            boxes = [t.xyxy for t in matched_t]
            lost_end = [(id(l), pred.get(id(l), l.xyxy)) for l in self.lost if l.state == LOST and l.cls == PERSON_CLS]   # incl. tracks lost this frame
            cur = {}
            for n, t in enumerate(matched_t):
                others = boxes[:n] + boxes[n + 1:]; low = bool(getattr(t, 'low_now', False)); t.low_now = False
                pb = pred.get(id(t)); kf_iou = 0.0 if pb is None else box_iou_max(t.xyxy, [pb])
                cur[id(t)] = self._fvec(t, t.xyxy, True, low, t.score, kf_iou, others, lost_end)
            tent = [t for t in matched_t if not t.gconf]; tent_ids = {id(t) for t in tent}
            if self.gate is not None and tent:
                p = self._gate([self._window(t, cur[id(t)]) for t in tent])
                for t, q in zip(tent, p):
                    t.p_post = float(q); t.is_activated = bool(q >= self.thr_out)
                    if t.is_activated:
                        t.gconf = True; self.stats['confirmed'] += 1; k = t.hits_n + 1; self.stats['confirm_at_hit'][k] = self.stats['confirm_at_hit'].get(k, 0) + 1
            elif self.gate is None:
                for t in tent:
                    if t.is_activated: t.gconf = True; self.stats['confirmed'] += 1; k = t.hits_n + 1; self.stats['confirm_at_hit'][k] = self.stats['confirm_at_hit'].get(k, 0) + 1
            if self.log:
                fr = []
                for t in matched_t:
                    fr.append(dict(kind='post', tid=t.track_id, box=t.xyxy.tolist(), win=self._window(t, cur[id(t)]), s1=id(t) in s1_ids, tent=id(t) in tent_ids,
                                   hit=t.hits_n + 1, conf_now=t.score, low=bool(cur[id(t)][1]), birth=t.birth == self.frame_id, gconf=bool(t.gconf)))
                for t in list(rt) + [t for t in removed if getattr(t, 'fh', None) is not None and id(t) in pre]:
                    if id(t) in pre and t.cls == PERSON_CLS:
                        fr.append(dict(kind='pre', tid=t.track_id, box=pred[id(t)].tolist(), win=self._window(t, pre[id(t)]), was_conf=bool(t.gconf), hit=t.hits_n, miss=t.miss_n, cmiss=t.cmiss,
                                       kept_low=bool(cur.get(id(t), np.zeros(GF))[1]) if id(t) in cur else False, then_lost=t.state == LOST, removed=t.state == REMOVED))
                self.rows_log.append(fr)
            for t in matched_t: self._commit(t, cur[id(t)], True)
            for t in self.lost:
                if t.cls == PERSON_CLS and getattr(t, 'fh', None) is not None:
                    pb = pred.get(id(t), t.xyxy); self._commit(t, self._fvec(t, pb, False, False, 0.0, 0.0, boxes, lost_end), False)
        return [[*t.xyxy.tolist(), t.track_id, t.score, t.cls, t.idx] for t in self.tracked if t.is_activated]




class NumpyGRUGate:
    """numpy forward of the frozen torch GRU32 gate (GATE_GRU32_FROZEN.json = state dict of failure_analysis_20261102/track_gate/cache/model_gru_full.pkl):
    windows [B, 11, 19] -> p(true person); h' = (1-z) n + z h, torch gate order r, z, n; head = linear on the last hidden state, sigmoid"""
    def __init__(self, path):
        import hashlib
        raw = Path(path).read_bytes(); self.sha = hashlib.sha256(raw).hexdigest(); sp = json.loads(raw.decode('utf-8')); self.spec = sp
        W = {k: np.asarray(v, np.float64) for k, v in sp['weights'].items()}; self.H = int(sp['hidden']); self.thr_out = float(sp['thr_out']); self.thr_low = float(sp['thr_low'])
        self.wih, self.whh, self.bih, self.bhh = W['gru.weight_ih_l0'], W['gru.weight_hh_l0'], W['gru.bias_ih_l0'], W['gru.bias_hh_l0']
        self.hw, self.hb = W['head.weight'], W['head.bias']
        assert sp['n_features'] == GF and sp['window'] == GWIN, 'gate spec does not match the tracker features'
    def __call__(self, X):
        X = np.asarray(X, np.float64); H = self.H; h = np.zeros((X.shape[0], H))
        for t in range(X.shape[1]):
            gi = X[:, t] @ self.wih.T + self.bih; gh = h @ self.whh.T + self.bhh
            r = 1.0 / (1.0 + np.exp(-(gi[:, :H] + gh[:, :H]))); z = 1.0 / (1.0 + np.exp(-(gi[:, H:2 * H] + gh[:, H:2 * H]))); n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h = (1.0 - z) * n + z * h
        return (1.0 / (1.0 + np.exp(-(h @ self.hw.T + self.hb)))).reshape(-1)
