"""Online lost-track ID relabel with the M1 crop budget (Claude 2026-11-02, user "M1에서해"; rule = PREREG_RELABEL.json, budget =
PREREG_RELABEL_M1.json). The same rule as relabel.relabel(), frame by frame, but appearance features exist only for the crops that were
embedded this frame:
  - new: every frame, up to NEW_K crops of NEW output IDs still awaiting their decision (clean detection this frame), smallest ID first;
  - refresh: on every EVERY-th frame of a video (per-video frame counter % EVERY == 0), up to K more crops of clean output person tracks,
    fewest gallery features first, then the oldest last feature, then the smallest ID.
budget=False embeds every clean output person track every frame (= relabel.relabel(), used for the regression check).
Clean = person output row with a detection key k in P (person detections, conf >= 0.1), detection height >= 32 px and no IoU > 0 with any
other detection in P. The tracker output (boxes, IDs before relabel, confirmation) is not changed; only person IDs are mapped.
embed_async (v7 --rl-async, budget mode only): the REFRESH crops are submitted asynchronously and their features join the galleries at the
start of the next step (refresh candidates then exclude IDs awaiting a decision, so no decision uses them); with new_k >= the number of
pending IDs this gives exactly the synchronous result. numpy only (runs on the M1 host and on the PC)."""
from collections import deque
import numpy as np
THR, LOST_MAX, GATE_H, GATE_PER10, HR, WAIT, KEEP, MIN_H = 0.596, 90, 3.5, 1.0, (0.5, 2.0), 3, 5, 32.0
# 11-14 (user, object-permanence reel): growing search range centred on the PREDICTED place. radius = last height x min(SR_BASE + SR_GROW x lost frames, SR_MAX);
# velocity = mean centre shift per frame over the last seen frames (at most SR_VMAX heights per frame). Shared with the overlay (track/position_memory.py).
SR_BASE, SR_GROW, SR_MAX, SR_VMAX, SR_HIST = 0.5, 0.05, 4.0, 0.15, 6


def search_range(hist, el, lh):
    """hist: [(t, cx, cy), ...] of the last seen frames (oldest first); el: lost frames; lh: last height -> (cx, cy, radius) of the search range."""
    t1, x1, y1 = hist[-1]; vx = vy = 0.0
    if len(hist) >= 2:
        t0, x0, y0 = hist[0]; dt = max(t1 - t0, 1); vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        s = (vx * vx + vy * vy) ** 0.5; m = SR_VMAX * lh
        if s > m: vx, vy = vx * m / s, vy * m / s
    return x1 + vx * el, y1 + vy * el, lh * min(SR_BASE + SR_GROW * el, SR_MAX)


def iou_any(b, others):
    for o in others:
        if min(b[2], o[2]) > max(b[0], o[0]) and min(b[3], o[3]) > max(b[1], o[1]):
            return True
    return False


class OnlineRelabel:
    def __init__(self, budget=True, every=2, k=2, new_k=2):
        self.budget, self.every, self.k, self.new_k = budget, every, k, new_k
        self.pred_gate = True   # 11-14: growing search range on the predicted place (page switch; on by default after the replay check: ID switches 8 -> 4, 0 -> 0, 2 -> 2); False = the original gate
        self.stats = dict(new_ids=0, decided_with_feature=0, relabelled=0, retired_reappearances=0, frames=0, crops_new=0, crops_refresh=0, crops_embedded=0)
        self.events, self.per_frame = [], []
        self.reset()

    def reset(self):
        """call at the first frame of every video (drains an asynchronous refresh of the previous video; its features are dropped)"""
        if getattr(self, 'deferred', None) is not None: self.deferred[2]()
        self.deferred = None
        self.t = -1; self.fresh = 10 ** 6
        self.id_map, self.retired, self.gallery, self.last, self.pending, self.last_emb = {}, {}, {}, {}, {}, {}
        self.hist = {}   # 11-14: output id -> deque of (t, cx, cy) of the last seen frames

    def step(self, rows, P, embed, frame=None, embed_async=None):
        """rows: this frame's tracker rows [x1, y1, x2, y2, id, conf, cls, k, ...]; P: {k: [x1, y1, x2, y2]} person detections (conf >= 0.1)
        keyed like the rows' k; embed(list of k) -> {k: L2-normalised feature} for the crops it could embed. Returns the rows (same order,
        same boxes) with relabelled person IDs."""
        self.t += 1; t = self.t; st = self.stats; st['frames'] += 1
        if self.deferred is not None:                                     # embed_async: last frame's refresh features join the galleries now
            t_prev, recs_d, collect_d = self.deferred; self.deferred = None; got_d = collect_d() or {}
            for bid_, oid_, k_ in recs_d:
                if k_ in got_d:
                    self.gallery.setdefault(oid_, deque(maxlen=KEEP)).append(got_d[k_]); self.last_emb[oid_] = t_prev; st['crops_embedded'] += 1
        prow = [r for r in rows if int(r[6]) == 0]
        elig = {}
        for r in prow:
            k = int(r[7])
            if k >= 0 and k in P and P[k][3] - P[k][1] >= MIN_H and not iou_any(P[k], [b for j, b in P.items() if j != k]):
                elig[int(r[4])] = k
        for r in prow:                                                    # new tracker IDs / reappearing retired ones
            bid = int(r[4])
            if bid in self.retired and self.retired[bid] is None:
                self.id_map[bid] = self.fresh; self.fresh += 1; self.retired[bid] = self.id_map[bid]; st['retired_reappearances'] += 1
            if bid not in self.id_map:
                self.id_map[bid] = bid; self.pending[bid] = dict(t0=t, feats=[]); st['new_ids'] += 1
        if not self.budget:
            req = sorted(elig.items()); n_new = 0
        else:
            req = [(bid, k) for bid, k in sorted(elig.items()) if bid in self.pending][:self.new_k]; n_new = len(req)
            if t % self.every == 0:
                taken = {bid for bid, _ in req}
                cand = sorted((len(self.gallery.get(self.id_map[bid], ())), self.last_emb.get(self.id_map[bid], -1), bid, k) for bid, k in elig.items()
                              if bid not in taken and (embed_async is None or bid not in self.pending))
                req += [(bid, k) for _, _, bid, k in cand[:self.k]]
        recs = None
        if embed_async is not None and self.budget and len(req) > n_new:   # refresh crops asynchronous, new-ID crops now
            got = (embed([k for _, k in req[:n_new]]) or {}) if n_new else {}
            recs = req[n_new:]; collect = embed_async([k for _, k in recs])
        else:
            got = (embed([k for _, k in req]) or {}) if req else {}
        clean = {bid: got[k] for bid, k in req if k in got}
        st['crops_new'] += n_new; st['crops_refresh'] += len(req) - n_new; st['crops_embedded'] += len(clean); self.per_frame.append((len(req), len(clean)))
        current_out = {self.id_map[int(r[4])] for r in prow}
        decide = []
        for bid, pnd in list(self.pending.items()):
            if bid in clean:
                pnd['feats'].append(clean[bid])
            if pnd['feats']:
                decide.append(bid)
            elif t - pnd['t0'] >= WAIT:
                del self.pending[bid]
        pairs = []
        for bid in decide:
            f = np.mean(self.pending[bid]['feats'], 0); f /= max(float(np.linalg.norm(f)), 1e-12)
            box = next(r[:4] for r in prow if int(r[4]) == bid); h = box[3] - box[1]; cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            for oid, g in self.gallery.items():
                if oid in current_out or oid == self.id_map[bid] or not g:
                    continue
                lt, lb = self.last[oid]; el = t - lt; lh = lb[3] - lb[1]
                if el > LOST_MAX or el <= 0 or not (HR[0] <= h / max(lh, 1e-6) <= HR[1]):
                    continue
                if self.pred_gate and oid in self.hist:   # 11-14: predicted place, growing radius
                    gx, gy, gr = search_range(self.hist[oid], el, lh)
                    if ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5 > gr: continue
                else:
                    d = ((cx - (lb[0] + lb[2]) / 2) ** 2 + (cy - (lb[1] + lb[3]) / 2) ** 2) ** 0.5
                    if d > GATE_H * lh + GATE_PER10 * lh * el / 10.0:
                        continue
                gm = np.mean(g, 0); gm /= max(float(np.linalg.norm(gm)), 1e-12); s = float(f @ gm)
                if s >= THR:
                    pairs.append((s, bid, oid))
        used_b, used_o = set(), set()
        for s, bid, oid in sorted(pairs, reverse=True):
            if bid in used_b or oid in used_o:
                continue
            used_b.add(bid); used_o.add(oid)
            for ob, om in list(self.id_map.items()):                     # the tracker ID that owned oid is retired
                if om == oid and ob != bid:
                    self.retired[ob] = None
            old_new = self.id_map[bid]; self.id_map[bid] = oid; st['relabelled'] += 1
            self.gallery[oid].extend(self.gallery.pop(old_new, [])); self.last.pop(old_new, None); self.last_emb.pop(old_new, None); self.hist.pop(old_new, None)
            self.events.append(dict(frame=frame, t=t, tracker_id=bid, new_id=old_new, old_id=oid, cos=round(s, 4)))
        for bid in decide:
            self.pending.pop(bid, None); st['decided_with_feature'] += 1
        if recs is not None: self.deferred = (t, [(bid, self.id_map[bid], k) for bid, k in recs], collect)
        out = []
        for r in rows:
            rr = list(r)
            if int(r[6]) == 0:
                bid = int(r[4]); oid = self.id_map[bid]; rr[4] = oid
                if bid in clean:
                    self.gallery.setdefault(oid, deque(maxlen=KEEP)).append(clean[bid]); self.last_emb[oid] = t
                self.last[oid] = (t, r[:4]); self.hist.setdefault(oid, deque(maxlen=SR_HIST)).append((t, (r[0] + r[2]) / 2, (r[1] + r[3]) / 2))
            out.append(rr)
        return out
