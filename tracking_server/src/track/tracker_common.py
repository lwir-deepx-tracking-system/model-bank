"""Shared track state, box geometry, and assignment for the tracking server."""

import numpy as np

from ..common import settings
from ..common.shared_values import PERSON_CLS, TRACKED
from .position_memory import POSP, pos_inn
from .stabilizer import STAB_M


try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None


ASSIGN = dict(cost_first=False, depth_order=False)   # 11-20 page switches 'cost_first', 'depth_order' (BoT-SORT, Python and C++): copied onto the live tracker every frame (app.py)

# 11-20 depth_order: which overlaps count and how 'in front' is decided (the C++ twin reads its own copy of these, set from Python)
DEPTH_IOU, DEPTH_CONTAIN, DEPTH_Y2, DEPTH_H = 0.5, 0.6, 0.02, 1.05


def depth_order_blocks(cost, pool, thresh, iou_thr=DEPTH_IOU, contain_thr=DEPTH_CONTAIN):
    """11-20 switch 'depth_order' (BoT-SORT first association, default off). User: when two people overlap almost completely, the hidden
    (back) person's number slides onto the front person's box. For every pair of seen (TRACKED, confirmed) person tracks whose predicted
    boxes overlap (IoU >= iou_thr or intersection / smaller area >= contain_thr), the one in front = bottom edge lower in the picture by more
    than DEPTH_Y2 x the taller height, else taller by more than DEPTH_H (else undecided: no action). When both tracks' best detection (lowest
    cost, first on ties) is the same one and both may take it (cost <= thresh), it is forbidden to the one behind; that track stays unmatched
    this frame unless it fits another detection (second association, then the lost path, number kept). A first version forbade every detection
    both MAY take: it also took the hidden person's own box from it and measured worse; letting lost tracks be the one behind measured far
    worse (HANDOFF 11-20).
    Returns the (track row, detection column) pairs to forbid, in a fixed order (same order and arithmetic in native_tracker_core.hpp)."""
    if cost.size == 0: return []
    idx = [i for i, t in enumerate(pool) if t.state == TRACKED and t.is_activated and t.cls == PERSON_CLS]
    if len(idx) < 2: return []
    b = xyxy_of([pool[i] for i in idx]); out = []
    for p in range(len(idx)):
        for q in range(p + 1, len(idx)):
            x1, y1, x2, y2 = (float(v) for v in b[p]); u1, v1, u2, v2 = (float(v) for v in b[q])
            iw = min(x2, u2) - max(x1, u1); ih = min(y2, v2) - max(y1, v1)
            if iw <= 0 or ih <= 0: continue
            inter = iw * ih; ap = (x2 - x1) * (y2 - y1); aq = (u2 - u1) * (v2 - v1)
            if inter / max(ap + aq - inter, 1e-9) < iou_thr and inter / max(min(ap, aq), 1e-9) < contain_thr: continue
            hp, hq = y2 - y1, v2 - v1; m = DEPTH_Y2 * max(hp, hq)
            if y2 > v2 + m: front, back = idx[p], idx[q]
            elif v2 > y2 + m: front, back = idx[q], idx[p]
            elif hp > DEPTH_H * hq: front, back = idx[p], idx[q]
            elif hq > DEPTH_H * hp: front, back = idx[q], idx[p]
            else: continue
            j = int(np.argmin(cost[front]))
            if int(np.argmin(cost[back])) == j and cost[front, j] <= thresh and cost[back, j] <= thresh: out.append((back, j))
    return out


def linear_assignment(cost, thresh, cost_first=False, owner=None):
    if cost.size == 0: return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    if cost.shape == (1, 1):                                   # trivial case: no solver call
        return ([(0, 0)], [], []) if cost[0, 0] <= thresh else ([], [0], [0])
    c = np.where(cost > thresh, 1e6, cost)
    r, k = _lsa(c)
    matches = [(int(i), int(j)) for i, j in zip(r, k) if cost[i, j] <= thresh]
    if cost_first:   # 11-20 switch 'cost_first' (BoT-SORT, default off): see _cost_first; owner.cost_first_changed counts the calls it changed
        cf = _cost_first(cost, thresh)
        if owner is not None and sorted(cf) != sorted(matches): owner.cost_first_changed += 1
        matches = cf
    ma = {i for i, _ in matches}; mb = {j for _, j in matches}
    return matches, [i for i in range(cost.shape[0]) if i not in ma], [j for j in range(cost.shape[1]) if j not in mb]


def _cost_first(cost, thresh):
    """11-20 (user: the front person takes the hidden person's number): the assignment of the original ByteTrack / BoT-SORT code
    (lap.lapjv(extend_cost=True, cost_limit=thresh)): leaving a track or a detection unpaired costs thresh / 2, so a pair is made only when
    it lowers the total cost. The rule above (pairs over the limit = 1e6) first makes as many pairs as possible, which can move a seen
    track onto the next person so that a lost track gets a pair too (two pairs at 0.85 + 0.59 instead of one at 0.15).
    Solved on the n x m matrix: allowed pair -> cost - thresh (<= 0), forbidden -> 0; the best full matching of it holds the best set of
    allowed pairs (padding pairs cost 0). Same arithmetic and tie order in the C++ twin (native_tracker_core.hpp linear_assignment)."""
    r, k = _lsa(np.where(cost > thresh, 0.0, cost - thresh))
    return [(int(i), int(j)) for i, j in zip(r, k) if cost[i, j] <= thresh]


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


def iou_xyxy(a, b):
    """IoU matrix of [n,4] and [m,4] x1,y1,x2,y2 boxes."""
    a = np.asarray(a, np.float64).reshape(-1, 4); b = np.asarray(b, np.float64).reshape(-1, 4)
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih; ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None, :] - inter, 1e-9)
