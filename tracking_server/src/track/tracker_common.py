"""Shared track state, box geometry, and assignment for the tracking server."""

import numpy as np

from ..common import settings
from ..common.shared_values import TRACKED
from .position_memory import POSP, pos_inn
from .stabilizer import STAB_M


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


def iou_xyxy(a, b):
    """IoU matrix of [n,4] and [m,4] x1,y1,x2,y2 boxes."""
    a = np.asarray(a, np.float64).reshape(-1, 4); b = np.asarray(b, np.float64).reshape(-1, 4)
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih; ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None, :] - inter, 1e-9)
