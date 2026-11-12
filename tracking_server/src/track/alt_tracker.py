"""Lifecycle shared by OC-SORT, SORT, and C-BIoU."""

import numpy as np

from ..common.shared_values import LOST, REMOVED, TRACKED
from .bytetrack import ByteTrackTracker
from .position_memory import pos_predict
from .tracker_common import Track, edge_ok, iou_dist, linear_assignment


class AltTracker(ByteTrackTracker):
    """Shared association cascade and compatibility constructor for the alternate trackers."""

    def __new__(cls, kind=None, **kw):
        if cls is AltTracker:
            assert kind in ('ocsort', 'biou', 'sort'), kind
            from .biou import BIoUTracker
            from .ocsort import OCSortTracker
            from .sort import SortTracker

            cls = {'ocsort': OCSortTracker, 'biou': BIoUTracker, 'sort': SortTracker}[kind]
        return super().__new__(cls)

    def __init__(self, kind=None, **kw):
        assert kind is None or kind == self._tracker_kind, kind
        super().__init__('bytetrack', **kw)
        self.kind = self._tracker_kind
        self.dt, self.vdc, self.iou_min = 3, 0.2, 0.3
        self.b1, self.b2 = 0.3, 0.5

    @staticmethod
    def _box(d):
        x, y, w, h = d.xywh0; return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

    def _prepare_prediction(self):
        """Update any state that needs to follow camera motion before Kalman prediction."""

    def _associate(self, pool, dets):
        raise NotImplementedError

    def _take(self, track, det):
        track.update(det, self.frame_id)

    def _recover(self, pool, ut, dets, ud, act, refind, done):
        return ud

    def _on_birth(self, det):
        """Initialize any algorithm-specific state for a new track."""

    def update(self, arr, det_feats=None, feat_fn=None):
        self.frame_id += 1; act, refind, lost, removed = [], [], [], []; self.new_this_frame, self.lost_this_frame = [], []
        wh = arr[:, 2:4] - arr[:, 0:2]; valid = (wh[:, 0] > 0) & (wh[:, 1] > 0)
        hi = np.flatnonzero(valid & (arr[:, 4] >= self.high)); lo = np.flatnonzero(valid & (arr[:, 4] > self.low) & (arr[:, 4] < self.high))
        mk = lambda i: Track(np.r_[(arr[i, :2] + arr[i, 2:4]) / 2, wh[i]], arr[i, 4], arr[i, 5], i)
        dets = [mk(i) for i in hi]; dets2 = [mk(i) for i in lo]
        self._prepare_prediction()
        unconfirmed = [t for t in self.tracked if not t.is_activated]; tracked = [t for t in self.tracked if t.is_activated]
        pool = tracked + [t for t in self.lost if t not in tracked]
        if pool:
            mean = np.asarray([t.mean for t in pool]); cov = np.asarray([t.cov for t in pool])
            for i, t in enumerate(pool):
                if t.state != TRACKED: mean[i, 6:8] = 0
            mean, cov = pos_predict(self.kf, pool, mean, cov)   # v32
            for i, t in enumerate(pool): t.mean, t.cov = mean[i], cov[i]
        m, ut, ud = self._associate(pool, dets)
        for i, j in m:
            t = pool[i]; was_lost = t.state == LOST; self._take(t, dets[j]); (refind if was_lost else act).append(t)
        rt = [pool[i] for i in ut if pool[i].state == TRACKED]; done = set()
        if rt and dets2:
            m2, _, _ = linear_assignment(self._cg(iou_dist(rt, dets2), rt, dets2), 0.5)
            for i, j in m2: self._take(rt[i], dets2[j]); act.append(rt[i]); done.add(id(rt[i]))
        ud = self._recover(pool, ut, dets, ud, act, refind, done)
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
                self._on_birth(dets[j])
        for t in self.lost:
            if self.frame_id - t.end_frame > self.buffer: t.state = REMOVED; removed.append(t)
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        for t in act + refind:
            if t not in self.tracked: self.tracked.append(t)
        self.lost = [t for t in self.lost if t.state == LOST and t not in self.tracked] + [t for t in lost if t not in self.tracked]
        self.lost = [t for t in self.lost if t.state != REMOVED]
        return [[*t.xyxy.tolist(), t.track_id, t.score, t.cls, t.idx] for t in self.tracked if t.is_activated]
