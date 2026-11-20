"""BoT-SORT appearance cost on the ByteTrack association cascade."""

import numpy as np

from .bytetrack import ByteTrackTracker
from .tracker_common import iou_dist


class BotSortTracker(ByteTrackTracker):
    """v38: BoT-SORT (Aharon et al., 2022) association cost on top of the ByteTrack path: min(IoU-score distance, appearance distance).
    Appearance distance = 0.5 * cosine distance, counted only when cosine distance < APP_THR and IoU distance < PROX_THR
    (paper defaults .25 / .5; not measured on our camera). Appearance can only LOWER a cost between boxes that already overlap."""
    APP_THR = 0.25; PROX_THR = 0.5
    
    def __init__(self, mode, new=0.5, match=0.9, unconfirmed=0.6, **kw):
        # runtime bench_mot17 스윕 (YOLO26s, 18개 시퀀스): IDSW 1,001 → 566, HOTA 58.2 → 60.7
        super().__init__(mode, **kw)
        self.new, self.match, self.unconfirmed = new, match, unconfirmed

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

    # 11-20 kpt_assoc (measurement switch, Python BoT-SORT only, default off; no C++ twin): torso-keypoint agreement (OKS) as an extra cost
    # in the first association, only for tracks whose predicted box overlaps another track's. Constants: id_steal_20261120/merge/RULE_BEFORE_RUN.txt
    KPT_W, KPT_OVL, KPT_VIS, KPT_IDX, KPT_K = 0.2, 0.3, 0.3, (5, 6, 11, 12), np.array([0.158, 0.158, 0.214, 0.214])
    kpt_acted = 0

    @staticmethod
    def _kpt_find(store, box):
        """keypoints [17, 3] of the stored detection with this box (IoU >= 0.9), else None"""
        if not store or not len(store['boxes']): return None
        b = np.asarray(store['boxes'], np.float64)
        iw = np.clip(np.minimum(b[:, 2], box[2]) - np.maximum(b[:, 0], box[0]), 0, None); ih = np.clip(np.minimum(b[:, 3], box[3]) - np.maximum(b[:, 1], box[1]), 0, None)
        inter = iw * ih; u = inter / np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter, 1e-9)
        j = int(np.argmax(u)); return np.asarray(store['kpts'][j], np.float64) if u[j] >= 0.9 else None

    def _kpt_cost(self, d, tracks, dets):
        if d.size == 0 or len(tracks) < 2: return d
        try:
            from ..detect.detect_frame import KP
        except Exception:
            return d
        cur, prev = KP.get('cur'), KP.get('prev')
        if not cur or not prev: return d
        ov = 1.0 - iou_dist(tracks, tracks); np.fill_diagonal(ov, 0.0)
        rows = [a for a in range(len(tracks)) if ov[a].max() >= self.KPT_OVL]
        if not rows: return d
        I = list(self.KPT_IDX); dk = {}
        for b, x in enumerate(dets):
            k = self._kpt_find(cur, x.xyxy)
            if k is not None and (k[I, 2] >= self.KPT_VIS).sum() >= 3: dk[b] = (k[I], max(float(x.xywh0[2] * x.xywh0[3]), 1.0))
        if not dk: return d
        d = d.copy()
        for a in rows:
            t = tracks[a]
            if t.last_det is None or t.last_det[0] != self.frame_id - 1: continue
            k = self._kpt_find(prev, t.last_det[2])
            if k is None or (k[I, 2] >= self.KPT_VIS).sum() < 3: continue
            lb = t.last_det[2]; sh = (t.xywh[0] - 0.5 * (lb[0] + lb[2]), t.xywh[1] - 0.5 * (lb[1] + lb[3])); tk = k[I]
            for b, (kk, area) in dk.items():
                if d[a, b] > self.match: continue
                vis = (tk[:, 2] >= self.KPT_VIS) & (kk[:, 2] >= self.KPT_VIS)
                if vis.sum() < 3: continue
                e = (tk[vis, 0] + sh[0] - kk[vis, 0]) ** 2 + (tk[vis, 1] + sh[1] - kk[vis, 1]) ** 2
                oks = float(np.mean(np.exp(-e / (2.0 * area * self.KPT_K[vis] ** 2))))
                d[a, b] = min(d[a, b] + self.KPT_W * (1.0 - oks), self.match); self.kpt_acted += 1
        return d
