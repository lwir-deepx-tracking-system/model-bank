"""BoT-SORT appearance cost on the ByteTrack association cascade."""

import numpy as np

from .bytetrack import ByteTrackTracker
from .tracker_common import iou_dist


class BotSortTracker(ByteTrackTracker):
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
