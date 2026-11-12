"""SORT association with plain IoU on the Kalman prediction."""

from .alt_tracker import AltTracker
from .tracker_common import iou_dist, linear_assignment


class SortTracker(AltTracker):
    _tracker_kind = 'sort'

    def __init__(self, kind=None, **kw):
        super().__init__(kind, **kw)
        # SORT does not associate low-score detections and uses max_age=1.
        self.low = self.high
        self.buffer = 1

    def _associate(self, pool, dets):
        return linear_assignment(self._cg(iou_dist(pool, dets), pool, dets), 1 - self.iou_min)
