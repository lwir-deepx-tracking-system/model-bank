"""Legacy C-BIoU association using buffered boxes on the Kalman prediction."""

import numpy as np

from .alt_tracker import AltTracker
from .tracker_common import iou_xyxy, linear_assignment, xyxy_of


def grow_xyxy(a, s):
    """C-BIoU buffered box: every side moved outwards by s x (width / height)."""
    a = np.asarray(a, np.float64).reshape(-1, 4); w = (a[:, 2] - a[:, 0]) * s; h = (a[:, 3] - a[:, 1]) * s
    return np.stack([a[:, 0] - w, a[:, 1] - h, a[:, 2] + w, a[:, 3] + h], 1)


class BIoUTracker(AltTracker):
    _tracker_kind = 'biou'

    def _bdist(self, T, D, s):
        if not T or not D: return np.zeros((len(T), len(D)), np.float32)
        iou = iou_xyxy(grow_xyxy(xyxy_of(T), s), grow_xyxy([self._box(d) for d in D], s))
        return (1 - iou * np.asarray([d.score for d in D])[None]).astype(np.float32)

    def _associate(self, pool, dets):
        matches, unmatched_tracks, unmatched_dets = linear_assignment(
            self._cg(self._bdist(pool, dets, self.b1), pool, dets), self.match
        )
        if not unmatched_tracks or not unmatched_dets:
            return matches, unmatched_tracks, unmatched_dets
        tracks = [pool[i] for i in unmatched_tracks]
        remaining = [dets[j] for j in unmatched_dets]
        second_matches, tracks_left, dets_left = linear_assignment(
            self._cg(self._bdist(tracks, remaining, self.b2), tracks, remaining), 0.7
        )
        matches += [(unmatched_tracks[i], unmatched_dets[j]) for i, j in second_matches]
        return matches, [unmatched_tracks[i] for i in tracks_left], [unmatched_dets[j] for j in dets_left]
