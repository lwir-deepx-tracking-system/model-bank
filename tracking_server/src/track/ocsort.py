"""Observation-centric association and gap recovery for OC-SORT."""

import numpy as np

from ..common.shared_values import LOST
from .alt_tracker import AltTracker
from .kalman import KF, gmc_warp
from .tracker_common import iou_xyxy, linear_assignment, xyxy_of


class OCSortTracker(AltTracker):
    """IoU and observation direction first, then last-observation matching and Kalman recovery."""

    _tracker_kind = 'ocsort'

    def _prev(self, obs):
        """observation delta t frames ago (else the nearest later one, else the newest) = OC-SORT k_previous_obs"""
        for dt in range(self.dt, 0, -1):
            for f, o in obs:
                if f == self.frame_id - dt: return o
        return obs[-1][1] if obs else None

    def _note(self, t, det):
        b = self._box(det); obs = getattr(t, 'obs', []); p = self._prev(obs); t.vdir = None
        if p is not None:
            v = (b[:2] + b[2:]) / 2 - (p[:2] + p[2:]) / 2; n = float(np.hypot(v[0], v[1]))
            if n > 1e-6: t.vdir = v / n
        t.obs = (obs + [(self.frame_id, b)])[-8:]; t.kf_obs = (t.mean.copy(), t.cov.copy())

    def _take(self, t, det):
        """t absorbs det; OC-SORT ORU first when the track was unobserved for more than one frame"""
        gap = self.frame_id - t.end_frame
        if gap > 1 and getattr(t, 'obs', None):
            (m, c), b = t.kf_obs, t.obs[-1][1]
            z0 = np.r_[(b[:2] + b[2:]) / 2, b[2:] - b[:2]]
            z1 = det.xywh0
            for k in range(1, gap):
                m, c = KF.multi_predict(self.kf, m[None], c[None])
                m, c = self.kf.update(m[0], c[0], z0 + (z1 - z0) * k / gap)
            m, c = KF.multi_predict(self.kf, m[None], c[None])
            t.mean, t.cov = m[0], c[0]
        t.update(det, self.frame_id)
        self._note(t, det)

    def _ocm(self, T, D):
        if not T or not D: return np.zeros((len(T), len(D)), np.float32)
        db = np.asarray([self._box(d) for d in D]); sc = np.asarray([d.score for d in D]); iou = iou_xyxy(xyxy_of(T), db)
        dc = (db[:, :2] + db[:, 2:]) / 2; ang = np.zeros_like(iou)
        for a, t in enumerate(T):
            v = getattr(t, 'vdir', None); p = self._prev(getattr(t, 'obs', []))
            if v is None or p is None: continue
            u = dc - (p[:2] + p[2:]) / 2; n = np.hypot(u[:, 0], u[:, 1])
            ang[a] = (np.pi / 2 - np.arccos(np.clip((u @ v) / np.maximum(n, 1e-6), -1, 1))) / np.pi * (n > 1e-6)
        cost = 1 - (iou + self.vdc * ang * sc[None]); cost[iou < self.iou_min] = 1.0
        return cost.astype(np.float32)

    def _prepare_prediction(self):
        # Stored observations and the saved Kalman state follow the same warp as the prediction.
        warp = getattr(self.kf, 'warp', None)
        if warp is None:
            return
        rotation, translation = warp[:2, :2], warp[:2, 2]
        for track in self.tracked + self.lost:
            if not getattr(track, 'obs', None):
                continue
            observations = []
            for frame, box in track.obs:
                center = rotation @ ((box[:2] + box[2:]) / 2) + translation
                half_size = (box[2:] - box[:2]) / 2
                observations.append((frame, np.r_[center - half_size, center + half_size]))
            track.obs = observations
            mean, cov = gmc_warp(track.kf_obs[0][None].copy(), track.kf_obs[1], warp)
            track.kf_obs = (mean[0], cov)

    def _associate(self, pool, dets):
        return linear_assignment(self._cg(self._ocm(pool, dets), pool, dets), 0.95)

    def _recover(self, pool, ut, dets, ud, act, refind, done):
        # OCR matches last observations of the remaining tracks against unused high-score detections.
        if not ud:
            return ud
        tracks = [pool[i] for i in ut if id(pool[i]) not in done and getattr(pool[i], 'obs', None)]
        if not tracks:
            return ud
        remaining = [dets[j] for j in ud]
        cost = (1 - iou_xyxy([t.obs[-1][1] for t in tracks], [self._box(d) for d in remaining])).astype(np.float32)
        matches, _, unmatched = linear_assignment(self._cg(cost, tracks, remaining), 1 - self.iou_min)
        for track_index, det_index in matches:
            track = tracks[track_index]
            was_lost = track.state == LOST
            self._take(track, remaining[det_index])
            (refind if was_lost else act).append(track)
            done.add(id(track))
        return [ud[index] for index in unmatched]

    def _on_birth(self, det):
        self._note(det, det)
