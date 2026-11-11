"""칼만 필터: 상자의 위치·크기를 다음 프레임으로 예측하고, 카메라가 움직인 만큼 보정한다."""
import numpy as np


class KF:
    """Ultralytics KalmanFilterXYWH (constant velocity, state x,y,w,h,vx,vy,vw,vh)."""
    def __init__(self):
        self.F = np.eye(8); self.F[np.arange(4), np.arange(4, 8)] = 1.0; self.sp = 1 / 20; self.sv = 1 / 160; self.I8 = np.eye(8)
    def initiate(self, m):
        std = [2 * self.sp * m[2], 2 * self.sp * m[3], 2 * self.sp * m[2], 2 * self.sp * m[3], 10 * self.sv * m[2], 10 * self.sv * m[3], 10 * self.sv * m[2], 10 * self.sv * m[3]]
        return np.r_[m, np.zeros(4)], np.diag(np.square(std))
    def multi_predict(self, mean, cov):
        w, h = mean[:, 2], mean[:, 3]
        q = np.square(np.stack([self.sp * w, self.sp * h, self.sp * w, self.sp * h, self.sv * w, self.sv * h, self.sv * w, self.sv * h], 1))
        return mean @ self.F.T, self.F @ cov @ self.F.T + self.I8 * q[:, None, :]       # == diag(q) per track (x1.0 / x0.0 exact)
    def update(self, mean, cov, z):
        R = np.diag(np.square([self.sp * mean[2], self.sp * mean[3], self.sp * mean[2], self.sp * mean[3]]))
        pc = cov[:4, :4] + R; K = np.linalg.solve(pc, cov[:, :4].T).T
        return mean + (z - mean[:4]) @ K.T, cov - K @ pc @ K.T


def gmc_warp(mean, cov, H):
    """--gmc: Ultralytics STrack.multi_gmc (BoT-SORT): state (x,y,w,h,vx,vy,vw,vh) <- kron(I4, R) @ state, position += t (rows = tracks)."""
    R = H[:2, :2]; R8 = np.kron(np.eye(4), R); t = H[:2, 2]
    mean = mean @ R8.T; mean[:, :2] += t
    return mean, R8 @ cov @ R8.T


class GKF(KF):
    """--gmc: KF whose multi_predict (called once per update for the tracked + lost pool) applies the pending warp right after predicting."""
    warp = None
    def multi_predict(self, mean, cov):
        mean, cov = super().multi_predict(mean, cov)
        if self.warp is not None: mean, cov = gmc_warp(mean, cov, self.warp); self.warp = None
        return mean, cov
