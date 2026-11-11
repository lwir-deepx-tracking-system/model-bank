"""카메라 움직임 추정 (광학 흐름으로 이전 프레임 -> 현재 프레임 변환을 구함)과, 그 계산을 따로 도는 작업 스레드에 맡기는 함수."""
import time
import numpy as np
from ..common.state import APP


class GMC:
    """--gmc: Ultralytics trackers/utils/gmc.py method 'sparseOptFlow' (BoT-SORT default): downscale 2, Ultralytics feature params; returns the
    2x3 partial-affine warp previous -> current frame (identity on the first frame, without keypoints, or when <= 4 points survive / RANSAC fails)."""
    def __init__(self, downscale=2):
        self.ds = downscale; self.prev = None; self.kp = None; self.n_fail = 0
        self.fp = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)
    def apply(self, gray):
        import cv2
        h, w = gray.shape; f = cv2.resize(gray, (w // self.ds, h // self.ds)) if self.ds > 1 else gray
        H = np.eye(2, 3); kp = cv2.goodFeaturesToTrack(f, mask=None, **self.fp)
        if self.prev is None or self.kp is None:
            self.prev, self.kp = f.copy(), kp; return H
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, f, self.kp, None)
        ok = st.reshape(-1).astype(bool) if st is not None else np.zeros(0, bool)
        M = cv2.estimateAffinePartial2D(self.kp[ok], nxt[ok], method=cv2.RANSAC)[0] if ok.sum() > 4 else None
        if M is not None: H = M; H[0, 2] *= self.ds; H[1, 2] *= self.ds
        else: self.n_fail += 1
        self.prev, self.kp = f.copy(), kp
        return H


def gmc_job(o_, g_):
    t_ = time.perf_counter_ns(); W = o_.apply(g_); return W, (time.perf_counter_ns() - t_) / 1e6


def gmc_submit(k, g_):
    if k == 0 or APP.images[k]['extra_info']['video_id'] != APP.images[k - 1]['extra_info']['video_id']:
        o_ = GMC(APP.args.gmc_ds); o_.fp = dict(o_.fp, maxCorners=APP.args.gmc_corners, minDistance=APP.args.gmc_mindist); APP.gmc_objs.append(o_)
    APP.gmc_futs[k] = APP.gmc_pool.submit(APP.gmc_job, APP.gmc_objs[-1], g_)


def gmc_submit_2(k, g_): return None
