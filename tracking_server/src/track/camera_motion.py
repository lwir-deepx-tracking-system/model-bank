"""카메라 움직임 추정 (광학 흐름으로 이전 프레임 -> 현재 프레임 변환을 구함)과, 그 계산을 따로 도는 작업 스레드에 맡기는 함수."""
import time
from dataclasses import dataclass
import numpy as np
from ..common.state import APP


@dataclass(frozen=True, eq=False)
class MotionEstimate:
    warp: np.ndarray
    valid: bool
    previous_frame: object
    frame: object

    def paired(self, previous_frame, frame, previous_time, frame_time):
        if (self.valid and previous_time is not None and frame_time is not None
                and self.previous_frame is previous_frame and self.frame is frame):
            return previous_time, frame_time, self.warp
        return None


class GMC:
    """--gmc: Ultralytics trackers/utils/gmc.py method 'sparseOptFlow' (BoT-SORT default): downscale 2, Ultralytics feature params; returns the
    2x3 partial-affine warp previous -> current frame (identity on the first frame, without keypoints, or when <= 4 points survive / RANSAC fails)."""
    def __init__(self, downscale=2):
        self.ds = downscale; self.prev = None; self.kp = None; self.n_fail = 0
        self.previous_frame = None
        self.fp = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)

    def apply(self, gray):
        return self.estimate(gray).warp

    def estimate(self, gray):
        import cv2
        h, w = gray.shape; f = cv2.resize(gray, (w // self.ds, h // self.ds)) if self.ds > 1 else gray
        H = np.eye(2, 3); kp = cv2.goodFeaturesToTrack(f, mask=None, **self.fp)
        previous_frame = self.previous_frame
        self.previous_frame = gray
        if self.prev is None or self.kp is None:
            self.prev, self.kp = f.copy(), kp
            return MotionEstimate(H, False, previous_frame, gray)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, f, self.kp, None)
        ok = st.reshape(-1).astype(bool) if st is not None else np.zeros(0, bool)
        M = cv2.estimateAffinePartial2D(self.kp[ok], nxt[ok], method=cv2.RANSAC)[0] if ok.sum() > 4 else None
        if M is not None: H = M; H[0, 2] *= self.ds; H[1, 2] *= self.ds
        else: self.n_fail += 1
        self.prev, self.kp = f.copy(), kp
        return MotionEstimate(H, M is not None and bool(np.isfinite(M).all()), previous_frame, gray)


def gmc_job(o_, g_):
    t_ = time.perf_counter_ns(); motion = o_.estimate(g_); return motion, (time.perf_counter_ns() - t_) / 1e6


def gmc_submit(k, g_):
    from ..camera import replay_eval
    live = APP.LIVE
    replay_start = live is not None and live.arr and replay_eval.first_pass_position(live.arr[-1]) == 0
    if replay_start or not APP.gmc_objs or k == 0 or APP.images[k]['extra_info']['video_id'] != APP.images[k - 1]['extra_info']['video_id']:
        o_ = GMC(APP.args.gmc_ds); o_.fp = dict(o_.fp, maxCorners=APP.args.gmc_corners, minDistance=APP.args.gmc_mindist); APP.gmc_objs.append(o_)
    APP.gmc_futs[k] = APP.gmc_pool.submit(APP.gmc_job, APP.gmc_objs[-1], g_)


def gmc_submit_2(k, g_): return None
