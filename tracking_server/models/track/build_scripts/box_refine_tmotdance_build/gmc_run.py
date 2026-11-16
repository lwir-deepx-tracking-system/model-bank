"""gmc_run.py (PC): camera motion per frame exactly as the tracking server computes it with the device options
(--gmc --gmc-corners 300 --gmc-mindist 5): src/track/camera_motion.GMC (Ultralytics sparseOptFlow, downscale 2) on the 640x512 canvas
made like detect_frame.to_canvas (picture in rows 16..496, edge rows replicated). A new GMC object per sequence (gmc_submit), identity on frame 0.
Writes data/gmc/<seq>.npz: warp [n,2,3] (previous -> current frame, canvas px), valid [n] (bool).
Usage: python gmc_run.py <out_dir> <seq_dir> [...]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


class GMC:
    """= tracking_server/src/track/camera_motion.GMC (copied; same code path, estimate() only)."""
    def __init__(self, downscale=2):
        self.ds = downscale; self.prev = None; self.kp = None; self.n_fail = 0
        self.fp = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)

    def estimate(self, gray):
        h, w = gray.shape; f = cv2.resize(gray, (w // self.ds, h // self.ds)) if self.ds > 1 else gray
        H = np.eye(2, 3); kp = cv2.goodFeaturesToTrack(f, mask=None, **self.fp)
        if self.prev is None or self.kp is None:
            self.prev, self.kp = f.copy(), kp
            return H, False
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, f, self.kp, None)
        ok = st.reshape(-1).astype(bool) if st is not None else np.zeros(0, bool)
        M = cv2.estimateAffinePartial2D(self.kp[ok], nxt[ok], method=cv2.RANSAC)[0] if ok.sum() > 4 else None
        if M is not None: H = M; H[0, 2] *= self.ds; H[1, 2] *= self.ds
        else: self.n_fail += 1
        self.prev, self.kp = f.copy(), kp
        return H, M is not None and bool(np.isfinite(M).all())


def to_canvas(g):
    c = np.empty((512, 640), np.uint8); c[16:496] = g; c[:16] = g[0]; c[496:] = g[-1]; return c


def one(args):
    out, sd = args; sd = Path(sd)
    d = sd / 'img1' if (sd / 'img1').is_dir() else sd / 'images'
    fs = sorted(p for p in d.iterdir() if p.suffix.lower() in ('.jpg', '.png'))
    o = GMC(2); o.fp = dict(o.fp, maxCorners=300, minDistance=5)
    W = np.zeros((len(fs), 2, 3)); V = np.zeros(len(fs), bool)
    for i, p in enumerate(fs):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE); assert g is not None and g.shape == (480, 640), p
        W[i], V[i] = o.estimate(to_canvas(g))
    np.savez_compressed(Path(out) / ('%s.npz' % sd.name), warp=W.astype(np.float64), valid=V, names=np.array([p.name for p in fs]))
    t = np.abs(W[:, :, 2]); return sd.name, len(fs), int(V.sum()), float(np.median(t[1:, 0])) if len(fs) > 1 else 0.0, float(np.percentile(t[1:], 95)) if len(fs) > 1 else 0.0


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(out), s) for s in sys.argv[2:] if not (out / ('%s.npz' % Path(s).name)).exists()]
    cv2.setNumThreads(1)
    with Pool(7) as pool:
        for r in pool.imap_unordered(one, jobs):
            print('GMC %s frames %d valid %d median |tx| %.2f p95 |t| %.2f' % r, flush=True)
    print('GMC_DONE', flush=True)
