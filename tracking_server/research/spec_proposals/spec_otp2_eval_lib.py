"""Sample extraction for OpenThermalPose2 (shared by train_otp2_spec.py); same rules as spec_otp2_eval.py split()."""
import glob
import os

import cv2
import numpy as np
from scipy import ndimage

from src.detect import spec_proposals as sp

rng = np.random.default_rng(0)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def sq(x0, y0, x1, y1):
    s = min(x1 - x0, y1 - y0); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return (cx - s // 2, cy - s // 2, cx - s // 2 + s, cy - s // 2 + s)


def split(ROOT, name, step):
    X, y, kind = [], [], []
    files = sorted(glob.glob(os.path.join(ROOT, name, 'images', '*.png')))[::step]
    for fp in files:
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = im[..., 0]
        H, W = im.shape
        lp = os.path.join(ROOT, name, 'labels', os.path.splitext(os.path.basename(fp))[0] + '.txt')
        gt = []
        for l in open(lp) if os.path.exists(lp) else []:
            v = l.split()
            if len(v) >= 5 and v[0] == '0':
                cx, cy, w, h = float(v[1]) * W, float(v[2]) * H, float(v[3]) * W, float(v[4]) * H
                gt.append((int(max(0, cx - w / 2)), int(max(0, cy - h / 2)), int(min(W, cx + w / 2)), int(min(H, cy + h / 2))))
        if not gt:
            continue
        pm = np.zeros((H, W), bool)
        for b in gt:
            pm[b[1]:b[3], b[0]:b[2]] = True
        pos = [sq(*b) for b in gt if min(b[2] - b[0], b[3] - b[1]) >= 32]
        warm, rnd = [], []
        if pm.any() and (~pm).any():
            lab, nl = ndimage.label((im >= np.percentile(im[pm], 25)) & ~ndimage.binary_dilation(pm, iterations=6))
            for sl, i in zip(ndimage.find_objects(lab), range(1, nl + 1)):
                if (lab[sl] == i).sum() >= 600:
                    s = sq(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
                    if s[2] - s[0] >= 32:
                        warm.append(s)
        for _ in range(4):
            z = int(rng.choice((64, 96, 128, 160)))
            if z >= min(H, W):
                continue
            x0, y0 = int(rng.integers(0, W - z)), int(rng.integers(0, H - z))
            if all(iou((x0, y0, x0 + z, y0 + z), b) < 0.05 for b in gt):
                rnd.append((x0, y0, x0 + z, y0 + z))
        boxes = pos + warm + rnd
        if boxes:
            X.append(sp.features(im, boxes)); y += [1] * len(pos) + [0] * (len(warm) + len(rnd)); kind += ['person'] * len(pos) + ['warm'] * len(warm) + ['random'] * len(rnd)
    return np.concatenate(X), np.array(y), np.array(kind), len(files)


