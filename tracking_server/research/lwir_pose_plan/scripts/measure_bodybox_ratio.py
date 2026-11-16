"""2026-11-15: how a body box (head top to hip, template fit of the 17 keypoints, pose_partial.to_body_boxes) sits inside each training set's own
person box, measured on well-labelled people (>= 12 visible keypoints), per data set and size bin; and how many people are box-only (< 2 usable
keypoints: they keep their full box in training today). Output: JSON with medians and IQR. Usage: python -I measure_bodybox_ratio.py <code dir>
<template_points.npy> <out.json> name=<labels dir> [name=<labels dir> ...]"""
import sys, json, glob, os
import numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1])
import pose_partial as PP
PP.set_body_box(sys.argv[2])
out_path = sys.argv[3]
res = {}
for arg in sys.argv[4:]:
    name, lab_dir = arg.split('=', 1)
    img_dir = lab_dir.replace('/labels', '/images')
    rows = []; box_only = []; n_people = 0; files = sorted(glob.glob(os.path.join(lab_dir, '*.txt')))
    for f in files:
        L = [l.split() for l in open(f) if l.strip()]
        if not L: continue
        stem = os.path.splitext(os.path.basename(f))[0]
        ims = glob.glob(os.path.join(img_dir, stem + '.*'))
        if not ims: continue
        W, H = Image.open(ims[0]).size
        for l in L:
            v = np.array(l[1:], float)
            if len(v) < 4 + 51: continue
            n_people += 1
            cx, cy, w, h = v[:4] * [W, H, W, H]
            g = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            k = v[4:4 + 51].reshape(17, 3).copy(); k[:, 0] *= W; k[:, 1] *= H
            use = (k[:, 2] > 0) & (k[:, 2] != PP.IGNORE)
            if use.sum() < 2:
                box_only.append(h); continue
            if (k[:, 2] == 2).sum() < 12: continue
            b = PP.to_body_boxes(g[None].astype(np.float32), k[None], (W, H))[0]
            if np.allclose(b, g): continue   # fit failed
            bh, bw = b[3] - b[1], b[2] - b[0]
            rows.append([(b[1] - g[1]) / h, bh / h, ((b[0] + b[2]) / 2 - cx) / w, bw / w, h])
    r = np.array(rows) if rows else np.zeros((0, 5))
    def stat(m):
        if not len(m): return None
        q = np.percentile(m[:, :4], [25, 50, 75], axis=0)
        return dict(n=int(len(m)), top=[round(x, 3) for x in q[:, 0]], height=[round(x, 3) for x in q[:, 1]], cx=[round(x, 3) for x in q[:, 2]], width=[round(x, 3) for x in q[:, 3]])
    bo = np.array(box_only)
    res[name] = dict(files=len(files), people=n_people, all=stat(r),
                     by_height={k: stat(r[(r[:, 4] >= lo) & (r[:, 4] < hi)]) for k, (lo, hi) in {'<50': (0, 50), '50-100': (50, 100), '100-200': (100, 200), '>=200': (200, 1e9)}.items()},
                     box_only=dict(n=int(len(bo)), share=round(len(bo) / max(n_people, 1), 3), h_median=round(float(np.median(bo)), 1) if len(bo) else None,
                                   h_lt50=int((bo < 50).sum()), h_50_100=int(((bo >= 50) & (bo < 100)).sum()), h_ge100=int((bo >= 100).sum())))
    print(name, json.dumps(res[name])[:600], flush=True)
json.dump(res, open(out_path, 'w'), indent=1)
