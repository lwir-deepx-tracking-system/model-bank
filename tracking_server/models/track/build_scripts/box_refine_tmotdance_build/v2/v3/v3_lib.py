"""v2/v3/scripts/v3_lib.py - box_refine v3 helpers (rules: v2/v3/PREREG_V3.json). Reuses v2/scripts/v2_lib.py (the server Refiner run frame by frame).
The metric aim point is top + V3_AIM x h (environment variable, default 0.37 = the v2 metric); set before the import, workers inherit it."""
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
sys.path.insert(0, str(V3.parent / 'scripts'))
import v2_lib as L   # noqa: E402

AIM = float(os.environ.get('V3_AIM', '0.37'))
L.R.AIM_Y = AIM; L.AIM = AIM
BRM = L.BRM
V2_SIZE = dict(mc_s=5.0, beta_s=3.0, dcut=1.0, anchor='centre', src='trk')   # the v2 choice (v2/results/size_valA.json)


def run3(inp, size, blend_s=0.0, aim_y=AIM):
    """L.run('size') with the anchor's aim_y and a blend time (the server Refiner) -> ({tid: out}, {tid: k})."""
    obj = BRM.Refiner('size', None, BRM.EURO, blend_s, size, aim_y=aim_y); ids = inp['ids']; W = inp['W']; fps = inp['fps']
    out = {tid: np.zeros((len(e['f']), 4)) for tid, e in ids.items()}; kk = {tid: np.zeros(len(e['f']), np.int64) for tid, e in ids.items()}
    for f in range(inp['n']):
        H = W[f] if (f > 0 and f < len(W)) else None; its = inp['by_f'].get(f, ())
        res = obj.step([(tid, z, kp, zr) for tid, j, z, kp, zr in its], f / fps, H, inp['keys'][f], True)
        for tid, j, *_ in its: out[tid][j] = res[tid]; kk[tid][j] = obj.tr[tid]['k']
    return out, kk


def edges(o, f, href, okrow):
    """one-frame top / bottom edge steps / reference height (= v2/review_numbers/scripts/review_edges.py)"""
    n = len(f); c1 = np.r_[False, np.diff(f) == 1] & okrow & np.r_[False, okrow[:-1]]; i = np.flatnonzero(c1)
    top = o[:, 1] - o[:, 3] / 2; bot = o[:, 1] + o[:, 3] / 2
    dt = np.full(n, np.nan); db = np.full(n, np.nan)
    dt[i] = np.abs(top[i] - top[i - 1]) / href[i]; db[i] = np.abs(bot[i] - bot[i - 1]) / href[i]
    return dt, db
