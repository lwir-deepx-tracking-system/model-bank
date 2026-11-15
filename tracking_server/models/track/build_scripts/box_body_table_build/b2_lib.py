"""body_v2/scripts/b2_lib.py - shared pieces of the body-mode v2 work (rules: body_v2/PREREG_BODY_V2.json).
Reuses, read only: gru_v2/v3b/scripts/g2_lib.py (corrected targets, metrics), scripts/prcommon, pos_lib, ref_lib, verify/combo/combo_lib.
The server module is imported from the worktree (tracking_server/src/track/box_refine.py); every method runs through its Refiner frame by frame."""
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B2 = HERE.parent                         # pos_refine_20261115/body_v2
ROOT = B2.parent                         # pos_refine_20261115
V3B = ROOT / 'gru_v2' / 'v3b'
sys.path.insert(0, str(V3B / 'scripts'))
import g2_lib as L        # noqa: E402  (reads gru_v2/v3b/data/lut_v3.pkl = the table of the pushed body mode)
from g2_lib import P, PL, R, RE, C   # noqa: E402

sys.path.insert(0, P.PATHS['worktree'])
from src.track import box_refine as BR   # noqa: E402

DK = ('near', 'far')
HIP_BODY = (0.93, 0.96)                  # hips of a body box (detectors' own convention on the train split, g2_lib.HIP_F 0.95)


def train_pairs():
    """every associated (detection, corrected target) pair of the TRAIN split, both pose detectors -> dict of arrays.
    Z det cxcywh, T corrected target cxcywh, G the dataset's whole-person box cxcywh, kp [n,17,3], det 0 near / 1 far, ds 0 TMOT / 1 DT, seq index."""
    out = defaultdict(list); names = []
    for si, r in enumerate([r for r in P.registry() if r['split'] == 'train']):
        S = PL.load_seq(ROOT, r); tg = P.xyxy2cxcywh(S['a']['tgt_h2h']); gb = P.xyxy2cxcywh(S['a']['gbox']); names.append(r['name'])
        for k in DK:
            dg = S['a']['dg_' + k]; m = dg >= 0; z = P.xyxy2cxcywh(S['d'][k]['box'][m]); kp = S['d'][k]['kpt'][m]; t, _ = L.fix_target(tg[dg[m]], z, kp)
            ok = np.isfinite(t[:, 0]); n = int(ok.sum())
            out['Z'].append(z[ok]); out['T'].append(t[ok]); out['G'].append(gb[dg[m]][ok]); out['kp'].append(kp[ok].astype(np.float32)); out['score'].append(S['d'][k]['score'][m][ok])
            out['det'].append(np.full(n, DK.index(k))); out['ds'].append(np.full(n, 0 if r['ds'] == 'TMOT' else 1)); out['seq'].append(np.full(n, si))
    d = {k: np.concatenate(v) for k, v in out.items()}; d['names'] = names
    return d


def cell_index(Z, lh_edges, ar_edges):
    lh = np.log(np.maximum(Z[:, 3], 1.)); ar = Z[:, 2] / np.maximum(Z[:, 3], 1.)
    return np.clip(np.searchsorted(lh_edges, lh, 'right') - 1, 0, len(lh_edges) - 2), np.clip(np.searchsorted(ar_edges, ar, 'right') - 1, 0, len(ar_edges) - 2)


def rel(Z, T):
    """(dx / h, dy / h, log w ratio, log h ratio) of T against Z (both cxcywh) = the table's cell content."""
    return np.stack([(T[:, 0] - Z[:, 0]) / Z[:, 3], (T[:, 1] - Z[:, 1]) / Z[:, 3], np.log(T[:, 2] / Z[:, 2]), np.log(T[:, 3] / Z[:, 3])], 1)


def hip_pos(Z, kp):
    """hip line of the detector's own keypoints as a share of the box height from its top (nan where a hip is under 0.5)."""
    hy = (kp[:, 11, 1] + kp[:, 12, 1]) / 2; ok = (kp[:, 11, 2] >= 0.5) & (kp[:, 12, 2] >= 0.5)
    return np.where(ok, (hy - (Z[:, 1] - Z[:, 3] / 2)) / np.maximum(Z[:, 3], 1.), np.nan)


def conv(tab, lh_cen, ar_cen, Z):
    """a table read with bilinear interpolation (= box_refine conv_delta) -> converted boxes cxcywh."""
    e = L.conv_delta(Z, tab, lh_cen, ar_cen)
    return np.stack([Z[:, 0] + e[:, 0] * Z[:, 3], Z[:, 1] + e[:, 1] * Z[:, 3], Z[:, 2] * np.exp(e[:, 2]), Z[:, 3] * np.exp(e[:, 3])], 1)


def jdump(o):
    import json

    def f(x):
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x))
    return json.dumps(o, indent=1, default=f, ensure_ascii=False)
