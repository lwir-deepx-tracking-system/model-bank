"""fix_lib.py (= gru_mismatch_20261116 pf_lib.py) - the position-memory data builder (prcommon / pos_lib of this folder) with the training-data fixes of 2026-11-16.

Fix 1 (picture rows per dataset): the picture-edge inputs measure the distance to the edges of the REAL picture. The old builder used the device rows
  16..496 of the 640x512 canvas for every dataset, but the DanceTrack copy is a letterbox (content rows 60..420 of the 480-row picture = canvas
  76..436), so DanceTrack people never came closer than 60 px to the 'bottom edge' the model saw.
Fix 2 (no target where the gt box is cut by the picture): the detector-definition target is the gt box through a per-height lookup; for a gt box cut
  by the picture edge the lookup is applied to the cut height and asks for a smaller, higher box. Those target frames are dropped (NaN = no loss,
  no score), the inputs are unchanged. Small: 2.5 % (DanceTrack) / 3.1 % (TMOT) of the train gt rows; cut-bottom targets are 0.94x / 0.85x the
  detector's own associated box (the 0.47-0.77x first quoted was against the plain Kalman path on 442 DanceTrack anchors = 0.35 % of v0).
Fix 3 (server warp order, optional): lost step 1 uses the camera warp of frame t+1 (the old builder and server: the warp of t+2 twice, t+1 never).
  fix_export writes opt_warp_fix = 1 into files trained this way; the server keeps the old order for files without it.
Fixes 1 + 2 alone (v1, edge inputs kept) closed about 41 % of the gap to Kalman on our camera (TEST 0.243, old file 0.336, Kalman 0.110). Most of the
  gain came from EDGES[0] = False (v2): the edge inputs carried the public cameras' scene layout (low in the picture = big person, cut by the bottom).
Still open (11-16 review): the target is ONE per-height lookup in head-to-hip mode, but for a person >= 64 px with no visible keypoint the detector
  mostly draws a whole-body box (train split, 64-130 px, far detector and TMOT near: det / gt 0.79-1.04 without keypoints, about 0.53 with), so
  those targets ask for 0.55-0.69x of the detector's box. Not fixed here; the
  server gives such losses the plain Kalman path (position_memory.KP_NOKPT_H) for files without opt_box_mode_targets = 1.
Everything else is pos_lib / prcommon as they are (imported, not copied).
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prcommon as P   # noqa: E402
import pos_lib as L    # noqa: E402

PR = Path(P.PATHS['root'])   # paths.json 'root': the data root (data/assoc, data/det, data/gmc)

ROWS = {'TMOT': (16.0, 496.0), 'DT': (76.0, 436.0)}   # canvas rows of the real picture
CUT_PX = 2.0
DETS = L.DETS
EDGES = [True]   # v2 (PREREG_FIX amendment 1): False -> the four picture-edge inputs are the constant 5 (= far from every edge) in seen and lost rows
NO_EDGE = 5.0


def edge_rows(b, y0, y1):
    h = max(b[3], 1.); w = max(b[2], 1.)
    return np.array([(b[0] - w / 2) / h, (P.XW - (b[0] + w / 2)) / h, (b[1] - h / 2 - y0) / h, (y1 - (b[1] + h / 2)) / h])


def feat_seen(v, z, zprev, inn, kp, y0, y1):
    """= prcommon.feat_seen with the picture rows of the dataset"""
    h = max(z[3], 1.); w = max(z[2], 1.); x = np.empty(P.KP_NF)
    x[0:2] = v / h; x[2:6] = inn; x[6:8] = (z[:2] - zprev[:2]) / h; x[8:10] = np.log(np.maximum(z[2:4], 1.) / np.maximum(zprev[2:4], 1.))
    x[10] = math.log(h / 100.); x[11] = math.log(w / h); x[12:16] = edge_rows(z, y0, y1) if EDGES[0] else NO_EDGE; x[16:67] = P.kp_block(z, kp); x[67] = 0.
    return np.clip(x, -P.KP_CLIP, P.KP_CLIP, out=x)


def pos_series(f0, f1, det_at, W, y0, y1):
    """= prcommon.pos_series with feat_seen of this file"""
    kf = P.KF(); mean = cov = None; lost = 0; zprev = None; run = -1; nrun = 0; out = {}
    for f in range(f0, f1 + 1):
        if mean is not None:
            m = mean.copy()
            if lost > 0: m[6:8] = 0
            mean, cov = kf.predict(m, cov)
            if W is not None and f < len(W): mean, cov = P.gmc_warp(mean, cov, W[f])
        if f in det_at:
            z, kp = det_at[f]; new = False
            if mean is None or lost > P.LOST_MAX:
                mean, cov = kf.initiate(z); inn = np.zeros(4); new = True
            else:
                inn = P.pos_inn(z, mean); mean, cov = kf.update(mean, cov, z); new = lost > 0
            if new: run += 1; zprev = None; nrun = 0
            x = feat_seen(mean[4:6], z, zprev if zprev is not None else z, inn, kp, y0, y1)
            zprev = z; nrun += 1; lost = 0
            out[f] = (x, mean.copy(), np.asarray(z, np.float64), run, nrun)
        elif mean is not None:
            lost += 1
    return out


def lost_path_batch(post, z_last, W, t, n_frames, y0, y1, fixwarp=True):
    """= prcommon.lost_path_batch with the picture rows; fixwarp: step s uses the warp of frame t+s (old: steps 1 and 2 both t+2)"""
    N = len(t); c = post[:, :2].copy(); v = post[:, 4:6].copy(); wh = post[:, 2:4].copy(); vwh = post[:, 6:8].copy(); pb = np.asarray(z_last, np.float64).copy()
    base = np.zeros((N, P.KP_STEPS, 4)); xl = np.zeros((N, P.KP_STEPS, P.KP_NF)); I = np.eye(2, 3)
    for s in range(1, P.KP_STEPS + 1):
        fw = t + s if fixwarp else (t + 2 if s <= 2 else t + s)
        if W is not None:
            Hs = np.where((fw < n_frames)[:, None, None], W[np.minimum(fw, n_frames - 1)], I)
            R = Hs[:, :, :2]; c = np.einsum('nij,nj->ni', R, c) + Hs[:, :, 2]; v = np.einsum('nij,nj->ni', R, v)
        c = c + v; wh = wh + (vwh if s == 1 else 0.)
        b = np.concatenate([c, wh], 1); hb = np.maximum(b[:, 3], 1.); wb = np.maximum(b[:, 2], 1.)
        x = xl[:, s - 1]
        x[:, 0:2] = v / hb[:, None]; x[:, 6:8] = (b[:, :2] - pb[:, :2]) / hb[:, None]; x[:, 8:10] = np.log(np.maximum(b[:, 2:4], 1.) / np.maximum(pb[:, 2:4], 1.))
        x[:, 10] = np.log(hb / 100.); x[:, 11] = np.log(wb / hb)
        x[:, 12] = (b[:, 0] - wb / 2) / hb; x[:, 13] = (P.XW - (b[:, 0] + wb / 2)) / hb; x[:, 14] = (b[:, 1] - hb / 2 - y0) / hb; x[:, 15] = (y1 - (b[:, 1] + hb / 2)) / hb
        if not EDGES[0]: x[:, 12:16] = NO_EDGE
        x[:, 67] = s / P.KP_STEPS; np.clip(x, -P.KP_CLIP, P.KP_CLIP, out=x)
        base[:, s - 1] = b; pb = b
    return base, xl


def tracks_of(S, k, mask_cut=True):
    """= pos_lib.tracks_of; mask_cut: the target frames whose gt box touches the real picture edge are left out (also kept as e['cut'])"""
    T = L.tracks_of(S, k, 'det'); a = S['a']; y0, y1 = ROWS[S['r']['ds']]; gb = a['gbox']
    cut = (gb[:, 0] <= CUT_PX) | (gb[:, 2] >= P.XW - CUT_PX) | (gb[:, 1] <= y0 + CUT_PX) | (gb[:, 3] >= y1 - CUT_PX)
    for row, (f, g) in enumerate(zip(a['gfr'], a['gid'])):
        e = T.get(int(g))
        if e is None: continue
        e.setdefault('cut', set())
        if cut[row]:
            e['cut'].add(int(f))
            if mask_cut: e['tgt'].pop(int(f), None); e['tgt2'].pop(int(f), None)
    return T


def anchors(S, k, T, W, n, rng=None, stride=1, offset=0, syn_only=False, vanish_p=0.6, fixwarp=True, rows=None):
    """= pos_lib.anchors with the builder of this file (rows = the dataset's picture rows unless given)"""
    y0, y1 = rows if rows is not None else ROWS[S['r']['ds']]
    Lq = defaultdict(list)
    for g, e in T.items():
        det = e['det']; e_syn = -1
        if rng is not None and vanish_p > 0: det, e_syn = L.vanish(det, rng, vanish_p)
        if not det: continue
        f0 = min(det); ser = pos_series(f0, max(e['f1'], max(det)), det, W, y0, y1)
        seen = sorted(ser); runs = defaultdict(list)
        for f in seen: runs[ser[f][3]].append(f)
        orig_seen = set(e['det']); sel = []
        for rid, fl in runs.items():
            for j, t in enumerate(fl):
                if j + 1 < P.KP_MINOBS: continue
                if syn_only and t != e_syn - 1: continue
                if not syn_only and stride > 1 and (t - offset) % stride: continue
                tg = np.full((P.KP_STEPS, 4), np.nan); t2 = np.full((P.KP_STEPS, 4), np.nan)
                for s in range(1, P.KP_STEPS + 1):
                    if t + s in e['tgt']: tg[s - 1] = e['tgt'][t + s]; t2[s - 1] = e['tgt2'][t + s]
                if not np.isfinite(tg[:, 0]).any(): continue
                hist = [ser[f][0] for f in fl[max(0, j + 1 - P.KP_HIST):j + 1]]
                hx = np.zeros((P.KP_HIST, P.KP_NF), np.float32); hm = np.zeros(P.KP_HIST, np.float32)
                hx[P.KP_HIST - len(hist):] = hist; hm[P.KP_HIST - len(hist):] = 1; sel.append(t)
                Lq['hx'].append(hx.astype(np.float16)); Lq['hm'].append(hm)
                Lq['h0'].append(max(ser[t][2][3], 1.)); Lq['tgt'].append(tg.astype(np.float32)); Lq['tgt2'].append(t2.astype(np.float32))
                Lq['real'].append((t + 1) not in orig_seen); Lq['syn'].append(t == e_syn - 1); Lq['hg'].append(e['hg'].get(t, np.nan)); Lq['gid'].append(g); Lq['t'].append(t)
        if sel:
            ta = np.array(sel); base, xl = lost_path_batch(np.array([ser[t][1] for t in sel]), np.array([ser[t][2] for t in sel]), W, ta, n, y0, y1, fixwarp)
            Lq['base'].extend(base.astype(np.float32)); Lq['xl'].extend(xl.astype(np.float16))
    return Lq
