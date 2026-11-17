"""prcommon.py - shared data code of pos_refine_20261115 (box refiner + keypoint position memory on our pose detector outputs).

Coordinates: the tracking server's frame canvas (x as the 640x480 picture, y = picture y + 16), boxes xyxy or (cx, cy, w, h) as named.
Paths come from paths.json next to this file (PC and GPU server differ).
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PATHS = json.loads((HERE / 'paths.json').read_text()) if (HERE / 'paths.json').exists() else {}
Y0, Y1, XW = 16.0, 496.0, 640.0           # picture rows inside the 640x512 canvas (position_memory KP_ROW0 / KP_ROW1)
KP_NF, KP_HIST, KP_STEPS, KP_C, KP_S, KP_CLIP, KP_MINOBS = 68, 20, 15, 1.0, math.log(1.5), 5.0, 4   # = position_memory.py
LOST_MAX = 30                             # ByteTrack buffer (frames a lost track is kept)
COCO_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
SIZE_BINS = ((0, 50), (50, 100), (100, 1e9))
SIZE_NAMES = ('<50', '50-100', '>=100')

TMOT_FPS = dict(seq1=9.26, seq11=9.29, seq12=9.24, seq13=9.26, seq14=9.08, seq15=9.13, seq16=8.63, seq18=5.05, seq19=5.05, seq20=5.07, seq21=5.05,
                seq25=9.46, seq26=9.49, seq39=9.28, seq41=9.25, seq42=9.19, seq43=9.33, seq48=9.21, seq57=9.34, seq60=9.20, seq62=9.23, seq63=9.28,
                seq64=9.23, seq65=9.29, seq17=5.05, seq2=9.34, seq22=5.08, seq47=9.20, seq54=9.37, seq66=9.29)   # thermal file time stamps (tmot_dataset.zip)

PREREG = json.loads((HERE.parent / 'PREREG_POS_REFINE_20261115.json').read_text(encoding='utf-8')) if (HERE.parent / 'PREREG_POS_REFINE_20261115.json').exists() else None


def registry():
    """-> list of dict(name, ds, split, folder (file stem of det / gmc outputs), gt, fps)."""
    sp = PREREG['splits'] if PREREG else json.loads((HERE / 'splits.json').read_text())
    out = []
    for split in ('train', 'valA', 'valB'):
        for s in sp['TMOT'][split]:
            if split == 'valB': out.append(dict(name=s, ds='TMOT', split=split, folder=s, gt=str(Path(PATHS['tmot_val']) / s / 'tracks_mot.txt'), fps=TMOT_FPS[s]))
            else: out.append(dict(name=s, ds='TMOT', split=split, folder='tmot_' + s, gt=str(Path(PATHS['tmot_train']) / ('tmot_' + s) / 'gt' / 'gt.txt'), fps=TMOT_FPS[s]))
        for s in sp['DanceTrack'][split]:
            sub = 'val' if s in ('dancetrack0087', 'dancetrack0096', 'dancetrack0098', 'dancetrack0099') else 'train'
            out.append(dict(name=s, ds='DT', split=split, folder=s, gt=str(Path(PATHS['dance']) / sub / s / 'gt' / 'gt.txt'), fps=20.0))
    return out


# ------------------------------------------------------------------------------------------------------------------ loading
def load_gt(path):
    """MOT text -> fr (0-based), id, box xyxy canvas."""
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    a = a[(a[:, 4] > 1) & (a[:, 5] > 1)]
    fr = a[:, 0].astype(np.int64) - 1; gid = a[:, 1].astype(np.int64)
    box = np.c_[a[:, 2], a[:, 3] + Y0, a[:, 2] + a[:, 4], a[:, 3] + a[:, 5] + Y0]
    o = np.lexsort((gid, fr)); return fr[o], gid[o], box[o]


def load_det(path):
    z = np.load(path)
    box = z['box'].astype(np.float64).copy(); box[:, 1] += Y0; box[:, 3] += Y0
    kp = z['kpt'].astype(np.float64).copy(); kp[:, :, 1] += Y0
    return dict(fi=z['fi'].astype(np.int64), box=box, score=z['score'].astype(np.float64), kpt=kp, n=int(z['n_frames']))


def load_gmc(path):
    z = np.load(path); return z['warp'].astype(np.float64), z['valid']


# ------------------------------------------------------------------------------------------------------------------ geometry
def xyxy2cxcywh(b): b = np.asarray(b, np.float64); return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2, b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], -1)
def cxcywh2xyxy(b): b = np.asarray(b, np.float64); return np.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)


def iou_m(a, b):
    a = np.asarray(a, np.float64).reshape(-1, 4); b = np.asarray(b, np.float64).reshape(-1, 4)
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih; ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None, :] - inter, 1e-9)


def iou_pair(a, b):
    """IoU of matching rows (xyxy)."""
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih; ar = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / np.maximum(ar(a) + ar(b) - inter, 1e-9)


def apply_rel(g, rel, wmode):
    """gt boxes xyxy [n,4] -> boxes through a relation (dx, dtop, lw, lh) per row or one for all; wmode 'w' (width vs gt width) or 'h' (vs gt height)."""
    g = np.asarray(g, np.float64); rel = np.broadcast_to(np.asarray(rel, np.float64), g.shape[:-1] + (4,))
    hg = g[..., 3] - g[..., 1]; wg = g[..., 2] - g[..., 0]; cx = (g[..., 0] + g[..., 2]) / 2 + rel[..., 0] * hg; top = g[..., 1] + rel[..., 1] * hg
    w = (wg if wmode == 'w' else hg) * np.exp(rel[..., 2]); h = hg * np.exp(rel[..., 3])
    return np.stack([cx - w / 2, top, cx + w / 2, top + h], -1)


def rel_of(d, g):
    """detection xyxy vs gt xyxy -> (dx, dtop, log w / w_gt, log w / h_gt, log h / h_gt) per row."""
    hg = g[:, 3] - g[:, 1]; wg = g[:, 2] - g[:, 0]; wd = d[:, 2] - d[:, 0]; hd = d[:, 3] - d[:, 1]
    return np.stack([((d[:, 0] + d[:, 2]) - (g[:, 0] + g[:, 2])) / 2 / hg, (d[:, 1] - g[:, 1]) / hg, np.log(wd / wg), np.log(wd / hg), np.log(hd / hg)], 1)


def touches_edge(b, m=2.0):
    return (b[:, 0] < m) | (b[:, 2] > XW - m) | (b[:, 1] < Y0 + m) | (b[:, 3] > Y1 - m)


# ------------------------------------------------------------------------------------------------------------------ association
PRIOR = (0.0, 0.0, math.log(0.9), math.log(0.55))   # top aligned, 0.9 gt width, 0.55 gt height (amendment 1, pass 0)


def associate(gfr, gbox, dfi, dbox, rel, wmode, s_min=0.35, ioa_min=0.5, top_tol=0.25):
    """Per frame Hungarian: cost 1 - max(IoU(det, expected head-to-hip box), IoU(det, gt)); gates containment and top alignment.
    -> det_gt [n_det] = row index into the gt arrays or -1, and the score s of the pair."""
    from scipy.optimize import linear_sum_assignment
    out = np.full(len(dfi), -1, np.int64); sc = np.zeros(len(dfi))
    gi_by_f = defaultdict(list); di_by_f = defaultdict(list)
    for i, f in enumerate(gfr): gi_by_f[int(f)].append(i)
    for i, f in enumerate(dfi): di_by_f[int(f)].append(i)
    for f, di in di_by_f.items():
        gi = gi_by_f.get(f)
        if not gi: continue
        di = np.array(di); gi = np.array(gi); D = dbox[di]; G = gbox[gi]
        E = apply_rel(G, rel, wmode)
        s = np.maximum(iou_m(D, E), iou_m(D, G))
        Dc = D.copy(); Dc[:, [0, 2]] = np.clip(Dc[:, [0, 2]], 0, XW); Dc[:, [1, 3]] = np.clip(Dc[:, [1, 3]], Y0, Y1)
        ac = np.maximum((Dc[:, 2] - Dc[:, 0]) * (Dc[:, 3] - Dc[:, 1]), 1e-9)
        iw = np.clip(np.minimum(Dc[:, None, 2], G[None, :, 2]) - np.maximum(Dc[:, None, 0], G[None, :, 0]), 0, None)
        ih = np.clip(np.minimum(Dc[:, None, 3], G[None, :, 3]) - np.maximum(Dc[:, None, 1], G[None, :, 1]), 0, None)
        ioa = iw * ih / ac[:, None]
        hg = G[:, 3] - G[:, 1]; top_ok = np.abs(Dc[:, None, 1] - G[None, :, 1]) <= top_tol * hg[None, :]
        ok = (ioa >= ioa_min) & top_ok & (s >= s_min)
        if not ok.any(): continue
        cost = np.where(ok, 1 - s, 1e6); r, c = linear_sum_assignment(cost)
        for a, b in zip(r, c):
            if ok[a, b]: out[di[a]] = gi[b]; sc[di[a]] = s[a, b]
    return out, sc


# ------------------------------------------------------------------------------------------------------------------ Kalman (tracking server kalman.KF / GKF)
class KF:
    """Ultralytics KalmanFilterXYWH as tracking_server/src/track/kalman.py (one track, float64)."""
    sp, sv = 1 / 20, 1 / 160
    F = np.eye(8); F[np.arange(4), np.arange(4, 8)] = 1.0

    def initiate(self, z):
        z = np.asarray(z, np.float64); sp, sv = self.sp, self.sv
        std = [2 * sp * z[2], 2 * sp * z[3], 2 * sp * z[2], 2 * sp * z[3], 10 * sv * z[2], 10 * sv * z[3], 10 * sv * z[2], 10 * sv * z[3]]
        return np.r_[z, np.zeros(4)], np.diag(np.square(std))

    def predict(self, m, c):
        w, h = m[2], m[3]; sp, sv = self.sp, self.sv
        q = np.array([sp * w, sp * h, sp * w, sp * h, sv * w, sv * h, sv * w, sv * h]); q *= q
        c2 = self.F @ c @ self.F.T; c2[np.arange(8), np.arange(8)] += q
        return self.F @ m, c2

    def update(self, m, c, z):
        R = np.diag(np.square([self.sp * m[2], self.sp * m[3], self.sp * m[2], self.sp * m[3]]))
        pc = c[:4, :4] + R; K = np.linalg.solve(pc, c[:, :4].T).T
        return m + (np.asarray(z, np.float64) - m[:4]) @ K.T, c - K @ pc @ K.T


def gmc_warp(m, c, H):
    """kalman.gmc_warp for one track (kron(I4, R) built block by block: the same matrix)."""
    R = H[:2, :2]; R8 = np.zeros((8, 8))
    R8[0:2, 0:2] = R; R8[2:4, 2:4] = R; R8[4:6, 4:6] = R; R8[6:8, 6:8] = R
    m = R8 @ m; m[:2] += H[:2, 2]; return m, R8 @ c @ R8.T


def pos_inn(z, p):
    h = max(p[3], 1.); return np.array([(z[0] - p[0]) / h, (z[1] - p[1]) / h, math.log(max(z[2], 1.) / max(p[2], 1.)), math.log(max(z[3], 1.) / h)])


ZERO51 = np.zeros(51)


def kp_block(z, kp):
    """17 keypoints (canvas x, y, visibility 0..1) relative to the box -> 51 numbers (position_memory._kp_feat_seen, vis = sigmoid >= 0.5)."""
    if kp is None: return ZERO51
    hz = max(z[3], 1.); vis = (kp[:, 2] >= 0.5).astype(np.float64)
    return np.stack([(kp[:, 0] - z[0]) / hz * vis, (kp[:, 1] - z[1]) / hz * vis, vis], 1).ravel()


def edge_of(b):
    h = max(b[3], 1.); w = max(b[2], 1.)
    return np.array([(b[0] - w / 2) / h, (XW - (b[0] + w / 2)) / h, (b[1] - h / 2 - Y0) / h, (Y1 - (b[1] + h / 2)) / h])


def feat_seen(v, z, zprev, inn, kp):
    """position_memory._kp_feat_seen (posetrack=False) -> 68 numbers."""
    h = max(z[3], 1.); w = max(z[2], 1.); x = np.empty(KP_NF)
    x[0:2] = v / h; x[2:6] = inn; x[6:8] = (z[:2] - zprev[:2]) / h; x[8:10] = np.log(np.maximum(z[2:4], 1.) / np.maximum(zprev[2:4], 1.))
    x[10] = math.log(h / 100.); x[11] = math.log(w / h); x[12:16] = edge_of(z); x[16:67] = kp_block(z, kp); x[67] = 0.
    return np.clip(x, -KP_CLIP, KP_CLIP, out=x)


# ------------------------------------------------------------------------------------------------------------------ position memory series (server feature builder)
def pos_series(f0, f1, det_at, W):
    """One gt person, frames f0..f1, det_at {f: (z cxcywh, kp [17,3])} = the detections the tracker would give this person.
    Tracker-like Kalman (GKF: predict, lost tracks vw = vh = 0, warp after predict; re-initiated after > LOST_MAX lost frames).
    -> dict f -> (x68, post8, z, run, n_in_run) for the seen frames."""
    kf = KF(); mean = cov = None; lost = 0; zprev = None; run = -1; nrun = 0; out = {}
    for f in range(f0, f1 + 1):
        if mean is not None:
            m = mean.copy()
            if lost > 0: m[6:8] = 0
            mean, cov = kf.predict(m, cov)
            if W is not None and f < len(W): mean, cov = gmc_warp(mean, cov, W[f])
        if f in det_at:
            z, kp = det_at[f]; new = False
            if mean is None or lost > LOST_MAX:
                mean, cov = kf.initiate(z); inn = np.zeros(4); new = True
            else:
                inn = pos_inn(z, mean); mean, cov = kf.update(mean, cov, z); new = lost > 0
            if new: run += 1; zprev = None; nrun = 0
            x = feat_seen(mean[4:6], z, zprev if zprev is not None else z, inn, kp)
            zprev = z; nrun += 1; lost = 0
            out[f] = (x, mean.copy(), np.asarray(z, np.float64), run, nrun)
        elif mean is not None:
            lost += 1
    return out


def lost_path(post, z_last, W, t, n_frames):
    """position_memory.pos_predict_kpt plain path from the last posterior after seen frame t -> base [15,4] (cx cy w h) and lost-step inputs [15,68].
    Warps as the server applies them: steps 1 and 2 both use the warp of frame t+2 (the first LOST call does two steps), step s >= 3 the warp of t+s."""
    c = post[:2].copy(); v = post[4:6].copy(); wh = post[2:4].copy(); vwh = post[6:8].copy(); pb = np.asarray(z_last, np.float64).copy()
    base = np.zeros((KP_STEPS, 4)); xl = np.zeros((KP_STEPS, KP_NF))
    for s in range(1, KP_STEPS + 1):
        fw = t + 2 if s <= 2 else t + s
        if W is not None and fw < n_frames:
            R = W[fw][:2, :2]; c = R @ c + W[fw][:2, 2]; v = R @ v
        c = c + v; wh = wh + (vwh if s == 1 else 0.)
        b = np.r_[c, wh]; hb = max(b[3], 1.); wb = max(b[2], 1.)
        chg = np.r_[(b[:2] - pb[:2]) / hb, np.log(np.maximum(b[2:4], 1.) / np.maximum(pb[2:4], 1.))]
        xl[s - 1] = np.clip(np.r_[v / hb, np.zeros(4), chg, math.log(hb / 100.), math.log(wb / hb), edge_of(b), np.zeros(KP_NF - 17), s / KP_STEPS], -KP_CLIP, KP_CLIP)
        base[s - 1] = b; pb = b
    return base, xl


def lost_path_batch(post, z_last, W, t, n_frames):
    """lost_path for N anchors at once (same operations, vectorised): post [N,8], z_last [N,4], t [N] -> base [N,15,4], xl [N,15,68]."""
    N = len(t); c = post[:, :2].copy(); v = post[:, 4:6].copy(); wh = post[:, 2:4].copy(); vwh = post[:, 6:8].copy(); pb = np.asarray(z_last, np.float64).copy()
    base = np.zeros((N, KP_STEPS, 4)); xl = np.zeros((N, KP_STEPS, KP_NF)); I = np.eye(2, 3)
    for s in range(1, KP_STEPS + 1):
        fw = t + 2 if s <= 2 else t + s
        if W is not None:
            Hs = np.where((fw < n_frames)[:, None, None], W[np.minimum(fw, n_frames - 1)], I)
            R = Hs[:, :, :2]; c = np.einsum('nij,nj->ni', R, c) + Hs[:, :, 2]; v = np.einsum('nij,nj->ni', R, v)
        c = c + v; wh = wh + (vwh if s == 1 else 0.)
        b = np.concatenate([c, wh], 1); hb = np.maximum(b[:, 3], 1.); wb = np.maximum(b[:, 2], 1.)
        x = xl[:, s - 1]
        x[:, 0:2] = v / hb[:, None]; x[:, 6:8] = (b[:, :2] - pb[:, :2]) / hb[:, None]; x[:, 8:10] = np.log(np.maximum(b[:, 2:4], 1.) / np.maximum(pb[:, 2:4], 1.))
        x[:, 10] = np.log(hb / 100.); x[:, 11] = np.log(wb / hb)
        x[:, 12] = (b[:, 0] - wb / 2) / hb; x[:, 13] = (XW - (b[:, 0] + wb / 2)) / hb; x[:, 14] = (b[:, 1] - hb / 2 - Y0) / hb; x[:, 15] = (Y1 - (b[:, 1] + hb / 2)) / hb
        x[:, 67] = s / KP_STEPS; np.clip(x, -KP_CLIP, KP_CLIP, out=x)
        base[:, s - 1] = b; pb = b
    return base, xl


# ------------------------------------------------------------------------------------------------------------------ refiner series (own Kalman, camera-compensated)
REF_NF = 70
REF_C, REF_S = 0.5, math.log(2.5)


def warp_box(b, H):
    """cxcywh box through a partial affine warp: centre warped, size x scale."""
    if H is None: return np.array(b, np.float64)
    s = math.sqrt(abs(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])); o = np.empty(4)
    o[0] = H[0, 0] * b[0] + H[0, 1] * b[1] + H[0, 2]; o[1] = H[1, 0] * b[0] + H[1, 1] * b[1] + H[1, 2]; o[2] = b[2] * s; o[3] = b[3] * s
    return o


def ref_series(f0, f1, det_at, W):
    """Refiner inputs for one person over frames f0..f1 (det_at as pos_series). Own tracker-like Kalman with GMC warp, NOT reset after a short gap
    (<= LOST_MAX frames: the drawn track keeps its id). -> dict f -> (x70, z) for the seen frames.
    x: [0:2] Kalman velocity / h, [2:6] innovation, [6:10] box change against the previous seen box moved by the camera warps, [10] log h / 100, [11] log w / h,
       [12:16] picture edges / h, [16:67] keypoints, [67] frames missed since the last seen frame / 15, [68:70] camera shift of the previous box centre this frame / h."""
    kf = KF(); mean = cov = None; lost = 0; zp = None; out = {}
    for f in range(f0, f1 + 1):
        H = W[f] if (W is not None and f < len(W)) else None
        cam = np.zeros(2)
        if mean is not None:
            m = mean.copy()
            if lost > 0: m[6:8] = 0
            mean, cov = kf.predict(m, cov)
            if H is not None: mean, cov = gmc_warp(mean, cov, H)
        if zp is not None and H is not None:
            zw = warp_box(zp, H); cam = zw[:2] - zp[:2]; zp = zw
        if f in det_at:
            z, kp = det_at[f]; z = np.asarray(z, np.float64)
            if mean is None or lost > LOST_MAX:
                mean, cov = kf.initiate(z); inn = np.zeros(4); zref = z; gap = 0; cam = np.zeros(2)
            else:
                inn = pos_inn(z, mean); mean, cov = kf.update(mean, cov, z); zref = zp if zp is not None else z; gap = lost
            h = max(z[3], 1.); w = max(z[2], 1.); x = np.empty(REF_NF)
            x[0:2] = mean[4:6] / h; x[2:6] = inn; x[6:8] = (z[:2] - zref[:2]) / h; x[8:10] = np.log(np.maximum(z[2:4], 1.) / np.maximum(zref[2:4], 1.))
            x[10] = math.log(h / 100.); x[11] = math.log(w / h); x[12:16] = edge_of(z); x[16:67] = kp_block(z, kp); x[67] = min(gap, LOST_MAX) / 15.; x[68:70] = cam / h
            out[f] = (np.clip(x, -KP_CLIP, KP_CLIP, out=x), z.copy())
            zp = z.copy(); lost = 0
        elif mean is not None:
            lost += 1
            if lost > LOST_MAX: mean = cov = None; zp = None; lost = 0
    return out


def ref_apply(z, o):
    """refiner output o (4) on the detector box z (cxcywh) -> corrected box."""
    h = max(z[3], 1.)
    return np.r_[z[:2] + REF_C * h * np.tanh(o[:2]), z[2:4] * np.exp(REF_S * np.tanh(o[2:4]))]


# ------------------------------------------------------------------------------------------------------------------ warps for augmentation
def h3(H): return np.vstack([H, [0, 0, 1]])


def flip_warps(W):
    Fm = np.array([[-1., 0, XW], [0, 1, 0], [0, 0, 1]]); return np.array([(Fm @ h3(H) @ Fm)[:2] for H in W])


def reverse_warps(W):
    n = len(W); out = np.zeros_like(W); out[0] = np.eye(2, 3)
    for r in range(1, n): out[r] = np.linalg.inv(h3(W[n - r]))[:2]
    return out


def half_warps(W):
    idx = np.arange(0, len(W), 2); out = np.zeros((len(idx), 2, 3)); out[0] = np.eye(2, 3)
    for k in range(1, len(idx)): out[k] = (h3(W[idx[k]]) @ h3(W[idx[k] - 1]))[:2]
    return out


def size_bin(h):
    for k, (a, b) in enumerate(SIZE_BINS):
        if a <= h < b: return k
    return len(SIZE_BINS) - 1
