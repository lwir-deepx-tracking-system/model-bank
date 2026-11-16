"""ref_lib.py - box refiner: One Euro filter, learned GRU refiner (data, model, loss), server-like rows and the switch blend (pos_refine_20261115)."""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prcommon as P
import pos_lib as PL

DETS = ('near', 'far')
HID = 48
JUMP_H = 1.5           # a new detection farther than this many box heights from the track's last one = another person: state reset
AIM_Y = 0.37           # gimbal aim point: top + 0.37 x height (LWIR_GIMBAL_AIM_Y for the body-box models)


# ------------------------------------------------------------------------------------------------------------------ One Euro (Casiez et al., CHI 2012)
class Euro:
    """Per track. Centre (cx, cy) with (mc, beta) and speed in box heights per second; log w / log h with (mc_s, beta_s). d_cutoff 1 Hz.
    comp=True: before each step the previous state is moved by the camera warp (centre and its speed by the warp, size by the warp scale)."""
    def __init__(self, mc, beta, mc_s, beta_s, dcut=1.0, comp=True):
        self.p = (mc, beta, mc_s, beta_s, dcut); self.comp = comp; self.x = None

    @staticmethod
    def _a(cut, dt): tau = 1.0 / (2 * math.pi * cut); return 1.0 / (1.0 + tau / dt)

    def step(self, z, dt, H=None):
        """z cxcywh (detector), dt s since the track's previous step, H = composed warp since then (2x3) or None -> filtered box."""
        mc, beta, mcs, bs, dcut = self.p; s = np.r_[z[:2], math.log(max(z[2], 1.)), math.log(max(z[3], 1.))]
        if self.x is None: self.x = s.copy(); self.dx = np.zeros(4); return self.box()
        if self.comp and H is not None:
            self.x[:2] = H[:2, :2] @ self.x[:2] + H[:2, 2]; self.dx[:2] = H[:2, :2] @ self.dx[:2]
            sc = math.log(max(math.sqrt(abs(np.linalg.det(H[:2, :2]))), 1e-6)); self.x[2:] += sc
        dt = max(dt, 1e-3); h = math.exp(self.x[3])
        d = (s - self.x) / dt; ad = self._a(dcut, dt); self.dx = ad * d + (1 - ad) * self.dx
        cut_c = mc + beta * np.abs(self.dx[:2]) / h; cut_s = mcs + bs * np.abs(self.dx[2:])
        a = np.r_[[self._a(c, dt) for c in cut_c], [self._a(c, dt) for c in cut_s]]
        self.x = a * s + (1 - a) * self.x; return self.box()

    def box(self): return np.r_[self.x[:2], math.exp(self.x[2]), math.exp(self.x[3])]


# ------------------------------------------------------------------------------------------------------------------ training pieces
def splice(Tn, Tf, g, rng, fps):
    """det_at for one person: near only, far only, or spliced at random switch frames (gaps 1-8 s). -> (det_at, src per frame dict, switch frames)"""
    srcs = [k for k, T in (('near', Tn), ('far', Tf)) if g in T]
    if not srcs: return {}, {}, []
    e = {k: (Tn if k == 'near' else Tf)[g] for k in srcs}
    frames = sorted(set().union(*[set(e[k]['det']) for k in srcs]))
    if len(srcs) == 1 or rng.uniform() < 0.5:
        k = srcs[int(rng.integers(len(srcs)))]; return dict(e[k]['det']), {f: k for f in e[k]['det']}, []
    f0, f1 = frames[0], frames[-1]; cur = int(rng.integers(2)); t = f0; det = {}; src = {}; sw = []
    while t <= f1:
        seg = max(1, int(round(rng.uniform(1.0, 8.0) * fps))); k = DETS[cur]
        for f in range(t, min(t + seg, f1 + 1)):
            if f in e[k]['det']: det[f] = e[k]['det'][f]; src[f] = k
        t += seg; cur = 1 - cur
        if t <= f1: sw.append(t)
    return det, src, sw


def occlude(det, rng, p=0.3):
    """synthetic occluder entering / staying / leaving (desk from below or a side): keypoints under it hidden; half the time the box shrinks to the visible part."""
    if rng.uniform() > p or len(det) < 10: return det
    fs = sorted(det); fa = fs[int(rng.integers(len(fs)))]; tin = int(rng.integers(3, 11)); stay = int(rng.integers(0, 41)); tout = int(rng.integers(3, 11))
    side = 'bottom' if rng.uniform() < 0.5 else ('left' if rng.uniform() < 0.5 else 'right'); fmax = rng.uniform(0.3, 0.7); shrink = rng.uniform() < 0.5
    det = dict(det)
    for f in range(fa, fa + tin + stay + tout):
        if f not in det: continue
        k = f - fa; fr = fmax * (min(1.0, (k + 1) / tin) if k < tin + stay else max(0.0, 1 - (k - tin - stay + 1) / tout))
        if fr <= 0: continue
        z, kp = det[f]; z = z.copy(); kp = kp.copy(); x1, y1, x2, y2 = z[0] - z[2] / 2, z[1] - z[3] / 2, z[0] + z[2] / 2, z[1] + z[3] / 2
        if side == 'bottom':
            line = y2 - fr * (y2 - y1); kp[kp[:, 1] > line, 2] = 0.
            if shrink: y2 = line
        elif side == 'left':
            line = x1 + fr * (x2 - x1); kp[kp[:, 0] < line, 2] = 0.
            if shrink: x1 = line
        else:
            line = x2 - fr * (x2 - x1); kp[kp[:, 0] > line, 2] = 0.
            if shrink: x2 = line
        z = np.r_[(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 2.), max(y2 - y1, 2.)]
        if shrink: z[:2] += rng.normal(0, 0.02, 2) * z[3]
        det[f] = (z, kp)
    return det


def pieces_of_seq(S, rng, aug=True, max_len=256):
    """training pieces of one sequence: per gt person a (possibly spliced, occluded) detection track -> refiner inputs on the seen frames."""
    Tn = PL.tracks_of(S, 'near', 'h2h'); Tf = PL.tracks_of(S, 'far', 'h2h'); W = S['W']; n = S['n']
    if aug and rng.uniform() < 0.5:   # left-right flip of the whole sequence
        Tn, W2, _ = PL.augment(Tn, W, n, rng, flip=True); Tf, _, _ = PL.augment(Tf, W, n, rng, flip=True); W = W2
    out = []
    for g in sorted(set(Tn) | set(Tf)):
        det, src, sw = splice(Tn, Tf, g, rng, S['r']['fps'])
        if len(det) < 3: continue
        if aug: det = occlude(det, rng)
        e = (Tn if g in Tn else Tf)[g]; ser = P.ref_series(min(det), max(det), det, W)
        fs = sorted(ser)
        x = np.array([ser[f][0] for f in fs], np.float32); z = np.array([ser[f][1] for f in fs], np.float32)
        tg = np.array([e['tgt'].get(f, [np.nan] * 4) for f in fs], np.float32)
        con = np.r_[0, (np.diff(fs) == 1).astype(np.int8)]
        for a in range(0, len(fs), max_len):
            sl = slice(a, a + max_len); con_ = con[sl].copy(); con_[0] = 0
            out.append(dict(x=x[sl], z=z[sl], tgt=tg[sl], con=con_))
    return out


def batch_pieces(pcs, idx):
    T = max(len(pcs[i]['z']) for i in idx); B = len(idx)
    X = np.zeros((B, T, P.REF_NF), np.float32); Z = np.ones((B, T, 4), np.float32); G = np.ones((B, T, 4), np.float32); OK = np.zeros((B, T), bool); C = np.zeros((B, T), bool); M = np.zeros((B, T), bool)
    for b, i in enumerate(idx):
        p = pcs[i]; n = len(p['z']); X[b, :n] = p['x']; Z[b, :n] = p['z']; g = p['tgt']; ok = np.isfinite(g[:, 0])
        G[b, :n][ok] = g[ok]; OK[b, :n] = ok; C[b, :n] = p['con'] > 0; M[b, :n] = True
    return X, Z, G, OK, C, M


# ------------------------------------------------------------------------------------------------------------------ model
def make_net(hidden=HID):
    import torch

    class RefNet(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.cell = torch.nn.GRUCell(P.REF_NF, hidden); self.head = torch.nn.Linear(hidden, 4)
            torch.nn.init.zeros_(self.head.weight); torch.nn.init.zeros_(self.head.bias); self.nh = hidden

        def forward(self, X, Z, M, h=None):
            """X [B,T,70], Z [B,T,4] detector boxes, M [B,T] valid -> corrected boxes [B,T,4], last state."""
            B, T = X.shape[:2]
            if h is None: h = X.new_zeros(B, self.nh)
            out = []
            for t in range(T):
                hn = self.cell(X[:, t], h); h = torch.where(M[:, t:t + 1], hn, h); o = self.head(h); z = Z[:, t]; hz = z[:, 3:4].clamp(min=1.)
                out.append(torch.cat([z[:, :2] + P.REF_C * hz * torch.tanh(o[:, :2]), z[:, 2:] * torch.exp(P.REF_S * torch.tanh(o[:, 2:]))], 1))
            return torch.stack(out, 1), h
    return RefNet()


def err_vec(q, g):
    """error vector ((cx - cx_t) / h_t, (cy - cy_t) / h_t, log w / w_t, log h / h_t) (torch or numpy)."""
    try:
        import torch
        if isinstance(q, torch.Tensor):
            return torch.cat([(q[..., :2] - g[..., :2]) / g[..., 3:4], torch.log(q[..., 2:].clamp(min=1e-3) / g[..., 2:].clamp(min=1e-3))], -1)
    except ImportError: pass
    return np.concatenate([(q[..., :2] - g[..., :2]) / g[..., 3:4], np.log(np.maximum(q[..., 2:], 1e-3) / np.maximum(g[..., 2:], 1e-3))], -1)


def sl1(x, beta=0.02):
    ax = np.abs(x); return np.where(ax < beta, 0.5 * ax * ax / beta, ax - 0.5 * beta)


def loss_torch(q, G, OK, C, lam=1.0):
    import torch, torch.nn.functional as TF
    e = err_vec(q, G); okf = OK.float()
    acc = (TF.smooth_l1_loss(e, torch.zeros_like(e), beta=0.02, reduction='none').sum(-1) * okf).sum() / okf.sum().clamp(min=1)
    d2 = e[:, 2:] - 2 * e[:, 1:-1] + e[:, :-2]; v = OK[:, 2:] & OK[:, 1:-1] & OK[:, :-2] & C[:, 2:] & C[:, 1:-1]; vf = v.float()
    jit = (TF.smooth_l1_loss(d2, torch.zeros_like(d2), beta=0.02, reduction='none').sum(-1) * vf).sum() / vf.sum().clamp(min=1)
    return acc + lam * jit, acc, jit


def loss_np(out, tgt, con, lam=1.0):
    """validation loss of one method on per-id row sequences: lists of (out [n,4], tgt [n,4] nan = none, con [n] consecutive flag). -> (loss, acc, jit, counts)"""
    sa = 0.; na = 0; sj = 0.; nj = 0
    for o, g, c in zip(out, tgt, con):
        ok = np.isfinite(g[:, 0])
        if not ok.any(): continue
        e = err_vec(o, np.where(ok[:, None], g, 1.)); sa += sl1(e[ok]).sum(); na += int(ok.sum())
        if len(o) >= 3:
            d2 = e[2:] - 2 * e[1:-1] + e[:-2]; v = ok[2:] & ok[1:-1] & ok[:-2] & (c[2:] > 0) & (c[1:-1] > 0)
            sj += sl1(d2[v]).sum(); nj += int(v.sum())
    acc = float(sa) / max(na, 1); jit = float(sj) / max(nj, 1); return acc + lam * jit, acc, jit, (na, nj)


# ------------------------------------------------------------------------------------------------------------------ numpy refiner (server forward)
class RefNumpy:
    def __init__(self, w):
        g = lambda k: np.asarray(w[k], np.float64)
        self.wih, self.whh, self.bih, self.bhh, self.wo, self.bo = g('cell.weight_ih'), g('cell.weight_hh'), g('cell.bias_ih'), g('cell.bias_hh'), g('head.weight'), g('head.bias'); self.n = self.whh.shape[1]

    def step(self, x, h):
        n = self.n; gi = self.wih @ x + self.bih; gh = self.whh @ h + self.bhh; sig = lambda a: 1. / (1. + np.exp(-a))
        r = sig(gi[:n] + gh[:n]); zz = sig(gi[n:2 * n] + gh[n:2 * n]); c = np.tanh(gi[2 * n:] + r * gh[2 * n:]); h = (1. - zz) * c + zz * h
        return self.wo @ h + self.bo, h


# ------------------------------------------------------------------------------------------------------------------ server-like rows
def rows_of(root, r, stream):
    """tracker rows of one stream -> per id dict(f [n], z [n,4] detector cxcywh, kp [n,17,3], trk [n,4] tracker cxcywh, tgt [n,4] h2h cxcywh (nan), hg [n], src [n], score [n])."""
    S = PL.load_seq(root, r); tr = np.load(Path(root) / 'data' / 'tracks' / ('%s.npz' % r['name'])); rows = tr[stream]
    tg = P.xyxy2cxcywh(S['a']['tgt_h2h']); hgt = S['a']['gbox'][:, 3] - S['a']['gbox'][:, 1]; ids = defaultdict(list)
    for row in rows: ids[int(row[1])].append(row)
    out = {}
    for i, rr in ids.items():
        rr = np.array(rr); rr = rr[np.argsort(rr[:, 0])]; n = len(rr)
        f = rr[:, 0].astype(np.int64); src = rr[:, 6].astype(np.int64); di = rr[:, 7].astype(np.int64)
        z = np.zeros((n, 4)); kp = np.zeros((n, 17, 3)); t = np.full((n, 4), np.nan); hg = np.full(n, np.nan); sc = np.zeros(n)
        for j in range(n):
            k = DETS[src[j]]; d = S['d'][k]; z[j] = P.xyxy2cxcywh(d['box'][di[j]]); kp[j] = d['kpt'][di[j]]; sc[j] = d['score'][di[j]]
            g = S['a']['dg_' + k][di[j]]
            if g >= 0: t[j] = tg[g]; hg[j] = hgt[g]
        out[i] = dict(f=f, z=z, kp=kp, trk=P.xyxy2cxcywh(rr[:, 2:6]), tgt=t, hg=hg, src=src, score=sc)
    return out, S


def segments(e):
    """split one id's rows where the gap is > LOST_MAX frames or the detector box jumps > JUMP_H heights (state reset rule of the server module)."""
    cut = [0]
    for j in range(1, len(e['f'])):
        gap = e['f'][j] - e['f'][j - 1]; jump = np.hypot(*(e['z'][j, :2] - e['z'][j - 1, :2])) > JUMP_H * max(e['z'][j - 1, 3], 1.)
        if gap > P.LOST_MAX or jump: cut.append(j)
    cut.append(len(e['f'])); return [slice(a, b) for a, b in zip(cut[:-1], cut[1:])]


def composed(W, f_prev, f):
    H = np.eye(3)
    for k in range(f_prev + 1, f + 1):
        if k < len(W): H = P.h3(W[k]) @ H
    return H[:2]


def run_euro(e, W, fps, params, comp=True):
    out = np.zeros_like(e['z'])
    for sl in segments(e):
        eu = Euro(*params, comp=comp); fp = None
        for j in range(sl.start, sl.stop):
            f = e['f'][j]; H = composed(W, fp, f) if fp is not None else None; dt = (f - fp) / fps if fp is not None else 1.0 / fps
            out[j] = eu.step(e['z'][j], dt, H); fp = f
    return out


def run_ref(e, W, net_np):
    """learned refiner over one id's rows (numpy forward, state kept across short gaps, reset at a segment start)."""
    out = np.zeros_like(e['z'])
    for sl in segments(e):
        fs = e['f'][sl]; det = {int(f): (e['z'][j], e['kp'][j]) for j, f in zip(range(sl.start, sl.stop), fs)}
        ser = P.ref_series(int(fs[0]), int(fs[-1]), det, W); h = np.zeros(net_np.n)
        for j, f in zip(range(sl.start, sl.stop), fs):
            x, z = ser[int(f)]; o, h = net_np.step(x.astype(np.float64), h); out[j] = P.ref_apply(z, o)
    return out


def blend(e, out, fps, T, switch_frames):
    """server guarantee: at a detector switch the box offset (centre / height, log size) between the track's previous output and its new output
    decays linearly to 0 over T seconds."""
    if T <= 0 or not len(switch_frames): return out
    out = out.copy(); sws = np.asarray(switch_frames)
    for j in range(1, len(e['f'])):
        f = e['f'][j]
        if e['src'][j] == e['src'][j - 1] or f - e['f'][j - 1] > 3: continue
        prev = out[j - 1]; new = out[j].copy(); h = max(new[3], 1.)
        off = np.r_[(prev[:2] - new[:2]) / h, np.log(max(prev[2], 1.) / max(new[2], 1.)), np.log(max(prev[3], 1.) / max(new[3], 1.))]
        for k in range(j, len(e['f'])):
            a = 1.0 - (e['f'][k] - f) / (T * fps)
            if a <= 0: break
            if k > j and e['src'][k] != e['src'][k - 1]: break
            b = out[k]; hk = max(b[3], 1.)
            out[k] = np.r_[b[:2] + a * off[:2] * hk, b[2] * math.exp(a * off[2]), b[3] * math.exp(a * off[3])]
    return out
