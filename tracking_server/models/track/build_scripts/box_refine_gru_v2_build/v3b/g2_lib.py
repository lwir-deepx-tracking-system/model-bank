"""gru_v2/scripts/g2_lib.py - GRU box refiner version 2 (rules: gru_v2/PREREG_GRU_V2.json).

One frame-by-frame, batch-over-people ENGINE builds every input (lookup conversion, fixed-gain velocity filter, camera warps, features) for
training, validation, the device-rate test and the cost test, so the offline numbers and the server port share one definition.
Round-1 code is imported read-only (scripts/prcommon, pos_lib, ref_lib, ref_eval; verify/combo/combo_lib). Nothing outside gru_v2/ is written.
"""
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G2 = HERE.parent                 # gru_v2/v3
ROOT = G2.parent.parent          # pos_refine_20261115
EXPORT_DIR = G2.parent / 'export_v3'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'verify' / 'combo'))
import prcommon as P      # noqa: E402
import pos_lib as PL      # noqa: E402
import ref_lib as R       # noqa: E402
import ref_eval as RE     # noqa: E402
import combo_lib as C     # noqa: E402

DETS = ('near', 'far')
LOST_MAX = 30            # state reset when more than this many frames were missed
JUMP_H = 1.5             # ... or the detection centre is farther than this many previous-box heights from the camera-moved previous detection
VB = 0.35                # fixed gain of the velocity filter (steady-state Kalman, no covariance)
NS, ND = 74, 4           # static inputs, inputs made from the previous output
NF = NS + ND
HID = 48
KC, KS = 0.4, math.log(2.2)   # correction range: centre +-0.4 converted-box heights, size x/ 2.2
CLIP = 5.0
EURO_V1 = (3.0, 3.0, 0.4, 0.3, 1.0)   # results/euro_valA.json (the round-1 / verifier One Euro)
AIM_Y = R.AIM_Y

# ------------------------------------------------------------------------------------------------ lookup conversion (train split, verify/gen_outputs.py)
# lut_train_fix.pkl = lut_train.pkl with the rows 'detector box >= 170 px' set to the 130-170 px row (identity): g2_lutfix.py, by-eye check 2026-11-15.
# The numbers before the fix (and this file before it) are kept in gru_v2/before_lutfix/.
# v3: the lookup refitted on the corrected targets (g2_data.py lut). Until it exists (only while it is being fitted) the v2 table is read.
_LP = G2 / 'data' / 'lut_v3.pkl'
_L = pickle.loads((_LP if _LP.exists() else G2.parent / 'data' / 'lut_train_fix.pkl').read_bytes())
LH_EDGES, AR_EDGES, TAB = _L['lh_edges'], _L['ar_edges'], _L['tab']
LH_CEN = (LH_EDGES[:-1] + LH_EDGES[1:]) / 2; LH_CEN[0] = math.log(20.); LH_CEN[-1] = math.log(280.)
AR_CEN = (AR_EDGES[:-1] + AR_EDGES[1:]) / 2; AR_CEN[0] = 0.27; AR_CEN[-1] = 0.90


def conv_hard(z):
    """the verifier's lookup: hard bins (baselines 'conv', 'conv_euro')."""
    z = np.asarray(z, np.float64); lh = np.log(np.maximum(z[:, 3], 1.)); ar = z[:, 2] / np.maximum(z[:, 3], 1.)
    bh = np.clip(np.searchsorted(LH_EDGES, lh, 'right') - 1, 0, len(LH_EDGES) - 2); ba = np.clip(np.searchsorted(AR_EDGES, ar, 'right') - 1, 0, len(AR_EDGES) - 2)
    e = TAB[bh, ba]; return np.stack([z[:, 0] + e[:, 0] * z[:, 3], z[:, 1] + e[:, 1] * z[:, 3], z[:, 2] * np.exp(e[:, 2]), z[:, 3] * np.exp(e[:, 3])], 1)


def conv_delta(z, tab=TAB, lh_cen=LH_CEN, ar_cen=AR_CEN):
    """the SAME table read with bilinear interpolation between the bin centres (no steps at the bin edges) -> delta [N,4] (dx / h, dy / h, log w ratio, log h ratio)."""
    z = np.asarray(z, np.float64); lh = np.log(np.maximum(z[:, 3], 1.)); ar = z[:, 2] / np.maximum(z[:, 3], 1.)
    i = np.clip(np.searchsorted(lh_cen, lh, 'right') - 1, 0, len(lh_cen) - 2); t = np.clip((lh - lh_cen[i]) / (lh_cen[i + 1] - lh_cen[i]), 0., 1.)[:, None]
    j = np.clip(np.searchsorted(ar_cen, ar, 'right') - 1, 0, len(ar_cen) - 2); u = np.clip((ar - ar_cen[j]) / (ar_cen[j + 1] - ar_cen[j]), 0., 1.)[:, None]
    return (1 - t) * (1 - u) * tab[i, j] + (1 - t) * u * tab[i, j + 1] + t * (1 - u) * tab[i + 1, j] + t * u * tab[i + 1, j + 1]


def conv_soft(z):
    z = np.asarray(z, np.float64); e = conv_delta(z)
    return np.stack([z[:, 0] + e[:, 0] * z[:, 3], z[:, 1] + e[:, 1] * z[:, 3], z[:, 2] * np.exp(e[:, 2]), z[:, 3] * np.exp(e[:, 3])], 1)


# ------------------------------------------------------------------------------------------------ numpy GRU (batch)
class GRUNp:
    def __init__(self, w):
        g = lambda k: np.asarray(w[k], np.float64)
        self.wih, self.whh, self.bih, self.bhh, self.wo, self.bo = g('cell.weight_ih'), g('cell.weight_hh'), g('cell.bias_ih'), g('cell.bias_hh'), g('head.weight'), g('head.bias')
        self.n = self.whh.shape[1]; self.arch = str(w['arch']) if 'arch' in w else 'gate'

    def step(self, x, h):
        """x [N,NF], h [N,n] -> o [N,8], h' [N,n]"""
        n = self.n; gi = x @ self.wih.T + self.bih; gh = h @ self.whh.T + self.bhh
        r = 1. / (1. + np.exp(-(gi[:, :n] + gh[:, :n]))); zz = 1. / (1. + np.exp(-(gi[:, n:2 * n] + gh[:, n:2 * n])))
        c = np.tanh(gi[:, 2 * n:] + r * gh[:, 2 * n:]); h = (1. - zz) * c + zz * h
        return h @ self.wo.T + self.bo, h


def _alpha(cut, dt): return 1.0 / (1.0 + 1.0 / (2 * math.pi * cut * dt))


# ------------------------------------------------------------------------------------------------ the engine
class Engine:
    """All people of a frame in one batch. step(ids, Z, KP, H, t) -> dict of arrays [N, ...] in the order of ids.
    State per id: previous detection box zp and previous output op (both kept in the CURRENT frame's coordinates: every live id is moved by every
    frame's camera warp), velocity v (px per frame), frames missed, rows since the state started, GRU state, size-filter state.
    net=None: inputs only (training / validation data). size=(mc_s, beta_s, dcut): One Euro on log w / log h after the GRU ('out_f')."""

    SPEC = dict(zp=((4,), np.float64, 1.), op=((4,), np.float64, 1.), v=((2,), np.float64, 0.), cam=((2,), np.float64, 0.), Ha=((3, 3), np.float64, 0.),
                h=((HID,), np.float64, 0.), sx=((2,), np.float64, 0.), sdx=((2,), np.float64, 0.), lsc=((), np.float64, 0.), tl=((), np.float64, 0.),
                miss=((), np.int64, 0), k=((), np.int64, 0), alive=((), bool, False))

    def __init__(self, net=None, size=None, cap=64):
        self.net = net; self.size = size; self.slot = {}; self.free = []; self.cap = 0
        for k, (shp, dt, fill) in self.SPEC.items(): setattr(self, k, np.full((0,) + shp, fill, dt))
        self._grow(cap)

    def _grow(self, cap):
        for k, (shp, dt, fill) in self.SPEC.items():
            b = np.full((cap,) + shp, fill, dt); b[:self.cap] = getattr(self, k); setattr(self, k, b)
        self.free = list(range(cap - 1, self.cap - 1, -1)) + self.free; self.cap = cap

    def _slot(self, i):
        s = self.slot.get(i)
        if s is None:
            if not self.free: self._grow(self.cap * 2)
            s = self.free.pop(); self.slot[i] = s; self.alive[s] = False
        return s

    def step(self, ids, Z, KP, H=None, t=0.0):
        # 1) every live id: one more frame, camera warp of the kept boxes
        a = np.flatnonzero(self.alive)
        if len(a):
            self.miss[a] += 1
            dead = a[self.miss[a] > LOST_MAX + 1]
            if len(dead):
                self.alive[dead] = False; ds = set(dead.tolist())
                for i in [i for i, s in self.slot.items() if s in ds]: self.free.append(self.slot.pop(i))
                a = np.flatnonzero(self.alive)
            if H is not None and len(a):
                Rm = H[:2, :2]; tr = H[:2, 2]; sc = math.sqrt(abs(Rm[0, 0] * Rm[1, 1] - Rm[0, 1] * Rm[1, 0])); c0 = self.zp[a, :2]
                self.zp[a, :2] = c0 @ Rm.T + tr; self.zp[a, 2:] *= sc; self.cam[a] = self.zp[a, :2] - c0
                self.op[a, :2] = self.op[a, :2] @ Rm.T + tr; self.op[a, 2:] *= sc; self.v[a] = self.v[a] @ Rm.T; self.lsc[a] += math.log(max(sc, 1e-6))
                H3 = np.vstack([H, [0, 0, 1]]); self.Ha[a] = H3 @ self.Ha[a]
            else: self.cam[a] = 0.
        N = len(ids)
        if N == 0: return dict(out=np.zeros((0, 4)), out_f=np.zeros((0, 4)), X=np.zeros((0, NS)), C=np.zeros((0, 4)), Ha=np.zeros((0, 2, 3)), start=np.zeros(0, bool), k=np.zeros(0, np.int64))
        Z = np.asarray(Z, np.float64).reshape(N, 4); Z = np.c_[Z[:, :2], np.maximum(Z[:, 2:], 1.)]
        s = np.array([self._slot(i) for i in ids]); new = ~self.alive[s]; zp = self.zp[s]; h = Z[:, 3]; w = Z[:, 2]
        jump = ~new & (np.hypot(Z[:, 0] - zp[:, 0], Z[:, 1] - zp[:, 1]) > JUMP_H * np.maximum(zp[:, 3], 1.))
        st = new | jump; gap = np.where(st, 0, self.miss[s] - 1); n1 = (gap + 1.)[:, None]
        v0 = np.where(st[:, None], 0., self.v[s]); zpc = np.where(st[:, None], Z[:, :2], zp[:, :2]); zps = np.where(st[:, None], Z[:, 2:], zp[:, 2:])
        inn = (Z[:, :2] - (zpc + v0 * n1)) / h[:, None]; v = v0 + VB * ((Z[:, :2] - zpc) / n1 - v0)
        dC = conv_delta(Z); Cb = np.stack([Z[:, 0] + dC[:, 0] * h, Z[:, 1] + dC[:, 1] * h, w * np.exp(dC[:, 2]), h * np.exp(dC[:, 3])], 1)
        X = np.empty((N, NS))
        X[:, 0:2] = v / h[:, None]; X[:, 2:4] = inn; X[:, 4:6] = (Z[:, :2] - zpc) / h[:, None]; X[:, 6:8] = np.log(Z[:, 2:] / np.maximum(zps, 1.))
        X[:, 8] = np.log(h / 100.); X[:, 9] = np.log(w / h)
        X[:, 10] = (Z[:, 0] - w / 2) / h; X[:, 11] = (P.XW - (Z[:, 0] + w / 2)) / h; X[:, 12] = (Z[:, 1] - h / 2 - P.Y0) / h; X[:, 13] = (P.Y1 - (Z[:, 1] + h / 2)) / h
        if KP is None: X[:, 14:65] = 0.
        else:
            KP = np.asarray(KP, np.float64).reshape(N, 17, 3); vis = (KP[:, :, 2] >= 0.5).astype(np.float64)
            X[:, 14:65] = np.stack([(KP[:, :, 0] - Z[:, None, 0]) / h[:, None] * vis, (KP[:, :, 1] - Z[:, None, 1]) / h[:, None] * vis, vis], 2).reshape(N, 51)
        kk = np.where(st, 0, self.k[s] + 1)
        X[:, 65] = np.minimum(gap, LOST_MAX) / 15.; X[:, 66:68] = np.where(st[:, None], 0., self.cam[s]) / h[:, None]; X[:, 68:72] = dC
        X[:, 72] = st; X[:, 73] = np.minimum(kk, 10) / 10.; np.clip(X, -CLIP, CLIP, out=X)
        Ha = np.where(st[:, None, None], np.eye(3)[None], self.Ha[s])[:, :2]
        res = dict(X=X, C=Cb, Ha=Ha, start=st, k=kk, gap=gap)
        if self.net is not None:
            out, hn = model_step_np(self.net, X, Cb, self.op[s], np.where(st[:, None], 0., self.h[s]), st)
            self.h[s] = hn; self.op[s] = out; res['out'] = out
            if self.size is not None:
                mcs, bs, dcut = self.size; ls = np.log(np.maximum(out[:, 2:], 1e-3)); dt = np.maximum(t - self.tl[s], 1e-3)[:, None]
                xp = self.sx[s] + self.lsc[s][:, None]; d = (ls - xp) / dt; ad = _alpha(dcut, dt); dx = ad * d + (1 - ad) * self.sdx[s]
                al = _alpha(mcs + bs * np.abs(dx), dt); xn = al * ls + (1 - al) * xp
                xn = np.where(st[:, None], ls, xn); dx = np.where(st[:, None], 0., dx); self.sx[s] = xn; self.sdx[s] = dx
                res['out_f'] = np.c_[out[:, :2], np.exp(xn)]
        self.zp[s] = Z; self.v[s] = v; self.miss[s] = 0; self.k[s] = kk; self.alive[s] = True; self.Ha[s] = np.eye(3); self.lsc[s] = 0.; self.tl[s] = t
        return res


def model_step_np(net, X, Cb, op, h, st):
    """one GRU step for N people. Cb = converted boxes, op = previous outputs in this frame's coordinates, st = state starts here."""
    Cs = np.c_[Cb[:, :2], np.log(Cb[:, 2:])]; hc = Cb[:, 3:4]; gate = net.arch == 'gate'
    ops = np.c_[op[:, :2], np.log(np.maximum(op[:, 2:], 1e-3))]; ops = np.where(st[:, None], Cs, ops)
    prel = np.c_[(ops[:, :2] - Cs[:, :2]) / hc, ops[:, 2:] - Cs[:, 2:]] if gate else np.zeros((len(X), ND))
    o, hn = net.step(np.c_[X, np.clip(prel, -CLIP, CLIP)], h)
    base = ops + (1. / (1. + np.exp(-o[:, :4]))) * (Cs - ops) if gate else Cs
    cen = base[:, :2] + KC * hc * np.tanh(o[:, 4:6]); ls = base[:, 2:] + KS * np.tanh(o[:, 6:8])
    return np.c_[cen, np.exp(ls)], hn


# ------------------------------------------------------------------------------------------------ torch model and loss
def make_net(arch='gate', hidden=HID):
    import torch

    class RefNet2(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.cell = torch.nn.GRUCell(NF, hidden); self.head = torch.nn.Linear(hidden, 8); self.nh = hidden; self.arch = arch
            torch.nn.init.zeros_(self.head.weight); torch.nn.init.zeros_(self.head.bias)
            with torch.no_grad(): self.head.bias[:4] = 1.0     # first gate 0.73: start close to the converted box

        def forward(self, X, Cb, Ha, ST, M, state=None):
            """X [B,T,NS], Cb [B,T,4] converted boxes, Ha [B,T,2,3] warp since the row before, ST [B,T] state start, M [B,T] valid -> boxes [B,T,4], state"""
            B, T = X.shape[:2]; gate = self.arch == 'gate'
            Cs = torch.cat([Cb[..., :2], torch.log(Cb[..., 2:].clamp(min=1e-3))], -1)
            h, op = state if state is not None else (X.new_zeros(B, self.nh), Cs[:, 0])
            out = []
            for t in range(T):
                st = ST[:, t:t + 1]; cs = Cs[:, t]; hc = Cb[:, t, 3:4]; Hm = Ha[:, t]
                sc = torch.log(torch.sqrt(torch.abs(Hm[:, 0, 0] * Hm[:, 1, 1] - Hm[:, 0, 1] * Hm[:, 1, 0])).clamp(min=1e-6))[:, None]
                opw = torch.cat([torch.einsum('bij,bj->bi', Hm[:, :, :2], op[:, :2]) + Hm[:, :, 2], op[:, 2:] + sc], 1); opw = torch.where(st, cs, opw)
                prel = torch.cat([(opw[:, :2] - cs[:, :2]) / hc, opw[:, 2:] - cs[:, 2:]], 1).clamp(-CLIP, CLIP) if gate else X.new_zeros(B, ND)
                hn = self.cell(torch.cat([X[:, t], prel], 1), torch.where(st, torch.zeros_like(h), h)); o = self.head(hn)
                base = opw + torch.sigmoid(o[:, :4]) * (cs - opw) if gate else cs
                q = torch.cat([base[:, :2] + KC * hc * torch.tanh(o[:, 4:6]), base[:, 2:] + KS * torch.tanh(o[:, 6:8])], 1)
                m = M[:, t:t + 1]; h = torch.where(m, hn, h); op = torch.where(m, q, opw); out.append(q)
            q = torch.stack(out, 1); return torch.cat([q[..., :2], torch.exp(q[..., 2:])], -1), (h, op)
    return RefNet2()


def loss_parts(q, G, OK, CON, K, Cb, Ha, M, TR, Zr):
    """sums and counts of the loss terms for one batch (torch). q, G, Cb [B,T,4] cxcywh; OK target present; CON consecutive frame with the row before
    in the same state; K rows since the state started. -> dict name -> (sum, count); lag -> signed sums per speed bin."""
    import torch, torch.nn.functional as TF
    sl1 = lambda x: TF.smooth_l1_loss(x, torch.zeros_like(x), beta=0.02, reduction='none')
    Gs = torch.where(OK[..., None], G, torch.ones_like(G)); e = R.err_vec(q, Gs); wr = (1. + 2. * (K < 5).float()) * OK.float()
    P_ = dict(acc=((sl1(e).sum(-1) * wr).sum(), wr.sum()))
    s = torch.cat([q[..., :2], torch.log(q[..., 2:].clamp(min=1e-3))], -1); hq = q[..., 3:4].detach().clamp(min=1.)
    Rm = Ha[:, 1:, :, :2]; sc = torch.log(torch.sqrt(torch.abs(Rm[..., 0, 0] * Rm[..., 1, 1] - Rm[..., 0, 1] * Rm[..., 1, 0])).clamp(min=1e-6))[..., None]
    p1 = torch.cat([torch.einsum('btij,btj->bti', Rm, s[:, :-1, :2]) + Ha[:, 1:, :, 2], s[:, :-1, 2:] + sc], -1)
    d1 = s[:, 1:] - p1; v1 = CON[:, 1:] & M[:, 1:]                                   # own first difference at t = 1.., camera-compensated
    d1n = torch.cat([d1[..., :2] / hq[:, 1:], d1[..., 2:]], -1)
    d1p = torch.cat([torch.einsum('btij,btj->bti', Rm[:, 1:], d1[:, :-1, :2]), d1[:, :-1, 2:]], -1)
    d2 = d1[:, 1:] - d1p; d2n = torch.cat([d2[..., :2] / hq[:, 2:], d2[..., 2:]], -1); v2 = (v1[:, 1:] & v1[:, :-1]).float()
    P_['jit'] = ((sl1(d2n).sum(-1) * v2).sum(), v2.sum())
    stp = torch.relu(d1n[..., 3].abs() - 0.04) + 0.5 * torch.relu(d1n[..., 2].abs() - 0.06); P_['step'] = ((stp * v1.float()).sum(), v1.float().sum())
    vt = Gs[:, 1:, :2] - Gs[:, :-1, :2]; sp = torch.hypot(vt[..., 0], vt[..., 1]); u = vt / sp.clamp(min=1e-9)[..., None]; ht = Gs[:, 1:, 3]
    al = ((q[:, 1:, :2] - Cb[:, 1:, :2]) * u).sum(-1) / ht; vl = OK[:, 1:] & OK[:, :-1] & v1; spn = sp / ht
    alt = ((TR[:, 1:, :2] - Zr[:, 1:, :2]) * u).sum(-1) / ht                       # the tracker row's own lag against the raw detection
    for nm, lo, hi in (('lag_mid', 0.03, 0.1), ('lag_fast', 0.1, 1e9)):
        mm = (vl & (spn >= lo) & (spn < hi)).float(); P_[nm] = ((al * mm).sum(), mm.sum()); P_[nm + '_trk'] = ((alt * mm).sum(), mm.sum())
    # v3: aim-point jitter of the output must not exceed the tracker row's on the same rows (camera-compensated second difference of top + 0.37 h)
    def d2aim(b):
        a = torch.cat([b[..., 0:1], b[..., 1:2] + (AIM_Y - 0.5) * b[..., 3:4]], -1)
        d = a[:, 1:] - (torch.einsum('btij,btj->bti', Rm, a[:, :-1]) + Ha[:, 1:, :, 2])
        return d[:, 1:] - torch.einsum('btij,btj->bti', Rm[:, 1:], d[:, :-1])
    mo = torch.linalg.norm(d2aim(q), dim=-1) / hq[:, 2:, 0]; mt = torch.linalg.norm(d2aim(TR), dim=-1) / hq[:, 2:, 0]
    P_['aimtrk'] = ((sl1(torch.relu(mo - mt)) * v2).sum(), v2.sum())
    return P_


def loss_total(P_, lam_j, lam_l, lam_s=2.0):
    """v3: acc + lam_j * (own jitter + aim jitter above the tracker row's) + lam_s * height / width step hinge
    + lam_l * sum over the two speed bins of (relu(tracker row's mean lag - mean lag) + relu(mean lag)): the mean lag may sit anywhere between the tracker row's and 0."""
    import torch
    f = lambda k: P_[k][0] / P_[k][1].clamp(min=1)
    lag = 0.
    for k in ('lag_mid', 'lag_fast'):
        if float(P_[k][1]) >= 8:
            mo = P_[k][0] / P_[k][1]; mt = torch.clamp(P_[k + '_trk'][0] / P_[k][1], max=0.); lag = lag + torch.relu(mt - mo) + torch.relu(mo)
    return f('acc') + lam_j * (f('jit') + f('aimtrk')) + lam_s * f('step') + lam_l * lag


def torch_abs(x):
    import torch
    return torch.abs(x)


# ------------------------------------------------------------------------------------------------ running the engine over rows
def run_rows(ids, W, fps, n, net=None, size=None, times=None):
    """ids {tid: dict(f [n], z [n,4], kp [n,17,3], ...)} -> per tid dict(X, C, Ha, start, k, [out, out_f]) arrays in row order."""
    by_f = defaultdict(list)
    for tid in sorted(ids):
        for j, f in enumerate(ids[tid]['f']): by_f[int(f)].append((tid, j))
    eng = Engine(net, size); keys = ['X', 'C', 'Ha', 'start', 'k'] + (['out'] if net is not None else []) + (['out_f'] if (net is not None and size is not None) else [])
    shp = dict(X=(NS,), C=(4,), Ha=(2, 3), start=(), k=(), out=(4,), out_f=(4,))
    o = {tid: {k: np.zeros((len(e['f']),) + shp[k], bool if k == 'start' else (np.int64 if k == 'k' else np.float64)) for k in keys} for tid, e in ids.items()}
    for f in range(n):
        its = by_f.get(f, ()); H = W[f] if (W is not None and 0 < f < len(W)) else None; t = float(times[f]) if times is not None else f / fps
        r = eng.step([a for a, _ in its], np.array([ids[a]['z'][j] for a, j in its]).reshape(-1, 4), np.array([ids[a]['kp'][j] for a, j in its]).reshape(-1, 17, 3), H, t)
        for q, (a, j) in enumerate(its):
            for k in keys: o[a][k][j] = r[k][q]
    return o


def size_filter(e, out, start, W, fps, params, times=None):
    """One Euro on log w / log h of the GRU output of one id (the engine's 'out_f', used for the val-A grid). centre untouched."""
    mcs, bs, dcut = params; f = e['f']; res = out.copy(); x = dx = None; fp = None
    for j in range(len(f)):
        ls = np.log(np.maximum(out[j, 2:], 1e-3)); tj = float(times[f[j]]) if times is not None else f[j] / fps
        if start[j] or x is None: x = ls.copy(); dx = np.zeros(2)
        else:
            sc = 0.
            for k in range(fp + 1, f[j] + 1):
                if W is not None and 0 < k < len(W): sc += math.log(max(math.sqrt(abs(W[k][0, 0] * W[k][1, 1] - W[k][0, 1] * W[k][1, 0])), 1e-6))
            dt = max(tj - tp, 1e-3); xp = x + sc; d = (ls - xp) / dt; ad = _alpha(dcut, dt); dx = ad * d + (1 - ad) * dx
            al = _alpha(mcs + bs * np.abs(dx), dt); x = al * ls + (1 - al) * xp
        res[j, 2:] = np.exp(x); fp = int(f[j]); tp = tj
    return res


def euro_rows(e, boxes, start, W, fps, params, times=None):
    """One Euro (ref_lib.Euro, camera compensated) over boxes [n,4] of one id with the ENGINE's state starts (same reset rule for every method)."""
    out = np.zeros_like(boxes); eu = None; fp = None; f = e['f']
    for j in range(len(f)):
        tj = float(times[f[j]]) if times is not None else f[j] / fps
        if start[j] or eu is None: eu = R.Euro(*params, comp=True); out[j] = eu.step(boxes[j], 1.0 / fps, None)
        else: out[j] = eu.step(boxes[j], max(tj - tp, 1e-3), R.composed(W, fp, int(f[j])) if W is not None else None)
        fp = int(f[j]); tp = tj
    return out


# ------------------------------------------------------------------------------------------------ metrics (verify/combo definitions + lag against the detection, first rows)
def aim_of(b): return np.stack([b[:, 0], b[:, 1] - b[:, 3] / 2 + AIM_Y * b[:, 3]], 1)


CONVERTED = ('conv', 'conv_euro', 'conv_euro_t', 'gru_v1', 'gru_v2', 'gru_v3', 'gru_v3_size')
KEYS = C.KEYS + ('along_det', 'along_raw', 'aim_det', 'k')


def rows_metrics(rec, o, m):
    """combo_lib.rows_metrics + lag against the CURRENT detection along the target motion.
    along_det: against the detection in the method's own box definition (converted methods: the lookup-converted detection; raw / tracker / euro: the raw detection).
    along_raw: against the raw detection centre for every method (the verifier's definition)."""
    d = C.rows_metrics(rec, o); e = rec['e']; g = e['tgt']; ok = d['ok']; f = e['f']; n = len(f); z = e['z']; ref = e['conv'] if m in CONVERTED else z
    gs = np.where(ok[:, None], g, 1.); ht = gs[:, 3]; j = np.flatnonzero(np.r_[False, np.diff(f) == 1] & ok & np.r_[False, ok[:-1]])
    ad = np.full(n, np.nan); ar = np.full(n, np.nan); aa = np.full(n, np.nan)
    if len(j):
        v = gs[j, :2] - gs[j - 1, :2]; sp = np.hypot(*v.T); u = v / np.maximum(sp, 1e-9)[:, None]
        ad[j] = np.sum((o[j, :2] - ref[j, :2]) * u, 1) / ht[j]; ar[j] = np.sum((o[j, :2] - z[j, :2]) * u, 1) / ht[j]
        aa[j] = np.sum((aim_of(o)[j] - aim_of(ref)[j]) * u, 1) / ht[j]
    d['along_det'] = ad; d['along_raw'] = ar; d['aim_det'] = aa; d['k'] = e['k']
    return d


def events(rec):
    """combo_lib.events with the track-start rows from the ENGINE's state (rows since the state started k): first5 = rows 1-4 (steps inside the first 5 rows),
    row1 = row 1, later = rows >= 5."""
    ev, sb = C.events(rec); kk = rec['e']['k']
    ev['start'] = (kk == 1) | (kk == 2); ev['first5'] = (kk >= 1) & (kk <= 4); ev['row1'] = kk == 1; ev['later'] = kk >= 5
    return ev, sb


def gather(recs, m, evs):
    names = sorted({r['name'] for r in recs}); loc = {n: i for i, n in enumerate(names)}
    rows = defaultdict(list); seqv = []; idv = []; ev = defaultdict(list); sbv = []
    for k, (rec, (em, sb)) in enumerate(zip(recs, evs)):
        d = rows_metrics(rec, rec['o'][m], m)
        for kk in KEYS: rows[kk].append(d[kk])
        n = len(rec['e']['f']); seqv.append(np.full(n, loc[rec['name']])); idv.append(np.full(n, k)); sbv.append(sb)
        for kk, v in em.items(): ev[kk].append(v)
    return {k: np.concatenate(v) for k, v in rows.items()}, np.concatenate(seqv), np.concatenate(idv), {k: np.concatenate(v) for k, v in ev.items()}, np.concatenate(sbv), names


def mean_(v):
    v = v[np.isfinite(v)]; return float(v.mean()) if len(v) else None


def summary(rows, ev, idv, sbv):
    """combo_lib.summary + lag against the detection per speed bin (all sizes and >= 100 px), first-5-rows numbers for people under 100 px."""
    s = C.summary(rows, ev, idv); ok = rows['ok']
    for a, b in C.SPEED_BINS:
        mm = ok & (rows['spd'] >= a) & (rows['spd'] < b); big = mm & (sbv == 2)
        s['lag_%g-%g' % (a, b)].update(along_det=mean_(rows['along_det'][mm]), along_raw=mean_(rows['along_raw'][mm]), aim_det=mean_(rows['aim_det'][mm]),
                                       n_big=int(big.sum()), along_det_big=mean_(rows['along_det'][big]), along_raw_big=mean_(rows['along_raw'][big]))
    small = ok & (sbv >= 0) & (sbv <= 1); fin = np.isfinite(rows['dh'])
    f5 = small & ev['first5'] & fin; r1 = small & ev['row1'] & fin
    s['first5'] = dict(n=int(f5.sum()), dh_mean=mean_(rows['dh'][f5]), dh_p95=float(np.percentile(rows['dh'][f5], 95)) if f5.sum() >= 5 else None,
                       da_mean=mean_(rows['da'][f5]), n_row1=int(r1.sum()), row1_over_015=float(np.mean(rows['dh'][r1] > 0.15)) if r1.any() else None,
                       hr_row0=float(np.median(rows['hr'][small & (rows['k'] == 0)])) if (small & (rows['k'] == 0)).any() else None,
                       hr_row1=float(np.median(rows['hr'][small & (rows['k'] == 1)])) if (small & (rows['k'] == 1)).any() else None,
                       hr_row4=float(np.median(rows['hr'][small & (rows['k'] == 4)])) if (small & (rows['k'] == 4)).any() else None)
    allf5 = ok & ev['first5'] & fin; s['first5_all'] = dict(n=int(allf5.sum()), dh_mean=mean_(rows['dh'][allf5]), da_mean=mean_(rows['da'][allf5]))
    return s


def sel_numbers(s):
    """the numbers of the val-A selection rule and the val-B pass rules from one summary."""
    return dict(dh_p99=s['dh_p99'], tmax=s['dh_trackmax_med'], d2a=s['d2a_rms'], d2c=s['d2c_rms'], first5=s['first5']['dh_mean'], row1_over=s['first5']['row1_over_015'],
                aim=s['aim'], ce=s['ce'], lh=s['lh'], da_p99=s['da_p99'], lag_fast=s['lag_0.1-99']['along_det'], lag_mid=s['lag_0.03-0.1']['along_det'],
                lag_fast_big=s['lag_0.1-99']['along_det_big'], lag_fast_raw_big=s['lag_0.1-99']['along_raw_big'], slag=s['slag_0.03-99']['salong'])


def load_net(path):
    z = np.load(path, allow_pickle=False); return GRUNp({k: z[k] for k in z.files})


# ------------------------------------------------------------------------------------------------ v3: corrected target (PREREG amendment 2)
HIP_F, HIP_MIN, HIP_CAP = 0.95, 0.80, 0.03


def fix_target(t, z, kp):
    """t [n,4] head-to-hip target cxcywh (nan = none), z [n,4] the matched detection cxcywh, kp [n,17,3] its keypoints -> corrected target.
    Where the detection is a body box by its own keypoints (both hips visible >= 0.5 and at >= 0.80 of the detection box height), the target bottom is
    at least top + (hip line - top) / 0.95 (hips at 0.95 of the box, the detectors' own convention on the train split), and not below the detection
    bottom + 3 % of its height. Top, centre x and width unchanged. Everything else (whole-person detections, hips not visible) unchanged."""
    t = np.array(t, np.float64); z = np.asarray(z, np.float64); kp = np.asarray(kp, np.float64)
    top = t[:, 1] - t[:, 3] / 2; bot = t[:, 1] + t[:, 3] / 2; zh = np.maximum(z[:, 3], 1.); ztop = z[:, 1] - z[:, 3] / 2
    hy = (kp[:, 11, 1] + kp[:, 12, 1]) / 2; body = (kp[:, 11, 2] >= 0.5) & (kp[:, 12, 2] >= 0.5) & ((hy - ztop) / zh >= HIP_MIN) & np.isfinite(t[:, 0]) & (hy > top)
    with np.errstate(invalid='ignore'):
        bk = top + (hy - top) / HIP_F; cap = np.maximum(bot, z[:, 1] + z[:, 3] / 2 + HIP_CAP * zh); nb = np.where(body, np.minimum(np.maximum(bot, bk), cap), bot)
    t[:, 2] = t[:, 2] * np.where(body, (nb - top) / np.maximum(bot - top, 1e-6), 1.)     # v3b (amendment 3): the width grows by the same factor as the height
    t[:, 3] = nb - top; t[:, 1] = top + t[:, 3] / 2
    return t, body & (nb > bot + 1e-9)
