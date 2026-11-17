"""m4_model: learned position predictor arms (PREREG_MOTION.json). Needs numpy; torch only for the arms (the Kalman filter and the GRU numpy forward work without torch).
  KalmanXYWH   numpy copy of the deployed filter (Ultralytics KalmanFilterXYWH: weights 1/20, 1/160, dt = 1 frame; BOTrack.multi_gmc warp), no ultralytics import
  GRUArm       1-layer GRU cell, hidden 40, linear head (7,124 parameters)
  MambaArm     Mamba-style selective state space block (Gu & Dao 2023), plain PyTorch: d_model 32, expand 2, d_state 8, d_conv 4, one-step recurrence (9,124 parameters)
  GRUNumpy     numpy forward of GRUArm (the tracker runs numpy on the M1); rollout_np = frame-by-frame reference of the whole gap prediction
Role of both arms: RESIDUAL on the deployed prediction. Every frame: p = Kalman predict + camera-motion warp (kf_gmc); the arm outputs o (4 numbers);
  centre q_c = p_c + 0.5 * p_h * tanh(o[:2])  (at most 0.5 box heights per frame),  size q_wh = p_wh * exp(ln(1.1) * tanh(o[2:]))  (at most 10 % per frame).
  The head starts at zero, so an untrained arm IS the deployed kf_gmc. While the person is lost the corrected box q replaces the position part of the Kalman state
  (own prediction fed back, MambaTrack rule); velocity is left to the Kalman filter.
Input per frame (NF = 16, all divided by the predicted box height): Kalman velocity (2), last innovation (4, zero while lost), observed flag (1), camera shift of the box centre (2),
  last box change (4), log h / 100, log w / h, lost frames so far / 60.
Self test (CPU):  python -I m4_model.py"""
import math
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script folder from the path
import numpy as np
try:
    import torch, torch.nn as nn, torch.nn.functional as TF
except ImportError:  # baselines and the numpy GRU do not need torch
    torch = None; nn = None

NF = 16; MAX_C = 0.5; MAX_S = math.log(1.1); HIST = 20; MINOBS = 5; STEPS = 60; CLIP = 5.0


# ----------------------------------------------------------------------------------------------------------------- deployed Kalman filter (numpy)
class KalmanXYWH:
    """State (cx, cy, w, h, vx, vy, vw, vh). Same numbers as ultralytics.trackers.utils.kalman_filter.KalmanFilterXYWH."""
    SP = 1. / 20; SV = 1. / 160

    def __init__(self):
        self.F = np.eye(8); self.F[:4, 4:] = np.eye(4); self.Hm = np.eye(4, 8)

    def initiate(self, z):
        z = np.asarray(z, np.float64); mean = np.r_[z, np.zeros(4)]; w, h = z[2], z[3]
        std = [2 * self.SP * w, 2 * self.SP * h, 2 * self.SP * w, 2 * self.SP * h, 10 * self.SV * w, 10 * self.SV * h, 10 * self.SV * w, 10 * self.SV * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov, lost=False):
        """lost=True: vw = vh = 0 before the step (BOTrack.predict for a track that is not in state Tracked)."""
        if lost: mean = mean.copy(); mean[6] = 0.; mean[7] = 0.
        w, h = mean[2], mean[3]
        std = [self.SP * w, self.SP * h, self.SP * w, self.SP * h, self.SV * w, self.SV * h, self.SV * w, self.SV * h]
        return mean @ self.F.T, self.F @ cov @ self.F.T + np.diag(np.square(std))

    def project(self, mean, cov):
        w, h = mean[2], mean[3]; std = [self.SP * w, self.SP * h, self.SP * w, self.SP * h]
        return self.Hm @ mean, self.Hm @ cov @ self.Hm.T + np.diag(np.square(std))

    def update(self, mean, cov, z):
        pm, pc = self.project(mean, cov); K = np.linalg.solve(pc, (cov @ self.Hm.T).T).T; inn = np.asarray(z, np.float64) - pm
        return mean + inn @ K.T, cov - K @ pc @ K.T

    @staticmethod
    def gmc(mean, cov, A):
        """BOTrack.multi_gmc: 2x3 warp A applied to the state after predict."""
        R8 = np.kron(np.eye(4), A[:2, :2]); m = R8 @ mean; m[:2] += A[:2, 2]
        return m, R8 @ cov @ R8.T


# ----------------------------------------------------------------------------------------------------------------- shared feature (numpy or torch)
def feat(lib, v, whp, inn, flag, cam, m1, m2, cnt):
    """All [B, k]. v Kalman velocity after predict + warp; whp predicted w, h; inn last innovation (normalised); flag 1 = last frame observed;
    cam camera shift of the last centre in px; m1, m2 last and second last box (cx, cy, w, h); cnt lost frames so far / 60."""
    if lib is np: cat = lambda a: np.concatenate(a, 1); log = np.log; lo = lambda a: np.maximum(a, 1.0); clip = lambda a: np.clip(a, -CLIP, CLIP)
    else: cat = lambda a: torch.cat(a, 1); log = torch.log; lo = lambda a: a.clamp(min=1.0); clip = lambda a: a.clamp(-CLIP, CLIP)
    h = lo(whp[:, 1:2]); w = lo(whp[:, 0:1])
    return clip(cat([v / h, inn, flag, cam / h, (m1[:, :2] - m2[:, :2]) / h, log(lo(m1[:, 2:]) / lo(m2[:, 2:])), log(h / 100.), log(w / h), cnt]))


def innovation(z, p):
    """numpy, single box: observed box z against predicted state p -> (dcx / h, dcy / h, log w ratio, log h ratio)."""
    h = max(p[3], 1.); return np.array([(z[0] - p[0]) / h, (z[1] - p[1]) / h, math.log(max(z[2], 1.) / max(p[2], 1.)), math.log(max(z[3], 1.) / h)])


# ----------------------------------------------------------------------------------------------------------------- numpy GRU (tracker side)
class GRUNumpy:
    def __init__(self, w):
        self.w = {k: np.asarray(v, np.float64) for k, v in w.items()}; self.nh = self.w['w_hh'].shape[1]

    @staticmethod
    def from_torch(model):
        sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        return GRUNumpy(dict(w_ih=sd['cell.weight_ih'], w_hh=sd['cell.weight_hh'], b_ih=sd['cell.bias_ih'], b_hh=sd['cell.bias_hh'], w_o=sd['head.weight'], b_o=sd['head.bias']))

    @staticmethod
    def load(path): d = np.load(path); return GRUNumpy({k: d[k] for k in d.files})
    def save(self, path): np.savez(path, **self.w)
    def init_state(self, B): return np.zeros((B, self.nh))

    def step(self, x, h):
        w = self.w; n = self.nh; gi = x @ w['w_ih'].T + w['b_ih']; gh = h @ w['w_hh'].T + w['b_hh']; sig = lambda a: 1. / (1. + np.exp(-a))
        r = sig(gi[:, :n] + gh[:, :n]); z = sig(gi[:, n:2 * n] + gh[:, n:2 * n]); c = np.tanh(gi[:, 2 * n:] + r * gh[:, 2 * n:]); h = (1. - z) * c + z * h
        return h @ w['w_o'].T + w['b_o'], h


def rollout_np(g, B, steps=STEPS, base='kf_gmc'):
    """Frame-by-frame numpy reference of rollout() for the GRU arm: B = batch dict of m4_common.build (float arrays). Returns [N, steps, 4]."""
    N = len(B['st']); h = g.init_state(N)
    for i in range(B['hx'].shape[1]):
        _, hn = g.step(B['hx'][:, i].astype(np.float64), h); m = B['hm'][:, i:i + 1] > 0; h = np.where(m, hn, h)
    st = (B['st_ng'] if base == 'kf' else B['st']).astype(np.float64); c, wh, v, vwh = st[:, :2].copy(), st[:, 2:4].copy(), st[:, 4:6].copy(), st[:, 6:8].copy()
    m1 = st[:, :4].copy(); m2 = B['m2'].astype(np.float64); inn = B['inn'].astype(np.float64); out = np.zeros((N, steps, 4)); one = np.ones((N, 1))
    for j in range(1, steps + 1):
        A = B['aff'][:, j - 1].astype(np.float64).reshape(N, 2, 3); R = A[:, :, :2]; tr = A[:, :, 2]; mv = lambda a: np.einsum('nij,nj->ni', R, a)
        cam = mv(c) + tr - c
        if base == 'kf': cp = c + v; whp = np.maximum(wh + vwh if j == 1 else wh, 1.0)
        else: cp = mv(c + v) + tr; v = mv(v); whp = np.maximum(mv(wh + vwh if j == 1 else wh), 1.0)
        x = feat(np, v, whp, inn if j == 1 else 0 * inn, one * (j == 1), cam, m1, m2, one * (j - 1) / 60.)
        o, h = g.step(x, h)
        c = cp + MAX_C * whp[:, 1:2] * np.tanh(o[:, :2]); wh = whp * np.exp(MAX_S * np.tanh(o[:, 2:])); m2 = m1; m1 = np.concatenate([c, wh], 1); out[:, j - 1] = m1
    return out


# ----------------------------------------------------------------------------------------------------------------- torch arms
if torch is not None:
    class GRUArm(nn.Module):
        def __init__(self, nf=NF, hidden=40):
            super().__init__(); self.cell = nn.GRUCell(nf, hidden); self.head = nn.Linear(hidden, 4); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias); self.nh = hidden
        def init_state(self, B, dev): return (torch.zeros(B, self.nh, device=dev),)
        def step(self, x, st): h = self.cell(x, st[0]); return self.head(h), (h,)

    class RMSNorm(nn.Module):
        def __init__(self, d): super().__init__(); self.w = nn.Parameter(torch.ones(d))
        def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.w

    class MambaArm(nn.Module):
        """One Mamba block written as a one-step recurrence (state = last d_conv - 1 inputs of the depthwise causal convolution + the SSM state [d_inner, d_state])."""
        def __init__(self, nf=NF, d=32, expand=2, d_state=8, d_conv=4):
            super().__init__(); di = d * expand; r = math.ceil(d / 16); self.di, self.ds, self.dc, self.r = di, d_state, d_conv, r
            self.emb = nn.Linear(nf, d); self.norm = RMSNorm(d); self.in_proj = nn.Linear(d, 2 * di, bias=False)
            self.conv_w = nn.Parameter(torch.empty(di, d_conv).uniform_(-1, 1) / math.sqrt(d_conv)); self.conv_b = nn.Parameter(torch.zeros(di))
            self.x_proj = nn.Linear(di, r + 2 * d_state, bias=False); self.dt_proj = nn.Linear(r, di)
            nn.init.uniform_(self.dt_proj.weight, -r ** -0.5, r ** -0.5)
            dt = torch.exp(torch.rand(di) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            with torch.no_grad(): self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus (Mamba init)
            self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(di, 1)); self.D = nn.Parameter(torch.ones(di))
            self.out_proj = nn.Linear(di, d, bias=False); self.norm_f = RMSNorm(d); self.head = nn.Linear(d, 4); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        def init_state(self, B, dev): return (torch.zeros(B, self.di, self.dc - 1, device=dev), torch.zeros(B, self.di, self.ds, device=dev))
        def step(self, x, st):
            buf, h = st; u = self.emb(x); xi, z = self.in_proj(self.norm(u)).chunk(2, -1)
            win = torch.cat([buf, xi[:, :, None]], 2); xc = TF.silu((win * self.conv_w).sum(-1) + self.conv_b)
            dt, Bm, Cm = torch.split(self.x_proj(xc), [self.r, self.ds, self.ds], -1); dt = TF.softplus(self.dt_proj(dt))[:, :, None]
            h = torch.exp(dt * -torch.exp(self.A_log)) * h + dt * Bm[:, None, :] * xc[:, :, None]      # selective state update (zero-order hold on A, Euler on B)
            y = ((h * Cm[:, None, :]).sum(-1) + self.D * xc) * TF.silu(z); u = u + self.out_proj(y)
            return self.head(self.norm_f(u)), (win[:, :, 1:], h)

    def make_arm(arm): return GRUArm() if arm == 'gru' else MambaArm()

    def rollout(model, T, steps=STEPS, base='kf_gmc'):
        """T = dict of torch tensors (hx, hm, st, m2, inn, aff). Warm-up over the last <= 20 observed frames, then `steps` lost frames with the own prediction fed back.
        Returns the corrected boxes [B, steps, 4] (cx, cy, w, h)."""
        N = T['st'].shape[0]; dev = T['st'].device; s = tuple(x.to(T['st'].dtype) for x in model.init_state(N, dev))
        for i in range(T['hx'].shape[1]):
            _, sn = model.step(T['hx'][:, i], s); m = T['hm'][:, i] > 0
            s = tuple(torch.where(m.view(-1, *([1] * (a.dim() - 1))), a, b) for a, b in zip(sn, s))
        st = T['st_ng'] if base == 'kf' else T['st']; c, wh, v, vwh = st[:, :2], st[:, 2:4], st[:, 4:6], st[:, 6:8]; m1 = st[:, :4]; m2 = T['m2']; inn = T['inn']; out = []; one = torch.ones(N, 1, device=dev, dtype=st.dtype)
        for j in range(1, steps + 1):
            A = T['aff'][:, j - 1].reshape(N, 2, 3); R = A[:, :, :2]; tr = A[:, :, 2]; mv = lambda a: torch.einsum('nij,nj->ni', R, a)
            cam = mv(c) + tr - c
            if base == 'kf': cp = c + v; whp = (wh + vwh if j == 1 else wh).clamp(min=1.0)
            else: cp = mv(c + v) + tr; v = mv(v); whp = mv(wh + vwh if j == 1 else wh).clamp(min=1.0)
            x = feat(torch, v, whp, inn if j == 1 else 0 * inn, one * float(j == 1), cam, m1, m2, one * (j - 1) / 60.)
            o, s = model.step(x, s)
            c = cp + MAX_C * whp[:, 1:2] * torch.tanh(o[:, :2]); wh = whp * torch.exp(MAX_S * torch.tanh(o[:, 2:])); m2 = m1; m1 = torch.cat([c, wh], 1); out.append(m1)
        return torch.stack(out, 1)

    KEYS = ('hx', 'hm', 'st', 'm2', 'inn', 'aff', 'st_ng')

    def to_t(B, idx, dev): return {k: torch.as_tensor(np.asarray(B[k][idx], np.float32), device=dev) for k in KEYS}

    @torch.no_grad()
    def predict_arm(model, B, dev, bs=1024, steps=STEPS, base='kf_gmc'):
        model.eval(); N = len(B['st']); out = np.zeros((N, steps, 4), np.float32)
        for a in range(0, N, bs): out[a:a + bs] = rollout(model, to_t(B, slice(a, a + bs), dev), steps, base).cpu().numpy()
        return out

    def load_ckpt(path, dev='cpu'):
        ck = torch.load(path, map_location=dev); m = make_arm(ck['arm']).to(dev); m.load_state_dict(ck['model']); m.eval(); return m, ck


# ----------------------------------------------------------------------------------------------------------------- self test
def selftest():
    import m4_common as C
    rng = np.random.default_rng(0); kf = KalmanXYWH(); n = 90; N = 6
    fn = np.arange(n); aff = np.tile(np.eye(2, 3), (n, 1, 1))
    for k in range(n):
        th = rng.normal(0, 0.004); s = 1 + rng.normal(0, 0.004); aff[k] = [[s * math.cos(th), -s * math.sin(th), rng.normal(0, 3)], [s * math.sin(th), s * math.cos(th), rng.normal(0, 2)]]
    box = np.zeros((n, 4)); box[0] = [300, 240, 40, 110]
    for k in range(1, n): box[k] = box[k - 1] + [rng.normal(1.5, 1), rng.normal(0, .6), rng.normal(0, .4), rng.normal(.1, .5)]
    B = C.build([dict(name='t', si=0, fn=fn, ctx=np.zeros(n, np.uint8), aff=aff, tracks={1: box})]); i = 3; t = int(B['t'][i])
    # 1. closed-form gap baseline == explicit Kalman class stepping (predict, lost flag from the 2nd lost frame, warp)
    mean, cov = kf.initiate(box[0])
    for k in range(1, t + 1): mean, cov = kf.predict(mean, cov); mean, cov = kf.gmc(mean, cov, aff[k]); mean, cov = kf.update(mean, cov, box[k])
    ref = []
    for j in range(1, 31): mean, cov = kf.predict(mean, cov, lost=j > 1); mean, cov = kf.gmc(mean, cov, aff[t + j]); ref.append(mean[:4].copy())
    base = C.baselines(B); d1 = float(np.abs(base['kf_gmc'][i, :30] - np.array(ref)).max()); print('kf_gmc closed form vs class stepping  max |diff| px = %.2e' % d1); assert d1 < 1e-6
    if torch is None: print('torch missing: arm tests skipped'); return
    torch.manual_seed(0)
    for arm in ('gru', 'mamba'):
        m = make_arm(arm).double(); npar = sum(p.numel() for p in m.parameters())
        T = {k: torch.as_tensor(np.asarray(B[k], np.float64)) for k in KEYS}
        with torch.no_grad(): z = rollout(m, T).numpy()
        d2 = float(np.abs(z - base['kf_gmc']).max()); print('%-5s parameters %d | zero head vs kf_gmc  max |diff| px = %.2e' % (arm, npar, d2)); assert d2 < 1e-6
        nn.init.normal_(m.head.weight, 0, 0.3); nn.init.normal_(m.head.bias, 0, 0.1)
        with torch.no_grad(): y = rollout(m, T).numpy()
        assert np.isfinite(y).all(); print('      random head: mean shift from kf_gmc at step 60 = %.1f px (finite, bounded)' % float(np.abs(y[:, -1, :2] - base['kf_gmc'][:, -1, :2]).mean()))
        if arm == 'gru':
            d3 = float(np.abs(rollout_np(GRUNumpy.from_torch(m), B) - y).max()); print('      torch vs numpy rollout (60 lost frames, float64)  max |diff| px = %.2e' % d3); assert d3 < 1e-6
            m32 = make_arm('gru'); nn.init.normal_(m32.head.weight, 0, 0.3)
            with torch.no_grad(): y32 = rollout(m32, {k: torch.as_tensor(np.asarray(B[k], np.float32)) for k in KEYS}).numpy()
            d4 = float(np.abs(rollout_np(GRUNumpy.from_torch(m32), B) - y32).max()); print('      torch float32 vs numpy float64 rollout  max |diff| px = %.2e' % d4); assert d4 < 0.05
    print('M4_MODEL_SELFTEST_OK')


if __name__ == '__main__':
    selftest()
