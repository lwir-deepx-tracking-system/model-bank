"""pos_lib.py - position-memory data (server feature builder on our detector outputs), augmentation, model, loss and score (pos_refine_20261115).
Anchor = last seen frame t of a run with >= KP_MINOBS seen features; inputs hx [20,68] (the run's seen-frame features up to t), xl [15,68] (lost steps),
base [15,4] (plain Kalman path, server warps), h0; targets [15,4] (cx cy w h, nan = no gt).
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prcommon as P

DETS = ('near', 'far')


def load_seq(root, r):
    a = dict(np.load(Path(root) / 'data' / 'assoc' / ('%s.npz' % r['name'])))
    d = {k: P.load_det(Path(root) / 'data' / 'det' / ('%s_%s.npz' % (k, r['folder']))) for k in DETS}
    W, _ = P.load_gmc(Path(root) / 'data' / 'gmc' / ('%s.npz' % r['folder']))
    return dict(r=r, a=a, d=d, W=W, n=d['near']['n'])


def tracks_of(S, k, target='det'):
    """per gt person of detector k: dict gid -> dict(det={f: (z cxcywh, kp)}, tgt={f: cxcywh}, tgt2={f: cxcywh h2h}, hg={f: gt height}, f1=last gt frame)."""
    a, d = S['a'], S['d'][k]; dg = a['dg_' + k]; T = {}
    tg = P.xyxy2cxcywh(a['tgt_det_' + k] if target == 'det' else a['tgt_h2h']); t2 = P.xyxy2cxcywh(a['tgt_h2h']); hg = a['gbox'][:, 3] - a['gbox'][:, 1]
    for row, (f, g) in enumerate(zip(a['gfr'], a['gid'])):
        e = T.setdefault(int(g), dict(det={}, tgt={}, tgt2={}, hg={}, f1=0))
        e['tgt'][int(f)] = tg[row]; e['tgt2'][int(f)] = t2[row]; e['hg'][int(f)] = float(hg[row]); e['f1'] = max(e['f1'], int(f))
    D = P.xyxy2cxcywh(d['box'])
    for i in np.flatnonzero(dg >= 0):
        g = int(a['gid'][dg[i]]); T[g]['det'][int(d['fi'][i])] = (D[i], d['kpt'][i])
    return {g: e for g, e in T.items() if e['det']}


def augment(T, W, n, rng, flip=False, reverse=False, half=False):
    """whole-sequence motion augmentation (all people and the warps together). -> (T', W', n')"""
    def fmap(f):
        if reverse: f = n - 1 - f
        if half:
            if f % 2: return None
            f //= 2
        return f

    def box(b):
        b = np.array(b, np.float64)
        if flip: b[0] = P.XW - b[0]
        return b

    def kpf(kp):
        kp = kp.copy()
        if flip: kp[:, 0] = P.XW - kp[:, 0]; kp = kp[P.COCO_FLIP]
        return kp
    W2 = W
    if flip: W2 = P.flip_warps(W2)
    if reverse: W2 = P.reverse_warps(W2)
    if half: W2 = P.half_warps(W2)
    n2 = len(W2); out = {}
    for g, e in T.items():
        o = dict(det={}, tgt={}, tgt2={}, hg={}, f1=0)
        for f, (z, kp) in e['det'].items():
            f2 = fmap(f)
            if f2 is not None: o['det'][f2] = (box(z), kpf(kp))
        for key in ('tgt', 'tgt2', 'hg'):
            for f, v in e[key].items():
                f2 = fmap(f)
                if f2 is not None: o[key][f2] = box(v) if key != 'hg' else v
        o['f1'] = max(o['tgt']) if o['tgt'] else 0
        if o['det']: out[g] = o
    return out, W2, n2


def vanish(det, rng, p=0.6):
    """m5 v2 synthetic disappearance on one person's detections: frames e..e+g-1 removed, an occluder sweeps over the body in the K frames before e
    (keypoints hidden step by step, more box noise). -> (det', e) ; e = -1 when not applied."""
    if rng.uniform() > p: return det, -1
    fs = sorted(det); cand = []
    run = 0
    for i, f in enumerate(fs):
        run = run + 1 if i and fs[i - 1] == f - 1 else 1
        if i + 1 < len(fs) and fs[i + 1] == f + 1 and run >= P.KP_MINOBS + 1: cand.append(f + 1)
    if not cand: return det, -1
    e = int(rng.choice(cand)); g = int(rng.integers(3, P.KP_STEPS + 1)); K = int(rng.integers(0, 7)); side = int(rng.integers(4)); ax = 0 if side < 2 else 1
    det = dict(det)
    for f in range(e, e + g): det.pop(f, None)
    for i, f in enumerate(range(e - K, e)):
        if f not in det: continue
        z, kp = det[f]; z = z.copy(); kp = kp.copy(); fr = 0.2 + 0.7 * (i + 1) / max(K, 1); v = kp[:, 2] >= 0.5
        if v.sum() >= 2:
            c = kp[v, ax]; lo, hi = c.min(), c.max(); cut = lo + fr * (hi - lo) if side % 2 == 0 else hi - fr * (hi - lo)
            hide = (kp[:, ax] < cut) if side % 2 == 0 else (kp[:, ax] > cut); kp[hide & v, 2] = 0.
        z[:2] += rng.normal(0, 0.05 * fr, 2) * z[3]; z[2:] *= np.exp(rng.normal(0, 0.08 * fr, 2)); det[f] = (z, kp)
    return det, e


def anchors(S, k, T, W, n, rng=None, stride=1, offset=0, syn_only=False, vanish_p=0.6, real_from=None):
    """-> dict of lists. syn_only: keep only the anchors right before a synthetic vanish (fixed-seed validation set 'syn')."""
    L = defaultdict(list)
    for g, e in T.items():
        det = e['det']; e_syn = -1
        if rng is not None and vanish_p > 0: det, e_syn = vanish(det, rng, vanish_p)
        if not det: continue
        f0 = min(det); ser = P.pos_series(f0, max(e['f1'], max(det)), det, W)
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
                L['hx'].append(hx.astype(np.float16)); L['hm'].append(hm)
                L['h0'].append(max(ser[t][2][3], 1.)); L['tgt'].append(tg.astype(np.float32)); L['tgt2'].append(t2.astype(np.float32))
                L['real'].append((t + 1) not in orig_seen); L['syn'].append(t == e_syn - 1); L['hg'].append(e['hg'].get(t, np.nan)); L['gid'].append(g); L['t'].append(t)
        if sel:   # the plain Kalman path and the lost-step inputs of all these anchors at once (P.lost_path_batch = P.lost_path vectorised)
            ta = np.array(sel); base, xl = P.lost_path_batch(np.array([ser[t][1] for t in sel]), np.array([ser[t][2] for t in sel]), W, ta, n)
            L['base'].extend(base.astype(np.float32)); L['xl'].extend(xl.astype(np.float16))
    return L


def stack(Ls, extra):
    out = defaultdict(list)
    for L, ex in zip(Ls, extra):
        m = len(L['h0'])
        for k, v in L.items(): out[k].extend(v)
        for k, v in ex.items(): out[k].extend([v] * m)
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------------------------ torch model / loss / score
def make_net(arm, hidden=64):
    import torch
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / 'ref_code')]   # m4_model.py next to this file (repository copy) or in ref_code/
    import m4_model as M

    class PosNet(torch.nn.Module):
        """m5_kpt_motion_v2.Net with the lost-step inputs precomputed (server-exact, with camera warps): warm-up over hx, then 15 steps over xl.
        Same parameter names (cell.cell.* / cell.emb.* ...) so the server's PosKpt reads the exported file."""
        def __init__(self):
            super().__init__()
            self.cell = M.GRUArm(nf=P.KP_NF, hidden=hidden) if arm.startswith('gru') else M.MambaArm(nf=P.KP_NF, d=max(16, hidden * 3 // 4))

        def forward(self, hx, hm, xl, base, h0):
            N = hx.shape[0]; s = tuple(x.to(hx.dtype) for x in self.cell.init_state(N, hx.device))
            for i in range(hx.shape[1]):
                _, sn = self.cell.step(hx[:, i], s); m = hm[:, i] > 0
                s = tuple(torch.where(m.view(-1, *([1] * (a.dim() - 1))), a, b) for a, b in zip(sn, s))
            out = []; h0 = h0[:, None]
            for j in range(xl.shape[1]):
                o, s = self.cell.step(xl[:, j], s); b = base[:, j]
                out.append(torch.cat([b[:, :2] + P.KP_C * h0 * torch.tanh(o[:, :2]), b[:, 2:] * torch.exp(P.KP_S * torch.tanh(o[:, 2:]))], 1))
            return torch.stack(out, 1)
    return PosNet()


def load_npz_into(net, path):
    import torch
    z = np.load(path); sd = net.state_dict(); new = {}
    for k in sd:
        assert k in z.files, (k, z.files); assert tuple(z[k].shape) == tuple(sd[k].shape), (k, z[k].shape, sd[k].shape)
        new[k] = torch.as_tensor(z[k], dtype=sd[k].dtype)
    net.load_state_dict(new)


def loss_fn(q, g, ok):
    import torch, torch.nn.functional as TF
    gs = torch.where(ok[..., None], g, torch.ones_like(g))
    off = torch.cat([(q[..., :2] - gs[..., :2]) / gs[..., 3:4], torch.log(q[..., 2:].clamp(min=1.) / gs[..., 2:].clamp(min=1.))], -1)
    px1, py1, px2, py2 = q[..., 0] - q[..., 2] / 2, q[..., 1] - q[..., 3] / 2, q[..., 0] + q[..., 2] / 2, q[..., 1] + q[..., 3] / 2
    gx1, gy1, gx2, gy2 = gs[..., 0] - gs[..., 2] / 2, gs[..., 1] - gs[..., 3] / 2, gs[..., 0] + gs[..., 2] / 2, gs[..., 1] + gs[..., 3] / 2
    inter = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(min=0) * (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(min=0)
    union = q[..., 2] * q[..., 3] + gs[..., 2] * gs[..., 3] - inter + 1e-7; hull = (torch.max(px2, gx2) - torch.min(px1, gx1)) * (torch.max(py2, gy2) - torch.min(py1, gy1)) + 1e-7
    l = TF.smooth_l1_loss(off, torch.zeros_like(off), beta=0.1, reduction='none').sum(-1) + (1 - (inter / union - (hull - union) / hull))
    okf = ok.float(); return (l * okf).sum() / okf.sum().clamp(min=1)


def ce_matrix(pred, tgt):
    """[N,15] min(centre error / target height, 2), nan where no target."""
    return np.minimum(np.hypot(pred[..., 0] - tgt[..., 0], pred[..., 1] - tgt[..., 1]) / tgt[..., 3], 2.0)


def score(ce, sel=None):
    m = np.ones(len(ce), bool) if sel is None else sel
    r = {}
    for h in (1, 5, 10, 15):
        v = ce[m, h - 1]; v = v[np.isfinite(v)]; r['h%d' % h] = dict(ce=float(v.mean()) if len(v) else float('nan'), n=int(len(v)))
    r['score'] = float(np.nanmean([r['h%d' % h]['ce'] for h in (5, 10, 15)])); return r


def predict(net, B, dev, bs=4096, zero_kpt12=False):
    import torch
    net.eval(); out = []
    with torch.no_grad():
        for a in range(0, len(B['h0']), bs):
            hx = torch.as_tensor(B['hx'][a:a + bs].astype(np.float32), device=dev)
            if zero_kpt12: hx[..., 19:25] = 0.   # server: 'posetrack' files get keypoints 1 and 2 (PoseTrack head points) as zeros
            out.append(net(hx, torch.as_tensor(B['hm'][a:a + bs], device=dev), torch.as_tensor(B['xl'][a:a + bs].astype(np.float32), device=dev),
                           torch.as_tensor(B['base'][a:a + bs], device=dev), torch.as_tensor(B['h0'][a:a + bs].astype(np.float32), device=dev)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, P.KP_STEPS, 4), np.float32)
