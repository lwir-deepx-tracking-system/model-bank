"""m5_kpt_motion: GRU / Mamba position memory WITH body keypoints, trained and scored on PoseTrack18 (video, track ids, 17 keypoints, real disappearances).

User 2026-11-14: "관절 있는 데이터로 해 그럼 학습을" (after: GRU/Mamba should remember where a person vanished, which the Kalman filter cannot).
Only the annotation files are used (no pictures): data/posetrack18/posetrack18_{train,val}.json (OpenMMLab merge of the official files).

Task: a person was seen for >= 5 frames up to frame t; predict the person box for t+1 .. t+15 without further detections.
  - simulated loss: the person IS labelled at t+1 (we hide it), most anchors
  - real loss:      the person is NOT labelled at t+1 (left the picture / fully hidden) but comes back later inside the labelled run -> the case the user means
Prediction = plain Kalman (Ultralytics KalmanFilterXYWH constants, the tracker's filter) + learned correction (head starts at zero = exactly Kalman).
Inputs per seen frame: Kalman velocity, innovation, box change, box size, distance to the 4 picture edges (all / box height) and, in the keypoint arms,
17 keypoints relative to the box centre / box height + visible flag. Lost frames: same vector with keypoints / innovation zero + lost counter.
Arms: gru_kpt, gru_box, mamba_kpt, mamba_box (box = same model without keypoints, to see what the keypoints add).
Selection on val half A (epoch), report on val half B (never used for choices). Score = mean over horizons 5, 10, 15 of mean min(centre error / true height, 2).

Usage (Vast): python -I m5_kpt_motion.py <arm> --data-dir data/posetrack18 --motion-dir /workspace/motion --out runs/m5_kpt [--epochs 30] [--seed 0]
"""
import argparse, json, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn.functional as TF

HIST = 20; MINOBS = 5; STEPS = 15; HORIZONS = (1, 5, 10, 15); SCORE_H = (5, 10, 15); MIN_H = 32
MAX_C = 1.0; MAX_S = math.log(1.5); CLIP = 5.0
NB = 16; NK = 51; NF_KPT = NB + NK + 1; NF_BOX = NB + 1


def load(path):
    """-> list of runs: dict(W, H, n, tracks={tid: (box [n,4] nan=absent, kp [n,17,3])}) over maximal runs of consecutive labelled frames."""
    d = json.load(open(path)); im = {i['id']: i for i in d['images']}; lab = defaultdict(list)
    for i in d['images']:
        if i['is_labeled']: lab[i['vid_id']].append(i)
    ann = defaultdict(list)
    for a in d['annotations']: ann[a['image_id']].append(a)
    runs = []
    for vid, ims in lab.items():
        ims = sorted(ims, key=lambda i: i['frame_id']); fr = [i['frame_id'] % 10000 for i in ims]; cuts = [0] + [k for k in range(1, len(fr)) if fr[k] != fr[k - 1] + 1] + [len(fr)]
        for a0, b0 in zip(cuts[:-1], cuts[1:]):
            seg = ims[a0:b0]; n = len(seg)
            if n < MINOBS + 2: continue
            tr = {}
            for k, i in enumerate(seg):
                for a in ann[i['id']]:
                    bb = a.get('bbox')
                    if not bb or bb[3] < MIN_H or bb[2] < 6: continue
                    t = tr.setdefault(a['track_id'], (np.full((n, 4), np.nan), np.zeros((n, 17, 3))))
                    t[0][k] = [bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]]; kp = np.array(a['keypoints'], float).reshape(17, 3); kp[:, 2] = kp[:, 2] > 0; t[1][k] = kp
            runs.append(dict(vid=vid, W=float(seg[0].get('width') or 1280), H=float(seg[0].get('height') or 720), n=n, tracks=tr))
    return runs


class KF:
    """Ultralytics KalmanFilterXYWH (weights 1/20, 1/160, dt 1)."""
    SP = 1. / 20; SV = 1. / 160
    def __init__(self): self.F = np.eye(8); self.F[:4, 4:] = np.eye(4); self.Hm = np.eye(4, 8)
    def initiate(self, z):
        w, h = z[2], z[3]; s = [2 * self.SP * w, 2 * self.SP * h, 2 * self.SP * w, 2 * self.SP * h, 10 * self.SV * w, 10 * self.SV * h, 10 * self.SV * w, 10 * self.SV * h]
        return np.r_[z, np.zeros(4)], np.diag(np.square(s))
    def predict(self, m, c, lost=False):
        if lost: m = m.copy(); m[6:] = 0
        w, h = m[2], m[3]; s = [self.SP * w, self.SP * h, self.SP * w, self.SP * h, self.SV * w, self.SV * h, self.SV * w, self.SV * h]
        return self.F @ m, self.F @ c @ self.F.T + np.diag(np.square(s))
    def update(self, m, c, z):
        w, h = m[2], m[3]; s = [self.SP * w, self.SP * h, self.SP * w, self.SP * h]; pm = self.Hm @ m; pc = self.Hm @ c @ self.Hm.T + np.diag(np.square(s))
        K = np.linalg.solve(pc, (c @ self.Hm.T).T).T; return m + (z - pm) @ K.T, c - K @ pc @ K.T


def boxfeat(v, b, prev, inn, W, H):
    h = max(b[3], 1.); w = max(b[2], 1.)
    edge = np.array([b[0] - w / 2, W - (b[0] + w / 2), b[1] - h / 2, H - (b[1] + h / 2)]) / h
    chg = np.r_[(b[:2] - prev[:2]) / h, np.log(np.maximum(b[2:], 1.) / np.maximum(prev[2:], 1.))]
    return np.clip(np.r_[v / h, inn, chg, math.log(h / 100.), math.log(w / h), edge], -CLIP, CLIP)


FLIP = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # PoseTrack18 order: nose, head_bottom, head_top, ears, shoulders, elbows, wrists, hips, knees, ankles


def aug_run(R, rng):
    """Motion augmentation of a whole run (all tracks together, targets move too): half frame rate (faster motion), time reversal, left-right flip, camera pan."""
    n, W = R['n'], R['W']; tr = {k: (v[0].copy(), v[1].copy()) for k, v in R['tracks'].items()}
    sl = slice(None, None, 2) if (rng.uniform() < 0.3 and n >= 2 * (MINOBS + 2)) else slice(None)
    if rng.uniform() < 0.5: sl = slice(sl.start, sl.stop, -(sl.step or 1)) if sl.step else slice(None, None, -1)
    tr = {k: (b[sl], p[sl]) for k, (b, p) in tr.items()}; n = len(next(iter(tr.values()))[0]) if tr else n
    flip = rng.uniform() < 0.5; pan = None
    if rng.uniform() < 0.5: pan = np.cumsum(np.c_[rng.normal(rng.uniform(-4, 4), 0.5, n), rng.normal(rng.uniform(-1.5, 1.5), 0.3, n)], 0)
    for k, (b, p) in tr.items():
        v = p[:, :, 2] > 0
        if flip: b[:, 0] = W - b[:, 0]; p[:, :, 0] = np.where(v, W - p[:, :, 0], 0); p[:] = p[:, FLIP]; v = p[:, :, 2] > 0
        if pan is not None: b[:, :2] += pan; p[:, :, :2] += pan[:, None] * v[..., None]
    return dict(R, n=n, tracks=tr)


def vanish(obs, box, kz, rng, p=0.6):
    """Synthetic disappearance: input frames e .. e+g-1 removed (targets stay); in the K frames before e an occluder sweeps over the body from one side,
    hiding keypoints step by step (20 % -> 90 % of the body), with more box noise (the body-box detector is less sure while the person is half hidden). Returns obs_in, e."""
    n = len(obs); obs_in = obs.copy()
    if rng.uniform() > p or n < MINOBS + 4: return obs_in, -1, 0
    e = int(rng.integers(MINOBS + 1, n - 1))
    if not obs[e - 1] or not obs[e]: return obs_in, -1, 0
    g = int(rng.integers(3, STEPS + 1)); obs_in[e:e + g] = False; K = int(rng.integers(0, 7)); side = int(rng.integers(4)); ax = 0 if side < 2 else 1
    for i, j in enumerate(range(max(0, e - K), e)):
        if not obs[j]: continue
        f = 0.2 + 0.7 * (i + 1) / max(K, 1); v = kz[j, :, 2] > 0
        if v.sum() < 2: continue
        c = kz[j, v, ax]; lo, hi = c.min(), c.max(); cut = lo + f * (hi - lo) if side % 2 == 0 else hi - f * (hi - lo)
        hide = (kz[j, :, ax] < cut) if side % 2 == 0 else (kz[j, :, ax] > cut); kz[j, hide & v] = 0
        box[j, :2] += rng.normal(0, 0.05 * f, 2) * box[j, 3]; box[j, 2:] *= np.exp(rng.normal(0, 0.08 * f, 2))
    return obs_in, e, K


def build(runs, rng=None, sig=(0.0, 0.0), aug=False):
    """Anchors -> arrays. Input boxes / keypoints get detector-like jitter when rng is given (one strength per track, 0..sig); targets stay clean.
    aug: motion augmentation (aug_run) and synthetic disappearance with a sweeping occluder (vanish); L['syn'] marks the anchor just before such a disappearance."""
    kf = KF(); L = defaultdict(list)
    for R in runs:
        if aug: R = aug_run(R, rng)
        n, W, H = R['n'], R['W'], R['H']
        for tid, (box, kp) in R['tracks'].items():
            obs = ~np.isnan(box[:, 0]); z = box.copy(); kz = kp.copy()
            if rng is not None:
                u = rng.uniform(); z[:, :2] += rng.normal(0, 1, (n, 2)) * u * sig[0] * box[:, 3:4]; z[:, 2:] *= np.exp(rng.normal(0, 1, (n, 2)) * u * sig[1])
                kz[:, :, :2] += rng.normal(0, 1, (n, 17, 2)) * u * sig[0] * np.nan_to_num(box[:, None, 3:4])
            e_syn = -1
            if aug: obs, e_syn, _ = vanish(obs, z, kz, rng)
            k = 0
            while k < n:
                if not obs[k]: k += 1; continue
                a = k
                while k + 1 < n and obs[k + 1]: k += 1
                b = k; k += 1
                if b - a + 1 < MINOBS: continue
                m, c = kf.initiate(z[a]); feats = [None] * n; post = {a: m.copy()}; feats[a] = None
                for j in range(a + 1, b + 1):
                    m, c = kf.predict(m, c); pm = m.copy(); h = max(pm[3], 1.)
                    inn = np.r_[(z[j, :2] - pm[:2]) / h, np.log(max(z[j, 2], 1.) / max(pm[2], 1.)), np.log(max(z[j, 3], 1.) / h)]
                    m, c = kf.update(m, c, z[j]); post[j] = m.copy()
                    f = boxfeat(m[4:6], z[j], z[j - 1], inn, W, H); kq = kz[j]; hk = max(z[j, 3], 1.)
                    kfe = np.c_[(kq[:, 0] - z[j, 0]) / hk * kq[:, 2], (kq[:, 1] - z[j, 1]) / hk * kq[:, 2], kq[:, 2]].ravel()
                    feats[j] = np.r_[f, np.clip(kfe, -CLIP, CLIP), 0.]
                for t in range(a + MINOBS - 1, b + 1):
                    if t + 1 >= n: continue
                    gt = np.full((STEPS, 4), np.nan); e = min(n, t + 1 + STEPS); gt[:e - t - 1] = box[t + 1:e]; ok = ~np.isnan(gt[:, 0])
                    if not ok.any(): continue
                    hx = np.zeros((HIST, NF_KPT)); hm = np.zeros(HIST); k0 = max(a + 1, t - HIST + 1); nh = t - k0 + 1
                    hx[HIST - nh:] = np.stack(feats[k0:t + 1]); hm[HIST - nh:] = 1
                    # plain Kalman over the gap (lost from the 2nd step: vw = vh = 0), the baseline and the base of the correction
                    mm, cc = post[t].copy(), None; cc = np.eye(8); base = np.zeros((STEPS, 4)); mmv = mm.copy()
                    for s in range(STEPS):
                        if s > 0: mmv[6:] = 0
                        mmv = kf.F @ mmv; base[s] = mmv[:4]
                    L['hx'].append(hx); L['hm'].append(hm); L['base'].append(base); L['gt'].append(np.nan_to_num(gt, nan=1.)); L['ok'].append(ok)
                    L['last'].append(z[t]); L['prev'].append(z[t - 1]); L['v'].append(post[t][4:6]); L['WH'].append([W, H])
                    L['real'].append(bool(t + 1 < n and not obs[t + 1])); L['syn'].append(t == e_syn - 1); L['vid'].append(R['vid']); L['tid'].append(tid)
    return {k: (np.array(v, np.float32) if k == 'hx' else np.array(v)) for k, v in L.items()}


class Net(torch.nn.Module):
    """History encoder + gap decoder with one recurrent cell (GRU or Mamba-style block from m4_model). Output = correction of the Kalman box."""
    def __init__(self, arm, nf, hidden=64):
        super().__init__(); import m4_model as M
        self.cell = M.GRUArm(nf=nf, hidden=hidden) if arm.startswith('gru') else M.MambaArm(nf=nf, d=max(16, hidden * 3 // 4)); self.nf = nf
    def forward(self, hx, hm, base, last, prev, v, WH):
        N = hx.shape[0]; dev = hx.device; s = self.cell.init_state(N, dev)
        for i in range(hx.shape[1]):
            _, sn = self.cell.step(hx[:, i], s); msk = hm[:, i] > 0
            s = tuple(torch.where(msk.view(-1, *([1] * (a.dim() - 1))), a, b) for a, b in zip(sn, s))
        h0 = last[:, 3:4].clamp(min=1.); out = []; pb = last
        for j in range(STEPS):
            b = base[:, j]; hb = b[:, 3:4].clamp(min=1.); wb = b[:, 2:3].clamp(min=1.)
            edge = torch.cat([b[:, 0:1] - wb / 2, WH[:, 0:1] - (b[:, 0:1] + wb / 2), b[:, 1:2] - hb / 2, WH[:, 1:2] - (b[:, 1:2] + hb / 2)], 1) / hb
            chg = torch.cat([(b[:, :2] - pb[:, :2]) / hb, torch.log(b[:, 2:].clamp(min=1.) / pb[:, 2:].clamp(min=1.))], 1)
            f = torch.cat([v / hb, torch.zeros(N, 4, device=dev), chg, torch.log(hb / 100.), torch.log(wb / hb), edge], 1)
            x = torch.cat([f, torch.zeros(N, self.nf - NB - 1, device=dev), torch.full((N, 1), (j + 1) / STEPS, device=dev)], 1).clamp(-CLIP, CLIP)
            o, s = self.cell.step(x, s)
            q = torch.cat([b[:, :2] + MAX_C * h0 * torch.tanh(o[:, :2]), b[:, 2:] * torch.exp(MAX_S * torch.tanh(o[:, 2:]))], 1); out.append(q); pb = b
        return torch.stack(out, 1)


def giou(p, g):
    px1, py1, px2, py2 = p[..., 0] - p[..., 2] / 2, p[..., 1] - p[..., 3] / 2, p[..., 0] + p[..., 2] / 2, p[..., 1] + p[..., 3] / 2
    gx1, gy1, gx2, gy2 = g[..., 0] - g[..., 2] / 2, g[..., 1] - g[..., 3] / 2, g[..., 0] + g[..., 2] / 2, g[..., 1] + g[..., 3] / 2
    inter = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(min=0) * (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(min=0)
    union = p[..., 2] * p[..., 3] + g[..., 2] * g[..., 3] - inter + 1e-7; hull = (torch.max(px2, gx2) - torch.min(px1, gx1)) * (torch.max(py2, gy2) - torch.min(py1, gy1)) + 1e-7
    return inter / union - (hull - union) / hull


def loss_fn(q, g, ok):
    off = torch.cat([(q[..., :2] - g[..., :2]) / g[..., 3:4], torch.log(q[..., 2:].clamp(min=1.) / g[..., 2:])], -1)
    l = TF.smooth_l1_loss(off, torch.zeros_like(off), beta=0.1, reduction='none').sum(-1) + (1 - giou(q, g))
    return (l * ok).sum() / ok.sum().clamp(min=1)


def score(pred, B, sel=None):
    """Per horizon mean min(centre error / true h, 2), share IoU >= .5; score = mean over SCORE_H. sel = boolean mask of anchors."""
    r = {}
    for h in HORIZONS:
        m = B['ok'][:, h - 1] & (sel if sel is not None else True); p = pred[m, h - 1]; g = B['gt'][m, h - 1]
        if not m.any(): r['h%d' % h] = dict(ce=float('nan'), n=0); continue
        e = np.minimum(np.hypot(p[:, 0] - g[:, 0], p[:, 1] - g[:, 1]) / g[:, 3], 2)
        r['h%d' % h] = dict(ce=float(e.mean()), n=int(m.sum()))
    r['score'] = float(np.nanmean([r['h%d' % h]['ce'] for h in SCORE_H])); return r


def baselines(B):
    last = np.repeat(B['last'][:, None], STEPS, 1); cv = last.copy(); d = (B['last'][:, :2] - B['prev'][:, :2])
    cv[:, :, :2] += np.arange(1, STEPS + 1)[None, :, None] * d[:, None]
    return dict(kalman=B['base'], last=last, const_vel=cv)


def T(B, idx, dev, nf):
    hx = B['hx'][idx]
    if nf == NF_BOX: hx = np.concatenate([hx[..., :NB], hx[..., -1:]], -1)
    f = lambda k: torch.as_tensor(np.asarray(B[k][idx], np.float32), device=dev)
    return torch.as_tensor(hx, dtype=torch.float32, device=dev), f('hm'), f('base'), f('last'), f('prev'), f('v'), f('WH')


@torch.no_grad()
def predict(net, B, dev, nf, bs=2048):
    net.eval(); out = []
    for a in range(0, len(B['hm']), bs): out.append(net(*T(B, slice(a, a + bs), dev, nf)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('arm', choices=['gru_kpt', 'gru_box', 'mamba_kpt', 'mamba_box'])
    ap.add_argument('--data-dir', default='data/posetrack18'); ap.add_argument('--motion-dir', default='/workspace/motion'); ap.add_argument('--out', default='runs/m5_kpt')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--sig-c', type=float, default=0.03); ap.add_argument('--sig-s', type=float, default=0.05)
    ap.add_argument('--aug', type=int, default=0, help='1 = motion augmentation + synthetic disappearance with a sweeping occluder in training')
    ap.add_argument('--hidden', type=int, default=64); ap.add_argument('--tag', default=''); a = ap.parse_args()
    sys.path.insert(0, a.motion_dir); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dd = Path(a.data_dir); nf = NF_KPT if a.arm.endswith('kpt') else NF_BOX
    cache = out / 'anchors_val.npz'
    tr_runs = load(dd / 'posetrack18_train.json'); va_runs = load(dd / 'posetrack18_val.json')
    vids = sorted({R['vid'] for R in va_runs}); half_a = set(vids[0::2])
    V = build(va_runs); isA = np.array([v in half_a for v in V['vid']]); isR = V['real']
    S = build(va_runs, np.random.default_rng(12345), (0.03, 0.05), aug=True); sA = np.array([v in half_a for v in S['vid']]) & S['syn']; sB = np.array([v not in half_a for v in S['vid']]) & S['syn']
    base = baselines(V); baseS = baselines(S); bs_ = {k: dict(A=score(p, V, isA), B=score(p, V, ~isA), B_real=score(p, V, ~isA & isR), A_real=score(p, V, isA & isR), A_syn=score(baseS[k], S, sA), B_syn=score(baseS[k], S, sB)) for k, p in base.items()}
    print('synthetic disappearance anchors (val, fixed seed): A %d B %d' % (sA.sum(), sB.sum()), flush=True); stem = '%s_s%d%s' % (a.arm, a.seed, ('_' + a.tag) if a.tag else '')
    print('arm %s nf %d dev %s | train runs %d val runs %d | val anchors %d (A %d, B %d, real-loss anchors A %d B %d)' % (a.arm, nf, dev, len(tr_runs), len(va_runs), len(isA), isA.sum(), (~isA).sum(), (isA & isR).sum(), (~isA & isR).sum()), flush=True)
    print('BASELINES ' + json.dumps({k: {p: round(v[p]['score'], 4) for p in v} for k, v in bs_.items()}), flush=True)
    net = Net(a.arm, nf, a.hidden).to(dev); print('parameters', sum(q.numel() for q in net.parameters()), flush=True); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, a.lr * 0.02)
    best = math.inf; log = open(out / (stem + '.jsonl'), 'w')
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); B = build(tr_runs, rng, (a.sig_c, a.sig_s), aug=bool(a.aug)); N = len(B['hm']); order = rng.permutation(N); net.train(); tot = 0.; nb = 0
        for s in range(0, N, a.bs):
            idx = order[s:s + a.bs]; q = net(*T(B, idx, dev, nf)); g = torch.as_tensor(B['gt'][idx], dtype=torch.float32, device=dev); ok = torch.as_tensor(B['ok'][idx], dtype=torch.float32, device=dev)
            loss = loss_fn(q, g, ok); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tot += float(loss); nb += 1
        sch.step(); p = predict(net, V, dev, nf); ps = predict(net, S, dev, nf); rA = score(p, V, isA); rAs = score(ps, S, sA); sel = (rA['score'] + rAs['score']) / 2; isb = sel < best; best = min(best, sel)
        rec = dict(epoch=ep, arm=a.arm, seed=a.seed, train_loss=tot / nb, train_anchors=int(N), A=rA, A_real=score(p, V, isA & isR), A_syn=rAs, sel=sel, best=isb, sec=round(time.time() - t0))
        if isb:
            rec['B'] = score(p, V, ~isA); rec['B_real'] = score(p, V, ~isA & isR); rec['B_syn'] = score(ps, S, sB); torch.save(dict(arm=a.arm, nf=nf, hidden=a.hidden, aug=a.aug, model=net.state_dict(), epoch=ep), out / (stem + '_best.pt'))
            bestrec = rec
        log.write(json.dumps(rec) + '\n'); log.flush()
        print('EPOCH %d %s A %.4f%s A_real %.4f A_syn %.4f | kalman A %.4f A_real %.4f A_syn %.4f | %ds' % (ep, a.arm, rA['score'], ' *' if isb else '', rec['A_real']['score'], rAs['score'], bs_['kalman']['A']['score'], bs_['kalman']['A_real']['score'], bs_['kalman']['A_syn']['score'], rec['sec']), flush=True)
    res = dict(arm=a.arm, seed=a.seed, best_epoch=bestrec['epoch'], A=bestrec['A']['score'], B=bestrec['B']['score'], B_real=bestrec['B_real']['score'], B_syn=bestrec['B_syn']['score'], kalman_B_syn=bs_['kalman']['B_syn']['score'], hidden=a.hidden, aug=a.aug, B_horizons=bestrec['B'], B_real_horizons=bestrec['B_real'],
               kalman_B=bs_['kalman']['B']['score'], kalman_B_real=bs_['kalman']['B_real']['score'], baselines=bs_)
    (out / (stem + '_RESULT.json')).write_text(json.dumps(res, indent=1)); print('TRAIN_DONE ' + json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}), flush=True)


if __name__ == '__main__':
    main()
