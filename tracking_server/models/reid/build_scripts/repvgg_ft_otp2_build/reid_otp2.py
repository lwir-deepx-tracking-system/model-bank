"""Appearance model (re-ID) trained with REAL identities from OpenThermalPose2 (user 2026-11-13: "26m 학습처럼 그 데이터로").

Identities: the indoor sitting part of OpenThermalPose2 has file names <subject>_<session>_... with one person per picture; pictures with the same
subject number show the same person (checked by eye: stage1/otp2_id_check.jpg) and the session number changes the clothes. Identity = (subject, session).
Train / val / test subjects do not overlap (the dataset's own split).
Crop = the server's crop of the new detector: the BODY BOX (tools/skeleton_box.py fit to the labelled keypoints), clipped to the picture, stretched to
64x128 (cv2 INTER_LINEAR), grey. Training views add: cut at one picture edge (keep 35-80 %, stretched again), an occluder (another crop's pixels
over the bottom or one side, 20-60 %), Gaussian blur, brightness / contrast, noise.

Model: the board's RepVGG-A0 128-d (reid7 start/repvgg_a0_live_128d.pt) as start. Loss = batch-hard triplet (margin 0.3, P identities x K crops per
batch) on the L2 features + 0.5 x (1 - cos) to the frozen start model on the start model's own public full-body crops (cache30), so the general
features and the cosine scale the relink judge was fitted on are not lost.
Evaluation: (a) OTP2 test identities, per identity half the crops = gallery mean, half = queries, same view kind on both sides, rank1 / AUC;
(b) our indoor crops (reid7 R7 val / test, r7_common.metrics with the views of reid_partial.py). Start model and trained model, same numbers.

Usage (Vast): REID7_ROOT=/workspace/reid7 python -u reid_otp2.py crops --otp2 <otp2_dataset> --template <template_points.npy> --out <crops.npz>
              REID7_ROOT=/workspace/reid7 python -u reid_otp2.py train --crops <crops.npz> --out <dir> [--steps 3000]
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, os.environ.get('REID7_ROOT', '/workspace/reid7'))
SEED = 20261114


def make_crops(a):
    import cv2
    import skeleton_box as SB
    T = SB.template_from_points(np.load(a.template))
    R = Path(a.otp2); crops, pid, split, subj = [], [], [], []
    for si, s in enumerate(('train', 'val', 'test')):
        for f in sorted((R / s / 'labels').glob('*.txt')):
            parts = f.stem.split('_')
            if len(parts) not in (6, 7): continue
            rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
            if len(rows) != 1 or len(rows[0]) != 56: continue
            ip = next((R / s / 'images').glob(f.stem + '.*'), None)
            if ip is None: continue
            g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE); h, w = g.shape
            v = np.array(rows[0], float); k = v[5:].reshape(17, 3); use = k[:, 2] > 0
            if use.sum() < 2: continue
            fit = SB.fit(T, k[:, :2] * [w, h], use)
            if fit is None: continue
            b = SB.box_of(T, fit); x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))
            if x2 - x1 < 8 or y2 - y1 < 16: continue
            crops.append(cv2.resize(g[y1:y2, x1:x2], (64, 128), interpolation=cv2.INTER_LINEAR))
            pid.append(int(parts[0]) * 10 + int(parts[1])); split.append(si); subj.append(int(parts[0]))
    crops, pid, split, subj = np.stack(crops), np.array(pid), np.array(split), np.array(subj)
    np.savez_compressed(a.out, crop=crops, pid=pid, split=split, subj=subj)
    print(json.dumps({s: dict(crops=int((split == i).sum()), identities=len(set(pid[split == i]))) for i, s in enumerate(('train', 'val', 'test'))}), flush=True)


def aug(torch, F, x, gen, kind):
    """x float [B,1,128,64]: 'train' random, 'cut' / 'occluded' deterministic-ish (seeded generator), 'full' as is."""
    B, dev = x.shape[0], x.device
    U = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi, generator=gen)
    P = lambda p: torch.rand(B, device=dev, generator=gen) < p
    if kind == 'full': return x
    one = torch.ones(B, dtype=torch.bool, device=dev); zero = ~one
    cut = P(0.4) if kind == 'train' else (one if kind == 'cut' else zero)
    side = torch.randint(0, 4, (B,), device=dev, generator=gen); keep = U(0.35, 0.8)
    y0 = torch.where(cut & (side == 1), 1 - keep, torch.zeros_like(keep)); y1 = torch.where(cut & (side == 0), keep, torch.ones_like(keep))
    x0 = torch.where(cut & (side == 3), 1 - keep, torch.zeros_like(keep)); x1 = torch.where(cut & (side == 2), keep, torch.ones_like(keep))
    flip = torch.where(P(0.5), -1.0, 1.0) if kind == 'train' else torch.ones(B, device=dev)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = (x1 - x0) * flip; th[:, 0, 2] = (x0 + x1) - 1; th[:, 1, 1] = (y1 - y0); th[:, 1, 2] = (y0 + y1) - 1
    v = F.grid_sample(x, F.affine_grid(th, (B, 1, 128, 64), align_corners=False), mode='bilinear', padding_mode='border', align_corners=False)
    occ = P(0.4) if kind == 'train' else (one if kind == 'occluded' else zero)
    rr = torch.arange(128, device=dev).view(1, 128, 1).float(); cc = torch.arange(64, device=dev).view(1, 1, 64).float()
    a3 = U(0.2, 0.6).view(B, 1, 1); left = P(0.5).view(B, 1, 1); mode = P(0.5).view(B, 1, 1)
    m = (mode & (rr >= 128 * (1 - a3))) | (~mode & ((left & (cc < 64 * a3)) | (~left & (cc >= 64 * (1 - a3)))))
    v = torch.where((m & occ.view(B, 1, 1)).unsqueeze(1), v[torch.roll(torch.arange(B, device=dev), 1)], v)
    if kind == 'train':
        sg = U(0.3, 2.0); k1 = torch.exp(-1.0 / (2 * sg * sg)); k = torch.stack([k1, torch.ones_like(k1), k1], 1); k = k / k.sum(1, keepdim=True)
        k2 = (k.view(B, 3, 1) * k.view(B, 1, 3)).view(B, 1, 3, 3)
        vb = F.conv2d(F.pad(v.view(1, B, 128, 64), (1, 1, 1, 1), mode='replicate'), k2, groups=B).view(B, 1, 128, 64)
        v = torch.where(P(0.3).view(B, 1, 1, 1), vb, v)
        v = torch.where(P(0.6).view(B, 1, 1, 1), (v - 128) * U(0.75, 1.25).view(B, 1, 1, 1) + 128 + U(-20, 20).view(B, 1, 1, 1), v)
        v = v + P(0.3).view(B, 1, 1, 1) * torch.randn(v.shape, device=dev, generator=gen) * U(1, 6).view(B, 1, 1, 1)
    return v.clamp(0, 255).round()


def feats(torch, F, model, crops, dev, kind, bs=256):
    gen = torch.Generator(device=dev); gen.manual_seed(SEED + {'full': 0, 'cut': 2, 'occluded': 3}[kind]); model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            x = torch.from_numpy(np.ascontiguousarray(crops[i:i + bs])).to(dev).float().unsqueeze(1)
            out.append(F.normalize(model(aug(torch, F, x, gen, kind).expand(-1, 3, -1, -1)).float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def otp2_eval(torch, F, model, dev, crops, pid):
    """gallery = mean of the even-indexed crops of each identity, queries = odd-indexed crops; same view kind on both sides."""
    res = {}
    for kind in ('full', 'cut', 'occluded'):
        E = feats(torch, F, model, crops, dev, kind); ids = sorted(set(pid.tolist()))
        gi, qi = [], []
        for p in ids:
            idx = np.flatnonzero(pid == p); gi.append(idx[0::2]); qi.append(idx[1::2])
        G = np.stack([E[g].mean(0) for g in gi]); G /= np.linalg.norm(G, axis=1, keepdims=True)
        ok, pos, neg = [], [], []
        for j, q in enumerate(qi):
            s = E[q] @ G.T; ok += (s.argmax(1) == j).tolist(); pos += s[:, j].tolist(); neg += np.delete(s, j, 1).ravel().tolist()
        pos, neg = np.array(pos), np.array(neg)
        auc = float((pos[:, None] > neg[None, ::max(1, len(neg) // 20000)]).mean())
        res[kind] = dict(rank1=round(float(np.mean(ok)), 4), auc=round(auc, 4), identities=len(ids), queries=int(sum(len(q) for q in qi)))
    return res


def indoor_eval(torch, F, model, dev, part):
    import r7_common as C
    import reid_partial as RP
    tr, q = C.load_crops('train'), C.load_crops(part); out = {}
    for kind in ('full', 'body', 'cut', 'occluded'):
        m = C.metrics(RP.embed_views(torch, F, model, tr['crop'], dev, kind), tr, RP.embed_views(torch, F, model, q['crop'], dev, kind), q)
        out[kind] = round(m['score'], 4)
    return out


def train(a):
    import torch
    import torch.nn.functional as F
    import r7_common as C
    torch.manual_seed(SEED); dev = torch.device('cuda'); t0 = time.time(); torch.backends.cudnn.benchmark = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    d = np.load(a.crops); crops, pid, split = d['crop'], d['pid'], d['split']
    trm = split == 0; Xtr = torch.from_numpy(crops[trm]).to(dev); Ptr = torch.from_numpy(pid[trm]).to(dev)
    ids = np.unique(pid[trm]); by_id = {int(p): np.flatnonzero(pid[trm] == p) for p in ids}
    pub = torch.from_numpy(np.load(os.environ.get('CACHE30', '/workspace/d30/cache30/crops.npy'), mmap_mode='r')[:]).to(dev)
    student, _ = C.load_student(torch); assert C.sha(C.START_STUDENT) == C.START_STUDENT_SHA
    teacher = copy.deepcopy(student).eval().to(dev)
    for p in teacher.parameters(): p.requires_grad_(False)
    student = student.to(dev); ema = copy.deepcopy(student).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    log = dict(start_sha=C.START_STUDENT_SHA, train_crops=int(trm.sum()), train_identities=len(ids), steps=a.steps, P=a.P, K=a.K, lr=a.lr, kd=a.kd, records=[])
    vt = split == 2
    log['start'] = dict(otp2_test=otp2_eval(torch, F, teacher, dev, crops[vt], pid[vt]), indoor_val=indoor_eval(torch, F, teacher, dev, 'val'))
    print(json.dumps(dict(start=log['start'])), flush=True)
    dec = [p for n, p in student.named_parameters() if p.ndim > 1]; nodec = [p for n, p in student.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([dict(params=dec, weight_decay=0.05), dict(params=nodec, weight_decay=0.0)], lr=a.lr)
    gen = torch.Generator(device=dev); gen.manual_seed(SEED); rng = np.random.default_rng(SEED)
    es, ms = ema.state_dict(), student.state_dict()
    ef = [es[k] for k in es if es[k].dtype.is_floating_point]; mf = [ms[k] for k in es if es[k].dtype.is_floating_point]
    warm = max(1, a.steps // 20); acc = {}
    for step in range(a.steps):
        student.train()
        chosen = rng.choice(ids, a.P, replace=False)
        idx = np.concatenate([rng.choice(by_id[int(p)], a.K, replace=len(by_id[int(p)]) < a.K) for p in chosen])
        idx_t = torch.from_numpy(idx).to(dev)
        x = aug(torch, F, Xtr[idx_t].float().unsqueeze(1), gen, 'train'); lab = Ptr[idx_t]
        pi = torch.from_numpy(rng.integers(0, len(pub), 128)).to(dev); xp = pub[pi].float().unsqueeze(1)
        with torch.no_grad(): tp = F.normalize(teacher(xp.expand(-1, 3, -1, -1)).float(), dim=1)
        lr = a.lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, step - warm) / max(1, a.steps - warm))))
        for g in opt.param_groups: g['lr'] = lr
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            e = student(x.expand(-1, 3, -1, -1)); ep = student(xp.expand(-1, 3, -1, -1))
        e = F.normalize(e.float(), dim=1); ep = F.normalize(ep.float(), dim=1)
        D = torch.cdist(e, e); same = lab.view(-1, 1) == lab.view(1, -1)
        hp = (D * same.float()).max(1).values; hn = (D + 1e4 * same.float()).min(1).values
        l_tri = F.relu(hp - hn + 0.3).mean(); l_kd = (1 - (ep * tp).sum(1)).mean()
        loss = l_tri + a.kd * l_kd
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(dec + nodec, 5.0); opt.step()
        with torch.no_grad():
            torch._foreach_mul_(ef, 0.998); torch._foreach_add_(ef, mf, alpha=0.002)
            for k in es:
                if not es[k].dtype.is_floating_point: es[k].copy_(ms[k])
            for k, val in dict(loss=loss, triplet=l_tri, kd=l_kd, frac_active=(hp - hn + 0.3 > 0).float().mean()).items(): acc[k] = acc.get(k, 0) + float(val)
        if (step + 1) % 250 == 0 or step + 1 == a.steps:
            n = 250 if (step + 1) % 250 == 0 else (step + 1) % 250
            rec = dict(step=step + 1, lr=round(lr, 7), minutes=round((time.time() - t0) / 60, 2), **{k: round(v / n, 5) for k, v in acc.items()}); acc = {}
            if (step + 1) % 1000 == 0 or step + 1 == a.steps:
                vm = split == 1
                rec['otp2_val'] = {k: v['rank1'] for k, v in otp2_eval(torch, F, ema, dev, crops[vm], pid[vm]).items()}
                rec['indoor_val'] = indoor_eval(torch, F, ema, dev, 'val')
            log['records'].append(rec); print(json.dumps(rec), flush=True)
            if not math.isfinite(rec['loss']): raise SystemExit('NaN')
            (out / 'TRAIN_LOG.json').write_text(json.dumps(log, indent=1))
    sd = {k: v.detach().cpu() for k, v in ema.state_dict().items()}
    torch.save(dict(kind='repvgg', state_dict=sd, start_sha=C.START_STUDENT_SHA, steps=a.steps, method='OTP2 identity triplet + KD on public crops'), out / 'final_ema.pt')
    log['end'] = dict(otp2_test=otp2_eval(torch, F, ema, dev, crops[vt], pid[vt]), indoor_val=indoor_eval(torch, F, ema, dev, 'val'), indoor_test=indoor_eval(torch, F, ema, dev, 'test'))
    log['start']['indoor_test'] = indoor_eval(torch, F, teacher, dev, 'test'); log['minutes'] = round((time.time() - t0) / 60, 1)
    (out / 'TRAIN_LOG.json').write_text(json.dumps(log, indent=1))
    print(json.dumps(dict(start=log['start'], end=log['end'])), flush=True); print('TRAIN_DONE', out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode')
    ap.add_argument('--otp2'); ap.add_argument('--template'); ap.add_argument('--out', required=True); ap.add_argument('--crops')
    ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--P', type=int, default=32); ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--kd', type=float, default=0.5)
    a = ap.parse_args()
    if a.mode == 'crops': make_crops(a)
    elif a.mode == 'train': train(a)
    elif a.mode == 'smoke': a.steps = 40; train(a)
