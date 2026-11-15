"""gru_v2/scripts/g2_train.py - train one GRU refiner v2 run on the PC CPU (PREREG_GRU_V2.json).
Usage: python g2_train.py <arch gate|res> <lam_j> <lam_l> [--epochs 30] [--lr 0.002] [--threads 2] [--versions 6] [--smoke N]
Train: data/train_v*.pkl (one version per epoch). Epoch choice inside the run: the run's own loss on the val-A server-like rows (data/rows_valA.pkl).
Writes runs/<stem>.jsonl, runs/<stem>_best.pt, runs/<stem>_RESULT.json.
"""
import argparse
import json
import math
import pickle
import time

import numpy as np
import torch

import g2_lib as L


def batch(pcs, idx, tkey='tgt'):
    T = max(len(pcs[i]['X']) for i in idx); B = len(idx)
    X = np.zeros((B, T, L.NS), np.float32); Cb = np.ones((B, T, 4), np.float32); Ha = np.zeros((B, T, 2, 3), np.float32); Ha[:, :, 0, 0] = 1; Ha[:, :, 1, 1] = 1
    ST = np.zeros((B, T), bool); M = np.zeros((B, T), bool); G = np.ones((B, T, 4), np.float32); TR = np.ones((B, T, 4), np.float32); Zr = np.ones((B, T, 4), np.float32); OK = np.zeros((B, T), bool); CON = np.zeros((B, T), bool); K = np.full((B, T), 99, np.int64)
    for b, i in enumerate(idx):
        p = pcs[i]; n = len(p['X']); X[b, :n] = p['X']; Cb[b, :n] = p['C']; Ha[b, :n] = p['Ha']; ST[b, :n] = p['st']; M[b, :n] = True; g = p[tkey]; ok = np.isfinite(g[:, 0])
        G[b, :n][ok] = g[ok]; OK[b, :n] = ok; CON[b, :n] = p['con']; K[b, :n] = p['k']; TR[b, :n] = p['trk']; Zr[b, :n] = p['z']
    return [torch.as_tensor(v) for v in (X, Cb, Ha, ST, M, G, OK, CON, K, TR, Zr)]


def val_parts(net, V, bs=128):
    order = np.argsort([len(v['X']) for v in V]); tot = None
    with torch.no_grad():
        for a in range(0, len(order), bs):
            X, Cb, Ha, ST, M, G, OK, CON, K, TR, Zr = batch(V, order[a:a + bs]); q, _ = net(X, Cb, Ha, ST, M)
            P_ = L.loss_parts(q, G, OK, CON, K, Cb, Ha, M, TR, Zr)
            tot = P_ if tot is None else {k: (tot[k][0] + P_[k][0], tot[k][1] + P_[k][1]) for k in P_}
    return tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('arch'); ap.add_argument('lam_j', type=float); ap.add_argument('lam_l', type=float)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--versions', type=int, default=6); ap.add_argument('--bs', type=int, default=64); ap.add_argument('--chunk', type=int, default=64); ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
    stem = 'g3_%s_j%g_l%g' % (a.arch, a.lam_j, a.lam_l); out = L.G2 / 'runs'; D = L.G2 / 'data'; t0 = time.time()
    vers = [pickle.loads((D / ('train_v%d.pkl' % v)).read_bytes()) for v in range(a.versions)]
    V = [dict(r['inp'], tgt=r['e']['tgt'].astype(np.float32), trk=r['e']['trk'].astype(np.float32), z=r['e']['z'].astype(np.float32)) for r in pickle.loads((D / 'rows_valA.pkl').read_bytes())['recs']]
    if a.smoke: vers = [v[:a.smoke] for v in vers]; V = V[:a.smoke]
    print('train versions %d pieces %d rows %d | val-A ids %d rows %d (%.0f s)' % (len(vers), len(vers[0]), sum(len(p['X']) for p in vers[0]), len(V), sum(len(v['X']) for v in V), time.time() - t0), flush=True)
    net = L.make_net(a.arch); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, a.lr * 0.02)
    best = math.inf; bestrec = None; log = open(out / (stem + '.jsonl'), 'w')
    for ep in range(1, a.epochs + 1):
        t1 = time.time(); pcs = vers[(ep - 1) % len(vers)]; order = rng.permutation(len(pcs)); lens = np.array([len(pcs[i]['X']) for i in order])
        order = order[np.argsort(lens, kind='stable')]; bl = [order[s:s + a.bs] for s in range(0, len(order), a.bs)]; rng.shuffle(bl); net.train(); tot = 0.; nb = 0
        for idx in bl:
            X, Cb, Ha, ST, M, G, OK, CON, K, TR, Zr = batch(pcs, idx); state = None
            for s in range(0, X.shape[1], a.chunk):
                s0 = max(s - 2, 0); sl = slice(s, s + a.chunk)     # the loss needs the two rows before the chunk for the second difference: recomputed without gradient below
                q, state2 = net(X[:, sl], Cb[:, sl], Ha[:, sl], ST[:, sl], M[:, sl], state)
                if s > 0: qq = torch.cat([qprev[:, -(s - s0):].detach(), q], 1); sl2 = slice(s0, s + a.chunk)
                else: qq = q; sl2 = sl
                P_ = L.loss_parts(qq, G[:, sl2], OK[:, sl2], CON[:, sl2], K[:, sl2], Cb[:, sl2], Ha[:, sl2], M[:, sl2], TR[:, sl2], Zr[:, sl2]); loss = L.loss_total(P_, a.lam_j, a.lam_l)
                if M[:, sl].any() and loss.requires_grad:
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tot += float(loss); nb += 1
                state = (state2[0].detach(), state2[1].detach()); qprev = q
        sch.step(); net.eval(); vp = val_parts(net, V); vl = float(L.loss_total(vp, a.lam_j, a.lam_l)); isb = vl < best; best = min(best, vl)
        f = lambda k: float(vp[k][0] / vp[k][1].clamp(min=1))
        rec = dict(epoch=ep, train_loss=tot / max(nb, 1), updates=nb, val_loss=vl, val_acc=f('acc'), val_jit=f('jit'), val_step=f('step'), val_lag_mid=f('lag_mid'), val_lag_fast=f('lag_fast'), val_aimtrk=f('aimtrk'), val_lag_fast_trk=f('lag_fast_trk'), best=bool(isb), sec=round(time.time() - t1))
        if isb: torch.save(dict(arch=a.arch, lam_j=a.lam_j, lam_l=a.lam_l, model=net.state_dict(), epoch=ep, val_loss=vl), out / (stem + '_best.pt')); bestrec = rec
        log.write(json.dumps(rec) + '\n'); log.flush()
        print('EPOCH %d train %.5f val %.5f (acc %.5f jit %.5f step %.5f lag mid %+.4f fast %+.4f)%s | %ds' % (ep, rec['train_loss'], vl, rec['val_acc'], rec['val_jit'], rec['val_step'], rec['val_lag_mid'], rec['val_lag_fast'], ' *' if isb else '', rec['sec']), flush=True)
    if not a.smoke: (out / (stem + '_RESULT.json')).write_text(json.dumps(dict(best=bestrec, arch=a.arch, lam_j=a.lam_j, lam_l=a.lam_l, lr=a.lr, epochs=a.epochs, versions=a.versions), indent=1))
    print('TRAIN_DONE', stem, bestrec['epoch'], bestrec['val_loss'], flush=True)


if __name__ == '__main__':
    main()
