"""fix_train.py (= gru_mismatch_20261116 pf_train.py) - pos_train.py (gru_kpt / mamba_kpt, config a) with the fixed builder fix_lib.
Same recipe: from scratch, AdamW lr 2e-3 wd 1e-4, cosine to 2 %, 30 epochs, batch 512, 4 training versions (v0 plain, v1..3 flip / reverse / half
frame rate + synthetic vanish 0.6, DanceTrack anchors every 3rd frame), torch seed 0, numpy seed 0, grad clip 1.0; selection on val-A:
sel = (score_all + score_syn) / 2 (m5 = mean over horizons 5, 10, 15 of mean min(centre error / target height, 2)).
Usage: python fix_train.py <out dir> <arm gru_kpt|mamba_kpt> [--epochs 30] [--versions 4] [--workers 6] [--threads 8] [--no-fixwarp] [--no-edges]
"""
import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fix_lib as F   # noqa: E402
P = F.P; L = F.L
FIXWARP = True


def build_train_version(args):
    v, names, fixwarp, edges = args; F.EDGES[0] = edges; rng = np.random.default_rng([1000 + v, sum(map(ord, names[0]))]); out = []
    reg = {r['name']: r for r in P.registry()}
    for nm in names:
        r = reg[nm]; S = L.load_seq(F.PR, r)
        for k in F.DETS:
            T = F.tracks_of(S, k)
            flip, rev, half = (rng.uniform() < 0.5, rng.uniform() < 0.5, rng.uniform() < 0.3) if v > 0 else (False, False, False)
            T2, W2, n2 = L.augment(T, S['W'], S['n'], rng, flip, rev, half)
            stride = 3 if r['ds'] == 'DT' else 1
            A = F.anchors(S, k, T2, W2, n2, rng=rng, stride=stride, offset=int(rng.integers(stride)), vanish_p=0.6 if v > 0 else 0.0, fixwarp=fixwarp)
            out.append((A, dict(seq=nm, ds=r['ds'], det=k)))
    return out


def build_val(args):
    nm, syn, fixwarp, edges = args; F.EDGES[0] = edges; r = {x['name']: x for x in P.registry()}[nm]; S = L.load_seq(F.PR, r); out = []
    for k in F.DETS:
        T = F.tracks_of(S, k)
        if syn: A = F.anchors(S, k, T, S['W'], S['n'], rng=np.random.default_rng(12345 + sum(map(ord, nm)) + (0 if k == 'near' else 7)), syn_only=True, vanish_p=1.0, fixwarp=fixwarp)
        else: A = F.anchors(S, k, T, S['W'], S['n'], fixwarp=fixwarp)
        out.append((A, dict(seq=nm, ds=r['ds'], det=k)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('arm'); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--versions', type=int, default=4); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--threads', type=int, default=8); ap.add_argument('--no-fixwarp', action='store_true'); ap.add_argument('--no-edges', action='store_true'); a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads); fixwarp = not a.no_fixwarp; edges = not a.no_edges; F.EDGES[0] = edges
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = 'pf_%s' % a.arm; cache = out / 'cache'; cache.mkdir(exist_ok=True)
    dev = 'cpu'; torch.manual_seed(0); rng = np.random.default_rng(0)
    reg = P.registry(); tr = [r['name'] for r in reg if r['split'] == 'train']; va = [r['name'] for r in reg if r['split'] == 'valA']
    t0 = time.time(); keep = ('hx', 'hm', 'xl', 'base', 'h0', 'tgt', 'tgt2', 'real', 'syn', 'hg', 't', 'seq', 'ds', 'det')
    if not (cache / 'DONE').exists():
        with Pool(a.workers) as pool:
            for v in range(a.versions):
                res = pool.map(build_train_version, [(v, [nm], fixwarp, edges) for nm in tr])
                items = [x for rr in res for x in rr]; B = L.stack([x[0] for x in items], [x[1] for x in items])
                np.savez(cache / ('v%d.npz' % v), **{k: B[k] for k in keep}); print('train version %d anchors %d (%.0f s)' % (v, len(B['h0']), time.time() - t0), flush=True)
            V = [x for rr in pool.map(build_val, [(nm, False, fixwarp, edges) for nm in va]) for x in rr]; V = L.stack([x[0] for x in V], [x[1] for x in V])
            Sy = [x for rr in pool.map(build_val, [(nm, True, fixwarp, edges) for nm in va]) for x in rr]; Sy = L.stack([x[0] for x in Sy], [x[1] for x in Sy])
        np.savez(cache / 'valA.npz', **{k: V[k] for k in keep}); np.savez(cache / 'synA.npz', **{k: Sy[k] for k in keep}); (cache / 'DONE').write_text('ok')
    ld = lambda f: dict(np.load(cache / f))
    vers = [ld('v%d.npz' % v) for v in range(a.versions)]; V = ld('valA.npz'); Sy = ld('synA.npz')
    print('train versions %d anchors %s val-A anchors %d syn %d (%.0f s)' % (len(vers), [len(x['h0']) for x in vers], len(V['h0']), len(Sy['h0']), time.time() - t0), flush=True)
    net = L.make_net(a.arm).to(dev); lr = 2e-3
    print('parameters', sum(p.numel() for p in net.parameters()), 'lr', lr, 'dev', dev, 'threads', a.threads, 'fixwarp', fixwarp, 'edges', edges, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, lr * 0.02)
    T = lambda B, k, idx: torch.as_tensor(B[k][idx].astype(np.float32), device=dev)
    ceV0 = L.ce_matrix(V['base'], V['tgt']); ceS0 = L.ce_matrix(Sy['base'], Sy['tgt'])
    kal = dict(all=L.score(ceV0)['score'], syn=L.score(ceS0)['score']); print('KALMAN val-A all %.4f syn %.4f' % (kal['all'], kal['syn']), flush=True)
    log = open(out / (stem + '.jsonl'), 'w'); best = math.inf; bestrec = None
    for ep in range(1, a.epochs + 1):
        t1 = time.time(); B = vers[(ep - 1) % len(vers)]; N = len(B['h0']); order = rng.permutation(N); net.train(); tot = 0.; nb = 0
        for s in range(0, N, a.bs):
            idx = order[s:s + a.bs]
            q = net(T(B, 'hx', idx), T(B, 'hm', idx), T(B, 'xl', idx), T(B, 'base', idx), T(B, 'h0', idx))
            g = torch.as_tensor(B['tgt'][idx], device=dev); ok = torch.isfinite(g[..., 0])
            loss = L.loss_fn(q, torch.nan_to_num(g, nan=1.0), ok); opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tot += float(loss); nb += 1
        sch.step()
        ceV = L.ce_matrix(L.predict(net, V, dev), V['tgt']); ceS = L.ce_matrix(L.predict(net, Sy, dev), Sy['tgt'])
        rA = L.score(ceV); rS = L.score(ceS); sel = (rA['score'] + rS['score']) / 2; isb = sel < best; best = min(best, sel)
        rec = dict(epoch=ep, arm=a.arm, lr=lr, train_loss=tot / nb, anchors=int(N), A=rA, A_syn=rS, A_real=L.score(ceV, V['real'].astype(bool)), sel=sel, best=isb, sec=round(time.time() - t1))
        if isb:
            torch.save(dict(arm=a.arm, cfg='a_fix', lr=lr, model=net.state_dict(), epoch=ep, sel=sel, fixwarp=fixwarp, edges=edges), out / (stem + '_best.pt')); bestrec = rec
        log.write(json.dumps(rec) + '\n'); log.flush()
        print('EPOCH %d %s A %.4f syn %.4f real %.4f sel %.4f%s | kalman %.4f %.4f | %ds' % (ep, a.arm, rA['score'], rS['score'], rec['A_real']['score'], sel, ' *' if isb else '', kal['all'], kal['syn'], rec['sec']), flush=True)
    (out / (stem + '_RESULT.json')).write_text(json.dumps(dict(best=bestrec, kalman_valA=kal, versions=a.versions, epochs=a.epochs, fixwarp=fixwarp, edges=edges), indent=1))
    print('TRAIN_DONE', stem, json.dumps(dict(epoch=bestrec['epoch'], sel=bestrec['sel'])), flush=True)


if __name__ == '__main__':
    main()
