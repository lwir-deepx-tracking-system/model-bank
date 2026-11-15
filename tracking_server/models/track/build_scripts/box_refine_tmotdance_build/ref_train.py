"""ref_train.py - learned causal box refiner (PREREG box_refiner + amendment 1).
Usage: python -I ref_train.py <root> <lr> [--epochs 30] [--versions 8] [--workers 10]
Train pieces: per gt person of the train split, near / far / spliced near+far, synthetic occluder, flips (8 random versions, one per epoch).
Selection: validation loss (same formula as the training loss) on the server-like val-A rows (tracker ids, streams near, far, sw) -> runs/ref_lr<lr>_best.pt.
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
import prcommon as P
import pos_lib as PL
import ref_lib as R


def build_version(args):
    root, v, nm = args; r = {x['name']: x for x in P.registry()}[nm]; S = PL.load_seq(root, r)
    return R.pieces_of_seq(S, np.random.default_rng([77 + v, sum(map(ord, nm))]), aug=True)


def build_val(args):
    """server-like val rows of one sequence -> list of segments dict(x, z, tgt, con, key) for every stream."""
    root, nm = args; r = {x['name']: x for x in P.registry()}[nm]; out = []
    for stream in ('near', 'far', 'sw'):
        ids, S = R.rows_of(root, r, stream)
        for i, e in ids.items():
            for sl in R.segments(e):
                fs = e['f'][sl]; det = {int(f): (e['z'][j], e['kp'][j]) for j, f in zip(range(sl.start, sl.stop), fs)}
                ser = P.ref_series(int(fs[0]), int(fs[-1]), det, S['W'])
                out.append(dict(x=np.array([ser[int(f)][0] for f in fs], np.float32), z=e['z'][sl].astype(np.float32), tgt=e['tgt'][sl].astype(np.float32),
                                con=np.r_[0, (np.diff(fs) == 1)].astype(np.int8), key=(nm, stream, i, sl.start)))
    return out


def val_loss(net, V, dev):
    import torch
    out = []
    with torch.no_grad():
        for a in range(0, len(V), 256):
            idx = list(range(a, min(a + 256, len(V)))); X, Z, G, OK, C, M = R.batch_pieces(V, idx)
            q, _ = net(torch.as_tensor(X, device=dev), torch.as_tensor(Z, device=dev), torch.as_tensor(M, device=dev))
            q = q.cpu().numpy()
            for b, i in enumerate(idx): out.append(q[b, :len(V[i]['z'])])
    return R.loss_np(out, [v['tgt'] for v in V], [v['con'] for v in V])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('lr', type=float); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--versions', type=int, default=8); ap.add_argument('--workers', type=int, default=10); ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--chunk', type=int, default=64); ap.add_argument('--mem', type=float, default=0.15); ap.add_argument('--cache', default='/dev/shm/posref_refcache.pkl'); ap.add_argument('--build-only', action='store_true'); a = ap.parse_args()
    import torch
    root = Path(a.root); out = root / 'runs'; out.mkdir(exist_ok=True); stem = 'ref_lr%g' % a.lr
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda': torch.backends.cudnn.benchmark = False; torch.cuda.set_per_process_memory_fraction(a.mem)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    reg = P.registry(); tr = [r['name'] for r in reg if r['split'] == 'train']; va = [r['name'] for r in reg if r['split'] == 'valA']
    t0 = time.time(); cache = Path(a.cache)
    if cache.exists():   # built once (pickle in RAM disk), the other lr run reads it
        import pickle; vers, V = pickle.loads(cache.read_bytes()); vers = vers[:a.versions]
    else:
        with Pool(a.workers) as pool:
            vers = []
            for v in range(a.versions):
                vers.append([p for pp in pool.map(build_version, [(str(root), v, nm) for nm in tr]) for p in pp])
                print('version %d pieces %d frames %d (%.0f s)' % (v, len(vers[-1]), sum(len(p['z']) for p in vers[-1]), time.time() - t0), flush=True)
            V = [s for ss in pool.map(build_val, [(str(root), nm) for nm in va]) for s in ss]
        import pickle; tmp = cache.with_suffix('.tmp'); tmp.write_bytes(pickle.dumps((vers, V), protocol=4)); tmp.replace(cache)
        if a.build_only: print('BUILD_DONE', flush=True); return
    print('val-A segments %d rows %d (%.0f s)' % (len(V), sum(len(v['z']) for v in V), time.time() - t0), flush=True)
    raw = R.loss_np([v['z'] for v in V], [v['tgt'] for v in V], [v['con'] for v in V]); print('RAW val-A loss %.5f acc %.5f jit %.5f' % raw[:3], flush=True)
    net = R.make_net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, a.lr * 0.02)
    best = math.inf; bestrec = None; log = open(out / (stem + '.jsonl'), 'w')
    for ep in range(1, a.epochs + 1):
        t1 = time.time(); pcs = vers[(ep - 1) % len(vers)]; order = rng.permutation(len(pcs)); lens = np.array([len(pcs[i]['z']) for i in order])
        order = order[np.argsort(lens, kind='stable')]; bl = [order[s:s + a.bs] for s in range(0, len(order), a.bs)]; rng.shuffle(bl); net.train(); tot = 0.; nb = 0
        for idx in bl:
            X, Z, G, OK, C, M = (torch.as_tensor(v, device=dev) for v in R.batch_pieces(pcs, idx)); h = None
            for s in range(0, X.shape[1], a.chunk):
                sl = slice(s, s + a.chunk); q, h = net(X[:, sl], Z[:, sl], M[:, sl], h)
                loss, acc, jit = R.loss_torch(q, G[:, sl], OK[:, sl], C[:, sl])
                if OK[:, sl].any():
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tot += float(loss); nb += 1
                h = h.detach()
        sch.step(); vl = val_loss(net, V, dev); isb = vl[0] < best; best = min(best, vl[0])
        rec = dict(epoch=ep, lr=a.lr, train_loss=tot / max(nb, 1), updates=nb, val_loss=float(vl[0]), val_acc=float(vl[1]), val_jit=float(vl[2]), raw_val=[float(x) for x in raw[:3]], best=bool(isb), sec=round(time.time() - t1))
        if isb: torch.save(dict(lr=a.lr, model=net.state_dict(), epoch=ep, val_loss=vl[0]), out / (stem + '_best.pt')); bestrec = rec
        log.write(json.dumps(rec) + '\n'); log.flush()
        print('EPOCH %d lr %g train %.5f val %.5f (acc %.5f jit %.5f)%s | raw %.5f | %ds' % (ep, a.lr, rec['train_loss'], vl[0], vl[1], vl[2], ' *' if isb else '', raw[0], rec['sec']), flush=True)
    (out / (stem + '_RESULT.json')).write_text(json.dumps(dict(best=bestrec, raw_valA=[float(x) for x in raw[:3]], versions=a.versions), indent=1))
    print('TRAIN_DONE', stem, bestrec['epoch'], bestrec['val_loss'], flush=True)


if __name__ == '__main__':
    main()
