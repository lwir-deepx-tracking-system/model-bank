"""verify/gen_outputs.py - CHECK B: regenerate the per-row outputs of every method on val-B (and val-A for tuning checks), two ways:
  (1) the reviewed offline code (scripts/ref_lib.run_euro / run_ref), exactly what ref_eval.py used;
  (2) the SERVER module src/track/box_refine.py (worktree), driven frame by frame (Refiner('gru'), Refiner('euro')), as an independent code path.
Also builds a 'conversion only' baseline: a lookup table fitted on the TRAIN split (detector box height x aspect ratio bins -> median
target/detector relation, both datasets and both detectors pooled, like the refiner), applied to the raw box, and One Euro on top of it.
Writes verify/preds/<split>_<name>_<stream>.pkl  (per id: f, z, trk, tgt, hg, src, kp vis count, outputs).
Usage: python gen_outputs.py <root> [valB|valA]
Reads only; writes only into verify/.
"""
import json
import math
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import prcommon as P   # noqa: E402
import ref_lib as R    # noqa: E402
import pos_lib as PL   # noqa: E402

sys.path.insert(0, P.PATHS['worktree'])
OUT = ROOT / 'verify' / 'preds'
G = {}

# ---------------------------------------------------------------- conversion lookup table (train split only)
LH_EDGES = np.log([1, 25, 35, 45, 55, 65, 80, 100, 130, 170, 230, 1e5])
AR_EDGES = np.array([0, 0.30, 0.36, 0.42, 0.48, 0.55, 0.65, 0.80, 99])


def lut_bins(z):
    lh = np.log(np.maximum(z[:, 3], 1.)); ar = z[:, 2] / np.maximum(z[:, 3], 1.)
    return np.clip(np.searchsorted(LH_EDGES, lh, 'right') - 1, 0, len(LH_EDGES) - 2), np.clip(np.searchsorted(AR_EDGES, ar, 'right') - 1, 0, len(AR_EDGES) - 2)


def fit_lut(root):
    reg = [r for r in P.registry() if r['split'] == 'train']; Z = []; T = []
    for r in reg:
        S = PL.load_seq(root, r); tg = P.xyxy2cxcywh(S['a']['tgt_h2h'])
        for k in ('near', 'far'):
            dg = S['a']['dg_' + k]; m = dg >= 0
            Z.append(P.xyxy2cxcywh(S['d'][k]['box'][m])); T.append(tg[dg[m]])
    Z = np.concatenate(Z); T = np.concatenate(T); ok = np.isfinite(T[:, 0]); Z = Z[ok]; T = T[ok]
    e = np.stack([(T[:, 0] - Z[:, 0]) / Z[:, 3], (T[:, 1] - Z[:, 1]) / Z[:, 3], np.log(T[:, 2] / Z[:, 2]), np.log(T[:, 3] / Z[:, 3])], 1)
    bh, ba = lut_bins(Z); tab = np.zeros((len(LH_EDGES) - 1, len(AR_EDGES) - 1, 4)); cnt = np.zeros(tab.shape[:2], int)
    glob = np.median(e, 0)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            m = (bh == i) & (ba == j); cnt[i, j] = m.sum()
            if m.sum() >= 30: tab[i, j] = np.median(e[m], 0)
            else:   # sparse cell: nearest populated aspect cell in the same height row, else global
                tab[i, j] = np.nan
    for i in range(tab.shape[0]):
        good = [j for j in range(tab.shape[1]) if np.isfinite(tab[i, j, 0])]
        for j in range(tab.shape[1]):
            if not np.isfinite(tab[i, j, 0]): tab[i, j] = tab[i, min(good, key=lambda g: abs(g - j))] if good else glob
    return tab, cnt


def apply_lut(z, tab):
    bh, ba = lut_bins(z); e = tab[bh, ba]
    return np.stack([z[:, 0] + e[:, 0] * z[:, 3], z[:, 1] + e[:, 1] * z[:, 3], z[:, 2] * np.exp(e[:, 2]), z[:, 3] * np.exp(e[:, 3])], 1)


# ---------------------------------------------------------------- server module driven frame by frame
def server_run(mode, ids, W, fps, n, model):
    from src.track import box_refine as BRm
    ref = BRm.Refiner(mode, model, BRm.EURO, 0.0); out = {i: np.full((len(e['f']), 4), np.nan) for i, e in ids.items()}
    byf = {}
    for i, e in ids.items():
        for j, f in enumerate(e['f']): byf.setdefault(int(f), []).append((i, j))
    for f in range(n):
        items = []; keys = []
        for i, j in byf.get(f, []):
            e = ids[i]; items.append((i, e['z'][j].astype(np.float64), e['kp'][j].astype(np.float64), e['trk'][j])); keys.append((i, j))
        H = W[f] if f < len(W) else None
        res = ref.step(items, f / fps, H, None, True)
        for (i, j) in keys: out[i][j] = res[i]
    return out


def init(root, split):
    G['root'] = Path(root); G['split'] = split
    G['euro'] = json.loads((G['root'] / 'results' / 'euro_valA.json').read_text())
    f = sorted((G['root'] / 'export').glob('box_refine_gru_*.npz'))[-1]; z = np.load(f); w = {k: z[k] for k in z.files}
    G['ref'] = R.RefNumpy(w)
    from src.track import box_refine as BRm
    G['srv'] = BRm.GRUNp(w)
    G['lut'] = pickle.loads((G['root'] / 'verify' / 'lut.pkl').read_bytes())['tab']


def job(args):
    name, stream = args; r = {x['name']: x for x in P.registry()}[name]
    ids, S = R.rows_of(G['root'], r, stream); W = S['W']; fps = r['fps']; n = S['n']
    eu = G['euro']['comp']; pe = (eu['mc'], eu['beta'], eu['mc_s'], eu['beta_s'], 1.0)
    srv_gru = server_run('gru', ids, W, fps, n, G['srv']); srv_euro = server_run('euro', ids, W, fps, n, None)
    recs = []
    for i, e in ids.items():
        o = dict(raw=e['z'].copy(), tracker=e['trk'].copy(), euro=R.run_euro(e, W, fps, pe, True), refiner=R.run_ref(e, W, G['ref']),
                 srv_refiner=srv_gru[i], srv_euro=srv_euro[i])
        conv = apply_lut(e['z'], G['lut']); o['conv'] = conv
        e2 = dict(e); e2['z'] = conv; o['conv_euro'] = R.run_euro(e2, W, fps, pe, True)
        vis = (e['kp'][:, :, 2] >= 0.5).sum(1)
        recs.append(dict(name=name, ds=r['ds'], stream=stream, id=i, fps=fps, f=e['f'], z=e['z'], trk=e['trk'], tgt=e['tgt'], hg=e['hg'], src=e['src'],
                         vis=vis, kp=e['kp'], score=e['score'], o=o))
    (OUT / ('%s_%s_%s.pkl' % (G['split'], name, stream))).write_bytes(pickle.dumps(recs, protocol=4))
    return name, stream, sum(len(x['f']) for x in recs)


def main():
    root = ROOT; split = sys.argv[2] if len(sys.argv) > 2 else 'valB'; OUT.mkdir(parents=True, exist_ok=True)
    lp = root / 'verify' / 'lut.pkl'
    if not lp.exists():
        tab, cnt = fit_lut(root); lp.write_bytes(pickle.dumps(dict(tab=tab, cnt=cnt, lh_edges=LH_EDGES, ar_edges=AR_EDGES), protocol=4))
        print('LUT fitted on train: cells with >=30 pairs %d / %d' % ((cnt >= 30).sum(), cnt.size), flush=True)
        np.set_printoptions(precision=3, suppress=True, linewidth=200); print('h ratio (target/det) per height row x aspect col:\n', np.exp(tab[:, :, 3])); print('counts\n', cnt)
    reg = [r for r in P.registry() if r['split'] == split]
    jobs = [(r['name'], s) for r in reg for s in ('near', 'far', 'sw')]
    with Pool(8, initializer=init, initargs=(str(root), split)) as pool:
        for name, stream, nrow in pool.imap_unordered(job, jobs):
            print('DONE', split, name, stream, nrow, flush=True)
    print('GEN_DONE', flush=True)


if __name__ == '__main__':
    main()
