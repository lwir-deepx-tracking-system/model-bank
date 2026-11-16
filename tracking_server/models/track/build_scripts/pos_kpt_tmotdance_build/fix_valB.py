"""fix_valB.py (= gru_mismatch_20261116 pf_valB.py) - val-B (a) and (b): the old file (pos_gru_kpt_tmotdance_20261115) vs the retrained file on val-B.
(a) each file on its own builder's anchors and targets (the old one = the published numbers).
(b) the anchors both builders give (same person, same last seen frame), the targets of the fixed builder (no target where the gt box is cut by the
    picture) for both, each file with the inputs of its own builder (old: device rows for every dataset + old warp order; new: real picture rows +
    fixed warp order); Kalman = the fixed builder's plain path. Per-sequence bootstrap 90 % (2000 draws, seed 0) of new - old, pooled detectors.
Pass (pre-registered 2026-11-16): new <= old + 0.005 in (b) for TMOT and for DanceTrack.
Usage: python fix_valB.py <old npz> <new npz> <out json> [--no-edges]   (--no-edges: the new file was trained without the picture-edge inputs)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fix_lib as F   # noqa: E402
P = F.P; L = F.L


class G:
    """the helpers of gru_mismatch_20261116 gm_lib.py this script uses"""
    @staticmethod
    def net_of(npz, arm='gru_kpt'):
        net = L.make_net(arm).double(); L.load_npz_into(net, npz); net.eval(); return net

    @staticmethod
    def run(net, D, bs=4096):
        import torch
        out = []
        with torch.no_grad():
            for a in range(0, len(D['h0']), bs):
                f = lambda v: torch.as_tensor(np.asarray(v[a:a + bs], np.float64))
                out.append(net(f(D['hx']), f(D['hm']), f(D['xl']), f(D['base']), f(D['h0'])).numpy())
        return np.concatenate(out)

    @staticmethod
    def err_valB(q, D, sel):
        ce = L.ce_matrix(q, D['tgt'])
        return float(np.mean([np.nanmean(np.where(sel, ce[:, h - 1], np.nan)) for h in (5, 10, 15)]))

    @staticmethod
    def step2(q, base, h0, sel):
        f = lambda v: round(float(np.nanmedian(v)), 3) if len(v) else None
        return dict(n=int(sel.sum()), h2=f(q[sel, 1, 3] / base[sel, 1, 3]), dy2=f((q[sel, 1, 1] - base[sel, 1, 1]) / h0[sel]),
                    h15=f(q[sel, 14, 3] / base[sel, 14, 3]), dy15=f((q[sel, 14, 1] - base[sel, 14, 1]) / h0[sel]))


def build(args):
    nm, edges = args; F.EDGES[0] = edges
    r = {x['name']: x for x in P.registry()}[nm]; S = L.load_seq(F.PR, r); out = []
    for k in F.DETS:
        A0 = L.anchors(S, k, L.tracks_of(S, k, 'det'), S['W'], S['n']); A1 = F.anchors(S, k, F.tracks_of(S, k), S['W'], S['n'])
        out.append((nm, r['ds'], k, A0, A1))
    return out


def arr(A, keys):
    return {k: np.asarray(A[k], np.float32 if k in ('hx', 'xl') else np.float64) for k in keys}


def m5_seq(ce, seq, nseq, draws, mask):
    S = []; C = []
    for h in (5, 10, 15):
        v = ce[:, h - 1]; m = mask & np.isfinite(v)
        S.append(np.bincount(seq[m], weights=v[m], minlength=nseq)); C.append(np.bincount(seq[m], minlength=nseq).astype(float))
    S = np.array(S); C = np.array(C)
    return float(np.mean(S.sum(1) / C.sum(1))), np.mean(S[:, draws].sum(2) / np.maximum(C[:, draws].sum(2), 1e-9), 0)


def main():
    import torch
    torch.set_num_threads(6)
    old = G.net_of(Path(sys.argv[1])); new = G.net_of(Path(sys.argv[2])); res = dict(old=Path(sys.argv[1]).name, new=Path(sys.argv[2]).name, new_edges='--no-edges' not in sys.argv, a={}, b={})
    reg = [r['name'] for r in P.registry() if r['split'] == 'valB']
    edges = '--no-edges' not in sys.argv
    with Pool(6) as pool: R = [x for xx in pool.map(build, [(nm, edges) for nm in reg]) for x in xx]
    keys = ('hx', 'hm', 'xl', 'base', 'h0', 'tgt')
    rows0 = []; rows1 = []; rowsC0 = []; rowsC1 = []
    for nm, ds, k, A0, A1 in R:
        if not len(A0['h0']) and not len(A1['h0']): continue
        rows0.append((nm, ds, A0)); rows1.append((nm, ds, A1))
        i0 = {(g, t): i for i, (g, t) in enumerate(zip(A0['gid'], A0['t']))}; i1 = {(g, t): i for i, (g, t) in enumerate(zip(A1['gid'], A1['t']))}
        com = sorted(set(i0) & set(i1)); rowsC0.append((nm, ds, A0, [i0[c] for c in com])); rowsC1.append((nm, ds, A1, [i1[c] for c in com]))

    def cat(rows, sel=False):
        D = {k: [] for k in keys}; seq = []; dsl = []
        for row in rows:
            nm, ds, A = row[:3]; idx = row[3] if sel else None
            a = arr(A, keys)
            for k in keys: D[k].append(a[k][idx] if sel else a[k])
            nn = len(idx) if sel else len(A['h0']); seq += [nm] * nn; dsl += [ds] * nn
        D = {k: np.concatenate(v) for k, v in D.items()}; D['seq'] = np.array(seq); D['ds'] = np.array(dsl); return D
    D0 = cat(rows0); D1 = cat(rows1); C0 = cat(rowsC0, True); C1 = cat(rowsC1, True)
    q0 = G.run(old, D0); q1 = G.run(new, D1); qc0 = G.run(old, C0); qc1 = G.run(new, C1)
    for ds in ('TMOT', 'DT'):
        s0 = D0['ds'] == ds; s1 = D1['ds'] == ds
        res['a'][ds] = dict(old=G.err_valB(q0, D0, s0), kalman_old_builder=G.err_valB(D0['base'], D0, s0), anchors_old=int(s0.sum()),
                            new=G.err_valB(q1, D1, s1), kalman_new_builder=G.err_valB(D1['base'], D1, s1), anchors_new=int(s1.sum()))
        s = C1['ds'] == ds; names = sorted(set(C1['seq'][s])); loc = {n_: i for i, n_ in enumerate(names)}; sq = np.array([loc.get(x, 0) for x in C1['seq']])
        draws = np.random.default_rng(0).integers(0, len(names), (2000, len(names)))
        tg = C1['tgt']; ce_old = L.ce_matrix(qc0, tg); ce_new = L.ce_matrix(qc1, tg); ce_k = L.ce_matrix(C1['base'], tg)
        po, bo = m5_seq(ce_old, sq, len(names), draws, s); pn, bn = m5_seq(ce_new, sq, len(names), draws, s); pk, bk = m5_seq(ce_k, sq, len(names), draws, s)
        d = bn - bo; dk = bn - bk
        real = None
        res['b'][ds] = dict(anchors=int(s.sum()), sequences=len(names), old=po, new=pn, kalman=pk, new_minus_old=[pn - po, float(np.percentile(d, 5)), float(np.percentile(d, 95))],
                            new_minus_kalman=[pn - pk, float(np.percentile(dk, 5)), float(np.percentile(dk, 95))], passes=bool(pn <= po + 0.005),
                            step2_old=G.step2(qc0, C0['base'], C0['h0'], s), step2_new=G.step2(qc1, C1['base'], C1['h0'], s))
        print(ds, 'a', json.dumps(res['a'][ds]), flush=True); print(ds, 'b', json.dumps(res['b'][ds]), flush=True)
    res['pass'] = all(res['b'][ds]['passes'] for ds in ('TMOT', 'DT'))
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1)); print('PF_VALB_DONE pass', res['pass'], flush=True)


if __name__ == '__main__':
    main()
