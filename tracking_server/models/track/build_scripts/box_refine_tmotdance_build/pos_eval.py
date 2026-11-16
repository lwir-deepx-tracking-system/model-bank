"""pos_eval.py (PC): position memory on val-B (PREREG position_memory + amendment 1). Plain Kalman path, the CURRENT posetrack18 files (keypoints 1, 2 zeroed
as the server does for 'posetrack' names) and the new files (chosen on val-A), all on the same detector inputs.
m5 score = mean over horizons 5, 10, 15 of mean min(centre error / target height, 2). Groups: all, sim (detector still had the person at t+1), real (no
detection at t+1), syn (fixed-seed synthetic vanish). Target: primary = detector-definition box, secondary = global head-to-hip box.
Per-sequence bootstrap 90 % (2000 draws). Writes results/pos_valB.json.
Usage: python pos_eval.py <root> <new_gru.pt> <new_mamba.pt>
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prcommon as P
import pos_lib as L


def build(args):
    root, nm, syn = args; r = {x['name']: x for x in P.registry()}[nm]; S = L.load_seq(root, r); out = []
    for k in L.DETS:
        T = L.tracks_of(S, k, 'det')
        if syn: A = L.anchors(S, k, T, S['W'], S['n'], rng=np.random.default_rng(12345 + sum(map(ord, nm)) + (0 if k == 'near' else 7)), syn_only=True, vanish_p=1.0)
        else: A = L.anchors(S, k, T, S['W'], S['n'])
        out.append((A, dict(seq=nm, ds=r['ds'], det=k)))
    return out


def m5_boot(ce, seq, nseq, draws, mask):
    """m5 score (mean over horizons 5, 10, 15 of the mean) with a per-sequence bootstrap. ce [N,15]."""
    S = []; C = []
    for h in (5, 10, 15):
        v = ce[:, h - 1]; m = mask & np.isfinite(v)
        S.append(np.bincount(seq[m], weights=v[m], minlength=nseq)); C.append(np.bincount(seq[m], minlength=nseq).astype(float))
    S = np.array(S); C = np.array(C)
    if (C.sum(1) == 0).any(): return None, None
    pt = float(np.mean(S.sum(1) / C.sum(1))); bs = np.mean(S[:, draws].sum(2) / np.maximum(C[:, draws].sum(2), 1e-9), 0)
    return pt, bs


def main():
    import torch
    root = Path(sys.argv[1]); dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.set_num_threads(8)
    reg = [r for r in P.registry() if r['split'] == 'valB']
    with Pool(8) as pool:
        V = [x for xx in pool.map(build, [(str(root), r['name'], False) for r in reg]) for x in xx]
        Sy = [x for xx in pool.map(build, [(str(root), r['name'], True) for r in reg]) for x in xx]
    V = L.stack([x[0] for x in V], [x[1] for x in V]); Sy = L.stack([x[0] for x in Sy], [x[1] for x in Sy])
    print('val-B anchors %d syn %d' % (len(V['h0']), len(Sy['h0'])), flush=True)
    nets = {}
    for arm in ('gru_kpt', 'mamba_kpt'):
        n = L.make_net(arm).to(dev); L.load_npz_into(n, root / 'export' / 'ref' / ('pos_%s_posetrack18_20261114.npz' % arm)); nets['current_' + arm] = (n, True)
        f_ = sys.argv[2] if arm == 'gru_kpt' else sys.argv[3]
        if f_ == '-': continue   # this arm's new file is not ready: only Kalman and the current file
        ck = torch.load(f_, map_location='cpu'); n2 = L.make_net(arm); n2.load_state_dict(ck['model']); nets['new_' + arm] = (n2.to(dev), False)
    preds = {'kalman': (V['base'], Sy['base'])}
    for k, (n, z12) in nets.items():
        preds[k] = (L.predict(n, V, dev, zero_kpt12=z12), L.predict(n, Sy, dev, zero_kpt12=z12)); print('predicted', k, flush=True)
    res = dict(anchors=len(V['h0']), syn_anchors=len(Sy['h0']), tables={}, beats={})
    for tname, tk in (('detector_box', 'tgt'), ('head_to_hip', 'tgt2')):
        ce = {m: (L.ce_matrix(p[0], V[tk]), L.ce_matrix(p[1], Sy[tk])) for m, p in preds.items()}
        for ds in ('TMOT', 'DT'):
            for det in ('near', 'far', 'pooled'):
                selV = (V['ds'] == ds) & ((V['det'] == det) if det != 'pooled' else True); selS = (Sy['ds'] == ds) & ((Sy['det'] == det) if det != 'pooled' else True)
                names = sorted(set(V['seq'][selV])); loc = {n: i for i, n in enumerate(names)}; ns = len(names)
                if not ns: continue
                draws = np.random.default_rng(0).integers(0, ns, (2000, ns))
                sqV = np.array([loc.get(s, 0) for s in V['seq']]); sqS = np.array([loc.get(s, 0) for s in Sy['seq']])
                hgV = V['hg']; groups = dict(all=selV, sim=selV & ~V['real'].astype(bool), real=selV & V['real'].astype(bool))
                for b, (lo, hi) in zip(P.SIZE_NAMES, P.SIZE_BINS): groups['all|' + b] = selV & (hgV >= lo) & (hgV < hi)
                tab = {}
                for g, mask in list(groups.items()) + [('syn', None)]:
                    row = {}; bsd = {}
                    for m in preds:
                        if g == 'syn': pt, bs = m5_boot(ce[m][1], sqS, ns, draws, selS)
                        else: pt, bs = m5_boot(ce[m][0], sqV, ns, draws, mask)
                        if pt is None: continue
                        row[m] = [round(pt, 4), round(float(np.percentile(bs, 5)), 4), round(float(np.percentile(bs, 95)), 4)]; bsd[m] = (pt, bs)
                    n_ = int((selS if g == 'syn' else mask).sum())
                    diffs = {}
                    for a_, b_ in (('new_gru_kpt', 'current_gru_kpt'), ('new_mamba_kpt', 'current_mamba_kpt'), ('new_gru_kpt', 'kalman'), ('new_mamba_kpt', 'kalman'),
                                   ('current_gru_kpt', 'kalman'), ('current_mamba_kpt', 'kalman')):
                        if a_ in bsd and b_ in bsd:
                            d = bsd[a_][1] - bsd[b_][1]; diffs['%s - %s' % (a_, b_)] = [round(bsd[a_][0] - bsd[b_][0], 4), round(float(np.percentile(d, 5)), 4), round(float(np.percentile(d, 95)), 4)]
                    tab[g] = dict(n=n_, score=row, diff=diffs)
                res['tables']['%s|%s|%s' % (tname, ds, det)] = dict(sequences=names, groups=tab)
                print(tname, ds, det, {g: (v['n'], v['score'].get('kalman', [None])[0], v['score'].get('current_gru_kpt', [None])[0], v['score'].get('new_gru_kpt', [None])[0]) for g, v in tab.items() if '|' not in g}, flush=True)
    for arm in [a_ for a_ in ('gru_kpt', 'mamba_kpt') if 'new_' + a_ in preds]:
        ok = []
        for ds in ('TMOT', 'DT'):
            d = res['tables']['detector_box|%s|pooled' % ds]['groups']['all']['diff'].get('new_%s - current_%s' % (arm, arm))
            ok.append(d is not None and d[0] < 0 and d[2] < 0)
        res['beats'][arm] = dict(beats_current=all(ok), rule='val-B all-anchor score lower with the 90 % interval of (new - current) below 0 in both TMOT and DanceTrack (pooled detectors, detector-box target)')
    name = 'pos_valB.json' if '-' not in sys.argv[2:4] else 'pos_valB_partial.json'
    (root / 'results' / name).write_text(json.dumps(res, indent=1)); print('POS_VALB_DONE', json.dumps(res['beats']), flush=True)


if __name__ == '__main__':
    main()
