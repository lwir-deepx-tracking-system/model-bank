"""assoc_build.py (PC): detections -> gt ids, the body-box targets and the fit report (PREREG amendment 1).
Inputs: data/det/<near|far>_<folder>.npz (det_run.py), gt files. Outputs: data/assoc/<name>.npz, results/assoc_report.json.
Usage: python assoc_build.py <pos_refine root>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prcommon as P

DETS = ('near', 'far')
HBINS = (0, 35, 50, 70, 100, 140, 200, 1e9)


def med(a): return float(np.median(a)) if len(a) else float('nan')


def main():
    root = Path(sys.argv[1]); (root / 'data' / 'assoc').mkdir(parents=True, exist_ok=True)
    reg = P.registry(); S = {}
    for r in reg:
        gfr, gid, gbox = P.load_gt(r['gt'])
        d = {k: P.load_det(root / 'data' / 'det' / ('%s_%s.npz' % (k, r['folder']))) for k in DETS}
        assert int(gfr.max()) < d['near']['n'], (r['name'], int(gfr.max()), d['near']['n'])
        S[r['name']] = dict(r=r, gfr=gfr, gid=gid, gbox=gbox, det=d)
    # ---- pass 0 (prior) -> global head-to-hip relation per dataset from TRAIN pairs (both detectors, gt >= 100 px, head-to-hip regime, no picture edge)
    rels = {}; wmode = {}
    for ds in ('TMOT', 'DT'):
        R = []
        for s in S.values():
            if s['r']['ds'] != ds or s['r']['split'] != 'train': continue
            for k in DETS:
                dg, _ = P.associate(s['gfr'], s['gbox'], s['det'][k]['fi'], s['det'][k]['box'], P.PRIOR, 'w')
                m = dg >= 0; D = s['det'][k]['box'][m]; G = s['gbox'][dg[m]]
                hg = G[:, 3] - G[:, 1]; ratio = (D[:, 3] - D[:, 1]) / hg
                ok = (hg >= 100) & (ratio >= 0.40) & (ratio <= 0.75) & ~P.touches_edge(D) & ~P.touches_edge(G)
                R.append(P.rel_of(D[ok], G[ok]))
        R = np.concatenate(R); mad = lambda a: float(np.median(np.abs(a - np.median(a))))
        wm = 'w' if mad(R[:, 2]) <= mad(R[:, 3]) else 'h'
        rels[ds] = [med(R[:, 0]), med(R[:, 1]), med(R[:, 2] if wm == 'w' else R[:, 3]), med(R[:, 4])]; wmode[ds] = wm
        print('GLOBAL %s pairs %d wmode %s (MAD w %.3f h %.3f) rel dx %.3f dtop %.3f lw %.3f lh %.3f (h ratio %.3f)' % (ds, len(R), wm, mad(R[:, 2]), mad(R[:, 3]), *rels[ds], np.exp(rels[ds][3])), flush=True)
    # ---- pass 1 with the global relation: final association
    for s in S.values():
        ds = s['r']['ds']
        for k in DETS:
            s['dg_' + k], s['ds_' + k] = P.associate(s['gfr'], s['gbox'], s['det'][k]['fi'], s['det'][k]['box'], rels[ds], wmode[ds])
    # ---- detector-definition relation (position-memory target) per dataset x detector, by gt-height bin, TRAIN pairs without picture edge
    drel = {}
    for ds in ('TMOT', 'DT'):
        for k in DETS:
            R = []; HG = []
            for s in S.values():
                if s['r']['ds'] != ds or s['r']['split'] != 'train': continue
                dg = s['dg_' + k]; m = dg >= 0; D = s['det'][k]['box'][m]; G = s['gbox'][dg[m]]; ok = ~P.touches_edge(D) & ~P.touches_edge(G)
                R.append(P.rel_of(D[ok], G[ok])); HG.append((G[ok, 3] - G[ok, 1]))
            R = np.concatenate(R); HG = np.concatenate(HG); bins = []
            for a, b in zip(HBINS[:-1], HBINS[1:]):
                m = (HG >= a) & (HG < b)
                if m.sum() < 20: continue
                rr = R[m]; bins.append(dict(lo=a, hi=b, n=int(m.sum()), hc=float(np.exp(np.median(np.log(HG[m])))),
                                            rel=[med(rr[:, 0]), med(rr[:, 1]), med(rr[:, 2] if wmode[ds] == 'w' else rr[:, 3]), med(rr[:, 4])]))
            drel['%s_%s' % (ds, k)] = bins
            print('DETREL %s %s' % (ds, k), ' | '.join('%g-%g n%d h%.2f' % (b['lo'], b['hi'] if b['hi'] < 1e8 else 999, b['n'], np.exp(b['rel'][3])) for b in bins), flush=True)

    def det_rel(ds, k, hg):
        bins = drel['%s_%s' % (ds, k)]; xc = np.log([b['hc'] for b in bins]); Rb = np.array([b['rel'] for b in bins]); x = np.log(np.maximum(hg, 1.))
        return np.stack([np.interp(x, xc, Rb[:, j]) for j in range(4)], 1)

    # ---- targets per gt row, per-track relation (secondary), save
    rep = defaultdict(lambda: defaultdict(list)); mrate = defaultdict(lambda: [0, 0]); dshare = defaultdict(lambda: [0, 0])
    for name, s in S.items():
        ds = s['r']['ds']; gb = s['gbox']; hg = gb[:, 3] - gb[:, 1]
        out = dict(gfr=s['gfr'], gid=s['gid'], gbox=gb, tgt_h2h=P.apply_rel(gb, rels[ds], wmode[ds]))
        for k in DETS:
            out['tgt_det_' + k] = P.apply_rel(gb, det_rel(ds, k, hg), wmode[ds])
            dg = s['dg_' + k]; m = dg >= 0; trk = np.full((len(gb), 4), np.nan)
            D = s['det'][k]['box']; byid = defaultdict(list)
            for i in np.flatnonzero(m):
                if not (P.touches_edge(D[i:i + 1])[0] or P.touches_edge(gb[dg[i]:dg[i] + 1])[0]): byid[int(s['gid'][dg[i]])].append(i)
            for g_, ii in byid.items():
                if len(ii) < 5: continue
                rr = P.rel_of(D[ii], gb[dg[ii]]); rel = [med(rr[:, 0]), med(rr[:, 1]), med(rr[:, 2] if wmode[ds] == 'w' else rr[:, 3]), med(rr[:, 4])]
                rows = s['gid'] == g_; trk[rows] = P.apply_rel(gb[rows], rel, wmode[ds])
            out['tgt_trk_' + k] = trk; out['dg_' + k] = dg; out['ds_' + k] = s['ds_' + k]
            # report: match rate, share of confident detections associated, fit of each target where both exist
            key = '%s|%s|%s' % (ds, k, s['r']['split'])
            big = hg >= 24; got = np.zeros(len(gb), bool); got[dg[m]] = True
            mrate[key][0] += int((got & big).sum()); mrate[key][1] += int(big.sum())
            hi = s['det'][k]['score'] >= 0.25; dshare[key][0] += int((m & hi).sum()); dshare[key][1] += int(hi.sum())
            Dm = D[m]; gi = dg[m]
            for tname in ('h2h', 'det', 'trk'):
                T = out['tgt_h2h'] if tname == 'h2h' else out['tgt_%s_%s' % (tname, k)]
                Tm = T[gi]; ok = np.isfinite(Tm[:, 0]); iou = P.iou_pair(Dm[ok], Tm[ok]); th = Tm[ok, 3] - Tm[ok, 1]
                ce = np.hypot((Dm[ok, 0] + Dm[ok, 2] - Tm[ok, 0] - Tm[ok, 2]) / 2, (Dm[ok, 1] + Dm[ok, 3] - Tm[ok, 1] - Tm[ok, 3]) / 2) / th
                hr = (Dm[ok, 3] - Dm[ok, 1]) / th; sb = np.array([P.size_bin(h) for h in hg[gi][ok]])
                for b in range(-1, 3):
                    mm = np.ones(len(iou), bool) if b < 0 else sb == b
                    rep[key + '|' + tname + '|' + ('all' if b < 0 else P.SIZE_NAMES[b])]['iou'].append(iou[mm])
                    rep[key + '|' + tname + '|' + ('all' if b < 0 else P.SIZE_NAMES[b])]['ce'].append(ce[mm])
                    rep[key + '|' + tname + '|' + ('all' if b < 0 else P.SIZE_NAMES[b])]['hr'].append(hr[mm])
        np.savez_compressed(root / 'data' / 'assoc' / ('%s.npz' % name), **out)
    fit = {}
    for kk, v in rep.items():
        iou = np.concatenate(v['iou']); ce = np.concatenate(v['ce']); hr = np.concatenate(v['hr'])
        if not len(iou): continue
        fit[kk] = dict(n=int(len(iou)), iou_median=round(med(iou), 3), iou_ge_05=round(float((iou >= 0.5).mean()), 3), centre_err_median=round(med(ce), 3),
                       h_det_over_target_median=round(med(hr), 3), h_ratio_iqr=[round(float(np.percentile(hr, 25)), 3), round(float(np.percentile(hr, 75)), 3)])
    res = dict(global_relation={ds: dict(dx=rels[ds][0], dtop=rels[ds][1], log_w=rels[ds][2], log_h=rels[ds][3], h_ratio=float(np.exp(rels[ds][3])),
                                          w_ratio=float(np.exp(rels[ds][2])), width_vs=('gt width' if wmode[ds] == 'w' else 'gt height')) for ds in rels},
               detector_relation=drel, match_rate={k: dict(matched=v[0], gt=v[1], rate=round(v[0] / max(v[1], 1), 3)) for k, v in sorted(mrate.items())},
               det_share={k: dict(associated=v[0], dets_ge_025=v[1], share=round(v[0] / max(v[1], 1), 3)) for k, v in sorted(dshare.items())}, fit=fit,
               note='match rate = gt boxes >= 24 px with an associated detection (score > 0.10); fit = target box vs the associated detection; h2h = global head-to-hip target (refiner), det = detector-definition target (position memory), trk = per-track median relation (secondary)')
    (root / 'results' / 'assoc_report.json').write_text(json.dumps(res, indent=1))
    for k, v in sorted(res['match_rate'].items()): print('MATCH', k, v, res['det_share'][k], flush=True)
    for k in sorted(fit):
        if k.endswith('|all'): print('FIT', k, fit[k], flush=True)
    print('ASSOC_DONE', flush=True)


if __name__ == '__main__':
    main()
