"""gru_v2/scripts/g2_lutcheck.py - TRAIN split only: the detector box vs the head-to-hip target per dataset, detector and detector-box height row of the lookup
(the pairs verify/gen_outputs.fit_lut pooled), with two independent signs of what the box is: the ground-truth box touching the picture edge (a cut person: the
fixed head-to-hip share of a cut box is too small) and where the detector's own hip keypoints sit inside the detector box and inside the target.
Writes results/lutcheck_train.json. Reads only round-1 data.   Usage: python g2_lutcheck.py
"""
import json

import numpy as np

import g2_lib as L
from g2_lib import P, PL

ROWS = np.log([1, 25, 35, 45, 55, 65, 80, 100, 130, 170, 230, 1e5])


def main():
    reg = [r for r in P.registry() if r['split'] == 'train']; acc = {}
    for r in reg:
        S = PL.load_seq(L.ROOT, r); tg = P.xyxy2cxcywh(S['a']['tgt_h2h']); gb = S['a']['gbox']
        for k in ('near', 'far'):
            dg = S['a']['dg_' + k]; m = dg >= 0; Z = P.xyxy2cxcywh(S['d'][k]['box'][m]); T = tg[dg[m]]; G = gb[dg[m]]; kp = S['d'][k]['kpt'][m]; ok = np.isfinite(T[:, 0])
            Z, T, G, kp = Z[ok], T[ok], G[ok], kp[ok]
            hv = (kp[:, 11, 2] >= 0.5) & (kp[:, 12, 2] >= 0.5); hy = (kp[:, 11, 1] + kp[:, 12, 1]) / 2
            d = acc.setdefault((r['ds'], k), dict(lh=[], ar=[], dlh=[], dlw=[], dy=[], cut=[], cutb=[], hip_det=[], hip_tgt=[], hg=[]))
            d['lh'].append(np.log(np.maximum(Z[:, 3], 1.))); d['ar'].append(Z[:, 2] / Z[:, 3]); d['dlh'].append(np.log(T[:, 3] / Z[:, 3])); d['dlw'].append(np.log(T[:, 2] / Z[:, 2])); d['dy'].append((T[:, 1] - Z[:, 1]) / Z[:, 3])
            d['cut'].append(P.touches_edge(G)); d['cutb'].append(G[:, 3] > P.Y1 - 2.0); d['hg'].append(G[:, 3] - G[:, 1])
            d['hip_det'].append(np.where(hv, (hy - (Z[:, 1] - Z[:, 3] / 2)) / Z[:, 3], np.nan)); d['hip_tgt'].append(np.where(hv, (hy - (T[:, 1] - T[:, 3] / 2)) / T[:, 3], np.nan))
    res = {}; med = lambda v: round(float(np.nanmedian(v)), 3) if len(v) and np.isfinite(v).any() else None
    for (ds, k), d in sorted(acc.items()):
        d = {a: np.concatenate(b) for a, b in d.items()}; bi = np.clip(np.searchsorted(ROWS, d['lh'], 'right') - 1, 0, len(ROWS) - 2); out = []
        for i in range(len(ROWS) - 1):
            m = bi == i
            if m.sum() < 30: continue
            for nm, mm in (('all', m), ('whole', m & ~d['cut']), ('cut', m & d['cut'])):
                if mm.sum() < 20: continue
                out.append(dict(det_h='%g-%g' % (np.exp(ROWS[i]), min(np.exp(ROWS[i + 1]), 999)), part=nm, n=int(mm.sum()), cut_share=round(float(d['cut'][mm].mean()), 3), cut_bottom=round(float(d['cutb'][mm].mean()), 3),
                                gt_h=med(d['hg'][mm]), ar=med(d['ar'][mm]), h_ratio=round(float(np.exp(np.median(d['dlh'][mm]))), 3), h_ratio_q25=round(float(np.exp(np.percentile(d['dlh'][mm], 25))), 3),
                                h_ratio_q75=round(float(np.exp(np.percentile(d['dlh'][mm], 75))), 3), w_ratio=round(float(np.exp(np.median(d['dlw'][mm]))), 3), dy=med(d['dy'][mm]),
                                hip_in_det=med(d['hip_det'][mm]), hip_in_target=med(d['hip_tgt'][mm]), n_hip=int(np.isfinite(d['hip_det'][mm]).sum())))
                o = out[-1]; print('%-4s %-4s det h %-8s %-5s n %6d cut %.2f (bottom %.2f) gt h %5.0f ar %.2f  target/det h %.2f [%.2f, %.2f] w %.2f dy %+.2f  hip in det %s in target %s (n %d)' % (
                    ds, k, o['det_h'], nm, o['n'], o['cut_share'], o['cut_bottom'], o['gt_h'], o['ar'], o['h_ratio'], o['h_ratio_q25'], o['h_ratio_q75'], o['w_ratio'], o['dy'], o['hip_in_det'], o['hip_in_target'], o['n_hip']), flush=True)
        res['%s_%s' % (ds, k)] = out
    (L.G2 / 'results' / 'lutcheck_train.json').write_text(json.dumps(res, indent=1)); print('LUTCHECK_DONE', flush=True)


if __name__ == '__main__':
    main()
