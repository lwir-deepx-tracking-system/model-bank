"""gru_v2/v3/scripts/g3_evidence.py - TRAIN split only: where does the fixed-ratio GT-derived head-to-hip target disagree with the detectors' own keypoints?
Per detector x dataset x detector-box-height bin x box kind (hips inside the box at >= 0.80 of its height = body box; < 0.80 = whole-person box):
n, target h / detector h, hip position in the detector box and in the target box ((mean hip y - top) / h), nose position, share where hips fall below the target bottom.
Writes results/evidence_train.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import g2_lib as L
from g2_lib import P, PL

BINS = ((0, 35), (35, 50), (50, 70), (70, 100), (100, 130), (130, 170), (170, 230), (230, 9999))
acc = {}
for r in [r for r in P.registry() if r['split'] == 'train']:
    S = PL.load_seq(L.ROOT, r); tg = S['a']['tgt_h2h']; gb = S['a']['gbox']
    for k in ('near', 'far'):
        dg = S['a']['dg_' + k]; m = dg >= 0; D = S['d'][k]['box'][m]; K = S['d'][k]['kpt'][m]; T = tg[dg[m]]; Gb = gb[dg[m]]
        ok = np.isfinite(T[:, 0]); D, K, T, Gb = D[ok], K[ok], T[ok], Gb[ok]
        hd = D[:, 3] - D[:, 1]; ht = T[:, 3] - T[:, 1]; hv = (K[:, 11, 2] >= 0.5) & (K[:, 12, 2] >= 0.5); hy = (K[:, 11, 1] + K[:, 12, 1]) / 2
        nv = K[:, 0, 2] >= 0.5
        for a, b in BINS:
            mm = (hd >= a) & (hd < b)
            e = acc.setdefault((k, r['ds'], '%d-%d' % (a, b)), dict(hd=[], ht=[], hv=[], fd=[], ft=[], nd=[], dtop=[], hg=[]))
            e['hd'].append(hd[mm]); e['ht'].append(ht[mm]); e['hv'].append(hv[mm]); e['fd'].append(np.where(hv[mm], (hy[mm] - D[mm, 1]) / hd[mm], np.nan)); e['ft'].append(np.where(hv[mm], (hy[mm] - T[mm, 1]) / ht[mm], np.nan))
            e['nd'].append(np.where(nv[mm], (K[mm, 0, 1] - D[mm, 1]) / hd[mm], np.nan)); e['dtop'].append((T[mm, 1] - D[mm, 1]) / hd[mm]); e['hg'].append(Gb[mm, 3] - Gb[mm, 1])
res = {}
med = lambda v: round(float(np.nanmedian(v)), 3) if len(v) and np.isfinite(v).any() else None
for key, e in sorted(acc.items()):
    e = {k: np.concatenate(v) for k, v in e.items()}
    if not len(e['hd']): continue
    for kind, mk in (('body box (hips >= 0.80)', e['fd'] >= 0.80), ('whole-person box (hips < 0.80)', e['fd'] < 0.80), ('hips not visible', ~e['hv'])):
        n = int(mk.sum())
        if n < 30: continue
        d = dict(n=n, tgt_over_det_h=med(e['ht'][mk] / e['hd'][mk]), det_over_gt_h=med(e['hd'][mk] / e['hg'][mk]), hip_in_det=med(e['fd'][mk]), hip_in_tgt=med(e['ft'][mk]), nose_in_det=med(e['nd'][mk]), tgt_top_minus_det_top=med(e['dtop'][mk]),
                 share_hip_below_tgt=round(float(np.nanmean(e['ft'][mk] > 1.0)), 3) if kind[0] != 'h' else None, share_tgt_short_10pct=round(float(np.nanmean(e['ft'][mk] > 1.05)), 3) if kind[0] != 'h' else None)
        res['|'.join(key) + '|' + kind] = d
        print('%-5s %-4s det h %-9s %-31s n %6d  tgt/det h %s  det/gt h %s  hip in det %s  hip in tgt %s  nose in det %s  top diff %s  hips below tgt bottom %s  (> 1.05: %s)' % (*key, kind, n, d['tgt_over_det_h'], d['det_over_gt_h'], d['hip_in_det'], d['hip_in_tgt'], d['nose_in_det'], d['tgt_top_minus_det_top'], d['share_hip_below_tgt'], d['share_tgt_short_10pct']), flush=True)
(Path(__file__).resolve().parents[1] / 'results' / 'evidence_train.json').write_text(json.dumps(res, indent=1)); print('EVIDENCE_DONE', flush=True)
