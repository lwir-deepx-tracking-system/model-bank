"""pan_test.py (PC): camera-pan test of the camera compensation (PREREG camera_test). TMOT val-B rows (near and far streams, tripod camera) get a synthetic
horizontal pan added in coordinates: every box, keypoint and target moves by p(f), and the frame warp carries the same shift (v(f) = 6 sin(2 pi f / 100) px per frame).
Methods: raw, One Euro with / without the warp compensation (val-A values), learned refiner (its Kalman uses the warp). The tracker is not rerun (its rows are not shifted).
Writes results/pan_test.json.  Usage: python pan_test.py <root>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prcommon as P
import ref_lib as R
import ref_eval as E


def main():
    root = Path(sys.argv[1]); E.init(root); eu = E.G['euro']['comp']; params = (eu['mc'], eu['beta'], eu['mc_s'], eu['beta_s'], 1.0); res = {}
    for pan in (0.0, 6.0):
        rows = {m: {k: [] for k in ('ce', 'along', 'd2c', 'spd', 'ok')} for m in ('raw', 'euro', 'euro_nocomp', 'refiner')}
        for r in [x for x in P.registry() if x['split'] == 'valB' and x['ds'] == 'TMOT']:
            for stream in ('near', 'far'):
                ids, S = R.rows_of(root, r, stream); n = len(S['W']); v = pan * np.sin(2 * np.pi * np.arange(n) / 100.); p = np.cumsum(v)
                W = S['W'].copy(); W[:, 0, 2] += v   # the picture content moves by v(f) between f-1 and f
                for e in ids.values():
                    e = dict(e); sh = p[e['f']]
                    e['z'] = e['z'].copy(); e['z'][:, 0] += sh; e['tgt'] = e['tgt'].copy(); e['tgt'][:, 0] += sh; e['kp'] = e['kp'].copy(); e['kp'][:, :, 0] += sh[:, None]
                    outs = dict(raw=e['z'], euro=R.run_euro(e, W, r['fps'], params, True), euro_nocomp=R.run_euro(e, W, r['fps'], params, False), refiner=R.run_ref(e, W, E.G['ref']))
                    rec = dict(e=e, o=outs, fps=r['fps'])
                    for m in outs:
                        d = E.per_row(rec, m)
                        for k in rows[m]: rows[m][k].append(d[k])
        out = {}
        for m, d in rows.items():
            d = {k: np.concatenate(v) for k, v in d.items()}; ok = d['ok']
            fast = ok & (d['spd'] > 0.03)
            out[m] = dict(n=int(ok.sum()), ce=round(float(np.nanmean(d['ce'][ok])), 4), along_all=round(float(np.nanmean(d['along'][ok])), 4),
                          along_fast=round(float(np.nanmean(d['along'][fast])), 4) if fast.any() else None, ce_fast=round(float(np.nanmean(d['ce'][fast])), 4) if fast.any() else None,
                          d2c_rms=round(float(np.sqrt(np.nanmean(d['d2c'][ok] ** 2))), 4))
        res['pan_%g' % pan] = out; print('PAN', pan, json.dumps(out), flush=True)
    res['note'] = 'TMOT val-B, near + far streams, synthetic horizontal pan 6 sin(2 pi f / 100) px per frame added to boxes, keypoints, targets and the warp; along = signed error along the target motion / height (negative = behind)'
    (root / 'results' / 'pan_test.json').write_text(json.dumps(res, indent=1)); print('PAN_DONE', flush=True)


if __name__ == '__main__':
    main()
