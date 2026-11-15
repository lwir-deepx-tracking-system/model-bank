"""body_v2/scripts/b2_plain_check.py - TRAIN split, report only: the whole-person -> body constant (fitted on dataset GT boxes) against the boxes of the
plain detector itself (YOLO26x FLIR fp32 on the PC, b2_yolo.py imgs on the frame sample data/train_sample.tsv).
Per sample frame: YOLO boxes (score >= 0.25) and our pose detections that are body boxes by their own hips (both hips >= 0.5, hip line 0.90-1.00 of
the box), each paired with the dataset GT box (IoU >= 0.5 for YOLO; the association of pos_refine for the pose detections). For GT people with both:
body box relative to the YOLO box (top offset / h, height / h, width / h) and the error of the constant applied to the YOLO box.
Writes results/plain_check.json.   Usage: python b2_plain_check.py
"""
import json

import numpy as np

import b2_lib as B
from b2_lib import P, PL


def main():
    tbl = np.load(B.B2 / 'data' / 'table_v2.npz', allow_pickle=False); to, kh, kw = (float(v) for v in tbl['person'])
    y = np.load(B.B2 / 'data' / 'yolo_train_sample.npz', allow_pickle=True); rows = [l.split('\t') for l in (B.B2 / 'data' / 'train_sample.tsv').read_text(encoding='utf-8').splitlines() if l.strip()]
    reg = {r['name']: r for r in P.registry()}; cache = {}; out = []; n_y = 0; n_ym = 0
    for k, (name, folder, fi, path) in enumerate(rows):
        fi = int(fi)
        if name not in cache: cache.clear(); cache[name] = PL.load_seq(B.ROOT, reg[name])
        S = cache[name]; a = S['a']; g = np.flatnonzero(a['gfr'] == fi); G = a['gbox'][g]
        yb = y['box'][(y['fi'] == k) & (y['score'] >= 0.25)].astype(np.float64).copy(); yb[:, [1, 3]] += P.Y0; n_y += len(yb)
        if len(G) == 0 or len(yb) == 0: continue
        iou = P.iou_m(yb, G); best = iou.argmax(1)
        for di in ('near', 'far'):
            d = S['d'][di]; dg = a['dg_' + di]; m = (d['fi'] == fi) & (dg >= 0)
            for j in np.flatnonzero(m):
                gi = np.flatnonzero(g == dg[j])
                if len(gi) == 0: continue
                gi = gi[0]; z = P.xyxy2cxcywh(d['box'][j]); hp = B.hip_pos(z[None], d['kpt'][j][None])[0]
                if not (np.isfinite(hp) and 0.90 <= hp <= 1.00): continue
                cand = np.flatnonzero((best == gi) & (iou[np.arange(len(yb)), best] >= 0.5))
                if len(cand) == 0: continue
                yc = P.xyxy2cxcywh(yb[cand[np.argmax(iou[cand, gi])]]); n_ym += 1
                pred = np.array([yc[0], yc[1] - yc[3] / 2 + to * yc[3] + kh * yc[3] / 2, kw * yc[3], kh * yc[3]])
                out.append(dict(ds=reg[name]['ds'], det=di, yh=yc[3], top=((z[1] - z[3] / 2) - (yc[1] - yc[3] / 2)) / yc[3], h=z[3] / yc[3], w=z[2] / yc[3], cx=(z[0] - yc[0]) / yc[3],
                                err_h=pred[3] / z[3], err_top=((pred[1] - pred[3] / 2) - (z[1] - z[3] / 2)) / z[3], err_w=pred[2] / z[2], iou=float(P.iou_pair(P.cxcywh2xyxy(pred[None]), P.cxcywh2xyxy(z[None]))[0]),
                                cut=bool(yb[cand[0], 3] >= (P.Y1 if reg[name]['ds'] == 'TMOT' else 436.0) - 2)))
    res = dict(constant=dict(top=to, height=kh, width_over_height=kw), yolo_boxes=n_y, pairs=len(out))
    q = lambda v: [round(float(x), 4) for x in np.percentile(v, [10, 25, 50, 75, 90])]
    for nm, sel in (('all', lambda o: True), ('TMOT', lambda o: o['ds'] == 'TMOT'), ('DT', lambda o: o['ds'] == 'DT'), ('yolo_h>=100', lambda o: o['yh'] >= 100), ('cut_at_bottom', lambda o: o['cut'])):
        oo = [o for o in out if sel(o)]
        if len(oo) < 10: res[nm] = dict(n=len(oo)); continue
        res[nm] = dict(n=len(oo), body_top_over_yolo_h=q([o['top'] for o in oo]), body_h_over_yolo_h=q([o['h'] for o in oo]), body_w_over_yolo_h=q([o['w'] for o in oo]),
                       constant_h_over_body_h=q([o['err_h'] for o in oo]), constant_top_err_over_body_h=q([o['err_top'] for o in oo]), constant_w_over_body_w=q([o['err_w'] for o in oo]), iou_constant_vs_body=q([o['iou'] for o in oo]))
    (B.B2 / 'results' / 'plain_check.json').write_text(B.jdump(res), encoding='utf-8'); print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
