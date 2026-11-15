"""body_v2/scripts/b2_train_look.py - TRAIN split only: what the pushed lookup table's cells rest on (before any rule is written).
Per cell (detector box height row x aspect column): pairs, sequences, the fitted relation, how stable it is (per-sequence spread of the cell
median, bootstrap over sequences), the detectors' own hip line in the detector box and in the converted box.
Also: the whole-person box (dataset GT) -> body box relation for pairs whose detection is a body box by its own keypoints.
Writes body_v2/results/train_look.json.   Usage: python b2_train_look.py
"""
import sys
import time

import numpy as np

import b2_lib as B
from b2_lib import L

np.set_printoptions(precision=3, suppress=True, linewidth=250)


def main():
    t0 = time.time(); d = B.train_pairs(); Z, T, G, kp = d['Z'], d['T'], d['G'], d['kp']; n = len(Z)
    print('train pairs %d, sequences %d (%.0f s)' % (n, len(d['names']), time.time() - t0), flush=True)
    bh, ba = B.cell_index(Z, L.LH_EDGES, L.AR_EDGES); e = B.rel(Z, T); hp = B.hip_pos(Z, kp)
    cv = B.conv(L.TAB, L.LH_CEN, L.AR_CEN, Z); hpc = np.where(np.isfinite(hp), (hp * Z[:, 3] + (Z[:, 1] - Z[:, 3] / 2) - (cv[:, 1] - cv[:, 3] / 2)) / cv[:, 3], np.nan)
    hpt = np.where(np.isfinite(hp), (hp * Z[:, 3] + (Z[:, 1] - Z[:, 3] / 2) - (T[:, 1] - T[:, 3] / 2)) / T[:, 3], np.nan)
    nh, na = L.TAB.shape[:2]; res = dict(n_pairs=n, lh_edges_px=np.exp(L.LH_EDGES).tolist(), ar_edges=L.AR_EDGES.tolist(), cells=[])
    rng = np.random.default_rng(0)
    print('row(px) col(ar)    n  seqs near/far TMOT/DT | fit h  w   dy    dx  | seq-median h: iqr | boot90 h | hip in det  n | hip in conv | hip in target')
    for i in range(nh):
        for j in range(na):
            m = (bh == i) & (ba == j); c = int(m.sum())
            if c == 0: continue
            sq = d['seq'][m]; us = np.unique(sq); med = np.median(e[m], 0)
            per = np.array([np.median(e[m][sq == s, 3]) for s in us if (sq == s).sum() >= 5])
            bt = []
            if len(us) >= 2:
                idx = {s: np.flatnonzero(sq == s) for s in us}
                for _ in range(300):
                    pick = rng.choice(us, len(us)); bt.append(np.median(np.concatenate([e[m][idx[s], 3] for s in pick])))
            hh = hp[m]; hh = hh[np.isfinite(hh)]; hc = hpc[m]; hc = hc[np.isfinite(hc)]; ht = hpt[m]; ht = ht[np.isfinite(ht)]
            cell = dict(row=i, col=j, n=c, seqs=int(len(us)), near=int((d['det'][m] == 0).sum()), far=int((d['det'][m] == 1).sum()), tmot=int((d['ds'][m] == 0).sum()), dt=int((d['ds'][m] == 1).sum()),
                        fit=dict(h=float(np.exp(med[3])), w=float(np.exp(med[2])), dy=float(med[1]), dx=float(med[0])), table=dict(h=float(np.exp(L.TAB[i, j, 3])), w=float(np.exp(L.TAB[i, j, 2])), dy=float(L.TAB[i, j, 1]), dx=float(L.TAB[i, j, 0])),
                        seq_med_h=[float(np.exp(np.percentile(per, q))) for q in (25, 50, 75)] if len(per) >= 3 else None,
                        boot90_h=[float(np.exp(np.percentile(bt, 5))), float(np.exp(np.percentile(bt, 95)))] if bt else None,
                        hip_det=dict(n=int(len(hh)), med=float(np.median(hh)) if len(hh) else None, q25=float(np.percentile(hh, 25)) if len(hh) else None, q75=float(np.percentile(hh, 75)) if len(hh) else None),
                        hip_conv=float(np.median(hc)) if len(hc) else None, hip_target=float(np.median(ht)) if len(ht) else None)
            res['cells'].append(cell)
            f = lambda v: '-' if v is None else '%.3f' % v
            print('%2d %7.0f %d %4.2f %6d %3d %6d/%-6d %6d/%-6d | %.3f %.3f %+.3f %+.3f | %s | %s | %s %5d | %s | %s' % (
                i, np.exp(L.LH_CEN[i]), j, L.AR_CEN[j], c, len(us), cell['near'], cell['far'], cell['tmot'], cell['dt'], cell['fit']['h'], cell['fit']['w'], med[1], med[0],
                ' '.join(f(v) for v in cell['seq_med_h']) if cell['seq_med_h'] else '-', ' '.join(f(v) for v in cell['boot90_h']) if cell['boot90_h'] else '-',
                f(cell['hip_det']['med']), cell['hip_det']['n'], f(cell['hip_conv']), f(cell['hip_target'])), flush=True)
    # ---- rows by height only: are boxes of this height body boxes (hips at 0.93-0.96 of the detector box)?
    print('\nby detector box height row, all aspects: n, share of pairs whose detection is a body box by its own hips (0.88-1.0), median hip, by detector')
    res['rows'] = []
    for i in range(nh):
        o = dict(row=i, px=float(np.exp(L.LH_CEN[i])))
        for k, nm in ((None, 'both'), (0, 'near'), (1, 'far')):
            m = (bh == i) & ((d['det'] == k) if k is not None else True); hh = hp[m]; fin = np.isfinite(hh)
            o[nm] = dict(n=int(m.sum()), hips_seen=int(fin.sum()), body_share=float(np.mean((hh[fin] >= 0.88) & (hh[fin] <= 1.0))) if fin.any() else None, hip_med=float(np.median(hh[fin])) if fin.any() else None,
                         h_ratio=float(np.exp(np.median(e[m, 3]))) if m.any() else None, ar_med=float(np.median(Z[m, 2] / Z[m, 3])) if m.any() else None)
        res['rows'].append(o); print(i, '%.0f' % o['px'], {k: (v['n'], v['hips_seen'], None if v['body_share'] is None else round(v['body_share'], 3), None if v['hip_med'] is None else round(v['hip_med'], 3), None if v['h_ratio'] is None else round(v['h_ratio'], 3), None if v['ar_med'] is None else round(v['ar_med'], 3)) for k, v in o.items() if isinstance(v, dict)})
    # ---- whole-person (dataset GT) box -> body box: pairs whose detection is a body box by its own keypoints (hips 0.88-1.0 of the box) and >= 100 px GT
    print('\nwhole-person box (GT) -> body box (detection that is a body box by its own hips), train split')
    body = np.isfinite(hp) & (hp >= 0.88) & (hp <= 1.0); gh = G[:, 3]; res['person'] = {}
    for nm, m0 in (('TMOT', d['ds'] == 0), ('DT', d['ds'] == 1), ('both', np.ones(n, bool))):
        for bn, lo, hi in (('all>=60', 60, 1e9), ('60-100', 60, 100), ('100-170', 100, 170), ('170-300', 170, 300), ('>=300', 300, 1e9)):
            m = m0 & body & (gh >= lo) & (gh < hi)
            if m.sum() < 50: continue
            top = ((Z[m, 1] - Z[m, 3] / 2) - (G[m, 1] - G[m, 3] / 2)) / G[m, 3]; hr = Z[m, 3] / G[m, 3]; cx = (Z[m, 0] - G[m, 0]) / G[m, 2]; wr = Z[m, 2] / G[m, 2]; wh = Z[m, 2] / G[m, 3]; ar = Z[m, 2] / Z[m, 3]
            q = lambda v: [round(float(np.percentile(v, p)), 4) for p in (25, 50, 75)]
            mad = lambda v: float(np.median(np.abs(np.log(v) - np.median(np.log(v)))))
            o = dict(n=int(m.sum()), top=q(top), height=q(hr), cx=q(cx), width_over_gt_width=q(wr), width_over_gt_height=q(wh), body_aspect=q(ar), mad_log_w_over_w=round(mad(wr), 4), mad_log_w_over_h=round(mad(wh), 4),
                     gt_aspect=q(G[m, 2] / G[m, 3]))
            res['person']['%s|%s' % (nm, bn)] = o; print(nm, bn, o)
    # the target's own relation to the GT box (the fixed per-dataset relation + the hip correction)
    for nm, m0 in (('TMOT', d['ds'] == 0), ('DT', d['ds'] == 1)):
        m = m0 & (gh >= 60)
        o = dict(n=int(m.sum()), top=float(np.median(((T[m, 1] - T[m, 3] / 2) - (G[m, 1] - G[m, 3] / 2)) / G[m, 3])), height=float(np.median(T[m, 3] / G[m, 3])), width_over_gt_width=float(np.median(T[m, 2] / G[m, 2])),
                 width_over_gt_height=float(np.median(T[m, 2] / G[m, 3])), aspect=float(np.median(T[m, 2] / T[m, 3])))
        res['person']['target|%s' % nm] = o; print('target', nm, o)
    (B.B2 / 'results' / 'train_look.json').write_text(B.jdump(res), encoding='utf-8'); print('TRAIN_LOOK_DONE %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
