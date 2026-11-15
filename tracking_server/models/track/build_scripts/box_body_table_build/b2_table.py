"""body_v2/scripts/b2_table.py - TRAIN split only: repair of the body mode's lookup table and the whole-person -> body constant
(rules: body_v2/PREREG_BODY_V2.json 'table_repair' and 'person_family').
  python b2_table.py stats  -> results/table_stats.json : per-pair spread and the design effect behind N (no table written)
  python b2_table.py fit    -> data/table_v2.npz + results/table_v2.json : the repaired table, every cell's rule trace, the person constant
"""
import json
import math
import sys
import time

import numpy as np

import b2_lib as B
from b2_lib import L

PR = json.loads((B.B2 / 'PREREG_BODY_V2.json').read_text(encoding='utf-8')) if (B.B2 / 'PREREG_BODY_V2.json').exists() else None


def seq_boot(lr, sq, rng, draws):
    """sequence bootstrap of the median of lr (log h ratio) -> array of medians"""
    us = np.unique(sq); idx = [np.flatnonzero(sq == s) for s in us]; out = np.empty(draws)
    for d in range(draws):
        pick = rng.integers(0, len(us), len(us)); out[d] = np.median(lr[np.concatenate([idx[p] for p in pick])])
    return out


def cells(d):
    Z, T, kp = d['Z'], d['T'], d['kp']; bh, ba = B.cell_index(Z, L.LH_EDGES, L.AR_EDGES); e = B.rel(Z, T); hp = B.hip_pos(Z, kp)
    return bh, ba, e, hp


def stats():
    """per-pair robust spread of the log height ratio and the design effect (sequence bootstrap variance / iid variance of the median) in the
    well-populated whole-person cells (fitted height ratio < 0.75, >= 1000 pairs): the inputs of N."""
    d = B.train_pairs(); bh, ba, e, hp = cells(d); rng = np.random.default_rng(0); res = []
    for i in range(L.TAB.shape[0]):
        for j in range(L.TAB.shape[1]):
            m = (bh == i) & (ba == j)
            if m.sum() < 1000 or math.exp(np.median(e[m, 3])) >= 0.75: continue
            lr = e[m, 3]; sig = 1.4826 * np.median(np.abs(lr - np.median(lr))); n = int(m.sum())
            bt = seq_boot(lr, d['seq'][m], rng, 400); v_iid = (1.2533 * sig) ** 2 / n; deff = float(np.var(bt) / v_iid)
            res.append(dict(row=i, col=j, n=n, seqs=int(len(np.unique(d['seq'][m]))), sigma=float(sig), deff=deff, n_eff=n / deff))
            print(res[-1], flush=True)
    sig = float(np.median([r['sigma'] for r in res])); deff = float(np.median([r['deff'] for r in res]))
    out = dict(cells=res, sigma_median=sig, deff_median=deff,
               N_for_se={'%.3f' % se: int(math.ceil(deff * (1.2533 * sig / se) ** 2)) for se in (0.02, 0.025, 0.03)})
    (B.B2 / 'results' / 'table_stats.json').write_text(B.jdump(out), encoding='utf-8'); print('STATS', json.dumps({k: v for k, v in out.items() if k != 'cells'}), flush=True)


def fit():
    t0 = time.time(); R_ = PR['table_repair']; d = B.train_pairs(); bh, ba, e, hp = cells(d); rng = np.random.default_rng(int(R_['bootstrap_seed']))
    nh, na = L.TAB.shape[:2]; tab = np.zeros((nh, na, 4)); kind = np.full((nh, na), '', object); trace = {}
    lo, hi = R_['body_hip_range']; n_hip = int(R_['min_hip_pairs']); N = int(R_['N']); S = int(R_['min_sequences']); W = math.log(float(R_['max_boot90_ratio']))
    clo, chi = R_['contradiction_hip_range']
    fitted = np.full((nh, na, 4), np.nan)
    for i in range(nh):
        for j in range(na):
            m = (bh == i) & (ba == j); n = int(m.sum()); hh = hp[m]; fin = np.isfinite(hh); t = dict(n=n, seqs=int(len(np.unique(d['seq'][m]))), hips=int(fin.sum()))
            if n: fitted[i, j] = np.median(e[m], 0); t['fit'] = dict(h=math.exp(fitted[i, j, 3]), w=math.exp(fitted[i, j, 2]), dy=float(fitted[i, j, 1]), dx=float(fitted[i, j, 0]))
            if fin.any(): t['hip_det'] = float(np.median(hh[fin]))
            if fin.sum() >= n_hip and lo <= t['hip_det'] <= hi:
                kind[i, j] = 'body'; tab[i, j] = 0.0; t['rule'] = 'body cell (detector box is a body box by its own hips): identity'
            elif n:
                why = []
                if n < N: why.append('n %d < N %d' % (n, N))
                if t['seqs'] < S: why.append('sequences %d < %d' % (t['seqs'], S))
                if n >= 2 and t['seqs'] >= 2:
                    bt = seq_boot(e[m, 3], d['seq'][m], rng, int(R_['bootstrap_draws'])); b5, b95 = np.percentile(bt, [5, 95]); t['boot90_h'] = [math.exp(b5), math.exp(b95)]
                    if b95 - b5 > W: why.append('bootstrap 90 %% interval of the height ratio %.3f-%.3f wider than x %.2f' % (math.exp(b5), math.exp(b95), math.exp(W)))
                else: why.append('one sequence')
                if fin.sum() >= n_hip:   # hips seen often enough: where does the fitted value put them?
                    z = d['Z'][m][fin]; c = np.c_[z[:, 0] + fitted[i, j, 0] * z[:, 3], z[:, 1] + fitted[i, j, 1] * z[:, 3], z[:, 2] * math.exp(fitted[i, j, 2]), z[:, 3] * math.exp(fitted[i, j, 3])]
                    hc = float(np.median((hh[fin] * z[:, 3] + (z[:, 1] - z[:, 3] / 2) - (c[:, 1] - c[:, 3] / 2)) / c[:, 3])); t['hip_conv_fit'] = hc
                    if not clo <= hc <= chi: why.append('contradiction: the fitted value puts the hips at %.3f of the converted box' % hc)
                if why: kind[i, j] = 'fill'; t['rule'] = 'fill: ' + '; '.join(why)
                else: kind[i, j] = 'fit'; tab[i, j] = fitted[i, j]; t['rule'] = 'kept: fitted median'
            else:
                kind[i, j] = 'fill'; t['rule'] = 'fill: no pairs'
            trace['%d,%d' % (i, j)] = t
    for i in range(nh):   # fill: the nearest kept or body cell of the same row; a tie -> the one whose height ratio is closer to the cell's own fitted median
        good = [j for j in range(na) if kind[i, j] in ('body', 'fit')]
        for j in range(na):
            if kind[i, j] != 'fill': continue
            if not good:
                src = None
                for di in range(1, nh):
                    for ii in (i - di, i + di):
                        if 0 <= ii < nh and kind[ii, j] in ('body', 'fit'): src = (ii, j); break
                    if src: break
            else:
                dmin = min(abs(g - j) for g in good); cand = [g for g in good if abs(g - j) == dmin]
                if len(cand) > 1:
                    own = fitted[i, j, 3] if np.isfinite(fitted[i, j, 3]) else 0.0
                    cand.sort(key=lambda g: (abs(tab[i, g, 3] - own), -g))
                src = (i, cand[0])
            tab[i, j] = tab[src]; trace['%d,%d' % (i, j)]['filled_from'] = list(src)
    # whole-person -> body constant (person family): train pairs whose detection is a body box by its own hips, against the dataset's whole-person box
    P_ = PR['person_family']; G = d['G']; Z = d['Z']; body = np.isfinite(hp) & (hp >= lo) & (hp <= hi) & (G[:, 3] >= float(P_['min_gt_height']))
    top = ((Z[body, 1] - Z[body, 3] / 2) - (G[body, 1] - G[body, 3] / 2)) / G[body, 3]; hr = Z[body, 3] / G[body, 3]; wh = Z[body, 2] / G[body, 3]; wr = Z[body, 2] / G[body, 2]
    cx = (Z[body, 0] - G[body, 0]) / G[body, 2]
    mad = lambda v: float(np.median(np.abs(np.log(v) - np.median(np.log(v)))))
    person = dict(n=int(body.sum()), top=float(np.median(top)), height=float(np.median(hr)), width_over_height=float(np.median(wh)), width_over_width=float(np.median(wr)),
                  cx_over_width=float(np.median(cx)), mad_log_w_over_h=mad(wh), mad_log_w_over_w=mad(wr), height_q25_q75=[float(np.percentile(hr, 25)), float(np.percentile(hr, 75))],
                  by_dataset={nm: dict(n=int((body & (d['ds'] == k)).sum()), height=float(np.median(Z[body & (d['ds'] == k), 3] / G[body & (d['ds'] == k), 3])),
                                       width_over_height=float(np.median(Z[body & (d['ds'] == k), 2] / G[body & (d['ds'] == k), 3]))) for k, nm in ((0, 'TMOT'), (1, 'DT'))})
    person['width_by'] = 'height' if person['mad_log_w_over_h'] <= person['mad_log_w_over_w'] else 'width'
    pvec = np.array([person['top'], person['height'], person['width_over_height'] if person['width_by'] == 'height' else person['width_over_width']])
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    print('kind:\n', kind); print('h ratio before:\n', np.exp(L.TAB[:, :, 3])); print('h ratio after:\n', np.exp(tab[:, :, 3])); print('w ratio after:\n', np.exp(tab[:, :, 2])); print('dx after:\n', tab[:, :, 0])
    print('person', json.dumps(person), flush=True)
    np.savez(B.B2 / 'data' / 'table_v2.npz', tab=tab, lh_cen=L.LH_CEN, ar_cen=L.AR_CEN, lh_edges=L.LH_EDGES, ar_edges=L.AR_EDGES, person=pvec, person_width_by=np.array(person['width_by']))
    out = dict(rules=R_, person_rules=P_, kind=kind.tolist(), tab=tab, before=L.TAB, cells=trace, person=person, lh_cen_px=np.exp(L.LH_CEN), ar_cen=L.AR_CEN, seconds=round(time.time() - t0, 1))
    (B.B2 / 'results' / 'table_v2.json').write_text(B.jdump(out), encoding='utf-8'); print('TABLE_DONE %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    {'stats': stats, 'fit': fit}[sys.argv[1]]()
