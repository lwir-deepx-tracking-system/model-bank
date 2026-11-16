"""v2/v3/scripts/v3_valB.py - box_refine v3 val-B run (PREREG_V3.json valB_once_v3; the 4th look at val-B for this family).
Methods raw, tracker, euro, size_centre (v2 choice), size_aim (v3 choice = box_refine.SIZE, must equal v3/results/aim_valA.json), size_aim_b03
(+ box_blend 0.3 s). Usage: python v3_valB.py <split valB|valA> <metric aim 0.37|0.08> [workers]. Writes v2/v3/results/aim_<split>_aim<aim>.json / .md
"""
import os
import sys
os.environ['V3_AIM'] = sys.argv[2] if len(sys.argv) > 2 else '0.37'
import json        # noqa: E402
import time        # noqa: E402
from multiprocessing import Pool   # noqa: E402

import numpy as np  # noqa: E402

import v3_lib as V  # noqa: E402
import v2_valB as VB  # noqa: E402

L = V.L
METHODS = ('raw', 'tracker', 'euro', 'size_centre', 'size_aim', 'size_aim_b03')
VB.NAMES.update(size_centre='size, centre anchor (v2)', size_aim='size, aim anchor (v3)', size_aim_b03='size aim + blend 0.3 s')
VB.PAIRS = (('size_aim', 'tracker'), ('size_centre', 'tracker'), ('size_aim', 'size_centre'), ('size_aim_b03', 'tracker'), ('size_aim_b03', 'size_aim'), ('euro', 'tracker'))
EDGE_STATS = ('top_p99', 'bot_p99', 'top_sw0_p95', 'bot_sw0_p95', 'top_sw0_mean', 'bot_sw0_mean', 'top_flip0_p95', 'bot_flip0_p95', 'top_flip0_mean', 'bot_flip0_mean',
              'edge_gt10')


def job(a):
    r, st = a; inp = L.stream_inputs(r, st); outs = {}
    _, kk = L.run(inp, 'off')
    outs['euro'], _ = L.run(inp, 'euro')
    outs['size_centre'], k2 = V.run3(inp, V.V2_SIZE, 0.0, V.AIM)
    outs['size_aim'], k3 = V.run3(inp, dict(L.BRM.SIZE), 0.0, V.AIM)
    outs['size_aim_b03'], _ = V.run3(inp, dict(L.BRM.SIZE), 0.3, V.AIM)
    assert all((k2[t] == kk[t]).all() and (k3[t] == kk[t]).all() for t in kk)
    recs = L.recs_of(r, st, inp, outs, kk)
    return recs, [L.events(x) for x in recs]


def next2(ev0, f):
    """the switch frame and the next 2 rows of the same id on consecutive frames"""
    m = ev0.copy()
    for i in np.flatnonzero(ev0):
        for d in (1, 2):
            if i + d < len(f) and f[i + d] == f[i] + d: m[i + d] = True
    return m


def stat(P, s):
    k = 'dt' if s.startswith('top') else ('db' if s.startswith('bot') else 'de')
    if s == 'edge_gt10': v = P['de'][np.isfinite(P['de'])]; return float((v > 0.10).mean()) if len(v) else np.nan
    ev = 'sw0' if '_sw0' in s else ('flip0' if '_flip0' in s else None)
    v = P[k] if ev is None else np.where(P[ev], P[k], np.nan); v = v[np.isfinite(v)]
    if len(v) < 5: return np.nan
    if s.endswith('_mean'): return float(v.mean())
    return float(np.percentile(v, 99 if s.endswith('p99') else 95))


def edge_table(recs, evs):
    names = sorted({r['name'] for r in recs}); loc = {nm: j for j, nm in enumerate(names)}; ns = len(names); per = {m: [] for m in METHODS}; seq = []
    for rec, (em, _) in zip(recs, evs):
        e = rec['e']; ok = np.isfinite(e['tgt'][:, 0]); ht = np.where(ok, e['tgt'][:, 3], 1.)
        for m in METHODS:
            dt, db = V.edges(rec['o'][m], e['f'], ht, ok); per[m].append(dict(dt=dt, db=db, de=np.fmax(dt, db), sw0=em['switch0'], flip0=em['flip0']))
        seq.append(loc[rec['name']])
    seq = np.array(seq); byseq = [np.flatnonzero(seq == s) for s in range(ns)]
    pool_ = lambda m, idx: {kk: np.concatenate([per[m][j][kk] for j in idx]) for kk in per[m][0]}
    T = {m: {s: stat(pool_(m, range(len(recs))), s) for s in EDGE_STATS} for m in METHODS}
    draws = np.random.default_rng(0).integers(0, ns, (2000, ns)); D = {}
    for a, b in (('size_aim', 'tracker'), ('size_centre', 'tracker'), ('size_aim_b03', 'tracker')):
        bs = {s: [] for s in EDGE_STATS}
        for d in draws:
            idx = np.concatenate([byseq[s] for s in d]); A = pool_(a, idx); B = pool_(b, idx)
            for s in EDGE_STATS: bs[s].append(stat(A, s) - stat(B, s))
        D['%s - %s' % (a, b)] = {s: [round(T[a][s] - T[b][s], 5), round(float(np.nanpercentile(bs[s], 5)), 5), round(float(np.nanpercentile(bs[s], 95)), 5)] for s in EDGE_STATS}
    n = dict(sw0=int(sum(p['sw0'].sum() for p in per['tracker'])), flip0=int(sum(p['flip0'].sum() for p in per['tracker'])))
    return dict(methods={m: {s: (None if not np.isfinite(v) else round(v, 5)) for s, v in T[m].items()} for m in T}, diffs=D, events=n)


def p7b(recs, evs, a='size_aim', b='tracker'):
    names = sorted({r['name'] for r in recs}); ns = len(names); draws = np.random.default_rng(0).integers(0, ns, (2000, ns)); Cc = L.C.draw_counts(draws, ns)
    RA, seqv, idv, ev, _, _ = L.gather(recs, [r['o'][a] for r in recs], evs); RB = L.gather(recs, [r['o'][b] for r in recs], evs)[0]
    m = np.concatenate([next2(em['switch0'], r['e']['f']) for r, (em, _) in zip(recs, evs)]) & RA['ok']
    n_ev = int(sum(int(em['switch0'].sum()) for em, _ in evs))
    d = L.C.boot_pct(RA['dh'], RB['dh'], seqv, m, ns, Cc, 95)['diff'] if n_ev >= 1 and (m & np.isfinite(RA['dh'])).sum() >= 5 else None
    return dict(events=n_ev, rows=int(m.sum()), dh_p95_diff=d, passes=bool(d is not None and n_ev >= 20 and d[2] < 0))


def decide(T, key='size_aim - tracker'):
    out = {}; allok = True
    for ds in ('TMOT', 'DT'):
        nf = T['%s|nf' % ds]; sw = T['%s|sw' % ds]; d = nf['diffs'][key]; dsw = sw['diffs'][key]; t2 = nf['methods']['tracker']['d2a_rms'] ** 2
        c = dict(P1_dh_p99=d['dh_p99'][2] < 0, P2_trackmax=d['dh_trackmax_med'][2] < 0, P3_aim=d['aim'][2] <= 0.003, P4_aim_jitter=d['d2a_sq'][2] <= 0.04 * t2,
                 P5_lh=d['lh'][2] <= 0.005, P6_aim_lag_det_fast=d['aim_det_0.1-99'] is not None and d['aim_det_0.1-99'][1] >= -0.01,
                 P7_switch1s_dh_p99=('ev_switch1s_dh_p99' in dsw) and dsw['ev_switch1s_dh_p99'][2] < 0)
        out[ds] = dict(c, values=dict(dh_p99=d['dh_p99'], trackmax=d['dh_trackmax_med'], aim=d['aim'], d2a_sq=d['d2a_sq'], d2a_sq_limit=round(0.04 * t2, 7), lh=d['lh'],
                                      aim_det_fast=d['aim_det_0.1-99'], switch1s_dh_p99=dsw.get('ev_switch1s_dh_p99')))
        allok &= all(c.values())
    out['passes'] = bool(allok); return out


def main():
    t0 = time.time(); split = sys.argv[1]; aim = sys.argv[2]; nw = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ch = json.loads((V.V3 / 'results' / 'aim_valA.json').read_text())['choice']
    assert ch['feasible'] and all(L.BRM.SIZE[k] == ch[k] for k in ('anchor', 'src', 'mc_s', 'beta_s', 'dcut')), (L.BRM.SIZE, ch)
    print('SIZE', L.BRM.SIZE, 'metric aim', V.AIM, flush=True)
    reg = [r for r in L.P.registry() if r['split'] == split]; jobs = [(r, st) for r in reg for st in ('near', 'far', 'sw')]
    with Pool(nw) as pool: res = pool.map(job, jobs, chunksize=1)
    recs = [x for rr, _ in res for x in rr]; evs = [x for _, ee in res for x in ee]
    out = dict(split=split, metric_aim=V.AIM, size=dict(L.BRM.SIZE), v2_size=V.V2_SIZE, tables={}, edges={}, p7b={})
    for ds in ('TMOT', 'DT'):
        for part, sel in (('nf', ('near', 'far')), ('sw', ('sw',))):
            k = [i for i, r in enumerate(recs) if r['ds'] == ds and r['stream'] in sel]; R_ = [recs[i] for i in k]; E_ = [evs[i] for i in k]
            T = VB.table(R_, E_, METHODS); key = '%s|%s' % (ds, part); out['tables'][key] = T; out['edges'][key] = edge_table(R_, E_)
            if part == 'sw': out['p7b'][ds] = dict(size_aim=p7b(R_, E_), size_centre=p7b(R_, E_, 'size_centre'), size_aim_b03=p7b(R_, E_, 'size_aim_b03'))
            print('TABLE', key, T['n_rows'], '%.0f s' % (time.time() - t0), flush=True)
            for m in METHODS:
                s = T['methods'][m]
                print('  %-12s aim %.4f d2a %.5f lh %.4f dh99 %.4f tmax %.4f sw1 %s aimdet> %s slag %+.4f | edge top99 %s bot99 %s' % (m, s['aim'], s['d2a_rms'], s['lh'], s['dh_p99'], s['dh_trackmax_med'],
                      VB.r5(s['events'].get('switch1s', {}).get('dh_p99')), VB.r5(s['lag_0.1-99']['aim_det']), s['slag_0.03-99']['salong'],
                      out['edges'][key]['methods'][m]['top_p99'], out['edges'][key]['methods'][m]['bot_p99']), flush=True)
    out['decision'] = decide(out['tables']); out['decision_v2_centre'] = decide(out['tables'], 'size_centre - tracker')
    print('DECISION', json.dumps(out['decision']), flush=True); print('P7B', json.dumps(out['p7b']), flush=True)
    tag = 'aim_%s_aim%s' % (split, aim)
    (V.V3 / 'results' / ('%s.json' % tag)).write_text(json.dumps(out, indent=1))
    print('VALB_DONE %.1f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
