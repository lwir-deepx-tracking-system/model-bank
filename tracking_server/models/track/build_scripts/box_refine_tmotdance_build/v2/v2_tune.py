"""v2/scripts/v2_tune.py - val-A choice of the 'size' mode (PREREG_V2.json selection_valA). Runs the server Refiner('size') over the val-A
near / far / sw rows of TMOT and DanceTrack for every grid point, pools the streams per dataset, applies the pre-registered rule.
Writes v2/results/size_valA.json. Usage: python v2_tune.py [workers]
"""
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import v2_lib as L

G = {}
GRID = dict(src=('det', 'trk'), mc_s=(0.2, 0.4, 0.8, 1.5, 3.0, 5.0), beta_s=(0.0, 0.1, 0.3, 1.0, 3.0))
ANCHORS = ('centre', 'top')


def init():
    reg = [r for r in L.P.registry() if r['split'] == 'valA']; G['data'] = []
    for r in reg:
        for st in ('near', 'far', 'sw'):
            inp = L.stream_inputs(r, st); _, kk = L.run(inp, 'off'); recs = L.recs_of(r, st, inp, {}, kk)
            G['data'].append((r, st, inp, kk, recs, [L.events(x) for x in recs]))


def pooled(method_outs):
    """method_outs: list (one per G['data'] entry) of {tid: out} or a base method name -> {ds: sel metrics}"""
    res = {}
    for ds in ('TMOT', 'DT'):
        recs, outs, evs = [], [], []
        for (r, st, inp, kk, rr, ee), mo in zip(G['data'], method_outs):
            if r['ds'] != ds: continue
            for rec, ev in zip(rr, ee):
                recs.append(rec); evs.append(ev); outs.append(rec['o'][mo] if isinstance(mo, str) else mo[rec['id']])
        rows, seqv, idv, ev, sb, names = L.gather(recs, outs, evs); res[ds] = L.sel_metrics(rows, ev, idv)
    return res


def job(cfg):
    anchor, src, mc, beta = cfg; size = dict(mc_s=mc, beta_s=beta, dcut=1.0, anchor=anchor, src=src); outs = []
    for r, st, inp, kk, rr, ee in G['data']:
        o, k2 = L.run(inp, 'size', size=size); outs.append(o)
        assert all((k2[t] == kk[t]).all() for t in kk)   # the server segments do not depend on the mode
    return cfg, pooled(outs)


def base():
    return {m: pooled([m] * len(G['data'])) for m in ('raw', 'tracker')}


def violation(m, t):
    """largest relative constraint violation (<= 0 = feasible) of metrics m against the tracker t, both datasets."""
    v = []
    for ds in ('TMOT', 'DT'):
        a, b = m[ds], t[ds]
        v += [a['aim'] / (1.02 * b['aim']) - 1, a['d2a'] / (1.02 * b['d2a']) - 1, a['lh'] / (1.02 * b['lh']) - 1,
              ((b['slag'] - 0.01) - a['slag']) / 0.01, ((b['aimdet_fast'] - 0.005) - a['aimdet_fast']) / 0.005]
    return max(v)


def J(m, t): return float(np.mean([(m[ds]['dh_p99'] / t[ds]['dh_p99'] + m[ds]['tmax'] / t[ds]['tmax']) / 2 for ds in ('TMOT', 'DT')]))


def pick(points, t):
    feas = [p for p in points if p['viol'] <= 0]
    if not feas:
        p = min(points, key=lambda p: p['viol']); return dict(p, feasible=False)
    best = min(p['J'] for p in feas); near = [p for p in feas if p['J'] <= best + 0.005]
    p = sorted(near, key=lambda p: (-p['mc_s'], -p['beta_s'], 0 if p['src'] == 'det' else 1))[0]; return dict(p, feasible=True)


def main():
    t0 = time.time(); nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    init(); B = base(); print('BASE', json.dumps(B), '%.1f s' % (time.time() - t0), flush=True)
    cfgs = list(itertools.product(ANCHORS, GRID['src'], GRID['mc_s'], GRID['beta_s']))
    with Pool(nw, initializer=init) as pool:
        res = pool.map(job, cfgs, chunksize=1)
    pts = []
    for (anchor, src, mc, beta), m in res:
        p = dict(anchor=anchor, src=src, mc_s=mc, beta_s=beta, dcut=1.0, J=J(m, B['tracker']), viol=violation(m, B['tracker']), m=m); pts.append(p)
        print('PT %-6s %-3s mc %.1f beta %.1f J %.4f viol %+.3f | TMOT dh99 %.4f tmax %.4f aim %.4f d2a %.4f lh %.4f slag %+.4f aimdet %+.4f | DT dh99 %.4f tmax %.4f aim %.4f d2a %.4f lh %.4f slag %+.4f aimdet %+.4f'
              % (anchor, src, mc, beta, p['J'], p['viol'], *[m['TMOT'][k] for k in ('dh_p99', 'tmax', 'aim', 'd2a', 'lh', 'slag', 'aimdet_fast')],
                 *[m['DT'][k] for k in ('dh_p99', 'tmax', 'aim', 'd2a', 'lh', 'slag', 'aimdet_fast')]), flush=True)
    picks = {a: pick([p for p in pts if p['anchor'] == a], B['tracker']) for a in ANCHORS}
    fe = [a for a in ANCHORS if picks[a]['feasible']]
    if fe:
        best = min(picks[a]['J'] for a in fe); anchor = 'top' if 'top' in fe and picks['top']['J'] <= best + 0.005 else min(fe, key=lambda a: picks[a]['J'])
    else:
        anchor = min(ANCHORS, key=lambda a: picks[a]['viol'])
    ch = picks[anchor]; choice = dict(anchor=anchor, src=ch['src'], mc_s=ch['mc_s'], beta_s=ch['beta_s'], dcut=1.0, feasible=ch['feasible'], J=ch['J'])
    out = dict(rules='v2/PREREG_V2.json selection_valA', base=B, points=pts, picks={a: {k: v for k, v in p.items() if k != 'm'} for a, p in picks.items()},
               pick_metrics={a: p['m'] for a, p in picks.items()}, choice=choice, seconds=round(time.time() - t0, 1))
    (L.V2 / 'results' / 'size_valA.json').write_text(json.dumps(out, indent=1))
    print('PICKS', json.dumps(out['picks']), flush=True); print('CHOICE', json.dumps(choice), flush=True); print('TUNE_DONE %.1f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
