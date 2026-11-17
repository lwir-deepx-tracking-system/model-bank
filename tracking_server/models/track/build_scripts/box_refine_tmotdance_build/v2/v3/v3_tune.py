"""v2/v3/scripts/v3_tune.py - val-A choice of the size mode with the 'aim' anchor (PREREG_V3.json selection_valA). Writes v2/v3/results/aim_valA.json.
Usage: python v3_tune.py [workers]"""
import itertools
import json
import sys
import time
from multiprocessing import Pool

import v3_lib as V
import v2_tune as T   # noqa: E402  (init, pooled, base, violation, J, pick of v2)

L = V.L
GRID = T.GRID


def init(): T.init()


def job(cfg):
    src, mc, beta, blend, aim = cfg; size = dict(mc_s=mc, beta_s=beta, dcut=1.0, anchor='aim', src=src); outs = []
    for r, st, inp, kk, rr, ee in T.G['data']:
        o, k2 = V.run3(inp, size, blend, aim); outs.append(o)
        assert all((k2[t] == kk[t]).all() for t in kk)
    return cfg, T.pooled(outs)


def main():
    t0 = time.time(); nw = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    assert abs(V.AIM - 0.37) < 1e-9, 'the choice is made at the v2 metric aim point'
    init(); B = T.base(); print('BASE', json.dumps(B), '%.1f s' % (time.time() - t0), flush=True)
    cfgs = [(s, m, b, 0.0, V.AIM) for s, m, b in itertools.product(GRID['src'], GRID['mc_s'], GRID['beta_s'])]
    with Pool(nw, initializer=init) as pool: res = pool.map(job, cfgs, chunksize=1)
    pts = []
    for (src, mc, beta, _, _), m in res:
        p = dict(anchor='aim', src=src, mc_s=mc, beta_s=beta, dcut=1.0, J=T.J(m, B['tracker']), viol=T.violation(m, B['tracker']), m=m); pts.append(p)
        print('PT aim %-3s mc %.1f beta %.1f J %.4f viol %+.3f | TMOT aim %.5f/%.5f d2a %.5f/%.5f slag %+.4f | DT aim %.5f/%.5f d2a %.5f/%.5f slag %+.4f'
              % (src, mc, beta, p['J'], p['viol'], m['TMOT']['aim'], B['tracker']['TMOT']['aim'], m['TMOT']['d2a'], B['tracker']['TMOT']['d2a'], m['TMOT']['slag'],
                 m['DT']['aim'], B['tracker']['DT']['aim'], m['DT']['d2a'], B['tracker']['DT']['d2a'], m['DT']['slag']), flush=True)
    ch = T.pick(pts, B['tracker']); choice = dict(anchor='aim', src=ch['src'], mc_s=ch['mc_s'], beta_s=ch['beta_s'], dcut=1.0, feasible=ch['feasible'], J=ch['J'])
    print('CHOICE', json.dumps(choice), flush=True)
    # report only: the choice with box_blend 0.3 s
    with Pool(nw, initializer=init) as pool: rep = pool.map(job, [(choice['src'], choice['mc_s'], choice['beta_s'], 0.3, V.AIM)], chunksize=1)
    out = dict(rules='v2/v3/PREREG_V3.json selection_valA', base=B, points=pts, choice=choice, pick_metrics=ch['m'], report_blend03=rep[0][1],
               seconds=round(time.time() - t0, 1))
    (V.V3 / 'results' / 'aim_valA.json').write_text(json.dumps(out, indent=1))
    print('BLEND03', json.dumps(rep[0][1]), flush=True); print('TUNE_DONE %.1f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
