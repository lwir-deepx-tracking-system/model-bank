"""euro_tune.py (PC): One Euro parameters on val-A only (PREREG box_refiner): server-like rows of the tracker (streams near, far, sw), per tracker id,
loss = the refiner's validation loss (accuracy + 1.0 x jitter). Centre (cx, cy) and size (log w, log h) are separate filters, so the centre loss part
(error components 0, 1) is minimised over (min_cutoff, beta) of the centre and the size part (components 2, 3) over those of the size.
Done with camera compensation (main) and without it (the camera test). Writes results/euro_valA.json.
Usage: python euro_tune.py <root>
"""
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prcommon as P
import ref_lib as R

MC = (0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 5.0, 8.0)
BETA = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
G = {}


def init(root):
    reg = [r for r in P.registry() if r['split'] == 'valA']; G['data'] = []
    for r in reg:
        for stream in ('near', 'far', 'sw'):
            ids, S = R.rows_of(root, r, stream); G['data'].append((r, ids, S['W']))


def parts(outs, tgts, cons):
    """loss parts: centre (acc, jit) and size (acc, jit) sums and counts."""
    s = np.zeros(4); n = np.zeros(2)
    for o, g, c in zip(outs, tgts, cons):
        ok = np.isfinite(g[:, 0])
        if not ok.any(): continue
        e = R.err_vec(o, np.where(ok[:, None], g, 1.)); a = R.sl1(e[ok]); s[0] += a[:, :2].sum(); s[2] += a[:, 2:].sum(); n[0] += ok.sum()
        if len(o) >= 3:
            d2 = e[2:] - 2 * e[1:-1] + e[:-2]; v = ok[2:] & ok[1:-1] & ok[:-2] & (c[2:] > 0) & (c[1:-1] > 0); j = R.sl1(d2[v])
            s[1] += j[:, :2].sum(); s[3] += j[:, 2:].sum(); n[1] += v.sum()
    return s, n


def job(args):
    params, comp = args; outs, tgts, cons = [], [], []
    for r, ids, W in G['data']:
        for e in ids.values():
            outs.append(R.run_euro(e, W, r['fps'], params, comp)); tgts.append(e['tgt']); cons.append(np.r_[0, (np.diff(e['f']) == 1)])
    s, n = parts(outs, tgts, cons)
    return params, comp, (s[0] / n[0] + s[1] / n[1], s[2] / n[0] + s[3] / n[1])


def main():
    root = Path(sys.argv[1]); res = {}
    with Pool(8, initializer=init, initargs=(str(root),)) as pool:
        for comp in (True, False):
            # stage 1: centre grid with a mild size filter; stage 2: size grid with the chosen centre
            r1 = pool.map(job, [((mc, b, 0.8, 0.3, 1.0), comp) for mc, b in itertools.product(MC, BETA)])
            bc = min(r1, key=lambda x: x[2][0]); mc, beta = bc[0][0], bc[0][1]
            r2 = pool.map(job, [((mc, beta, ms, bs, 1.0), comp) for ms, bs in itertools.product(MC, BETA)])
            bs_ = min(r2, key=lambda x: x[2][1])
            res['comp' if comp else 'nocomp'] = dict(mc=mc, beta=beta, mc_s=bs_[0][2], beta_s=bs_[0][3], dcut=1.0, centre_loss=bc[2][0], size_loss=bs_[2][1],
                                                    centre_grid=[[p[0], p[1], round(l[0], 6)] for p, _, l in r1], size_grid=[[p[2], p[3], round(l[1], 6)] for p, _, l in r2])
            print('EURO comp=%s centre mc %g beta %g (loss %.5f) size mc %g beta %g (loss %.5f)' % (comp, mc, beta, bc[2][0], bs_[0][2], bs_[0][3], bs_[2][1]), flush=True)
    res['note'] = 'val-A only; grids MC %s x BETA %s; loss = refiner validation loss parts (smooth L1 beta 0.02 of the error + 1.0 x its second difference)' % (MC, BETA)
    (root / 'results' / 'euro_valA.json').write_text(json.dumps(res, indent=1)); print('EURO_DONE', flush=True)


if __name__ == '__main__':
    main()
