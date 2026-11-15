"""gru_v2/scripts/g2_cost.py - PC cost of the v2 engine (lookup + features + GRU + size filter) for all people of a frame in one batch, one thread."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import json
import time

import numpy as np

import g2_lib as L

exp = json.loads((L.EXPORT_DIR / 'EXPORT.json').read_text()); net = L.load_net(L.EXPORT_DIR / exp['npz']); size = tuple(float(x) for x in np.load(L.EXPORT_DIR / exp['npz'])['size_euro'])
res = {}; rng = np.random.default_rng(0)
for N in (1, 10, 20, 30):
    runs = []
    for rep in range(3):
        eng = L.Engine(net, size); c = np.c_[rng.uniform(50, 590, N), rng.uniform(80, 420, N)]; wh = np.c_[rng.uniform(20, 60, N), rng.uniform(50, 160, N)]; v = rng.normal(0, 1.5, (N, 2)); ts = []
        for f in range(350):
            c = c + v + rng.normal(0, 0.5, (N, 2)); Z = np.c_[c, wh * np.exp(rng.normal(0, 0.02, (N, 2)))]
            KP = np.dstack([Z[:, None, 0] + rng.normal(0, 0.2, (N, 17)) * Z[:, None, 3], Z[:, None, 1] + rng.normal(0, 0.3, (N, 17)) * Z[:, None, 3], rng.uniform(0, 1, (N, 17))])
            H = np.array([[1., 0, rng.normal(0, 0.5)], [0, 1., rng.normal(0, 0.5)]]); t0 = time.perf_counter(); eng.step(list(range(N)), Z, KP, H, f / 25.0); ts.append(time.perf_counter() - t0)
        runs.append(float(np.median(ts[50:])) * 1e3)
    res[N] = dict(ms_per_frame=round(min(runs), 4), ms_per_person=round(min(runs) / N, 4), runs=[round(x, 4) for x in runs])
    print('COST people %2d  %.3f ms per frame  %.4f ms per person  (runs %s)' % (N, min(runs), min(runs) / N, ', '.join('%.3f' % x for x in runs)), flush=True)
json.dump(dict(weights=exp['npz'], size=size, threads=1, cost=res), open(L.G2 / 'results' / 'cost.json', 'w'), indent=1); print('COST_DONE', flush=True)
