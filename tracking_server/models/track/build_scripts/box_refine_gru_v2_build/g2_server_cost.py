"""gru_v2/scripts/g2_server_cost.py - PC cost of the SERVER mode gru2 (box_refine.Refiner.step: hand-over search, batched engine, size filter), one thread.
N people moving a little, every one with keypoints, a camera warp every frame; 400 frames, the first 50 dropped; 3 runs, the best median.
Usage: python g2_server_cost.py <tracking_server dir> <npz> [gru2|body]   -> results/server_cost_<mode>.json"""
import json
import os
import sys
import time

for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import numpy as np   # noqa: E402

import g2_lib as L   # noqa: E402

sys.path.insert(0, sys.argv[1])
from src.track import box_refine as BR   # noqa: E402

npz = sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else 'gru2'; z = np.load(npz, allow_pickle=False); model = BR.G2Model({k: z[k] for k in z.files}); res = {}
for n in (1, 10, 20, 30):
    runs = []
    for rep in range(3):
        rng = np.random.default_rng(rep); r = BR.Refiner(mode, model); zs = np.c_[rng.uniform(40, 600, n), rng.uniform(100, 400, n), rng.uniform(15, 60, n), rng.uniform(40, 150, n)]; ts = []
        kp = rng.uniform(0, 1, (n, 17, 3)); H = np.array([[1., 0, 0.5], [0, 1., 0.2]])
        for k in range(400):
            zs[:, :2] += rng.normal(0, 1, (n, 2)); items = [(i, zs[i].copy(), np.c_[zs[i, 0] + 20 * kp[i, :, 0], zs[i, 1] + 40 * kp[i, :, 1], kp[i, :, 2]], zs[i].copy()) for i in range(n)]
            t0 = time.perf_counter(); r.step(items, k / 24., H); ts.append(time.perf_counter() - t0)
        runs.append(float(np.median(ts[50:]) * 1000)); p99 = float(np.percentile(ts[50:], 99) * 1000)
    res[n] = dict(ms_per_frame=round(min(runs), 4), ms_per_person=round(min(runs) / n, 4), runs=[round(v, 4) for v in runs], p99_last_run=round(p99, 4))
    print('SERVER_COST %2d people: %.3f ms per frame (median, best of 3; p99 %.3f)' % (n, min(runs), p99), flush=True)
(L.G2 / 'results' / ('server_cost_%s.json' % mode)).write_text(json.dumps(dict(weights=os.path.basename(npz), threads=1, what='box_refine.Refiner(%s).step' % mode, cost=res), indent=1)); print('SERVER_COST_DONE', mode, flush=True)
