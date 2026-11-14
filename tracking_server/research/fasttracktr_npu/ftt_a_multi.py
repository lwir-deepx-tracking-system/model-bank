"""Device: one engine per NPU core (the runtime refuses multi-core binding for this model), frames dealt round-robin,
each engine with up to K frames in flight. Usage: python ftt_a_multi.py <a.dxnn> <cores e.g. 1,2> [K] [frames]"""
import json
import sys
import time

import dx_engine as dx
import numpy as np

path, cores = sys.argv[1], [int(c) for c in sys.argv[2].split(',')]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
N = int(sys.argv[4]) if len(sys.argv) > 4 else 90
BO = dx.InferenceOption.BOUND_OPTION
engs = []
for c in cores:
    o = dx.InferenceOption(); o.devices = [0]; o.bound_option = getattr(BO, 'NPU_%d' % c); o.use_ort = True; o.buffer_count = K + 1
    engs.append(dx.InferenceEngine(path, o))
i = engs[0].get_input_tensors_info()[0]
rng = np.random.default_rng(0)
imgs = [np.ascontiguousarray(rng.integers(0, 256, size=i['shape']).astype(i['dtype'])) for _ in range(4)]
for e in engs:
    for _ in range(3):
        e.run([imgs[0]])
q, ts, lat = [], [], []
t0 = time.perf_counter()
for f in range(N):
    e = engs[f % len(engs)]
    ts.append(time.perf_counter()); q.append((e, e.run_async([imgs[f % 4]])))
    if len(q) >= K * len(engs):
        e_, r_ = q.pop(0); e_.wait(r_); lat.append((time.perf_counter() - ts[len(lat)]) * 1e3)
while q:
    e_, r_ = q.pop(0); e_.wait(r_); lat.append((time.perf_counter() - ts[len(lat)]) * 1e3)
wall = time.perf_counter() - t0
w = sorted(lat)
print(json.dumps(dict(file=path.split('/')[-1], cores=cores, in_flight_per_core=K, fps=round(N / wall, 1),
                      latency_mean_ms=round(sum(w) / len(w), 1), latency_p95_ms=round(w[int(0.95 * (len(w) - 1))], 1))))
