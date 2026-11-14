"""Device: where does the time of one decoder-chunk call go?  wall time of run() vs the runtime's own latency and NPU
inference time, and the cost of only converting / copying the inputs. Usage: python ftt_call_cost.py <chunk.dxnn> [core] [n]"""
import json
import sys
import time

import dx_engine as dx
import numpy as np

path = sys.argv[1]
core = int(sys.argv[2]) if len(sys.argv) > 2 else 2
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
o = dx.InferenceOption(); o.devices = [0]; o.bound_option = getattr(dx.InferenceOption.BOUND_OPTION, 'NPU_%d' % core); o.use_ort = True
e = dx.InferenceEngine(path, o)
ins = e.get_input_tensors_info()
rng = np.random.default_rng(0)
feed = [np.ascontiguousarray(rng.standard_normal(t['shape']).astype(t['dtype'])) for t in ins]
for _ in range(5):
    e.run(feed)
w = []
for _ in range(n):
    t = time.perf_counter(); e.run(feed); w.append((time.perf_counter() - t) * 1e3)
t = time.perf_counter()
for _ in range(n):
    [np.ascontiguousarray(f.reshape(ti['shape']).astype(ti['dtype'])) for f, ti in zip(feed, ins)]
prep = (time.perf_counter() - t) / n * 1e3


def safe(f):
    try:
        return round(float(f()), 2)
    except Exception as x:
        return str(x)[:60]


print(json.dumps(dict(file=path.split('/')[-1], inputs=[(t['name'], list(t['shape']), t['dtype'].__name__) for t in ins],
                      outputs=[(t['name'], list(t['shape']), t['dtype'].__name__) for t in e.get_output_tensors_info()],
                      run_wall_ms_mean=round(sum(w) / n, 2), run_wall_ms_min=round(min(w), 2),
                      runtime_latency_mean_ms=safe(lambda: e.get_latency_mean() / 1000), npu_inference_mean_ms=safe(lambda: e.get_npu_inference_time_mean() / 1000),
                      input_prep_ms=round(prep, 2), compile_type=safe(lambda: 0) if False else str(e.get_compile_type()) if hasattr(e, 'get_compile_type') else None)))
