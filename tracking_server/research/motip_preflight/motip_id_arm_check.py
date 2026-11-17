"""Synthetic fixed-state ID-head comparison using ARM CPU only."""
import json
import time
from pathlib import Path
import os
import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parent
fixture = np.load(root / 'motip_id_fixture.npz', allow_pickle=False)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
sess = ort.InferenceSession(str(root / 'motip_id_fixed_probe.onnx'), sess_options=opts,
                            providers=['CPUExecutionProvider'])
start = time.perf_counter()
actual = sess.run(None, {i.name: fixture[i.name] for i in sess.get_inputs()})[0]
elapsed = round((time.perf_counter() - start) * 1000, 3)
error = float(np.max(np.abs(actual-fixture['expected'])))
ok = bool(np.allclose(actual, fixture['expected'], rtol=1e-3, atol=1e-4))
print(json.dumps(dict(scope='synthetic fixed state, CPU only, not tracking', matches=ok,
                      max_abs_error=error, elapsed_ms=elapsed, npu_used=False)))
if not ok:
    raise SystemExit(1)
