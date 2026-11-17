"""PC/ARM CPU correctness smoke for fixed-shape ONNX detector; never runs the NPU."""
import json
import sys
import time
from pathlib import Path
import os
import numpy as np
import onnxruntime as ort

model = Path(sys.argv[1])
output = Path(sys.argv[2])
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
session = ort.InferenceSession(str(model), sess_options=so, providers=['CPUExecutionProvider'])
rng = np.random.default_rng(20261113)
available = dict(image=rng.standard_normal((1, 3, 128, 160), dtype=np.float32),
                 mask=np.zeros((1, 128, 160), dtype=np.bool_))
inputs = {x.name: available[x.name] for x in session.get_inputs()}
start = time.perf_counter()
values = session.run(None, inputs)
elapsed = 1000 * (time.perf_counter() - start)
np.savez_compressed(output, **{x.name: v for x, v in zip(session.get_outputs(), values)})
print(json.dumps(dict(scope='single synthetic static 128x160 detector inference, not tracking or NPU',
                      providers=session.get_providers(), elapsed_ms=round(elapsed, 3),
                      shapes=[list(v.shape) for v in values], result=str(output))))
