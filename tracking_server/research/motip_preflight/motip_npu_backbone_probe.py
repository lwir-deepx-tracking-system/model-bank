"""Four held-frame M1 inference for a tiny 128x160 backbone; not deployment."""
import json
import os
import time
from pathlib import Path
import dx_engine as dx
import numpy as np

root = Path(__file__).resolve().parent
height = int(os.environ.get('MOTIP_PROBE_H','128'))
width = int(os.environ.get('MOTIP_PROBE_W','160'))
samples = np.load(root/f'motip_backbone_{height}x{width}_calib_probe.npz', allow_pickle=False)
opts = dx.InferenceOption()
opts.devices = [0]
opts.bound_option = dx.InferenceOption.BOUND_OPTION.NPU_2
opts.use_ort = True
engine = dx.InferenceEngine(str(root/f'motip_backbone_raw_{height}x{width}_probe.dxnn'), opts)
names = [x['name'] for x in engine.get_output_tensors_info()]
outputs = {}
latencies = []
for i in samples['test_indices']:
    image = np.ascontiguousarray(samples['images'][int(i)][None])
    start = time.perf_counter()
    values = engine.run(image)
    latencies.append(round((time.perf_counter()-start)*1000, 3))
    for j, value in enumerate(values):
        outputs[f'frame_{int(i)}_level_{j}'] = np.asarray(value).copy()
np.savez_compressed(root/f'motip_npu_backbone_outputs_{height}x{width}.npz', **outputs)
print(json.dumps(dict(scope=f'four frames; {height}x{width} feature extractor only, diagnostic calibration',
                      indices=samples['test_indices'].tolist(), output_names=names,
                      latency_ms=latencies, finite=all(np.isfinite(x).all() for x in outputs.values()))))
