"""Read-only numerical check of a tiny ONNX component using ARM CPU only."""
import json
from pathlib import Path
import os
import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parent
fixture = np.load(root / 'attention_reference_fixture.npz', allow_pickle=False)
session = ort.InferenceSession(str(root / 'attention_reference_static.onnx'), providers=['CPUExecutionProvider'])
actual = session.run(None, {k: fixture[k] for k in ('value', 'sampling', 'weights')})[0]
error = float(np.max(np.abs(actual - fixture['expected'])))
success = bool(np.allclose(actual, fixture['expected'], rtol=1e-4, atol=1e-5))
print(json.dumps(dict(provider=session.get_providers(), max_abs_error=error,
                      matches=success, full_model=False, npu_used=False)))
if not success:
    raise SystemExit(1)
