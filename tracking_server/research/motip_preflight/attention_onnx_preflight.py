"""Export a tiny fixed-shape MOTIP reference attention component, not the model."""
import json
import sys
import types
from pathlib import Path
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

root = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ['MOTIP_UPSTREAM'])))
sys.modules['MultiScaleDeformableAttention'] = types.ModuleType('MultiScaleDeformableAttention')
from models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch


class Attention(torch.nn.Module):
    def forward(self, value, sampling, weights):
        # Two small, fixed feature levels. This is not the deployed input shape.
        shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long)
        return ms_deform_attn_core_pytorch(value, shapes, sampling, weights)


torch.manual_seed(7)
model = Attention().eval()
value = torch.randn(1, 20, 2, 4)
sampling = torch.rand(1, 3, 2, 2, 2, 2) * 1.4 - .2
weights = torch.softmax(torch.randn(1, 3, 2, 4), -1).reshape(1, 3, 2, 2, 2)
dest = root / 'attention_reference_static.onnx'
with torch.no_grad():
    reference = model(value, sampling, weights).numpy()
    torch.onnx.export(model, (value, sampling, weights), str(dest), opset_version=17,
                      input_names=['value', 'sampling', 'weights'], output_names=['output'],
                      do_constant_folding=True)
onnx_model = onnx.load(str(dest))
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(str(dest), providers=['CPUExecutionProvider'])
observed = session.run(None, {'value': value.numpy(), 'sampling': sampling.numpy(),
                              'weights': weights.numpy()})[0]
diff = float(np.max(np.abs(reference - observed)))
assert np.allclose(reference, observed, rtol=1e-4, atol=1e-5)
np.savez(root / 'attention_reference_fixture.npz', value=value.numpy(), sampling=sampling.numpy(),
         weights=weights.numpy(), expected=reference)
result = dict(scope='tiny fixed-shape attention component, no complete model or M1',
              opset=17, operators=sorted(set(n.op_type for n in onnx_model.graph.node)),
              max_abs_error=diff, inputs_fixed=True, m1_compiler_tested=False)
(root / 'attention_onnx_preflight.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
