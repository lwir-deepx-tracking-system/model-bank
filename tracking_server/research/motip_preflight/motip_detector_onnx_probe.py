"""Fixed-shape DETR-only export probe; not MOTIP tracking, DXNN, or deployment."""
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

root = Path(__file__).resolve().parent
upstream = Path(os.environ['MOTIP_UPSTREAM'])
sys.path.insert(0, str(upstream))
sys.modules['MultiScaleDeformableAttention'] = types.ModuleType('MultiScaleDeformableAttention')
os.chdir(upstream)
from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch
MSDeformAttnFunction.apply = staticmethod(lambda value, shapes, starts, sample, weights, step:
                                          ms_deform_attn_core_pytorch(value, shapes, sample, weights))
import models.deformable_detr.backbone as backbone
backbone.is_main_process = lambda: False
from utils.misc import yaml_to_dict, distributed_device
from configs.util import load_super_config
from models.motip import build
from utils.nested_tensor import NestedTensor


class Wrapper(torch.nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, image, mask):
        out = self.detector(NestedTensor(image, mask))
        return out['pred_logits'], out['pred_boxes'], out['outputs']


height = int(os.environ.get('MOTIP_PROBE_H', '128'))
width = int(os.environ.get('MOTIP_PROBE_W', '160'))
result_path = root / ('motip_detector_onnx_probe.json' if (height, width) == (128, 160)
                      else f'motip_detector_{height}x{width}_onnx_probe.json')
output_path = (Path(os.environ['MOTIP_ARTIFACTS']) / f'motip_detector_{height}x{width}_probe.onnx')
result = dict(scope=f'fixed {height}x{width} detector only; no ID decoder, NPU or production use',
              complete=False, output=str(output_path))
try:
    config = yaml_to_dict(str(upstream / 'configs/r50_deformable_detr_motip_dancetrack.yaml'))
    config = load_super_config(config, config['SUPER_CONFIG_PATH'])
    device = distributed_device()
    config['DEVICE'] = str(device)
    model, _ = build(config)
    state = torch.load(str(Path(os.environ['MOTIP_ARTIFACTS']) / 'motip_dancetrack_model_only.pth'),
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    wrapper = Wrapper(model.detr).eval().to(device)
    torch.manual_seed(7)
    image = torch.randn(1, 3, height, width, device=device)
    mask = torch.zeros((1, height, width), dtype=torch.bool, device=device)
    start = time.perf_counter()
    with torch.no_grad():
        reference = wrapper(image, mask)
        torch.onnx.export(wrapper, (image, mask), str(output_path), opset_version=17,
                          input_names=['image', 'mask'],
                          output_names=['logits', 'boxes', 'features'], do_constant_folding=True)
    import onnx
    import onnxruntime as ort
    import numpy as np
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    inputs = {'image': image.cpu().numpy(), 'mask': mask.cpu().numpy()}
    inputs = {k: inputs[k] for k in [item.name for item in session.get_inputs()]}
    actual = session.run(None, inputs)
    errors = [float(np.max(np.abs(a - b.detach().cpu().numpy()))) for a, b in zip(actual, reference)]
    result.update(complete=True, max_abs_error=errors,
                  output_shapes=[list(a.shape) for a in actual],
                  operators=sorted({n.op_type for n in graph.graph.node}),
                  elapsed_seconds=round(time.perf_counter()-start, 2),
                  numerical_match=all(np.allclose(a, b.detach().cpu().numpy(), rtol=1e-3, atol=1e-4)
                                      for a, b in zip(actual, reference)))
except Exception as exc:
    result.update(error_type=type(exc).__name__, error=str(exc)[:1000])
    raise
finally:
    result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
