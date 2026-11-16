"""Static ID-head ONNX export probe with synthetic trajectory state, not a tracker."""
import json
import os
import sys
import types
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parent
upstream = Path(os.environ['MOTIP_UPSTREAM'])
sys.path.insert(0, str(upstream))
sys.modules['MultiScaleDeformableAttention'] = types.ModuleType('MultiScaleDeformableAttention')
os.chdir(upstream)
import models.deformable_detr.backbone as backbone
backbone.is_main_process = lambda: False
from utils.misc import yaml_to_dict
from configs.util import load_super_config
from models.motip import build


class IDWrapper(torch.nn.Module):
    def __init__(self, trajectory_modeling, decoder):
        super().__init__()
        self.trajectory_modeling = trajectory_modeling
        self.decoder = decoder

    def forward(self, trajectory_features, unknown_features, trajectory_id_labels,
                trajectory_times, unknown_times, trajectory_masks, unknown_masks):
        seq = dict(trajectory_features=trajectory_features, unknown_features=unknown_features,
                   trajectory_id_labels=trajectory_id_labels, trajectory_times=trajectory_times,
                   unknown_times=unknown_times, trajectory_masks=trajectory_masks,
                   unknown_masks=unknown_masks)
        seq = self.trajectory_modeling(seq)
        logits, _, _ = self.decoder(seq, use_decoder_checkpoint=False)
        return logits


result_path = root / 'motip_id_onnx_probe.json'
output_path = (Path(os.environ['MOTIP_ARTIFACTS']) / 'motip_id_fixed_probe.onnx')
result = dict(scope='synthetic fixed-shape ID head only; not complete tracking or NPU',
              complete=False, output=str(output_path))
try:
    config = yaml_to_dict(str(upstream / 'configs/r50_deformable_detr_motip_dancetrack.yaml'))
    config = load_super_config(config, config['SUPER_CONFIG_PATH'])
    config['DEVICE'] = 'cpu'
    model, _ = build(config)
    state = torch.load(str(Path(os.environ['MOTIP_ARTIFACTS']) / 'motip_dancetrack_model_only.pth'),
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    wrapper = IDWrapper(model.trajectory_modeling, model.id_decoder).eval()
    torch.manual_seed(8)
    inputs = (torch.randn(1, 1, 2, 2, 256), torch.randn(1, 1, 1, 2, 256),
              torch.tensor([[[[0, 1], [0, 1]]]], dtype=torch.int64),
              torch.tensor([[[[0, 0], [1, 1]]]], dtype=torch.int64),
              torch.tensor([[[[2, 2]]]], dtype=torch.int64),
              torch.zeros(1, 1, 2, 2, dtype=torch.bool),
              torch.zeros(1, 1, 1, 2, dtype=torch.bool))
    names = ['trajectory_features', 'unknown_features', 'trajectory_id_labels',
             'trajectory_times', 'unknown_times', 'trajectory_masks', 'unknown_masks']
    with torch.no_grad():
        expected = wrapper(*inputs).numpy()
        torch.onnx.export(wrapper, inputs, str(output_path), opset_version=17,
                          input_names=names, output_names=['id_logits'], do_constant_folding=True)
    import onnx
    import onnxruntime as ort
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    sess = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    provided = {name: value.numpy() for name, value in zip(names, inputs)}
    actual = sess.run(None, {item.name: provided[item.name] for item in sess.get_inputs()})[0]
    np.savez_compressed(root / 'motip_id_fixture.npz', **provided, expected=actual)
    result.update(complete=True, output_shape=list(actual.shape),
                  max_abs_error=float(np.max(np.abs(expected-actual))),
                  numerical_match=bool(np.allclose(expected, actual, rtol=1e-3, atol=1e-4)),
                  operators=sorted({n.op_type for n in graph.graph.node}),
                  m1_compiler_tested=False)
except Exception as exc:
    result.update(error_type=type(exc).__name__, error=str(exc)[:1000])
    raise
finally:
    result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
