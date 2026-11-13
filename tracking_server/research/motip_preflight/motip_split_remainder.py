"""Verify fixed-shape ONNX detector split at the three ResNet feature outputs."""
import json
import os
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

work = Path(os.environ['MOTIP_ARTIFACTS'])
height = int(os.environ.get('MOTIP_PROBE_H','128'))
width = int(os.environ.get('MOTIP_PROBE_W','160'))
source = work / f'motip_detector_{height}x{width}_probe.onnx'
features = [f'/detector/backbone/backbone.0/body/{level}/{block}/relu_2/Relu_output_0'
            for level, block in [('layer2', 'layer2.3'), ('layer3', 'layer3.5'), ('layer4', 'layer4.2')]]
target = work / f'motip_detector_remainder_{height}x{width}_probe.onnx'
result_path = Path(__file__).with_name(f'motip_split_remainder_{height}x{width}.json')
result = dict(scope=f'PC FP32 static {height}x{width} detector split only; no NPU quantization or ID', success=False)
try:
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
    remainder = onnx.utils.Extractor(graph).extract_model(features + ['mask'], ['logits', 'boxes', 'features'])
    onnx.checker.check_model(remainder)
    onnx.save(remainder, str(target))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    def session(p):
        return ort.InferenceSession(str(p), sess_options=opts, providers=['CPUExecutionProvider'])
    full = session(source)
    backbone = session(work / f'motip_backbone_{height}x{width}_probe.onnx')
    rest = session(target)
    image = np.random.default_rng(20261113).standard_normal((1,3,height,width), dtype=np.float32)
    mask = np.zeros((1,height,width), dtype=np.bool_)
    available = dict(image=image, mask=mask)
    full_out = full.run(None, {x.name: available[x.name] for x in full.get_inputs()})
    feature_values = backbone.run(None, {'image': image})
    provided = dict(zip(features, feature_values))
    provided['mask'] = mask
    rest_out = rest.run(None, {x.name: provided[x.name] for x in rest.get_inputs()})
    errors = [float(np.max(np.abs(a-b))) for a,b in zip(full_out, rest_out)]
    matched = all(np.allclose(a,b,rtol=1e-3,atol=1e-4) for a,b in zip(full_out, rest_out))
    result.update(success=bool(matched), max_abs_errors=errors,
                  remainder_inputs=[x.name for x in rest.get_inputs()], output=str(target))
    if not matched:
        raise RuntimeError('Split outputs did not match')
except Exception as exc:
    result.update(error_type=type(exc).__name__, error=str(exc)[:1000])
    raise
finally:
    result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
