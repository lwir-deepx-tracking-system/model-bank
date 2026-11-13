"""Extract three ResNet outputs from the fixed-shape reference ONNX for M1 probing."""
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
target = work / f'motip_backbone_{height}x{width}_probe.onnx'
names = [f'/detector/backbone/backbone.0/body/{level}/{block}/relu_2/Relu_output_0'
         for level, block in [('layer2', 'layer2.3'), ('layer3', 'layer3.5'), ('layer4', 'layer4.2')]]
original = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
extracted = onnx.utils.Extractor(original).extract_model(['image'], names)
onnx.checker.check_model(extracted)
onnx.save(extracted, str(target))
infos = {v.name: v for v in original.graph.value_info}
assert all(n in infos for n in names)
del original.graph.output[:]
original.graph.output.extend(infos[n] for n in names)
original_path = work / f'motip_detector_backbone_outputs_{height}x{width}_probe.onnx'
onnx.save(original, str(original_path))
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
rng = np.random.default_rng(20261113)
image = rng.standard_normal((1, 3, height, width), dtype=np.float32)
available = dict(image=image, mask=np.zeros((1, height, width), dtype=np.bool_))
def run(path):
    s = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
    return s.run(None, {i.name: available[i.name] for i in s.get_inputs()})
reference = run(original_path)
output = run(target)
errors = [float(np.max(np.abs(a-b))) for a, b in zip(reference, output)]
assert all(np.allclose(a,b,atol=1e-5,rtol=1e-4) for a,b in zip(reference,output))
result = dict(scope='fixed detector feature subgraph only, synthetic input; no ID and no M1',
              source=str(source), output=str(target), output_names=names,
              output_shapes=[list(x.shape) for x in output], max_abs_error=errors)
Path(__file__).with_name(f'motip_backbone_extract_{height}x{width}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
