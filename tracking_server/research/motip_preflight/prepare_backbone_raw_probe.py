"""Build raw 8-bit-input preprocessing into a small backbone test graph."""
import json
import os
import zipfile
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

root = Path(__file__).resolve().parent
work = Path(os.environ['MOTIP_ARTIFACTS'])
height = int(os.environ.get('MOTIP_PROBE_H','128'))
width = int(os.environ.get('MOTIP_PROBE_W','160'))
source = work / f'motip_backbone_{height}x{width}_probe.onnx'
target = work / f'motip_backbone_raw_{height}x{width}_probe.onnx'
archive = Path(os.environ['LWIR_SOURCE_ARCHIVE'])
graph = onnx.load(str(source))
for node in graph.graph.node:
    for i, name in enumerate(node.input):
        if name == 'image':
            node.input[i] = 'normalized_image'
for name, value in [('raw_divisor', np.array(255, np.float32)),
                    ('raw_mean', np.array([.485,.456,.406], np.float32).reshape(1,3,1,1)),
                    ('raw_std', np.array([.229,.224,.225], np.float32).reshape(1,3,1,1))]:
    graph.graph.initializer.append(numpy_helper.from_array(value, name))
for i, node in enumerate([
    helper.make_node('Div', ['image','raw_divisor'], ['scaled_raw']),
    helper.make_node('Sub', ['scaled_raw','raw_mean'], ['centered_raw']),
    helper.make_node('Div', ['centered_raw','raw_std'], ['normalized_image']),
]):
    graph.graph.node.insert(i, node)
onnx.checker.check_model(graph)
onnx.save(graph, str(target))

samples = []
meta = []
with zipfile.ZipFile(archive) as z:
    for seq in ('seq_001','seq_014'):
        prefix = f'LWIR_labels/opi_sequences/{seq}/images/'
        names = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith('.png'))[:6]
        assert len(names) == 6
        for n in names:
            arr = cv2.imdecode(np.frombuffer(z.read(n),dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            assert arr is not None and arr.ndim == 2 and arr.dtype in (np.uint8, np.uint16)
            bits = 16 if arr.dtype == np.uint16 else 8
            if bits == 16:
                arr = (arr // 256).astype(np.uint8)
            if arr.shape != (height,width):
                arr = cv2.resize(arr, (width,height), interpolation=cv2.INTER_LINEAR)
            rgb = np.repeat(arr[:,:,None],3,axis=2)
            samples.append(rgb)
            meta.append(dict(sequence=seq, frame=n.rsplit('/',1)[-1], bits=bits))
data = np.stack(samples)
np.savez_compressed(work / f'motip_backbone_{height}x{width}_calib_probe.npz', images=data,
                    calibration_indices=np.array([0,1,2,3,6,7,8,9]),
                    test_indices=np.array([4,5,10,11]))
opts = ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
a = ort.InferenceSession(str(source), sess_options=opts, providers=['CPUExecutionProvider'])
b = ort.InferenceSession(str(target), sess_options=opts, providers=['CPUExecutionProvider'])
raw = data[4].transpose(2,0,1)[None].astype(np.float32)
norm = (raw/255.0 - np.array([.485,.456,.406],np.float32).reshape(1,3,1,1)) / np.array([.229,.224,.225],np.float32).reshape(1,3,1,1)
x = a.run(None,{'image':norm}); y=b.run(None,{'image':raw})
errors=[float(np.max(np.abs(i-j))) for i,j in zip(x,y)]
assert all(np.allclose(i,j,rtol=1e-4,atol=1e-5) for i,j in zip(x,y))
report = dict(scope=f'diagnostic {height}x{width}, not deployment; real-frame calibration, no independent accuracy claim',
              preprocessing='raw grayscale duplicated to RGB; 16-bit diagnostic conversion integer divide 256',
              calibration_indices=[0,1,2,3,6,7,8,9], test_indices=[4,5,10,11], samples=meta,
              float_preprocessing_max_abs_error=errors)
(root/f'backbone_raw_preflight_{height}x{width}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
