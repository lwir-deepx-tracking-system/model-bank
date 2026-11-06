"""WSL (dx_com env). Six-head DXNN at 640x640 per the model bank's tensor_requirements.md: [1,4,80,80] [1,4,40,40] [1,4,20,20] [1,16,80,80] [1,16,40,40] [1,16,20,20], float32.
Recipe of claude_ft_deploy_o1_01/build_split6.py: same ONNX export flags, explicit Div(255) (raw 0..255 input), ema, 8 calibration tensors, opt_level 1, aggressive_partitioning False. Input window = the whole 640x640 letterboxed tensor.
Usage: python -B build640.py <name> <trace.pt>"""
import os, sys, json, time, hashlib, resource
from pathlib import Path
P = Path(__file__).resolve().parent
B0 = Path('/mnt/d/LIG-Runs/flir_yolo26/yolo26m_m1_wholegraph_20261022_v1/claude_b0_tile_01')
name, trace = sys.argv[1], sys.argv[2]; out = P / name; out.mkdir(exist_ok=False); tmp = P / 'tmp'; tmp.mkdir(exist_ok=True)
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 ** 3,) * 2); os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='', TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp))
import numpy as np, torch, onnx, onnxruntime as ort, dx_com
from onnx import helper, numpy_helper
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(2); torch.set_num_interop_threads(1)
m = json.loads((B0 / 'MANIFEST.json').read_text())
for v in m['calibration']: assert sha(v['path']) == v['SHA']
model = torch.jit.load(trace, map_location='cpu').eval()
xs = [np.ascontiguousarray(np.load(v['path'], allow_pickle=False)) for v in m['calibration']]; assert xs[0].shape == (1, 3, 640, 640), xs[0].shape
raw = [np.rint(x * 255).astype(np.float32) for x in xs]
with torch.inference_mode():
    tr = [[v.numpy().copy() for v in model(torch.from_numpy(x))] for x in xs]
    assert [v.shape for v in tr[0]] == [(1, 4, 8400), (1, 16, 8400)]
    base = out / 'base.onnx'
    torch.onnx.export(model, (torch.from_numpy(xs[0]),), str(base), input_names=['images'], output_names=['boxes', 'scores'], opset_version=17, dynamo=False, optimize=False, do_constant_folding=False, external_data=False)
SRC6 = ['/branch/0/2/Conv_output_0', '/branch/1/2/Conv_output_0', '/branch/2/2/Conv_output_0', '/branch/0/2_1/Conv_output_0', '/branch/1/2_1/Conv_output_0', '/branch/2/2_1/Conv_output_0']
NAMES6 = ['box_s8', 'box_s16', 'box_s32', 'cls_s8', 'cls_s16', 'cls_s32']
base6 = out / 'base6.onnx'; onnx.utils.extract_model(str(base), str(base6), ['images'], SRC6)
g = onnx.load(str(base6)); ren = dict(zip(SRC6, NAMES6))
for n in g.graph.node:
    for i, v in enumerate(n.output):
        if v in ren: n.output[i] = ren[v]
    for i, v in enumerate(n.input): assert v not in ren
for o in g.graph.output: o.name = ren[o.name]
for n in g.graph.node:
    for i, v in enumerate(n.input):
        if v == 'images': n.input[i] = 'explicit_normalized_images'
g.graph.initializer.append(numpy_helper.from_array(np.array(255, np.float32), 'explicit_divisor_255'))
g.graph.node.insert(0, helper.make_node('Div', ['images', 'explicit_divisor_255'], ['explicit_normalized_images'], name='ExplicitImageDiv255'))
onnx.checker.check_model(g, full_check=True); onx = out / (name + '.onnx'); onnx.save(g, str(onx))
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s = ort.InferenceSession(str(onx), sess_options=so, providers=['CPUExecutionProvider'])
want = [(1, 4, 80, 80), (1, 4, 40, 40), (1, 4, 20, 20), (1, 16, 80, 80), (1, 16, 40, 40), (1, 16, 20, 20)]; chk = []
for i, r in enumerate(raw):
    o = s.run(NAMES6, {'images': r}); assert [x.shape for x in o] == want and all(x.dtype == np.float32 for x in o)
    got = [np.concatenate([x.reshape(1, 4, -1) for x in o[:3]], 2), np.concatenate([x.reshape(1, 16, -1) for x in o[3:]], 2)]
    chk.append([float(np.max(np.abs(a - b))) for a, b in zip(got, tr[i])]); assert all(np.allclose(a, b, atol=1e-3, rtol=1e-4) for a, b in zip(got, tr[i])), chk
print('PARITY ok', chk, flush=True); del s; os.remove(base); os.remove(base6)
class C(torch.utils.data.Dataset):
    def __len__(self): return len(raw)
    def __getitem__(self, i): return torch.from_numpy(raw[i][0].copy())
t = time.monotonic()
dx_com.compile(model=str(onx), output_dir=str(out / 'compiler'), dataloader=torch.utils.data.DataLoader(C(), batch_size=1, num_workers=0, shuffle=False), calibration_num=8, calibration_method='ema', quantization_device='cpu', opt_level=1, aggressive_partitioning=False, gen_log=True, export_html=False)
f = list((out / 'compiler').glob('*.dxnn')); assert len(f) == 1
import importlib.util
spec = importlib.util.spec_from_file_location('fr', m['metadata_parser']['path']); rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
meta = rd.read(f[0]); (out / 'DXNN_METADATA.json').write_text(json.dumps(meta, indent=1, default=str))
g2 = meta['graph']; res = dict(name=name, trace=trace, trace_sha=sha(trace), dxnn=str(f[0]), sha=sha(f[0]), bytes=f[0].stat().st_size, compile_s=time.monotonic() - t, outputs=g2.get('outputs'), subgraphs=[dict(n=x.get('name'), d=x.get('device')) for x in g2.get('graphs', [])], parity=chk)
(out / 'RESULT.json').write_text(json.dumps(res, indent=1)); print('DONE', json.dumps({k: res[k] for k in ('name', 'sha', 'bytes', 'compile_s', 'outputs', 'subgraphs')}), flush=True)
