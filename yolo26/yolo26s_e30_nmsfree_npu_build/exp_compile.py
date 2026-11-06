"""Experiment: compile one ONNX (raw255 ABI: float 0..255 input, explicit Div255 already inside) with DX-COM and report which ops land on NPU / CPU.
Usage (WSL, dx_com env): python exp_compile.py <onnx> <out_dir> <aggressive:0|1> [opt_level]
Calibration: the same 8 TRAIN tensors (claude_b0_tile_01/MANIFEST.json), raw 0..255, full 640x640 letterbox. Writes <out_dir>/SPLIT.json."""
import os, sys, json, time, resource, collections
from pathlib import Path
import numpy as np
B0DIR = Path('/mnt/d/LIG-Runs/flir_yolo26/yolo26m_m1_wholegraph_20261022_v1/claude_b0_tile_01')
onx, out, aggr = Path(sys.argv[1]), Path(sys.argv[2]), bool(int(sys.argv[3])); opt = int(sys.argv[4]) if len(sys.argv) > 4 else 0
out.mkdir(parents=True, exist_ok=True)
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 ** 3,) * 2); os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
import torch, onnx, dx_com
m = json.loads((B0DIR / 'MANIFEST.json').read_text())
raw = [np.rint(np.load(v['path'], allow_pickle=False) * 255).astype(np.float32) for v in m['calibration']]
class Calib(torch.utils.data.Dataset):
    def __len__(self): return len(raw)
    def __getitem__(self, i): return torch.from_numpy(raw[i][0].copy())
g = onnx.load(str(onx)); ops = collections.Counter(n.op_type for n in g.graph.node)
t = time.monotonic()
dx_com.compile(model=str(onx), output_dir=str(out / 'compiler'), dataloader=torch.utils.data.DataLoader(Calib(), batch_size=1, num_workers=0, shuffle=False),
               calibration_num=8, calibration_method='ema', quantization_device='cpu', opt_level=opt, aggressive_partitioning=aggr, gen_log=True, export_html=False)
secs = time.monotonic() - t
files = list((out / 'compiler').glob('*.dxnn')); assert len(files) == 1
import importlib.util
parser = m['metadata_parser']; spec = importlib.util.spec_from_file_location('fr', parser['path']); rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
meta = rd.read(files[0]); gr = meta['graph']
res = dict(onnx=str(onx), aggressive=aggr, opt_level=opt, onnx_ops=dict(ops), compile_s=secs, dxnn=str(files[0]), bytes=files[0].stat().st_size,
           outputs=[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.graph.output],
           subgraphs=[dict(name=s.get('name'), device=s.get('device'), n_in=len(s.get('inputs', [])), outs=[o.get('name') for o in s.get('outputs', [])]) for s in gr.get('graphs', [])])
(out / 'SPLIT.json').write_text(json.dumps(res, indent=1))
print('SPLIT', json.dumps(res['subgraphs']), flush=True)
