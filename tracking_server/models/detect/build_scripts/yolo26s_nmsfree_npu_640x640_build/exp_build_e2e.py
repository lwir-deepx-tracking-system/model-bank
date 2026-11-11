"""WSL (dx_com env): ONNX variants of the YOLO26s one2one (NMS-free) model that put the post-processing INSIDE the graph, to test how much of it DX-COM can run on the NPU.
Input: float 0..255 (raw255 ABI), explicit Div(255) in front (same as build_s_raw255.py). Trace: S_o2o.pt (outputs raw ltrb distances (1,4,8400) and class logits (1,16,8400)).
Decode = runtime.Decoder: anchor = grid centre (+.5) per stride 8/16/32 (row-major), x1y1 = (anchor - lt) * stride, x2y2 = (anchor + rb) * stride, scores = sigmoid(logits).
Post = ultralytics Detect.postprocess for end2end (agnostic False): (1) top-300 anchors by max class score, (2) top-300 over the 300*16 flattened scores, rows [x1,y1,x2,y2,score,cls].
The post-processing is exported as its own small ONNX (pure torch) and merged behind the exported backbone+heads ONNX (a traced ScriptModule cannot be nested in another export).
Variants:  D  decode + sigmoid               -> boxes (1,8400,4), scores (1,8400,16)
           M  D + per-anchor max            -> boxes (1,8400,4), scores (1,8400,16), maxscore (1,8400)
           E  full end-to-end               -> output0 (1,300,6)
Usage: python exp_build_e2e.py <D|M|E> <out.onnx>   (checks the merged ONNX against a numpy reference on 2 calibration tensors)"""
import sys, json, collections
from pathlib import Path
import numpy as np, torch, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, compose
from torch import nn
P = Path(__file__).resolve().parent
B0DIR = Path('<YOUR_PATH>')
variant, outp = sys.argv[1], Path(sys.argv[2]); outp.parent.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
tr = torch.jit.load(str(P / 'S_o2o.pt'), map_location='cpu').eval()
H = W = 640; NC = 16; K = 300
pts, strd = [], []
for s in (8, 16, 32):
    y, x = np.meshgrid(np.arange(H // s, dtype=np.float32) + .5, np.arange(W // s, dtype=np.float32) + .5, indexing='ij')
    pts.append(np.stack([x.ravel(), y.ravel()])); strd.append(np.full((1, x.size), s, np.float32))
ANCHOR = torch.from_numpy(np.concatenate(pts, 1)[None]); STRIDE = torch.from_numpy(np.concatenate(strd, 1)[None])   # (1,2,N) (1,1,N)
N = ANCHOR.shape[-1]; assert N == 8400
man = json.loads((B0DIR / 'MANIFEST.json').read_text())
x01 = [np.load(v['path'], allow_pickle=False) for v in man['calibration'][:2]]; xs = [np.rint(a * 255).astype(np.float32) for a in x01]


class Post(nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode = mode
        self.register_buffer('anchor', ANCHOR); self.register_buffer('stride', STRIDE)
    def forward(self, b, lg):
        low = (self.anchor - b[:, :2]) * self.stride; high = (self.anchor + b[:, 2:]) * self.stride
        boxes = torch.cat([low, high], 1).transpose(1, 2)            # (1,N,4) xyxy
        sc = torch.sigmoid(lg).transpose(1, 2)                       # (1,N,16)
        if self.mode == 'D': return boxes, sc
        if self.mode in ('M3', 'T3', 'E3'):   # per-anchor max as a chain of elementwise Max (no ReduceMax)
            mx = sc[:, :, 0]
            for c in range(1, NC): mx = torch.maximum(mx, sc[:, :, c])
            if self.mode == 'M3': return boxes, sc, mx
            if self.mode == 'T3':
                v, ori = mx.topk(K, dim=1); return boxes, sc, v, ori
        else: mx = sc.max(-1)[0]
        if self.mode == 'M': return boxes, sc, mx
        ori = mx.topk(K, dim=1)[1]                                   # (1,K) anchors
        s2 = torch.gather(sc, 1, ori[:, :, None].expand(-1, -1, NC)).flatten(1)   # (1,K*NC)
        sv, idx = s2.topk(K, dim=1)
        cls = (idx % NC).to(torch.float32); anc = torch.gather(ori, 1, idx // NC)
        bx = torch.gather(boxes, 1, anc[:, :, None].expand(-1, -1, 4))
        return torch.cat([bx, sv[:, :, None], cls[:, :, None]], 2)   # (1,K,6)


out_names = {'D': ['boxes', 'scores'], 'M': ['boxes', 'scores', 'maxscore'], 'M3': ['boxes', 'scores', 'maxscore'], 'T3': ['boxes', 'scores', 'topv', 'topi'], 'E': ['output0'], 'E3': ['output0']}[variant]
tmp = outp.parent
# 1) backbone + heads (recipe of build_s_raw255.py) + explicit Div255
with torch.inference_mode():
    torch.onnx.export(tr, (torch.from_numpy(x01[0]),), str(tmp / 'base.onnx'), input_names=['images'], output_names=['rawbox', 'rawlogit'], opset_version=17, dynamo=False, optimize=False, do_constant_folding=False, external_data=False)
g = onnx.load(str(tmp / 'base.onnx'))
for node in g.graph.node:
    for i, v in enumerate(node.input):
        if v == 'images': node.input[i] = 'explicit_normalized_images'
g.graph.initializer.append(numpy_helper.from_array(np.array(255, np.float32), 'explicit_divisor_255'))
g.graph.node.insert(0, helper.make_node('Div', ['images', 'explicit_divisor_255'], ['explicit_normalized_images'], name='ExplicitImageDiv255'))
# 2) post-processing as its own ONNX
pm = Post(variant).eval()
with torch.inference_mode():
    torch.onnx.export(pm, (torch.zeros(1, 4, N), torch.zeros(1, NC, N)), str(tmp / 'post.onnx'), input_names=['pbox', 'plogit'], output_names=out_names, opset_version=17, dynamo=False, optimize=False, do_constant_folding=False, external_data=False)
p = onnx.load(str(tmp / 'post.onnx'))
mg = compose.merge_models(g, p, io_map=[('rawbox', 'pbox'), ('rawlogit', 'plogit')])
mg.ir_version = g.ir_version
onnx.checker.check_model(mg, full_check=True); onnx.save(mg, str(outp))
# 3) check against numpy reference
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ss = ort.InferenceSession(str(outp), sess_options=so, providers=['CPUExecutionProvider'])
sb = ort.InferenceSession(str(tmp / 'base.onnx'), sess_options=so, providers=['CPUExecutionProvider'])
chk = []
for i, x in enumerate(xs):
    got = ss.run(out_names, {'images': x})
    b, lg = sb.run(['rawbox', 'rawlogit'], {'images': x01[i]})
    low = (ANCHOR.numpy() - b[:, :2]) * STRIDE.numpy(); high = (ANCHOR.numpy() + b[:, 2:]) * STRIDE.numpy()
    xyxy = np.concatenate([low, high], 1)[0].T; s = (1 / (1 + np.exp(-lg)))[0].T
    c = dict(image=i)
    if variant in ('D', 'M', 'M3'): c.update(boxes=float(np.max(np.abs(got[0][0] - xyxy))), scores=float(np.max(np.abs(got[1][0] - s))))
    if variant in ('M', 'M3'): c['maxscore'] = float(np.max(np.abs(got[2][0] - s.max(1))))
    if variant in ('E', 'E3'):
        ori = np.argsort(-s.max(1), kind='stable')[:K]; flat = s[ori].reshape(-1); idx = np.argsort(-flat, kind='stable')[:K]
        rows = np.concatenate([xyxy[ori[idx // NC]], flat[idx][:, None], (idx % NC)[:, None].astype(np.float32)], 1)
        o = got[0][0]; c.update(rows_vs_numpy_maxabs=float(np.max(np.abs(o - rows))), same_class_rows=int(np.sum(o[:, 5] == rows[:, 5])), top_score=float(o[0, 4]), n_over_025=int(np.sum(o[:, 4] >= .25)))
    chk.append(c)
print('OPS', dict(collections.Counter(n.op_type for n in mg.graph.node)))
print('CHECK', json.dumps(chk))
