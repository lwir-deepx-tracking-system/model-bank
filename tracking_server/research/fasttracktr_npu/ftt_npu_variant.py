"""FastTrackTr, NPU variant: the whole per-frame network as ONE static graph for the DX-M1 (speed test with random weights;
the same module is what would be trained). Changes against the official RT-DETR-cross decoder (all because the M1 has no
TopK / GridSample / Log on the NPU, see the MOTIP operator table):
  1. queries = the 300 cells of the coarsest level (15x20 at 480x640) instead of TopK over all cells (no TopK / Gather)
  2. decoder cross-attention = plain multi-head attention over the levels after the first (30x40 + 15x20 + 8x10 = 1580 keys,
     softmax width <= 3072) with a fixed sine position code on the keys, instead of deformable sampling (no GridSample)
  3. box refinement carries the unactivated box (ref_unact += bbox_head(out)) instead of inverse_sigmoid(sigmoid(.)) (no Log)
  4. the previous frame's decoder output comes in as an input; on the first frame it is zeros (the official code uses tgt
     itself there; a 0/1 switch input made DX-COM put the mixing on the CPU and fail in calibration, so: train with zeros)
Everything else (backbone, hybrid encoder, input projections, encoder head, query position MLP, self-attention, FFN, norms,
score / box / id heads) is the official module with its own weights.

Usage (WSL / PC):  python ftt_npu_variant.py export <config name> <out dir> [code dir]
                   python ftt_npu_variant.py compile <config name> <out dir>     (dx-com env)"""
import glob
import json
import math
import os
import sys
import time
import types

mode, cfg_name, out = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
CODE = sys.argv[4] if len(sys.argv) > 4 else os.environ['FTT_CODE']   # clone of github.com/liaopan-lp/fasttracktr
FRAMES = os.environ.get('FTT_FRAMES', '')   # folder with <seq>/images/*.png 8-bit frames for calibration
H, W = 480, 640
os.makedirs(out, exist_ok=True)
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(3)
ONNX = os.path.join(out, cfg_name + '_npu_full_480x640.onnx')
CAL = os.path.join(out, 'calib.npz')


def frames(n):
    files = sorted(glob.glob(os.path.join(FRAMES, 'seq_003', 'images', '*.png')))[:n]
    from PIL import Image
    return [np.repeat(np.asarray(Image.open(f).convert('L'), dtype=np.float32)[None], 3, 0) for f in files]


def sine_pos(h, w, dim=256, temperature=10000.):
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
    pd = dim // 4
    om = 1. / (temperature ** (torch.arange(pd, dtype=torch.float32) / pd))
    ox, oy = gx.flatten()[:, None] * om[None], gy.flatten()[:, None] * om[None]
    return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], 1)[None]   # [1, h*w, dim]


class MHA(nn.Module):
    """Multi-head attention written as MatMul / Softmax / Reshape only (export-friendly)."""
    def __init__(self, dim=256, heads=8, mha=None):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        if mha is not None:   # reuse nn.MultiheadAttention weights
            wq, wk, wv = mha.in_proj_weight.chunk(3)
            bq, bk, bv = mha.in_proj_bias.chunk(3)
            for lin, w_, b_ in ((self.q, wq, bq), (self.k, wk, bk), (self.v, wv, bv)):
                lin.weight.data.copy_(w_); lin.bias.data.copy_(b_)
            self.o.weight.data.copy_(mha.out_proj.weight); self.o.bias.data.copy_(mha.out_proj.bias)

    def forward(self, q, k, v):
        B, Lq, _ = q.shape
        Lk = k.shape[1]
        q = self.q(q).reshape(B, Lq, self.h, self.d).transpose(1, 2)
        k = self.k(k).reshape(B, Lk, self.h, self.d).permute(0, 2, 3, 1)
        v = self.v(v).reshape(B, Lk, self.h, self.d).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k) * (1.0 / math.sqrt(self.d)), dim=-1)
        return self.o(torch.matmul(a, v).transpose(1, 2).reshape(B, Lq, self.h * self.d))


class FullNPU(nn.Module):
    def __init__(self, detr):
        super().__init__()
        self.backbone, self.encoder, dec = detr.backbone, detr.encoder, detr.decoder
        self.input_proj, self.enc_output, self.enc_bbox_head = dec.input_proj, dec.enc_output, dec.enc_bbox_head
        self.query_pos_head, self.num_levels = dec.query_pos_head, dec.num_levels
        self.dec_bbox_head, self.dec_score_head, self.dec_id_head = dec.dec_bbox_head, dec.dec_score_head, dec.dec_id_head
        self.layers = dec.decoder.layers
        self.sa = nn.ModuleList(MHA(mha=l.self_attn) for l in self.layers)
        self.ca = nn.ModuleList(MHA() for _ in self.layers)   # new: dense cross-attention (random weights here)
        x = torch.zeros(1, 3, H, W)
        with torch.no_grad():
            shapes = [p.shape[-2:] for p in self.proj(self.encoder(self.backbone(x)))]
        self.shapes = [tuple(int(v) for v in s) for s in shapes]
        assert self.shapes[2][0] * self.shapes[2][1] == 300, self.shapes
        self.kv_from = int(os.environ.get('FTT_KV_FROM', '1'))   # cross-attention keys: levels kv_from.. (1 = 30x40 + 15x20 = 1500, 2 = 15x20 = 300)
        self.register_buffer('kpos', torch.cat([sine_pos(h, w) for h, w in self.shapes[self.kv_from:]], 1))
        anchors, valid = dec._generate_anchors([list(s) for s in self.shapes])
        a0 = sum(h * w for h, w in self.shapes[:2])
        a = anchors[:, a0:a0 + 300]
        assert torch.isfinite(a).all(), 'coarse-level anchors must be finite'
        self.register_buffer('anchors_q', a)

    def proj(self, feats):
        p = [self.input_proj[i](f) for i, f in enumerate(feats)]
        for i in range(len(p), self.num_levels):
            p.append(self.input_proj[i](feats[-1] if i == len(feats) else p[-1]))
        return p

    def part_a(self, image):   # per frame, no state: backbone, encoder, projections, encoder head -> decoder inputs
        feats = self.proj(self.encoder(self.backbone(image * (1.0 / 255.0))))
        flat = [f.flatten(2).transpose(1, 2) for f in feats]                       # [1, h*w, 256]
        mem_q = self.enc_output(flat[2])                                             # 300 coarse cells
        return mem_q, self.enc_bbox_head(mem_q) + self.anchors_q, torch.cat(flat[self.kv_from:], 1)

    def part_a_maps(self, image):   # NPU-only part A: the projected maps the decoder reads (levels 1, 2), NCHW; the host flattens them
        feats = self.proj(self.encoder(self.backbone(image * (1.0 / 255.0))))
        return feats[1], feats[2]

    def part_ae(self, image):   # NPU-only part A + encoder head on the 15x20 level, kept 4-D (NHWC): kv, tgt, ref_unact
        feats = self.proj(self.encoder(self.backbone(image * (1.0 / 255.0))))
        x = feats[2].permute(0, 2, 3, 1)                                   # [1, 15, 20, 256]
        mem = self.enc_output(x)
        h_, w_ = x.shape[1], x.shape[2]
        return x, mem, self.enc_bbox_head(mem) + self.anchors_q.reshape(1, h_, w_, 4)

    def part_b_head(self, kv, traj, hi=1):   # first decoder chunk with the encoder head: kv (flattened maps) -> queries -> layers 0..hi-1
        mem_q = self.enc_output(kv[:, -300:])
        ref_unact = self.enc_bbox_head(mem_q) + self.anchors_q
        return self.part_b(mem_q, ref_unact, kv, traj, 0, hi)

    def forward(self, image, traj):
        tgt, ref_unact, kv = self.part_a(image)
        return self.part_b(tgt, ref_unact, kv, traj)

    def part_b(self, tgt, ref_unact, kv, traj, lo=0, hi=None):   # decoder layers lo..hi-1; needs the previous frame's output (traj)
        hi = len(self.layers) if hi is None else hi
        kv_pos = kv + self.kpos
        qpos = None
        for i in range(lo, hi):
            l = self.layers[i]
            qpos = self.query_pos_head(torch.sigmoid(ref_unact))
            mix = traj
            q = tgt + qpos
            tgt = l.norm1(tgt + self.sa[i](q, torch.cat([tgt, mix], 1), torch.cat([tgt, mix], 1)))
            tgt = l.norm2(tgt + self.ca[i](tgt + qpos, kv_pos, kv))
            tgt = l.norm3(tgt + l.linear2(F.relu(l.linear1(tgt))))
            ref_unact = ref_unact + self.dec_bbox_head[i](tgt)
        if hi < len(self.layers):   # a middle chunk: hand the state to the next chunk
            return tgt, ref_unact
        return (self.dec_score_head[-1](tgt), torch.sigmoid(ref_unact), tgt, self.dec_id_head[-1](tgt), qpos)


if mode == 'export':
    sys.path.insert(0, CODE)
    sys.modules['MultiScaleDeformableAttention'] = types.ModuleType('MultiScaleDeformableAttention')
    from utils.utils import yaml_to_dict
    from models import build_rt_model
    torch.manual_seed(0)
    cfg = yaml_to_dict(os.path.join(CODE, 'configs', cfg_name + '.yaml'))
    cfg['DEVICE'] = 'cpu'
    cfg['pretrained'] = False
    model = FullNPU(build_rt_model(config=cfg).detr).eval()
    if os.environ.get('FTT_DEC_LAYERS'):   # fewer decoder layers (a design choice for training; speed test here)
        nl_ = int(os.environ['FTT_DEC_LAYERS'])
        model.layers, model.sa, model.ca = model.layers[:nl_], model.sa[:nl_], model.ca[:nl_]
        model.dec_bbox_head, model.dec_score_head, model.dec_id_head = model.dec_bbox_head[:nl_], model.dec_score_head[:nl_], model.dec_id_head[:nl_]
    print('levels', model.shapes, 'decoder layers', len(model.layers), 'params %.1f M' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    imgs = frames(17)
    cal_img, cal_traj, cal_use = [], [], []
    traj, use = torch.zeros(1, 300, 256), torch.zeros(1, 1, 1)
    t = time.perf_counter()
    with torch.no_grad():
        for im in imgs:
            x = torch.from_numpy(im)[None]
            cal_img.append(im); cal_traj.append(traj[0].numpy().copy()); cal_use.append(use[0].numpy().copy())
            o = model(x, traj)
            traj, use = o[2], torch.ones(1, 1, 1)
    print('torch CPU %.1f ms/frame' % ((time.perf_counter() - t) / len(imgs) * 1e3), flush=True)
    np.savez(CAL, image=np.stack(cal_img[:16]), traj=np.stack(cal_traj[:16]), use=np.stack(cal_use[:16]))
    x, tr, us = torch.from_numpy(imgs[1])[None], torch.from_numpy(cal_traj[1])[None], torch.from_numpy(cal_use[1])[None]
    with torch.no_grad():
        ref = model(x, tr)
    names = ['logits', 'boxes', 'hs', 'id_words', 'query_pos']
    torch.onnx.export(model, (x, tr), ONNX, input_names=['image', 'traj'], output_names=names, opset_version=17,
                      do_constant_folding=True)
    import onnx
    import onnxruntime as ort
    g = onnx.load(ONNX)
    ops = {}
    for n in g.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    s = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
    got = s.run(None, {'image': x.numpy(), 'traj': tr.numpy()})
    diff = [float(np.abs(a - b.numpy()).max()) for a, b in zip(got, ref)]
    print('EXPORTED', ONNX, 'onnxruntime vs torch max abs', ['%.1e' % d for d in diff], json.dumps(dict(sorted(ops.items(), key=lambda kv: -kv[1]))), flush=True)

    # split for two NPU cores in a pipeline: A (stateless, per frame) and B (decoder, carries the previous frame)
    class A(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, image): return self.m.part_a(image)

    class B(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, tgt, ref_unact, kv, traj): return self.m.part_b(tgt, ref_unact, kv, traj)

    a_m, b_m = A(model).eval(), B(model).eval()
    if model.kv_from == 2:
        class AE(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, image): return self.m.part_ae(image)
        ae = AE(model).eval()
        torch.onnx.export(ae, (x,), ONNX.replace('_npu_full_', '_npu_ae_'), input_names=['image'], output_names=['kv4', 'tgt4', 'ref4'],
                          opset_version=17, do_constant_folding=True)
        with torch.no_grad():
            k4, t4, r4 = ae(x); ta_, ra_, ka_ = a_m(x)
        print('EXPORTED AE (NHWC) vs part A max abs', ['%.1e' % float((p_.reshape(q_.shape) - q_).abs().max()) for p_, q_ in ((k4, ka_), (t4, ta_), (r4, ra_))], flush=True)
    cal_b = dict(tgt=[], ref=[], kv=[])
    with torch.no_grad():
        for im in cal_img[:16]:
            ta, ra, ka = a_m(torch.from_numpy(im)[None])
            cal_b['tgt'].append(ta[0].numpy()); cal_b['ref'].append(ra[0].numpy()); cal_b['kv'].append(ka[0].numpy())
        ta, ra, ka = a_m(x)
        refb = b_m(ta, ra, ka, tr)
    np.savez(os.path.join(out, 'calib_b.npz'), tgt=np.stack(cal_b['tgt']), ref=np.stack(cal_b['ref']), kv=np.stack(cal_b['kv']),
             traj=np.stack(cal_traj[:16]), use=np.stack(cal_use[:16]))
    torch.onnx.export(a_m, (x,), ONNX.replace('_npu_full_', '_npu_a_'), input_names=['image'], output_names=['tgt', 'ref_unact', 'kv'],
                      opset_version=17, do_constant_folding=True)
    torch.onnx.export(b_m, (ta, ra, ka, tr), ONNX.replace('_npu_full_', '_npu_b_'), input_names=['tgt', 'ref_unact', 'kv', 'traj'],
                      output_names=names, opset_version=17, do_constant_folding=True)
    sb = ort.InferenceSession(ONNX.replace('_npu_full_', '_npu_b_'), providers=['CPUExecutionProvider'])
    gb = sb.run(None, {'tgt': ta.numpy(), 'ref_unact': ra.numpy(), 'kv': ka.numpy(), 'traj': tr.numpy()})
    CH = int(os.environ.get('FTT_CHUNK', '1'))   # layers per decoder graph: 3 fail and real 2-layer chunks fail in DX-COM 2.4.1 CODEGEN; 1 works
    chunks = [(lo, min(lo + CH, len(model.layers))) for lo in range(0, len(model.layers), CH)]
    cal_c = [dict(tgt=[], ref=[]) for _ in chunks]
    with torch.no_grad():
        for j in range(len(cal_b['tgt'])):
            t_, r_ = torch.from_numpy(cal_b['tgt'][j])[None], torch.from_numpy(cal_b['ref'][j])[None]
            k_, tr_ = torch.from_numpy(cal_b['kv'][j])[None], torch.from_numpy(cal_traj[j])[None]
            for c, (lo, hi) in enumerate(chunks):
                cal_c[c]['tgt'].append(t_[0].numpy().copy()); cal_c[c]['ref'].append(r_[0].numpy().copy())
                o_ = model.part_b(t_, r_, k_, tr_, lo, hi)
                if hi < len(model.layers):
                    t_, r_ = o_
        t_, r_ = ta, ra
        for c, (lo, hi) in enumerate(chunks):
            class Cm(nn.Module):
                def __init__(self, m, lo, hi): super().__init__(); self.m, self.lo, self.hi = m, lo, hi
                def forward(self, tgt, ref_unact, kv, traj): return self.m.part_b(tgt, ref_unact, kv, traj, self.lo, self.hi)
            cm = Cm(model, lo, hi).eval()
            onames = ['tgt_out', 'ref_out'] if hi < len(model.layers) else names
            torch.onnx.export(cm, (t_, r_, ka, tr), ONNX.replace('_npu_full_', '_npu_b%d_' % c), input_names=['tgt', 'ref_unact', 'kv', 'traj'],
                              output_names=onames, opset_version=17, do_constant_folding=True)
            np.savez(os.path.join(out, 'calib_b%d.npz' % c), tgt=np.stack(cal_c[c]['tgt']), ref=np.stack(cal_c[c]['ref']),
                     kv=np.stack(cal_b['kv']), traj=np.stack(cal_traj[:len(cal_b['kv'])]))
            o_ = cm(t_, r_, ka, tr)
            if hi < len(model.layers):
                t_, r_ = o_
        chain = o_
    class AM(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, image): return self.m.part_a_maps(image)

    class BH(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, kv, traj): return self.m.part_b_head(kv, traj, 1)

    am, bh = AM(model).eval(), BH(model).eval()
    torch.onnx.export(am, (x,), ONNX.replace('_npu_full_', '_npu_am_'), input_names=['image'], output_names=['m1', 'm2'], opset_version=17, do_constant_folding=True)
    torch.onnx.export(bh, (ka, tr), ONNX.replace('_npu_full_', '_npu_bh_'), input_names=['kv', 'traj'],
                      output_names=['tgt_out', 'ref_out'] if len(model.layers) > 1 else names, opset_version=17, do_constant_folding=True)
    with torch.no_grad():
        m1, m2 = am(x)
        kv_host = torch.cat([m1.flatten(2).transpose(1, 2), m2.flatten(2).transpose(1, 2)][model.kv_from - 1:], 1)
        hb = bh(kv_host, tr)
        hb_ref = model.part_b(ta, ra, ka, tr, 0, 1)
    np.savez(os.path.join(out, 'calib_bh.npz'), kv=np.stack(cal_b['kv']), traj=np.stack(cal_traj[:len(cal_b['kv'])]))
    print('EXPORTED A-maps + head chunk; host-flattened kv vs graph kv max abs %.1e; head chunk vs chunk 0 max abs %s' % (
        float((kv_host - ka).abs().max()), ['%.1e' % float((p_ - q_).abs().max()) for p_, q_ in zip(hb, hb_ref)]), flush=True)
    print('EXPORTED decoder chunks', chunks, 'chained vs full torch max abs', ['%.1e' % float((a_ - b_).abs().max()) for a_, b_ in zip(chain, ref)], flush=True)
    print('EXPORTED A and B; B onnxruntime vs full torch max abs', ['%.1e' % float(np.abs(g_ - r_.numpy()).max()) for g_, r_ in zip(gb, ref)],
          'A out shapes', [tuple(v.shape) for v in (ta, ra, ka)], flush=True)
else:   # mode: compile (full graph) | compile_a | compile_b ; each into its own folder (the compiler writes one .dxnn per folder)
    import dx_com
    part = {'compile': 'full', 'compile_a': 'a', 'compile_b': 'b'}.get(mode) or mode.replace('compile_', '')   # compile_b0, compile_b1, ...: decoder chunks
    if part == 'full':
        cal = np.load(CAL)
        items = [(cal['image'][i], cal['traj'][i]) for i in range(len(cal['image']))]
    elif part == 'a':
        cal = np.load(CAL)
        items = [(cal['image'][i],) for i in range(len(cal['image']))]
    elif part in ('am', 'ae'):
        cal = np.load(CAL)
        items = [(cal['image'][i],) for i in range(len(cal['image']))]
    elif part == 'bh':
        cal = np.load(os.path.join(out, 'calib_bh.npz'))
        items = [(cal['kv'][i], cal['traj'][i]) for i in range(len(cal['kv']))]
    elif part.startswith('b') and part != 'b':
        cal = np.load(os.path.join(out, 'calib_%s.npz' % part))
        items = [(cal['tgt'][i], cal['ref'][i], cal['kv'][i], cal['traj'][i]) for i in range(len(cal['tgt']))]
    else:
        cal = np.load(os.path.join(out, 'calib_b.npz'))
        items = [(cal['tgt'][i], cal['ref'][i], cal['kv'][i], cal['traj'][i]) for i in range(len(cal['tgt']))]
        if os.environ.get('FTT_RANDOM_CAL') == '1':   # bisect: same shapes, gaussian values
            rg = np.random.default_rng(0)
            items = [tuple(rg.standard_normal(v.shape).astype(np.float32) for v in it) for it in items]

    import onnx as _onnx
    in_names = [i.name for i in _onnx.load(ONNX.replace('_npu_full_', '_npu_%s_' % part)).graph.input]

    class Cal(torch.utils.data.Dataset):   # several inputs: a dict by ONNX input name, in the ONNX input order (DX-COM 2.4.1)
        def __len__(self): return len(items)
        def __getitem__(self, i):
            t_ = [torch.from_numpy(np.ascontiguousarray(v)) for v in items[i]]
            return t_[0] if len(t_) == 1 else {n: v for n, v in zip(in_names, t_)}

    od = out if part == 'full' else os.path.join(out, 'part_' + part + os.environ.get('FTT_OUT_SUFFIX', ''))
    os.makedirs(od, exist_ok=True)
    t = time.monotonic()
    dx_com.compile(model=ONNX.replace('_npu_full_', '_npu_%s_' % part), output_dir=od,
                   dataloader=torch.utils.data.DataLoader(Cal(), batch_size=1, num_workers=0, shuffle=False),
                   calibration_num=len(items), calibration_method='ema', quantization_device='cpu', opt_level=1,
                   aggressive_partitioning=False, gen_log=True)
    print('COMPILED', part, 'in %.1f s' % (time.monotonic() - t), sorted(os.listdir(od)), flush=True)
