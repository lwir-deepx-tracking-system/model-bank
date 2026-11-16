"""Trained FastTrackTr (ftt_npu_train.py checkpoint) -> the device graphs of the NPU port (model-bank research/fasttracktr_npu): AE (backbone +
encoder + encoder head on the 15x20 level, NHWC) and one decoder graph per layer (b0, b1), plus calibration files, for DX-COM.
The module math is the same as in training (ftt_npu_train._npu_forward): queries = the 300 /32 cells, self-attention keys [tgt, traj],
dense cross-attention over levels >= 2 with a sine code (trained npu_ca weights), box refine on the unactivated box, ReLU FFN.
Usage (Vast): FTT_CODE=<official clone> python ftt_trained_export.py <config.yaml> <checkpoint.pth> <calibration frames dir (8-bit png)> <out dir>"""
import glob, math, os, sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

cfg_path, ckpt, frames_dir, out = sys.argv[1], sys.argv[2], sys.argv[3], os.path.abspath(sys.argv[4]); os.makedirs(out, exist_ok=True)
sys.argv = [sys.argv[0]]                       # ftt_npu_train reads argv for the COCO start weights: none here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftt_npu_train as P                      # noqa: E402  (patches the official decoder class)
from utils.utils import yaml_to_dict           # noqa: E402
import models as M                             # noqa: E402
H, W = 480, 640


def sine_pos(h, w, dim=256, temperature=10000.):
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
    pd = dim // 4; om = 1. / (temperature ** (torch.arange(pd, dtype=torch.float32) / pd))
    ox, oy = gx.flatten()[:, None] * om[None], gy.flatten()[:, None] * om[None]
    return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], 1)[None]


class MHA(nn.Module):
    """nn.MultiheadAttention written as MatMul / Softmax / Reshape (export form, as the NPU port)."""
    def __init__(self, mha, dim=256, heads=8):
        super().__init__(); self.h, self.d = heads, dim // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        wq, wk, wv = mha.in_proj_weight.chunk(3); bq, bk, bv = mha.in_proj_bias.chunk(3)
        for lin, w_, b_ in ((self.q, wq, bq), (self.k, wk, bk), (self.v, wv, bv)): lin.weight.data.copy_(w_); lin.bias.data.copy_(b_)
        self.o.weight.data.copy_(mha.out_proj.weight); self.o.bias.data.copy_(mha.out_proj.bias)

    def forward(self, q, k, v):
        B, Lq, _ = q.shape; Lk = k.shape[1]
        q = self.q(q).reshape(B, Lq, self.h, self.d).transpose(1, 2); k = self.k(k).reshape(B, Lk, self.h, self.d).permute(0, 2, 3, 1)
        v = self.v(v).reshape(B, Lk, self.h, self.d).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k) * (1.0 / math.sqrt(self.d)), dim=-1)
        return self.o(torch.matmul(a, v).transpose(1, 2).reshape(B, Lq, self.h * self.d))


class NPUNet(nn.Module):
    def __init__(self, detr):
        super().__init__(); dec = detr.decoder
        self.backbone, self.encoder = detr.backbone, detr.encoder
        self.input_proj, self.enc_output, self.enc_bbox_head, self.enc_score_head = dec.input_proj, dec.enc_output, dec.enc_bbox_head, dec.enc_score_head
        self.query_pos_head, self.num_levels = dec.query_pos_head, dec.num_levels
        self.dec_bbox_head, self.dec_score_head, self.dec_id_head = dec.dec_bbox_head, dec.dec_score_head, dec.dec_id_head
        self.layers = dec.decoder.layers
        self.sa = nn.ModuleList(MHA(l.self_attn) for l in self.layers); self.ca = nn.ModuleList(MHA(m) for m in dec.npu_ca)
        with torch.no_grad(): shapes = [tuple(int(v) for v in p.shape[-2:]) for p in self.proj(self.encoder(self.backbone(torch.zeros(1, 3, H, W))))]
        assert shapes[2] == (15, 20), shapes
        self.register_buffer('kpos', torch.cat([sine_pos(h, w) for h, w in shapes[P.KV_FROM:]], 1))
        anchors, _ = dec._generate_anchors([[1, 1], [1, 1], [15, 20]]); self.register_buffer('anchors_q', anchors[:, -300:].float())

    def proj(self, feats):
        p = [self.input_proj[i](f) for i, f in enumerate(feats)]
        for i in range(len(p), self.num_levels): p.append(self.input_proj[i](feats[-1] if i == len(feats) else p[-1]))
        return p

    def part_ae(self, image):   # NPU only, NHWC: kv (15x20 level), tgt, ref_unact
        feats = self.proj(self.encoder(self.backbone(image * (1.0 / 255.0))))
        x = feats[2].permute(0, 2, 3, 1); mem = self.enc_output(x)
        return x, mem, self.enc_bbox_head(mem) + self.anchors_q.reshape(1, 15, 20, 4)

    def part_b(self, tgt, ref_unact, kv, traj, lo, hi):
        kv_pos = kv + self.kpos; qpos = None
        for i in range(lo, hi):
            l = self.layers[i]; qpos = self.query_pos_head(torch.sigmoid(ref_unact)); kvs = torch.cat([tgt, traj], 1)
            tgt = l.norm1(tgt + self.sa[i](tgt + qpos, kvs, kvs)); tgt = l.norm2(tgt + self.ca[i](tgt + qpos, kv_pos, kv))
            tgt = l.norm3(tgt + l.linear2(F.relu(l.linear1(tgt)))); ref_unact = ref_unact + self.dec_bbox_head[i](tgt)
        if hi < len(self.layers): return tgt, ref_unact
        return self.dec_score_head[hi - 1](tgt), torch.sigmoid(ref_unact), tgt, self.dec_id_head[hi - 1](tgt), qpos


cfg = yaml_to_dict(cfg_path); cfg['DEVICE'] = 'cpu'; cfg['pretrained'] = False
model = M.build_rt_model(cfg); sd = torch.load(ckpt, map_location='cpu'); sd = sd.get('model', sd)
missing, unexpected = model.load_state_dict(sd, strict=False); print('checkpoint loaded, missing %d unexpected %d' % (len(missing), len(unexpected)), flush=True)
model.eval(); net = NPUNet(model.detr).eval()

# reference: the training forward (patched official decoder) on one frame vs this export form
files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))[:17]
from PIL import Image   # noqa: E402
imgs = [np.repeat(np.asarray(Image.open(f).convert('L'), np.float32)[None], 3, 0) for f in files]
x0 = torch.from_numpy(imgs[0])[None]
with torch.no_grad():
    kv4, tgt4, ref4 = net.part_ae(x0); kv = kv4.reshape(1, 300, -1); tgt = tgt4.reshape(1, 300, -1); ref = ref4.reshape(1, 300, 4); traj = torch.zeros_like(tgt)
    t_, r_ = net.part_b(tgt, ref, kv, traj, 0, 1); outs = net.part_b(t_, r_, kv, traj, 1, 2)
    feats = model.detr.encoder(model.detr.backbone(x0 / 255.0)); ref_out = model.detr.decoder(feats)
print('export form vs training forward: logits %.2e boxes %.2e' % (float((outs[0] - ref_out['pred_logits']).abs().max()), float((outs[1] - ref_out['pred_boxes']).abs().max())), flush=True)
print('frame 0: queries with score >= 0.5:', int((torch.sigmoid(outs[0]) >= 0.5).sum()), flush=True)

class AE(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, image): return s.m.part_ae(image)

class Cm(nn.Module):
    def __init__(s, m, lo, hi): super().__init__(); s.m, s.lo, s.hi = m, lo, hi
    def forward(s, tgt, ref_unact, kv, traj): return s.m.part_b(tgt, ref_unact, kv, traj, s.lo, s.hi)

stem = os.path.join(out, 'ftt_npu_trained')
torch.onnx.export(AE(net), (x0,), stem + '_ae.onnx', input_names=['image'], output_names=['kv4', 'tgt4', 'ref4'], opset_version=17, do_constant_folding=True)
cal = dict(image=[], b0=[], b1=[]); tr_prev = torch.zeros(1, 300, 256)
with torch.no_grad():
    for im in imgs[:16]:
        xi = torch.from_numpy(im)[None]; k4, t4, r4 = net.part_ae(xi); k, t, r = k4.reshape(1, 300, -1), t4.reshape(1, 300, -1), r4.reshape(1, 300, 4)
        cal['image'].append(im); cal['b0'].append((t[0].numpy(), r[0].numpy(), k[0].numpy(), tr_prev[0].numpy()))
        t1, r1 = net.part_b(t, r, k, tr_prev, 0, 1); cal['b1'].append((t1[0].numpy(), r1[0].numpy(), k[0].numpy(), tr_prev[0].numpy()))
        tr_prev = net.part_b(t1, r1, k, tr_prev, 1, 2)[2]
torch.onnx.export(Cm(net, 0, 1), (tgt, ref, kv, traj), stem + '_b0.onnx', input_names=['tgt', 'ref_unact', 'kv', 'traj'], output_names=['tgt_out', 'ref_out'], opset_version=17, do_constant_folding=True)
torch.onnx.export(Cm(net, 1, 2), (t_, r_, kv, traj), stem + '_b1.onnx', input_names=['tgt', 'ref_unact', 'kv', 'traj'], output_names=['logits', 'boxes', 'hs', 'id_words', 'query_pos'], opset_version=17, do_constant_folding=True)
np.savez(os.path.join(out, 'calib_ae.npz'), image=np.stack(cal['image']))
for c in ('b0', 'b1'):
    a = list(zip(*cal[c])); np.savez(os.path.join(out, 'calib_%s.npz' % c), tgt=np.stack(a[0]), ref=np.stack(a[1]), kv=np.stack(a[2]), traj=np.stack(a[3]))
json.dump(dict(ckpt=ckpt, config=cfg_path, frames=len(imgs), kv_from=P.KV_FROM), open(os.path.join(out, 'EXPORT.json'), 'w'))
print('EXPORT DONE', sorted(os.listdir(out)), flush=True)
