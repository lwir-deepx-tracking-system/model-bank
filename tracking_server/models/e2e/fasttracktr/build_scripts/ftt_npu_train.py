"""Train FastTrackTr in the DX-M1 NPU form (user 2026-11-14: "올려논 패스트트랙도 학습시켜야하는데").

The device port (model-bank tracking_server/research/fasttracktr_npu/ftt_npu_variant.py, FullNPU) changed the RT-DETR-cross decoder so the
whole frame runs on the NPU; training must use the same form or the exported graph would not match the weights:
  1. queries = the cells of the /32 level (300 = 15x20 at 480x640) instead of TopK, encoder head on those cells
  2. cross-attention = plain multi-head attention over levels >= 2 (FTT_KV_FROM=2, 300 keys at 480x640) with a fixed sine code on the keys
  3. box refinement adds to the unactivated box (no inverse sigmoid)
  4. the previous frame's decoder output always enters the self-attention keys; zeros on the first frame (no tgt copy, no 0/1 switch)
  5. ReLU in the decoder FFN (the measured NPU graph), 2 decoder layers (config DETR_DEC_LAYERS: 2), no denoising queries
Everything else (backbone, hybrid encoder, input projections, heads, ID decoder, losses, data pipeline, training loop) is the official code,
run unchanged through its main.py. Start: official ImageNet ResNet18-vd backbone + matching tensors of the RT-DETR r18vd COCO checkpoint.

Usage (Vast): FTT_CODE=<clone of liaopan-lp/fasttracktr> python ftt_npu_train.py <config.yaml> <data root> [rtdetr_coco.pth] [extra main.py args]
"""
import math, os, runpy, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = os.environ['FTT_CODE']; sys.path.insert(0, CODE); os.chdir(CODE)
import types   # noqa: E402
try: import MultiScaleDeformableAttention   # noqa: F401  (CUDA op of the deformable-DETR variants; the NPU form does not use it)
except ImportError: sys.modules['MultiScaleDeformableAttention'] = types.ModuleType('MultiScaleDeformableAttention')
try: import cython_bbox   # noqa: F401  (only the JDE tracker of submit_engine imports it; the build fails on this machine)
except ImportError:
    import numpy as _np
    def _bbox_overlaps(a, b):
        a = _np.asarray(a, float); b = _np.asarray(b, float)
        if not len(a) or not len(b): return _np.zeros((len(a), len(b)))
        iw = _np.clip(_np.minimum(a[:, None, 2], b[None, :, 2]) - _np.maximum(a[:, None, 0], b[None, :, 0]) + 1, 0, None)
        ih = _np.clip(_np.minimum(a[:, None, 3], b[None, :, 3]) - _np.maximum(a[:, None, 1], b[None, :, 1]) + 1, 0, None)
        aa = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1); bb = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
        return iw * ih / (aa[:, None] + bb[None, :] - iw * ih)
    _m = types.ModuleType('cython_bbox'); _m.bbox_overlaps = _bbox_overlaps; sys.modules['cython_bbox'] = _m
from models.rt_detr_cross import rtdetr_decoder_cross as RDC   # noqa: E402

KV_FROM = int(os.environ.get('FTT_KV_FROM', '2')); QH, QW = 15, 20


def sine_pos(h, w, dim, device, temperature=10000.):
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device), torch.arange(w, dtype=torch.float32, device=device), indexing='ij')
    pd = dim // 4; om = 1. / (temperature ** (torch.arange(pd, dtype=torch.float32, device=device) / pd))
    ox, oy = gx.flatten()[:, None] * om[None], gy.flatten()[:, None] * om[None]
    return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], 1)[None]


_orig_init = RDC.RTDETRTransformerCross.__init__


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    d = self.hidden_dim if hasattr(self, 'hidden_dim') else 256
    n = len(self.decoder.layers)
    self.npu_ca = nn.ModuleList(nn.MultiheadAttention(d, 8, batch_first=True) for _ in range(n))   # new dense cross-attention (exported as MHA)
    self._npu_cache = {}


def _npu_forward(self, feats, targets=None, max_gt_num=0, trajectory_inputs=None):
    memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)            # [B, sum hw, C], [(h, w)] per level
    shapes = [(int(h), int(w)) for h, w in (spatial_shapes.tolist() if torch.is_tensor(spatial_shapes) else spatial_shapes)]
    starts = [0];
    for h, w in shapes: starts.append(starts[-1] + h * w)
    B, _, C = memory.shape; dev = memory.device
    key = (tuple(shapes), str(dev))
    if key not in self._npu_cache:
        # queries: always the 15x20 grid of the /32 level (= exactly its cells at 480x640, the device size); anchors of level index 2
        g3 = [[1, 1], [1, 1], [QH, QW]]
        anchors, _ = self._generate_anchors(g3, device=dev) if 'device' in self._generate_anchors.__code__.co_varnames else self._generate_anchors(g3)
        kpos = torch.cat([sine_pos(h, w, C, dev) for h, w in shapes[KV_FROM:]], 1)
        self._npu_cache[key] = (anchors.to(dev)[:, -QH * QW:], kpos)
    anchors_q, kpos = self._npu_cache[key]
    h2, w2 = shapes[2]; lv2 = memory[:, starts[2]:starts[3]]
    if (h2, w2) != (QH, QW):   # other training picture sizes (crop / resize augmentation): resample the /32 map to 15x20 so the query count stays 300
        lv2 = F.interpolate(lv2.transpose(1, 2).reshape(B, C, h2, w2), size=(QH, QW), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
    mem_q = self.enc_output(lv2)                                                         # queries = /32 cells
    enc_logits = self.enc_score_head(mem_q)
    ref_unact = self.enc_bbox_head(mem_q) + anchors_q
    enc_boxes = torch.sigmoid(ref_unact)
    kv = memory[:, starts[KV_FROM]:]; kv_pos = kv + kpos
    tgt = mem_q
    traj = trajectory_inputs['src'] if (trajectory_inputs is not None and trajectory_inputs.get('src') is not None and trajectory_inputs['src'].shape == tgt.shape) else torch.zeros_like(tgt)
    logits, boxes, hs, ids = [], [], [], []
    qpos = None
    for i, l in enumerate(self.decoder.layers):
        qpos = self.query_pos_head(torch.sigmoid(ref_unact))
        kvs = torch.cat([tgt, traj], 1)
        tgt = l.norm1(tgt + l.self_attn(tgt + qpos, kvs, kvs, need_weights=False)[0])
        tgt = l.norm2(tgt + self.npu_ca[i](tgt + qpos, kv_pos, kv, need_weights=False)[0])
        tgt = l.norm3(tgt + l.linear2(F.relu(l.linear1(tgt))))
        ref_unact = ref_unact + self.dec_bbox_head[i](tgt)
        logits.append(self.dec_score_head[i](tgt)); boxes.append(torch.sigmoid(ref_unact)); hs.append(tgt); ids.append(self.dec_id_head[i](tgt))
    out = {'pred_logits': logits[-1], 'pred_boxes': boxes[-1], 'outputs': hs[-1], 'id_words': ids[-1], 'query_pos_embed_history_encoder': qpos}
    if self.training and self.aux_loss:
        out['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, 'id_words': c} for a, b, c in zip(logits[:-1], boxes[:-1], ids[:-1])]
        out['enc_outputs'] = {'pred_logits': enc_logits, 'pred_boxes': enc_boxes}
    return out


RDC.RTDETRTransformerCross.__init__ = _init
RDC.RTDETRTransformerCross.forward = _npu_forward

# start weights: matching tensors of the RT-DETR r18vd COCO checkpoint (backbone / encoder / decoder input projections, heads where shapes fit)
COCO = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].endswith('.pth') else ''
import models as M   # noqa: E402
_orig_build = M.build_rt_model


def _build(config):
    model = _orig_build(config)
    if COCO:
        sd = torch.load(COCO, map_location='cpu'); sd = sd.get('ema', {}).get('module', sd.get('model', sd))
        own = model.detr.state_dict(); took = {}
        for k, v in sd.items():
            for kk in (k, k.replace('decoder.decoder.', 'decoder.decoder.')):
                if kk in own and own[kk].shape == v.shape and not kk.startswith('decoder.dec_score_head') and not kk.startswith('decoder.enc_score_head') and 'denoising_class_embed' not in kk:
                    took[kk] = v; break
        model.detr.load_state_dict(took, strict=False)
        print('NPU-TRAIN start weights from %s: %d of %d detr tensors' % (os.path.basename(COCO), len(took), len(own)), flush=True)
    n = sum(p.numel() for p in model.parameters()); print('NPU-TRAIN parameters %.1f M, decoder layers %d, kv levels >= %d' % (n / 1e6, len(model.detr.decoder.decoder.layers), KV_FROM), flush=True)
    return model


M.build_rt_model = _build

# 11-14 (user: 빠르게 학습): bf16 autocast around the model forward (FTT_AMP=1). The official loop has no AMP; bf16 needs no loss scaler.
# Outputs go back to float32 so the official criterion and matcher run as before.
if os.environ.get('FTT_AMP') == '1':
    from models import fasttracktr as FT

    def _to32(o):
        if torch.is_tensor(o): return o.float() if o.is_floating_point() else o
        if isinstance(o, dict): return {k: _to32(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return type(o)(_to32(v) for v in o)
        return o
    _orig_fwd = FT.FastTrackTr.forward

    def _amp_fwd(self, *a, **k):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = _orig_fwd(self, *a, **k)
        return _to32(out)
    FT.FastTrackTr.forward = _amp_fwd
    print('NPU-TRAIN bf16 autocast on', flush=True)
# 11-14 fixes after the ID check (ftt_id_check.py: id_words matched people worse than the nearest box centre, and worse after the 2nd epoch;
# a code audit found the ID loss falls to ~0 for collapsed embeddings). Both are off unless asked for, so the run started earlier is unchanged.
# FTT_ID_FIX=1: the circle loss on L2-normalised id_words (the tracker compares normalised features), standard margin m=0.25, gamma=64,
#   no division by the feature size and detection count, times FTT_ID_W (default 0.05: 8 terms per step, ~30 each at the start).
#   The official one used raw dot products, m=0.8 and /301 /N, so shrinking every embedding towards 0 already gave ~0 loss.
if os.environ.get('FTT_ID_FIX') == '1':
    import einops   # noqa: E402
    from models import criterion as CR   # noqa: E402
    _ID_W = float(os.environ.get('FTT_ID_W', '0.05')); _circle = CR.CircleLoss(m=0.25, gamma=64)

    def _id_forward(self, outputs, targets, emb_reid):
        f = F.normalize(einops.rearrange(outputs, 'b n c -> (b n) c').float(), dim=1); t = targets.squeeze(0)
        sp, sn, _ = CR.convert_label_to_similarity(f, t)
        if not len(sp) or not len(sn): return f.sum() * 0.0   # no same-person pair or no other person in this clip
        return _ID_W * self.IDLoss_fixed(sp, sn, 1)
    CR.IDCriterion.forward = _id_forward; CR.IDCriterion.IDLoss_fixed = _circle
    print('NPU-TRAIN ID loss: normalised circle loss m=0.25 gamma=64 x %.3g' % _ID_W, flush=True)

# FTT_ASPECT=1: the official multi-scale squashes every frame to a square (480..800), but the device runs 640x480 as it is; scale the
# frame keeping 3:4 instead (most often 480x640). Boxes are normalised (cx, cy, w, h), so the targets stay as they are.
if os.environ.get('FTT_ASPECT') == '1':
    import numpy as _np2   # noqa: E402
    from models.rt_detr_cross import rtdetr_cross as RC   # noqa: E402
    _SIZES = [(384, 512), (416, 544), (448, 608), (480, 640), (480, 640), (480, 640), (512, 672), (544, 736), (576, 768)]
    _rc_fwd = RC.RTDETRCROSS.forward

    def _aspect_fwd(self, x, *a, **k):
        if not (self.training and self.multi_scale): return _rc_fwd(self, x, *a, **k)
        h, w = _SIZES[_np2.random.randint(len(_SIZES))]; x.tensors = F.interpolate(x.tensors, size=[h, w]); ms = self.multi_scale; self.multi_scale = None
        try: return _rc_fwd(self, x, *a, **k)
        finally: self.multi_scale = ms
    RC.RTDETRCROSS.forward = _aspect_fwd
    print('NPU-TRAIN multi-scale keeps 3:4 (%s)' % ', '.join('%dx%d' % (w, h) for h, w in _SIZES), flush=True)
import train_engine as TE   # noqa: E402
if hasattr(TE, 'build_rt_model'): TE.build_rt_model = _build

if __name__ == '__main__':   # imported by ftt_trained_export.py for the same patches without training
    cfg, root = sys.argv[1], sys.argv[2]; extra = [a for a in sys.argv[3:] if not a.endswith('.pth')]
    sys.argv = ['main.py', '--mode', 'train', '--config-path', cfg, '--data-root', root, '--device', 'cuda'] + extra
    runpy.run_path(os.path.join(CODE, 'main.py'), run_name='__main__')
