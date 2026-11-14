"""Which part of the FastTrackTr NPU-variant decoder fails in DX-COM CODEGEN?  Exports small blocks (random weights, the
same module code as ftt_npu_variant.py) and compiles each with random calibration data.
Usage: python ftt_bisect.py export <out dir>   (lwir_motip venv)   |   python ftt_bisect.py compile <out dir> <block>  (dx-com env)"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

mode, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
D, NQ, NK = 256, 300, 1500


class MHA(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, q, k, v):
        B, Lq, _ = q.shape
        Lk = k.shape[1]
        q = self.q(q).reshape(B, Lq, self.h, self.d).transpose(1, 2)
        k = self.k(k).reshape(B, Lk, self.h, self.d).permute(0, 2, 3, 1)
        v = self.v(v).reshape(B, Lk, self.h, self.d).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k) * (1.0 / math.sqrt(self.d)), dim=-1)
        return self.o(torch.matmul(a, v).transpose(1, 2).reshape(B, Lq, self.h * self.d))


class MLP(nn.Module):
    def __init__(self, i, h, o, n):
        super().__init__()
        dims = [i] + [h] * (n - 1) + [o]
        self.l = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for j, l in enumerate(self.l):
            x = l(x) if j == len(self.l) - 1 else F.relu(l(x))
        return x


class SA(nn.Module):        # self-attention over [tgt, traj] with residual + norm
    def __init__(self): super().__init__(); self.a, self.n = MHA(), nn.LayerNorm(D)
    def forward(self, tgt, qpos, traj): return self.n(tgt + self.a(tgt + qpos, torch.cat([tgt, traj], 1), torch.cat([tgt, traj], 1)))


class SA_noconcat(nn.Module):   # same without the concat (keys = tgt only)
    def __init__(self): super().__init__(); self.a, self.n = MHA(), nn.LayerNorm(D)
    def forward(self, tgt, qpos): return self.n(tgt + self.a(tgt + qpos, tgt, tgt))


class CA(nn.Module):        # dense cross-attention to 1500 keys with residual + norm
    def __init__(self): super().__init__(); self.a, self.n = MHA(), nn.LayerNorm(D); self.register_buffer('kpos', torch.randn(1, NK, D) * 0.1)
    def forward(self, tgt, qpos, kv): return self.n(tgt + self.a(tgt + qpos, kv + self.kpos, kv))


class CA_nores(nn.Module):  # cross-attention without residual / norm
    def __init__(self): super().__init__(); self.a = MHA(); self.register_buffer('kpos', torch.randn(1, NK, D) * 0.1)
    def forward(self, tgt, qpos, kv): return self.a(tgt + qpos, kv + self.kpos, kv)


class FFN(nn.Module):
    def __init__(self): super().__init__(); self.l1, self.l2, self.n = nn.Linear(D, 1024), nn.Linear(1024, D), nn.LayerNorm(D)
    def forward(self, tgt): return self.n(tgt + self.l2(F.relu(self.l1(tgt))))


class HEADS(nn.Module):     # query position MLP from the box + box refinement + heads
    def __init__(self):
        super().__init__(); self.qp, self.bb, self.sc, self.idh = MLP(4, 2 * D, D, 2), MLP(D, D, 4, 3), nn.Linear(D, 1), nn.Linear(D, 301)
    def forward(self, tgt, ref):
        qpos = self.qp(torch.sigmoid(ref)); ref2 = ref + self.bb(tgt)
        return qpos, torch.sigmoid(ref2), self.sc(tgt), self.idh(tgt)


class LAYER(nn.Module):     # one whole decoder layer
    def __init__(self): super().__init__(); self.h, self.sa, self.ca, self.f = HEADS(), SA(), CA(), FFN()
    def forward(self, tgt, ref, kv, traj):
        qpos = self.h.qp(torch.sigmoid(ref))
        t = self.f(self.ca(self.sa(tgt, qpos, traj), qpos, kv))
        return t, ref + self.h.bb(t)


class LAYERS(nn.Module):    # n chained decoder layers + the outputs part B gives
    def __init__(self, n):
        super().__init__(); self.ls = nn.ModuleList(LAYER() for _ in range(n)); self.sc, self.idh = nn.Linear(D, 1), nn.Linear(D, 301)
    def forward(self, tgt, ref, kv, traj):
        for l in self.ls:
            tgt, ref = l(tgt, ref, kv, traj)
        return self.sc(tgt), torch.sigmoid(ref), tgt, self.idh(tgt)


class L2(LAYERS):
    def __init__(self): super().__init__(2)


class L3(LAYERS):
    def __init__(self): super().__init__(3)


class L1(LAYERS):
    def __init__(self): super().__init__(1)


BLOCKS = {
    'sa': (SA, dict(tgt=(1, NQ, D), qpos=(1, NQ, D), traj=(1, NQ, D))),
    'sa_noconcat': (SA_noconcat, dict(tgt=(1, NQ, D), qpos=(1, NQ, D))),
    'ca': (CA, dict(tgt=(1, NQ, D), qpos=(1, NQ, D), kv=(1, NK, D))),
    'ca_nores': (CA_nores, dict(tgt=(1, NQ, D), qpos=(1, NQ, D), kv=(1, NK, D))),
    'ffn': (FFN, dict(tgt=(1, NQ, D))),
    'heads': (HEADS, dict(tgt=(1, NQ, D), ref=(1, NQ, 4))),
    'layer': (LAYER, dict(tgt=(1, NQ, D), ref=(1, NQ, 4), kv=(1, NK, D), traj=(1, NQ, D))),
    'l1out': (L1, dict(tgt=(1, NQ, D), ref=(1, NQ, 4), kv=(1, NK, D), traj=(1, NQ, D))),
    'l2out': (L2, dict(tgt=(1, NQ, D), ref=(1, NQ, 4), kv=(1, NK, D), traj=(1, NQ, D))),
    'l3out': (L3, dict(tgt=(1, NQ, D), ref=(1, NQ, 4), kv=(1, NK, D), traj=(1, NQ, D))),
}

if mode == 'export':
    for name, (cls, ins) in BLOCKS.items():
        m = cls().eval()
        x = tuple(torch.randn(*s) for s in ins.values())
        torch.onnx.export(m, x, os.path.join(out, name + '.onnx'), input_names=list(ins), opset_version=17, do_constant_folding=True)
        np.savez(os.path.join(out, name + '_cal.npz'), **{k: np.random.randn(16, *s[1:]).astype(np.float32) for k, s in ins.items()})
    print('exported', list(BLOCKS))
else:
    import dx_com
    name = sys.argv[3]
    cal = np.load(os.path.join(out, name + '_cal.npz'))
    keys = list(BLOCKS[name][1])

    class Cal(torch.utils.data.Dataset):
        def __len__(self): return 16
        def __getitem__(self, i):
            d = {k: torch.from_numpy(cal[k][i].copy()) for k in keys}
            return d[keys[0]] if len(keys) == 1 else d

    od = os.path.join(out, 'c_' + name)
    os.makedirs(od, exist_ok=True)
    t = time.monotonic()
    try:
        dx_com.compile(model=os.path.join(out, name + '.onnx'), output_dir=od,
                       dataloader=torch.utils.data.DataLoader(Cal(), batch_size=1, num_workers=0, shuffle=False),
                       calibration_num=16, calibration_method='ema', quantization_device='cpu', opt_level=1,
                       aggressive_partitioning=False, gen_log=True)
        ok = any(f.endswith('.dxnn') for f in os.listdir(od))
    except Exception as e:
        ok = False
    print('BLOCK', name, 'OK' if ok else 'FAIL', '%.0f s' % (time.monotonic() - t), flush=True)
