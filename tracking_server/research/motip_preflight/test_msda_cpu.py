"""Check msda_cpu.cpp against MOTIP's pure PyTorch reference (grid_sample) and time both.
Usage (device): python test_msda_cpu.py <MOTIP clone> [threads]"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.abspath(sys.argv[1]))
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
import torch
from torch.utils.cpp_extension import load

torch.set_num_threads(threads)
sys.modules["MultiScaleDeformableAttention"] = types.ModuleType("MultiScaleDeformableAttention")
from models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch

here = os.path.dirname(os.path.abspath(__file__))
ext = load(name="msda_cpu", sources=[os.path.join(here, "msda_cpu.cpp")], extra_cflags=["-O3", "-mcpu=native", "-fopenmp"], extra_ldflags=["-fopenmp"],
           build_directory=os.path.join(here, "build_msda"), verbose=False)
torch.manual_seed(0)
shapes = torch.tensor([[60, 80], [30, 40], [15, 20], [8, 10]])
starts = torch.cat([torch.zeros(1, dtype=torch.long), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
S = int((shapes[:, 0] * shapes[:, 1]).sum())
worst = 0.0
for Lq in (S, 300, 7):   # encoder (queries = all positions), decoder, a few
    value = torch.randn(1, S, 8, 32)
    loc = torch.rand(1, Lq, 8, 4, 4, 2) * 1.3 - 0.15          # some points outside the map, some on the edge
    loc[0, :3, 0, 0, 0] = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-0.01, 0.5]])
    attn = torch.rand(1, Lq, 8, 4, 4).softmax(-1)
    t = time.perf_counter(); ref = ms_deform_attn_core_pytorch(value, shapes, loc, attn); tr = time.perf_counter() - t
    t = time.perf_counter(); got = ext.forward(value, shapes, starts, loc, attn); tg = time.perf_counter() - t
    d = float((ref - got).abs().max())
    worst = max(worst, d)
    print("Lq %5d  max abs diff %.2e  (ref |max| %.2f)  reference %.1f ms  cpp %.1f ms" % (Lq, d, float(ref.abs().max()), tr * 1e3, tg * 1e3))
print("PASS" if worst < 1e-4 else "FAIL", "worst %.2e" % worst)
