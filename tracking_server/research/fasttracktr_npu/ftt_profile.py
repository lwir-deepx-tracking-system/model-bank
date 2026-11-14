"""Time FastTrackTr (random weights, no checkpoint) per frame on the CPU, part by part.
Usage: python ftt_profile.py <code dir> <config yaml> [threads] [frames] [H] [W]
Only speed: the weights are random, so the boxes mean nothing. The previous frame's output is fed back as in submit_engine."""
import os
import sys
import time

code, cfg_path = os.path.abspath(sys.argv[1]), sys.argv[2]
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
frames = int(sys.argv[4]) if len(sys.argv) > 4 else 6
H = int(sys.argv[5]) if len(sys.argv) > 5 else 480
W = int(sys.argv[6]) if len(sys.argv) > 6 else 640
os.environ["OMP_NUM_THREADS"] = str(threads)
sys.path.insert(0, code)
import torch

torch.set_num_threads(threads)
torch.manual_seed(0)
import types
sys.modules["MultiScaleDeformableAttention"] = types.ModuleType("MultiScaleDeformableAttention")   # CUDA extension: only imported (Deformable-DETR path), not used by RT-DETR
from utils.utils import yaml_to_dict
from models import build_rt_model
from utils.nested_tensor import tensor_list_to_nested_tensor

cfg = yaml_to_dict(cfg_path)
if cfg.get("SUPER_CONFIG_PATH"):
    from configs.utils import load_super_config
    cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
cfg["DEVICE"] = "cpu"
cfg["pretrained"] = False
t = time.perf_counter()
model = build_rt_model(config=cfg).eval()
print("built in %.1f s; params %.1f M" % (time.perf_counter() - t, sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
print("detr children:", [n for n, _ in model.detr.named_children()], flush=True)

PROF, T0 = {}, {}
parts = list(model.detr.named_children()) + [("decoder." + n, m) for n, m in model.detr.decoder.named_children()] + [("dec.layer0." + n, m) for n, m in model.detr.decoder.decoder.layers[0].named_children()]
for name, mod in parts:
    def pre(m, a, _k=name):
        T0[_k] = time.perf_counter()

    def post(m, a, o, _k=name):
        PROF[_k] = PROF.get(_k, 0.0) + time.perf_counter() - T0.pop(_k)
    mod.register_forward_pre_hook(pre)
    mod.register_forward_hook(post)

traj = None
tot = []
with torch.no_grad():
    for i in range(frames):
        img = torch.rand(3, H, W)
        fr = tensor_list_to_nested_tensor([img], is_training=False)
        if i == 1:
            PROF.clear()
        t = time.perf_counter()
        out = model(frames=fr, trajectory_output=traj)
        dt = time.perf_counter() - t
        if i >= 1:
            tot.append(dt)
        v = out["outputs"]
        mask = torch.zeros((1, v.size(1)), dtype=v.dtype)
        traj = {"src": v, "mask": mask}
        print("frame %d  %.1f ms  outputs %s  keys %s" % (i, dt * 1e3, tuple(v.shape), sorted(out.keys())[:12]), flush=True)
n = len(tot)
print("MEAN %.1f ms/frame over %d frames (first frame excluded), threads %d, input %dx%d" % (sum(tot) / n * 1e3, n, threads, H, W))
for k, v in sorted(PROF.items(), key=lambda kv: -kv[1]):
    print("PART %-20s %8.1f ms/frame" % (k, v / n * 1e3))
