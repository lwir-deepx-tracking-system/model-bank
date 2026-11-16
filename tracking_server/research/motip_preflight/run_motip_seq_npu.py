#!/usr/bin/env python3
"""[NPU variant, made by make_npu_runner.py] Run the ORIGINAL MOTIP model (official code, DanceTrack config + checkpoint) on a folder of 8-bit
grayscale PNG frames on the CPU, and write MOT rows plus per-frame timing.

NPU VARIANT (--npu-backbone): the ResNet-50 body of the detector (outputs of layer2, layer3, layer4) runs on the DX-M1
NPU from backbone_480x640.dxnn (INT8, input = raw 0..255 gray replicated to 3 channels as uint8 NHWC, the /255 and ImageNet
mean/std are inside the compiled graph). Its three feature maps replace the PyTorch body output; the rest of the
official model runs unchanged on the CPU. Only the native 640x480 input (--shorter 480) is possible.

Nothing in the MOTIP clone is modified.  Deviations from the official submit_and_evaluate.py, all stated
in the output JSON as well:
  1. The CUDA extension MultiScaleDeformableAttention is not built.  MSDeformAttnFunction.apply is routed
     to the official pure PyTorch reference ms_deform_attn_core_pytorch (models/ops/functions/
     ms_deform_attn_func.py), which the official model code itself never calls.
  2. backbone.is_main_process is patched to False so the ImageNet ResNet-50 download is skipped; the
     checkpoint holds every backbone tensor and is loaded strictly.
  3. The checkpoint is read with torch.load(weights_only=True) instead of weights_only=False.
  4. Frames come from a plain folder loop instead of the DanceTrack dataset class / DataLoader / accelerate.
The per-frame tracking logic is the official models.runtime_tracker.RuntimeTracker, unchanged, built with
every inference value taken from the official yaml (its own constructor defaults are different).

RESIZE.  The official runtime (data/seq_dataset.py + submit_and_evaluate.py) resizes the PIL image so the
shorter side is 800 with the longer side capped at INFERENCE_MAX_LONGER (1440): a 640x480 frame becomes
1066x800.  This script applies exactly that when --shorter 800 --longer 1440 (the default).  With
--shorter 480 the frame goes in at its native 640x480 (the resize is a no-op); that is NOT the official
setting and is recorded as such.

INPUT.  Each PNG must be 8-bit single channel (PIL mode "L").  It is replicated to 3 channels with
Image.convert("RGB"), then the official transform: Resize, /255, ImageNet mean/std.  Batch of one, FP32.

OUTPUT (all next to --out-prefix):
  <prefix>.mot.txt       frame,id,x,y,w,h,score,-1,-1,-1   frame = PNG number (000123.png -> 123, the
                         numbering tracks_edited_mot.txt uses), x,y = top-left in source pixels, id =
                         official output id (starts at 0), score = detector score (the official file
                         writer hard-codes 1 here).  Boxes are not clipped, like the official output.
  <prefix>.frames.jsonl  one JSON line per processed frame: timing split and counts
  <prefix>.state.pt      tracker state after the last saved frame (for --resume)
  <prefix>.json          run description: settings, load result, frames processed, timing summary, status

RANGE / RESUME.  --start/--end are PNG numbers (inclusive).  --resume continues after the frame stored in
<prefix>.state.pt with the tracker state restored (history, id queue, id map, next id), so a resumed run
is the same computation as an uninterrupted one.  Starting at --start > first PNG without a state file
starts a fresh tracker at that frame, which is recorded in the JSON.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import types

ap = argparse.ArgumentParser()
ap.add_argument("--motip", required=True, help="official MOTIP clone")
ap.add_argument("--ckpt", required=True, help="r50_deformable_detr_motip_dancetrack.pth")
ap.add_argument("--frames", required=True, help="folder with <number>.png")
ap.add_argument("--out-prefix", required=True)
ap.add_argument("--start", type=int, default=0, help="first PNG number (0 = first in folder)")
ap.add_argument("--end", type=int, default=0, help="last PNG number, inclusive (0 = last in folder)")
ap.add_argument("--shorter", type=int, default=800, help="official: 800")
ap.add_argument("--longer", type=int, default=1440, help="official DanceTrack yaml: 1440")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--device", default="cpu", help="cpu, or cuda in an environment with a CUDA build of torch")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--save-every", type=int, default=25, help="frames between tracker state saves")
ap.add_argument("--expect-sha256", default="", help="refuse to run if the checkpoint hash differs")
ap.add_argument("--npu-backbone", default="", help="backbone_480x640.dxnn: ResNet-50 body on the DX-M1 NPU")
ap.add_argument("--npu-core", type=int, default=1, help="NPU core (the tracking server uses core 0 for detection)")
ap.add_argument("--profile", action="store_true", help="time the parts of the model (forward hooks) and print a table")
ap.add_argument("--qlinear", choices=["none", "enc", "transformer"], default="none",
                help="dynamic int8 Linear layers (oneDNN) in the encoder or the whole transformer (changes the numbers slightly)")
ap.add_argument("--msda", choices=["reference", "cpp"], default="reference",
                help="deformable attention sampling: official PyTorch reference, or msda_cpu.cpp (same arithmetic, faster)")
args = ap.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["MKL_NUM_THREADS"] = str(args.threads)
import torch  # noqa: E402

torch.set_num_threads(args.threads)

MOTIP = os.path.abspath(args.motip)
sys.path.insert(0, MOTIP)

# (1) stand-in for the CUDA extension, then the official pure PyTorch attention
sys.modules["MultiScaleDeformableAttention"] = types.ModuleType("MultiScaleDeformableAttention")
import models.ops.functions.ms_deform_attn_func as msda_func  # noqa: E402
import models.ops.modules.ms_deform_attn as msda_mod  # noqa: E402

MSDA_CALLS = [0]
PROF = {}
MSDA_EXT, MSDA_CHECK = [None], []
if args.msda == "cpp":
    from torch.utils.cpp_extension import load as _load
    _here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(_here, "build_msda"), exist_ok=True)
    MSDA_EXT[0] = _load(name="msda_cpu", sources=[os.path.join(_here, "msda_cpu.cpp")], extra_cflags=["-O3", "-mcpu=native", "-fopenmp"],
                        extra_ldflags=["-fopenmp"], build_directory=os.path.join(_here, "build_msda"), verbose=False)


class PureTorchMSDA:
    @staticmethod
    def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        MSDA_CALLS[0] += 1
        _t = time.perf_counter()
        if MSDA_EXT[0] is None:
            _o = msda_func.ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
        else:
            _o = MSDA_EXT[0].forward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
            if not MSDA_CHECK:
                _r = msda_func.ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
                MSDA_CHECK.append(float((_r - _o).abs().max()))
                log("msda cpp vs reference, first call: max abs diff %.2e" % MSDA_CHECK[0])
                if MSDA_CHECK[0] > 1e-3:
                    raise SystemExit("msda cpp differs from the reference")
        PROF["msda_core(" + ("enc" if value.shape[1] > 1000 and sampling_locations.shape[1] == value.shape[1] else "dec") + ")"] = PROF.get("msda_core(" + ("enc" if value.shape[1] > 1000 and sampling_locations.shape[1] == value.shape[1] else "dec") + ")", 0.0) + time.perf_counter() - _t
        return _o


msda_mod.MSDeformAttnFunction = PureTorchMSDA

# (2) no ImageNet download; the checkpoint overwrites the backbone
import models.deformable_detr.backbone as backbone_mod  # noqa: E402

backbone_mod.is_main_process = lambda: False

import numpy  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from configs.util import load_super_config  # noqa: E402
from models.motip import build as build_motip  # noqa: E402
from models.runtime_tracker import RuntimeTracker  # noqa: E402
from structures.ordered_set import OrderedSet  # noqa: E402
from utils.misc import yaml_to_dict, distributed_device  # noqa: E402
from utils.nested_tensor import nested_tensor_from_tensor_list  # noqa: E402

P = args.out_prefix
MOT, JSONL, STATE, SIDE = P + ".mot.txt", P + ".frames.jsonl", P + ".state.pt", P + ".json"
os.makedirs(os.path.dirname(os.path.abspath(P)), exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo):
    try:
        head = open(os.path.join(repo, ".git", "HEAD")).read().strip()
        if head.startswith("ref:"):
            return open(os.path.join(repo, ".git", head.split(None, 1)[1])).read().strip()
        return head
    except OSError:
        return None


# ---- frames -------------------------------------------------------------------------------------
all_png = sorted(int(os.path.splitext(n)[0]) for n in os.listdir(args.frames)
                 if n.lower().endswith(".png") and os.path.splitext(n)[0].isdigit())
names = {int(os.path.splitext(n)[0]): n for n in os.listdir(args.frames)
         if n.lower().endswith(".png") and os.path.splitext(n)[0].isdigit()}
if not all_png:
    raise SystemExit("no <number>.png in %s" % args.frames)
start = args.start or all_png[0]
end = args.end or all_png[-1]

# ---- config, model, checkpoint ------------------------------------------------------------------
cfg_path = os.path.join(MOTIP, "configs", "r50_deformable_detr_motip_dancetrack.yaml")
cfg = yaml_to_dict(cfg_path)
cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
cfg["DEVICE"] = args.device

ckpt_sha = sha256_file(args.ckpt)
if args.expect_sha256 and ckpt_sha != args.expect_sha256:
    raise SystemExit("checkpoint sha256 %s != expected %s" % (ckpt_sha, args.expect_sha256))

t0 = time.perf_counter()
model, _ = build_motip(config=cfg)
state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
model_state = state["model"]
# First a non-strict load only to LIST missing / unexpected keys, then the strict load the official code does.
probe = model.load_state_dict(model_state, strict=False)
missing, unexpected = list(probe.missing_keys), list(probe.unexpected_keys)
strict_error = None
try:
    strict_result = str(model.load_state_dict(model_state, strict=True))
except RuntimeError as e:  # reported, never hidden
    strict_result, strict_error = None, str(e)[:2000]
model.eval()
model.to(args.device)
build_load_s = time.perf_counter() - t0
load_info = dict(strict=True, strict_result=strict_result, strict_error=strict_error,
                 missing_keys=missing, unexpected_keys=unexpected,
                 checkpoint_model_tensors=len(model_state), model_state_dict_tensors=len(model.state_dict()),
                 checkpoint_top_keys=list(state.keys()), weights_only=True, build_and_load_s=round(build_load_s, 2))
log("load_state_dict strict:", strict_result, "missing", len(missing), "unexpected", len(unexpected))
if strict_error or missing or unexpected:
    with open(SIDE, "w") as fh:
        json.dump(dict(status="failed_load", load=load_info), fh, indent=1)
    raise SystemExit("strict load failed: %s" % (strict_error or (missing, unexpected)))
del state

if args.profile:
    import re as _re
    _T0 = {}

    def _cat(name):
        n = _re.sub(r"\.\d+", ".*", name)
        for pat, lab in ((r"backbone$", "backbone (NPU body + position)"), (r"input_proj\.\*$", "input_proj"),
                         (r"encoder\.layers\.\*\.self_attn$", "enc self_attn (MSDeformAttn)"),
                         (r"encoder\.layers\.\*\.self_attn\.(value_proj|sampling_offsets|attention_weights|output_proj)$", "enc attn linears"),
                         (r"encoder\.layers\.\*\.(linear1|linear2)$", "enc FFN linears"),
                         (r"encoder\.layers\.\*\.(norm1|norm2)$", "enc norms"),
                         (r"encoder$", "encoder total"), (r"decoder$", "decoder total"),
                         (r"decoder\.layers\.\*\.cross_attn$", "dec cross_attn"), (r"decoder\.layers\.\*\.self_attn$", "dec self_attn"),
                         (r"decoder\.layers\.\*\.(linear1|linear2)$", "dec FFN linears"),
                         (r"transformer$", "transformer total"), (r"^detr$", "detr total"),
                         (r"id_decoder$", "id_decoder"), (r"trajectory_modeling$", "trajectory_modeling")):
            if _re.search(pat, n):
                return lab
        return None

    for _name, _m in model.named_modules():
        _c = _cat(_name)
        if _c is None:
            continue
        def _pre(m, a, _k=_name):
            _T0[_k] = time.perf_counter()
        def _post(m, a, o, _k=_name, _c=_c):
            PROF[_c] = PROF.get(_c, 0.0) + time.perf_counter() - _T0.pop(_k, time.perf_counter())
        _m.register_forward_pre_hook(_pre)
        _m.register_forward_hook(_post)
    log("profile hooks on", sum(1 for n, _ in model.named_modules() if _cat(n)), "modules")

if args.qlinear != "none":
    torch.backends.quantized.engine = "onednn"
    _tr = [m for n, m in model.named_modules() if n.endswith("detr.transformer") or n == "detr.transformer"][0]
    _target = _tr.encoder if args.qlinear == "enc" else _tr
    _q = torch.ao.quantization.quantize_dynamic(_target, {torch.nn.Linear}, dtype=torch.qint8)
    if args.qlinear == "enc":
        _tr.encoder = _q
    else:
        _par = [m for n, m in model.named_modules() if n.endswith("detr") and hasattr(m, "transformer")][0]
        _par.transformer = _q
    log("dynamic int8 Linear layers:", args.qlinear)

# ---- NPU body (only with --npu-backbone) -----------------------------------------------------------
NPU = dict(raw=None, ms=0.0, calls=0, sha=None, names=None)
if args.npu_backbone:
    import collections
    import numpy as np
    import dx_engine as dx
    from models.deformable_detr.backbone import BackboneBase
    if args.shorter != 480:
        raise SystemExit("--npu-backbone needs --shorter 480 (the NPU graph is fixed at 480x640)")
    NPU["sha"] = sha256_file(args.npu_backbone)
    opt = dx.InferenceOption()
    opt.devices = [0]
    opt.bound_option = getattr(dx.InferenceOption.BOUND_OPTION, "NPU_%d" % args.npu_core)
    opt.use_ort = True
    ENG = dx.InferenceEngine(args.npu_backbone, opt)
    NPU["names"] = [x["name"] for x in ENG.get_output_tensors_info()]
    SHAPES = [(1, 512, 60, 80), (1, 1024, 30, 40), (1, 2048, 15, 20)]

    class NpuBody(torch.nn.Module):
        def forward(self, x):
            if tuple(x.shape[-2:]) != (480, 640) or NPU["raw"] is None:
                raise RuntimeError("NPU body: input %s, raw frame %s" % (tuple(x.shape), NPU["raw"] is not None))
            t = time.perf_counter()
            outs = ENG.run(NPU["raw"])
            NPU["ms"] += (time.perf_counter() - t) * 1e3
            NPU["calls"] += 1
            NPU["raw"] = None   # one NPU call per frame; a second call without a new frame is an error
            feats = []
            for o, shp in zip(outs, SHAPES):
                a = np.asarray(o, dtype=np.float32)
                if a.size != int(np.prod(shp)):
                    raise RuntimeError("NPU body: output size %d, expected %s" % (a.size, shp))
                feats.append(torch.from_numpy(a.reshape(shp).copy()))
            return collections.OrderedDict((str(k), f) for k, f in enumerate(feats))

    bb = [m for m in model.modules() if isinstance(m, BackboneBase)]
    if len(bb) != 1:
        raise SystemExit("expected one BackboneBase in the model, found %d" % len(bb))
    bb[0].body = NpuBody()
    log("NPU body:", args.npu_backbone, "core", args.npu_core, "outputs", NPU["names"])

settings = dict(
    config=os.path.relpath(cfg_path, MOTIP).replace("\\", "/"),
    ASSIGNMENT_PROTOCOL=cfg.get("ASSIGNMENT_PROTOCOL", "hungarian"), MISS_TOLERANCE=cfg["MISS_TOLERANCE"],
    DET_THRESH=cfg["DET_THRESH"], NEWBORN_THRESH=cfg["NEWBORN_THRESH"], ID_THRESH=cfg["ID_THRESH"],
    AREA_THRESH=cfg.get("AREA_THRESH", 0), USE_FOCAL_LOSS=cfg.get("USE_FOCAL_LOSS", False),
    INFERENCE_DTYPE=cfg.get("INFERENCE_DTYPE", "FP32"), SIZE_DIVISIBILITY=cfg.get("SIZE_DIVISIBILITY", 0),
    NUM_ID_VOCABULARY=cfg["NUM_ID_VOCABULARY"], INFERENCE_MAX_LONGER_in_yaml=cfg["INFERENCE_MAX_LONGER"],
    only_detr=False, resize_shorter=args.shorter, resize_longer_cap=args.longer,
)
if settings["INFERENCE_DTYPE"] != "FP32":
    raise SystemExit("this runner is FP32 only")

first_img = Image.open(os.path.join(args.frames, names[all_png[0]]))
W, H = first_img.size

# data/seq_dataset.py:28-33, literally
transform = v2.Compose([
    v2.Resize(size=args.shorter, max_size=args.longer),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tracker = RuntimeTracker(
    model=model, sequence_hw=(H, W),
    use_sigmoid=settings["USE_FOCAL_LOSS"], assignment_protocol=settings["ASSIGNMENT_PROTOCOL"],
    miss_tolerance=settings["MISS_TOLERANCE"], det_thresh=settings["DET_THRESH"],
    newborn_thresh=settings["NEWBORN_THRESH"], id_thresh=settings["ID_THRESH"],
    area_thresh=settings["AREA_THRESH"], only_detr=False, dtype=torch.float32,
)

# ---- timing hooks on the official model (no logic change) ---------------------------------------
PART = dict(t=0.0, detr=0.0, other=0.0, cand=0, max_score=None)


def _pre(module, a, kw):
    PART["t"] = time.perf_counter()


def _post(module, a, kw, out):
    dt = time.perf_counter() - PART["t"]
    if kw.get("part") == "detr":
        PART["detr"] += dt
        sc = out["pred_logits"][0].sigmoid().max(dim=-1).values
        PART["cand"] = int((sc > settings["DET_THRESH"]).sum())
        PART["max_score"] = float(sc.max())
    else:
        PART["other"] += dt


model.register_forward_pre_hook(_pre, with_kwargs=True)
model.register_forward_hook(_post, with_kwargs=True)

# ---- tracker state save / restore ---------------------------------------------------------------
TENSORS = ["trajectory_features", "trajectory_boxes", "trajectory_id_labels", "trajectory_times", "trajectory_masks"]


def save_state(last_frame, n_done, first_frame, fresh_at):
    st = dict(last_frame=int(last_frame), n_done=int(n_done), first_frame=int(first_frame), fresh_at=int(fresh_at),
              next_id=int(tracker.next_id), id_label_to_id={int(k): int(v) for k, v in tracker.id_label_to_id.items()},
              id_queue=[int(v) for v in tracker.id_queue], settings_json=json.dumps(settings, sort_keys=True),
              ckpt_sha256=ckpt_sha)
    for k in TENSORS:
        st[k] = getattr(tracker, k).clone()
    torch.save(st, STATE + ".tmp")
    os.replace(STATE + ".tmp", STATE)


def keep_rows_up_to(path, last_frame, key):
    if not os.path.exists(path):
        return 0
    kept = []
    with open(path) as fh:
        for line in fh:
            if line.strip() and key(line) <= last_frame:
                kept.append(line)
    with open(path + ".tmp", "w") as fh:
        fh.writelines(kept)
    os.replace(path + ".tmp", path)
    return len(kept)


n_done, first_frame, fresh_at, resumed_from = 0, start, start, None
if args.resume and os.path.exists(STATE):
    st = torch.load(STATE, map_location="cpu", weights_only=True)
    if st["settings_json"] != json.dumps(settings, sort_keys=True) or st["ckpt_sha256"] != ckpt_sha:
        raise SystemExit("state file was made with other settings or weights; refusing to resume")
    tracker.next_id = st["next_id"]
    tracker.id_label_to_id = dict(st["id_label_to_id"])
    q = OrderedSet()
    for v in st["id_queue"]:
        q.add(v)
    tracker.id_queue = q
    for k in TENSORS:
        setattr(tracker, k, st[k])
    n_done, first_frame, fresh_at, resumed_from = st["n_done"], st["first_frame"], st["fresh_at"], st["last_frame"]
    keep_rows_up_to(MOT, resumed_from, lambda s: int(s.split(",", 1)[0]))
    kept = keep_rows_up_to(JSONL, resumed_from, lambda s: json.loads(s)["frame"])
    if kept != n_done:
        raise SystemExit("frames.jsonl has %d rows up to frame %d but the state says %d" % (kept, resumed_from, n_done))
    start = resumed_from + 1
    log("resumed after frame", resumed_from, "frames done", n_done)
else:
    if args.resume:
        log("no state file, starting fresh")
    for pth in (MOT, JSONL):
        open(pth, "w").close()

todo = [n for n in all_png if start <= n <= end]
meta = dict(
    what=("official MOTIP (MCG-NJU/MOTIP) RuntimeTracker; ResNet-50 body on the DX-M1 NPU (INT8), the rest on the CPU"
          if args.npu_backbone else "official MOTIP (MCG-NJU/MOTIP) RuntimeTracker on CPU"),
    npu_backbone=(dict(file=os.path.basename(args.npu_backbone), sha256=NPU["sha"], core=args.npu_core, outputs=NPU["names"],
                       input="raw 0..255 gray replicated to 3 channels, uint8 NHWC [1,480,640,3]; /255 and mean/std inside the graph")
                  if args.npu_backbone else None),
    motip_commit=git_head(MOTIP), checkpoint=os.path.basename(args.ckpt), checkpoint_sha256=ckpt_sha,
    load=load_info, settings=settings,
    attention=("msda_cpu.cpp (C++ CPU kernel, same arithmetic as the official CUDA kernel; checked against the reference on the first call of the run: msda_check_max_abs)"
               if args.msda == "cpp" else "official pure PyTorch reference ms_deform_attn_core_pytorch (grid_sample); CUDA extension not built"),
    resize=dict(official="shorter side 800, longer side capped at 1440 (640x480 -> 1066x800)",
                applied="shorter side %d, longer side capped at %d" % (args.shorter, args.longer),
                is_official=(args.shorter == 800 and args.longer == cfg["INFERENCE_MAX_LONGER"])),
    input="8-bit PNG mode L -> Image.convert('RGB') (gray replicated to 3 channels) -> Resize -> /255 -> ImageNet mean/std, batch 1, FP32",
    frames_dir=args.frames, source_wh=[W, H], sequence_total=len(all_png), png_first=all_png[0], png_last=all_png[-1],
    requested=[first_frame, end], tracker_fresh_at_frame=fresh_at,
    device=str(distributed_device()), threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
    versions=dict(python=sys.version.split()[0], torch=torch.__version__, torchvision=torchvision.__version__,
                  numpy=numpy.__version__),
    host=os.uname().nodename if hasattr(os, "uname") else None, cpu_count=os.cpu_count(),
)


def summary(status, error=None):
    ms = []
    n_rows = 0
    last = None
    if os.path.exists(JSONL):
        for line in open(JSONL):
            if line.strip():
                d = json.loads(line)
                ms.append(d)
                last = d["frame"]
                n_rows += d["n_out"]
    tot = sorted(d["total_ms"] for d in ms)
    tot_warm = sorted(d["total_ms"] for d in ms[1:]) or tot

    def pc(v, q):
        return round(v[min(len(v) - 1, int(q * len(v)))], 1) if v else None

    def mean(key):
        return round(sum(d[key] for d in ms) / len(ms), 1) if ms else None

    out = dict(meta, status=status, error=error, frames_processed=len(ms),
               first_processed=ms[0]["frame"] if ms else None, last_processed=last,
               covers_whole_sequence=(len(ms) == len(all_png) and fresh_at == all_png[0]),
               rows=n_rows, next_id=int(tracker.next_id), msda_reference_calls_this_process=MSDA_CALLS[0],
               network_input_hw=ms[-1]["net_hw"] if ms else None,
               timing_ms=dict(note="wall clock per frame in this process; total = read + preprocess + tracker.update; "
                                   "the first frame of each process includes warm-up and is excluded from *_warm",
                              total_mean=mean("total_ms"), total_median_warm=pc(tot_warm, 0.5), total_p95_warm=pc(tot_warm, 0.95),
                              total_max=pc(tot, 0.9999), detector_mean=mean("detr_ms"), id_head_mean=mean("id_ms"),
                              read_pre_mean=mean("pre_ms"), npu_backbone_mean=(mean("npu_ms") if args.npu_backbone else None),
                              seconds_per_frame_mean=round(sum(tot) / len(tot) / 1e3, 3) if tot else None,
                              wall_sum_s=round(sum(tot) / 1e3, 1)),
               msda_check_max_abs=(MSDA_CHECK[0] if MSDA_CHECK else None), msda=args.msda, qlinear=args.qlinear, mot_file=os.path.basename(MOT), mot_sha256=sha256_file(MOT) if os.path.exists(MOT) else None,
               written=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(SIDE + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(SIDE + ".tmp", SIDE)
    return out


log("frames to do:", len(todo), "range", start, end, "of", len(all_png), "threads", torch.get_num_threads())
summary("running")
status, error = "running", None
fm = open(MOT, "a")
fj = open(JSONL, "a")
last_frame = resumed_from
t_run = time.perf_counter()
with torch.no_grad():
    for k, png in enumerate(todo):
        ta = time.perf_counter()
        img = Image.open(os.path.join(args.frames, names[png]))
        if img.mode != "L" or img.size != (W, H):
            status, error = "failed_input", "frame %d is mode %s size %s, expected 8-bit L %dx%d" % (png, img.mode, img.size, W, H)
            break
        x = transform(img.convert("RGB"))
        if args.npu_backbone:
            NPU["raw"] = np.ascontiguousarray(np.repeat(np.asarray(img, dtype=np.uint8)[None, :, :, None], 3, axis=3))   # [1,480,640,3] uint8 NHWC, as the compiled graph takes it
            NPU["ms"] = 0.0
        nt = nested_tensor_from_tensor_list([x], settings["SIZE_DIVISIBILITY"]).to(args.device)
        tb = time.perf_counter()
        PART["detr"] = PART["other"] = 0.0
        try:
            tracker.update(image=nt)
        except SystemExit as e:  # runtime_tracker.py:135-137 calls exit(-1) on duplicate id labels
            status, error = "failed_tracker_exit", "RuntimeTracker called exit(%r) at frame %d (duplicate id labels)" % (e.code, png)
            break
        res = tracker.get_track_results()
        tc = time.perf_counter()
        ids, scores, boxes = res["id"].tolist(), res["score"].tolist(), res["bbox"].tolist()
        for oid, sc, b in zip(ids, scores, boxes):
            fm.write("%d,%d,%.3f,%.3f,%.3f,%.3f,%.5f,-1,-1,-1\n" % (png, oid, b[0], b[1], b[2], b[3], sc))
        n_done += 1
        last_frame = png
        rec = dict(frame=png, n_out=len(ids), n_cand=PART["cand"], max_score=round(PART["max_score"], 4),
                   pre_ms=round((tb - ta) * 1e3, 2), detr_ms=round(PART["detr"] * 1e3, 1), id_ms=round(PART["other"] * 1e3, 1),
                   update_ms=round((tc - tb) * 1e3, 1), total_ms=round((tc - ta) * 1e3, 1), npu_ms=round(NPU["ms"], 1),
                   hist_T=int(tracker.trajectory_features.shape[0]), hist_N=int(tracker.trajectory_features.shape[1]),
                   net_hw=list(nt.tensors.shape[-2:]))
        fj.write(json.dumps(rec) + "\n")
        fm.flush()
        fj.flush()
        if (k + 1) % args.save_every == 0 or k + 1 == len(todo):
            save_state(png, n_done, first_frame, fresh_at)
            summary("running")
            el = time.perf_counter() - t_run
            log("frame", png, "done", k + 1, "/", len(todo), "%.2f s/frame" % (el / (k + 1)),
                "eta %.1f min" % ((len(todo) - k - 1) * el / (k + 1) / 60), "out", len(ids), "cand", PART["cand"],
                "hist", rec["hist_T"], rec["hist_N"], "next_id", tracker.next_id)
fm.close()
fj.close()
if status == "running":
    status = "complete_range" if last_frame == end else "stopped_early"
s = summary(status, error)
log("status", status, error or "", "frames_processed", s["frames_processed"], "of", len(all_png),
    "whole sequence:", s["covers_whole_sequence"])
log(json.dumps(s["timing_ms"]))
if args.profile and s["frames_processed"]:
    nf = s["frames_processed"]
    for k, v in sorted(PROF.items(), key=lambda kv: -kv[1]):
        log("PROFILE %-34s %8.1f ms/frame" % (k, v * 1e3 / nf))
if status.startswith("failed"):
    sys.exit(2)
