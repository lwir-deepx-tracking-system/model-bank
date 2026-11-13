"""Two-frame, no-training PC smoke test using upstream's PyTorch attention reference.

The reference is not the CUDA kernel and this is not a latency/accuracy benchmark.
It does not modify the upstream tree, the GUI, or the M1 server.
"""
import argparse
import json
import os
import sys
import time
import types
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

root = Path(__file__).resolve().parent
upstream = Path(os.environ['MOTIP_UPSTREAM'])
sys.path.insert(0, str(upstream))
sys.modules['MultiScaleDeformableAttention'] = types.ModuleType('MultiScaleDeformableAttention')
os.chdir(upstream)

from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch


def reference_apply(value, shapes, starts, sampling, weights, im2col_step):
    return ms_deform_attn_core_pytorch(value, shapes, sampling, weights)


MSDeformAttnFunction.apply = staticmethod(reference_apply)
import models.deformable_detr.backbone as backbone
backbone.is_main_process = lambda: False  # checkpoint includes backbone; no network download
from utils.misc import yaml_to_dict, distributed_device
from configs.util import load_super_config
from models.motip import build
from models.runtime_tracker import RuntimeTracker
from utils.nested_tensor import nested_tensor_from_tensor_list

# Model-only copy was extracted with torch 2.7 weights_only=True from the official file.
checkpoint = (Path(os.environ['MOTIP_ARTIFACTS']) / 'motip_dancetrack_model_only.pth')
archive = Path(os.environ['LWIR_SOURCE_ARCHIVE'])
config = yaml_to_dict(str(upstream / 'configs/r50_deformable_detr_motip_dancetrack.yaml'))
config = load_super_config(config, config['SUPER_CONFIG_PATH'])
device = distributed_device()
config['DEVICE'] = str(device)
model, _ = build(config)
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(state['model'], strict=True)
del state
model.eval().to(device)

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', choices=['seq_001', 'seq_014'], default='seq_014')
parser.add_argument('--shorter', type=int, choices=[256, 480], default=256)
args = parser.parse_args()
prefix = f'LWIR_labels/opi_sequences/{args.sequence}/images/'
rows = []
tracker = None
with zipfile.ZipFile(archive) as z:
    names = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith('.png'))[:2]
    assert len(names) == 2
    for name in names:
        image = cv2.imdecode(np.frombuffer(z.read(name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert image is not None and image.ndim == 2
        dtype, lo, hi = str(image.dtype), int(image.min()), int(image.max())
        if image.dtype == np.uint16:
            image = (image // 256).astype(np.uint8)
        assert image.dtype == np.uint8
        height, width = image.shape
        if tracker is None:
            tracker = RuntimeTracker(model=model, sequence_hw=(height, width),
                                     assignment_protocol='object-max', miss_tolerance=30,
                                     det_thresh=.5, newborn_thresh=.5, id_thresh=.2)
        rgb = np.repeat(image[:, :, None], 3, axis=2)
        tensor = F.to_tensor(rgb)
        tensor = F.resize(tensor, size=args.shorter, max_size=640 if args.shorter == 480 else 455)
        tensor = F.normalize(tensor, mean=[.485, .456, .406], std=[.229, .224, .225])
        nested = nested_tensor_from_tensor_list([tensor.to(device)])
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        tracker.update(nested)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        results = tracker.get_track_results()
        rows.append(dict(frame=name.rsplit('/', 1)[-1], source_dtype=dtype,
                         source_min=lo, source_max=hi, source_hw=[height, width],
                         resized_hw=list(tensor.shape[-2:]),
                         boxes=results['bbox'].detach().cpu().tolist(),
                         ids=results['id'].detach().cpu().tolist(),
                         scores=results['score'].detach().cpu().tolist(),
                         elapsed_ms=round((time.perf_counter()-start)*1000, 3)))

result = dict(scope=f'two-frame functionality smoke only; {args.shorter} short side; reference attention',
              device=str(device), inference_not_benchmark=True, checkpoint_strict_load=True,
              source=f'{args.sequence} archive', frames=rows)
path = root / f'motip_reference_smoke_{args.sequence}_{args.shorter}.json'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
