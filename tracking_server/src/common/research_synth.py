"""가짜 가림 계획: 정해 둔 프레임에서 사람 검출을 일부러 숨겨 추적기를 시험한다 (연구용 옵션 --synth-plan)."""
import json
from pathlib import Path
import numpy as np
from ..common.shared_values import HERE, PERSON_CLS, sha


# ------------------------------------------------------------------------------------------------------- synthetic occlusion
def load_synth_plan(path):
    """--synth-plan: tracker_ab/SYNTH_PLAN.json (make_synth_plan.py). Returns (plan, {image_name: float64 [m,4] GT boxes to hide})."""
    p = Path(path)
    if not p.is_absolute() and not p.exists(): p = HERE / p
    plan = json.loads(p.read_text(encoding='utf-8'))
    hide = {k: np.asarray([b[2:6] for b in v], np.float64).reshape(-1, 4) for k, v in plan['frames'].items()}
    plan['_path'] = str(p); plan['_sha256'] = sha(p)
    return plan, hide


def apply_synth_plan(arr, boxes, thr=0.5):
    """Hide every PERSON detection whose IoU with a planned GT box is >= thr by setting its conf to 0 (the row is kept so the detection
    index space of TRACKS.jsonl is unchanged; the tracker ignores conf <= .10, so the hidden person is invisible to the tracker and to
    the ViT crop selection, which only crops tracker detections / tracks). Returns (arr', n_hidden, n_planned_boxes_without_detection)."""
    if not len(arr) or not len(boxes): return arr, 0, len(boxes)
    a = arr[:, :4].astype(np.float64); b = boxes
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)
    hit = (iou >= thr) & (arr[:, 5].astype(int) == PERSON_CLS)[:, None]
    drop = hit.any(1)
    if not drop.any(): return arr, 0, len(boxes)
    arr = arr.copy(); arr[drop, 4] = 0.0
    return arr, int(drop.sum()), int((~hit.any(0)).sum())
