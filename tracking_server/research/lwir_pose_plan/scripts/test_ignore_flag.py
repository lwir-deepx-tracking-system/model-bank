"""11-14 check of flag 3 = IGNORE (pose_partial.py) on the LLVIP-Pose set: through the full training augmentation (PartialCrop, Occlude,
ThermalLook, flips, body boxes) the eye/ear slots must stay 3 (never become "hidden" 1 or "visible" 2), the body-box fit must not use them,
and the patched loss must run (finite, gradients) with them present.
Usage: python test_ignore_flag.py <data.yaml whose train split is LLVIP-Pose> <weights.pt> <template_points.npy> [n=300]"""
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_partial  # noqa: E402


def main():
    data_yaml, weights, template = sys.argv[1], sys.argv[2], sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    pose_partial.install(); pose_partial.set_body_box(template)
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG
    np.random.seed(0); torch.manual_seed(0)
    import random; random.seed(0)
    cfg = get_cfg(DEFAULT_CFG, overrides=dict(task='pose', mode='train', imgsz=640, mosaic=0.0, close_mosaic=0, scale=0.5, translate=0.2, degrees=5.0,
                                               fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, mixup=0.0, copy_paste=0.0, data=data_yaml))
    data = check_det_dataset(data_yaml)
    ds = build_yolo_dataset(cfg, data['train'], batch=4, data=data, mode='train', rect=False, stride=32)
    pose_partial.add_partial_crop(ds, p=0.5, seed=0, occlude_p=0.5, thermal_look=True)
    st = dict(samples=0, people_before=0, people_after=0, ignore_before=0, ignore_after=0, eye_ear_not_ignore_after=0, other_slots_ignore_after=0)
    n = min(n, len(ds))
    for i in range(n):
        k0 = ds.labels[i]['keypoints']; st['people_before'] += len(k0); st['ignore_before'] += int((k0[..., 2] == 3).sum())
        s = ds[i]; k = s['keypoints'].numpy(); st['samples'] += 1; st['people_after'] += len(k)
        f = k[..., 2]
        st['ignore_after'] += int((f == 3).sum())
        st['eye_ear_not_ignore_after'] += int((f[:, 1:5] != 3).sum())
        st['other_slots_ignore_after'] += int((f[:, [0] + list(range(5, 17))] == 3).sum())
    # body box: an ignore slot at a far-away position must not move the fit
    kp = np.zeros((1, 17, 3), np.float32); kp[0, :, 2] = 2
    T = np.array([[320, 100], [0, 0], [0, 0], [0, 0], [0, 0], [290, 140], [350, 140], [280, 190], [360, 190], [275, 240], [365, 240],
                  [300, 240], [340, 240], [300, 320], [340, 320], [300, 400], [340, 400]], np.float32); kp[0, :, :2] = T
    kp[0, 1:5, 2] = 3; a = pose_partial.to_body_boxes(np.array([[0, 0, 1, 1]], np.float32), kp.copy())
    kp[0, 1:5, :2] = [[5000, 5000]] * 4; b = pose_partial.to_body_boxes(np.array([[0, 0, 1, 1]], np.float32), kp.copy())
    st['body_box_ignores_flag3'] = bool(np.allclose(a, b))
    # loss with flag-3 slots present
    from ultralytics import YOLO
    model = YOLO(weights).model; model.args = cfg
    for p in model.parameters(): p.requires_grad_(True)
    model.train(); batch = ds.collate_fn([ds[i] for i in range(4)]); batch['img'] = batch['img'].float() / 255
    loss, items = model(batch); loss.sum().backward()
    st['loss_finite'] = bool(torch.isfinite(loss).all()); st['grad_positive'] = float(sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)) > 0
    st['loss_parts'] = [round(float(v), 4) for v in (items.values() if isinstance(items, dict) else items)]
    st['ALL_OK'] = (st['eye_ear_not_ignore_after'] == 0 and st['other_slots_ignore_after'] == 0 and st['ignore_after'] > 0 and st['body_box_ignores_flag3']
                    and st['loss_finite'] and st['grad_positive'])
    print(json.dumps(st, indent=1))


if __name__ == '__main__':
    main()
