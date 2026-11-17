"""Does "colour pose -> copy to the thermal picture" give good thermal keypoints? (2026-11-15, user's plan: colour/thermal pairs -> thermal pose.)
LLVIP pairs are pixel-registered, and LLVIP-Pose test has human-corrected thermal keypoints, so the copy can be scored directly (no training).
Methods, all scored against LLVIP-Pose test (3,463 frames, never used for training here):
  rgb_copy      big public pose model (yolo26x-pose) on the COLOUR picture, keypoints used as they are on the thermal picture
  thermal_x     the same model directly on the THERMAL picture (grey, 3 channels)
  thermal_ours  our thermal model (experiment E weights) on the thermal picture (optional)
Matching (the same for every method): each labelled person takes the unused prediction with the smallest mean distance over the person's
labelled body joints (nose, shoulders, elbows, wrists, hips, knees, ankles), if under 0.5 x the person's box height.
Scores: found share, PCK@0.05 / @0.1 of the box height over all labelled joints (missed people count as wrong), median error of matched
joints; split by person size (box height at 1280x1024: <150, 150-250, >=250 px) and by colour-picture brightness (mean grey <50, 50-100, >=100).
Usage (Vast): python llvip_transfer_check.py <llvip_pose dir> <LLVIP dir> <out.json> <yolo26x-pose.pt> [ours=<weights.pt>]"""
import json, sys
from pathlib import Path

import numpy as np
from PIL import Image

pose_dir, llvip, outp, teacher = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
ours = dict(a.split('=', 1) for a in sys.argv[5:]).get('ours')
from ultralytics import YOLO   # noqa: E402

J = [0] + list(range(5, 17))            # LLVIP-Pose (PoseTrack order) indices that equal COCO indices
d = json.load(open(pose_dir / 'test_llvip.json'))
anns = {}
for a in d['annotations']: anns.setdefault(a['image_id'], []).append(a)
models = {'rgb_copy': (YOLO(teacher), 'visible'), 'thermal_x': (YOLO(teacher), 'infrared')}
if ours: models['thermal_ours'] = (YOLO(ours), 'infrared')


def size_bin(h): return '<150' if h < 150 else ('150-250' if h < 250 else '>=250')
def light_bin(v): return '<50' if v < 50 else ('50-100' if v < 100 else '>=100')


stat = {m: {} for m in models}
def add(m, key, found, n_j, ok05, ok10, errs):
    s = stat[m].setdefault(key, dict(people=0, found=0, joints=0, pck05=0, pck10=0, errs=[]))
    s['people'] += 1; s['found'] += int(found); s['joints'] += n_j; s['pck05'] += ok05; s['pck10'] += ok10; s['errs'] += errs


for k, im in enumerate(d['images']):
    people = anns.get(im['id'], [])
    if not people: continue
    vis = np.asarray(Image.open(llvip / 'visible' / 'test' / im['file_name']).convert('L'))
    light = light_bin(float(vis.mean()))
    for m, (model, side) in models.items():
        pic = np.asarray(Image.open(llvip / side / 'test' / im['file_name']).convert('RGB'))[..., ::-1]
        r = model.predict(np.ascontiguousarray(pic), imgsz=640, conf=0.25, verbose=False)[0]
        pk = r.keypoints.data.cpu().numpy() if r.keypoints is not None and len(r.keypoints) else np.zeros((0, 17, 3))
        used = set()
        for a in sorted(people, key=lambda a: -a['bbox'][3]):
            g = np.asarray(a['keypoints'], float).reshape(17, 3)[J]; lab = g[:, 2] > 0; h = float(a['bbox'][3])
            if lab.sum() == 0: continue
            best, bd = None, 1e9
            for p in range(len(pk)):
                if p in used: continue
                dist = np.linalg.norm(pk[p][J][lab, :2] - g[lab, :2], axis=1).mean() / h
                if dist < bd: best, bd = p, dist
            found = best is not None and bd < 0.5
            if found:
                used.add(best); e = np.linalg.norm(pk[best][J][lab, :2] - g[lab, :2], axis=1) / h
                ok05, ok10, errs = int((e < 0.05).sum()), int((e < 0.1).sum()), e.tolist()
            else: ok05 = ok10 = 0; errs = []
            for key in ('all', 'size ' + size_bin(h), 'light ' + light):
                add(m, key, found, int(lab.sum()), ok05, ok10, errs)
    if k % 500 == 0: print(k, flush=True)

rep = {}
for m, groups in stat.items():
    rep[m] = {key: dict(people=s['people'], found=round(100 * s['found'] / max(1, s['people']), 1), pck05=round(100 * s['pck05'] / max(1, s['joints']), 1),
                        pck10=round(100 * s['pck10'] / max(1, s['joints']), 1), median_err=round(float(np.median(s['errs'])), 4) if s['errs'] else None)
              for key, s in sorted(groups.items())}
json.dump(dict(test='LLVIP-Pose test_llvip.json', teacher=teacher, ours=ours, imgsz=640, conf=0.25, joints='nose, shoulders, elbows, wrists, hips, knees, ankles',
               methods=rep), open(outp, 'w'), indent=1)
print(json.dumps(rep, indent=1))
