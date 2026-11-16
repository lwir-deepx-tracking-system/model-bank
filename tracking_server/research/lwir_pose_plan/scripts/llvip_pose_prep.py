"""LLVIP-Pose (Cormier et al., ACCVW 2024; annotations MIT, github.com/MickaelCormier/llvip-pose, release v1.0.0 annotations.zip) on the
LLVIP thermal pictures -> a YOLO pose set in our COCO-17 layout (2026-11-14, user: follow the literature study, research_transfer_20261114).

LLVIP-Pose uses the PoseTrack order: 0 nose, 1 head_bottom, 2 head_top, 3/4 ears (never labelled), 5-16 = COCO 5-16. COCO 1/2 are the eyes.
Eyes and ears are not labelled here, and "not labelled" must not teach "not visible": they get flag 3 = IGNORE (pose_partial.py), which the
patched loss leaves out of both the position and the visibility terms. head_bottom / head_top are dropped (the body box comes from the
template fit, tools/skeleton_box.py). Flags 0 (not labelled), 1 (labelled, hidden), 2 (visible) are kept.
Pictures: LLVIP infrared train jpg 1280x1024 -> grey 640x512 (3 equal channels, as our other grey sets). Only frames that LLVIP-Pose labelled.
People with fewer than 2 labelled body points keep their box (pose_partial keeps the box when the template cannot be fitted); counted.
Usage (Vast): python llvip_pose_prep.py <llvip_pose dir with train_llvip.json> <LLVIP dir with infrared/train> <out dir>"""
import json, sys
from pathlib import Path

import numpy as np
from PIL import Image

ann_dir, llvip, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
IGNORE = 3
MAP = {0: 0, **{k: k for k in range(5, 17)}}          # LLVIP-Pose index -> COCO index (same where it exists)
W2, H2 = 640, 512
d = json.load(open(ann_dir / 'train_llvip.json'))
anns = {}
for a in d['annotations']: anns.setdefault(a['image_id'], []).append(a)
(out / 'train' / 'images').mkdir(parents=True, exist_ok=True); (out / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
rep = dict(images=0, people=0, few_points=0, flags={0: 0, 1: 0, 2: 0, 3: 0}, missing_picture=0, size_mismatch=0)
for im in d['images']:
    if not im.get('is_labeled', True): continue
    src = llvip / 'infrared' / 'train' / im['file_name']
    if not src.exists(): rep['missing_picture'] += 1; continue
    pic = Image.open(src).convert('L')
    if pic.size != (im['width'], im['height']): rep['size_mismatch'] += 1
    W, H = pic.size
    pic.resize((W2, H2), Image.BILINEAR).convert('RGB').save(out / 'train' / 'images' / im['file_name'], quality=95)
    rows = []
    for a in anns.get(im['id'], []):
        x, y, w, h = a['bbox']
        if w < 2 or h < 2: continue
        k = np.asarray(a['keypoints'], float).reshape(17, 3); kp = np.zeros((17, 3)); kp[:, 2] = IGNORE
        for s, t in MAP.items():
            if k[s, 2] > 0: kp[t] = [k[s, 0] / W, k[s, 1] / H, k[s, 2]]
            else: kp[t] = [0, 0, 0]
        for t in range(17): rep['flags'][int(kp[t, 2])] += 1
        if ((kp[:, 2] == 1) | (kp[:, 2] == 2)).sum() < 2: rep['few_points'] += 1
        cx, cy = np.clip((x + w / 2) / W, 0, 1), np.clip((y + h / 2) / H, 0, 1)
        bw, bh = min(w / W, 1.0), min(h / H, 1.0)
        kp[:, :2] = kp[:, :2].clip(0, 1)
        rows.append('0 %.6f %.6f %.6f %.6f ' % (cx, cy, bw, bh) + ' '.join('%.6f %.6f %d' % (p[0], p[1], int(p[2])) for p in kp))
        rep['people'] += 1
    (out / 'train' / 'labels' / (Path(im['file_name']).stem + '.txt')).write_text('\n'.join(rows) + ('\n' if rows else ''))
    rep['images'] += 1
rep['flags'] = {str(k): v for k, v in rep['flags'].items()}
(out / 'PREP.json').write_text(json.dumps(dict(source='LLVIP-Pose v1.0.0 train_llvip.json + LLVIP infrared/train', ignore_flag=IGNORE,
                                               map='PoseTrack 0,5..16 -> COCO same index; COCO 1-4 (eyes, ears) = ignore; head_bottom/top dropped',
                                               size='%dx%d grey' % (W2, H2), **rep), indent=1))
print(json.dumps(rep))
