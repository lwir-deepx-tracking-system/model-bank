"""Experiment D vs E on the MOT17 occlusion check set (MOT17-09, -11 grey; never trained on): body-box pose validation (mAP) and the
share of labelled people found, split by how much of them is visible (MOT17 visibility). User 2026-11-14 criterion 1: does E find hidden people better?
Usage (Vast): python eval_mot17_check.py <mot17_gray dir> <mot17_kpt dir> <template_points.npy> <out.json> name=weights.pt ..."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_partial as PP   # noqa: E402

gray, kd, tp, outp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]; models = dict(a.split('=', 1) for a in sys.argv[5:])
PP.install(); PP.set_body_box(tp)
yaml = gray / 'check.yaml'
yaml.write_text('path: %s\ntrain: check/images\nval: check/images\nkpt_shape: [17, 3]\nflip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\nnames:\n  0: person\n' % gray)
from ultralytics import YOLO   # noqa: E402
from ultralytics.models.yolo.pose import PoseValidator   # noqa: E402


class BVal(PoseValidator):
    def build_dataset(self, img_path, mode='val', batch=None):
        return PP.add_body_box_val(super().build_dataset(img_path, mode, batch))


# ground truth body boxes + visibility per person (same row selection as mot17_det_prep.py: frame % 3 == 1, visibility >= 0.1)
gt = []
for lab in sorted((gray / 'check' / 'labels').glob('*.txt')):
    nm, f = lab.stem.rsplit('_', 1); f = int(f); z = np.load(kd / (nm + '.npz'))
    rows = np.flatnonzero((z['frame'] == f) & (z['vis'] >= 0.1)); L = [l.split() for l in lab.read_text().splitlines() if l.strip()]
    assert len(rows) == len(L), (lab, len(rows), len(L))
    W, H = float(z['W']), float(z['H'])
    for r, l in zip(rows, L):
        v = np.array(l, float); k = v[5:].reshape(17, 3).copy(); k[:, 0] *= 640; k[:, 1] *= 640 * H / W
        bb = PP.to_body_boxes(np.array([[0, 0, 1, 1]], float), k[None], size=None)
        gt.append(dict(img=lab.stem, vis=float(z['vis'][r]), box=np.asarray(bb, float).reshape(-1)[:4]))

res = {}
for name, w in models.items():
    v = BVal(args=dict(model=w, data=str(yaml), imgsz=640, batch=16, split='val', plots=False, verbose=False)); m = v()
    r = dict(pose_map50_95=float(m.get('metrics/mAP50-95(P)', np.nan)), box_map50=float(m.get('metrics/mAP50(B)', np.nan)), box_map50_95=float(m.get('metrics/mAP50-95(B)', np.nan)))
    model = YOLO(w); found = {}
    for img in sorted({g['img'] for g in gt}):
        p = model.predict(str(gray / 'check' / 'images' / (img + '.jpg')), imgsz=640, conf=0.25, verbose=False)[0]
        found[img] = p.boxes.xyxy.cpu().numpy() if p.boxes is not None else np.zeros((0, 4))
    bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)]; rec = {}
    for lo, hi in bins:
        sel = [g for g in gt if lo <= g['vis'] < hi]; hit = 0
        for g in sel:
            b = found[g['img']]; x = g['box']
            if not len(b): continue
            iw = np.clip(np.minimum(b[:, 2], x[2]) - np.maximum(b[:, 0], x[0]), 0, None); ih = np.clip(np.minimum(b[:, 3], x[3]) - np.maximum(b[:, 1], x[1]), 0, None)
            iou = iw * ih / ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + (x[2] - x[0]) * (x[3] - x[1]) - iw * ih + 1e-9); hit += int(iou.max() >= 0.5)
        rec['vis_%.1f-%.1f' % (lo, min(hi, 1.0))] = dict(people=len(sel), found_share=hit / max(len(sel), 1))
    r['found_by_visibility_iou50_conf25'] = rec; r['detections'] = int(sum(len(b) for b in found.values())); res[name] = r
    print(name, json.dumps(r), flush=True)
Path(outp).write_text(json.dumps(res, indent=1)); print('EVAL DONE')
