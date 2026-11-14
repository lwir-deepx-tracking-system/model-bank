"""MOT17 -> grey YOLO-pose training set with keypoints for EVERY person (user 2026-11-14: "다 관절달고 흑백으로 바꿔서 하면되겠다").

Why: MOT17 labels the full body of a person even while other people or objects hide most of it (visibility 0..1) - real occlusions with the true
body position, the case our close-range body-box detector must learn. Our training (pose_partial body-box mode) turns 17 keypoints into the body
box (head top .. hips, template fit), so every person needs all 17 keypoints:
  - visible ones from the public COCO pose model (mot17_kpt_label.py full frame) and, for people it missed, a second pass on an enlarged crop;
    kept as flag 2 (visible);
  - the rest from the body template (tools/skeleton_box.py) fitted to the visible ones (>= 2 points, rms check), else fitted to the labelled
    full-body box (head top -> box top, ankles -> box bottom); flag 1 (labelled position, not visible) - the same meaning as COCO / pose_partial.
People with visibility < 0.1 are left out (nothing of them can be seen; labelling them would teach the detector to invent people).
Pictures: grey, longest side 640, every 3rd frame. MOT17-09 and MOT17-11 form a separate occlusion check set (never trained on).

Usage (Vast): python mot17_det_prep.py <MOT17/train> <mot17_kpt dir> <template_points.npy> <skeleton_box.py dir> <yolo26x-pose.pt> <out>
"""
import configparser, json, sys
from pathlib import Path
import numpy as np, cv2

TRAIN = ('MOT17-02', 'MOT17-04', 'MOT17-10', 'MOT17-13'); CHECK = ('MOT17-09', 'MOT17-11'); EVERY = 3; MINVIS = 0.1


def main():
    root, kd, tp, sbd, wts, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], Path(sys.argv[6])
    sys.path.insert(0, sbd); import skeleton_box as SB
    tmpl = SB.template_from_points(np.load(tp)); P = tmpl['points'].copy(); P[:, 1] *= -1   # image orientation
    top_t = -tmpl['box'][3]; ank_t = P[[15, 16], 1].mean()                                   # head top / ankles in template units (y down)
    from ultralytics import YOLO
    model = YOLO(wts); rep = {}
    for split, names in (('train', TRAIN), ('check', CHECK)):
        (out / split / 'images').mkdir(parents=True, exist_ok=True); (out / split / 'labels').mkdir(parents=True, exist_ok=True)
        st = dict(images=0, people=0, from_pose=0, from_crop=0, template_from_points=0, template_from_box=0, visible_kpts=0, hidden_kpts=0)
        for nm in names:
            z = np.load(kd / (nm + '.npz')); W, H = int(z['W']), int(z['H']); sd = root / (nm + '-FRCNN')
            fr, box, vis, kpt, matched = z['frame'], z['box'], z['vis'], z['kpt'], z['matched']
            for f in range(1, int(fr.max()) + 1, EVERY):
                rows = np.flatnonzero((fr == f) & (vis >= MINVIS))
                img = cv2.imread(str(sd / 'img1' / ('%06d.jpg' % f)))
                if img is None: continue
                kp_all = kpt[rows].copy(); src = np.where(matched[rows], 1, 0)
                need = [i for i, j in enumerate(rows) if not matched[j] and vis[j] >= 0.3]
                if need:   # second pass: enlarged crops of the people the full-frame pass missed
                    crops, offs = [], []
                    for i in need:
                        cx, cy, w, h = box[rows[i]]; x1, y1 = int(max(0, cx - 0.7 * w)), int(max(0, cy - 0.6 * h)); x2, y2 = int(min(W, cx + 0.7 * w)), int(min(H, cy + 0.6 * h))
                        c = img[y1:y2, x1:x2]
                        if c.size == 0 or c.shape[0] < 8 or c.shape[1] < 8: crops.append(None); offs.append(None); continue
                        s = 384.0 / c.shape[0]; crops.append(cv2.resize(c, (max(8, int(c.shape[1] * s)), 384))); offs.append((x1, y1, s))
                    ok = [k for k, c in enumerate(crops) if c is not None]
                    res = model.predict([crops[k] for k in ok], imgsz=384, conf=0.2, verbose=False) if ok else []
                    for k, r in zip(ok, res):
                        if r.keypoints is None or not len(r.boxes): continue
                        x1, y1, s = offs[k]; cxl, cyl = box[rows[need[k]]][:2]
                        b = r.boxes.xyxy.cpu().numpy() / s + [x1, y1, x1, y1]; d = np.hypot((b[:, 0] + b[:, 2]) / 2 - cxl, (b[:, 1] + b[:, 3]) / 2 - cyl); m = int(np.argmin(d))
                        kp = r.keypoints.data.cpu().numpy()[m].copy(); kp[:, :2] = kp[:, :2] / s + [x1, y1]; v = kp[:, 2] >= 0.5; kp[:, 2] = v; kp[~v, :2] = 0
                        if v.sum() >= 2: kp_all[need[k]] = kp; src[need[k]] = 2
                lines = []
                for i, j in enumerate(rows):
                    cx, cy, w, h = box[j]; kp = kp_all[i]; v = kp[:, 2] > 0; full = np.zeros((17, 3))
                    fit = SB.fit(tmpl, kp[:, :2], v) if v.sum() >= 2 else None
                    if fit is not None and fit['rms'] / fit['scale'] < 0.35:   # rms in template units (torso lengths)
                        pts = (fit['scale'] * (fit['R'] @ P.T).T + fit['t']); st['template_from_points'] += 1
                    else:
                        s = h / max(ank_t - top_t, 1e-6); pts = np.c_[cx + s * P[:, 0], (cy - h / 2) + s * (P[:, 1] - top_t)]; st['template_from_box'] += 1
                    full[:, :2] = pts; full[:, 2] = 1; full[v, :2] = kp[v, :2]; full[v, 2] = 2
                    st['people'] += 1; st['from_pose'] += int(src[i] == 1); st['from_crop'] += int(src[i] == 2); st['visible_kpts'] += int(v.sum()); st['hidden_kpts'] += int((~v).sum())
                    vals = [0, cx / W, cy / H, w / W, h / H] + np.c_[full[:, 0] / W, full[:, 1] / H, full[:, 2]].ravel().tolist()
                    lines.append(' '.join('%d' % x if q == 0 or (q > 4 and (q - 5) % 3 == 2) else '%.6f' % x for q, x in enumerate(vals)))
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); g = cv2.resize(g, (640, int(round(H * 640 / W))), interpolation=cv2.INTER_AREA)
                stem = '%s_%06d' % (nm, f); cv2.imwrite(str(out / split / 'images' / (stem + '.jpg')), g, [cv2.IMWRITE_JPEG_QUALITY, 90])
                (out / split / 'labels' / (stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else '')); st['images'] += 1
            print(split, nm, json.dumps(st), flush=True)
        rep[split] = st
    (out / 'PREP.json').write_text(json.dumps(rep, indent=1)); print('PREP DONE', flush=True)


if __name__ == '__main__':
    main()
