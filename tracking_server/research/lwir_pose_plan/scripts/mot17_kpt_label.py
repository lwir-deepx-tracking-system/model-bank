"""MOT17 train -> long person tracks with keypoints for the long-horizon position memory (user 2026-11-14: "학습길이를 엄청 늘려봐", "관절없으면 안되지않나").

MOT17 gt.txt gives, per frame and person, the FULL body box even while the person is hidden (amodal) plus the visible share (visibility 0..1). That is the same idea
as our body box, and the hidden stretches are real disappearances with known positions. Keypoints are not labelled in MOT17, so the public COCO pose model
(Ultralytics yolo26m-pose) is run on every frame and its people are matched to the labelled boxes (pseudo labels, COCO keypoint order = our detector's order).

Usage (Vast, GPU): python mot17_kpt_label.py <MOT17/train> <yolo26m-pose.pt> <out_dir> [imgsz=1280]
Out: <out>/<seq>.npz with fps, W, H, frame [n], tid [n], box [n,4] (cx, cy, w, h of the labelled full body), vis [n] (visible share), kpt [n,17,3] (x, y, visible 0/1;
all 0 when no pose person matched), matched [n]; <out>/LABEL.json (counts per sequence).
"""
import configparser, json, sys, time
from pathlib import Path
import numpy as np


def iou_mat(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1]); x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None); aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-6), inter / np.maximum(bb[None, :], 1e-6)   # IoU, share of the pose box inside the labelled box


def main():
    root, wts, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); imgsz = int(sys.argv[4]) if len(sys.argv) > 4 else 1280
    out.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(wts); rep = {}
    for sd in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith('-FRCNN')):
        ini = configparser.ConfigParser(); ini.read(sd / 'seqinfo.ini'); s = ini['Sequence']; fps, W, H, n = float(s['frameRate']), int(s['imWidth']), int(s['imHeight']), int(s['seqLength'])
        g = np.loadtxt(sd / 'gt' / 'gt.txt', delimiter=','); g = g[(g[:, 6] == 1) & (g[:, 7] == 1)]   # used for scoring, class pedestrian
        fr, tid = g[:, 0].astype(int), g[:, 1].astype(int); xywh = g[:, 2:6]; vis = g[:, 8].astype(np.float32)
        box = np.c_[xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]].astype(np.float32)
        kpt = np.zeros((len(g), 17, 3), np.float32); matched = np.zeros(len(g), bool); t0 = time.time()
        for f in range(1, n + 1):
            rows = np.flatnonzero(fr == f)
            r = model.predict(str(sd / 'img1' / ('%06d.jpg' % f)), imgsz=imgsz, conf=0.15, verbose=False, half=True)[0]
            if not len(rows) or r.boxes is None or not len(r.boxes): continue
            pb = r.boxes.xyxy.cpu().numpy(); pk = r.keypoints.data.cpu().numpy()   # [m,17,3] x, y, conf
            gb = np.c_[box[rows, 0] - box[rows, 2] / 2, box[rows, 1] - box[rows, 3] / 2, box[rows, 0] + box[rows, 2] / 2, box[rows, 1] + box[rows, 3] / 2]
            iou, inside = iou_mat(gb, pb)
            score = np.where((inside > 0.8) & (iou > 0.25), iou + 0.5 * vis[rows, None], 0.)   # visible part of a hidden person is smaller than the full box
            used_g, used_p = set(), set()
            for k in np.argsort(-score, axis=None):
                a, b = divmod(int(k), score.shape[1])
                if score[a, b] <= 0: break
                if a in used_g or b in used_p: continue
                used_g.add(a); used_p.add(b); j = rows[a]; kp = pk[b].copy(); v = kp[:, 2] >= 0.5; kp[:, 2] = v; kp[~v, :2] = 0; kpt[j] = kp; matched[j] = True
            if f % 200 == 0: print(sd.name, f, n, '%.1f s' % (time.time() - t0), flush=True)
        np.savez_compressed(out / (sd.name.replace('-FRCNN', '') + '.npz'), fps=fps, W=W, H=H, frame=fr, tid=tid, box=box, vis=vis, kpt=kpt, matched=matched)
        rep[sd.name] = dict(frames=n, fps=fps, size=[W, H], boxes=int(len(g)), tracks=int(len(np.unique(tid))), matched=int(matched.sum()),
                            matched_share_vis_ge_05=float(matched[vis >= 0.5].mean()) if (vis >= 0.5).any() else None, hidden_share_vis_lt_02=float((vis < 0.2).mean()), seconds=round(time.time() - t0))
        print(sd.name, json.dumps(rep[sd.name]), flush=True)
    (out / 'LABEL.json').write_text(json.dumps(rep, indent=1)); print('LABEL DONE', flush=True)


if __name__ == '__main__':
    main()
