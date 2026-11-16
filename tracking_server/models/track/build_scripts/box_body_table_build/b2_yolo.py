"""body_v2/scripts/b2_yolo.py (PC CPU) - the plain whole-person detector the device runs today (YOLO26x FLIR, models/detect/yolo26x/yolo26x.pt =
bank:yolo26x_flir_nms_640x480) on grey 640x480 frames, decoded like the server's 'six' path (one-to-many branch, post.cpp decode + per-class NMS 0.7,
person class 0, score > 0.10, drop_giant) = scripts/det_run.py without the keypoint heads. fp32 on the CPU (the device runs the INT8 DXNN).
  python b2_yolo.py time <weights.pt>
  python b2_yolo.py imgs <weights.pt> <out.npz> <list.txt>            list = one grey 640x480 image path per line -> fi, box, score, names
  python b2_yolo.py rec  <weights.pt> <out_dir> <MEASURE dir> [...]   device recordings (frames.raw.gz, 16 -> 8 bit as the device) -> <out_dir>/yolo_<rec>.npz
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'v2' / 'scripts'))
import det_run as D     # noqa: E402
import dev_det as DD    # noqa: E402  (To8, frames)

BATCH = 4


class Heads(torch.nn.Module):
    """backbone + neck, then the one-to-many box / class convolutions of a Detect head (raw)."""
    def __init__(self, net):
        super().__init__(); self.net = net; self.head = net.model[-1]

    def forward(self, images):
        x = images / 255.0; saved = []
        for m in self.net.model[:-1]:
            if m.f != -1: x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x); saved.append(x if m.i in self.net.save else None)
        f = [saved[j] for j in self.head.f]; h = self.head
        return torch.cat([h.cv2[i](f[i]).flatten(2) for i in range(3)], 2), torch.cat([h.cv3[i](f[i]).flatten(2) for i in range(3)], 2)


def post(box, cls):
    score = (1.0 / (1.0 + np.exp(-cls[0].astype(np.float64)))).astype(np.float32); cand = np.flatnonzero(score > D.SCORE_KEEP)
    if len(cand) == 0: return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    ax, ay, st = D.GX[cand], D.GY[cand], D.GS[cand]; bb = box[:, cand]; l = ax - bb[0]; t = ay - bb[1]; r = ax + bb[2]; btm = ay + bb[3]
    x = ((l + r) * np.float32(.5)) * st; y = ((t + btm) * np.float32(.5)) * st; w = (r - l) * st; h = (btm - t) * st
    xyxy = np.stack([x - w * np.float32(.5), y - h * np.float32(.5), x + w * np.float32(.5), y + h * np.float32(.5)], 1).astype(np.float32)
    k = D.nms_server(xyxy, score[cand]); keep = (xyxy[k, 3] - xyxy[k, 1]) <= D.MAX_H
    return xyxy[k][keep], score[cand][k][keep]


def load(w):
    from ultralytics import YOLO
    torch.set_num_threads(int(os.environ.get('DET_THREADS', '8')))
    net = YOLO(w).model.float().eval(); hd = net.model[-1]
    info = dict(weights=Path(w).name, sha256=D.sha(w), head=type(hd).__name__, nc=int(hd.nc), reg_max=int(hd.reg_max), stride=[float(s) for s in hd.stride], names0=str(net.names[0]) if hasattr(net, 'names') else None)
    assert info['reg_max'] == 1 and info['stride'] == [8.0, 16.0, 32.0], info
    return Heads(net).eval(), info


def run(wrap, greys):
    x = torch.from_numpy(np.stack(greys)).float()[:, None].expand(-1, 3, -1, -1).contiguous()
    with torch.inference_mode(): bo, cl = wrap(x)
    bo, cl = bo.float().numpy(), cl.float().numpy()
    return [post(bo[j], cl[j]) for j in range(len(greys))]


def main():
    mode, w = sys.argv[1], sys.argv[2]; wrap, info = load(w); print('MODEL', json.dumps(info), flush=True)
    if mode == 'time':
        g = [np.full((480, 640), 90, np.uint8)] * BATCH; run(wrap, g); t0 = time.time(); run(wrap, g); run(wrap, g); print('s per frame %.3f' % ((time.time() - t0) / (2 * BATCH))); return
    if mode == 'imgs':
        out, lst = sys.argv[3], sys.argv[4]; files = [l.strip() for l in open(lst, encoding='utf-8') if l.strip()]; fi_, bx_, sc_ = [], [], []; t0 = time.time()
        for b0 in range(0, len(files), BATCH):
            gs = []
            for p in files[b0:b0 + BATCH]:
                g = cv2.imread(p, cv2.IMREAD_GRAYSCALE); assert g is not None and g.shape == (480, 640), (p, None if g is None else g.shape); gs.append(g)
            for j, (b, s) in enumerate(run(wrap, gs)): fi_.append(np.full(len(s), b0 + j, np.int32)); bx_.append(b); sc_.append(s)
            if (b0 // BATCH) % 25 == 0: print('IMG %d / %d  %.0f s' % (b0, len(files), time.time() - t0), flush=True)
        np.savez_compressed(out, fi=np.concatenate(fi_), box=np.concatenate(bx_), score=np.concatenate(sc_), names=np.array(files), n_frames=np.int32(len(files)))
        print('IMGS_DONE %d files %d boxes %.0f s' % (len(files), sum(map(len, sc_)), time.time() - t0), flush=True); return
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    for rec in [Path(s) for s in sys.argv[4:]]:
        meta = json.loads((rec / 'frames.json').read_text(encoding='utf-8')); ts = np.asarray(meta['times_s'], np.float64); n = int(meta['frames']); to8 = DD.To8(); buf = []; fi_, bx_, sc_ = [], [], []; t0 = time.time()

        def flush(buf):
            for (i, _), (b, s) in zip(buf, run(wrap, [g for _, g in buf])): fi_.append(np.full(len(s), i, np.int32)); bx_.append(b); sc_.append(s)
        for i, raw in DD.frames(rec):
            buf.append((i, to8(raw)))
            if len(buf) == BATCH: flush(buf); buf = []
            if i % 200 == 0: print('REC %s frame %d / %d  %.0f s' % (rec.name, i, n, time.time() - t0), flush=True)
        if buf: flush(buf)
        nb = sum(map(len, sc_))
        np.savez_compressed(out / ('yolo_%s.npz' % rec.name), fi=np.concatenate(fi_) if nb else np.zeros(0, np.int32), box=np.concatenate(bx_) if nb else np.zeros((0, 4), np.float32),
                            score=np.concatenate(sc_) if nb else np.zeros(0, np.float32), kpt=np.zeros((nb, 17, 3), np.float32), anchor=np.zeros(nb, np.int32), n_frames=np.int32(n),
                            names=np.array(['%06d' % i for i in range(n)]), times_s=ts)
        print('REC_DONE %s frames %d boxes %d (>= 0.25: %d) %.0f s' % (rec.name, n, nb, int(sum((s >= 0.25).sum() for s in sc_)), time.time() - t0), flush=True)
    print('YOLO_DONE', flush=True)


if __name__ == '__main__':
    main()
