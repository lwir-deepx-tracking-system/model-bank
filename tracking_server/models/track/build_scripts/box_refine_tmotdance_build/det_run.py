"""det_run.py (GPU server, run with the close_range venv: python -I det_run.py ...): our pose detectors on grey 640x480 frames, decoded and NMS'd exactly
like the tracking server (post.cpp decode_raw + nms_decoded, detect_frame.attach_kpts), keeping the person boxes the tracker can use (score > 0.10).

Head = the ONE-TO-MANY branch (cv2 / cv3 / cv4 / cv4_kpts), the branch the device DXNN files export (build_bodybox_kpt.py SixHeads).
NMS exactness: greedy NMS in score order -> whether a box above 0.10 is kept depends only on boxes with higher scores, so filtering the candidates
to score > 0.10 before the NMS gives exactly the kept set above 0.10 of the server's NMS (candidates > 0.001, at most 300 kept).

Usage: python -I det_run.py <weights.pt> <tag> <out_dir> <seq_dir> [<seq_dir> ...]
  seq_dir holds img1/ (MOT layout) or images/ (our TMOT val copy). Writes <out_dir>/<tag>_<seq>.npz and <out_dir>/<tag>_DONE.json.
"""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

SCORE_KEEP = 0.10      # ByteTrack low threshold (bytetrack.py self.low): boxes at or below it are never used
NMS_IOU = np.float32(0.7)
MAX_H = 1440.0         # detect_frame.drop_giant
BATCH = int(__import__('os').environ.get('DET_BATCH', '4'))
H, W = 480, 640


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


class Heads(torch.nn.Module):
    """build_bodybox_kpt.SixHeads: backbone + neck, then the one-to-many box / person / keypoint convolutions (raw, before any decode)."""
    def __init__(self, net):
        super().__init__(); self.net = net; self.head = net.model[-1]

    def feats(self, images):
        x = images / 255.0; saved = []
        for m in self.net.model[:-1]:
            if m.f != -1: x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x); saved.append(x if m.i in self.net.save else None)
        return [saved[j] for j in self.head.f]

    def forward(self, images):
        f = self.feats(images); h = self.head
        box = torch.cat([h.cv2[i](f[i]).flatten(2) for i in range(3)], 2)              # [B, 4, N] ltrb distances in stride units (reg_max 1)
        cls = torch.cat([h.cv3[i](f[i]).flatten(2) for i in range(3)], 2)              # [B, 1, N] person logit
        kp = torch.cat([h.cv4_kpts[i](h.cv4[i](f[i])).flatten(2) for i in range(3)], 2)  # [B, 51, N] raw keypoints (x, y, v) x 17
        return box, cls, kp


def grid(h, w):
    gx, gy, gs = [], [], []
    for s in (8, 16, 32):
        hh, ww = h // s, w // s; j = np.arange(hh * ww); gx.append(j % ww + .5); gy.append(j // ww + .5); gs.append(np.full(hh * ww, s))
    return (np.concatenate(gx).astype(np.float32), np.concatenate(gy).astype(np.float32), np.concatenate(gs).astype(np.float32))


GX, GY, GS = grid(H, W)


def nms_server(b, s):
    """post.cpp nms_decoded on one class: stable sort by score (descending), greedy, a later box is suppressed when IoU > 0.7 (float32). -> kept indices."""
    order = np.argsort(-s, kind='stable'); b = b[order]; area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); sup = np.zeros(len(order), bool); keep = []
    for a in range(len(order)):
        if sup[a]: continue
        keep.append(a)
        if len(keep) >= 300: break
        r = b[a + 1:]
        iw = np.maximum(np.float32(0), np.minimum(b[a, 2], r[:, 2]) - np.maximum(b[a, 0], r[:, 0]))
        ih = np.maximum(np.float32(0), np.minimum(b[a, 3], r[:, 3]) - np.maximum(b[a, 1], r[:, 1]))
        inter = iw * ih; ratio = inter / (area[a] + area[a + 1:] - inter)
        sup[a + 1:] |= ratio > NMS_IOU
    return order[np.array(keep, int)]


def post(box, cls, kp):
    """one picture: raw [4,N], [1,N], [51,N] (numpy float32) -> kept boxes x1,y1,x2,y2 (picture px), scores, keypoints [n,17,3], anchors."""
    score = (1.0 / (1.0 + np.exp(-cls[0].astype(np.float64)))).astype(np.float32)
    cand = np.flatnonzero(score > SCORE_KEEP)
    if len(cand) == 0: return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 17, 3), np.float32), np.zeros(0, np.int32)
    ax, ay, st = GX[cand], GY[cand], GS[cand]; bb = box[:, cand]
    l = ax - bb[0]; t = ay - bb[1]; r = ax + bb[2]; btm = ay + bb[3]                               # decode_raw (float32)
    x = ((l + r) * np.float32(.5)) * st; y = ((t + btm) * np.float32(.5)) * st; w = (r - l) * st; h = (btm - t) * st
    xyxy = np.stack([x - w * np.float32(.5), y - h * np.float32(.5), x + w * np.float32(.5), y + h * np.float32(.5)], 1).astype(np.float32)   # nms_decoded rows
    k = nms_server(xyxy, score[cand]); a = cand[k]
    kr = kp[:, a].reshape(17, 3, -1).transpose(2, 0, 1)
    out = np.empty_like(kr); out[:, :, 0] = (kr[:, :, 0] + GX[a, None]) * GS[a, None]; out[:, :, 1] = (kr[:, :, 1] + GY[a, None]) * GS[a, None]
    out[:, :, 2] = 1. / (1. + np.exp(-kr[:, :, 2]))
    keep = (xyxy[k, 3] - xyxy[k, 1]) <= MAX_H                                                       # drop_giant
    return xyxy[k][keep], score[a][keep], out[keep].astype(np.float32), a[keep].astype(np.int32)


def frames_of(sd):
    d = sd / 'img1' if (sd / 'img1').is_dir() else sd / 'images'
    fs = sorted(p for p in d.iterdir() if p.suffix.lower() in ('.jpg', '.png'))
    return fs


def read(p):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    assert g is not None and g.shape == (H, W), (p, None if g is None else g.shape)
    return g


def main():
    wpt, tag, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); seqs = [Path(s) for s in sys.argv[4:]]
    out.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    import ultralytics
    dev = 'cuda'; torch.backends.cudnn.benchmark = False; torch.cuda.set_per_process_memory_fraction(0.15)
    net = YOLO(wpt).model.float().eval().to(dev); wrap = Heads(net).eval()
    hd = net.model[-1]; info = dict(weights=wpt, sha256=sha(wpt), ultralytics=ultralytics.__version__, torch=torch.__version__, head=type(hd).__name__,
                                     nc=int(hd.nc), kpt_shape=list(hd.kpt_shape), stride=[float(s) for s in hd.stride], batch=BATCH, precision='fp32', branch='one2many',
                                     score_keep=SCORE_KEEP, nms_iou=float(NMS_IOU), max_h=MAX_H)
    assert info['nc'] == 1 and info['stride'] == [8.0, 16.0, 32.0] and info['kpt_shape'] == [17, 3], info
    print('MODEL', json.dumps(info), flush=True)
    pool = ThreadPoolExecutor(6); done = {}
    for sd in seqs:
        fs = frames_of(sd); t0 = time.time(); fi_, bx_, sc_, kp_, an_ = [], [], [], [], []
        futs = [pool.submit(read, p) for p in fs[:BATCH]]
        for b0 in range(0, len(fs), BATCH):
            ims = [f.result() for f in futs]
            nxt = fs[b0 + BATCH:b0 + 2 * BATCH]; futs = [pool.submit(read, p) for p in nxt]
            x = torch.from_numpy(np.stack(ims)).to(dev).float()[:, None].expand(-1, 3, -1, -1).contiguous()
            with torch.inference_mode():
                bo, cl, kq = wrap(x)
            bo, cl, kq = bo.float().cpu().numpy(), cl.float().cpu().numpy(), kq.float().cpu().numpy()
            for j in range(len(ims)):
                b, s, k, a = post(bo[j], cl[j], kq[j]); fi_.append(np.full(len(s), b0 + j, np.int32)); bx_.append(b); sc_.append(s); kp_.append(k); an_.append(a)
        cat = lambda v, sh, dt: np.concatenate(v).astype(dt) if v else np.zeros(sh, dt)
        res = dict(fi=cat(fi_, (0,), np.int32), box=cat(bx_, (0, 4), np.float32), score=cat(sc_, (0,), np.float32), kpt=cat(kp_, (0, 17, 3), np.float32),
                   anchor=cat(an_, (0,), np.int32), n_frames=np.int32(len(fs)), names=np.array([p.name for p in fs]))
        np.savez_compressed(out / ('%s_%s.npz' % (tag, sd.name)), **res)
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        done[sd.name] = dict(frames=len(fs), dets=int(len(res['score'])), dets_ge_025=int((res['score'] >= 0.25).sum()), sec=round(time.time() - t0, 1), max_gpu_gb=round(mem, 3))
        print('SEQ', tag, sd.name, json.dumps(done[sd.name]), flush=True)
    (out / ('%s_DONE.json' % tag)).write_text(json.dumps(dict(info, seqs=done), indent=1))
    print('DET_DONE', tag, flush=True)


if __name__ == '__main__':
    main()
