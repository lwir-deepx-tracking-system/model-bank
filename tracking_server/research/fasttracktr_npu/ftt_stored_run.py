"""Device: run the FastTrackTr NPU variant over a recording and write a stored replay result (<key>.mot.txt + <key>.json),
so the replay / tracker tree can pick it like any stored result.  Replace the .dxnn files later (trained weights) and run
again: the stored result is rewritten.

Usage: python3 ftt_stored_run.py --ae AE.dxnn --chunks B0.dxnn [B1.dxnn ...] --frames <recording>/images
                                 --out <replay tracks root>/<recording>/<key> [--label NAME] [--score 0.5] [--max-det 30]
                                 [--ae-core 1] [--b-core 2] [--untrained]
Per frame (8-bit PNG, gray -> 3 channels, uint8 NHWC): AE (backbone + encoder + encoder head) -> decoder chunks in order
(the previous frame's decoder output goes in as traj; zeros on the first frame) -> 300 queries: score = sigmoid(logit),
box = sigmoid output (cx, cy, w, h in 0..1 of 640x480). Kept: score >= --score, NMS IoU 0.7, at most --max-det.
IDs: a PLACEHOLDER greedy IoU linker (IoU >= 0.3 to a live track, else a new ID; a track ends after 30 missed frames).
FastTrackTr's own association (id_words + Hybrid-SORT-style tracker) is to replace it once there are trained weights."""
import argparse
import hashlib
import json
import os
import time

import dx_engine as dx
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--ae', required=True); ap.add_argument('--chunks', nargs='+', required=True)
ap.add_argument('--frames', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--label', default='FastTrackTr (장치 NPU)'); ap.add_argument('--score', type=float, default=0.5)
ap.add_argument('--max-det', type=int, default=30); ap.add_argument('--ae-core', type=int, default=1); ap.add_argument('--b-core', type=int, default=2)
ap.add_argument('--untrained', action='store_true', help='the .dxnn were compiled from random weights (speed test files)')
a = ap.parse_args()
BO = dx.InferenceOption.BOUND_OPTION


def engine(p, core):
    o = dx.InferenceOption(); o.devices = [0]; o.bound_option = getattr(BO, 'NPU_%d' % core); o.use_ort = True
    return dx.InferenceEngine(p, o)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def iou_m(A, B):   # A [n,4], B [m,4] x1 y1 x2 y2
    if not len(A) or not len(B):
        return np.zeros((len(A), len(B)))
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = ix * iy
    ar = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return inter / (ar(A)[:, None] + ar(B)[None, :] - inter + 1e-9)


ea = engine(a.ae, a.ae_core)
ebs = [engine(p, a.b_core) for p in a.chunks]
ae_out = [t['name'] for t in ea.get_output_tensors_info()]
pngs = sorted((int(os.path.splitext(n)[0]), n) for n in os.listdir(a.frames) if n.endswith('.png') and os.path.splitext(n)[0].isdigit())
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
mot_path, meta_path = a.out + '.mot.txt', a.out + '.json'
traj = np.zeros((1, 300, 256), np.float32)
tracks, next_id, rows, times = {}, 1, [], []
W = H = None
for num, name in pngs:
    im = np.asarray(Image.open(os.path.join(a.frames, name)).convert('L'), np.uint8)
    H, W = im.shape
    t0 = time.perf_counter()
    ao = dict(zip(ae_out, [np.asarray(v, np.float32) for v in ea.run([np.ascontiguousarray(np.repeat(im[None, :, :, None], 3, axis=3))])]))
    st = dict(kv=ao['kv4'].reshape(1, -1, 256), tgt=ao['tgt4'].reshape(1, -1, 256), ref_unact=ao['ref4'].reshape(1, -1, 4), traj=traj)
    for e in ebs:
        out = e.run([np.ascontiguousarray(st[t['name']].reshape(t['shape']).astype(t['dtype'])) for t in e.get_input_tensors_info()])
        res = dict(zip([t['name'] for t in e.get_output_tensors_info()], [np.asarray(v, np.float32) for v in out]))
        if 'tgt_out' in res:
            st['tgt'], st['ref_unact'] = res['tgt_out'], res['ref_out']
    traj = res['hs'].reshape(1, 300, 256)
    times.append((time.perf_counter() - t0) * 1e3)
    sc = 1 / (1 + np.exp(-res['logits'].reshape(-1)))
    bx = res['boxes'].reshape(-1, 4)
    xyxy = np.c_[(bx[:, 0] - bx[:, 2] / 2) * W, (bx[:, 1] - bx[:, 3] / 2) * H, (bx[:, 0] + bx[:, 2] / 2) * W, (bx[:, 1] + bx[:, 3] / 2) * H]
    xyxy = np.clip(xyxy, 0, [W, H, W, H])
    keep = np.flatnonzero((sc >= a.score) & (xyxy[:, 2] - xyxy[:, 0] > 2) & (xyxy[:, 3] - xyxy[:, 1] > 2))
    keep = keep[np.argsort(-sc[keep])]
    sel = []
    for k in keep:   # NMS
        if all(iou_m(xyxy[k:k + 1], xyxy[j:j + 1])[0, 0] < 0.7 for j in sel):
            sel.append(k)
        if len(sel) >= a.max_det:
            break
    dets = xyxy[sel]
    live = [tid for tid, t in tracks.items() if num - t['last'] <= 30]
    M = iou_m(dets, np.array([tracks[t]['box'] for t in live]) if live else np.zeros((0, 4)))
    used_d, used_t = set(), set()
    for di, ti in sorted(((i, j) for i in range(M.shape[0]) for j in range(M.shape[1]) if M[i, j] >= 0.3), key=lambda p: -M[p]):
        if di in used_d or ti in used_t:
            continue
        used_d.add(di); used_t.add(ti); tracks[live[ti]].update(box=dets[di], last=num)
        x1, y1, x2, y2 = dets[di]; rows.append('%d,%d,%.3f,%.3f,%.3f,%.3f,%.5f,-1,-1,-1' % (num, live[ti], x1, y1, x2 - x1, y2 - y1, sc[sel[di]]))
    for di in range(len(dets)):
        if di not in used_d:
            tracks[next_id] = dict(box=dets[di], last=num)
            x1, y1, x2, y2 = dets[di]; rows.append('%d,%d,%.3f,%.3f,%.3f,%.3f,%.5f,-1,-1,-1' % (num, next_id, x1, y1, x2 - x1, y2 - y1, sc[sel[di]]))
            next_id += 1
with open(mot_path + '.tmp', 'w') as f:
    f.write('\n'.join(rows) + ('\n' if rows else ''))
os.replace(mot_path + '.tmp', mot_path)
w_ = sorted(times[1:]) or times
meta = dict(what='FastTrackTr NPU variant (research/fasttracktr_npu): AE + %d decoder chunks on the DX-M1; IDs by a placeholder IoU linker' % len(ebs),
            weights='UNTRAINED (random weights, speed-test files): boxes and IDs mean nothing until trained weights replace them' if a.untrained else 'see the .dxnn files',
            label=a.label, files={os.path.basename(p): sha(p) for p in [a.ae] + a.chunks}, cores=dict(ae=a.ae_core, decoder=a.b_core),
            settings=dict(score=a.score, nms_iou=0.7, max_det=a.max_det, link_iou=0.3, max_missed=30, input='8-bit PNG gray -> 3 channels uint8 NHWC'),
            frames_dir=os.path.abspath(a.frames), source_wh=[W, H], sequence_total=len(pngs), png_first=pngs[0][0], png_last=pngs[-1][0],
            status='complete', error=None, frames_processed=len(pngs), first_processed=pngs[0][0], last_processed=pngs[-1][0], covers_whole_sequence=True,
            rows=len(rows), tracks=next_id - 1, mot_file=os.path.basename(mot_path), mot_sha256=sha(mot_path),
            timing_ms=dict(npu_per_frame_mean=round(float(np.mean(w_)), 1), npu_per_frame_p95=round(float(w_[int(0.95 * (len(w_) - 1))]), 1)),
            written=time.strftime('%Y-%m-%d %H:%M:%S'))
with open(meta_path + '.tmp', 'w') as f:
    json.dump(meta, f, indent=1, ensure_ascii=False)
os.replace(meta_path + '.tmp', meta_path)
print(json.dumps(dict(out=a.out, frames=len(pngs), rows=len(rows), tracks=next_id - 1, ms=meta['timing_ms']), ensure_ascii=False))
