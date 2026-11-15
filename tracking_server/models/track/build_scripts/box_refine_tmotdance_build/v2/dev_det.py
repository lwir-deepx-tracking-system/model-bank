"""v2/scripts/dev_det.py (PC CPU): our near and far pose detectors and the server GMC on DEVICE recordings (raw16 measurement recordings, 23-24 fps).

Per recording (a MEASURE_* folder with frames.raw.gz + frames.json):
  - every frame of frames.raw.gz, 16 -> 8 bit exactly like the device default (camera_link.lv_to8, --live-map pct: smoothed 1..99 percentile window,
    state from the first frame of the recording);
  - camera motion = server GMC with the device options (--gmc --gmc-corners 300 --gmc-mindist 5) on the 640x512 canvas (gmc_run.GMC / to_canvas);
  - detections = scripts/det_run.py Heads + post (one-to-many branch, post.cpp decode + NMS, score > 0.10, drop_giant), fp32 on the CPU, for the
    near detector (expE_v1_best.pt) and the far detector (C_control_v1_epoch2.pt) -- the same two weight files as the first round (sha256 checked).
Writes <out>/det/{near,far}_<rec>.npz (det_run format + times_s), <out>/gmc/<rec>.npz (warp, valid, times_s), <out>/det/<rec>_DONE.json.
Usage: python dev_det.py <out_dir> <near.pt> <far.pt> <MEASURE dir> [...]
"""
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'scripts'))
import det_run as D     # noqa: E402  (Heads, post, the same decode / NMS as the first round)
import gmc_run as G     # noqa: E402  (GMC, to_canvas)

W_, H_ = 640, 480
BATCH = 4
WANT_SHA = dict(near='6d79d275d357bfc09c690a2c4bec22f920d502ae8c9a5a71f7d55fd5944bc618', far='78525f5e987b5a5a429f19dbfd30e8ce610047148794488dc77ecbc7fb6d4ebb')


class To8:
    """= tracking_server/src/camera/camera_link.lv_to8 with --live-map pct (the device default)."""
    def __init__(self): self.lo = None; self.hi = None

    def __call__(self, raw):
        hist = np.bincount((raw >> 4).ravel(), minlength=4096).cumsum(); tot = hist[-1]
        lo = float(np.searchsorted(hist, tot * 0.01)) * 16; hi = float(np.searchsorted(hist, tot * 0.99)) * 16 + 16
        if hi - lo < 200: hi = lo + 200
        if self.lo is None: self.lo, self.hi = lo, hi
        else: self.lo += 0.1 * (lo - self.lo); self.hi += 0.1 * (hi - self.hi)
        return np.clip((raw.astype(np.float32) - self.lo) * (255.0 / (self.hi - self.lo)), 0, 255).astype(np.uint8)


def frames(rec):
    meta = json.loads((rec / 'frames.json').read_text(encoding='utf-8')); n = int(meta['frames']); nb = W_ * H_ * 2
    assert meta['width'] == W_ and meta['height'] == H_ and 'uint16' in meta['dtype']
    with gzip.open(rec / 'frames.raw.gz', 'rb') as f:
        for i in range(n):
            b = f.read(nb); assert len(b) == nb, (rec, i)
            yield i, np.frombuffer(b, '<u2').reshape(H_, W_)
    return


def main():
    out = Path(sys.argv[1]); wn, wf = sys.argv[2], sys.argv[3]; recs = [Path(s) for s in sys.argv[4:]]
    (out / 'det').mkdir(parents=True, exist_ok=True); (out / 'gmc').mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('DET_THREADS', '4')))
    from ultralytics import YOLO
    nets = {}
    for k, w in (('near', wn), ('far', wf)):
        s = D.sha(w); assert s == WANT_SHA[k], (k, s)
        net = YOLO(w).model.float().eval(); nets[k] = D.Heads(net).eval()
    for rec in recs:
        meta = json.loads((rec / 'frames.json').read_text(encoding='utf-8')); ts = np.asarray(meta['times_s'], np.float64); n = int(meta['frames'])
        to8 = To8(); gmc = G.GMC(2); gmc.fp = dict(gmc.fp, maxCorners=300, minDistance=5)
        Wp = np.zeros((n, 2, 3)); V = np.zeros(n, bool); acc = {k: dict(fi=[], box=[], score=[], kpt=[], anchor=[]) for k in nets}; buf = []; t0 = time.time()

        def flush(buf):
            x = torch.from_numpy(np.stack([g for _, g in buf])).float()[:, None].expand(-1, 3, -1, -1).contiguous()
            for k, wrap in nets.items():
                with torch.inference_mode():
                    bo, cl, kq = wrap(x)
                bo, cl, kq = bo.float().numpy(), cl.float().numpy(), kq.float().numpy()
                for j, (i, _) in enumerate(buf):
                    b, s, kp, a = D.post(bo[j], cl[j], kq[j]); A = acc[k]
                    A['fi'].append(np.full(len(s), i, np.int32)); A['box'].append(b); A['score'].append(s); A['kpt'].append(kp); A['anchor'].append(a)

        for i, raw in frames(rec):
            g = to8(raw); Wp[i], V[i] = gmc.estimate(G.to_canvas(g)); buf.append((i, g))
            if len(buf) == BATCH: flush(buf); buf = []
            if i % 200 == 0: print('REC %s frame %d / %d  %.1f s' % (rec.name, i, n, time.time() - t0), flush=True)
        if buf: flush(buf)
        cat = lambda v, sh, dt: np.concatenate(v).astype(dt) if v else np.zeros(sh, dt)
        info = dict(rec=rec.name, frames=n, seconds_video=float(ts[-1] - ts[0]), fps=float((n - 1) / (ts[-1] - ts[0])), to8='camera_link.lv_to8 pct',
                    gmc='server GMC corners 300 mindist 5 downscale 2', gmc_valid=int(V.sum()), weights=dict(near=wn, far=wf), sha256=WANT_SHA, precision='fp32 cpu', branch='one2many',
                    score_keep=D.SCORE_KEEP, sec=round(time.time() - t0, 1), torch=torch.__version__)
        for k, A in acc.items():
            res = dict(fi=cat(A['fi'], (0,), np.int32), box=cat(A['box'], (0, 4), np.float32), score=cat(A['score'], (0,), np.float32), kpt=cat(A['kpt'], (0, 17, 3), np.float32),
                       anchor=cat(A['anchor'], (0,), np.int32), n_frames=np.int32(n), names=np.array(['%06d' % i for i in range(n)]), times_s=ts)
            np.savez_compressed(out / 'det' / ('%s_%s.npz' % (k, rec.name)), **res)
            info['dets_' + k] = int(len(res['score'])); info['dets_ge_025_' + k] = int((res['score'] >= 0.25).sum())
        np.savez_compressed(out / 'gmc' / ('%s.npz' % rec.name), warp=Wp, valid=V, times_s=ts)
        (out / 'det' / ('%s_DONE.json' % rec.name)).write_text(json.dumps(info, indent=1))
        print('REC_DONE', json.dumps(info), flush=True)
    print('DEV_DET_DONE', flush=True)


if __name__ == '__main__':
    main()
