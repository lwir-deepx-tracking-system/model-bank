"""Device: run a FastTrackTr file set over a recording with EXACTLY the live server code (src/track/e2e_live.FTTLive, incl. its
number linking) and write a stored replay result (<key>.mot.txt + <key>.json), so two file sets can be scored by the server's
replay on the same recordings without touching the running server's models (2026-11-15: 6-epoch vs thermal fine-tune).

Usage (from tracking_server/): python3 research/fasttracktr_npu/ftt_stored_run_live.py --dir <folder with ae/b0/b1 .dxnn> --prefix <file prefix>
         --frames <recording>/images --out <replay tracks root>/<recording>/<key> [--label NAME] [--link id|iou]
Pictures: 8-bit PNG as they are; 16-bit PNG -> per-frame min-max to 8 bit (the labelled recordings used here are 8-bit).
Rows: e2e_live gives 640x512 canvas boxes (y + 16); written back in 640x480 picture pixels."""
import argparse, hashlib, json, os, sys, time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import dx_engine as dx                      # noqa: E402
from src.track import e2e_live as E         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True); ap.add_argument('--prefix', required=True)
ap.add_argument('--frames', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--label', default='FastTrackTr'); ap.add_argument('--link', default='id', choices=('id', 'iou'))
a = ap.parse_args()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''): h.update(c)
    return h.hexdigest()


E._DIR[0] = os.path.abspath(a.dir); E.E2E['link'] = a.link
d = dict(files={k: '%s_%s.dxnn' % (a.prefix, k) for k in ('ae', 'b0', 'b1')})
eng = E.FTTLive(dx, d)
pngs = sorted((int(os.path.splitext(n)[0]), n) for n in os.listdir(a.frames) if n.endswith('.png') and os.path.splitext(n)[0].isdigit())
rows, times, bits = [], [], set()
for num, name in pngs:
    im = Image.open(os.path.join(a.frames, name)); g = np.asarray(im)
    if g.ndim == 3: g = g[..., 0]
    if g.dtype != np.uint8:
        bits.add(16); g = g.astype(np.float32); lo, hi = float(g.min()), float(g.max()); g = ((g - lo) * (255.0 / max(hi - lo, 1.0))).astype(np.uint8)
    else: bits.add(8)
    assert g.shape == (480, 640), (name, g.shape)
    t0 = time.perf_counter(); out = eng.step(np.ascontiguousarray(g)); times.append((time.perf_counter() - t0) * 1e3)
    for x1, y1, x2, y2, tid, sc, _, _ in out:
        rows.append('%d,%d,%.3f,%.3f,%.3f,%.3f,%.5f,-1,-1,-1' % (num, tid, x1, y1 - 16.0, x2 - x1, y2 - y1, sc))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
mot = a.out + '.mot.txt'
with open(mot + '.tmp', 'w') as f: f.write('\n'.join(rows) + ('\n' if rows else ''))
os.replace(mot + '.tmp', mot)
w_ = sorted(times[1:]) or times
meta = dict(what='FastTrackTr NPU variant run with the live server code (src/track/e2e_live.FTTLive), number linking %r' % a.link,
            label=a.label, files={f: sha(os.path.join(a.dir, f)) for f in d['files'].values()}, settings=dict(link=a.link, score=eng.score, max_det=eng.max_det),
            frames_dir=os.path.abspath(a.frames), source_wh=[640, 480], input_bits=sorted(bits), sequence_total=len(pngs), png_first=pngs[0][0], png_last=pngs[-1][0],
            status='complete', error=None, frames_processed=len(pngs), first_processed=pngs[0][0], last_processed=pngs[-1][0], covers_whole_sequence=True,
            rows=len(rows), tracks=eng.next_id - 1, mot_file=os.path.basename(mot), mot_sha256=sha(mot),
            timing_ms=dict(step_mean=round(float(np.mean(w_)), 1), step_p95=round(float(w_[int(0.95 * (len(w_) - 1))]), 1)), written=time.strftime('%Y-%m-%d %H:%M:%S'))
with open(a.out + '.json.tmp', 'w') as f: json.dump(meta, f, indent=1, ensure_ascii=False)
os.replace(a.out + '.json.tmp', a.out + '.json')
print(json.dumps(dict(out=a.out, frames=len(pngs), rows=len(rows), tracks=eng.next_id - 1, ms=meta['timing_ms'], bits=sorted(bits)), ensure_ascii=False))
