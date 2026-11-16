"""track_run.py (PC): the tracking server's own Python ByteTrack (worktree code, imported) on our detections, as the device runs it:
mode 'bytetrack', n_init 3 (--n-init 2 with --confirm-ms 130 gives 3 frames at 5-30 fps), class gate, edge birth 0.5 (EDGE_PX 4), GKF + GMC warp
(also on unconfirmed tracks, as app.py), position memory Kalman. Streams: near, far, and 'sw' = near/far spliced at the sequence level
(switch every 2-6 s, seed 1, first detector alternating over the sequences; PREREG amendment 1).
Writes data/tracks/<name>.npz: for each stream rows f, id, box (xyxy canvas), src (0 near / 1 far), det (index into that detector's npz rows).
Usage: python track_run.py <pos_refine root>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prcommon as P

sys.path.insert(0, P.PATHS['worktree'])
from src.common import settings                      # noqa: E402
from src.track.bytetrack import ByteTrackTracker      # noqa: E402
from src.track.kalman import GKF, gmc_warp            # noqa: E402

DETS = ('near', 'far')


def schedule(n, fps, idx):
    rng = np.random.default_rng(1 + idx); cur = idx % 2; src = np.zeros(n, np.int64); t = 0; sw = []
    while t < n:
        seg = int(round(rng.uniform(2.0, 6.0) * fps)); src[t:t + seg] = cur
        t += seg
        if t < n: sw.append(t)
        cur = 1 - cur
    return src, sw


def run(n, W, frames_dets):
    settings.EDGE_BIRTH = 0.5; settings.EDGE_PX = 4.0
    tr = ByteTrackTracker('bytetrack', n_init=3, class_gate=True); tr.kf = GKF(); out = []
    for f in range(n):
        Wf = W[f]
        tr.kf.warp = Wf
        for t_ in tr.tracked:
            if not t_.is_activated and t_.mean is not None:
                m_, c_ = gmc_warp(t_.mean[None].copy(), t_.cov[None], Wf); t_.mean, t_.cov = m_[0], c_[0]
        arr, idx, srcv = frames_dets[f]
        rows = tr.update(arr.astype(np.float32))
        for r in rows:
            j = int(r[7]); out.append((f, int(r[4]), float(r[0]), float(r[1]), float(r[2]), float(r[3]), int(srcv[j]), int(idx[j])))
    return out


def main():
    root = Path(sys.argv[1]); (root / 'data' / 'tracks').mkdir(parents=True, exist_ok=True); reg = P.registry(); info = {}
    for i, r in enumerate(reg):
        W, _ = P.load_gmc(root / 'data' / 'gmc' / ('%s.npz' % r['folder']))
        d = {k: P.load_det(root / 'data' / 'det' / ('%s_%s.npz' % (k, r['folder']))) for k in DETS}; n = d['near']['n']
        per = {k: [np.flatnonzero(d[k]['fi'] == f) for f in range(n)] for k in DETS}
        mk = lambda k, f: (np.c_[d[k]['box'][per[k][f]], d[k]['score'][per[k][f]], np.zeros(len(per[k][f]))], per[k][f], np.full(len(per[k][f]), DETS.index(k)))
        src, sw = schedule(n, r['fps'], i); out = {}
        for stream in ('near', 'far', 'sw'):
            fd = [mk(stream, f) if stream != 'sw' else mk(DETS[src[f]], f) for f in range(n)]
            rows = np.array(run(n, W, fd), np.float64).reshape(-1, 8)
            out[stream] = rows
        np.savez_compressed(root / 'data' / 'tracks' / ('%s.npz' % r['name']), near=out['near'], far=out['far'], sw=out['sw'], sw_src=src, sw_frames=np.array(sw, np.int64))
        info[r['name']] = {k: dict(rows=int(len(v)), ids=int(len(np.unique(v[:, 1]))) if len(v) else 0) for k, v in out.items()}
        info[r['name']]['switches'] = len(sw)
        print('TRACK', r['name'], json.dumps(info[r['name']]), flush=True)
    (root / 'results' / 'track_run.json').write_text(json.dumps(info, indent=1)); print('TRACK_DONE', flush=True)


if __name__ == '__main__':
    main()
