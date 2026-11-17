"""v2/scripts/v2_device.py - box_refine v2 on DEVICE recordings (23-24 fps, no ground truth; PREREG_V2.json device_recordings).

Inputs: v2/data/det/{near,far}_<rec>.npz and v2/data/gmc/<rec>.npz from dev_det.py (our near / far pose detectors on the PC CPU, device 16 -> 8 bit
mapping, server GMC). Tracker = the server's Python ByteTrack exactly as scripts/track_run.py runs it (n_init 3, class gate, edge birth, GKF + GMC);
streams near, far and sw (near / far switched every 2-6 s, track_run.schedule). Methods = raw, tracker row, server euro, gru, combo_a, size (server SIZE),
all run through the server Refiner with the real frame times.
Reference height for every number = the tracker row's height (the same for all methods). Writes v2/results/device.json and device_tables.md.
Usage: python v2_device.py <rec name> [...]
"""
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import v2_lib as L
import track_run as TR   # scripts/track_run.py: run (server ByteTrack), schedule

D = L.V2 / 'data'
BINS = ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 99))
SBINS = ((0.005, 0.015), (0.015, 99))


def load(rec):
    d = {k: L.P.load_det(D / 'det' / ('%s_%s.npz' % (k, rec))) for k in L.DETS}; g = np.load(D / 'gmc' / ('%s.npz' % rec))
    return d, g['warp'].astype(np.float64), g['times_s'].astype(np.float64)


def streams(rec, idx):
    d, W, ts = load(rec); n = d['near']['n']; fps = (n - 1) / (ts[-1] - ts[0])
    per = {k: [np.flatnonzero(d[k]['fi'] == f) for f in range(n)] for k in L.DETS}
    mk = lambda k, f: (np.c_[d[k]['box'][per[k][f]], d[k]['score'][per[k][f]], np.zeros(len(per[k][f]))], per[k][f], np.full(len(per[k][f]), L.DETS.index(k)))
    src, sw = TR.schedule(n, fps, idx); out = {}
    for st in ('near', 'far', 'sw'):
        fd = [mk(st, f) if st != 'sw' else mk(L.DETS[src[f]], f) for f in range(n)]
        rows = np.array(TR.run(n, W, fd), np.float64).reshape(-1, 8); ids = defaultdict(list)
        for r in rows: ids[int(r[1])].append(r)
        E = {}
        for i, rr in ids.items():
            rr = np.array(rr); rr = rr[np.argsort(rr[:, 0])]; f = rr[:, 0].astype(np.int64); s = rr[:, 6].astype(np.int64); di = rr[:, 7].astype(np.int64)
            z = np.zeros((len(rr), 4)); kp = np.zeros((len(rr), 17, 3))
            for j in range(len(rr)):
                dd = d[L.DETS[s[j]]]; z[j] = L.P.xyxy2cxcywh(dd['box'][di[j]]); kp[j] = dd['kpt'][di[j]]
            E[i] = dict(f=f, z=z, kp=kp, trk=L.P.xyxy2cxcywh(rr[:, 2:6]), src=s)
        keys = [L.DETS[int(x)] for x in src] if st == 'sw' else [st] * n
        out[st] = dict(ids=E, by_f=L.frame_items(E), W=W, n=n, keys=keys, fps=fps, sw=np.array(sw if st == 'sw' else [], np.int64), times=ts)
    return out, dict(frames=n, fps=round(fps, 2), seconds=round(float(ts[-1] - ts[0]), 1), dets_near=int(len(d['near']['score'])), dets_far=int(len(d['far']['score'])), switches=len(sw))


def aim_of(b): return np.stack([b[:, 0], b[:, 1] - b[:, 3] / 2 + L.AIM * b[:, 3]], 1)


def per_id(e, o, kk, sw_frames, fps):
    """metric rows of one id (no GT): reference height = tracker row height."""
    f = e['f']; n = len(f); hr = np.maximum(e['trk'][:, 3], 1.); z = e['z']; a = aim_of(o); az = aim_of(z)
    c1 = np.r_[False, np.diff(f) == 1]; c2 = c1 & np.r_[False, c1[:-1]]; c3 = c2 & np.r_[False, c2[:-1]]
    dh = np.full(n, np.nan); da = np.full(n, np.nan); d2a = np.full(n, np.nan); i1 = np.flatnonzero(c1); i2 = np.flatnonzero(c2)
    dh[i1] = np.abs(o[i1, 3] - o[i1 - 1, 3]) / hr[i1]; da[i1] = np.hypot(*(a[i1] - a[i1 - 1]).T) / hr[i1]
    d2a[i2] = np.hypot(*(a[i2] - 2 * a[i2 - 1] + a[i2 - 2]).T) / hr[i2]
    spd = np.full(n, np.nan); lag = np.full(n, np.nan); alag = np.full(n, np.nan); sspd = np.full(n, np.nan); slag = np.full(n, np.nan)
    i3 = np.flatnonzero(c3)
    if len(i3):
        v = (z[i3, :2] - z[i3 - 3, :2]) / 3.; sp = np.hypot(*v.T); u = v / np.maximum(sp, 1e-9)[:, None]; spd[i3] = sp / hr[i3]
        lag[i3] = np.sum((o[i3, :2] - z[i3, :2]) * u, 1) / hr[i3]; alag[i3] = np.sum((a[i3] - az[i3]) * u, 1) / hr[i3]
        dl = (np.log(z[i3, 3]) - np.log(z[i3 - 3, 3])) / 3.; sspd[i3] = np.abs(dl); slag[i3] = np.sign(dl) * (np.log(np.maximum(o[i3, 3], 1e-3)) - np.log(z[i3, 3]))
    src = e['src']; sw0 = np.r_[False, (src[1:] != src[:-1]) & (np.diff(f) <= 3)]; sw1 = np.zeros(n, bool)
    for i in np.flatnonzero(sw0): sw1 |= (f >= f[i]) & (f <= f[i] + fps)
    rj = np.r_[False, c1[1:] & (np.abs(np.diff(np.log(z[:, 3]))) > math.log(1.3))]   # the raw detection height jumps by > 30 % in one frame (definition flip / bad box)
    near = np.convolve(rj.astype(float), np.ones(5), 'full')[2:2 + n] > 0
    return dict(dh=dh, da=da, d2a=d2a, spd=spd, lag=lag, alag=alag, sspd=sspd, slag=slag, start=(kk >= 1) & (kk <= 4), sw0=sw0, sw1=sw1, rawjump=near)


def summarize(parts):
    """parts: list of per-id dicts -> numbers of one method"""
    cat = lambda k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)
    dh, da, d2a = cat('dh'), cat('da'), cat('d2a'); fin = lambda v: v[np.isfinite(v)]
    q = lambda v, p: round(float(np.percentile(fin(v), p)), 5) if len(fin(v)) else None
    s = dict(n_rows=int(np.isfinite(dh).sum()), dh_p95=q(dh, 95), dh_p99=q(dh, 99), dh_max=q(dh, 100), da_p95=q(da, 95), da_p99=q(da, 99),
             aim_jitter=round(float(np.sqrt(np.mean(fin(d2a) ** 2))), 5) if len(fin(d2a)) else None)
    tm = [np.nanmax(p['dh']) for p in parts if np.isfinite(p['dh']).sum() >= 10]; s['dh_trackmax_med'] = round(float(np.median(tm)), 5) if tm else None; s['n_tracks'] = len(tm)
    spd, lag, alag, sspd, slag = cat('spd'), cat('lag'), cat('alag'), cat('sspd'), cat('slag')
    for a, b in BINS:
        mm = np.isfinite(spd) & (spd >= a) & (spd < b)
        s['lag_%g-%g' % (a, b)] = dict(n=int(mm.sum()), centre=round(float(np.mean(lag[mm])), 5) if mm.any() else None, aim=round(float(np.mean(alag[mm])), 5) if mm.any() else None)
    for a, b in SBINS:
        mm = np.isfinite(sspd) & (sspd >= a) & (sspd < b)
        s['slag_%g-%g' % (a, b)] = dict(n=int(mm.sum()), size=round(float(np.mean(slag[mm])), 5) if mm.any() else None)
    for ev in ('start', 'sw0', 'sw1', 'rawjump'):
        m = cat(ev).astype(bool); vh = dh[m & np.isfinite(dh)]; va = da[m & np.isfinite(da)]
        if len(vh) >= 5: s['ev_' + ev] = dict(n=int(len(vh)), dh_p95=q(vh, 95), dh_max=q(vh, 100), da_p95=q(va, 95), da_max=q(va, 100))
    return s


def main():
    t0 = time.time(); recs = sys.argv[1:]; model = L.gru_model(); res = dict(size=dict(L.BRM.SIZE), combo_a=L.COMBO_A, euro=dict(L.BRM.EURO), recs={}, pooled={})
    allp = defaultdict(lambda: defaultdict(list))
    for ri, rec in enumerate(recs):
        S, info = streams(rec, ri); res['recs'][rec] = dict(info=info, streams={})
        for st, inp in S.items():
            _, kk = L.run(inp, 'off'); outs = dict(euro=L.run(inp, 'euro')[0], gru=L.run(inp, 'gru', model)[0], combo_a=L.run(inp, 'gru', model, ca=L.COMBO_A)[0], size=L.run(inp, 'size', size=L.BRM.SIZE)[0])
            per = defaultdict(list)
            for tid, e in inp['ids'].items():
                for m in ('raw', 'tracker', 'euro', 'gru', 'combo_a', 'size'):
                    o = e['z'] if m == 'raw' else e['trk'] if m == 'tracker' else outs[m][tid]
                    p = per_id(e, o, kk[tid], inp['sw'], inp['fps']); per[m].append(p); allp[st][m].append(p)
            res['recs'][rec]['streams'][st] = dict(ids=len(inp['ids']), rows=int(sum(len(e['f']) for e in inp['ids'].values())), methods={m: summarize(v) for m, v in per.items()})
            print('REC %s %s ids %d rows %d (%.1f s)' % (rec, st, len(inp['ids']), res['recs'][rec]['streams'][st]['rows'], time.time() - t0), flush=True)
    for st, pm in allp.items():
        res['pooled'][st] = {m: summarize(v) for m, v in pm.items()}
        for m, s in res['pooled'][st].items():
            print('  %-3s %-8s dh95 %s dh99 %s tmax %s da99 %s jit %s lag>0.03 c %s a %s size lag %s sw1 dh95 %s' % (st, m, s['dh_p95'], s['dh_p99'], s['dh_trackmax_med'], s['da_p99'], s['aim_jitter'],
                  s['lag_0.03-0.1']['centre'], s['lag_0.03-0.1']['aim'], s['slag_0.015-99']['size'], s.get('ev_sw1', {}).get('dh_p95')), flush=True)
    (L.V2 / 'results' / 'device.json').write_text(json.dumps(res, indent=1))
    print('DEVICE_DONE %.1f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
