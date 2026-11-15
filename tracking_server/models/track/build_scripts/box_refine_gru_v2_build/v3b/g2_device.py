"""gru_v3/scripts/g2_device.py - GRU refiner v2 on DEVICE-rate recordings (23-24 fps, no ground truth; PREREG_GRU_V2.json device_rate).
Inputs (read only): ../v2/data/det/{near,far}_<rec>.npz and ../v2/data/gmc/<rec>.npz made by the size-mode workflow (our two pose detectors on the PC CPU with the
device 16 -> 8 bit mapping, server GMC, real frame times). Tracker = the server's Python ByteTrack as scripts/track_run.py runs it (imported from the worktree, not written).
Streams near, far, sw. Methods: raw, tracker, euro, conv_euro, conv_euro_t, gru_v1, gru_v3, gru_v3_size. Reference height of every number = the tracker row's height.
Lag = (output centre - current detection centre in the method's own box definition) along the detection's own motion over the last 3 frames.
Writes results/device.json, results/tables_device.md.   Usage: python g2_device.py <rec> [...]
"""
import json
import sys
import time
from collections import defaultdict

import numpy as np

import g2_lib as L
from g2_lib import P, R

sys.path.insert(0, str(L.ROOT / 'scripts'))
import track_run as TR   # noqa: E402  (server ByteTrack; reads the worktree)

D = L.ROOT / 'v2' / 'data'
BINS = ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 99))
METHODS = ('raw', 'tracker', 'euro', 'conv_euro', 'conv_euro_t', 'gru_v1', 'gru_v2', 'gru_v3', 'gru_v3_size')


def streams(rec, idx):
    d = {k: P.load_det(D / 'det' / ('%s_%s.npz' % (k, rec))) for k in L.DETS}; g = np.load(D / 'gmc' / ('%s.npz' % rec)); W = g['warp'].astype(np.float64); ts = g['times_s'].astype(np.float64)
    n = d['near']['n']; fps = (n - 1) / (ts[-1] - ts[0]); per = {k: [np.flatnonzero(d[k]['fi'] == f) for f in range(n)] for k in L.DETS}
    mk = lambda k, f: (np.c_[d[k]['box'][per[k][f]], d[k]['score'][per[k][f]], np.zeros(len(per[k][f]))], per[k][f], np.full(len(per[k][f]), L.DETS.index(k)))
    src, sw = TR.schedule(n, fps, idx); out = {}
    for st in ('near', 'far', 'sw'):
        fd = [mk(st, f) if st != 'sw' else mk(L.DETS[src[f]], f) for f in range(n)]; rows = np.array(TR.run(n, W, fd), np.float64).reshape(-1, 8); ids = defaultdict(list)
        for r in rows: ids[int(r[1])].append(r)
        E = {}
        for i, rr in ids.items():
            rr = np.array(rr); rr = rr[np.argsort(rr[:, 0])]; f = rr[:, 0].astype(np.int64); s = rr[:, 6].astype(np.int64); di = rr[:, 7].astype(np.int64); z = np.zeros((len(rr), 4)); kp = np.zeros((len(rr), 17, 3))
            for j in range(len(rr)):
                dd = d[L.DETS[s[j]]]; z[j] = P.xyxy2cxcywh(dd['box'][di[j]]); kp[j] = dd['kpt'][di[j]]
            E[i] = dict(f=f, z=z, kp=kp, trk=P.xyxy2cxcywh(rr[:, 2:6]), src=s)
        out[st] = dict(ids=E, W=W, n=n, fps=fps, times=ts)
    return out, dict(frames=n, fps=round(fps, 2), seconds=round(float(ts[-1] - ts[0]), 1), switches=len(sw))


def per_id(e, o, ref, kk, fps):
    f = e['f']; n = len(f); hr = np.maximum(e['trk'][:, 3], 1.); a = L.aim_of(o); z = e['z']
    c1 = np.r_[False, np.diff(f) == 1]; c2 = c1 & np.r_[False, c1[:-1]]; c3 = c2 & np.r_[False, c2[:-1]]
    dh = np.full(n, np.nan); da = np.full(n, np.nan); d2a = np.full(n, np.nan); d2c = np.full(n, np.nan); i1 = np.flatnonzero(c1); i2 = np.flatnonzero(c2)
    dh[i1] = np.abs(o[i1, 3] - o[i1 - 1, 3]) / hr[i1]; da[i1] = np.hypot(*(a[i1] - a[i1 - 1]).T) / hr[i1]
    d2a[i2] = np.hypot(*(a[i2] - 2 * a[i2 - 1] + a[i2 - 2]).T) / hr[i2]; d2c[i2] = np.hypot(*(o[i2, :2] - 2 * o[i2 - 1, :2] + o[i2 - 2, :2]).T) / hr[i2]
    spd = np.full(n, np.nan); lag = np.full(n, np.nan); i3 = np.flatnonzero(c3)
    if len(i3):
        v = (z[i3, :2] - z[i3 - 3, :2]) / 3.; sp = np.hypot(*v.T); u = v / np.maximum(sp, 1e-9)[:, None]; spd[i3] = sp / hr[i3]; lag[i3] = np.sum((o[i3, :2] - ref[i3, :2]) * u, 1) / hr[i3]
    return dict(dh=dh, da=da, d2a=d2a, d2c=d2c, spd=spd, lag=lag, first5=(kk >= 1) & (kk <= 4), row1=kk == 1, small=z[:, 3] < 100, hrel=o[:, 3] / hr, close=z[:, 3] >= 240)


def summarize(parts):
    cat = lambda k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)
    dh, da, d2a, d2c = cat('dh'), cat('da'), cat('d2a'), cat('d2c'); fin = lambda v: v[np.isfinite(v)]; q = lambda v, p: round(float(np.percentile(fin(v), p)), 5) if len(fin(v)) else None
    s = dict(n_rows=int(np.isfinite(dh).sum()), dh_p95=q(dh, 95), dh_p99=q(dh, 99), dh_max=q(dh, 100), da_p95=q(da, 95), da_p99=q(da, 99), aim_jitter=round(float(np.sqrt(np.mean(fin(d2a) ** 2))), 5),
             centre_jitter=round(float(np.sqrt(np.mean(fin(d2c) ** 2))), 5), h_over_tracker=round(float(np.median(cat('hrel'))), 3),
             h_over_tracker_close=round(float(np.median(cat('hrel')[cat('close').astype(bool)])), 3) if cat('close').any() else None, n_close=int(cat('close').sum()),
             h_over_tracker_other=round(float(np.median(cat('hrel')[~cat('close').astype(bool)])), 3) if (~cat('close').astype(bool)).any() else None)
    tm = [np.nanmax(p['dh']) for p in parts if np.isfinite(p['dh']).sum() >= 10]; s['dh_trackmax_med'] = round(float(np.median(tm)), 5) if tm else None; s['n_tracks'] = len(tm)
    spd, lag = cat('spd'), cat('lag')
    for a, b in BINS:
        mm = np.isfinite(spd) & (spd >= a) & (spd < b); s['lag_%g-%g' % (a, b)] = dict(n=int(mm.sum()), centre=round(float(np.mean(lag[mm])), 5) if mm.any() else None)
    f5 = cat('first5').astype(bool) & np.isfinite(dh); r1 = cat('row1').astype(bool) & np.isfinite(dh)
    s['first5'] = dict(n=int(f5.sum()), dh_mean=round(float(dh[f5].mean()), 5) if f5.any() else None, dh_max=round(float(dh[f5].max()), 5) if f5.any() else None,
                       da_mean=round(float(da[f5].mean()), 5) if f5.any() else None, n_row1=int(r1.sum()), row1_over_015=round(float(np.mean(dh[r1] > 0.15)), 3) if r1.any() else None)
    return s


def main():
    t0 = time.time(); recs = sys.argv[1:]; sel = json.loads((L.G2 / 'results' / 'valA_selection.json').read_text()); exp = json.loads((L.EXPORT_DIR / 'EXPORT.json').read_text())
    net = L.load_net(L.EXPORT_DIR / exp['npz']); size = tuple(sel['size_chosen']); pe = tuple(sel['conv_euro_t'])
    z = np.load(sorted((L.ROOT / 'export').glob('box_refine_gru_*.npz'))[-1]); v1 = R.RefNumpy({k: z[k] for k in z.files})
    z2 = np.load(L.G2.parent / 'export' / 'box_refine_gru_v2_tmotdance_20261115.npz'); v2 = L.GRUNp({k: z2[k] for k in z2.files}); v2size = tuple(float(x) for x in z2['size_euro'])
    lut2 = (z2['lut_tab'].astype(np.float64), z2['lut_lh_cen'].astype(np.float64), z2['lut_ar_cen'].astype(np.float64)); lut3 = L.conv_delta.__defaults__
    res = dict(weights=exp['npz'], size=list(size), conv_euro_t=list(pe), recs={}, pooled={}); allp = defaultdict(lambda: defaultdict(list))
    for ri, rec in enumerate(recs):
        S, info = streams(rec, ri); res['recs'][rec] = dict(info=info, streams={})
        for st, inp in S.items():
            W, fps, ts = inp['W'], inp['fps'], inp['times']; eo = L.run_rows(inp['ids'], W, fps, inp['n'], net, size, ts); per = defaultdict(list)
            L.conv_delta.__defaults__ = lut2; e2 = L.run_rows(inp['ids'], W, fps, inp['n'], v2, v2size, ts); L.conv_delta.__defaults__ = lut3
            for tid, e in inp['ids'].items():
                stt = eo[tid]['start']; cv = L.conv_hard(e['z'])
                o = dict(raw=e['z'], tracker=e['trk'], euro=L.euro_rows(e, e['z'], stt, W, fps, L.EURO_V1, ts), conv_euro=L.euro_rows(e, cv, stt, W, fps, L.EURO_V1, ts),
                         conv_euro_t=L.euro_rows(e, cv, stt, W, fps, pe, ts), gru_v1=R.run_ref(e, W, v1), gru_v2=e2[tid]['out_f'], gru_v3=eo[tid]['out'], gru_v3_size=eo[tid]['out_f'])
                for m in METHODS:
                    p = per_id(e, o[m], cv if m in L.CONVERTED else e['z'], eo[tid]['k'], fps); per[m].append(p); allp[st][m].append(p); allp['all'][m].append(p)
            res['recs'][rec]['streams'][st] = dict(ids=len(inp['ids']), rows=int(sum(len(e['f']) for e in inp['ids'].values())), methods={m: summarize(v) for m, v in per.items()})
            print('REC %s %s ids %d rows %d (%.1f s)' % (rec, st, len(inp['ids']), res['recs'][rec]['streams'][st]['rows'], time.time() - t0), flush=True)
    Lm = ['# GRU refiner v3 on device-rate recordings (no GT; reference height = tracker row height)', '', 'recordings: %s' % json.dumps({k: v['info'] for k, v in res['recs'].items()}), '']
    for st, pm in allp.items():
        res['pooled'][st] = {m: summarize(v) for m, v in pm.items()}
        Lm += ['## stream %s' % st, '', '| method | dh p95 | dh p99 | dh max | per-track max dh (median) | aim step p99 | aim jitter | centre jitter | lag 0.01-0.03 | lag 0.03-0.1 (n) | lag > 0.1 (n) | first 5 rows dh mean / max (n) | row-1 steps > 0.15 | height / tracker height: all | detector box >= 240 px (n) | other |', '|' + '---|' * 16]
        for m, s in res['pooled'][st].items():
            lg = lambda k: ('%+.4f' % s[k]['centre']) if s[k]['centre'] is not None else '-'
            Lm.append('| %s | %.4f | %.4f | %.3f | %s | %.4f | %.4f | %.4f | %s | %s (%d) | %s (%d) | %s / %s (%d) | %s | %.2f | %s (%d) | %s |' % (m, s['dh_p95'], s['dh_p99'], s['dh_max'], s['dh_trackmax_med'], s['da_p99'], s['aim_jitter'], s['centre_jitter'],
                      lg('lag_0.01-0.03'), lg('lag_0.03-0.1'), s['lag_0.03-0.1']['n'], lg('lag_0.1-99'), s['lag_0.1-99']['n'], s['first5']['dh_mean'], s['first5']['dh_max'], s['first5']['n'], s['first5']['row1_over_015'], s['h_over_tracker'], s['h_over_tracker_close'], s['n_close'], s['h_over_tracker_other']))
            print('  %-4s %-12s dh95 %.4f dh99 %.4f tmax %s da99 %.4f jit %.4f lag mid %s fast %s first5 %s h/trk close %s other %s' % (st, m, s['dh_p95'], s['dh_p99'], s['dh_trackmax_med'], s['da_p99'], s['aim_jitter'], lg('lag_0.03-0.1'), lg('lag_0.1-99'), s['first5']['dh_mean'], s['h_over_tracker_close'], s['h_over_tracker_other']), flush=True)
        Lm.append('')
    (L.G2 / 'results' / 'device.json').write_text(json.dumps(res, indent=1)); (L.G2 / 'results' / 'tables_device.md').write_text('\n'.join(Lm), encoding='utf-8')
    print('DEVICE_DONE %.1f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
