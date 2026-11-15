"""ref_eval.py (PC): box refiner evaluation on server-like tracker rows (PREREG box_refiner + amendment 1).
  python ref_eval.py <root> blend   -> results/blend_valA.json : switch blend window T per method on val-A switch streams (validation loss)
  python ref_eval.py <root> valB    -> results/ref_valB.json   : every metric on val-B, per dataset / stream / size bin, per-sequence bootstrap 90 %
Methods: raw (detector box), tracker (Kalman posterior row), euro (One Euro, camera compensation, val-A values), euro_nocomp (same without compensation,
its own val-A values), refiner (learned GRU, numpy forward of export/box_refine_gru_*.npz); + '+blend' variants on the switch stream.
"""
import json
import math
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prcommon as P
import ref_lib as R

T_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)
G = {}


def init(root):
    root = Path(root); G['root'] = root
    G['euro'] = json.loads((root / 'results' / 'euro_valA.json').read_text())
    f = sorted((root / 'export').glob('box_refine_gru_*.npz'))[-1]; z = np.load(f); G['ref'] = R.RefNumpy({k: z[k] for k in z.files}); G['ref_file'] = f.name
    bl = root / 'results' / 'blend_valA.json'; G['blend'] = json.loads(bl.read_text())['chosen'] if bl.exists() else None


def outputs(e, W, fps):
    eu = G['euro']['comp']; en = G['euro']['nocomp']
    return dict(raw=e['z'].copy(), tracker=e['trk'].copy(),
                euro=R.run_euro(e, W, fps, (eu['mc'], eu['beta'], eu['mc_s'], eu['beta_s'], 1.0), True),
                euro_nocomp=R.run_euro(e, W, fps, (en['mc'], en['beta'], en['mc_s'], en['beta_s'], 1.0), False),
                refiner=R.run_ref(e, W, G['ref']))


def seq_job(args):
    """one sequence x stream -> list of per-id dicts with outputs of every method (and blended ones for the switch stream)."""
    r, stream, blend_ts = args; ids, S = R.rows_of(G['root'], r, stream); tr = np.load(G['root'] / 'data' / 'tracks' / ('%s.npz' % r['name']))
    sw = tr['sw_frames'] if stream == 'sw' else np.zeros(0, np.int64); out = []
    for i, e in ids.items():
        o = outputs(e, S['W'], r['fps'])
        if stream == 'sw':
            for m in list(o):
                if m == 'euro_nocomp': continue
                for T in (blend_ts.get(m, []) if isinstance(blend_ts, dict) else blend_ts):
                    o['%s+blend%g' % (m, T)] = R.blend(e, o[m], r['fps'], T, sw)
        out.append(dict(name=r['name'], ds=r['ds'], stream=stream, id=i, e={k: e[k] for k in ('f', 'tgt', 'hg', 'src', 'kp', 'z')}, o=o, fps=r['fps']))
    return out


# ------------------------------------------------------------------------------------------------------------------ metrics
def aim_of(b): return np.stack([b[:, 0], b[:, 1] - b[:, 3] / 2 + R.AIM_Y * b[:, 3]], 1)


def per_row(rec, m):
    """per-row metric arrays of one id for method m (or 'target')."""
    e = rec['e']; g = e['tgt']; ok = np.isfinite(g[:, 0]); o = g.copy() if m == 'target' else rec['o'][m]; n = len(o); f = e['f']
    gs = np.where(ok[:, None], g, 1.); ht = gs[:, 3]
    ev = R.err_vec(o, gs); ce = np.minimum(np.hypot(ev[:, 0], ev[:, 1]), 2.0)
    iou = P.iou_pair(P.cxcywh2xyxy(o), P.cxcywh2xyxy(gs)); a = aim_of(o); at = aim_of(gs); aim = np.hypot(*(a - at).T) / ht
    c1 = np.r_[False, np.diff(f) == 1]; c2 = np.r_[False, c1[:-1]] & c1
    dh = np.full(n, np.nan); da = np.full(n, np.nan); d2c = np.full(n, np.nan); d2h = np.full(n, np.nan); d2a = np.full(n, np.nan)
    i1 = np.flatnonzero(c1 & ok)
    dh[i1] = np.abs(o[i1, 3] - o[i1 - 1, 3]) / ht[i1]; da[i1] = np.hypot(*(a[i1] - a[i1 - 1]).T) / ht[i1]
    i2 = np.flatnonzero(c2 & ok)
    d2c[i2] = np.hypot(*(o[i2, :2] - 2 * o[i2 - 1, :2] + o[i2 - 2, :2]).T) / ht[i2]
    lh = np.log(np.maximum(o[:, 3], 1e-3)); d2h[i2] = np.abs(lh[i2] - 2 * lh[i2 - 1] + lh[i2 - 2])
    d2a[i2] = np.hypot(*(a[i2] - 2 * a[i2 - 1] + a[i2 - 2]).T) / ht[i2]
    # target speed per frame (heights), along-motion signed error (negative = behind)
    spd = np.full(n, np.nan); along = np.full(n, np.nan)
    j = np.flatnonzero(c1 & ok & np.r_[False, ok[:-1]])
    if len(j):
        v = gs[j, :2] - gs[j - 1, :2]; sp = np.hypot(*v.T); spd[j] = sp / ht[j]; u = v / np.maximum(sp, 1e-9)[:, None]
        along[j] = np.sum((o[j, :2] - gs[j, :2]) * u, 1) / ht[j]
    return dict(ok=ok, ce=ce, lw=np.abs(ev[:, 2]), lh=np.abs(ev[:, 3]), iou=iou, aim=aim, hr=o[:, 3] / ht, dh=dh, da=da, d2c=d2c, d2h=d2h, d2a=d2a, spd=spd, along=along)


def events(rec):
    """per-row event masks: switch (switch frame and the next 0.5 s), switch0 (the switch frame), size (gt height crossing 100 px, +-2), occl (>= 4 visible keypoints lost within 3 frames, +-2), edge (target box crossing the picture edge, +-2)."""
    e = rec['e']; f = e['f']; n = len(f); src = e['src']; fps = rec['fps']; hg = e['hg']; g = e['tgt']
    sw0 = np.r_[False, (src[1:] != src[:-1]) & (np.diff(f) <= 3)]; swn = np.zeros(n, bool)
    for i in np.flatnonzero(sw0): swn |= (f >= f[i]) & (f < f[i] + 0.5 * fps)
    near = lambda m: np.convolve(m.astype(float), np.ones(5), 'full')[2:2 + len(m)] > 0   # +-2 rows ('same' would return 5 values for shorter ids)
    cross = np.r_[False, np.isfinite(hg[1:]) & np.isfinite(hg[:-1]) & ((hg[1:] >= 100) != (hg[:-1] >= 100)) & (np.diff(f) == 1)]
    vis = (e['kp'][:, :, 2] >= 0.5).sum(1); occl = np.zeros(n, bool); lo = np.searchsorted(f, f - 3, 'left')
    for i in range(1, n):
        if lo[i] < i and vis[lo[i]:i].max() - vis[i] >= 4: occl[i] = True
    xy = P.cxcywh2xyxy(np.where(np.isfinite(g), g, 0)); out = (xy[:, 0] < 0) | (xy[:, 2] > P.XW) | (xy[:, 1] < P.Y0) | (xy[:, 3] > P.Y1)
    edge = np.r_[False, (out[1:] != out[:-1]) & np.isfinite(g[1:, 0]) & np.isfinite(g[:-1, 0])]
    sb = np.array([P.size_bin(h) if np.isfinite(h) else -1 for h in hg])
    return dict(switch=swn, switch0=sw0, size=near(cross), occl=near(occl), edge=near(edge)), sb


def boot_mean(vals, seqs, nseq, draws, mask=None):
    """pooled mean and per-sequence bootstrap 90 % interval. vals / seqs per row (nan ignored)."""
    m = np.isfinite(vals) if mask is None else (np.isfinite(vals) & mask)
    if not m.any(): return None
    s = np.bincount(seqs[m], weights=vals[m], minlength=nseq); c = np.bincount(seqs[m], minlength=nseq).astype(float)
    pt = s.sum() / c.sum(); bs = s[draws].sum(1) / np.maximum(c[draws].sum(1), 1e-9)
    return [round(float(pt), 5), round(float(np.percentile(bs, 5)), 5), round(float(np.percentile(bs, 95)), 5), int(c.sum())]


def boot_diff(va, vb, seqs, nseq, draws, mask=None):
    m = np.isfinite(va) & np.isfinite(vb) if mask is None else (np.isfinite(va) & np.isfinite(vb) & mask)
    if not m.any(): return None
    d = va - vb; s = np.bincount(seqs[m], weights=d[m], minlength=nseq); c = np.bincount(seqs[m], minlength=nseq).astype(float)
    bs = s[draws].sum(1) / np.maximum(c[draws].sum(1), 1e-9)
    return [round(float(s.sum() / c.sum()), 5), round(float(np.percentile(bs, 5)), 5), round(float(np.percentile(bs, 95)), 5)]


def pct(vals, mask, q):
    v = vals[mask & np.isfinite(vals)]; return round(float(np.percentile(v, q)), 5) if len(v) else None


def run_valB(root):
    reg = [r for r in P.registry() if r['split'] == 'valB']; bl = G['blend']
    with Pool(8, initializer=init, initargs=(str(root),)) as pool:
        recs = [x for xx in pool.map(seq_job, [(r, s, {m: [bl[m]] for m in bl} if s == 'sw' else {}) for r in reg for s in ('near', 'far', 'sw')]) for x in xx]
    res = dict(ref_file=G['ref_file'], euro=G['euro']['comp'], euro_nocomp={k: G['euro']['nocomp'][k] for k in ('mc', 'beta', 'mc_s', 'beta_s')}, blend_T=bl, tables={})
    seq_names = sorted({r['name'] for r in recs}); sidx = {n: i for i, n in enumerate(seq_names)}
    for ds in ('TMOT', 'DT'):
        for stream in ('near', 'far', 'sw', 'pooled'):
            R_ = [r for r in recs if r['ds'] == ds and (stream == 'pooled' and r['stream'] in ('near', 'far') or r['stream'] == stream)]
            if not R_: continue
            names = sorted({r['name'] for r in R_}); loc = {n: i for i, n in enumerate(names)}; ns = len(names)
            draws = np.random.default_rng(0).integers(0, ns, (2000, ns))
            methods = [m for m in R_[0]['o']] + ['target']
            rows = {m: defaultdict(list) for m in methods}; seqv = []; ev = defaultdict(list); sbv = []; idv = []
            for k, rec in enumerate(R_):
                evm, sb = events(rec); seqv.append(np.full(len(rec['e']['f']), loc[rec['name']])); sbv.append(sb); idv.append(np.full(len(sb), k))
                for kk, v in evm.items(): ev[kk].append(v)
                for m in methods:
                    for kk, v in per_row(rec, m).items(): rows[m][kk].append(v)
            seqv = np.concatenate(seqv); sbv = np.concatenate(sbv); idv = np.concatenate(idv); ev = {k: np.concatenate(v) for k, v in ev.items()}
            rows = {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in rows.items()}
            tab = {}
            for m in methods:
                d = rows[m]; ok = d['ok']; t = {}
                for b in range(-1, 3):
                    bm = ok if b < 0 else ok & (sbv == b); key = 'all' if b < 0 else P.SIZE_NAMES[b]
                    if not bm.any(): continue
                    tb = {k: boot_mean(np.where(bm, d[k], np.nan), seqv, ns, draws) for k in ('ce', 'lw', 'lh', 'iou', 'aim')}
                    for k in ('d2c', 'd2h', 'd2a'):
                        v = np.where(bm, d[k] ** 2, np.nan); bmn = boot_mean(v, seqv, ns, draws)
                        tb[k + '_rms'] = [round(math.sqrt(max(x, 0)), 5) if isinstance(x, float) else x for x in bmn] if bmn else None
                    for k in ('dh', 'da'):
                        tb[k + '_p95'] = pct(d[k], bm, 95); tb[k + '_p99'] = pct(d[k], bm, 99)
                    hr = d['hr'][bm]; tb['h_over_target'] = [round(float(np.median(hr)), 3), round(float(np.percentile(hr, 25)), 3), round(float(np.percentile(hr, 75)), 3)]
                    t[key] = tb
                # largest single-frame jump per track (aim and height), median / p90 over ids with >= 10 steps
                mx = []
                for k in np.unique(idv):
                    mm = (idv == k) & np.isfinite(d['da'])
                    if mm.sum() >= 10: mx.append((np.nanmax(d['da'][mm]), np.nanmax(d['dh'][mm])))
                if mx: mx = np.array(mx); t['max_jump_per_track'] = dict(n_tracks=len(mx), aim_median=round(float(np.median(mx[:, 0])), 4), aim_p90=round(float(np.percentile(mx[:, 0], 90)), 4),
                                                                      h_median=round(float(np.median(mx[:, 1])), 4), h_p90=round(float(np.percentile(mx[:, 1], 90)), 4))
                # events: step size at event frames and the error over them
                t['events'] = {}
                for k, em in ev.items():
                    mm = em & ok
                    if mm.sum() < 5: continue
                    t['events'][k] = dict(n=int(mm.sum()), da_mean=boot_mean(np.where(mm, d['da'], np.nan), seqv, ns, draws), da_p95=pct(d['da'], mm, 95),
                                          dh_mean=boot_mean(np.where(mm, d['dh'], np.nan), seqv, ns, draws), dh_p95=pct(d['dh'], mm, 95), ce=boot_mean(np.where(mm, d['ce'], np.nan), seqv, ns, draws))
                # lag: error by target speed bin (heights per frame)
                t['lag'] = {}
                for a, b in ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 99)):
                    mm = ok & (d['spd'] >= a) & (d['spd'] < b)
                    if mm.sum() < 5: continue
                    t['lag']['%g-%g' % (a, b)] = dict(n=int(mm.sum()), ce=boot_mean(np.where(mm, d['ce'], np.nan), seqv, ns, draws), along=boot_mean(np.where(mm, d['along'], np.nan), seqv, ns, draws))
                tab[m] = t
            # differences: refiner (and its blend) against the others, on 'all' rows
            diffs = {}
            for a_ in [m for m in methods if m.startswith('refiner')]:
                for b_ in [m for m in methods if m != a_ and m != 'target']:
                    ok = rows[a_]['ok']
                    diffs['%s - %s' % (a_, b_)] = {k: boot_diff(np.where(ok, rows[a_][k], np.nan), np.where(ok, rows[b_][k], np.nan), seqv, ns, draws) for k in ('ce', 'aim', 'iou')}
                    diffs['%s - %s' % (a_, b_)].update({k + '_sq': boot_diff(np.where(ok, rows[a_][k] ** 2, np.nan), np.where(ok, rows[b_][k] ** 2, np.nan), seqv, ns, draws) for k in ('d2c', 'd2a', 'd2h')})
            res['tables']['%s|%s' % (ds, stream)] = dict(sequences=names, methods=tab, diffs=diffs)
            print('TABLE', ds, stream, 'rows', int(rows['raw']['ok'].sum()), flush=True)
    (root / 'results' / 'ref_valB.json').write_text(json.dumps(res, indent=1))
    print('REF_VALB_DONE', flush=True)


def run_blend(root):
    """T per method on the val-A switch streams: validation loss over all rows of the switch streams."""
    reg = [r for r in P.registry() if r['split'] == 'valA']
    with Pool(8, initializer=init, initargs=(str(root),)) as pool:
        recs = [x for xx in pool.map(seq_job, [(r, 'sw', list(T_GRID)) for r in reg]) for x in xx]
    res = dict(grid=list(T_GRID), loss={}, chosen={})
    for m in ('raw', 'tracker', 'euro', 'refiner'):
        L_ = {}
        for T in T_GRID:
            key = '%s+blend%g' % (m, T)
            outs = [r['o'][key] for r in recs]; tg = [r['e']['tgt'] for r in recs]; cn = [np.r_[0, (np.diff(r['e']['f']) == 1)] for r in recs]
            L_[T] = R.loss_np(outs, tg, cn)[:3]
        res['loss'][m] = {str(k): v for k, v in L_.items()}; res['chosen'][m] = min(L_, key=lambda k: L_[k][0])
        print('BLEND', m, ' '.join('T%g %.5f' % (k, v[0]) for k, v in L_.items()), '-> T', res['chosen'][m], flush=True)
    (root / 'results' / 'blend_valA.json').write_text(json.dumps(res, indent=1)); print('BLEND_DONE', flush=True)


if __name__ == '__main__':
    root = Path(sys.argv[1]); init(root)
    if sys.argv[2] == 'blend': run_blend(root)
    else: run_valB(root)
