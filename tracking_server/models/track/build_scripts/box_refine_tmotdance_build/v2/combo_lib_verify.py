"""verify/combo/combo_lib.py - refiner + One Euro combinations on the saved server-like tracker rows (rules: COMBO_RULES.json).
Uses the reviewed code as is: scripts/ref_lib (Euro, segments, composed, blend, run_euro, run_ref, rows_of), scripts/ref_eval (per_row, boot_mean, boot_diff).
Reads only; writes only into verify/combo/.
"""
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import prcommon as P    # noqa: E402
import ref_lib as R     # noqa: E402
import ref_eval as RE   # noqa: E402

EURO_VALA = (3.0, 3.0, 0.4, 0.3, 1.0)      # results/euro_valA.json comp (the server EURO dict)
SPEED_BINS = ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 99))
SSPEED_BINS = ((0, 0.01), (0.01, 0.03), (0.03, 99))
WHOLE, H2H = 0.85, 0.75


# ------------------------------------------------------------------------------------------------ methods on top of the refiner
def euro_on(e, inp, W, fps, params, centre=True):
    """One Euro (ref_lib.Euro, camera compensated) over the boxes inp [n,4] of one id; per-id reset = ref_lib.segments of the DETECTOR rows
    (the server's reset rule). centre=False: size only, the centre of inp is passed through untouched."""
    out = np.zeros_like(inp)
    for sl in R.segments(e):
        eu = R.Euro(*params, comp=True); fp = None
        for j in range(sl.start, sl.stop):
            f = e['f'][j]; H = R.composed(W, fp, f) if fp is not None else None; dt = (f - fp) / fps if fp is not None else 1.0 / fps
            b = eu.step(inp[j], dt, H)
            if not centre:
                eu.x[:2] = inp[j][:2]; b = np.r_[inp[j][:2], b[2:]]
            out[j] = b; fp = f
    return out


def combo(rec, W, kind, size=None, cen=None, T=0.0):
    """kind 'a' (size only), 'b' (size + centre), 'c' ((a) + blend T) on rec['o']['refiner']."""
    e = rec['e']; ref = rec['o']['refiner']; fps = rec['fps']
    if kind in ('a', 'c'):
        o = euro_on(e, ref, W, fps, (1.0, 0.0, size[0], size[1], 1.0), centre=False)
    else:
        o = euro_on(e, ref, W, fps, (cen[0], cen[1], size[0], size[1], 1.0), centre=True)
    if kind == 'c' and T > 0 and rec['stream'] == 'sw':
        o = R.blend(e, o, fps, T, rec['sw'])
    return o


# ------------------------------------------------------------------------------------------------ cache of the base outputs
def build_seq(args):
    """one sequence, all three streams -> (name, W, list of per-id records with raw / tracker / euro / refiner outputs)."""
    r, root = args; root = Path(root)
    z = np.load(sorted((root / 'export').glob('box_refine_gru_*.npz'))[-1]); net = R.RefNumpy({k: z[k] for k in z.files})
    tr = np.load(root / 'data' / 'tracks' / ('%s.npz' % r['name'])); out = []; W = None
    for stream in ('near', 'far', 'sw'):
        ids, S = R.rows_of(root, r, stream); W = S['W']; sw = tr['sw_frames'] if stream == 'sw' else np.zeros(0, np.int64)
        for i, e in ids.items():
            o = dict(raw=e['z'].copy(), tracker=e['trk'].copy(), euro=R.run_euro(e, W, r['fps'], EURO_VALA, True), refiner=R.run_ref(e, W, net))
            ee = {k: e[k] for k in ('f', 'z', 'tgt', 'hg', 'src')}; ee['vis'] = (e['kp'][:, :, 2] >= 0.5).sum(1).astype(np.int16)
            out.append(dict(name=r['name'], ds=r['ds'], split=r['split'], stream=stream, id=i, fps=r['fps'], sw=sw, e=ee, o=o))
    return r['name'], W, out


def load_cache(split):
    d = pickle.loads((HERE / 'cache' / ('%s.pkl' % split)).read_bytes()); return d['recs'], d['W']


# ------------------------------------------------------------------------------------------------ per-row metrics and events
def rows_metrics(rec, o):
    """ref_eval.per_row on output o + size lag (sign of the target's log-height change x log h / h_target)."""
    rr = dict(rec); rr['o'] = {'m': o}; d = RE.per_row(rr, 'm')
    e = rec['e']; g = e['tgt']; ok = d['ok']; f = e['f']; n = len(f)
    lt = np.log(np.where(ok, g[:, 3], 1.)); lo = np.log(np.maximum(o[:, 3], 1e-3)); ls = lo - lt
    sspd = np.full(n, np.nan); salong = np.full(n, np.nan)
    j = np.flatnonzero(np.r_[False, np.diff(f) == 1] & ok & np.r_[False, ok[:-1]])
    if len(j):
        dl = lt[j] - lt[j - 1]; sspd[j] = np.abs(dl); salong[j] = np.sign(dl) * ls[j]
    d['sspd'] = sspd; d['salong'] = salong
    return d


def events(rec):
    """ref_eval.events (same definitions; visible keypoint counts stored instead of keypoints) + switch window 1 s, flip event, 50 / 70 px crossings."""
    e = rec['e']; f = e['f']; n = len(f); src = e['src']; fps = rec['fps']; hg = e['hg']; g = e['tgt']
    sw0 = np.r_[False, (src[1:] != src[:-1]) & (np.diff(f) <= 3)]; swn = np.zeros(n, bool); sw1 = np.zeros(n, bool)
    for i in np.flatnonzero(sw0):
        swn |= (f >= f[i]) & (f < f[i] + 0.5 * fps); sw1 |= (f >= f[i]) & (f <= f[i] + 1.0 * fps)
    near = lambda m: np.convolve(m.astype(float), np.ones(5), 'full')[2:2 + len(m)] > 0
    fin = np.isfinite(hg[1:]) & np.isfinite(hg[:-1]) & (np.diff(f) == 1)
    cross = lambda th: np.r_[False, fin & ((hg[1:] >= th) != (hg[:-1] >= th))]
    vis = e['vis'].astype(int); occl = np.zeros(n, bool); lo = np.searchsorted(f, f - 3, 'left')
    for i in range(1, n):
        if lo[i] < i and vis[lo[i]:i].max() - vis[i] >= 4: occl[i] = True
    xy = P.cxcywh2xyxy(np.where(np.isfinite(g), g, 0)); out = (xy[:, 0] < 0) | (xy[:, 2] > P.XW) | (xy[:, 1] < P.Y0) | (xy[:, 3] > P.Y1)
    edge = np.r_[False, (out[1:] != out[:-1]) & np.isfinite(g[1:, 0]) & np.isfinite(g[:-1, 0])]
    rt = e['z'][:, 3] / np.where(np.isfinite(hg), hg, np.nan)
    reg = np.where(rt >= WHOLE, 1, np.where(rt <= H2H, 0, -1))
    for i in range(1, n):   # in-between ratios keep the last regime of the same detector run (a gradual change is flagged where it ends)
        if reg[i] < 0 and src[i] == src[i - 1] and f[i] - f[i - 1] == 1: reg[i] = reg[i - 1]
    flip = np.r_[False, fin & (src[1:] == src[:-1]) & (reg[1:] >= 0) & (reg[:-1] >= 0) & (reg[1:] != reg[:-1])]
    kk = np.zeros(n, int)
    for sl in R.segments(e): kk[sl] = np.arange(sl.stop - sl.start)
    ev = dict(switch0=sw0, switch=swn, switch1s=sw1, size=near(cross(100)), size70=near(cross(70)), size50=near(cross(50)), flip=near(flip), flip0=flip,
              occl=near(occl), edge=near(edge), start=(kk == 1) | (kk == 2), later=kk >= 3)
    sb = np.array([P.size_bin(h) if np.isfinite(h) else -1 for h in hg])
    return ev, sb


# ------------------------------------------------------------------------------------------------ pooled tables
KEYS = ('ok', 'ce', 'lw', 'lh', 'iou', 'aim', 'hr', 'dh', 'da', 'd2c', 'd2h', 'd2a', 'spd', 'along', 'sspd', 'salong')


def gather(recs, outs, evs):
    """recs and their outputs (list of [n,4]) and events -> concatenated row arrays + seq index + id index."""
    names = sorted({r['name'] for r in recs}); loc = {n: i for i, n in enumerate(names)}
    rows = defaultdict(list); seqv = []; idv = []; ev = defaultdict(list); sbv = []
    for k, (rec, o, (em, sb)) in enumerate(zip(recs, outs, evs)):
        d = rows_metrics(rec, o)
        for kk in KEYS: rows[kk].append(d[kk])
        n = len(rec['e']['f']); seqv.append(np.full(n, loc[rec['name']])); idv.append(np.full(n, k)); sbv.append(sb)
        for kk, v in em.items(): ev[kk].append(v)
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    return rows, np.concatenate(seqv), np.concatenate(idv), {k: np.concatenate(v) for k, v in ev.items()}, np.concatenate(sbv), names


def summary(rows, ev, idv):
    """point estimates used for the val-A choices and the val-B tables."""
    ok = rows['ok']; s = {}
    for k in ('ce', 'aim', 'lh', 'lw', 'iou'): s[k] = float(np.mean(rows[k][ok]))
    for k in ('d2c', 'd2a', 'd2h'):
        v = rows[k][ok & np.isfinite(rows[k])]; s[k + '_rms'] = float(math.sqrt(np.mean(v ** 2)))
    for k in ('dh', 'da'):
        v = rows[k][ok & np.isfinite(rows[k])]
        for q in (95, 99, 99.9): s['%s_p%g' % (k, q)] = float(np.percentile(v, q))
        s[k + '_max'] = float(v.max())
    mx = []
    for k in np.unique(idv):
        mm = (idv == k) & np.isfinite(rows['da'])
        if mm.sum() >= 10: mx.append((np.nanmax(rows['dh'][mm]), np.nanmax(rows['da'][mm])))
    mx = np.array(mx); s['n_tracks'] = len(mx)
    s['dh_trackmax_med'] = float(np.median(mx[:, 0])); s['dh_trackmax_p90'] = float(np.percentile(mx[:, 0], 90))
    s['da_trackmax_med'] = float(np.median(mx[:, 1])); s['da_trackmax_p90'] = float(np.percentile(mx[:, 1], 90))
    for a, b in SPEED_BINS:
        mm = ok & (rows['spd'] >= a) & (rows['spd'] < b)
        s['lag_%g-%g' % (a, b)] = dict(n=int(mm.sum()), ce=float(np.mean(rows['ce'][mm])) if mm.any() else None, along=float(np.mean(rows['along'][mm])) if mm.any() else None)
    for a, b in SSPEED_BINS:
        mm = ok & (rows['sspd'] >= a) & (rows['sspd'] < b)
        s['slag_%g-%g' % (a, b)] = dict(n=int(mm.sum()), lh=float(np.mean(rows['lh'][mm])) if mm.any() else None, salong=float(np.mean(rows['salong'][mm])) if mm.any() else None)
    s['events'] = {}
    for k, em in ev.items():
        mm = em & ok
        vh = rows['dh'][mm & np.isfinite(rows['dh'])]; va = rows['da'][mm & np.isfinite(rows['da'])]
        if len(vh) < 5: continue
        s['events'][k] = dict(n=int(len(vh)), dh_mean=float(vh.mean()), dh_p95=float(np.percentile(vh, 95)), dh_p99=float(np.percentile(vh, 99)), dh_max=float(vh.max()),
                              da_mean=float(va.mean()), da_p95=float(np.percentile(va, 95)), da_p99=float(np.percentile(va, 99)), da_max=float(va.max()),
                              ce=float(np.mean(rows['ce'][mm])))
    return s


# ------------------------------------------------------------------------------------------------ bootstrap of percentiles / per-track maxima
EDGES = np.r_[0.0, np.logspace(-6, 1, 4000), np.inf]


def seq_hist(vals, seqv, mask, ns):
    m = mask & np.isfinite(vals); H = np.zeros((ns, len(EDGES) - 1))
    for s in range(ns):
        v = vals[m & (seqv == s)]
        if len(v): H[s] = np.histogram(v, EDGES)[0]
    return H


def hist_q(Hd, q):
    """Hd [D, nb] summed histograms -> q-quantile per draw (upper bin edge, inf bin -> last finite edge)."""
    c = np.cumsum(Hd, 1); tot = c[:, -1:]; idx = np.argmax(c >= q / 100.0 * tot, 1)
    up = EDGES[1:].copy(); up[-1] = EDGES[-2]; return up[idx]


def draw_counts(draws, ns):
    C = np.zeros((len(draws), ns))
    for s in range(ns): C[:, s] = (draws == s).sum(1)
    return C


def boot_pct(vals_a, vals_b, seqv, mask, ns, C, q):
    """point (exact) and 90 % interval of the q-th percentile of a, and of (a - b) when b is given."""
    Ha = seq_hist(vals_a, seqv, mask, ns); qa = hist_q(C @ Ha, q)
    pa = float(np.percentile(vals_a[mask & np.isfinite(vals_a)], q))
    res = dict(a=[round(pa, 5), round(float(np.percentile(qa, 5)), 5), round(float(np.percentile(qa, 95)), 5)])
    if vals_b is not None:
        Hb = seq_hist(vals_b, seqv, mask, ns); qb = hist_q(C @ Hb, q); pb = float(np.percentile(vals_b[mask & np.isfinite(vals_b)], q))
        dd = qa - qb; res['diff'] = [round(pa - pb, 5), round(float(np.percentile(dd, 5)), 5), round(float(np.percentile(dd, 95)), 5)]
    return res


def track_max(rows, idv, seqv, key):
    tv = []; ts = []
    for k in np.unique(idv):
        mm = (idv == k) & np.isfinite(rows['da'])
        if mm.sum() >= 10: tv.append(np.nanmax(rows[key][mm])); ts.append(seqv[mm][0])
    return np.array(tv), np.array(ts, int)


def wq(v, w, q):
    o = np.argsort(v); v = v[o]; w = w[o]; c = np.cumsum(w)
    if c[-1] <= 0: return np.nan
    return v[min(np.searchsorted(c, q / 100.0 * c[-1]), len(v) - 1)]


def boot_trackmax(va, sa, vb, C, q):
    pa = float(np.percentile(va, q)); qa = np.array([wq(va, Cd[sa], q) for Cd in C])
    res = dict(a=[round(pa, 5), round(float(np.nanpercentile(qa, 5)), 5), round(float(np.nanpercentile(qa, 95)), 5)])
    if vb is not None:
        pb = float(np.percentile(vb, q)); qb = np.array([wq(vb, Cd[sa], q) for Cd in C]); dd = qa - qb
        res['diff'] = [round(pa - pb, 5), round(float(np.nanpercentile(dd, 5)), 5), round(float(np.nanpercentile(dd, 95)), 5)]
    return res
