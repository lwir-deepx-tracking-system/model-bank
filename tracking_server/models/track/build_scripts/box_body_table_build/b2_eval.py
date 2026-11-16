"""body_v2/scripts/b2_eval.py - numbers of body mode v2 (rules: body_v2/PREREG_BODY_V2.json). Every method runs through the SERVER Refiner
(worktree tracking_server/src/track/box_refine.py) frame by frame, hand-over on, the server's detection rule (the matched detection when its IoU
with the tracker row is >= 0.3, else the row object itself), t = frame / fps (device recordings: real frame times), server GMC warps.
  python b2_eval.py valA <workers>                      -> results/valA_grid.json  (the size-filter grid and the pre-registered choice)
  python b2_eval.py valB <euro mc> <beta>              -> results/valB.json, results/tables_valB.md   (once)
  python b2_eval.py device <euro mc> <beta> <rec ...>  -> results/device.json, results/tables_device.md (once)
Tables: T1 = the pushed table (gru_v2/v3b lut_v3, = models/track/box_body_table_20261115.npz), T2 = body_v2/data/table_v2.npz.
"""
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import b2_lib as B
from b2_lib import BR, L, P, R, RE, C

DK = ('near', 'far')
HB = ((0, 40), (40, 70), (70, 100), (100, 170), (170, 240), (240, 99999))
SP_BINS = ((0.03, 0.1), (0.1, 99))
INF = float('inf')
PR = json.loads((B.B2 / 'PREREG_BODY_V2.json').read_text(encoding='utf-8'))
G = {}


def tables():
    t1 = BR.BodyTable(dict(tab=L.TAB, lh_cen=L.LH_CEN, ar_cen=L.AR_CEN))
    z = np.load(B.B2 / 'data' / 'table_v2.npz', allow_pickle=False); t2 = BR.BodyTable(dict(tab=z['tab'], lh_cen=z['lh_cen'], ar_cen=z['ar_cen'], person=z['person']))
    return t1, t2


def method_specs(mc, beta, aim_y):
    """name -> (mode, table key, body / size dict)"""
    return dict(size=('size', None, dict(BR.SIZE)),
                body_pushed=('body', 't1', dict(anchor='top', mc_s=0.4, beta_s=0.0, dcut=1.0, sw=False)),
                conv1=('body', 't1', dict(anchor='top', mc_s=INF, beta_s=0.0, dcut=1.0, sw=False)),
                body_v2=('body', 't2', dict(anchor='aim', mc_s=mc, beta_s=beta, dcut=1.0, sw=True)),
                conv=('body', 't2', dict(anchor='aim', mc_s=INF, beta_s=0.0, dcut=1.0, sw=True)))


# ------------------------------------------------------------------------------------------------ running the server module
def items_of(ids):
    """per frame: [(tid, row j, z = the matched detection (a new array) when IoU(row, detection) >= 0.3 else the row object, row)]"""
    by_f = defaultdict(list)
    for tid in sorted(ids):
        e = ids[tid]; trk = np.c_[e['trk'][:, :2], np.maximum(e['trk'][:, 2:], 1.)]; z = np.c_[e['z'][:, :2], np.maximum(e['z'][:, 2:], 1.)]
        iou = P.iou_pair(P.cxcywh2xyxy(trk), P.cxcywh2xyxy(z))
        for j, f in enumerate(e['f']):
            zr = trk[j].copy(); by_f[int(f)].append((tid, j, z[j].copy() if iou[j] >= 0.3 else zr, zr))
    return by_f


def run(ids, by_f, W, n, keys, times, fps, mode, table, cfg, aim_y, family='pose'):
    """-> ({tid: out [n_rows, 4]}, {tid: k [n_rows]}, {tid: t [n_rows]}, refiner)"""
    if mode == 'size': r = BR.Refiner('size', None, 0.0, cfg, aim_y)
    else: r = BR.Refiner('body', table, 0.0, None, aim_y, cfg)
    r.family_of = (lambda k: family)
    out = {tid: np.zeros((len(e['f']), 4)) for tid, e in ids.items()}; kk = {tid: np.zeros(len(e['f']), np.int64) for tid, e in ids.items()}; tt = {tid: np.zeros(len(e['f'])) for tid, e in ids.items()}
    for f in range(n):
        its = by_f.get(f, ()); H = W[f] if (W is not None and 0 < f < len(W)) else None; t = float(times[f]) if times is not None else f / fps
        res = r.step([(tid, z, zr) for tid, j, z, zr in its], t, H, keys[f], True)
        for tid, j, *_ in its: out[tid][j] = res[tid]; kk[tid][j] = r.tr[tid]['k']; tt[tid][j] = t
    return out, kk, tt, r


def own_ref(table, z, family='pose'):
    """the matched detection in the method's own definition (body: converted with its table)"""
    z = np.c_[z[:, :2], np.maximum(z[:, 2:], 1.)]; d = table.conv_delta(z, family)
    return np.stack([z[:, 0] + d[:, 0] * z[:, 3], z[:, 1] + d[:, 1] * z[:, 3], z[:, 2] * np.exp(d[:, 2]), z[:, 3] * np.exp(d[:, 3])], 1)


def aim_of(b, ay): return np.stack([b[:, 0], b[:, 1] - b[:, 3] / 2 + ay * b[:, 3]], 1)


# ------------------------------------------------------------------------------------------------ per-row metrics (one id, one method)
def per_id(e, o, ref, conv, kk, tt, ay, gt=True):
    """e: f, z, trk, src, [tgt]; o output, ref = own-definition detection, conv = the same method without the filter (or None), kk rows since the
    state start, tt times. Reference height = target height (gt) or the tracker row's height (device)."""
    f = e['f']; n = len(f); trk = e['trk']
    if gt:
        g = e['tgt']; ok = np.isfinite(g[:, 0]); gs = np.where(ok[:, None], g, 1.); hr = gs[:, 3]
    else:
        ok = np.ones(n, bool); hr = np.maximum(trk[:, 3], 1.); gs = None
    a = aim_of(o, ay); c1 = np.r_[False, np.diff(f) == 1]; c2 = c1 & np.r_[False, c1][:n]; c3 = c2 & np.r_[False, c2][:n]
    i1 = np.flatnonzero(c1 & ok & np.r_[False, ok][:n]); i2 = np.flatnonzero(c2 & ok)
    dh = np.full(n, np.nan); da = np.full(n, np.nan); d2a = np.full(n, np.nan)
    dh[i1] = np.abs(o[i1, 3] - o[i1 - 1, 3]) / hr[i1]; da[i1] = np.hypot(*(a[i1] - a[i1 - 1]).T) / hr[i1]; d2a[i2] = np.hypot(*(a[i2] - 2 * a[i2 - 1] + a[i2 - 2]).T) / hr[i2]
    m = dict(ok=ok, dh=dh, da=da, d2a=d2a, k=kk, src=e['src'], trk_h=trk[:, 3], hrel=o[:, 3] / np.maximum(trk[:, 3], 1.), dx=np.abs(o[:, 0] - trk[:, 0]) / np.maximum(trk[:, 3], 1.))
    # aim lag vs the own-definition detection along the motion (gt: the target's motion this frame; device: the detection's motion over 3 frames)
    spd = np.full(n, np.nan); alag = np.full(n, np.nan); slag = np.full(n, np.nan); ar = aim_of(ref, ay)
    if gt:
        j = i1
        if len(j):
            v = gs[j, :2] - gs[j - 1, :2]; sp = np.hypot(*v.T); u = v / np.maximum(sp, 1e-9)[:, None]; spd[j] = sp / hr[j]; alag[j] = np.sum((a[j] - ar[j]) * u, 1) / hr[j]
            dl = np.log(gs[j, 3]) - np.log(gs[j - 1, 3]); fast = np.abs(dl) >= 0.03
            slag[j[fast]] = np.sign(dl[fast]) * (np.log(np.maximum(o[j[fast], 3], 1e-3)) - np.log(np.maximum(ref[j[fast], 3], 1e-3)))
        aim_err = np.hypot(*(a - aim_of(gs, ay)).T) / hr; lh = np.abs(np.log(np.maximum(o[:, 3], 1e-3) / gs[:, 3])); m.update(aim=aim_err, lh=lh, htgt=o[:, 3] / gs[:, 3])
    else:
        j = np.flatnonzero(c3)
        if len(j):
            z = e['z']; v = (z[j, :2] - z[j - 3, :2]) / 3.; sp = np.hypot(*v.T); u = v / np.maximum(sp, 1e-9)[:, None]; spd[j] = sp / hr[j]; alag[j] = np.sum((a[j] - ar[j]) * u, 1) / hr[j]
    m.update(spd=spd, alag=alag, slag=slag)
    # under-size against the same method without the filter: first second of a state segment, fast growth of the converted height
    if conv is not None:
        u = o[:, 3] / np.maximum(conv[:, 3], 1e-6); first = np.zeros(n, bool); grow = np.zeros(n, bool); t0 = None
        for q in range(n):
            if kk[q] == 0: t0 = tt[q]
            first[q] = kk[q] >= 1 and t0 is not None and tt[q] - t0 <= 1.0 + 1e-9
            if kk[q] >= 3 and c3[q] and tt[q] > tt[q - 3]: grow[q] = (math.log(conv[q, 3]) - math.log(conv[q - 3, 3])) / (tt[q] - tt[q - 3]) >= 0.5
        m.update(u=u, first=first, grow=grow)
        if ay is not None: m['aimdiff'] = np.hypot(*(a - aim_of(conv, ay)).T) / np.maximum(o[:, 3], 1.)
    sw0 = np.r_[False, (e['src'][1:] != e['src'][:-1])] & c1; sw3 = sw0 | (np.r_[False, sw0][:n] & c1) | (np.r_[False, False, sw0][:n] & c2)
    m.update(sw0=sw0, sw3=sw3)
    return m


def pool(parts, key):
    return np.concatenate([p[key] for p in parts]) if parts else np.zeros(0)


def summ(parts, gt=True):
    """pooled numbers of a list of per_id dicts"""
    fin = lambda v: v[np.isfinite(v)]; q = lambda v, p: float(np.percentile(fin(v), p)) if len(fin(v)) else None; mean = lambda v: float(np.mean(fin(v))) if len(fin(v)) else None
    ok = pool(parts, 'ok').astype(bool); dh = pool(parts, 'dh'); da = pool(parts, 'da'); d2a = pool(parts, 'd2a')
    s = dict(n_rows=int(ok.sum()), dh_p99=q(dh, 99), dh_p95=q(dh, 95), da_p99=q(da, 99), aim_jitter=float(np.sqrt(np.mean(fin(d2a) ** 2))) if len(fin(d2a)) else None)
    tm = [np.nanmax(p['dh']) for p in parts if np.isfinite(p['dh']).sum() >= 10]; s['tmax'] = float(np.median(tm)) if tm else None; s['n_tracks'] = len(tm)
    k = pool(parts, 'k'); f5 = (k >= 1) & (k <= 4) & np.isfinite(dh); s['first5'] = mean(dh[f5]); s['first5_n'] = int(f5.sum())
    spd = pool(parts, 'spd'); al = pool(parts, 'alag')
    for lo, hi in SP_BINS:
        mm = np.isfinite(spd) & (spd >= lo) & (spd < hi); s['aim_lag_%g' % lo] = mean(al[mm]); s['aim_lag_%g_n' % lo] = int(mm.sum())
    if gt:
        s['aim'] = mean(pool(parts, 'aim')[ok]); s['lh'] = mean(pool(parts, 'lh')[ok]); sl = pool(parts, 'slag'); s['size_lag'] = mean(sl); s['size_lag_n'] = int(np.isfinite(sl).sum())
    if 'u' in parts[0]:
        u = pool(parts, 'u'); fi = pool(parts, 'first').astype(bool); gr = pool(parts, 'grow').astype(bool)
        s['under_first'] = dict(n=int(fi.sum()), share_lt090=float(np.mean(u[fi] < 0.90)) if fi.any() else None, median=float(np.median(u[fi])) if fi.any() else None, p10=float(np.percentile(u[fi], 10)) if fi.any() else None)
        s['under_grow'] = dict(n=int(gr.sum()), median=float(np.median(u[gr])) if gr.any() else None, p10=float(np.percentile(u[gr], 10)) if gr.any() else None, share_lt090=float(np.mean(u[gr] < 0.90)) if gr.any() else None)
        if 'aimdiff' in parts[0]: s['aim_vs_conv_max'] = float(np.nanmax(pool(parts, 'aimdiff')))
    sw = pool(parts, 'sw0').astype(bool) & np.isfinite(dh); sw3 = pool(parts, 'sw3').astype(bool) & np.isfinite(dh)
    s['switch'] = dict(n=int(sw.sum()), mean=mean(dh[sw]), p95=q(dh[sw], 95) if sw.sum() >= 5 else None, max=float(dh[sw].max()) if sw.any() else None,
                       n3=int(sw3.sum()), p95_3=q(dh[sw3], 95) if sw3.sum() >= 5 else None, da_mean=mean(da[sw]))
    th = pool(parts, 'trk_h'); src = pool(parts, 'src'); hrel = pool(parts, 'hrel'); dx = pool(parts, 'dx'); s['bins'] = {}
    for sn, sv in (('both', None), ('near', 0), ('far', 1)):
        for lo, hi in HB:
            mm = (th >= lo) & (th < hi) & ((src == sv) if sv is not None else True)
            if mm.sum() < 20: continue
            b = dict(n=int(mm.sum()), h_trk=[float(np.percentile(hrel[mm], x)) for x in (10, 50, 90)], dx=[float(np.percentile(dx[mm], x)) for x in (50, 90)])
            if gt:
                ht = pool(parts, 'htgt'); m2 = mm & ok; b['h_tgt'] = [float(np.percentile(ht[m2], x)) for x in (10, 50, 90)] if m2.sum() >= 20 else None; b['n_tgt'] = int(m2.sum())
            s['bins']['%s|%d-%d' % (sn, lo, hi)] = b
    return s


# ------------------------------------------------------------------------------------------------ val-A / val-B groups
def groups(split):
    d = pickle.loads((L.G2 / 'data' / ('rows_%s.pkl' % split)).read_bytes()); recs, Wd, nd = d['recs'], d['W'], d['n']; gr = defaultdict(list)
    for r in recs: gr[(r['name'], r['stream'])].append(r)
    out = []
    for (name, stream), rr in sorted(gr.items()):
        ids = {r['id']: dict(f=r['e']['f'], z=r['e']['z'], trk=r['e']['trk'], src=r['e']['src'], tgt=r['e']['tgt']) for r in rr}; n = nd[name]
        keys = [None] * n
        for r in rr:
            for f, s in zip(r['e']['f'], r['e']['src']): keys[int(f)] = DK[int(s)]
        last = 'near' if stream != 'far' else 'far'
        for f in range(n):
            if keys[f] is None: keys[f] = last
            last = keys[f]
        out.append(dict(name=name, stream=stream, ds=rr[0]['ds'], fps=rr[0]['fps'], ids=ids, by_f=items_of(ids), W=Wd[name], n=n, keys=keys, recs=rr))
    return out


def init_valA():
    G['groups'] = groups('valA'); G['t1'], G['t2'] = tables()


def run_group(gp, mode, table, cfg, ay):
    return run(gp['ids'], gp['by_f'], gp['W'], gp['n'], gp['keys'], None, gp['fps'], mode, table, cfg, ay)


def valA_point(cfg):
    """one grid point (mc, beta) or a named base method -> {ds: numbers}"""
    ay = 0.5; t1, t2 = G['t1'], G['t2']; res = {}
    if isinstance(cfg, str):
        spec = method_specs(0.4, 0.0, ay)
        mode, tk, c = spec[cfg] if cfg != 'tracker' else (None, None, None)
    else:
        mode, tk, c = 'body', 't2', dict(anchor='aim', mc_s=cfg[0], beta_s=cfg[1], dcut=1.0, sw=True)
    conv_c = dict(c or {}, mc_s=INF) if mode == 'body' else None
    parts = defaultdict(list)
    for gp in G['groups']:
        if mode is None: o = {tid: e['trk'] for tid, e in gp['ids'].items()}; kk, tt = run_group(gp, 'body', t2, dict(mc_s=INF, beta_s=0, dcut=1, anchor='aim'), ay)[1:3]; cv = None
        else:
            o, kk, tt, _ = run_group(gp, mode, t2 if tk == 't2' else t1, c, ay); cv = run_group(gp, mode, t2 if tk == 't2' else t1, conv_c, ay)[0] if conv_c else None
        for tid, e in gp['ids'].items():
            ref = e['z'] if mode in (None, 'size') else own_ref(t2 if tk == 't2' else t1, e['z'])
            parts[gp['ds']].append(per_id(e, o[tid], ref, cv[tid] if cv is not None else None, kk[tid], tt[tid], ay))
    for ds, pp in parts.items(): res[ds] = summ(pp)
    return cfg, res


def valA(nw):
    t0 = time.time(); F = PR['filter_valA']; grid = [(mc, b) for mc in (0.2, 0.4, 0.8, 1.5, 3.0, 5.0, 8.0) for b in (0.0, 0.3, 1.0, 3.0, 10.0)]
    with Pool(nw, initializer=init_valA) as pool_:
        base = dict(pool_.map(valA_point, ['tracker', 'conv', 'body_pushed']))
        pts = pool_.map(valA_point, grid, chunksize=1)
    out = dict(rules=F, base=base, points=[]); dss = ('TMOT', 'DT')
    for (mc, b), m in pts:
        J = float(np.mean([(m[ds]['dh_p99'] / base['conv'][ds]['dh_p99'] + m[ds]['tmax'] / base['conv'][ds]['tmax']) / 2 for ds in dss]))
        v = []
        for ds in dss:
            v.append(((base['tracker'][ds]['size_lag'] - 0.02) - m[ds]['size_lag']) / 0.02)                       # C1
            v.append(m[ds]['under_first']['share_lt090'] / 0.05 - 1)                                             # C2
            if m[ds]['under_grow']['n'] >= 50: v.append((0.90 - m[ds]['under_grow']['median']) / 0.10)         # C3
            v.append((m[ds]['lh'] - (base['conv'][ds]['lh'] + 0.005)) / 0.005)                                  # C4
        out['points'].append(dict(mc_s=mc, beta_s=b, J=J, viol=float(max(v)), m=m))
        print('PT mc %.1f beta %.1f J %.4f viol %+.3f | %s' % (mc, b, J, max(v), ' | '.join('%s dh99 %.4f tmax %.4f slag %+.4f first<0.9 %.3f grow med %s lh %.4f' % (
            ds, m[ds]['dh_p99'], m[ds]['tmax'], m[ds]['size_lag'], m[ds]['under_first']['share_lt090'], ('%.3f' % m[ds]['under_grow']['median']) if m[ds]['under_grow']['median'] is not None else '-', m[ds]['lh']) for ds in dss)), flush=True)
    feas = [p for p in out['points'] if p['viol'] <= 0]
    if feas:
        best = min(p['J'] for p in feas); near = [p for p in feas if p['J'] <= best + 0.005]; ch = sorted(near, key=lambda p: (-p['mc_s'], -p['beta_s']))[0]; ch = dict(ch, feasible=True)
    else:
        ch = dict(min(out['points'], key=lambda p: p['viol']), feasible=False)
    out['choice'] = {k: v for k, v in ch.items() if k != 'm'}; out['choice_metrics'] = ch['m']
    pu = [p for p in out['points'] if p['mc_s'] == 0.4 and p['beta_s'] == 0.0][0]; out['pushed_filter_point'] = {k: v for k, v in pu.items() if k != 'm'}
    for nm in ('tracker', 'conv', 'body_pushed'):
        print('BASE %s | %s' % (nm, ' | '.join('%s dh99 %.4f tmax %.4f slag %s lh %s' % (ds, base[nm][ds]['dh_p99'], base[nm][ds]['tmax'], base[nm][ds].get('size_lag'), base[nm][ds].get('lh')) for ds in dss)), flush=True)
    print('PUSHED_FILTER_POINT', json.dumps(out['pushed_filter_point']), flush=True); print('CHOICE', json.dumps(out['choice']), flush=True)
    out['seconds'] = round(time.time() - t0, 1); (B.B2 / 'results' / 'valA_grid.json').write_text(B.jdump(out), encoding='utf-8'); print('VALA_DONE %.0f s' % (time.time() - t0), flush=True)


# ------------------------------------------------------------------------------------------------ val-B once
AIMS = (0.5, 0.37, 0.08)
METHODS = ('tracker', 'size', 'body_pushed', 'body_v2', 'conv', 'conv1')
AIM_DEP = ('size', 'body_v2', 'conv')   # methods whose box depends on aim_y (aim anchors)


def run_methods(gp, t1, t2, mc, beta, ay, only=None, times=None, family='pose'):
    """-> {method: (out, kk, tt, refiner)} for one group / stream at one aim_y"""
    spec = method_specs(mc, beta, ay); res = {}
    for m in (only or METHODS):
        if m == 'tracker': continue
        mode, tk, c = spec[m]; fam = family if m in ('body_v2', 'conv') else 'pose'
        res[m] = run(gp['ids'], gp['by_f'], gp['W'], gp['n'], gp['keys'], times, gp['fps'], mode, t2 if tk == 't2' else t1, c, ay, fam)
    return res


def seq_boot_mean(vals, seqv, mask, draws):
    """per-sequence bootstrap of a mean -> (point, draws array)"""
    ns = draws.shape[1]; v = np.where(mask & np.isfinite(vals), vals, 0.); c = (mask & np.isfinite(vals)).astype(float)
    S = np.bincount(seqv, v, ns); N = np.bincount(seqv, c, ns); Cm = C.draw_counts(draws, ns)
    return float(S.sum() / max(N.sum(), 1)), (Cm @ S) / np.maximum(Cm @ N, 1)


def diff_ci(a, b):
    """point and 90 % interval of a - b from (point, draws) pairs"""
    d = a[1] - b[1]; return [round(a[0] - b[0], 5), round(float(np.percentile(d, 5)), 5), round(float(np.percentile(d, 95)), 5)]


def valB(mc, beta):
    t0 = time.time(); t1, t2 = tables(); gr = groups('valB'); parts = defaultdict(list); seqs = {}; hand = defaultdict(int); sw = defaultdict(int)
    for gp in gr:
        part = 'nf' if gp['stream'] in ('near', 'far') else 'sw'; si = seqs.setdefault(gp['name'], len(seqs))
        for ay in AIMS:
            rm = run_methods(gp, t1, t2, mc, beta, ay, None if ay == 0.5 else AIM_DEP)
            kk, tt = rm['conv'][1], rm['conv'][2]
            for m in (METHODS if ay == 0.5 else AIM_DEP):
                o = {tid: e['trk'] for tid, e in gp['ids'].items()} if m == 'tracker' else rm[m][0]
                cv = rm['conv'][0] if m == 'body_v2' else rm['conv1'][0] if m == 'body_pushed' and 'conv1' in rm else None
                tk = t2 if m in ('body_v2', 'conv') else t1 if m in ('body_pushed', 'conv1') else None
                if ay == 0.5 and m in rm: hand[m] += rm[m][3].handovers; sw[m] += getattr(rm[m][3], 'switches', 0)
                for tid, e in gp['ids'].items():
                    ref = own_ref(tk, e['z']) if tk is not None else e['z']
                    d = per_id(e, o[tid], ref, cv[tid] if cv is not None else None, kk[tid], tt[tid], ay); d['seq'] = np.full(len(e['f']), si)
                    parts[(gp['ds'], part, m, ay)].append(d)
        print('GROUP %s %s (%.0f s)' % (gp['name'], gp['stream'], time.time() - t0), flush=True)
    res = dict(rules=PR['pass_rules'], euro=[mc, beta, 1.0], handovers=dict(hand), switches=dict(sw), tables={}, boot={}, seqs=seqs)
    for key, pp in parts.items(): res['tables']['%s|%s|%s|%g' % key] = summ(pp)
    rng = np.random.default_rng(0); ns = len(seqs); draws = rng.integers(0, ns, (2000, ns)); Cm = C.draw_counts(draws, ns)
    for ds in ('TMOT', 'DT'):
        # per-sequence bootstrap: only the sequences of this dataset are drawn (indices remapped)
        names = sorted({n for (n, i) in seqs.items() if any(d['seq'][0] == i for d in parts[(ds, 'nf', 'tracker', 0.5)])}); loc = {seqs[n]: k for k, n in enumerate(names)}; nsd = len(names)
        dr = np.random.default_rng(0).integers(0, nsd, (2000, nsd)); Cd = C.draw_counts(dr, nsd); bo = {}
        def cat(m, key, part='nf', ay=0.5): return pool(parts[(ds, part, m, ay)], key)
        def sq(m, part='nf', ay=0.5): return np.array([loc[x] for x in pool(parts[(ds, part, m, ay)], 'seq')], int)
        okm = cat('tracker', 'ok').astype(bool); sv = sq('tracker')
        for m in ('body_v2', 'body_pushed', 'size'):
            dhm = cat(m, 'dh'); dht = cat('tracker', 'dh'); okk = okm & np.isfinite(dhm) & np.isfinite(dht)
            bo['%s-tracker|dh_p99' % m] = C.boot_pct(dhm, dht, sv, okk, nsd, Cd, 99)['diff']
            ta, sa = C.track_max(dict(dh=dhm, da=cat(m, 'da')), np.concatenate([np.full(len(p['ok']), k) for k, p in enumerate(parts[(ds, 'nf', m, 0.5)])]), sv, 'dh')
            tb, _ = C.track_max(dict(dh=dht, da=cat('tracker', 'da')), np.concatenate([np.full(len(p['ok']), k) for k, p in enumerate(parts[(ds, 'nf', 'tracker', 0.5)])]), sv, 'dh')
            bo['%s-tracker|tmax_median' % m] = C.boot_trackmax(ta, sa, tb, Cd, 50)['diff']
            bo['%s-tracker|aim' % m] = diff_ci(seq_boot_mean(cat(m, 'aim'), sv, okm, dr), seq_boot_mean(cat('tracker', 'aim'), sv, okm, dr))
            big = okm & (cat('tracker', 'trk_h') >= 100)
            ja = seq_boot_mean(cat(m, 'd2a') ** 2, sv, big, dr); jt = seq_boot_mean(cat('tracker', 'd2a') ** 2, sv, big, dr)
            bo['%s-tracker|aim_jitter2_big' % m] = diff_ci(ja, jt); bo['%s|tracker_jitter2_big' % m] = round(jt[0], 6); bo['n_big'] = int(big.sum())
            bo['%s-tracker|aim_jitter2_all' % m] = diff_ci(seq_boot_mean(cat(m, 'd2a') ** 2, sv, okm, dr), seq_boot_mean(cat('tracker', 'd2a') ** 2, sv, okm, dr)); bo['tracker_jitter2_all'] = round(seq_boot_mean(cat('tracker', 'd2a') ** 2, sv, okm, dr)[0], 6)
            fast = okm & (cat('tracker', 'spd') >= 0.1)
            bo['%s-tracker|aim_lag_fast' % m] = diff_ci(seq_boot_mean(cat(m, 'alag'), sv, fast, dr), seq_boot_mean(cat('tracker', 'alag'), sv, fast, dr))
        bo['body_v2-body_pushed|lh'] = diff_ci(seq_boot_mean(cat('body_v2', 'lh'), sv, okm, dr), seq_boot_mean(cat('body_pushed', 'lh'), sv, okm, dr))
        bo['body_v2-body_pushed|dh_p99'] = C.boot_pct(cat('body_v2', 'dh'), cat('body_pushed', 'dh'), sv, okm & np.isfinite(cat('body_v2', 'dh')) & np.isfinite(cat('body_pushed', 'dh')), nsd, Cd, 99)['diff']
        # switch stream: the switch frame and the next 2 frames
        svw = sq('tracker', 'sw'); m3 = cat('tracker', 'sw3', 'sw').astype(bool) & cat('tracker', 'ok', 'sw').astype(bool)
        for m in ('body_v2', 'body_pushed', 'size'):
            a_ = cat(m, 'dh', 'sw'); b_ = cat('tracker', 'dh', 'sw'); mm = m3 & np.isfinite(a_) & np.isfinite(b_)
            bo['%s-tracker|switch3_dh_p95' % m] = C.boot_pct(a_, b_, svw, mm, nsd, Cd, 95)['diff'] if mm.sum() >= 20 else None
        bo['n_switch3'] = int(m3.sum())
        res['boot'][ds] = bo
    # pass rules
    R_ = {}; T = res['tables']; Bo = res['boot']
    for ds in ('TMOT', 'DT'):
        g = lambda m, ay=0.5, part='nf': T['%s|%s|%s|%g' % (ds, part, m, ay)]; b = Bo[ds]
        R_['%s|B1' % ds] = b['body_v2-tracker|dh_p99'][2] < 0 and b['body_v2-tracker|tmax_median'][2] < 0
        R_['%s|B2' % ds] = b['body_v2-tracker|aim'][2] <= 0.003
        R_['%s|B3' % ds] = all(g('body_v2', ay)['aim_vs_conv_max'] < 1e-6 for ay in AIMS) and b['body_v2-tracker|aim_jitter2_big'][2] <= 0.04 * b['body_v2|tracker_jitter2_big']
        R_['%s|B4' % ds] = b['body_v2-tracker|aim_lag_fast'][1] >= -0.01
        bins_ok = all(0.9 <= v['h_tgt'][1] <= 1.1 for k, v in g('body_v2')['bins'].items() if k.startswith('both|') and v.get('h_tgt') and v['n_tgt'] >= 100)
        R_['%s|B5' % ds] = b['body_v2-body_pushed|lh'][2] <= 0.005 and bins_ok
        R_['%s|B6' % ds] = g('body_v2')['under_first']['share_lt090'] <= 0.05
        R_['%s|B7' % ds] = b['body_v2-tracker|switch3_dh_p95'] is not None and b['body_v2-tracker|switch3_dh_p95'][2] < 0
    res['pass'] = R_; res['seconds'] = round(time.time() - t0, 1)
    (B.B2 / 'results' / 'valB.json').write_text(B.jdump(res), encoding='utf-8'); (B.B2 / 'results' / 'tables_valB.md').write_text(md_valB(res), encoding='utf-8')
    print('PASS', json.dumps(R_), flush=True); print('VALB_DONE %.0f s' % (time.time() - t0), flush=True)


def f3(v, n=3): return '-' if v is None else ('%.' + str(n) + 'f') % v


def md_tables(T, prefix, methods, gt=True, aims=AIMS):
    L_ = []
    for ay in aims:
        L_ += ['', '#### %s, aim_y %g' % (prefix, ay), '', '| method | rows | ' + ('aim err | ' if gt else '') + 'aim jitter | aim step p99 | aim lag 0.03-0.1 | aim lag > 0.1 (n) | height step p99 | per-track largest | first-5 step | switch frame step mean / p95 / max (n) | switch +2 frames p95 (n) | first 1 s: share < 0.9 of conv (n) | fast growth: median, p10 (n) | ' + ('size lag (n) | abs log h | ' if gt else '') + 'aim vs conv max |',
               '|' + '---|' * (16 if gt else 13)]
        for m in methods:
            k = '%s|%s|%g' % (prefix, m, ay)
            if k not in T: continue
            s = T[k]; w = s['switch']; uf = s.get('under_first'); ug = s.get('under_grow')
            row = '| %s | %d | ' % (m, s['n_rows']) + ('%s | ' % f3(s.get('aim')) if gt else '') + '%s | %s | %s | %s (%d) | %s | %s | %s | %s / %s / %s (%d) | %s (%d) | %s | %s | ' % (
                f3(s['aim_jitter'], 4), f3(s['da_p99']), f3(s['aim_lag_0.03'], 4), f3(s['aim_lag_0.1'], 4), s['aim_lag_0.1_n'], f3(s['dh_p99']), f3(s['tmax']), f3(s['first5']),
                f3(w['mean']), f3(w['p95']), f3(w['max']), w['n'], f3(w['p95_3']), w['n3'], ('%s (%d)' % (f3(uf['share_lt090']), uf['n'])) if uf else '-',
                ('%s, %s (%d)' % (f3(ug['median']), f3(ug['p10']), ug['n'])) if ug else '-')
            if gt: row += '%s (%d) | %s | ' % (f3(s.get('size_lag'), 4), s.get('size_lag_n', 0), f3(s.get('lh'), 4))
            row += '%s |' % (('%.1e' % s['aim_vs_conv_max']) if s.get('aim_vs_conv_max') is not None else '-'); L_.append(row)
        if ay == 0.5:
            keys = sorted({b for m in methods if '%s|%s|%g' % (prefix, m, ay) in T for b in T['%s|%s|%g' % (prefix, m, ay)]['bins']}, key=lambda x: (x.split('|')[0], int(x.split('|')[1].split('-')[0])))
            L_ += ['', 'height / tracker-row height median [10 %, 90 %]; |centre x - tracker centre x| / tracker h median, p90' + ('; height / target median' if gt else '') + ' — by detector and tracker-row height (px):', '',
                   '| detector, height | n | ' + ' | '.join(methods) + ' |', '|---|---|' + '---|' * len(methods)]
            for b in keys:
                cells = []
                for m in methods:
                    k = '%s|%s|%g' % (prefix, m, ay); v = T.get(k, {}).get('bins', {}).get(b)
                    cells.append('-' if v is None else '%.2f [%.2f, %.2f]; dx %.3f, %.3f%s' % (v['h_trk'][1], v['h_trk'][0], v['h_trk'][2], v['dx'][0], v['dx'][1], ('; tgt %.2f' % v['h_tgt'][1]) if gt and v.get('h_tgt') else ''))
                n = next(T['%s|%s|%g' % (prefix, m, ay)]['bins'][b]['n'] for m in methods if b in T.get('%s|%s|%g' % (prefix, m, ay), {}).get('bins', {}))
                L_.append('| %s | %d | ' % (b.replace('|', ', '), n) + ' | '.join(cells) + ' |')
    return L_


def md_valB(res):
    L_ = ['# body v2: val-B (once; val-B was already looked at about seven times in this line of work: weak evidence)', '', 'euro %s; hand-overs %s; switch transitions %s' % (res['euro'], json.dumps(res['handovers']), json.dumps(res['switches'])), '',
          '## pass rules (PREREG_BODY_V2.json)', '', '```', json.dumps(res['pass'], indent=1), '```', '', '## bootstrap (per sequence, 2000 draws, seed 0; [point, 5 %, 95 %])', '', '```', json.dumps(res['boot'], indent=1), '```']
    for ds in ('TMOT', 'DT'):
        for part in ('nf', 'sw'):
            L_ += ['', '### %s, %s' % (ds, part)] + md_tables(res['tables'], '%s|%s' % (ds, part), METHODS)
    return '\n'.join(L_)


# ------------------------------------------------------------------------------------------------ device-rate recordings once
def plain_stream(rec):
    """the plain YOLO26x detections (b2_yolo.py rec, body_v2/data/det/yolo_<rec>.npz) through the server ByteTrack -> one stream like g2_device.streams"""
    import g2_device as GD
    fp = B.B2 / 'data' / 'det' / ('yolo_%s.npz' % rec)
    if not fp.exists(): return None
    d = P.load_det(fp); g = np.load(GD.D / 'gmc' / ('%s.npz' % rec)); W = g['warp'].astype(np.float64); ts = g['times_s'].astype(np.float64); n = d['n']; fps = (n - 1) / (ts[-1] - ts[0])
    per = [np.flatnonzero(d['fi'] == f) for f in range(n)]
    fd = [(np.c_[d['box'][per[f]], d['score'][per[f]], np.zeros(len(per[f]))], per[f], np.zeros(len(per[f]), int)) for f in range(n)]
    rows = np.array(GD.TR.run(n, W, fd), np.float64).reshape(-1, 8); ids = defaultdict(list)
    for r in rows: ids[int(r[1])].append(r)
    E_ = {}
    for i, rr in ids.items():
        rr = np.array(rr); rr = rr[np.argsort(rr[:, 0])]; di = rr[:, 7].astype(np.int64)
        E_[i] = dict(f=rr[:, 0].astype(np.int64), z=P.xyxy2cxcywh(d['box'][di]), trk=P.xyxy2cxcywh(rr[:, 2:6]), src=np.zeros(len(rr), np.int64))
    return dict(ids=E_, W=W, n=n, fps=fps, times=ts)


def device(mc, beta, recs):
    import g2_device as GD
    t0 = time.time(); t1, t2 = tables(); parts = defaultdict(list); info = {}; plain_info = {}
    for ri, rec in enumerate(recs):
        S, info[rec] = GD.streams(rec, ri); pl = plain_stream(rec)
        if pl is not None: S['plain'] = pl; plain_info[rec] = dict(rows=int(sum(len(e['f']) for e in pl['ids'].values())), ids=len(pl['ids']))
        for st, inp in S.items():
            ids = {tid: dict(f=e['f'], z=e['z'], trk=e['trk'], src=e['src']) for tid, e in inp['ids'].items()}
            keys = [None] * inp['n']
            for e in ids.values():
                for f, s_ in zip(e['f'], e['src']): keys[int(f)] = ('yolo26x_flir' if st == 'plain' else DK[int(s_)])
            last = 'far' if st == 'far' else ('yolo26x_flir' if st == 'plain' else 'near')
            for f in range(inp['n']):
                if keys[f] is None: keys[f] = last
                last = keys[f]
            gp = dict(ids=ids, by_f=items_of(ids), W=inp['W'], n=inp['n'], keys=keys, fps=inp['fps'])
            fam = 'person' if st == 'plain' else 'pose'
            for ay in AIMS:
                rm = run_methods(gp, t1, t2, mc, beta, ay, None if ay == 0.5 else AIM_DEP, times=inp['times'], family=fam); kk, tt = rm['conv'][1], rm['conv'][2]
                for m in (METHODS if ay == 0.5 else AIM_DEP):
                    o = {tid: e['trk'] for tid, e in ids.items()} if m == 'tracker' else rm[m][0]
                    cv = rm['conv'][0] if m == 'body_v2' else rm['conv1'][0] if m == 'body_pushed' and 'conv1' in rm else None
                    tk = t2 if m in ('body_v2', 'conv') else t1 if m in ('body_pushed', 'conv1') else None
                    for tid, e in ids.items():
                        ref = own_ref(tk, e['z'], fam if m in ('body_v2', 'conv') else 'pose') if tk is not None else e['z']
                        d = per_id(e, o[tid], ref, cv[tid] if cv is not None else None, kk[tid], tt[tid], ay, gt=False)
                        parts[(st if st == 'plain' else 'pose_' + st, m, ay)].append(d); parts[(rec + '|' + st, m, ay)].append(d)
                        if st in ('near', 'far'): parts[('pose_nf', m, ay)].append(d)
            print('REC %s %s (%.0f s)' % (rec, st, time.time() - t0), flush=True)
    res = dict(rules=PR['pass_rules']['device'], euro=[mc, beta, 1.0], recordings=info, plain=plain_info, tables={})
    for key, pp in parts.items(): res['tables']['%s|%s|%g' % key] = summ(pp, gt=False)
    T = res['tables']; R_ = {}
    for st in ('pose_near', 'pose_far'):
        b = T['%s|body_v2|0.5' % st]['bins']; cl = [v for k, v in b.items() if k.startswith('both|') and int(k.split('|')[1].split('-')[0]) >= 170]
        R_['D1|%s' % st] = dict(ok=all(0.98 <= v['h_trk'][1] <= 1.02 and v['dx'][0] <= 0.01 for v in cl) and len(cl) > 0, bins={k: (v['n'], v['h_trk'][1], v['dx'][0]) for k, v in b.items() if k.startswith('both|')})
    g = lambda m, ay=0.5: T['pose_nf|%s|%g' % (m, ay)]
    R_['D2'] = g('body_v2')['dh_p99'] <= g('tracker')['dh_p99']
    R_['D3'] = all(T['pose_nf|body_v2|%g' % ay]['aim_vs_conv_max'] < 1e-6 for ay in AIMS) and g('body_v2')['aim_jitter'] <= 1.02 * g('tracker')['aim_jitter']
    R_['D4'] = g('body_v2')['under_first']['share_lt090'] <= 0.05
    res['pass'] = R_; res['seconds'] = round(time.time() - t0, 1)
    (B.B2 / 'results' / 'device.json').write_text(B.jdump(res), encoding='utf-8')
    L_ = ['# body v2: device-rate recordings (once; no ground truth; numbers relative to the tracker-row height; these recordings were used before: weak evidence)', '',
          'recordings %s; plain detector (YOLO26x FLIR fp32 on the PC) streams %s' % (json.dumps(info), json.dumps(plain_info)), '', '## pass rules', '', '```', json.dumps(R_, indent=1), '```']
    for st in ('pose_nf', 'pose_sw', 'pose_near', 'pose_far', 'plain'):
        if '%s|tracker|0.5' % st in T: L_ += ['', '### stream %s' % st] + md_tables(T, st, METHODS, gt=False, aims=AIMS if st == 'pose_nf' else (0.5,))
    (B.B2 / 'results' / 'tables_device.md').write_text('\n'.join(L_), encoding='utf-8'); print('PASS', json.dumps({k: (v if isinstance(v, bool) else v['ok']) for k, v in R_.items()}), flush=True)
    print('DEVICE_DONE %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'valA': valA(int(sys.argv[2]))
    elif sys.argv[1] == 'valB': valB(float(sys.argv[2]), float(sys.argv[3]))
    elif sys.argv[1] == 'device': device(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:])
