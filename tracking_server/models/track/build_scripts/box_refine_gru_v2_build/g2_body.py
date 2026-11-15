"""gru_v2/scripts/g2_body.py - numbers of the SERVER modes 'body' (lookup conversion of the tracker row + size-only One Euro) and 'gru2' against the tracker row.
Both modes run through tracking_server/src/track/box_refine.Refiner frame by frame, exactly as the server does (hand-over on), with the lookup table / weights of <npz>.
val-B (public data, already looked at several times in this line of work; nothing is chosen here) and the device-rate recordings (no ground truth).
Per method: height / tracker-row height by tracker-row height bin and detector, height steps (p99, per-track largest), aim-point jitter, aim step, lag against the
current detection in the method's own box definition (tracker row: raw detection; body / gru2: lookup-converted detection), first rows of a track, the step at a
detector switch (near <-> far stream). val-B numbers are relative to the target height, device numbers to the tracker-row height (as g2_eval / g2_device).
Usage: python g2_body.py <tracking_server dir> <npz> <rows_valB.pkl> <out stem> [device rec ...]   -> results/<stem>.json, results/tables_<stem>.md
"""
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

import g2_lib as L

sys.path.insert(0, sys.argv[1])
from src.track import box_refine as BR   # noqa: E402

# body = the mode as it ships (top anchor, size filter 0.4 Hz beta 0). For comparison only: body_c = the same with the converted CENTRE kept instead of the top edge,
# body5 = top anchor with the size filter of the 'size' mode (5 Hz, beta 3).
METHODS = ('tracker', 'body', 'body_c', 'body5', 'gru2')
RUNS = (('body', 'body', None), ('body_c', 'body', dict(anchor='centre')), ('body5', 'body', dict(mc_s=5.0, beta_s=3.0, dcut=1.0)), ('gru2', 'gru2', None))
HB = ((0, 40), (40, 70), (70, 100), (100, 170), (170, 240), (240, 9999))
BINS = ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 99))
DK = ('near', 'far')


def server(model, mode, ids, W, n, times, fps, body=None):
    """ids {tid: dict(f, z, kp, trk, src)} -> per tid out [n,4], k [n] (rows since the state started)."""
    by_f = defaultdict(list)
    for tid in sorted(ids):
        for j, f in enumerate(ids[tid]['f']): by_f[int(f)].append((tid, j))
    r = BR.Refiner(mode, model, body=body); o = {tid: np.zeros((len(e['f']), 4)) for tid, e in ids.items()}; kk = {tid: np.zeros(len(e['f']), np.int64) for tid, e in ids.items()}
    for f in range(n):
        its = by_f.get(f, ()); H = W[f] if (W is not None and 0 < f < len(W)) else None; t = float(times[f]) if times is not None else f / fps
        dk = DK[int(ids[its[0][0]]['src'][its[0][1]])] if its else None
        res = r.step([(a, ids[a]['z'][j], ids[a]['kp'][j], ids[a]['trk'][j]) for a, j in its], t, H, dk, True)
        for a, j in its: o[a][j] = res[a]; kk[a][j] = r.tr[a]['k']
    return o, kk, r.handovers


def conv_soft(model, z):
    z = np.asarray(z, np.float64); z = np.c_[z[:, :2], np.maximum(z[:, 2:], 1.)]; e = model.conv_delta(z)
    return np.stack([z[:, 0] + e[:, 0] * z[:, 3], z[:, 1] + e[:, 1] * z[:, 3], z[:, 2] * np.exp(e[:, 2]), z[:, 3] * np.exp(e[:, 3])], 1)


def hbins(trk_h, src, ratio):
    out = {}
    for s, nm in ((None, 'both'), (0, 'near'), (1, 'far')):
        for a, b in HB:
            m = (trk_h >= a) & (trk_h < b) & ((src == s) if s is not None else True)
            if m.sum() >= 20: out['%s|%d-%d' % (nm, a, b)] = dict(n=int(m.sum()), med=round(float(np.median(ratio[m])), 3), q10=round(float(np.percentile(ratio[m], 10)), 3), q90=round(float(np.percentile(ratio[m], 90)), 3))
    return out


# ------------------------------------------------------------------------------------------------ val-B
def valb(model, rows_path):
    d = pickle.loads(open(rows_path, 'rb').read()); recs, Wd, nd = d['recs'], d['W'], d['n']; groups = defaultdict(list); t0 = time.time(); hand = defaultdict(int)
    for r in recs: groups[(r['name'], r['stream'])].append(r)
    L.CONVERTED = tuple(L.CONVERTED) + ('body', 'body_c', 'body5', 'gru2')
    for key, rr in sorted(groups.items()):
        ids = {r['id']: dict(f=r['e']['f'], z=r['e']['z'], kp=r['e']['kp'], trk=r['o']['tracker'], src=r['e']['src']) for r in rr}
        for name, mode, bp in RUNS:
            o, kk, nh = server(model, mode, ids, Wd[key[0]], nd[key[0]], None, rr[0]['fps'], bp); hand[name] += nh
            for r in rr:
                r['o'][name] = o[r['id']]
                if name == 'body': r['e']['k'] = kk[r['id']]      # rows since the SERVER state started (Refiner rule + hand-over), for the first-rows numbers of every method
        for r in rr: r['e']['conv'] = conv_soft(model, r['e']['z'])   # the detection in the converted definition (this file's table), the lag reference of body / gru2
    print('VALB server runs done (%.0f s), hand-overs %s' % (time.time() - t0, dict(hand)), flush=True)
    evs = [L.events(r) for r in recs]; res = dict(handovers=dict(hand), tables={})
    for ds in ('TMOT', 'DT'):
        for part, streams in (('nf', ('near', 'far')), ('sw', ('sw',))):
            idx = [i for i, r in enumerate(recs) if r['ds'] == ds and r['stream'] in streams]; T = dict(n_rows=int(sum(len(recs[i]['e']['f']) for i in idx)), methods={})
            for m in METHODS:
                rows, seqv, idv, ev, sbv, names = L.gather([recs[i] for i in idx], m, [evs[i] for i in idx]); s = L.summary(rows, ev, idv, sbv); ok = rows['ok']
                trk_h = np.concatenate([recs[i]['o']['tracker'][:, 3] for i in idx]); src = np.concatenate([recs[i]['e']['src'] for i in idx]); oh = np.concatenate([recs[i]['o'][m][:, 3] for i in idx])
                ow = np.concatenate([recs[i]['o'][m][:, 2] for i in idx]); trk_w = np.concatenate([recs[i]['o']['tracker'][:, 2] for i in idx]); tg = np.concatenate([recs[i]['e']['tgt'][:, 3] for i in idx])
                sw0 = ev['switch0'] & ok & np.isfinite(rows['dh']); j8 = []
                for i in idx:   # aim-point jitter with the device's aim point (top + 0.08 x height) instead of 0.37, same rows and normalisation as d2a
                    e = recs[i]['e']; o = recs[i]['o'][m]; f = e['f']; g = e['tgt']; okr = np.isfinite(g[:, 0]); c1 = np.r_[False, np.diff(f) == 1]; i2 = np.flatnonzero(np.r_[False, c1[:-1]] & c1 & okr)
                    a8 = np.stack([o[:, 0], o[:, 1] - o[:, 3] / 2 + 0.08 * o[:, 3]], 1); j8.append(np.hypot(*(a8[i2] - 2 * a8[i2 - 1] + a8[i2 - 2]).T) / g[i2, 3])
                j8 = np.concatenate(j8)
                T['methods'][m] = dict(aim=s['aim'], ce=s['ce'], aim_jitter=s['d2a_rms'], aim_jitter_008=float(np.sqrt(np.mean(j8 ** 2))), dh_p99=s['dh_p99'], dh_p95=s['dh_p95'], tmax=s['dh_trackmax_med'], da_p99=s['da_p99'], first5=s['first5']['dh_mean'], first5_n=s['first5']['n'],
                                       row1_over=s['first5']['row1_over_015'], lag_mid=s['lag_0.03-0.1']['along_det'], lag_fast=s['lag_0.1-99']['along_det'], aim_lag_mid=s['lag_0.03-0.1']['aim_det'], aim_lag_fast=s['lag_0.1-99']['aim_det'],
                                       n_fast=s['lag_0.1-99']['n'], size_lag=s['slag_0.03-99']['salong'],
                                       switch=dict(n=int(sw0.sum()), dh_mean=float(rows['dh'][sw0].mean()) if sw0.any() else None, dh_p95=float(np.percentile(rows['dh'][sw0], 95)) if sw0.sum() >= 5 else None,
                                                   dh_max=float(rows['dh'][sw0].max()) if sw0.any() else None, da_mean=float(rows['da'][sw0].mean()) if sw0.any() else None),
                                       switch1s=s['events'].get('switch1s'), h_over_tracker=hbins(trk_h, src, oh / np.maximum(trk_h, 1.)), w_over_tracker=hbins(trk_h, src, ow / np.maximum(trk_w, 1.)),
                                       h_over_target=hbins(trk_h[ok], src[ok], oh[ok] / tg[ok]))
            res['tables']['%s|%s' % (ds, part)] = T; print('VALB table %s %s rows %d (%.0f s)' % (ds, part, T['n_rows'], time.time() - t0), flush=True)
    return res


# ------------------------------------------------------------------------------------------------ device-rate recordings
def per_id(e, o, ref, kk):
    f = e['f']; n = len(f); hr = np.maximum(e['trk'][:, 3], 1.); a = L.aim_of(o); z = e['z']
    c1 = np.r_[False, np.diff(f) == 1]; c2 = c1 & np.r_[False, c1[:-1]]; c3 = c2 & np.r_[False, c2[:-1]]
    dh = np.full(n, np.nan); da = np.full(n, np.nan); d2a = np.full(n, np.nan); i1 = np.flatnonzero(c1); i2 = np.flatnonzero(c2)
    dh[i1] = np.abs(o[i1, 3] - o[i1 - 1, 3]) / hr[i1]; da[i1] = np.hypot(*(a[i1] - a[i1 - 1]).T) / hr[i1]; d2a[i2] = np.hypot(*(a[i2] - 2 * a[i2 - 1] + a[i2 - 2]).T) / hr[i2]
    a8 = np.stack([o[:, 0], o[:, 1] - o[:, 3] / 2 + 0.08 * o[:, 3]], 1); d2a8 = np.full(n, np.nan); d2a8[i2] = np.hypot(*(a8[i2] - 2 * a8[i2 - 1] + a8[i2 - 2]).T) / hr[i2]   # the device's gimbal aims at 0.08 of the height
    spd = np.full(n, np.nan); lag = np.full(n, np.nan); alag = np.full(n, np.nan); i3 = np.flatnonzero(c3)
    if len(i3):
        v = (z[i3, :2] - z[i3 - 3, :2]) / 3.; sp = np.hypot(*v.T); u = v / np.maximum(sp, 1e-9)[:, None]; spd[i3] = sp / hr[i3]; lag[i3] = np.sum((o[i3, :2] - ref[i3, :2]) * u, 1) / hr[i3]
        alag[i3] = np.sum((a[i3] - L.aim_of(ref)[i3]) * u, 1) / hr[i3]
    sw = np.r_[False, (e['src'][1:] != e['src'][:-1])] & c1
    return dict(dh=dh, da=da, d2a=d2a, d2a8=d2a8, spd=spd, lag=lag, alag=alag, first5=(kk >= 1) & (kk <= 4), sw=sw, hrel=o[:, 3] / hr, wrel=o[:, 2] / np.maximum(e['trk'][:, 2], 1.), trk_h=e['trk'][:, 3], src=e['src'])


def summarize(parts):
    cat = lambda k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)
    dh, da, d2a, d2a8 = cat('dh'), cat('da'), cat('d2a'), cat('d2a8'); fin = lambda v: v[np.isfinite(v)]; q = lambda v, p: round(float(np.percentile(fin(v), p)), 5) if len(fin(v)) else None
    s = dict(n_rows=int(np.isfinite(dh).sum()), dh_p95=q(dh, 95), dh_p99=q(dh, 99), dh_max=q(dh, 100), da_p99=q(da, 99), aim_jitter=round(float(np.sqrt(np.mean(fin(d2a) ** 2))), 5),
             aim_jitter_008=round(float(np.sqrt(np.mean(fin(d2a8) ** 2))), 5))
    tm = [np.nanmax(p['dh']) for p in parts if np.isfinite(p['dh']).sum() >= 10]; s['tmax'] = round(float(np.median(tm)), 5) if tm else None; s['n_tracks'] = len(tm)
    spd, lag, alag = cat('spd'), cat('lag'), cat('alag')
    for a, b in BINS:
        mm = np.isfinite(spd) & (spd >= a) & (spd < b); s['lag_%g-%g' % (a, b)] = dict(n=int(mm.sum()), centre=round(float(np.mean(lag[mm])), 5) if mm.any() else None, aim=round(float(np.mean(alag[mm])), 5) if mm.any() else None)
    f5 = cat('first5').astype(bool) & np.isfinite(dh); s['first5'] = dict(n=int(f5.sum()), dh_mean=round(float(dh[f5].mean()), 5) if f5.any() else None, dh_max=round(float(dh[f5].max()), 5) if f5.any() else None)
    sw = cat('sw').astype(bool) & np.isfinite(dh); s['switch'] = dict(n=int(sw.sum()), dh_mean=round(float(dh[sw].mean()), 5) if sw.any() else None, dh_p95=round(float(np.percentile(dh[sw], 95)), 5) if sw.sum() >= 5 else None,
                                                                     dh_max=round(float(dh[sw].max()), 5) if sw.any() else None, da_mean=round(float(da[sw].mean()), 5) if sw.any() else None)
    s['h_over_tracker'] = hbins(cat('trk_h'), cat('src'), cat('hrel')); s['w_over_tracker'] = hbins(cat('trk_h'), cat('src'), cat('wrel'))
    return s


def device(model, recs):
    import g2_device as GD
    allp = defaultdict(lambda: defaultdict(list)); info = {}; hand = defaultdict(int)
    for ri, rec in enumerate(recs):
        S, info[rec] = GD.streams(rec, ri)
        for st, inp in S.items():
            o = {'tracker': {i: e['trk'] for i, e in inp['ids'].items()}}; kk = None
            for name, mode, bp in RUNS:
                o[name], k_, nh = server(model, mode, inp['ids'], inp['W'], inp['n'], inp['times'], inp['fps'], bp); hand[name] += nh
                if name == 'body': kk = k_
            for tid, e in inp['ids'].items():
                cv = conv_soft(model, e['z'])
                for m in METHODS:
                    p = per_id(e, o[m][tid], e['z'] if m == 'tracker' else cv, kk[tid]); allp[st][m].append(p); allp['all'][m].append(p)
    return dict(recordings=info, handovers=dict(hand), pooled={st: {m: summarize(v) for m, v in pm.items()} for st, pm in allp.items()})


# ------------------------------------------------------------------------------------------------ tables
def f_(v, n=3): return '-' if v is None else ('%+.' + str(n) + 'f') % v if n < 0 else ('%.' + str(n) + 'f') % v


def sg(v, n=3): return '-' if v is None else ('%+.' + str(n) + 'f') % v


def md(res):
    Lm = ['# body / gru2 (server modes) against the tracker row', '', 'file %s sha256 %s; body size filter %s; gru2 size filter %s' % (res['npz'], res['sha256'], res['body_euro'], res['gru2_size']), '']
    if 'valB' in res:
        Lm += ['## val-B (relative to the target height; val-B was already looked at several times, nothing is chosen here)', '', 'hand-overs in the server runs: %s' % json.dumps(res['valB']['handovers']), '']
        for key, T in res['valB']['tables'].items():
            Lm += ['### %s (%d rows)' % (key, T['n_rows']), '', '| method | aim err | aim jitter (aim at 0.37) | aim jitter (aim at 0.08) | aim step p99 | height step p99 | per-track largest | first-5 rows step (n) | lag 0.03-0.1 | lag > 0.1 (n) | aim lag > 0.1 | size lag | switch frame step mean / p95 / max (n) |', '|' + '---|' * 13]
            for m in METHODS:
                s = T['methods'][m]; w = s['switch']
                Lm.append('| %s | %.3f | %.4f | %.4f | %.3f | %.3f | %.3f | %s (%d) | %s | %s (%d) | %s | %s | %s / %s / %s (%d) |' % (m, s['aim'], s['aim_jitter'], s['aim_jitter_008'], s['da_p99'], s['dh_p99'], s['tmax'], f_(s['first5']), s['first5_n'], sg(s['lag_mid']), sg(s['lag_fast']), s['n_fast'],
                          sg(s['aim_lag_fast']), sg(s['size_lag']), f_(w['dh_mean']), f_(w['dh_p95']), f_(w['dh_max']), w['n']))
            for nm, kk in (('height / tracker-row height', 'h_over_tracker'), ('width / tracker-row width', 'w_over_tracker'), ('height / target height', 'h_over_target')):
                keys = sorted(T['methods']['body'][kk], key=lambda k: (k.split('|')[0], int(k.split('|')[1].split('-')[0])))
                Lm += ['', '%s, median [10 %%, 90 %%] by detector and tracker-row height (px):' % nm, '', '| detector, height | n | ' + ' | '.join(METHODS) + ' |', '|---|---|' + '---|' * len(METHODS)]
                for k in keys: Lm.append('| %s | %d | ' % (k.replace('|', ', '), T['methods']['body'][kk][k]['n']) + ' | '.join('%.2f [%.2f, %.2f]' % (T['methods'][m][kk][k]['med'], T['methods'][m][kk][k]['q10'], T['methods'][m][kk][k]['q90']) if k in T['methods'][m][kk] else '-' for m in METHODS) + ' |')
            Lm.append('')
    if 'device' in res:
        Lm += ['## device-rate recordings (no ground truth; relative to the tracker-row height)', '', 'recordings: %s; hand-overs %s' % (json.dumps(res['device']['recordings']), json.dumps(res['device']['handovers'])), '']
        for st, pm in res['device']['pooled'].items():
            Lm += ['### stream %s' % st, '', '| method | height step p99 | per-track largest | aim step p99 | aim jitter (aim at 0.37) | aim jitter (aim at 0.08) | lag 0.03-0.1 (n) | aim lag 0.03-0.1 | lag > 0.1 (n) | first-5 rows step mean / max (n) | switch frame step mean / p95 / max (n) |', '|' + '---|' * 11]
            for m in METHODS:
                s = pm[m]; w = s['switch']
                Lm.append('| %s | %.4f | %s | %.4f | %.4f | %.4f | %s (%d) | %s | %s (%d) | %s / %s (%d) | %s / %s / %s (%d) |' % (m, s['dh_p99'], f_(s['tmax'], 4), s['da_p99'], s['aim_jitter'], s['aim_jitter_008'], sg(s['lag_0.03-0.1']['centre'], 4), s['lag_0.03-0.1']['n'], sg(s['lag_0.03-0.1']['aim'], 4),
                          sg(s['lag_0.1-99']['centre'], 4), s['lag_0.1-99']['n'], f_(s['first5']['dh_mean'], 4), f_(s['first5']['dh_max'], 4), s['first5']['n'], f_(w['dh_mean'], 4), f_(w['dh_p95'], 4), f_(w['dh_max'], 4), w['n']))
            for nm, kk in (('height / tracker-row height', 'h_over_tracker'), ('width / tracker-row width', 'w_over_tracker')):
                keys = sorted(pm['body'][kk], key=lambda k: (k.split('|')[0], int(k.split('|')[1].split('-')[0])))
                Lm += ['', '%s, median [10 %%, 90 %%]:' % nm, '', '| detector, height | n | ' + ' | '.join(METHODS) + ' |', '|---|---|' + '---|' * len(METHODS)]
                for k in keys: Lm.append('| %s | %d | ' % (k.replace('|', ', '), pm['body'][kk][k]['n']) + ' | '.join('%.2f [%.2f, %.2f]' % (pm[m][kk][k]['med'], pm[m][kk][k]['q10'], pm[m][kk][k]['q90']) if k in pm[m][kk] else '-' for m in METHODS) + ' |')
            Lm.append('')
    return '\n'.join(Lm)


def main():
    npz, rows_path, stem, recs = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]; z = np.load(npz, allow_pickle=False); model = BR.G2Model({k: z[k] for k in z.files})
    res = dict(npz=npz.replace('\\', '/').split('/')[-1], sha256=hashlib.sha256(open(npz, 'rb').read()).hexdigest(), body_euro=dict(BR.BODY, **(model.body or {})), gru2_size=list(model.size))
    if rows_path != '-': res['valB'] = valb(model, rows_path)
    if recs: res['device'] = device(model, recs)
    (L.G2 / 'results' / (stem + '.json')).write_text(json.dumps(res, indent=1)); (L.G2 / 'results' / ('tables_%s.md' % stem)).write_text(md(res), encoding='utf-8'); print('BODY_DONE', stem, flush=True)


if __name__ == '__main__':
    main()
