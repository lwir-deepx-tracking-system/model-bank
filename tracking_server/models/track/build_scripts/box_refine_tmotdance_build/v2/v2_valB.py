"""v2/scripts/v2_valB.py - the ONE val-B run of box_refine v2 (PREREG_V2.json valB_once): raw, tracker, euro, gru, combo_a, size (val-A choice,
taken from the server module's SIZE, which must equal v2/results/size_valA.json choice). Per dataset: nf (near + far pooled) and sw.
Per-sequence bootstrap 90 % intervals (2000 draws, seed 0). Writes v2/results/size_valB.json and v2/results/tables_v2.md.
Usage: python v2_valB.py [valB|valA] [workers]   (valA = code check only, writes *_check_valA.*)
"""
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

import v2_lib as L

METHODS = ('raw', 'tracker', 'euro', 'gru', 'combo_a', 'size')
NAMES = dict(raw='raw detector', tracker='tracker row (today)', euro='One Euro (server euro)', gru='GRU (server gru)', combo_a='GRU + size One Euro (combo a)',
             size='size only (new)')
PAIRS = (('size', 'tracker'), ('size', 'euro'), ('size', 'gru'), ('size', 'combo_a'), ('size', 'raw'), ('euro', 'tracker'), ('gru', 'tracker'), ('combo_a', 'tracker'))
G = {}


def init(): G['m'] = L.gru_model()


def job(a):
    r, st = a; inp = L.stream_inputs(r, st); outs = {}
    _, kk = L.run(inp, 'off')
    outs['euro'], _ = L.run(inp, 'euro'); outs['gru'], _ = L.run(inp, 'gru', G['m']); outs['combo_a'], _ = L.run(inp, 'gru', G['m'], ca=L.COMBO_A)
    outs['size'], k2 = L.run(inp, 'size', size=L.BRM.SIZE)
    assert all((k2[t] == kk[t]).all() for t in kk)
    recs = L.recs_of(r, st, inp, outs, kk)
    return recs, [L.events(x) for x in recs]


def r5(x): return None if x is None else round(float(x), 5)


def table(recs, evs, methods):
    names = sorted({r['name'] for r in recs}); ns = len(names)
    draws = np.random.default_rng(0).integers(0, ns, (2000, ns)); Cc = L.C.draw_counts(draws, ns); Rm = {}
    for m in methods:
        rows, seqv, idv, ev, sb, _ = L.gather(recs, [r['o'][m] for r in recs], evs); Rm[m] = rows
    ok = Rm['raw']['ok']; fin = lambda m, k: np.where(ok, Rm[m][k], np.nan)
    res = dict(sequences=names, n_rows=int(ok.sum()), n_ids=len(recs), methods={}, diffs={})
    for m in methods:
        s = L.summary(Rm[m], ev, idv); rows = Rm[m]
        s['boot'] = {k: L.RE.boot_mean(fin(m, k), seqv, ns, draws) for k in ('ce', 'aim', 'lh', 'iou')}
        for k in ('d2c', 'd2a', 'd2h'):
            b = L.RE.boot_mean(np.where(ok, rows[k] ** 2, np.nan), seqv, ns, draws); s['boot'][k + '_rms'] = [round(math.sqrt(max(x, 0)), 5) for x in b[:3]] + [b[3]]
        s['boot']['dh_p99'] = L.C.boot_pct(rows['dh'], None, seqv, ok, ns, Cc, 99)['a']; s['boot']['da_p99'] = L.C.boot_pct(rows['da'], None, seqv, ok, ns, Cc, 99)['a']
        res['methods'][m] = s
    tv = {m: L.C.track_max(Rm[m], idv, seqv, 'dh') for m in methods}
    for a, b in PAIRS:
        if a not in methods or b not in methods: continue
        A, B = Rm[a], Rm[b]; d = {}
        for k in ('ce', 'aim', 'lh', 'iou'): d[k] = L.RE.boot_diff(fin(a, k), fin(b, k), seqv, ns, draws)
        for k in ('d2c', 'd2a', 'd2h'): d[k + '_sq'] = L.RE.boot_diff(np.where(ok, A[k] ** 2, np.nan), np.where(ok, B[k] ** 2, np.nan), seqv, ns, draws)
        for k, q in (('dh', 95), ('dh', 99), ('da', 95), ('da', 99)): d['%s_p%d' % (k, q)] = L.C.boot_pct(A[k], B[k], seqv, ok, ns, Cc, q)['diff']
        d['dh_trackmax_med'] = L.C.boot_trackmax(tv[a][0], tv[a][1], tv[b][0], Cc, 50)['diff']
        d['dh_trackmax_p90'] = L.C.boot_trackmax(tv[a][0], tv[a][1], tv[b][0], Cc, 90)['diff']
        for lo, hi in L.SPEED_BINS:
            mm = ok & (A['spd'] >= lo) & (A['spd'] < hi)
            for k in ('along', 'along_det', 'aim_det', 'aim_t', 'ce'):
                d['%s_%g-%g' % (k, lo, hi)] = L.RE.boot_diff(np.where(mm, A[k], np.nan), np.where(mm, B[k], np.nan), seqv, ns, draws)
        mm = ok & (A['sspd'] >= 0.03)
        for k in ('salong', 'sdet'): d['%s_0.03-99' % k] = L.RE.boot_diff(np.where(mm, A[k], np.nan), np.where(mm, B[k], np.nan), seqv, ns, draws)
        for evk, q in (('start', 95), ('start5', 95), ('switch0', 95), ('switch1s', 99), ('flip', 95), ('size', 95), ('size70', 95), ('size50', 95), ('later', 99)):
            mm = ok & ev[evk]
            if (mm & np.isfinite(A['dh'])).sum() < 5: continue
            d['ev_%s_dh_p%d' % (evk, q)] = L.C.boot_pct(A['dh'], B['dh'], seqv, mm, ns, Cc, q)['diff']
            d['ev_%s_da_p%d' % (evk, q)] = L.C.boot_pct(A['da'], B['da'], seqv, mm, ns, Cc, q)['diff']
        d['ev_start_dh_mean'] = L.RE.boot_diff(np.where(ok & ev['start'], A['dh'], np.nan), np.where(ok & ev['start'], B['dh'], np.nan), seqv, ns, draws)
        res['diffs']['%s - %s' % (a, b)] = d
    return res


def decide(T):
    """PREREG_V2 pass_rules_size_vs_tracker, both datasets."""
    out = {}; allok = True
    for ds in ('TMOT', 'DT'):
        nf = T['%s|nf' % ds]; sw = T['%s|sw' % ds]; d = nf['diffs']['size - tracker']; dsw = sw['diffs']['size - tracker']
        t2 = nf['methods']['tracker']['d2a_rms'] ** 2
        c = dict(P1_dh_p99=d['dh_p99'][2] < 0, P2_trackmax=d['dh_trackmax_med'][2] < 0, P3_aim=d['aim'][2] <= 0.003, P4_aim_jitter=d['d2a_sq'][2] <= 0.04 * t2,
                 P5_lh=d['lh'][2] <= 0.005, P6_aim_lag_det_fast=d['aim_det_0.1-99'] is not None and d['aim_det_0.1-99'][1] >= -0.01,
                 P7_switch1s_dh_p99=('ev_switch1s_dh_p99' in dsw) and dsw['ev_switch1s_dh_p99'][2] < 0)
        out[ds] = dict(c, values=dict(dh_p99=d['dh_p99'], trackmax=d['dh_trackmax_med'], aim=d['aim'], d2a_sq=d['d2a_sq'], d2a_sq_limit=round(0.04 * t2, 6), lh=d['lh'],
                                      aim_det_fast=d['aim_det_0.1-99'], switch1s_dh_p99=dsw.get('ev_switch1s_dh_p99')))
        allok &= all(c.values())
    out['passes'] = bool(allok); return out


def fmt(v, n=3):
    if v is None: return '-'
    if isinstance(v, list): return '%.*f [%.*f, %.*f]' % (n, v[0], n, v[1], n, v[2])
    return '%.*f' % (n, v)


def md(res):
    L_ = ['# box_refine v2: size-only mode, val-B (one run)', '', 'Rules: v2/PREREG_V2.json. val-A choice: %s' % json.dumps(res['size']), '',
          'Units: target heights (aim point = top + 0.37 x height). dh / da = one-frame height / aim-point change. Jitter = RMS second difference. '
          'Lag: signed along the target motion (negative = behind), by target speed (heights per frame): vs target ("tgt") and vs the CURRENT DETECTION ("det"). '
          'Size lag = sign(d log h_target) x log(h / reference), rows with |d log h_target| >= 0.03 per frame. [90 % per-sequence bootstrap].', '']
    for ds in ('TMOT', 'DT'):
        for part in ('nf', 'sw'):
            T = res['tables']['%s|%s' % (ds, part)]; Mm = list(T['methods'])
            L_ += ['## %s, %s (%d sequences, %d rows with a target)' % ('TMOT (thermal)' if ds == 'TMOT' else 'DanceTrack', 'near + far pooled' if part == 'nf' else 'switch stream', len(T['sequences']), T['n_rows']), '',
                   '| method | centre err | aim err | abs log h | centre jitter | aim jitter | dh p95 | dh p99 | dh max | per-track max dh med / p90 | da p99 |', '|' + '---|' * 11]
            for m in Mm:
                s = T['methods'][m]; b = s['boot']
                L_.append('| %s | %s | %s | %s | %s | %s | %.4f | %s | %.3f | %.3f / %.3f | %.3f |' % (NAMES[m], fmt(b['ce']), fmt(b['aim']), fmt(b['lh']), fmt(b['d2c_rms']), fmt(b['d2a_rms']),
                          s['dh_p95'], fmt(b['dh_p99']), s['dh_max'], s['dh_trackmax_med'], s['dh_trackmax_p90'], s['da_p99']))
            L_ += ['', '| method | start rows 1-2 dh mean / p95 | start rows 1-4 dh p95 | later dh p99 | switch frame dh / da p95 | 1 s after switch dh / da p99 | 1 s after switch dh / da max | def. flip dh / da p95 (n) |', '|' + '---|' * 8]
            for m in Mm:
                e = T['methods'][m]['events']; g = lambda k, f: e[k][f] if k in e else None
                L_.append('| %s | %s / %s | %s | %s | %s / %s | %s / %s | %s / %s | %s / %s (%s) |' % (NAMES[m], fmt(g('start', 'dh_mean')), fmt(g('start', 'dh_p95')), fmt(g('start5', 'dh_p95')), fmt(g('later', 'dh_p99')),
                          fmt(g('switch0', 'dh_p95')), fmt(g('switch0', 'da_p95')), fmt(g('switch1s', 'dh_p99')), fmt(g('switch1s', 'da_p99')), fmt(g('switch1s', 'dh_max')), fmt(g('switch1s', 'da_max')),
                          fmt(g('flip', 'dh_p95')), fmt(g('flip', 'da_p95')), g('flip', 'n')))
            L_ += ['', '| method | centre lag tgt / det, 0.03-0.1 | > 0.1 h/frame | aim lag tgt / det, 0.03-0.1 | > 0.1 | size lag tgt / det (fast size change) |', '|' + '---|' * 6]
            for m in Mm:
                s = T['methods'][m]; c = lambda k, a, b: '%+.3f / %+.3f' % (s[k][a], s[k][b]) if s[k][a] is not None and s[k][b] is not None else '-'
                L_.append('| %s | %s | %s | %s | %s | %+.3f / %+.3f |' % (NAMES[m], c('lag_0.03-0.1', 'along', 'along_det'), c('lag_0.1-99', 'along', 'along_det'),
                          c('lag_0.03-0.1', 'aim_t', 'aim_det'), c('lag_0.1-99', 'aim_t', 'aim_det'), s['slag_0.03-99']['salong'], s['slag_0.03-99']['sdet']))
            L_ += ['', 'Differences [90 % bootstrap]:', '', '| pair | centre err | aim err | abs log h | aim jitter^2 | dh p99 | per-track max dh med | aim lag det > 0.1 | aim lag det 0.03-0.1 | size lag tgt | start 1-4 dh p95 | switch 1 s dh p99 | flip dh p95 |', '|' + '---|' * 13]
            for k, d in T['diffs'].items():
                L_.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (k, fmt(d['ce'], 4), fmt(d['aim'], 4), fmt(d['lh'], 4), fmt(d['d2a_sq'], 5), fmt(d['dh_p99'], 4), fmt(d['dh_trackmax_med'], 4),
                          fmt(d['aim_det_0.1-99'], 4), fmt(d['aim_det_0.03-0.1'], 4), fmt(d['salong_0.03-99'], 4), fmt(d.get('ev_start5_dh_p95'), 4), fmt(d.get('ev_switch1s_dh_p99'), 4), fmt(d.get('ev_flip_dh_p95'), 4)))
            L_.append('')
    L_ += ['## Decision (PREREG_V2 pass_rules_size_vs_tracker)', '', '```', json.dumps(res['decision'], indent=1), '```']
    return '\n'.join(L_)


def main():
    t0 = time.time(); split = sys.argv[1] if len(sys.argv) > 1 else 'valB'; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ch = json.loads((L.V2 / 'results' / 'size_valA.json').read_text())['choice']
    assert all(L.BRM.SIZE[k] == ch[k] for k in ('anchor', 'src', 'mc_s', 'beta_s', 'dcut')), (L.BRM.SIZE, ch)
    print('SIZE', L.BRM.SIZE, 'GRU', L.GRU_FILE.name, flush=True)
    reg = [r for r in L.P.registry() if r['split'] == split]; jobs = [(r, st) for r in reg for st in ('near', 'far', 'sw')]
    with Pool(nw, initializer=init) as pool:
        res = pool.map(job, jobs, chunksize=1)
    recs = [x for rr, _ in res for x in rr]; evs = [x for _, ee in res for x in ee]
    print('RUNS %.1f s, %d ids' % (time.time() - t0, len(recs)), flush=True)
    out = dict(split=split, size=dict(L.BRM.SIZE), combo_a=L.COMBO_A, euro=dict(L.BRM.EURO), gru_file=L.GRU_FILE.name, tables={})
    for ds in ('TMOT', 'DT'):
        for part, sel in (('nf', ('near', 'far')), ('sw', ('sw',))):
            k = [i for i, r in enumerate(recs) if r['ds'] == ds and r['stream'] in sel]
            T = table([recs[i] for i in k], [evs[i] for i in k], METHODS); out['tables']['%s|%s' % (ds, part)] = T
            print('TABLE %s %s rows %d (%.1f s)' % (ds, part, T['n_rows'], time.time() - t0), flush=True)
            for m in METHODS:
                s = T['methods'][m]
                print('  %-8s ce %.4f aim %.4f lh %.4f d2a %.4f dh99 %.4f tmax %.4f start5 %s sw1 %s aimdet> %s slag %+.4f' % (m, s['ce'], s['aim'], s['lh'], s['d2a_rms'], s['dh_p99'], s['dh_trackmax_med'],
                      r5(s['events'].get('start5', {}).get('dh_p95')), r5(s['events'].get('switch1s', {}).get('dh_p99')), r5(s['lag_0.1-99']['aim_det']), s['slag_0.03-99']['salong']), flush=True)
    out['decision'] = decide(out['tables']); print('DECISION', json.dumps(out['decision']), flush=True)
    tag = 'size_valB' if split == 'valB' else 'size_check_valA'
    (L.V2 / 'results' / ('%s.json' % tag)).write_text(json.dumps(out, indent=1))
    (L.V2 / 'results' / ('tables_v2%s.md' % ('' if split == 'valB' else '_check_valA'))).write_text(md(out), encoding='utf-8')
    print('VALB_DONE %.1f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
