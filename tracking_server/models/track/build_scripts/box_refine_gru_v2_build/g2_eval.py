"""gru_v2/scripts/g2_eval.py - GRU refiner v2: val-A choices and the single val-B report (PREREG_GRU_V2.json).
  python g2_eval.py valA [workers]   -> results/valA_selection.json  (GRU run, size filter, tuned conv_euro_t; val-A rows only)
  python g2_eval.py valB             -> results/valB.json, results/tables_valB.md  (chosen things from valA_selection.json + the exported weights; pass rules P1-P5)
  python g2_eval.py checkA           -> results/check_valA.json, tables_check_valA.md  (the val-B report code run on val-A: code test, no choice)
"""
import itertools
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import g2_lib as L
from g2_lib import C, RE

RES = L.G2 / 'results'
SIZE_GRID = [(a, b, 1.0) for a in (1.5, 3.0, 5.0) for b in (0.0, 0.3, 1.0)]
EURO_GRID = [(a, b, c, d, 1.0) for a in (0.4, 1.0, 3.0) for b in (0.3, 1.0, 3.0) for c in (0.4, 1.5, 5.0) for d in (0.0, 0.3, 1.0)]
G = {}


def load_rows(split):
    d = pickle.loads((L.G2 / 'data' / ('rows_%s.pkl' % split)).read_bytes()); return d['recs'], d['W'], d['n']


def groups(recs):
    g = defaultdict(list)
    for i, r in enumerate(recs): g[(r['name'], r['stream'])].append(i)
    return g


def add_v2(recs, Wd, nd, net, size=None, key='gru_v2'):
    """the numpy engine frame by frame over every (sequence, stream): rec['o'][key] (and key + '_size')."""
    for (nm, st), idx in groups(recs).items():
        ids = {recs[i]['id']: dict(f=recs[i]['e']['f'], z=recs[i]['e']['z'], kp=recs[i]['e']['kp']) for i in idx}
        o = L.run_rows(ids, Wd[nm], recs[idx[0]]['fps'], nd[nm], net, size)
        for i in idx:
            r = recs[i]; assert np.array_equal(o[r['id']]['start'], r['e']['start']); r['o'][key] = o[r['id']]['out']
            if size is not None: r['o'][key + '_size'] = o[r['id']]['out_f']


def numbers(recs, methods, evs=None, parts=('TMOT', 'DT'), streams=('near', 'far', 'sw')):
    """per dataset: method -> summary (point estimates)."""
    out = {}
    for ds in parts:
        idx = [i for i, r in enumerate(recs) if r['ds'] == ds and r['stream'] in streams]; R_ = [recs[i] for i in idx]
        ev_ = [evs[i] for i in idx] if evs is not None else [L.events(r) for r in R_]; out[ds] = {}
        for m in methods:
            rows, seqv, idv, ev, sbv, _ = L.gather(R_, m, ev_); out[ds][m] = L.summary(rows, ev, idv, sbv)
    return out


def judge(sel, ref):
    """PREREG selection_valA: sel / ref = {ds: sel_numbers} of the candidate / of {'tracker', 'conv_euro'}. -> (J, violation, detail)"""
    J = []; viol = 0.; det = {}
    for ds in ('TMOT', 'DT'):
        s = sel[ds]; ce = ref[ds]['conv_euro']; tk = ref[ds]['tracker']
        rat = [s[k] / ce[k] for k in ('dh_p99', 'tmax', 'd2a')] + [(s['first5'] + 0.02) / (ce['first5'] + 0.02)]; J.append(float(np.mean(rat)))   # first5 term: amendment 1
        g1 = s['aim'] - (ce['aim'] * 1.02 + 0.001); g2a = (min(tk['lag_fast'], ce['lag_fast']) - 0.015) - s['lag_fast']; g2b = (min(tk['lag_mid'], ce['lag_mid']) - 0.010) - s['lag_mid']
        g3 = s['row1_over'] - 0.10; v = [max(0., x) for x in (g1, g2a, g2b, g3)]; viol += sum(v)
        det[ds] = dict(ratios=dict(zip(('dh_p99', 'tmax', 'd2a', 'first5'), [round(x, 4) for x in rat])), gate_excess=dict(aim=round(g1, 5), lag_fast=round(g2a, 5), lag_mid=round(g2b, 5), row1=round(g3, 5)), numbers=s)
    return float(np.mean(J)), float(viol), det


def pick(cands):
    """cands {name: (J, viol, det)} -> name by the rule (lowest J among viol == 0, else smallest violation)."""
    ok = [k for k, v in cands.items() if v[1] <= 0]
    return (min(ok, key=lambda k: cands[k][0]), True) if ok else (min(cands, key=lambda k: (cands[k][1], cands[k][0])), False)


# ------------------------------------------------------------------------------------------------ val-A
def _init(split):
    G['recs'], G['W'], G['n'] = load_rows(split); G['evs'] = [L.events(r) for r in G['recs']]


def _euro_job(p):
    recs = G['recs']
    for r in recs: r['o']['cand'] = L.euro_rows(r['e'], r['e']['conv'], r['e']['start'], G['W'][r['name']], r['fps'], p)
    L.CONVERTED = L.CONVERTED + ('cand',) if 'cand' not in L.CONVERTED else L.CONVERTED
    nb = numbers(recs, ['cand'], G['evs']); return p, {ds: L.sel_numbers(nb[ds]['cand']) for ds in nb}


def _run_job(stem):
    import torch
    ck = torch.load(L.G2 / 'runs' / (stem + '_best.pt'), map_location='cpu'); w = {k: v.numpy() for k, v in ck['model'].items()}; w['arch'] = np.array(ck['arch'])
    net = L.GRUNp(w); recs = G['recs']; add_v2(recs, G['W'], G['n'], net, None, 'cand')
    L.CONVERTED = L.CONVERTED + ('cand',) if 'cand' not in L.CONVERTED else L.CONVERTED
    nb = numbers(recs, ['cand'], G['evs']); return stem, {ds: L.sel_numbers(nb[ds]['cand']) for ds in nb}, ck['epoch']


def run_valA(nw):
    t0 = time.time(); _init('valA'); recs = G['recs']
    base = numbers(recs, ['raw', 'tracker', 'euro', 'conv', 'conv_euro', 'gru_v1'], G['evs']); ref = {ds: {m: L.sel_numbers(s) for m, s in base[ds].items()} for ds in base}
    res = dict(base=ref, base_judge={}, runs={}, size={}, euro={})
    for m in ('raw', 'tracker', 'euro', 'conv', 'conv_euro', 'gru_v1'):
        J, v, d = judge({ds: ref[ds][m] for ds in ref}, ref); res['base_judge'][m] = dict(J=J, viol=v)
        print('BASE %-10s J %.4f viol %.4f | ' % (m, J, v) + ' | '.join('%s dh99 %.4f tmax %.4f d2a %.4f f5 %.4f row1 %.2f aim %.4f lagF %+.4f lagM %+.4f' % (ds, *[ref[ds][m][k] for k in ('dh_p99', 'tmax', 'd2a', 'first5', 'row1_over', 'aim', 'lag_fast', 'lag_mid')]) for ds in ref), flush=True)
    stems = sorted(p.name[:-len('_RESULT.json')] for p in (L.G2 / 'runs').glob('g2_*_RESULT.json'))
    with Pool(nw, initializer=_init, initargs=('valA',)) as pool:
        cands = {}
        for stem, sel, ep in pool.imap_unordered(_run_job, stems):
            J, v, d = judge(sel, ref); cands[stem] = (J, v, d); res['runs'][stem] = dict(J=J, viol=v, epoch=ep, detail=d)
            print('RUN %-16s ep %2d J %.4f viol %.4f | ' % (stem, ep, J, v) + ' | '.join('%s dh99 %.4f tmax %.4f d2a %.4f f5 %.4f row1 %.2f aim %.4f lagF %+.4f lagM %+.4f' % (ds, *[sel[ds][k] for k in ('dh_p99', 'tmax', 'd2a', 'first5', 'row1_over', 'aim', 'lag_fast', 'lag_mid')]) for ds in sel), flush=True)
        best, feas = pick(cands); res['chosen_run'] = best; res['chosen_run_feasible'] = feas; print('CHOSEN RUN %s feasible %s (%.0f s)' % (best, feas, time.time() - t0), flush=True)
        ec = {}
        for p, sel in pool.imap_unordered(_euro_job, EURO_GRID):
            J, v, d = judge(sel, ref); ec[p] = (J, v, d); res['euro'][str(p)] = dict(J=J, viol=v, detail=d)
        pe, fe = pick(ec); res['conv_euro_t'] = list(pe); res['conv_euro_t_feasible'] = fe
        print('CONV_EURO_T %s feasible %s J %.4f viol %.4f (%.0f s)' % (pe, fe, ec[pe][0], ec[pe][1], time.time() - t0), flush=True)
    # size filter on the chosen run
    import torch
    ck = torch.load(L.G2 / 'runs' / (best + '_best.pt'), map_location='cpu'); w = {k: v.numpy() for k, v in ck['model'].items()}; w['arch'] = np.array(ck['arch'])
    add_v2(recs, G['W'], G['n'], L.GRUNp(w), None, 'gru_v2'); g0 = numbers(recs, ['gru_v2'], G['evs']); s0 = {ds: L.sel_numbers(g0[ds]['gru_v2']) for ds in g0}; sc = {}
    for p in SIZE_GRID:
        for r in recs: r['o']['gru_v2_size'] = L.size_filter(r['e'], r['o']['gru_v2'], r['e']['start'], G['W'][r['name']], r['fps'], p)
        nb = numbers(recs, ['gru_v2_size'], G['evs']); s = {ds: L.sel_numbers(nb[ds]['gru_v2_size']) for ds in nb}
        ok = all(s[ds]['lh'] <= s0[ds]['lh'] + 0.003 and s[ds]['aim'] <= s0[ds]['aim'] + 0.001 and s[ds]['slag'] >= s0[ds]['slag'] - 0.015 for ds in s)
        Js = float(np.mean([(s[ds]['dh_p99'] / s0[ds]['dh_p99'] + s[ds]['tmax'] / s0[ds]['tmax']) / 2 for ds in s])); sc[p] = (Js, ok)
        J, v, d = judge(s, ref); res['size'][str(p)] = dict(Js=Js, ok=bool(ok), J=J, viol=v, detail=d)
        print('SIZE %s Js %.4f ok %s J %.4f viol %.4f | ' % (p, Js, ok, J, v) + ' | '.join('%s dh99 %.4f tmax %.4f lh %.4f slag %+.4f aim %.4f' % (ds, s[ds]['dh_p99'], s[ds]['tmax'], s[ds]['lh'], s[ds]['slag'], s[ds]['aim']) for ds in s), flush=True)
    okp = [p for p in sc if sc[p][1]]; ps = min(okp, key=lambda p: sc[p][0]) if okp else (5.0, 1.0, 1.0)
    res['size_chosen'] = list(ps); res['size_feasible'] = bool(okp); res['gru_v2_alone'] = s0
    print('SIZE CHOSEN %s feasible %s' % (ps, bool(okp)), flush=True)
    (RES / 'valA_selection.json').write_text(json.dumps(res, indent=1)); print('VALA_DONE %.0f s' % (time.time() - t0), flush=True)


# ------------------------------------------------------------------------------------------------ report (val-B once; val-A as a code check)
METHODS = ('raw', 'tracker', 'euro', 'conv', 'conv_euro', 'conv_euro_t', 'gru_v1', 'gru_v2', 'gru_v2_size')
NAMES = dict(raw='raw detector', tracker='tracker row (today)', euro='One Euro (round 1)', conv='lookup', conv_euro='lookup + One Euro (conv_euro)', conv_euro_t='lookup + tuned One Euro',
             gru_v1='GRU v1', gru_v2='GRU v2', gru_v2_size='GRU v2 + size filter')


def table(recs, evs):
    names = sorted({r['name'] for r in recs}); ns = len(names); draws = np.random.default_rng(0).integers(0, ns, (2000, ns)); Cc = C.draw_counts(draws, ns); R_ = {}
    for m in METHODS: rows, seqv, idv, ev, sbv, _ = L.gather(recs, m, evs); R_[m] = rows
    ok = R_['raw']['ok']; fin = lambda m, k, mm=None: np.where(ok if mm is None else mm, R_[m][k], np.nan); res = dict(sequences=names, n_rows=int(ok.sum()), methods={}, diffs={})
    small = ok & (sbv >= 0) & (sbv <= 1); f5 = small & ev['first5']; bins = {('%g-%g' % (a, b)): ok & (R_['raw']['spd'] >= a) & (R_['raw']['spd'] < b) for a, b in C.SPEED_BINS}
    tv = {m: C.track_max(R_[m], idv, seqv, 'dh') for m in METHODS}
    for m in METHODS:
        s = L.summary(R_[m], ev, idv, sbv); rows = R_[m]; b = {k: RE.boot_mean(fin(m, k), seqv, ns, draws) for k in ('ce', 'aim', 'lh')}
        for k in ('d2c', 'd2a'):
            x = RE.boot_mean(np.where(ok, rows[k] ** 2, np.nan), seqv, ns, draws); b[k + '_rms'] = [round(math.sqrt(max(v, 0)), 5) for v in x[:3]] + [x[3]]
        b['dh_p99'] = C.boot_pct(rows['dh'], None, seqv, ok, ns, Cc, 99)['a']; b['dh_p95'] = C.boot_pct(rows['dh'], None, seqv, ok, ns, Cc, 95)['a']; b['da_p99'] = C.boot_pct(rows['da'], None, seqv, ok, ns, Cc, 99)['a']
        b['tmax_med'] = C.boot_trackmax(tv[m][0], tv[m][1], None, Cc, 50)['a']; b['first5_dh'] = RE.boot_mean(fin(m, 'dh', f5), seqv, ns, draws)
        for bn, mm in bins.items():
            b['lag_' + bn] = RE.boot_mean(fin(m, 'along_det', mm), seqv, ns, draws); b['lagraw_big_' + bn] = RE.boot_mean(fin(m, 'along_raw', mm & (sbv == 2)), seqv, ns, draws)
            b['lag_big_' + bn] = RE.boot_mean(fin(m, 'along_det', mm & (sbv == 2)), seqv, ns, draws); b['lagtgt_' + bn] = RE.boot_mean(fin(m, 'along', mm), seqv, ns, draws)
        s['boot'] = b; s['by_size'] = {}
        for bi, bn in enumerate(L.P.SIZE_NAMES):
            mm = ok & (sbv == bi)
            if mm.sum() < 20: continue
            s['by_size'][bn] = dict(n=int(mm.sum()), ce=float(np.mean(rows['ce'][mm])), aim=float(np.mean(rows['aim'][mm])), d2c=float(math.sqrt(np.nanmean(rows['d2c'][mm] ** 2))),
                                    d2a=float(math.sqrt(np.nanmean(rows['d2a'][mm] ** 2))), dh_p99=float(np.nanpercentile(rows['dh'][mm], 99)), hr=float(np.median(rows['hr'][mm])))
        res['methods'][m] = s
    for a in ('gru_v2', 'gru_v2_size'):
        for bm in ('tracker', 'conv_euro', 'conv_euro_t', 'euro', 'gru_v1'):
            A, B = R_[a], R_[bm]; d = {k: RE.boot_diff(fin(a, k), fin(bm, k), seqv, ns, draws) for k in ('ce', 'aim', 'lh')}
            for k in ('d2c', 'd2a'): d[k + '_sq'] = RE.boot_diff(np.where(ok, A[k] ** 2, np.nan), np.where(ok, B[k] ** 2, np.nan), seqv, ns, draws)
            for k, q in (('dh', 95), ('dh', 99), ('da', 99)): d['%s_p%d' % (k, q)] = C.boot_pct(A[k], B[k], seqv, ok, ns, Cc, q)['diff']
            d['tmax_med'] = C.boot_trackmax(tv[a][0], tv[a][1], tv[bm][0], Cc, 50)['diff']; d['first5_dh'] = RE.boot_diff(fin(a, 'dh', f5), fin(bm, 'dh', f5), seqv, ns, draws)
            for bn, mm in bins.items(): d['lag_' + bn] = RE.boot_diff(fin(a, 'along_det', mm), fin(bm, 'along_det', mm), seqv, ns, draws)
            res['diffs']['%s - %s' % (a, bm)] = d
    return res


def decide(T):
    """PREREG valB_once.pass_rules on the near + far tables."""
    out = {}
    for cand in ('gru_v2', 'gru_v2_size'):
        chk = {}; allok = True
        for ds in ('TMOT', 'DT'):
            t = T[ds + '|nf']; M = t['methods']; c = {}
            for rule, bm in (('P1_tracker', 'tracker'), ('P2_conv_euro', 'conv_euro'), ('P2b_conv_euro_t', 'conv_euro_t')):
                d = t['diffs']['%s - %s' % (cand, bm)]; three = [d['dh_p99'], d['tmax_med'], d['d2a_sq']]
                c[rule] = dict(ok=bool(all(x[0] < 0 for x in three) and sum(x[2] < 0 for x in three) >= 2), dh_p99=d['dh_p99'], tmax_med=d['tmax_med'], d2a_sq=d['d2a_sq'])
            dce = t['diffs'][cand + ' - conv_euro']; dtk = t['diffs'][cand + ' - tracker']
            c['P3_aim'] = dict(ok=bool(dce['aim'][2] <= 0.005 and dtk['aim'][2] <= 0.005), vs_conv_euro=dce['aim'], vs_tracker=dtk['aim'])
            lf = lambda m, b: M[m]['lag_' + b]['along_det']; p4 = lf(cand, '0.1-99') >= max(lf('tracker', '0.1-99'), lf('conv_euro', '0.1-99')) - 0.02 and lf(cand, '0.03-0.1') >= max(lf('tracker', '0.03-0.1'), lf('conv_euro', '0.03-0.1')) - 0.015
            v = dict(fast={m: lf(m, '0.1-99') for m in (cand, 'tracker', 'conv_euro')}, mid={m: lf(m, '0.03-0.1') for m in (cand, 'tracker', 'conv_euro')})
            if ds == 'TMOT': rb = M[cand]['lag_0.1-99']['along_raw_big']; p4 = p4 and rb >= -0.05; v['tmot_big_fast_along_raw'] = rb
            c['P4_lag'] = dict(ok=bool(p4), values=v)
            f = M[cand]['first5']; fc = M['conv_euro']['first5']
            if f['n_row1'] >= 20: c['P5_first_rows'] = dict(ok=bool(f['dh_mean'] <= fc['dh_mean'] + 0.010 and f['row1_over_015'] <= 0.05), first5_dh=f['dh_mean'], conv_euro=fc['dh_mean'], row1_over_015=f['row1_over_015'], n_row1=f['n_row1'])
            else: c['P5_first_rows'] = dict(ok=True, skipped='fewer than 20 new small-person states', n_row1=f['n_row1'])
            chk[ds] = c; allok &= all(c[k]['ok'] for k in ('P1_tracker', 'P2_conv_euro', 'P3_aim', 'P4_lag', 'P5_first_rows'))
        out[cand] = dict(passes=bool(allok), beats_tuned_filter=bool(all(chk[ds]['P2b_conv_euro_t']['ok'] for ds in chk)), checks=chk)
    return out


def fmt(v, n=3):
    if v is None: return '-'
    if isinstance(v, (list, tuple)): return '%.*f [%.*f, %.*f]' % (n, v[0], n, v[1], n, v[2])
    return '%.*f' % (n, v)


def md(res, split):
    Lm = ['# GRU box refiner v2, %s (gru_v2)' % split, '', 'Choices (val-A): %s' % json.dumps(res['choices']), '',
          'Units: target heights. aim point = top + 0.37 x height. dh / da = one-frame height / aim-point change. Jitter = RMS second difference of the output itself. '
          'Lag = (output centre - current detection centre) along the target motion, negative = behind; the detection is in the method\'s own box definition. [90 % per-sequence bootstrap].', '']
    for ds in ('TMOT', 'DT'):
        for part in ('nf', 'sw'):
            T = res['tables']['%s|%s' % (ds, part)]
            Lm += ['## %s, %s (%d sequences, %d rows)' % ('TMOT (thermal)' if ds == 'TMOT' else 'DanceTrack', 'near + far streams' if part == 'nf' else 'switch stream', len(T['sequences']), T['n_rows']), '',
                   '| method | centre err | aim err | abs log h | centre jitter | aim jitter | dh p95 | dh p99 | per-track max dh (median) | da p99 |', '|' + '---|' * 10]
            for m in METHODS:
                b = T['methods'][m]['boot']; Lm.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (NAMES[m], fmt(b['ce']), fmt(b['aim']), fmt(b['lh']), fmt(b['d2c_rms']), fmt(b['d2a_rms']), fmt(b['dh_p95']), fmt(b['dh_p99']), fmt(b['tmax_med']), fmt(b['da_p99'])))
            Lm += ['', '| method | first 5 rows, people < 100 px: mean dh | share of row-1 steps > 0.15 (n) | height / target at rows 0, 1, 4 | first 5 rows all sizes mean dh / da | shape flip dh p95 (n) | 1 s after switch dh p99 |', '|' + '---|' * 7]
            for m in METHODS:
                s = T['methods'][m]; f = s['first5']; e = s['events']; g = lambda k, q: e[k][q] if k in e else None
                Lm.append('| %s | %s | %s (%d) | %s / %s / %s | %s / %s | %s (%s) | %s |' % (NAMES[m], fmt(s['boot']['first5_dh']), fmt(f['row1_over_015'], 2), f['n_row1'], fmt(f['hr_row0'], 2), fmt(f['hr_row1'], 2), fmt(f['hr_row4'], 2),
                          fmt(s['first5_all']['dh_mean']), fmt(s['first5_all']['da_mean']), fmt(g('flip', 'dh_p95')), g('flip', 'n'), fmt(g('switch1s', 'dh_p99'))))
            Lm += ['', '| method | lag vs detection 0-0.01 | 0.01-0.03 | 0.03-0.1 | > 0.1 h/frame | >= 100 px, > 0.1: vs own-definition det / vs raw det (n) | lag vs GT target > 0.1 | size lag |', '|' + '---|' * 8]
            for m in METHODS:
                s = T['methods'][m]; b = s['boot']; lg = s['lag_0.1-99']
                Lm.append('| %s | %s | %s | %s | %s | %s / %s (%d) | %s | %+.3f |' % (NAMES[m], fmt(b['lag_0-0.01']), fmt(b['lag_0.01-0.03']), fmt(b['lag_0.03-0.1']), fmt(b['lag_0.1-99']), fmt(lg['along_det_big']), fmt(lg['along_raw_big']), lg['n_big'],
                          fmt(b['lagtgt_0.1-99']), s['slag_0.03-99']['salong'] if s['slag_0.03-99']['salong'] is not None else float('nan')))
            Lm += ['', '| by GT height | ' + ' | '.join(NAMES[m] for m in METHODS) + ' |', '|' + '---|' * (len(METHODS) + 1)]
            for bn in L.P.SIZE_NAMES:
                if bn not in T['methods']['raw']['by_size']: continue
                Lm.append('| %s px: centre err / centre jitter / dh p99 / height:target | ' % bn + ' | '.join('%.3f / %.3f / %.3f / %.2f' % (T['methods'][m]['by_size'][bn]['ce'], T['methods'][m]['by_size'][bn]['d2c'], T['methods'][m]['by_size'][bn]['dh_p99'], T['methods'][m]['by_size'][bn]['hr']) for m in METHODS) + ' |')
            Lm += ['', '| difference | centre err | aim err | aim jitter^2 | dh p99 | per-track max dh | first-5 dh | lag 0.03-0.1 | lag > 0.1 |', '|' + '---|' * 9]
            for k, d in T['diffs'].items():
                Lm.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (k, fmt(d['ce'], 4), fmt(d['aim'], 4), fmt(d['d2a_sq'], 5), fmt(d['dh_p99'], 4), fmt(d['tmax_med'], 4), fmt(d['first5_dh'], 4), fmt(d['lag_0.03-0.1'], 4), fmt(d['lag_0.1-99'], 4)))
            Lm.append('')
    Lm += ['## Pass rules (PREREG_GRU_V2.json valB_once)', '', '```', json.dumps(res['decision'], indent=1), '```']
    return '\n'.join(Lm)


def run_report(split):
    t0 = time.time(); recs, Wd, nd = load_rows(split); sel = json.loads((RES / 'valA_selection.json').read_text()); exp = json.loads((L.G2 / 'export' / 'EXPORT.json').read_text())
    net = L.load_net(L.G2 / 'export' / exp['npz']); size = tuple(sel['size_chosen']); pe = tuple(sel['conv_euro_t'])
    add_v2(recs, Wd, nd, net, size, 'gru_v2')
    for r in recs:
        r['o']['conv_euro_t'] = L.euro_rows(r['e'], r['e']['conv'], r['e']['start'], Wd[r['name']], r['fps'], pe)
        chk = L.size_filter(r['e'], r['o']['gru_v2'], r['e']['start'], Wd[r['name']], r['fps'], size); assert np.abs(chk - r['o']['gru_v2_size']).max() < 1e-6
    evs = [L.events(r) for r in recs]; res = dict(split=split, choices=dict(run=sel['chosen_run'], run_feasible=sel['chosen_run_feasible'], size=list(size), size_feasible=sel['size_feasible'], conv_euro_t=list(pe), weights=exp['npz'], sha256=exp['sha256']), tables={})
    for ds in ('TMOT', 'DT'):
        for part, streams in (('nf', ('near', 'far')), ('sw', ('sw',))):
            idx = [i for i, r in enumerate(recs) if r['ds'] == ds and r['stream'] in streams]; res['tables']['%s|%s' % (ds, part)] = table([recs[i] for i in idx], [evs[i] for i in idx])
            print('TABLE %s %s rows %d (%.0f s)' % (ds, part, res['tables']['%s|%s' % (ds, part)]['n_rows'], time.time() - t0), flush=True)
    res['decision'] = decide(res['tables']); stem = 'valB' if split == 'valB' else 'check_valA'
    (RES / (stem + '.json')).write_text(json.dumps(res, indent=1)); (RES / ('tables_%s.md' % stem)).write_text(md(res, split), encoding='utf-8')
    for cand, d in res['decision'].items():
        print('DECISION %s passes %s beats tuned filter %s | ' % (cand, d['passes'], d['beats_tuned_filter']) + ' | '.join('%s %s' % (ds, ' '.join('%s=%s' % (k, v['ok']) for k, v in d['checks'][ds].items())) for ds in d['checks']), flush=True)
    print('REPORT_DONE %s %.0f s' % (split, time.time() - t0), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'valA': run_valA(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif sys.argv[1] == 'valB': run_report('valB')
    else: run_report('valA')
