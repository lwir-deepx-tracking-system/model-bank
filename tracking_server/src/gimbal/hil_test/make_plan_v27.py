#!/usr/bin/env python3
"""Writes the plan files of the three stages (2026-11-11, second round). Standard library only; runs on the device or on any PC.
  python3 make_plan_v27.py stage1                    -> plan_stage1.json                       (fixed, needs no result)
  python3 make_plan_v27.py stage2 <stage 1 folder>   -> plan_stage2.json   from <folder>/STAGE1_RESULT.json  (written by: python3 analyze_v27.py stage1 <folder>)
  python3 make_plan_v27.py stage3 <stage 2 folder>   -> plan_stage3.json   from <folder>/STAGE2_CHOICE.json  (written by: python3 analyze_v27.py stage2 <folder>)
  python3 make_plan_v27.py dry                       -> plan_stage1_dry.json, plan_stage2_dry.json, plan_stage3_dry.json   (short; for dry runs only)
  python3 make_plan_v27.py assumed <folder>          -> full-length stage 2 and stage 3 plans from ASSUMED numbers, written into <folder> (PC dry checks only)
A plan made from assumed numbers carries "assumed": true in its first step; the test tool refuses to run such a plan on the real gimbal.
Every rule below is written in PLAN_V27_HILS_20261111_KO.md before any run; this file only applies it.
Third round (2026-11-12, second safety review, lines marked "v27c"):
  - stage2 / stage3 make NO plan when the stage before did not run to its end, or its last STOP was not written, or the view still moved after its last STOP.
    Only a person can decide otherwise:   python3 make_plan_v27.py stage2 <stage 1 folder> --fixed   (fixed candidate values)
                                          python3 make_plan_v27.py stage3 <stage 2 folder> --fixed   (current gains)
  - stage 3 leaves the 240 px/s walk (near240) out. It is put in only by the rule near240_rule() below (plan, section 3), never by hand."""
import json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LEAD = 1                                             # 예측 제어 on, 150 ms, as in live use
CURRENT = dict(kp=30.0, ki=0.0, kd=6.0, il=3.0, db=0.06)   # the live gains (tuned 11-10 with speed cap 10)
V27_EXTRA = ['--deadband-exit', '0.10', '--axis-reversal', '1']
BOX_NEAR = [120, 300]                                # a person close to the camera: about 300 px tall
ERRLIM = [240.0, 70.0]                               # px: a block ends as "person lost" when the person is this far from the picture centre, left-right / up-down (this morning's values; the same for every walk)
LATENCY_S = 0.040                                  # camera -> bridge delay the tool imitates (an assumption, same as on 11-10 and this morning)
# stage 1
S1_SPEEDS = [1, 2, 5, 10, 20, 50, 100]; S1_BUDGET = {'x': 180.0, 'y': 80.0}; S1_PAIRS = 3
# stage 2
S2_SPEEDS = [(40, 100.0), (80, 120.0), (160, 100.0)]; S2_DUR = 10.0      # walking speed px/s, how far to each side px. Smaller than the 150 of this morning: the view limit is 230 px now, and the pretend gimbal's view went to 210 px with 150 and to 196 px with 120 at 160 px/s (it runs on past the turning point)
KD_HIGH = 9.0; KI_SMALL, IL_SMALL = 3.0, 2.0
KP_LOW_DEFAULT, KP_LOW_MIN, KP_LOW_MAX = 20.0, 10.0, 24.0; DB_WIDE_DEFAULT, DB_WIDE_MIN, DB_WIDE_MAX = 0.08, 0.07, 0.12
# stage 3
S3_DUR = 20.0; S3_AMP = {40: 120.0, 80: 120.0, 160: 100.0, 240: 90.0}   # seconds per block; how far to each side each walk goes (px): the faster the walk, the shorter, because the view runs on further past the turning point
REV_LEGS = [90, -60, 120, -120, 30, -90, 110, -40, 80, -120, 60, -100, 120, -30, 100, -110]   # turning points (px) of the walk that reverses suddenly
NEAR240_PX = 25.0                                    # v27c (second review; plan section 3, "near240 규칙"): the 240 px/s walk is part of stage 3 only when the view moves at most this many px per picture at SPD 20 (the picture matching follows 36 px per picture at most)


def opts(g):
    f = lambda v: ('%g' % v)
    return ['--kp', f(g['kp']) if g['kp'] != 30.0 else '30.0', '--controller', 'pid', '--ki', f(g['ki']), '--kd', f(g['kd']), '--integral-limit', f(g['il']), '--deadband', f(g['db']),
            '--manual-speed', '20', '--speed-max', '100', '--reacquire-ms', '10000']          # the live option line (scripts/run_real_gui.sh, 2026-11-11) with the gains filled in


def walk(speed, amp, dur):
    """left-right walk of the virtual person: 0.5 s standing, to +amp, then back and forth between -amp and +amp at <speed> px/s"""
    wp = [[0, 0, 0], [0.5, 0, 0]]; t = 0.5 + amp / speed; x = amp
    while True:
        wp.append([round(t, 3), float(x), 0])
        if t > dur: break
        t += 2 * amp / speed; x = -x
    return wp


def walk_rev(speed, dur):
    """a walk that turns round suddenly at uneven places: 0.5 s standing, then from turning point to turning point at <speed> px/s"""
    wp = [[0, 0, 0], [0.5, 0, 0]]; t = 0.5; x = 0.0; i = 0
    while t <= dur:
        nx = float(REV_LEGS[i % len(REV_LEGS)]); t += abs(nx - x) / speed; x = nx; i += 1; wp.append([round(t, 3), x, 0])
    return wp


def block(name, bridge, g, wp, dur, box=None, extra=None, **more):
    st = {'type': 'track', 'name': name, 'bridge': bridge, 'opts': opts(g), 'extra': list(extra or []), 'dur': dur, 'wp': wp, 'lead': LEAD, 'errlim': ERRLIM}
    if box: st['box'] = box
    st.update(more); return st


def start(): return [{'type': 'static', 'T': 10, 'blind': 1}, {'type': 'static', 'T': 4}]


def plan_stage1(speeds=S1_SPEEDS, pairs=S1_PAIRS, short=False):
    p = [{'type': 'meta', 'stage': 1, 'made': time.strftime('%Y-%m-%d %H:%M:%S'), 'assumed': False, 'speeds': speeds, 'budget_px': S1_BUDGET, 'pairs': pairs}]
    p += [{'type': 'static', 'T': 3, 'blind': 1}, {'type': 'static', 'T': 2}] if short else start()
    for ax in ('x', 'y'):
        for sp in speeds: p.append({'type': 'pulse', 'axis': ax, 'speed': sp, 'budget': S1_BUDGET[ax], 'pairs': pairs})
    return p


def ceil2(v): return math.ceil(v * 100 - 1e-9) / 100.0


def candidates(r):
    """the stage 2 candidates from the stage 1 result (rule in the plan, section 2단계). r = STAGE1_RESULT.json or None"""
    why = {}; kp_low = KP_LOW_DEFAULT; db_wide = DB_WIDE_DEFAULT
    x = (r or {}).get('axes', {}).get('x') or {}; y = (r or {}).get('axes', {}).get('y') or {}; fp = (r or {}).get('frame_s')
    if x.get('slope') and x['slope'] > 0 and x.get('delay_s') is not None and fp:
        theta = x['delay_s'] + LATENCY_S + fp; raw = 320.0 / (2.0 * theta * x['slope']); kp_low = float(min(KP_LOW_MAX, max(KP_LOW_MIN, round(raw))))
        why['kp_low'] = 'Kp = 320 / (2 x delay %.3f s x slope %.1f px/s per SPD) = %.1f -> %.0f (kept inside %.0f..%.0f)' % (theta, x['slope'], raw, kp_low, KP_LOW_MIN, KP_LOW_MAX)
    else: why['kp_low'] = 'stage 1 gave no slope or delay for the left-right axis: the fixed value %.0f' % KP_LOW_DEFAULT
    need = [a['floor'] * (a['delay_s'] + LATENCY_S) / 2.0 / half for a, half in ((x, 320.0), (y, 240.0)) if a.get('floor') and a.get('delay_s') is not None]
    if need:
        db_wide = float(min(DB_WIDE_MAX, max(DB_WIDE_MIN, ceil2(1.25 * max(need)))))
        why['db_wide'] = 'deadband = 1.25 x half of (lowest speed x (delay + %.0f ms)), the larger axis = 1.25 x %.4f -> %.2f (kept inside %.2f..%.2f)' % (1e3 * LATENCY_S, max(need), db_wide, DB_WIDE_MIN, DB_WIDE_MAX)
    else: why['db_wide'] = 'stage 1 gave no lowest speed or delay: the fixed value %.2f' % DB_WIDE_DEFAULT
    c = [('A', 'current gains', dict(CURRENT)), ('B', 'Kp lower', dict(CURRENT, kp=kp_low)), ('C', 'Kd higher', dict(CURRENT, kd=KD_HIGH)),
         ('D', 'deadband wider', dict(CURRENT, db=db_wide)), ('E', 'small Ki with an integral limit', dict(CURRENT, ki=KI_SMALL, il=IL_SMALL))]
    return c, why


def plan_stage2(r, src, assumed, dur=S2_DUR, speeds=S2_SPEEDS, short=False):
    c, why = candidates(r)
    p = [{'type': 'meta', 'stage': 2, 'made': time.strftime('%Y-%m-%d %H:%M:%S'), 'assumed': bool(assumed), 'from': src, 'why': why,
          'candidates': [{'tag': t, 'label': l, 'gains': g} for t, l, g in c]}]
    p += [{'type': 'static', 'T': 3, 'blind': 1}, {'type': 'static', 'T': 2}] if short else start(); k = 0
    for speed, amp in speeds:
        for order in ((c,) if short else (c, c[::-1])):          # second round in the opposite order: slow changes fall on all candidates alike
            for tag, label, g in order: p.append(block('walk%d_c%s_%d' % (speed, tag, k), 'v25', g, walk(speed, amp, dur), dur, cand=tag)); k += 1
    return p


def near240_rule(r1):
    """v27c (second review). The rule of the plan, section 3: is the 240 px/s walk part of stage 3? r1 = STAGE1_RESULT.json of the stage 1 run that the stage 2 plan was made from, or None.
    Yes only when that stage 1 ran to its end on the real gimbal, SPD 20 left-right was measured in BOTH directions, and
    (the largest of the measured speeds of the two directions) x (the time between two pictures) <= NEAR240_PX. Anything missing = no. Returns (yes or no, the reason in words)."""
    if not r1: return False, 'no stage 1 result: the 240 px/s walk is left out'
    if r1.get('dry') or r1.get('assumed'): return False, 'the stage 1 result does not come from the real gimbal: the 240 px/s walk is left out'
    if not r1.get('ok') or not_ended(r1.get('end_state')): return False, 'stage 1 did not run to its end as it should (%s): the 240 px/s walk is left out' % (not_ended(r1.get('end_state'), bool(r1.get('ok'))) or 'ok is false')
    fp = r1.get('frame_s'); rows = [e for e in (r1.get('table') or []) if e.get('axis') == 'x' and e.get('spd') == 20 and e.get('dir') in ('+', '-') and e.get('v_best')]
    if not fp or fp <= 0 or {e['dir'] for e in rows} != {'+', '-'}: return False, 'stage 1 did not measure SPD 20 left-right in both directions (or the time between two pictures): the 240 px/s walk is left out'
    v = max(float(x) for e in rows for x in (e.get('v_pulse'), e.get('v_fit'), e.get('v_best')) if x); px = v * fp
    if px <= NEAR240_PX: return True, 'SPD 20 left-right, faster direction %.0f px/s x %.3f s between two pictures = %.1f px, not above %.0f px: the 240 px/s walk is PUT IN' % (v, fp, px, NEAR240_PX)
    return False, 'SPD 20 left-right, faster direction %.0f px/s x %.3f s between two pictures = %.1f px, above %.0f px: the 240 px/s walk is left out' % (v, fp, px, NEAR240_PX)


def plan_stage3(g, src, assumed, dur=S3_DUR, short=False, near240=False, why240='the 240 px/s walk is left out (the default)'):   # v27c: + near240, why240
    walks = [('slow20', walk(20, 60.0, dur), None), ('walk40', walk(40, S3_AMP[40], dur), None), ('walk80', walk(80, S3_AMP[80], dur), None),
             ('near160', walk(160, S3_AMP[160], dur), BOX_NEAR), ('near240', walk(240, S3_AMP[240], dur), BOX_NEAR), ('rev120', walk_rev(120, dur), None)]
    if not near240: walks = [w for w in walks if w[0] != 'near240']   # v27c (second review): out by default
    p = [{'type': 'meta', 'stage': 3, 'made': time.strftime('%Y-%m-%d %H:%M:%S'), 'assumed': bool(assumed), 'from': src, 'gains': g, 'v27_extra': V27_EXTRA, 'near240': bool(near240), 'near240_why': why240}]   # v27c: + near240, near240_why
    p += [{'type': 'static', 'T': 3, 'blind': 1}, {'type': 'static', 'T': 2}] if short else start(); k = 0
    for i, (name, wp, box) in enumerate(walks):
        order = ('v25', 'v27', 'v27', 'v25') if i % 2 == 0 else ('v27', 'v25', 'v25', 'v27')      # ABBA; every second walk starts with the other program
        for b in (order[:2] if short else order): p.append(block('%s_%s_%d' % (name, b, k), b, g, wp, dur, box=box, extra=V27_EXTRA if b == 'v27' else [], walk=name)); k += 1
    return p


def write(path, p):
    open(path, 'w', newline='\n').write(json.dumps(p) + '\n'); tr = [s for s in p if s['type'] == 'track']; pu = [s for s in p if s['type'] == 'pulse']
    print('%s: steps %d, blocks %d (%.0f s), pulse steps %d, assumed %s' % (os.path.basename(path), len(p), len(tr), sum(s['dur'] for s in tr), len(pu), p[0].get('assumed')))
    if p[0].get('why'):
        for k, v in p[0]['why'].items(): print('   %s: %s' % (k, v))
    for c in p[0].get('candidates', []): print('   candidate %s (%s): %s' % (c['tag'], c['label'], ' '.join(opts(c['gains'])[:10])))
    if p[0].get('gains'): print('   gains: %s' % ' '.join(opts(p[0]['gains'])[:10]))
    if p[0].get('stage') == 3: print('   walks: %s' % ', '.join(dict.fromkeys(s['walk'] for s in tr))); print('   240 px/s walk (near240): %s. %s' % ('IN THE PLAN' if p[0].get('near240') else 'not in the plan', p[0].get('near240_why', '')))   # v27c
    if p[0].get('fixed'): print('   NOTE: made with --fixed although the stage before did not end as it should: %s' % p[0]['fixed'])   # v27c


# numbers of the 11-10 record (PID_TUNING_RECORD_20261110.md), used ONLY for plans marked "assumed"
ASSUMED_S1 = {'ok': True, 'assumed': True, 'frame_s': 0.045, 'axes': {'x': {'floor': 160.0, 'slope': 35.6, 'delay_s': 0.16}, 'y': {'floor': 165.0, 'slope': 33.0, 'delay_s': 0.12}}}


def not_ended(es, need_ok=None):
    """v27c (second review): why the run behind this result must not be built on, in words ('' = nothing against it). es = its end_state; need_ok = the result's own "ok" (stage 1)."""
    es = es or {}; why = []
    if need_ok is False: why.append('the stage did not run to its end or measured fewer than 3 left-right commands (ok is false)')
    if es.get('why') != 'PLAN DONE': why.append('the run did not end with PLAN DONE (%s)' % es.get('why'))
    if es.get('last_stop_written') is not True: why.append('its last STOP was not written to the gimbal (or the record does not say)')
    if es.get('view_moving_after_stop'): why.append('the view was still moving after its last STOP')
    if es.get('end_view_checked') is not True: why.append('its end view was not checked after the last STOP (END VIEW NOT CHECKED, or the record does not say)')   # v27d (third review, minor 1)
    return '; '.join(why)


def stage1_of(folder2):
    """v27c: the STAGE1_RESULT.json that the stage 2 plan of this stage 2 folder was made from (named in plan_used.json), or None"""
    try:
        meta = json.load(open(os.path.join(folder2, 'plan_used.json')))[0]; f = meta.get('from', '')
        return json.load(open(f)) if meta.get('stage') == 2 and os.path.basename(f) == 'STAGE1_RESULT.json' else None
    except Exception: return None


def main():
    m = sys.argv[1] if len(sys.argv) > 1 else ''; fixed = '--fixed' in sys.argv[3:]   # v27c: + fixed
    if m == 'stage1': write(os.path.join(HERE, 'plan_stage1.json'), plan_stage1())
    elif m == 'stage2' and len(sys.argv) > 2:
        f = os.path.join(sys.argv[2], 'STAGE1_RESULT.json'); r = json.load(open(f))
        if r.get('dry'): sys.exit('%s comes from a run without the gimbal: not used for the real stage 2 plan' % f)
        bad = not_ended(r.get('end_state'), bool(r.get('ok')))   # v27c (second review): was: a plan with the fixed values, without a word
        if bad and not fixed: sys.exit('NO PLAN MADE. %s: %s.\nRead the stage 1 table first. Only if a person decides to go on with the FIXED candidate values:  python3 make_plan_v27.py stage2 %s --fixed' % (f, bad, sys.argv[2]))
        p = plan_stage2(None if bad else r, os.path.abspath(f), False)
        if bad: p[0]['fixed'] = bad
        write(os.path.join(HERE, 'plan_stage2.json'), p)
    elif m == 'stage3' and len(sys.argv) > 2:
        f = os.path.join(sys.argv[2], 'STAGE2_CHOICE.json'); r = json.load(open(f))
        if r.get('dry'): sys.exit('%s comes from a run without the gimbal: not used for the real stage 3 plan' % f)
        bad = not_ended(r.get('end_state'))            # v27c (second review)
        if bad and not fixed: sys.exit('NO PLAN MADE. %s: %s.\nRead the stage 2 table first. Only if a person decides to go on with the CURRENT gains:  python3 make_plan_v27.py stage3 %s --fixed' % (f, bad, sys.argv[2]))
        yes, why = near240_rule(stage1_of(sys.argv[2]))   # v27c (second review): the rule decides, never a person
        p = plan_stage3(dict(CURRENT) if bad else r['gains'], os.path.abspath(f), False, near240=yes, why240=why)
        if bad: p[0]['fixed'] = bad
        write(os.path.join(HERE, 'plan_stage3.json'), p)
    elif m == 'dry':
        write(os.path.join(HERE, 'plan_stage1_dry.json'), plan_stage1([1, 5, 20, 100], 2, short=True))
        write(os.path.join(HERE, 'plan_stage2_dry.json'), plan_stage2(ASSUMED_S1, '11-10 record (assumed)', True, dur=6.0, speeds=[(80, 120.0), (160, 100.0)], short=True))
        write(os.path.join(HERE, 'plan_stage3_dry.json'), plan_stage3(dict(CURRENT), 'current gains (assumed)', True, dur=6.0, short=True))
    elif m == 'assumed' and len(sys.argv) > 2:
        write(os.path.join(sys.argv[2], 'plan_stage2_assumed.json'), plan_stage2(ASSUMED_S1, '11-10 record (assumed)', True))
        write(os.path.join(sys.argv[2], 'plan_stage3_assumed.json'), plan_stage3(dict(CURRENT), 'current gains (assumed)', True))
    else: sys.exit(__doc__)


if __name__ == '__main__': main()
