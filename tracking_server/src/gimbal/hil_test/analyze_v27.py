#!/usr/bin/env python3
"""Real gimbal + virtual person, three stages (2026-11-11): reading of one run folder. Reads only; writes its table and result files into that folder.
  python3 analyze_v27.py stage1 <folder>   speed map of the gimbal            -> STAGE1_TABLE_KO.md, STAGE1_RESULT.json   (input of make_plan_v27.py stage2)
  python3 analyze_v27.py stage2 <folder>   PD candidates at speed cap 100     -> STAGE2_TABLE_KO.md, STAGE2_CHOICE.json   (input of make_plan_v27.py stage3)
  python3 analyze_v27.py stage3 <folder>   gimbal_bridge_v25 against v27      -> STAGE3_TABLE_KO.md, STAGE3_ANALYSIS.json
Every rule is fixed before any run in PLAN_V27_HILS_20261111_KO.md. Values per tracking block (phase 'run', first seconds left out, blocks that ran to the end):
  A  mean |dx|: the real (not shifted) virtual person's distance from the picture centre, left-right, px
  B  shake: standard deviation of dx after its 1 s moving average is removed, px (the measure of analyze_hil2_extra.py, 2026-11-11)
  C  lost ratio: share of samples in which the bridge reports the person as lost or the person is outside the picture; a block that ended as "person lost" counts as 1
  starts / stops / reversals per minute: from the commands that reached the gimbal line (serial.jsonl), reference only"""
import csv, json, math, os, sys
import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}
R_GOOD, R_BAD, C_GOOD, C_BAD = 0.85, 1.15, 0.02, 0.05          # stage 3, written in the plan before the run
S3_SKIP, S3_MIN_N, S3_MIN_PAIRS = 3.0, 150, 10
S3_MIN_PAIRS_5 = 8                                               # v27c (second review; written in the plan first): complete pairs needed when the plan has 5 walks = 10 pairs (the 240 px/s walk left out). With 6 walks = 12 pairs it stays S3_MIN_PAIRS
S2_SKIP, S2_MIN_N, S2_A_MEAN, S2_A_ONE, S2_B_NEED, S2_C_MAX = 2.0, 100, 1.15, 1.30, 0.90, 0.02   # stage 2
S2_REG_LOST = 2                                                  # v27c (second review; plan section 2, rule 1 (마)): a candidate with this many more attempts thrown away after a lost picture matching than the current gains is not chosen
S1_MOVE, S1_FLAT = 5.0, 1.15                                     # stage 1


def smooth(v, k):
    return np.convolve(np.pad(v, k // 2, mode='edge'), np.ones(k) / k, mode='valid')


def load(D):
    rows = list(csv.DictReader(open(os.path.join(D, 'frames.csv')))); blocks = list(csv.DictReader(open(os.path.join(D, 'blocks.csv')))); cmds = []
    for line in open(os.path.join(D, 'serial.jsonl')):
        try: d = json.loads(line)
        except ValueError: continue
        if 'cmd' in d and not d.get('blocked') and not d.get('werr'): cmds.append((d['ns'], d['cmd']))
        elif ('estop' in d or 'pulse_stop' in d) and d.get('ok', 1): cmds.append((d['ns'], {'T': 135}))
    try: es = json.load(open(os.path.join(D, 'end_state.json')))
    except Exception: es = None
    try: plan = json.load(open(os.path.join(D, 'plan_used.json')))
    except Exception: plan = []
    return rows, blocks, cmds, es, plan


def block_values(name, rows, cmds, skip):
    rr = [r for r in rows if r['name'] == name and r['phase'] == 'run' and float(r['t']) >= skip and r['ximg'] != '']
    if len(rr) < 10: return None
    t = np.array([float(r['t']) for r in rr]); dx = np.array([float(r['ximg']) - 320.0 for r in rr]); ns = [int(r['ns']) for r in rr]
    sx = np.array([float(r['sx']) for r in rr]); dur = float(t[-1] - t[0])
    k = max(3, int(round(1.0 / np.median(np.diff(t))))) | 1
    with np.errstate(all='ignore'): vx = np.nan_to_num(np.gradient(smooth(sx, 5), t), nan=0.0, posinf=0.0, neginf=0.0)   # equal time stamps (PC dry run) must not break the reference value
    # commands that reached the gimbal line inside the block; state at the block start = last such command before it
    before = [c for n, c in cmds if n < ns[0]]; seq = [(ns[0], before[-1] if before else {'T': 135})] + [(n, c) for n, c in cmds if ns[0] <= n <= ns[-1]]
    starts = stops = rev_x = rev_y = moves = 0; on_prev = None; dir_x = dir_y = 0; move_s = 0.0; spd_max = 0
    for i, (n, c) in enumerate(seq):
        on = c.get('T') == 141 and bool(c.get('X', 0) or c.get('Y', 0)) and c.get('SPD', 0) > 0
        n1 = seq[i + 1][0] if i + 1 < len(seq) else ns[-1]
        if on:
            move_s += (n1 - n) / 1e9; spd_max = max(spd_max, int(c.get('SPD', 0)))
            if i > 0:
                moves += 1; x, y = c.get('X', 0), c.get('Y', 0)
                if x and dir_x and x != dir_x: rev_x += 1
                if y and dir_y and y != dir_y: rev_y += 1
            if c.get('X', 0): dir_x = c.get('X', 0)
            if c.get('Y', 0): dir_y = c.get('Y', 0)
        if i > 0 and on and on_prev is False: starts += 1
        if i > 0 and not on and on_prev is True: stops += 1
        on_prev = on
    lost = [(r['lost'] not in ('0', '', '0.0')) or r['vis'] in ('0', '') for r in rr]
    return dict(n=len(rr), dur=round(dur, 2), fps=round(len(rr) / max(dur, 1e-6), 1), A_mean_abs_dx=round(float(np.abs(dx).mean()), 2),
                B_shake=round(float(np.std(dx - smooth(dx, k))), 2), C_lost=round(float(np.mean(lost)), 4),
                p95_abs_dx=round(float(np.percentile(np.abs(dx), 95)), 1), max_abs_dx=round(float(np.abs(dx).max()), 1),
                starts_per_min=round(60 * starts / dur, 1), stops_per_min=round(60 * stops / dur, 1), reversals_per_min=round(60 * rev_x / dur, 1),
                reversals_y_per_min=round(60 * rev_y / dur, 1), move_cmds=moves, moving_time_ratio=round(move_s / dur, 3), spd_max=spd_max,
                cam_speed_judder_px_s=round(float(np.std(vx - smooth(vx, k))), 2), resp_min=round(min(float(r['resp']) for r in rr), 2),
                sy_range=[round(min(float(r['sy']) for r in rr), 1), round(max(float(r['sy']) for r in rr), 1)])


def blocks_of(D, skip):
    rows, blocks, cmds, es, plan = load(D); res = []
    for b in blocks:
        base = b['name'].split('_try')[0]; f = base.split('_')          # <walk>_<tag>_<k>
        e = dict(name=b['name'], base=base, walk=f[0], tag=f[1], k=int(f[2]), status=b['status'], lead=int(b['lead']), bridge=b.get('bridge', ''))
        if b['status'] == 'ok':
            v = block_values(b['name'], rows, cmds, skip)
            if v is None: e['status'] = 'too short'
            else: e.update(v)
        res.append(e)
    return res, es, plan


def end_lines(es):
    if not es: return ['', '## 끝난 자리', '', '- end_state.json 이 없다 (도구가 끝까지 정리하지 못했다).']
    return ['', '## 끝난 자리', '', '- 끝난 이유: %s' % es.get('why'), '- 시작 화면과 다시 맞춘 값 (px): %s (마지막 측정값 %s, 한계 %s)' % (es.get('end_view_px'), es.get('last_measured_px'), es.get('limits_px')),
            '- 가장 높았던 속도 명령: %s (지킴이 상한 %s), 짐벌 선에 나간 멈춤 명령: %s' % (es.get('highest_spd'), es.get('spd_abort'), es.get('stops_on_device_line')),
            '- 마지막 비상 멈춤이 장치에 써졌나: %s (못 쓴 횟수 %s, 장치를 다시 연 횟수 %s)' % ('예' if es.get('last_stop_written', True) else '**아니오**', es.get('stops_not_written'), es.get('device_opened_again')),
            '- 명령이 0 일 때 같은 측정값이 3번 이어진 횟수 (세기만 함): %s, 사람을 놓쳐 끝난 구간: %s, 화면 맞추기가 끊긴 횟수: %s' % (es.get('same_while_zero'), es.get('person_lost_blocks'), es.get('lost')),
            '- 마지막 멈춤 뒤에도 화면이 움직였나: %s' % ('**예 — 두 화면이 %s px 달랐다. 멈춤이 써졌지만 짐벌이 듣지 않았을 수 있다. 다음 단계로 가지 않는다.**' % es.get('view_moving_after_stop_px') if es.get('view_moving_after_stop') else ('기록에 없음' if 'view_moving_after_stop' not in es else '아니오' if es.get('end_view_px') is not None else '**확인하지 못함 (END VIEW NOT CHECKED: 끝난 화면을 시작 화면과 맞추지 못했다). 짐벌을 눈으로 볼 것. 다음 계획은 만들어지지 않는다.**'))]   # v27c (second review)


def dry_note(es):
    return ['- **짐벌 없이 돌린 확인 기록이다 (가짜 배경, 가짜 짐벌). 아래 숫자는 도구가 도는지 보는 용도일 뿐, 실제 짐벌의 값이 아니다.**', ''] if (es or {}).get('dry_truth_px') is not None else []


def out(D, name, L):
    open(os.path.join(D, name), 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n'); sys.stdout.buffer.write(('\n'.join(L) + '\n').encode('utf-8'))


# ---------------------------------------------------------------- stage 1
def fit_map(pts):
    """pts = [(SPD, px/s)] of one axis. Lowest speed = mean of the commands within 15 % of the slowest; above them a straight line."""
    if not pts: return None
    pts = sorted(pts); vmin = min(v for _, v in pts); flat = [(s, v) for s, v in pts if v <= S1_FLAT * vmin]; up = [(s, v) for s, v in pts if v > S1_FLAT * vmin]
    r = dict(floor=round(float(np.mean([v for _, v in flat])), 1), floor_cmds=[s for s, _ in flat], slope=None, intercept=None, knee=None, n_cmd=len(pts), line_cmds=[s for s, _ in up])
    if len(up) >= 2:
        a, b = np.polyfit([s for s, _ in up], [v for _, v in up], 1); r.update(slope=round(float(a), 2), intercept=round(float(b), 1), knee=round(float((r['floor'] - b) / a), 1) if a > 0 else None)
    elif len(up) == 1 and flat:
        s0, v0 = flat[-1]; s1, v1 = up[0]; a = (v1 - v0) / (s1 - s0); r.update(slope=round(float(a), 2), intercept=round(float(v0 - a * s0), 1), knee=float(s0))
    if r['knee'] is not None and up and not (max(s for s, _ in flat) <= r['knee'] <= min(s for s, _ in up)): r['knee'] = None   # the two lines meet outside the gap between the slow and the faster commands: not a usable number
    r['faster_from'] = [max(s for s, _ in flat), min(s for s, _ in up)] if up else None      # the largest command still at the lowest speed, the smallest command above it
    return r


def stage1(D):
    rows = list(csv.DictReader(open(os.path.join(D, 'frames.csv')))); idr = list(csv.DictReader(open(os.path.join(D, 'ident.csv'))))
    try: es = json.load(open(os.path.join(D, 'end_state.json')))
    except Exception: es = None
    ns = [int(r['ns']) for r in rows]; dt = sorted((b - a) / 1e9 for a, b in zip(ns, ns[1:]) if 0 < b - a < 1e9); frame_s = round(dt[len(dt) // 2], 4) if dt else None
    table = []; axes = {}
    for ax in ('x', 'y'):
        pts = []; delays = []
        for sp in sorted({int(r['spd']) for r in idr if r['axis'] == ax}):
            rr = [r for r in idr if r['axis'] == ax and int(r['spd']) == sp]; skip = next((r['note'] for r in rr if r['status'] == 'skip'), None)
            if skip:
                sr = next(r for r in rr if r['status'] == 'skip')
                ko = '화면 맞추기가 더 작은 명령을 따라가지 못했다' if 'picture matching' in skip else ('앞 명령에서 본 속도로 미루어 이 명령은 %s px/s 까지 나올 수 있어, 가장 짧은 이동으로도 여유 거리를 넘을 수 있다' % sr['v_plan'] if sr.get('v_plan') else '더 작은 명령이 이미 여유 거리를 넘을 수 있어 실행하지 않았다')
                table.append(dict(axis=ax, spd=sp, dir='', status='건너뜀', note=ko)); continue
            both = []
            for dr in ('1', '-1'):
                mv = [r for r in rr if r['dir'] == dr and r['status'] in ('ok', 'lost') and r['travel_px'] != '']
                ok = [r for r in mv if r['status'] == 'ok' and abs(float(r['travel_px'])) >= S1_MOVE]
                e = dict(axis=ax, spd=sp, dir='+' if dr == '1' else '-', moves=len(mv), lost=sum(r['status'] == 'lost' for r in mv), status='측정' if ok else ('화면 맞추기 끊김' if mv and all(r['status'] == 'lost' for r in mv) else '움직임 안 보임'))
                if mv:
                    lg = max(mv, key=lambda r: float(r['T_act_ms'])); e.update(T_long_ms=round(float(lg['T_act_ms'])), travel_px=round(abs(float(lg['travel_px'])), 1), v_pulse=round(abs(float(lg['travel_px'])) / (float(lg['T_act_ms']) / 1e3)))
                if ok:
                    T = np.array([float(r['T_act_ms']) / 1e3 for r in ok]); Dd = np.array([abs(float(r['travel_px'])) for r in ok]); e['v_peak'] = round(float(np.median([float(r['v_peak']) for r in ok])))
                    if len(ok) >= 2 and T.max() - T.min() >= 0.03:
                        a, b = np.polyfit(T, Dd, 1)
                        if a > 0: e.update(v_fit=round(float(a)), dead_ms=round(float(-b / a * 1e3)))
                    hi = [float(r['delay_hi_ms']) for r in ok if r['delay_hi_ms'] != '']; lo = [float(r['delay_lo_ms']) for r in ok if r['delay_lo_ms'] != '']
                    if hi: e.update(delay_hi_ms=round(float(np.median(hi))), delay_lo_ms=round(float(np.median(lo))) if lo else None); delays += [(h + (l if l is not None else h)) / 2e3 for h, l in zip(hi, lo + [None] * (len(hi) - len(lo)))]
                    v = e.get('v_fit') if e.get('v_fit') and 0.5 * e['v_pulse'] <= e['v_fit'] <= 2.0 * e['v_pulse'] else e['v_pulse']; e['v_best'] = v; both.append(v)
                table.append(e)
            if both: pts.append((sp, float(np.mean(both))))
        m = fit_map(pts)
        if m: m.update(delay_s=round(float(np.median(delays)), 3) if delays else None, points=[[s, round(v)] for s, v in pts]); axes[ax] = m
    done = (es or {}).get('why') == 'PLAN DONE'; x = axes.get('x') or {}
    stopped = (es or {}).get('last_stop_written') is True and not (es or {}).get('view_moving_after_stop') and (es or {}).get('end_view_checked') is True   # v27d: + end view checked   # v27c (second review): a run whose last STOP was not written, or whose view still moved after it, is not a result to build the next plan on
    res = dict(folder=os.path.abspath(D), ok=bool(done and stopped and x.get('n_cmd', 0) >= 3), plan_done=done, stopped=stopped, dry=(es or {}).get('dry_truth_px') is not None, frame_s=frame_s, axes=axes, table=table, end_state=es)
    json.dump(res, open(os.path.join(D, 'STAGE1_RESULT.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    f = lambda e, k, u='': '' if e.get(k) is None else '%s%s' % (e[k], u)
    L = ['# 1단계 — 짐벌 속도 지도 (analyze_v27.py stage1 이 만든 파일)', '', '- 기록 폴더: %s' % res['folder'], '- 화면 한 장 간격: %s 초' % frame_s, *dry_note(es),
         '- 읽는 법: "이동÷시간" = 가장 긴 이동의 거리 ÷ 명령이 살아 있던 시간. "두 길이로 맞춤" = 길이가 다른 이동들의 거리를 직선으로 맞춘 기울기. "첫 움직임" = 명령이 짐벌 선에 나간 뒤 화면에서 2 px 넘게 움직인 것이 처음 보인 때 (카메라 지연 포함, 화면 한 장 간격만큼 거칠다).', '',
         '| 축 | 속도 명령 | 방향 | 이동 횟수 | 가장 긴 이동 ms | 그때 이동 px | 속도 px/s (이동÷시간) | 속도 px/s (두 길이로 맞춤) | 사진 두 장 사이 최고 px/s | 첫 움직임 ms (안 보인 마지막 ~ 보인 처음) | 상태 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for e in table:
        if e['status'] == '건너뜀': L.append('| %s | %d | | | | | | | | | 건너뜀: %s |' % ('좌우' if e['axis'] == 'x' else '상하', e['spd'], e['note'])); continue
        L.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % ('좌우' if e['axis'] == 'x' else '상하', e['spd'], e['dir'], f(e, 'moves'), f(e, 'T_long_ms'), f(e, 'travel_px'), f(e, 'v_pulse'), f(e, 'v_fit'), f(e, 'v_peak'),
                 '' if e.get('delay_hi_ms') is None else '%s ~ %s' % ('?' if e.get('delay_lo_ms') is None else e['delay_lo_ms'], e['delay_hi_ms']), e['status'] + (' (끊긴 이동 %d번)' % e['lost'] if e.get('lost') else '')))
    L += ['', '## 맞춘 속도 지도와 판정', '', '| 축 | 가장 느린 속도 px/s | 그 속도가 나오는 명령 | 그 위의 직선 (px/s) | 어디서부터 빨라지나 | 첫 움직임까지 (초) | 잰 명령 수 |', '|---|---|---|---|---|---|---|']
    for ax in ('x', 'y'):
        m = axes.get(ax)
        if not m: L.append('| %s | 재지 못함 | | | | | 0 |' % ('좌우' if ax == 'x' else '상하')); continue
        L.append('| %s | %s | %s | %s | %s | %s | %d |' % ('좌우' if ax == 'x' else '상하', m['floor'], ', '.join(str(s) for s in m['floor_cmds']), '맞출 점이 모자람' if m['slope'] is None else '%.1f x 명령 %+.0f (명령 %s 로 맞춤)' % (m['slope'], m['intercept'], ', '.join(str(s) for s in m['line_cmds'])),
                 '' if not m['faster_from'] else '명령 %d 까지 가장 느린 속도, %d 부터 빨라짐%s' % (m['faster_from'][0], m['faster_from'][1], '' if m['knee'] is None else ' (두 선이 만나는 곳 %s)' % m['knee']), '' if m['delay_s'] is None else m['delay_s'], m['n_cmd']))
    L += ['', '- 판정: %s' % ('**1단계 끝. 2단계 계획을 만들 수 있다** (좌우 축에서 명령 %d개를 쟀다).' % x.get('n_cmd', 0) if res['ok'] else '**1단계가 끝까지 가지 못했거나, 좌우 축에서 잰 명령이 3개 미만이거나, 마지막 멈춤이 써지지 않았거나 그 뒤에도 화면이 움직였다. make_plan_v27.py stage2 는 계획을 만들지 않는다. 사람이 이 표를 읽고 그래도 가기로 정할 때만 --fixed 로 고정 값 계획을 만든다.**'),   # v27c: the words (second review)
          '- 표의 숫자는 방향마다 이동 %s번 이하에서 나온 것이다 (한 번의 실행). 건너뜀 = 미리 적은 규칙으로 실행하지 않은 명령.' % max([e.get('moves', 0) for e in table] or [0])] + end_lines(es)
    out(D, 'STAGE1_TABLE_KO.md', L); return res


# ---------------------------------------------------------------- stage 2
def choose(cands, per):
    """the stage 2 rule. cands = [(tag, label, gains)], per[tag] = list of block dicts (status, walk, k, n, A, B, C). Returns (chosen tag, reason, table rows)"""
    walks = sorted({e['walk'] for v in per.values() for e in v}); rows = {}; cur = 'A'
    def mean(tag, key, walk=None, rnd=None):
        v = [e[key] for e in per.get(tag, []) if e['status'] == 'ok' and (walk is None or e['walk'] == walk) and (rnd is None or e['round'] == rnd)]
        return float(np.mean(v)) if v else None
    def complete(tag):
        v = per.get(tag, []); return bool(v) and all(any(e['status'] == 'ok' and e['walk'] == w and e['round'] == r and e['n'] >= S2_MIN_N for e in v) for w in walks for r in (0, 1))
    reg = lambda tag: sum(e['status'] == 'lost' for e in per.get(tag, []))   # v27c (second review): attempts of this candidate that were thrown away because the picture matching was lost
    for tag, label, g in cands:
        v = per.get(tag, []); lostb = sum(e['status'] == 'personlost' for e in v); cmax = max([e['C_lost'] for e in v if e['status'] == 'ok'] or [0.0])
        ra = [mean(tag, 'A_mean_abs_dx', w) / mean(cur, 'A_mean_abs_dx', w) for w in walks if mean(tag, 'A_mean_abs_dx', w) is not None and mean(cur, 'A_mean_abs_dx', w)]
        rb = [mean(tag, 'B_shake', w) / mean(cur, 'B_shake', w) for w in walks if mean(tag, 'B_shake', w) is not None and mean(cur, 'B_shake', w)]
        e = dict(tag=tag, label=label, gains=g, blocks=len([x for x in v if x['status'] == 'ok']), lost_blocks=lostb, c_max=cmax, A=mean(tag, 'A_mean_abs_dx'), B=mean(tag, 'B_shake'),
                 A_ratio=float(np.mean(ra)) if len(ra) == len(walks) and ra else None, A_ratio_max=max(ra) if ra else None, B_ratio=float(np.mean(rb)) if len(rb) == len(walks) and rb else None,
                 B_round=[mean(tag, 'B_shake', None, 0), mean(tag, 'B_shake', None, 1)], complete=complete(tag), reg_lost=reg(tag), skipped=sum(x['status'] == 'skipped' for x in v))   # v27c: + reg_lost, skipped
        why = []
        if not e['complete']: why.append('끝까지 돈 구간이 모자람')
        if lostb: why.append('사람을 놓쳐 끝난 구간 %d개' % lostb)
        if tag != cur and e['reg_lost'] >= reg(cur) + S2_REG_LOST: why.append('화면 맞추기가 끊겨 버린 시도 %d번 (현재 값 %d번보다 %d번 이상 많음)' % (e['reg_lost'], reg(cur), S2_REG_LOST))   # v27c (second review; plan section 2, rule 1 (마))
        if cmax > S2_C_MAX: why.append('놓친 비율 %.1f %%' % (100 * cmax))
        if e['A_ratio'] is not None and (e['A_ratio'] > S2_A_MEAN or e['A_ratio_max'] > S2_A_ONE): why.append('가운데 오차가 뚜렷이 나쁨 (평균 비 %.2f, 가장 나쁜 속도 %.2f)' % (e['A_ratio'], e['A_ratio_max']))
        e['out'] = '; '.join(why); rows[tag] = e
    c = rows.get(cur)
    if c is None or not c['complete']: return cur, '현재 값의 구간이 다 끝나지 않아 견줄 수 없다: 현재 값을 그대로 쓴다 (판정 불가)', rows
    elig = [e for e in rows.values() if not e['out'] and e['B_ratio'] is not None]
    if not elig: return cur, '조건을 채운 후보가 없다: 현재 값을 그대로 쓴다', rows
    best = min(elig, key=lambda e: e['B_ratio'])
    if best['tag'] == cur: return cur, '현재 값이 조건을 채운 후보 가운데 흔들림이 가장 작다: 현재 값을 그대로 쓴다', rows
    both = all(best['B_round'][r] is not None and c['B_round'][r] is not None and best['B_round'][r] < c['B_round'][r] for r in (0, 1))
    if best['B_ratio'] <= S2_B_NEED and both: return best['tag'], '후보 %s (%s): 흔들림이 현재 값의 %.2f 배이고 두 바퀴 모두 현재 값보다 작다' % (best['tag'], best['label'], best['B_ratio']), rows
    return cur, '흔들림이 가장 작은 후보 %s 도 현재 값보다 뚜렷이 낫지 않다 (비 %.2f, 필요 %.2f 이하 + 두 바퀴 모두 작을 것: %s): 현재 값을 그대로 쓴다' % (best['tag'], best['B_ratio'], S2_B_NEED, '채움' if both else '못 채움'), rows


def stage2(D):
    res, es, plan = blocks_of(D, S2_SKIP); meta = next((s for s in plan if s.get('type') == 'meta'), {}); cands = [(c['tag'], c['label'], c['gains']) for c in meta.get('candidates', [])]
    seen = {}; per = {}
    for e in res:
        tag = e['tag'][1:]; key = (e['walk'], tag); e['round'] = seen.get(key, 0)
        if e['status'] in ('ok', 'personlost', 'too short'): seen[key] = e['round'] + 1
        per.setdefault(tag, []).append(e)
    tag, reason, rows = choose(cands, per) if cands else ('A', 'plan_used.json 에 후보 목록이 없다: 현재 값을 그대로 쓴다 (판정 불가)', {})
    gains = dict(next((g for t, _, g in cands if t == tag), dict(kp=30.0, ki=0.0, kd=6.0, il=3.0, db=0.06)))
    choice = dict(folder=os.path.abspath(D), tag=tag, gains=gains, reason=reason, dry=(es or {}).get('dry_truth_px') is not None, assumed_plan=bool(meta.get('assumed')), candidates=list(rows.values()), blocks=res, end_state=es, why_candidates=meta.get('why'))
    json.dump(choice, open(os.path.join(D, 'STAGE2_CHOICE.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    g = lambda d: 'Kp %g · Ki %g (한계 %g) · Kd %g · 불감대 %g' % (d['kp'], d['ki'], d['il'], d['kd'], d['db']); n = lambda v, p='%.2f': '' if v is None else p % v
    L = ['# 2단계 — 속도 상한 100 에서 PD 값 고르기 (analyze_v27.py stage2 가 만든 파일)', '', '- 기록 폴더: %s' % choice['folder'], '- 구간 기록 %d줄 (끝까지 %d, 화면 맞추기가 끊겨 버린 시도 %d, 사람을 놓쳐 끝난 구간 %d, 빠른 걸음에서 두 번 끊겨 건너뛴 구간 %d)' % (len(res), sum(e['status'] == 'ok' for e in res), sum(e['status'] == 'lost' for e in res), sum(e['status'] == 'personlost' for e in res), sum(e['status'] == 'skipped' for e in res)),   # v27c: + skipped
         '- 값: 각 구간 앞 %g초 제외. 비 = 같은 걷는 속도에서 (후보 ÷ 현재 값) 을 속도별로 구해 평균한 것.' % S2_SKIP, *dry_note(es), *(['- **이 계획의 후보는 가정한 숫자로 만든 것이다 (확인용).**', ''] if meta.get('assumed') else []),
         '| 후보 | 값 | 끝까지 돈 구간 | 평균 중심 오차 px | 오차 비 (평균 / 가장 나쁜 속도) | 흔들림 px | 흔들림 비 | 흔들림 1바퀴 / 2바퀴 px | 놓친 비율 최고 | 사람을 놓쳐 끝난 구간 | 화면 맞추기 끊김 (버린 시도) | 건너뛴 구간 | 탈락 이유 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']   # v27c: + three count columns
    for t, label, gg in cands:
        e = rows[t]; L.append('| %s%s (%s) | %s | %d | %s | %s / %s | %s | %s | %s / %s | %.1f %% | %d | %d | %d | %s |' % (t, ' **(고름)**' if t == tag else '', label, g(gg), e['blocks'], n(e['A']), n(e['A_ratio']), n(e['A_ratio_max']), n(e['B']), n(e['B_ratio']), n(e['B_round'][0]), n(e['B_round'][1]), 100 * e['c_max'], e['lost_blocks'], e['reg_lost'], e['skipped'], e['out'] or '-'))   # v27c: + the counts
    L += ['', '- **판정 (미리 적은 규칙): %s**' % reason, '- **3단계에 쓸 값: %s**' % g(gains), '- 후보마다 걷는 속도별로 2구간씩뿐이다 (한 번의 실행). "뚜렷이 낫다"의 문턱(흔들림 비 %.2f 이하 + 두 바퀴 모두)을 못 넘으면 현재 값을 그대로 쓴다.' % S2_B_NEED, '',
          '## 구간별', '', '| 구간 | 상태 | 표본 | 평균 중심 오차 px | 흔들림 px | 놓친 비율 | 방향 뒤집힘 /분 | 최고 속도 명령 |', '|---|---|---|---|---|---|---|---|']
    for e in res:
        if e['status'] == 'ok': L.append('| %s | 끝까지 | %d | %.2f | %.2f | %.1f %% | %.1f | %d |' % (e['name'], e['n'], e['A_mean_abs_dx'], e['B_shake'], 100 * e['C_lost'], e['reversals_per_min'], e['spd_max']))
        else: L.append('| %s | %s | | | | | | |' % (e['name'], {'lost': '화면 맞추기 끊김 (버림)', 'personlost': '사람을 놓쳐 끝남 (세로 지킴이로 끝난 것 포함)', 'too short': '너무 짧음', 'skipped': '건너뜀 (빠른 걸음에서 두 번 끊김, 돌지 않음)'}.get(e['status'], e['status'])))   # v27c: + skipped
    out(D, 'STAGE2_TABLE_KO.md', L + end_lines(es)); return choice


# ---------------------------------------------------------------- stage 3
def log_interval(r):
    """mean and 95 % interval of the per-pair ratios, on the log scale (geometric mean)"""
    r = np.log(np.array(r, float)); n = len(r)
    if n < 2: return dict(pairs=n, ratio=round(float(np.exp(r.mean())), 3) if n else None, lo=None, hi=None)
    h = T975.get(n - 1, 2.0) * float(r.std(ddof=1)) / n ** 0.5
    return dict(pairs=n, ratio=round(float(np.exp(r.mean())), 3), lo=round(float(np.exp(r.mean() - h)), 3), hi=round(float(np.exp(r.mean() + h)), 3))


def verdict(pairs, need, walks=None):
    """the stage 3 rule of the plan, on a list of (v25 block, v27 block). A block that ended as "person lost" has status personlost and counts as lost ratio 1."""
    full = [(a, b) for a, b in pairs if a['status'] == 'ok' and b['status'] == 'ok']
    if len(full) < need or any(a['n'] < S3_MIN_N or b['n'] < S3_MIN_N for a, b in full) or (walks is not None and any(not any(a['walk'] == w for a, _ in full) for w in walks)): return dict(verdict='표본 부족', pairs=len(full))
    o = dict(pairs=len(full)); cl = lambda e: 1.0 if e['status'] == 'personlost' else e['C_lost']
    for key in ('A_mean_abs_dx', 'B_shake'):
        o[key] = dict(v25=round(float(np.mean([a[key] for a, _ in full])), 3), v27=round(float(np.mean([b[key] for _, b in full])), 3), **log_interval([max(b[key], 1e-6) / max(a[key], 1e-6) for a, b in full]))
    for key in ('starts_per_min', 'stops_per_min', 'reversals_per_min'): o[key] = dict(v25=round(float(np.mean([a[key] for a, _ in full])), 2), v27=round(float(np.mean([b[key] for _, b in full])), 2))
    c25 = float(np.mean([cl(a) for a, _ in pairs])); c27 = float(np.mean([cl(b) for _, b in pairs])); dC = c27 - c25; o['C_lost'] = dict(v25=round(c25, 4), v27=round(c27, 4), diff=round(dC, 4))
    A, B = o['A_mean_abs_dx'], o['B_shake']
    shake_less = B['ratio'] <= R_GOOD and B['hi'] is not None and B['hi'] < 1; shake_more = B['ratio'] >= R_BAD and B['lo'] is not None and B['lo'] > 1; err_worse = A['ratio'] >= R_BAD and A['lo'] is not None and A['lo'] > 1
    if shake_more or dC > C_BAD: v = '나빠짐'
    elif shake_less and err_worse and dC <= C_GOOD: v = '맞바꿈 (덜 흔들리지만 가운데에서 더 벗어남)'
    elif err_worse: v = '나빠짐'
    elif shake_less and dC <= C_GOOD: v = '덜 흔들림'
    else: v = '차이 없음'
    o.update(verdict=v, lost_diff=round(dC, 4)); return o


def stage3(D):
    res, es, plan = blocks_of(D, S3_SKIP); meta = next((s for s in plan if s.get('type') == 'meta'), {})
    final = {}
    for e in res:
        if e['status'] in ('ok', 'personlost', 'too short'): final[e['k']] = e
    pairs = []
    for k in sorted(final):
        if k % 2 == 0 and k + 1 in final and final[k]['walk'] == final[k + 1]['walk'] and {final[k]['tag'], final[k + 1]['tag']} == {'v25', 'v27'}:
            a, b = (final[k], final[k + 1]) if final[k]['tag'] == 'v25' else (final[k + 1], final[k]); pairs.append((a, b))
    walks = [w for w in dict.fromkeys(s.get('walk') for s in plan if s.get('type') == 'track') if w] or list(dict.fromkeys(e['walk'] for e in res))
    need = S3_MIN_PAIRS if len(walks) >= 6 else S3_MIN_PAIRS_5   # v27c (second review; plan section 3): 6 walks = 12 pairs planned -> 10 complete pairs needed (as before); 5 walks = 10 pairs planned -> 8
    regl = lambda tag, w=None: sum(e['status'] == 'lost' and e['tag'] == tag and (w is None or e['walk'] == w) for e in res)   # v27c: attempts thrown away after a lost picture matching, per program
    o = dict(folder=os.path.abspath(D), gains=meta.get('gains'), blocks=res, pairs=[[a['name'], b['name']] for a, b in pairs], per_walk={w: verdict([p for p in pairs if p[0]['walk'] == w], 2) for w in walks},
             all=verdict(pairs, need, walks), pairs_needed=need, walks=walks, attempts=len(res), lost_attempts=sum(e['status'] == 'lost' for e in res), lost_attempts_by_program={'v25': regl('v25'), 'v27': regl('v27')}, skipped_blocks=sum(e['status'] == 'skipped' for e in res),
             person_lost=sum(e['status'] == 'personlost' for e in res), end_state=es, dry=(es or {}).get('dry_truth_px') is not None)   # v27c: need (was S3_MIN_PAIRS), + pairs_needed, walks, lost_attempts_by_program, skipped_blocks
    json.dump(o, open(os.path.join(D, 'STAGE3_ANALYSIS.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    gg = meta.get('gains') or {}
    L = ['# 3단계 — v25 대 v27, 실제 짐벌 + 가상의 사람 (analyze_v27.py stage3 이 만든 파일)', '', '- 기록 폴더: %s' % o['folder'], '- 쓴 값: %s' % ('Kp %g · Ki %g (한계 %g) · Kd %g · 불감대 %g' % (gg['kp'], gg['ki'], gg['il'], gg['kd'], gg['db']) if gg else '계획 파일에 없음'),
         '- 구간 기록 %d줄, 그중 화면 맞추기가 끊겨 버린 시도 %d번 (v25 %d번, v27 %d번), 사람을 놓쳐 끝난 구간 %d개 (세로 지킴이로 끝난 것 포함), 빠른 걸음에서 두 번 끊겨 건너뛴 구간 %d개, 짝 %d개' % (o['attempts'], o['lost_attempts'], regl('v25'), regl('v27'), o['person_lost'], o['skipped_blocks'], len(pairs)),   # v27c: + per program, skipped
         '- 걸음 %d가지: %s. 판정에 필요한 끝까지 돈 짝: %d개 이상 (계획에 미리 적음: 5가지면 8, 6가지면 10)' % (len(walks), ', '.join(walks), need),   # v27c
         '- 값: 각 구간 앞 %g초 제외, 끝까지 돈 구간만. 흔들림 = 가운데 오차에서 1초 이동 평균을 뺀 나머지의 표준편차. 비 = 짝마다 (v27 ÷ v25) 를 구해 곱 평균한 것.' % S3_SKIP, *dry_note(es), *(['- **이 계획은 가정한 값으로 만든 것이다 (확인용).**', ''] if meta.get('assumed') else []),
         '| 걸음 | 짝 | 값 | v25 | v27 | 비 v27 ÷ v25 (95 % 구간) | 판정 |', '|---|---|---|---|---|---|---|']
    names = (('B_shake', '흔들림 px (주 지표)'), ('A_mean_abs_dx', '평균 중심 오차 px'), ('C_lost', '놓친 비율'), ('reversals_per_min', '방향 뒤집힘 /분 (참고)'), ('starts_per_min', '출발 /분 (참고)'))
    for label, v in [('**전체**', o['all'])] + [(w, o['per_walk'][w]) for w in walks]:
        lrow = '| | | 화면 맞추기 끊김 (버린 시도, 번 · 참고) | %d | %d | | |' % (regl('v25', None if label == '**전체**' else label), regl('v27', None if label == '**전체**' else label))   # v27c (second review): the count per program, always shown
        if 'B_shake' not in v: L.append('| %s | %d | | | | | **%s** |' % (label, v['pairs'], v['verdict'])); L.append(lrow); continue   # v27c: + lrow
        for i, (key, kn) in enumerate(names):
            x = v[key]
            if key == 'C_lost': mid = '%.1f %% → %.1f %% (차이 %+.1f %%p)' % (100 * x['v25'], 100 * x['v27'], 100 * x['diff']); a, b = '%.1f %%' % (100 * x['v25']), '%.1f %%' % (100 * x['v27'])
            elif 'ratio' in x: mid = '%.2f (%s)' % (x['ratio'], '구간 없음' if x['lo'] is None else '%.2f ~ %.2f' % (x['lo'], x['hi'])); a, b = '%.2f' % x['v25'], '%.2f' % x['v27']
            else: mid = ''; a, b = '%.1f' % x['v25'], '%.1f' % x['v27']
            L.append('| %s | %s | %s | %s | %s | %s | %s |' % (label if i == 0 else '', v['pairs'] if i == 0 else '', kn, a, b, mid, ('**%s**' % v['verdict'] if label == '**전체**' else '(참고) %s' % v['verdict']) if i == 0 else ''))
        L.append(lrow)                                 # v27c
    L += ['', '- **공식 판정은 "전체" 줄 하나다: %s.** 걸음별 줄은 짝이 2개뿐이라 참고로만 읽는다 (구간이 나오지 않는다).' % o['all']['verdict'],
          '- 판정 규칙: PLAN_V27_HILS_20261111_KO.md 3단계 (실행 전에 적음). 출발·방향 뒤집힘·화면 맞추기 끊김 횟수는 판정에 쓰지 않는다. 한 번의 실행, 한 장소, 가상의 사람이다.', '',
          '## 구간별', '', '| 구간 | 프로그램 | 상태 | 표본 | 평균 중심 오차 px | 흔들림 px | 놓친 비율 | 출발 /분 | 방향 뒤집힘 /분 | 최고 속도 명령 |', '|---|---|---|---|---|---|---|---|---|---|']
    for e in res:
        if e['status'] == 'ok': L.append('| %s | %s | 끝까지 | %d | %.2f | %.2f | %.1f %% | %.1f | %.1f | %d |' % (e['name'], e['tag'], e['n'], e['A_mean_abs_dx'], e['B_shake'], 100 * e['C_lost'], e['starts_per_min'], e['reversals_per_min'], e['spd_max']))
        else: L.append('| %s | %s | %s | | | | | | | |' % (e['name'], e['tag'], {'lost': '화면 맞추기 끊김 (버림)', 'personlost': '사람을 놓쳐 끝남 (세로 지킴이로 끝난 것 포함. 놓친 비율 100 %로 셈)', 'too short': '너무 짧음', 'skipped': '건너뜀 (빠른 걸음에서 두 번 끊김, 돌지 않음)'}.get(e['status'], e['status'])))   # v27c: + skipped
    out(D, 'STAGE3_TABLE_KO.md', L + end_lines(es)); return o


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('stage1', 'stage2', 'stage3'): sys.exit(__doc__)
    {'stage1': stage1, 'stage2': stage2, 'stage3': stage3}[sys.argv[1]](sys.argv[2])
