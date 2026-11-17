"""make_tables.py: Markdown tables for SUMMARY_KO.md from results/pos_valB.json and results/ref_valB.json (no new computation).
Usage: python make_tables.py <root>  -> results/tables.md"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]); R = root / 'results'; out = []
f3 = lambda v: '–' if v is None else ('%.3f' % v[0] if isinstance(v, list) else '%.3f' % v)
ci = lambda v: '–' if not v else '%.3f [%.3f, %.3f]' % (v[0], v[1], v[2])
DSN = dict(TMOT='TMOT (열화상)', DT='DanceTrack')

if (R / 'pos_valB.json').exists():
    P = json.loads((R / 'pos_valB.json').read_text())
    M = [('kalman', 'Kalman'), ('current_gru_kpt', '지금 GRU'), ('new_gru_kpt', '새 GRU'), ('current_mamba_kpt', '지금 Mamba'), ('new_mamba_kpt', '새 Mamba')]
    for tname, tk in (('detector_box', '검출기 상자 목표 (주)'), ('head_to_hip', '몸 네모 하나 목표 (참고)')):
        for ds in ('TMOT', 'DT'):
            t = P['tables'].get('%s|%s|pooled' % (tname, ds))
            if not t: continue
            out.append('\n#### 위치 기억 · %s · %s (val-B %d편, 근거리+원거리 검출 합침, 오차/키, 낮을수록 좋음, [90%% 구간])\n' % (DSN[ds], tk, len(t['sequences'])))
            out.append('| 경우 | 수 | ' + ' | '.join(n for _, n in M) + ' |'); out.append('|' + '---|' * (len(M) + 2))
            for g, gn in (('all', '전체'), ('sim', '가린 척(검출 있음)'), ('real', '실제 놓침'), ('syn', '합성 사라짐'), ('all|<50', '키<50'), ('all|50-100', '키 50-100'), ('all|>=100', '키≥100')):
                v = t['groups'].get(g)
                if not v: continue
                out.append('| %s | %d | ' % (gn, v['n']) + ' | '.join(ci(v['score'].get(m)) for m, _ in M) + ' |')
            d = t['groups']['all']['diff']
            out.append('\n차이(전체): ' + '; '.join('%s %s' % (k.replace('_kpt', ''), ci(v)) for k, v in d.items()))
    out.append('\n이김 판정: ' + json.dumps(P.get('beats'), ensure_ascii=False))

if (R / 'ref_valB.json').exists():
    Q = json.loads((R / 'ref_valB.json').read_text())
    for ds in ('TMOT', 'DT'):
        for stream, sn in (('pooled', '근거리·원거리 따로 (합침)'), ('sw', '근거리↔원거리 2~6초마다 바꿈')):
            t = Q['tables'].get('%s|%s' % (ds, stream))
            if not t: continue
            ms = [m for m in ('raw', 'tracker', 'euro', 'refiner', 'euro_nocomp') if m in t['methods']] + [m for m in t['methods'] if '+blend' in m] + ['target']
            names = dict(raw='검출 그대로', tracker='추적기(지금 화면)', euro='One Euro', euro_nocomp='One Euro(카메라 보정 없음)', refiner='학습 GRU', target='목표(참고)')
            out.append('\n#### 상자 · %s · %s (val-B %d편)\n' % (DSN[ds], sn, len(t['sequences'])))
            out.append('| 방법 | 중심 오차/키 | 조준점 오차/키 | IoU | 흔들림 중심 RMS | 흔들림 조준점 RMS | 키 변화 p99 | 조준점 변화 p99 | 키/목표 (작은 사람<50) |')
            out.append('|---|---|---|---|---|---|---|---|---|')
            for m in ms:
                a = t['methods'][m].get('all', {}); s = t['methods'][m].get('<50', {})
                nm = names.get(m, m.replace('+blend', ' + 섞기 ') + '초')
                out.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (nm, ci(a.get('ce')), ci(a.get('aim')), f3(a.get('iou')), ci(a.get('d2c_rms')), ci(a.get('d2a_rms')),
                                                                         f3(a.get('dh_p99')), f3(a.get('da_p99')), ('%.2f' % s['h_over_target'][0]) if s.get('h_over_target') else '–'))
            # events and lag in a second, smaller table
            out.append('\n| 방법 | 바꾼 순간 조준점 변화 p95 | 바꾼 뒤 0.5초 조준점 변화 평균 | 100 px 지날 때 키 변화 p95 | 가려질 때 조준점 변화 p95 | 가장 큰 한 번 튐(조준점, 중앙값) | 빠를 때(>0.1 키/장) 뒤처짐 |')
            out.append('|---|---|---|---|---|---|---|')
            for m in ms:
                e = t['methods'][m].get('events', {}); lg = t['methods'][m].get('lag', {}); mx = t['methods'][m].get('max_jump_per_track', {})
                nm = names.get(m, m.replace('+blend', ' + 섞기 ') + '초')
                out.append('| %s | %s | %s | %s | %s | %s | %s |' % (nm, f3(e.get('switch0', {}).get('da_p95')), f3((e.get('switch', {}).get('da_mean') or [None])[0]),
                                                                f3(e.get('size', {}).get('dh_p95')), f3(e.get('occl', {}).get('da_p95')), f3(mx.get('aim_median')),
                                                                f3((lg.get('0.1-99', {}).get('along') or [None])[0])))
            dd = t.get('diffs', {})
            for k in ('refiner - raw', 'refiner - tracker', 'refiner - euro'):
                if k in dd: out.append('\n%s: 중심 오차 %s, 조준점 오차 %s, 중심 흔들림(제곱) %s' % (k, ci(dd[k].get('ce')), ci(dd[k].get('aim')), ci(dd[k].get('d2c_sq'))))
(R / 'tables.md').write_text('\n'.join(out), encoding='utf-8'); print('\n'.join(out))
