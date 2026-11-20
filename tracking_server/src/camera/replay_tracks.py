"""녹화 재생의 추적 결과와 검출을 파일로 남긴다 (화면은 바꾸지 않음).

8090 녹화 재생의 첫 회차(처음부터 끝까지 한 번)를 프레임마다 MOT 형식으로 적는다.
  - 추적기를 바꿔 같은 녹화를 다시 재생하면 추적기별 결과 폴더가 하나씩 생기고
  - 검출 파일로 다른 추적기를 PC에서 같은 검출에 다시 돌릴 수 있다 (mb_tracks.py)

저장 위치: tracking_server/result/replay_tracks/<시각>_<추적기>/
  (서버 출력 폴더 real_gui_1 등은 재시작 때 바뀌거나 지워질 수 있어 밖의 고정 폴더에 둔다.
   result/는 .gitignore에 넣어 공개 저장소에 올라가지 않게 한다: 녹화에서 나온 자료다)
  <녹화>.txt                 화면에 보이는 최종 결과 (KeepID · 상자 다듬기까지 거친 tr_rows)
  <녹화>_tracker.txt         추적기만의 결과 (KeepID 전 base_rows). KeepID가 켜져 있을 때만 생김
  <녹화>-SERVER/det/det.txt  추적기에 들어간 검출
  info.txt                   녹화 이름, 장 수, 장 구간별 추적기, KeepID 사용 여부
  MOT 형식: frame,id,x,y,w,h,conf,-1,-1,-1 (frame = 녹화의 1부터 센 장 번호) · 검출은 frame,-1,x,y,w,h,conf,class,-1,-1
  좌표는 카메라 640x480 기준 (TRACKS.jsonl과 같게 위쪽 여백 FRAME_DY를 뺌).

폴더 이름의 추적기는 회차가 끝날 때 마지막 장에서 쓰인 추적기로 정한다. 메뉴에서 추적기를 바꾼 뒤 바로 재생하면
변경이 재생 첫 장 뒤에 적용될 수 있는데, 이처럼 회차 중에 추적기가 바뀌면 이름 끝에 '_변경'을 붙이고 info.txt에
몇 번째 장부터 바뀌었는지 적는다. 중간에 장을 건너뛰거나(±10초) 되돌리면 거기서 닫고 '_중단'을 붙인다.
"""
import os
import re
import time

from ..common.shared_values import HERE
from ..common.state import APP

RESULT_DIR = os.path.join(str(HERE), 'result', 'replay_tracks')

_S = dict(fh=None, bh=None, dh=None, dir=None, stamp='', name='', last=-1, rows=0, kinds=[], keepid=False)


def _safe(s):
    return re.sub(r'[^\w.-]+', '_', str(s)).strip('_')[:80] or 'replay'


def _name():
    rp = getattr(APP.LIVE, 'replay', None) if getattr(APP, 'LIVE', None) is not None else None
    name = rp.get('name') if isinstance(rp, dict) else None
    return _safe(os.path.basename(str(name).rstrip('/'))) if name else 'replay'


def _free(base):
    d, k = base, 2
    while os.path.exists(d):   # 같은 초에 두 번 시작해도 덮어쓰지 않게
        d = '%s_%d' % (base, k); k += 1
    return d


def _close(note=''):
    for k in ('fh', 'bh', 'dh'):
        if _S[k] is not None:
            _S[k].close(); _S[k] = None
    d = _S['dir']
    if d is not None:
        kinds = _S['kinds']                          # [(첫 장, 추적기), ...]
        last_kind = kinds[-1][1] if kinds else 'unknown'
        tail = ('_변경' if len(kinds) > 1 else '') + note
        try:
            with open(os.path.join(d, 'info.txt'), 'w', encoding='utf-8') as fi:
                fi.write('녹화: %s\n장 수: %d\nKeepID: %s\n' % (_S['name'], _S['last'] + 1, '켜짐' if _S['keepid'] else '꺼짐'))
                for i, (f0, kind) in enumerate(kinds):
                    f1 = kinds[i + 1][0] - 1 if i + 1 < len(kinds) else _S['last'] + 1
                    fi.write('추적기: %s (%d ~ %d장)\n' % (kind, f0, f1))
                if note:
                    fi.write('중단: %d장 뒤에 건너뛰기 · 되돌리기\n' % (_S['last'] + 1))
            final = _free(os.path.join(RESULT_DIR, '%s_%s%s' % (_S['stamp'], _safe(last_kind), tail)))
            os.replace(d, final); d = final
        except OSError as e:
            print('REPLAY tracks rename failed:', repr(e)[:120], flush=True)
        print('REPLAY tracks saved: %s (%d frames, %d rows)' % (d, _S['last'] + 1, _S['rows']), flush=True)
    _S.update(dir=None, last=-1, rows=0, kinds=[], keepid=False)


def _write(fh, f, rows, dy):
    for r in rows:
        x1, y1, x2, y2 = float(r[0]), float(r[1]) - dy, float(r[2]), float(r[3]) - dy
        fh.write('%d,%d,%.2f,%.2f,%.2f,%.2f,%.5f,-1,-1,-1\n' % (f, int(r[4]), x1, y1, x2 - x1, y2 - y1, float(r[5])))


def frame(eval_pos, tr_rows, arr, dy, tracker_key, base_rows=None):
    """프레임 루프에서 매 프레임 부른다.
    eval_pos = replay_eval.first_pass_position (라이브 카메라 · 반복 회차 · 정답 없는 녹화는 None)
    tracker_key = 이 장에서 쓰인 추적기 이름, base_rows = KeepID 전 추적기 결과 (tr_rows와 같은 객체면 따로 안 씀)"""
    try:
        if eval_pos is None:
            if _S['fh'] is not None:
                _close()
            return
        if eval_pos == 0:
            if _S['fh'] is not None:
                _close()
            _S['stamp'] = time.strftime('%Y%m%d_%H%M%S'); _S['name'] = name = _name()
            d = _free(os.path.join(RESULT_DIR, '%s_기록중' % _S['stamp']))
            os.makedirs(os.path.join(d, name + '-SERVER', 'det'), exist_ok=True)
            _S['fh'] = open(os.path.join(d, name + '.txt'), 'w', buffering=1)
            _S['dh'] = open(os.path.join(d, name + '-SERVER', 'det', 'det.txt'), 'w', buffering=1)
            _S['dir'] = d
        elif _S['fh'] is None:
            return
        elif eval_pos == _S['last']:          # 같은 장이 한 번 더 온 경우 (멈춤 등): 무시
            return
        elif eval_pos != _S['last'] + 1:      # 건너뛰기 · 되돌리기: 여기까지만 남긴다
            _close('_중단')
            return
        f = eval_pos + 1
        key = str(tracker_key)
        if not _S['kinds'] or _S['kinds'][-1][1] != key:
            _S['kinds'].append((f, key))
        _write(_S['fh'], f, tr_rows, dy)
        _S['rows'] += len(tr_rows)
        if base_rows is not None and base_rows is not tr_rows:   # KeepID가 번호를 바꾼 경우: 추적기 원래 결과도
            if _S['bh'] is None:
                _S['bh'] = open(os.path.join(_S['dir'], _S['name'] + '_tracker.txt'), 'w', buffering=1)
                _S['keepid'] = True
            _write(_S['bh'], f, base_rows, dy)
        for r in arr:
            x1, y1, x2, y2 = float(r[0]), float(r[1]) - dy, float(r[2]), float(r[3]) - dy
            _S['dh'].write('%d,-1,%.2f,%.2f,%.2f,%.2f,%.5f,%d,-1,-1\n' % (f, x1, y1, x2 - x1, y2 - y1, float(r[4]), int(r[5])))
        _S['last'] = eval_pos
    except Exception as e:   # 기록 실패가 추적을 멈추지 않게
        print('REPLAY tracks not saved:', repr(e)[:120], flush=True)
        for k in ('fh', 'bh', 'dh'):
            try:
                if _S[k] is not None: _S[k].close()
            except Exception:
                pass
            _S[k] = None
        _S['dir'] = None


def close():
    """서버 종료 때 (열린 파일 닫기)"""
    if _S['fh'] is not None:
        _close()