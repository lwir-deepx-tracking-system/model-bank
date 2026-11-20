"""녹화 재생의 추적 결과와 검출을 파일로 남긴다 (화면은 바꾸지 않음).

8090 녹화 재생의 첫 회차(처음부터 끝까지 한 번)를 프레임마다 MOT 형식으로 적는다. 화면에 보이는 것과 같은 최종 추적
결과(KeepID까지 거친 tr_rows)와, 추적기에 들어간 검출(arr)을 함께 적으므로
  - 추적기를 바꿔 같은 녹화를 다시 재생하면 추적기별 결과 파일이 하나씩 생기고
  - 검출 파일로 다른 추적기를 PC에서 같은 검출에 다시 돌릴 수 있다 (mb_tracks.py)

저장 위치: <서버 출력 폴더>/replay_tracks/<시각>_<추적기>/
  <녹화>.txt              추적 결과 MOT: frame,id,x,y,w,h,conf,-1,-1,-1   (frame = 녹화의 1부터 센 장 번호)
  <녹화>-SERVER/det/det.txt 검출 MOT:  frame,-1,x,y,w,h,conf,class,-1,-1
좌표는 카메라 640x480 기준 (TRACKS.jsonl과 같게 위쪽 여백 FRAME_DY를 뺌).

첫 회차를 끊김 없이 재생한 것만 남긴다. 중간에 장을 건너뛰거나(±10초) 되돌리면 그 파일은 거기서 닫고 이름 끝에
'_중단'을 붙인다 (건너뛴 뒤의 번호는 처음부터 이어 재생한 것과 다르기 때문).
"""
import os
import re
import time

from ..common.state import APP

_S = dict(fh=None, dh=None, dir=None, last=-1, rows=0)


def _safe(s):
    return re.sub(r'[^\w.-]+', '_', str(s)).strip('_')[:80] or 'replay'


def _name():
    rp = getattr(APP.LIVE, 'replay', None) if getattr(APP, 'LIVE', None) is not None else None
    name = rp.get('name') if isinstance(rp, dict) else None
    return _safe(os.path.basename(str(name).rstrip('/'))) if name else 'replay'


def _close(note=''):
    for k in ('fh', 'dh'):
        if _S[k] is not None:
            _S[k].close(); _S[k] = None
    if _S['dir'] is not None:
        d = _S['dir']
        if note:
            try:
                os.replace(d, d + note); d = d + note
            except OSError:
                pass
        print('REPLAY tracks saved: %s (%d frames, %d rows)%s' % (d, _S['last'] + 1, _S['rows'], ' ' + note if note else ''), flush=True)
    _S.update(dir=None, last=-1, rows=0)


def frame(eval_pos, tr_rows, arr, dy, tracker_key):
    """프레임 루프에서 매 프레임 부른다. eval_pos = replay_eval.first_pass_position (라이브 카메라·반복 회차는 None)."""
    try:
        if eval_pos is None:
            if _S['fh'] is not None:
                _close()
            return
        if eval_pos == 0:
            if _S['fh'] is not None:
                _close()
            out = getattr(APP, 'out', None)
            if out is None:
                return
            name = _name()
            base = os.path.join(str(out), 'replay_tracks', '%s_%s' % (time.strftime('%Y%m%d_%H%M%S'), _safe(tracker_key)))
            d, k = base, 2
            while os.path.exists(d) or os.path.exists(d + '_중단'):   # 같은 초에 두 번 시작해도 덮어쓰지 않게
                d = '%s_%d' % (base, k); k += 1
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
        for r in tr_rows:
            x1, y1, x2, y2 = float(r[0]), float(r[1]) - dy, float(r[2]), float(r[3]) - dy
            _S['fh'].write('%d,%d,%.2f,%.2f,%.2f,%.2f,%.5f,-1,-1,-1\n' % (f, int(r[4]), x1, y1, x2 - x1, y2 - y1, float(r[5])))
            _S['rows'] += 1
        for r in arr:
            x1, y1, x2, y2 = float(r[0]), float(r[1]) - dy, float(r[2]), float(r[3]) - dy
            _S['dh'].write('%d,-1,%.2f,%.2f,%.2f,%.2f,%.5f,%d,-1,-1\n' % (f, x1, y1, x2 - x1, y2 - y1, float(r[4]), int(r[5])))
        _S['last'] = eval_pos
    except Exception as e:   # 기록 실패가 추적을 멈추지 않게
        print('REPLAY tracks not saved:', repr(e)[:120], flush=True)
        _S['fh'] = _S['dh'] = None; _S['dir'] = None


def close():
    """서버 종료 때 (열린 파일 닫기)"""
    if _S['fh'] is not None:
        _close()
