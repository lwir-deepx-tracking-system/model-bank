"""녹화 재생 (2026-11-12): 웹 화면에서 라벨 사이트의 녹화 폴더 하나를 골라 카메라 대신 서버에 넣고, 끝나면 진짜 카메라로 돌아간다.

- 녹화 목록: --replay-root 아래에서 images/ 폴더에 번호 PNG(000001.png ...)가 있는 폴더. 이름 = root 기준 상대 경로. 60초 동안 기억해 둔다
  (PNG 를 열지 않고 첫 파일의 머리 33 바이트만 읽어 8/16 비트와 크기를 본다).
- 재생: 카메라 프로그램 자리에 replay_cam.py 를 띄운다. 띄우는 길은 카메라와 같다 (camera_link.lv_spawn / camera_extra.spawn 이 cam_argv() 를 부른다).
  바꾸는 동안은 카메라 초기화 버튼과 같은 LIVE.cam_hold 로 lv_cam_wait 가 다른 프로그램을 띄우지 못하게 한다.
- 짐벌 안전: 재생을 시작할 때 짐벌 프로그램에 멈춤(H). 재생 중(바꾸는 중 포함)에는 짐벌에 가는 줄에 사람이 없고(blocked), /gimbal/click·move 는 거절한다.
  재생을 멈춘 뒤에도, 멈추기 전에 도착한 프레임(처리 중이던 몇 장)은 사람 없이 보낸다 (도착 시각 < src_t). 멈출 때도 H 를 보낸다.
  옛 영상의 마지막 프레임과 새 영상의 첫 프레임 사이는 언제나 1.5 초보다 길어서, 전부터 있던 "1초 넘는 입력 끊김" 규칙이 추적을 새로 시작하고 고른 사람을 놓는다.
- 측정 버튼 녹화도 목록에 나온다 (11-12): --measure-dir 아래 <날짜>/MEASURE_*/ 의 frames.raw.gz (16비트 원본). 이름 = '측정/<날짜>/<MEASURE 폴더>'.
  정답 라벨이 없으므로 채점하지 않는다. 메모리 한도(LWIR_REPLAY_MAX_MB, 기본 400 MB)까지 앞에서부터만 재생한다.
- 재생을 한 번도 쓰지 않으면 이 파일은 아무것도 바꾸지 않는다 (cam_argv() 는 전과 같은 명령을 돌려준다, LIVE.replay 는 None).
- 정답과 비교 (11-12, replay_eval.py): 재생 카메라는 늘 프레임 머리(LWIR_CAM_STAMP=1)와 PNG 번호(--tag)를 붙여 띄운다 (spawn).
  카메라 쪽 설정(LWIR_CAM_STAMP)과 상관없이 재생 프로그램에만 그렇게 하고, 진짜 카메라를 띄우는 길은 바꾸지 않는다.
"""
import os
import re
import sys
import time

from ..common.state import APP
from ..common.shared_values import HERE
from . import camera_extra
from . import replay_eval

FPS_MIN, FPS_MAX, FPS_DEF = 5.0, 30.0, 25.0
LIST_TTL = 60.0
MAX_DEPTH = 4                     # root 아래 이 깊이까지 찾는다 (opi_sequences/seq_018 = 2)
NUM_PNG = re.compile(r'^\d+\.png$', re.I)
NOTE_8BIT = '8비트 녹화: 온도 Filter 가 맞지 않아 사람이 걸러질 수 있음. 보려면 온도 Filter 를 잠시 끄세요'
FRAME_BYTES = 640 * 480 * 2
_C = dict(t=-1e9, items=[], root=None)


def root():   # 쓸 수 있는 녹화 폴더 (없으면 None)
    r_ = getattr(APP.args, 'replay_root', '') or ''
    return r_ if r_ and os.path.isdir(r_) else None


MEAS = '측정'


def mroot():   # 측정 버튼 녹화 폴더 (없으면 None). 라벨 폴더를 못 쓰는 실행에서도 목록에 나오려면 --replay-root 가 있어야 한다 (재생을 켜는 스위치)
    m_ = getattr(APP.args, 'measure_dir', '') or ''
    return m_ if getattr(APP.args, 'replay_root', '') and m_ and os.path.isdir(m_) else None


def on():   # 재생 기능을 쓸 수 있는가 (라벨 폴더나 측정 폴더 가운데 하나)
    return root() is not None or mroot() is not None


def _png_head(p_):   # (비트, 폭, 높이, 색 종류) 또는 None. PNG 머리(IHDR)만 읽는다
    try:
        with open(p_, 'rb') as f_: b_ = f_.read(33)
        if len(b_) < 33 or b_[:8] != b'\x89PNG\r\n\x1a\n' or b_[12:16] != b'IHDR': return None
        return b_[24], int.from_bytes(b_[16:20], 'big'), int.from_bytes(b_[20:24], 'big'), b_[25]
    except OSError: return None


def _scan(r_):
    out_ = []; base_ = r_.rstrip('/\\'); n0_ = base_.count(os.sep)
    for d_, ds_, fs_ in os.walk(r_):
        ds_[:] = sorted(x_ for x_ in ds_ if not x_.startswith('.') and x_ != 'images')
        if d_.count(os.sep) - n0_ >= MAX_DEPTH: ds_[:] = []
        im_ = os.path.join(d_, 'images')
        if not os.path.isdir(im_) or os.path.samefile(d_, r_): continue
        try: names_ = sorted(e_.name for e_ in os.scandir(im_) if NUM_PNG.match(e_.name))
        except OSError: continue
        if not names_: continue
        h_ = _png_head(os.path.join(im_, names_[0]))
        rel_ = os.path.relpath(d_, r_).replace(os.sep, '/')
        it_ = dict(name=rel_, frames=len(names_), bits=h_[0] if h_ else None, size='%dx%d' % (h_[1], h_[2]) if h_ else None)
        it_['usable'] = bool(h_ and h_[0] in (8, 16) and (h_[1], h_[2]) == (640, 480))
        if not it_['usable']: it_['note'] = '쓸 수 없음: 640x480 의 8/16 비트 PNG 가 아님'
        elif h_[0] == 8: it_['note'] = NOTE_8BIT
        out_.append(it_)
    return sorted(out_, key=lambda x_: x_['name'])


def _scan_measure(m_):   # 측정 버튼 녹화: <날짜>/MEASURE_*/frames.raw.gz + frames.json (머리만 읽는다, 큰 파일은 열지 않는다)
    import json
    out_ = []
    try: days_ = sorted(e_.name for e_ in os.scandir(m_) if e_.is_dir() and not e_.name.startswith('.'))
    except OSError: return out_
    for day_ in days_:
        try: recs_ = sorted(e_.name for e_ in os.scandir(os.path.join(m_, day_)) if e_.is_dir() and e_.name.startswith('MEASURE_'))
        except OSError: continue
        for rec_ in recs_:
            d_ = os.path.join(m_, day_, rec_); fp_ = os.path.join(d_, 'frames.raw.gz')
            if not os.path.isfile(fp_): continue
            try:
                with open(os.path.join(d_, 'frames.json'), 'rb') as f_: j_ = json.loads(f_.read(4_000_000).decode('utf-8', 'replace').split('"times_s"')[0].rstrip(', \n') + '}')
            except (OSError, ValueError): j_ = {}
            w_, h_, n_ = j_.get('width'), j_.get('height'), j_.get('frames')
            it_ = dict(name='%s/%s/%s' % (MEAS, day_, rec_), frames=n_ if isinstance(n_, int) else None, bits=16, size='%sx%s' % (w_, h_) if w_ and h_ else None)
            it_['usable'] = bool(isinstance(n_, int) and n_ > 0 and (w_, h_) == (640, 480) and 'uint16' in str(j_.get('dtype', '')))
            if not it_['usable']: it_['note'] = '쓸 수 없음: frames.json 이 없거나 640x480 16비트 원본이 아님'
            else:
                u_ = _used(n_); it_['note'] = '측정 버튼 녹화(16비트 원본). 정답 라벨이 없어 채점하지 않음' + ('. 메모리 한도로 앞 %d 장만 재생 (전체 %d 장)' % (u_, n_) if u_ < n_ else '')
            out_.append(it_)
    return out_


def listing(refresh_=False):
    r_ = root(); m_ = mroot()
    if r_ is None and m_ is None: return []
    key_ = (r_, m_)
    if refresh_ or _C['root'] != key_ or time.monotonic() - _C['t'] > LIST_TTL:
        items_ = []
        try: items_ += _scan(r_) if r_ else []
        except OSError: pass
        try: items_ += _scan_measure(m_) if m_ else []
        except OSError: pass
        _C.update(t=time.monotonic(), items=items_, root=key_)
    return _C['items']


def find(name_):   # 이름 -> (목록 항목, images 폴더의 절대 경로) 또는 (None, 이유)
    r_ = root()
    if r_ is None and mroot() is None: return None, '녹화 폴더가 없음 (--replay-root)'
    n_ = str(name_ or '')
    parts_ = n_.replace('\\', '/').split('/')
    if not n_ or '\x00' in n_ or n_.startswith(('/', '\\')) or ':' in n_ or any(p_ in ('', '.', '..') for p_ in parts_): return None, '이름이 올바르지 않음'
    it_ = next((x_ for x_ in listing() if x_['name'] == n_), None) or next((x_ for x_ in listing(True) if x_['name'] == n_), None)
    if it_ is None: return None, '목록에 없는 이름'
    if parts_[0] == MEAS:   # 측정 버튼 녹화: 파일 하나 (frames.raw.gz)
        m_ = mroot()
        if m_ is None or len(parts_) != 3: return None, '측정 폴더가 없음'
        mr_ = os.path.realpath(m_); f_ = os.path.realpath(os.path.join(m_, parts_[1], parts_[2], 'frames.raw.gz'))
        if os.path.commonpath([mr_, f_]) != mr_ or not os.path.isfile(f_): return None, '측정 폴더 밖을 가리킴'
        if not it_['usable']: return None, it_.get('note', '쓸 수 없는 녹화')
        return it_, f_
    if r_ is None: return None, '녹화 폴더가 없음 (--replay-root)'
    rr_ = os.path.realpath(r_); d_ = os.path.realpath(os.path.join(r_, *parts_, 'images'))
    if os.path.commonpath([rr_, d_]) != rr_ or not os.path.isdir(d_): return None, '녹화 폴더 밖을 가리킴'
    if not it_['usable']: return None, it_.get('note', '쓸 수 없는 녹화')
    return it_, d_


def blocking():   # 짐벌을 막아야 하는가 (재생 중 또는 바꾸는 중)
    L_ = APP.LIVE
    return L_ is not None and (getattr(L_, 'replay', None) is not None or getattr(L_, 'replay_busy', False))


def frame_blocked(t_in_):   # live_publish: 이 프레임은 짐벌에 사람 없이 보낸다
    L_ = APP.LIVE
    if getattr(L_, 'replay_busy', False) or getattr(L_, 'replay', None) is not None: return True
    s_ = getattr(L_, 'replay_src_t', None)
    return s_ is not None and t_in_ < s_


def cam_argv():   # 카메라 자리에 띄울 명령. 재생이 없으면 전과 똑같다
    r_ = getattr(APP.LIVE, 'replay', None)
    if r_ is None: return [APP.args.live_cam or str(HERE / 'src' / 'camera' / 'cam_pipe')]
    return [sys.executable, '-B', str(HERE / 'src' / 'camera' / 'replay_cam.py'), '--dir', r_['dir'], '--fps', '%g' % r_['fps'], '--loops', '0' if r_['loop'] else '1', '--tag']


def spawn():   # lv_spawn 의 재생 판 (LIVE.replay 가 있을 때만): 머리와 PNG 번호를 늘 붙인다. 머리는 camera_extra 의 읽기(reader)가 읽는다
    env_ = dict(os.environ); env_['LWIR_CAM_STAMP'] = '1'; env_['LWIR_CAM_SHM'] = '0'
    p_ = APP.lv_sp.Popen(cam_argv(), stdout=APP.lv_sp.PIPE, stderr=APP.LIVE.log, bufsize=0, env=env_)
    p_.lwir_stamp = True; p_.lwir_eval = replay_eval.EV   # 이 프로그램의 프레임은 이 재생의 채점에만 들어간다
    APP.LIVE.proc = p_; APP.LIVE.cam_lost = False; APP.LIVE.proc_t0 = time.monotonic()
    camera_extra.log_start()
    APP.lv_th.Thread(target=APP.lv_reader, args=(p_,), daemon=True).start()


def top_text(r_):   # 화면 그림 맨 윗줄 앞에 붙는 글 (cv2 는 한글을 못 그려서 ASCII 만)
    return 'REPLAY ' + r_['name'].encode('ascii', 'replace').decode('ascii')


def _msg(t_):
    APP.LIVE.replay_msg = dict(text=t_, at=time.strftime('%H:%M:%S')); camera_extra.say('REPLAY ' + t_.encode('ascii', 'backslashreplace').decode('ascii'))


def _gimbal_stop():
    if APP.GB is not None:
        APP.GB.mover = None; APP.GB.state['answer'] = ''; APP.gb_send('H')


def state():
    L_ = APP.LIVE; r_ = getattr(L_, 'replay', None) if L_ is not None else None
    st_ = dict(on=r_ is not None, busy=bool(getattr(L_, 'replay_busy', False)), name=None, fps=None, loop=None, bits=None, frames=None, shown=None, pos=None, note=None,
               msg=getattr(L_, 'replay_msg', None))
    if r_ is not None:
        n_ = max(0, L_.frames_in - r_['fin0']) if r_.get('fin0') is not None else 0
        st_.update(name=r_['name'], fps=r_['fps'], loop=r_['loop'], bits=r_['bits'], frames=r_['frames'], shown=n_, pos=(n_ - 1) % r_['used'] + 1 if n_ and r_['used'] else None,
                   note=NOTE_8BIT if r_['bits'] == 8 else None)
    return st_


def get_info(q_):
    r_ = on(); refresh_ = q_.get('refresh', ['0'])[0] == '1'
    why_ = None if r_ else ('녹화 폴더가 정해지지 않음 (--replay-root)' if not getattr(APP.args, 'replay_root', '') else '녹화 폴더가 없음')
    if r_ and APP.args.live_wait: why_ = '이 실행 방식에서는 쓸 수 없음 (--live-wait 0 필요)'
    return dict(available=why_ is None, why=why_, fps_min=FPS_MIN, fps_max=FPS_MAX, fps_default=FPS_DEF, list=listing(refresh_) if r_ else [], state=state(), eval=replay_eval.state())


def _used(frames_):   # replay_cam 이 메모리에 올리는 장 수 (--max-mb, 기본 400 MB)
    try: mb_ = float(os.environ.get('LWIR_REPLAY_MAX_MB', '400') or 400)
    except ValueError: mb_ = 400.0
    return min(frames_, max(1, int(mb_ * 1024 * 1024 // FRAME_BYTES)))


def start_request(q_):   # -> (http 코드, 글)
    try: fps_ = float(q_.get('fps', [str(FPS_DEF)])[0])
    except ValueError: return 400, 'fps 가 숫자가 아님'
    if not FPS_MIN <= fps_ <= FPS_MAX: return 400, 'fps 는 %g ~ %g' % (FPS_MIN, FPS_MAX)
    lp_ = q_.get('loop', ['1'])[0]
    if lp_ not in ('0', '1'): return 400, 'loop 는 0 또는 1'
    it_, d_ = find(q_.get('seq', [''])[0])
    if it_ is None: return (409 if not on() else 400), d_
    L_ = APP.LIVE
    with L_.sys_lock, L_.cam_lock:
        if L_.stopped or APP.args.live_wait: return 409, '이 실행 방식에서는 쓸 수 없음 (--live-wait 0 필요)'
        if L_.cam_hold: return 409, '카메라 초기화나 다른 재생 바꾸기가 진행 중. 잠시 뒤 다시 누르세요'
        L_.cam_hold = True; L_.replay_busy = True   # 이 순간부터 짐벌에 사람을 보내지 않는다
    _gimbal_stop()
    tgt_ = dict(name=it_['name'], dir=d_, fps=fps_, loop=lp_ == '1', bits=it_['bits'], frames=it_['frames'], used=_used(it_['frames']), fin0=None)
    APP.lv_th.Thread(target=_switch, args=(tgt_,), daemon=True).start()
    return 200, '재생 시작: %s' % it_['name']


def stop_request():
    L_ = APP.LIVE
    with L_.sys_lock, L_.cam_lock:
        if getattr(L_, 'replay', None) is None: return 409, '재생 중이 아님'
        if L_.cam_hold: return 409, '재생 바꾸기가 진행 중. 잠시 뒤 다시 누르세요'
        L_.cam_hold = True; L_.replay_busy = True
    _gimbal_stop()
    APP.lv_th.Thread(target=_switch, args=(None,), daemon=True).start()
    return 200, '재생 멈춤: 카메라로 돌아가는 중 (영상은 10~60초 뒤)'


def _end_old(p_, replay_):   # 지금 카메라 자리의 프로그램을 끝낸다. True = 끝남
    if p_.poll() is not None: return True
    p_.terminate()
    try: p_.wait(10 if replay_ else 45); return True
    except Exception: pass
    if not replay_: return False   # cam_pipe 는 카메라를 연 채 강제 종료하지 않는다 (camera_link 와 같은 규칙)
    p_.kill()
    try: p_.wait(5)
    except Exception: pass
    return True


def _switch(tgt_):   # 따로 도는 스레드. tgt_ = 새 녹화 또는 None (카메라로)
    L_ = APP.LIVE
    ev_ = replay_eval.prepare(tgt_) if tgt_ is not None else None   # 11-12: 정답 라벨을 미리 읽는다
    try:
        old_ = getattr(L_, 'replay', None)
        if not _end_old(L_.proc, old_ is not None):
            _msg('카메라 프로그램이 45초 안에 끝나지 않아 재생을 시작하지 않음. 잠시 뒤 다시 누르세요'); return
        time.sleep(1.5)   # 옛 프로그램의 읽기 스레드가 파이프 끝을 먼저 본다. 옛 영상과 새 영상 사이가 1초를 넘어 추적이 새로 시작한다
        with L_.cam_lock:
            if L_.stopped: return
            if hasattr(L_, 'cam_eof_t'): del L_.cam_eof_t
            L_.cam_fail_n = 0; L_.cam_unrec_n = 0; L_.cam_next_spawn = None
            L_.replay_src_t = time.perf_counter()   # 이보다 먼저 도착한 프레임 = 옛 영상: 짐벌에 사람 없이
            with L_.cond: L_.q.clear()
            if tgt_ is not None: tgt_['fin0'] = L_.frames_in
            L_.replay = tgt_
            replay_eval.activate(ev_, L_.replay_src_t, 'switched' if tgt_ is not None else 'stopped')   # 옛 재생의 채점은 파일로 쓰고 끝낸다
            _gimbal_stop()
            APP.lv_spawn()
        if tgt_ is not None:
            if APP.LIVE.meas is not None:
                with L_.meas_lock:
                    if L_.meas is not None: L_.meas.setdefault('replay', []).append(tgt_['name'])
            _msg('재생 중: %s (%g fps, %s)' % (tgt_['name'], tgt_['fps'], '반복' if tgt_['loop'] else '한 번'))
        else: _msg('재생 멈춤: 카메라 프로그램을 다시 시작함. 영상은 10~60초 뒤. 짐벌은 사람을 다시 누를 때까지 따라가지 않음')
    except Exception as e_: _msg('재생 바꾸기 오류: %s' % type(e_).__name__)
    finally:
        L_.replay_busy = False; L_.cam_hold = False


def on_cam_lost(now_):   # lv_cam_wait (cam_lock 안, 주 스레드): 재생 프로그램이 스스로 끝남 -> 카메라로 돌아간다. True = 여기서 처리함
    L_ = APP.LIVE; r_ = getattr(L_, 'replay', None)
    if r_ is None or not L_.cam_lost: return False
    rc_ = L_.proc.poll()
    if rc_ is None:
        try: rc_ = L_.proc.wait(2)
        except Exception:
            if now_ - L_.proc_t0 > 30: L_.proc.kill()
            return True
    L_.replay_src_t = time.perf_counter(); L_.replay = None
    replay_eval.activate(None, None, 'ended' if rc_ == 0 else 'replay program ended with code %s' % rc_)
    with L_.cond: L_.q.clear()
    L_.cam_fail_n = 0; L_.cam_unrec_n = 0; L_.cam_next_spawn = None
    _gimbal_stop()
    _msg(('재생 끝: %s. 카메라로 돌아감 (영상은 10~60초 뒤)' % r_['name']) if rc_ == 0 else ('재생 프로그램이 코드 %s 로 끝남 (%s). 카메라로 돌아감' % (rc_, r_['name'])))
    APP.lv_spawn()
    return True
