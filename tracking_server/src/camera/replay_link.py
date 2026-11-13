"""녹화 재생 (2026-11-12): 웹 화면에서 라벨 사이트의 녹화 폴더 하나를 골라 카메라 대신 서버에 넣고, 끝나면 진짜 카메라로 돌아간다.

- 녹화 목록: --replay-root 아래에서 images/ 폴더에 번호 PNG(000001.png ...)가 있는 폴더. 이름 = root 기준 상대 경로. 60초 동안 기억해 둔다
  (PNG 를 열지 않고 첫 파일의 머리 33 바이트만 읽어 8/16 비트와 크기를 본다).
- 재생: 카메라 프로그램 자리에 replay_cam.py 를 띄운다. 띄우는 길은 카메라와 같다 (camera_link.lv_spawn / camera_extra.spawn 이 cam_argv() 를 부른다).
  바꾸는 동안은 카메라 초기화 버튼과 같은 LIVE.cam_hold 로 lv_cam_wait 가 다른 프로그램을 띄우지 못하게 한다.
- 짐벌 안전: 재생을 시작할 때 짐벌 프로그램에 멈춤(H). 재생 중(바꾸는 중 포함)에는 짐벌에 가는 줄에 사람이 없고(blocked), /gimbal/click·move 는 거절한다.
  재생을 멈춘 뒤에도, 멈추기 전에 도착한 프레임(처리 중이던 몇 장)은 사람 없이 보낸다 (도착 시각 < src_t). 멈출 때도 H 를 보낸다.
  옛 영상의 마지막 프레임과 새 영상의 첫 프레임 사이는 언제나 1.5 초보다 길어서, 전부터 있던 "1초 넘는 입력 끊김" 규칙이 추적을 새로 시작하고 고른 사람을 놓는다.
- 측정 버튼 녹화도 목록에 나온다 (11-12): --measure-dir 아래 <날짜>/MEASURE_*/ 의 frames.raw.gz (16비트 원본). 이름 = '측정/<날짜>/<MEASURE 폴더>'.
  정답 라벨이 없으므로 채점하지 않는다. 길이 제한은 없다 (프레임은 그때그때 읽는다, 11-12).
- 영상 플레이어처럼 조종 (11-12): 재생 프로그램의 표준 입력에 줄을 쓴다 (SEEK <번호>, PAUSE, PLAY, SPEED <fps>; 쓰는 곳 = Ctl). 같은 종류는 모아서 마지막 것만 보낸다.
  재생 프로그램이 이미 끝났거나(끊어진 파이프) 줄이 몰려도 서버는 멈추지 않는다. 위치(pos)는 프레임 머리에서 읽은 값이다 (replay_eval.Eval.position).
  건너뛰면 재생 프로그램이 1.2 초 쉬고 새 프레임을 보내서, 서버의 "입력이 1 초 넘게 끊기면 추적을 새로 시작" 규칙이 추적을 비운다. 멈춘 동안은 같은 프레임을
  초당 5 장 "되풀이" 표시를 달아 보낸다 (서버가 카메라가 죽었다고 다시 띄우지 않게. 채점하지 않는다). 정답 라벨이 있는 녹화는 되풀이 장을
  처리 줄에 넣지 않으므로 그림을 새로 만들지 않고, 이미 나간 그 장면의 그림을 그대로 둔다 (11-13: camera_link.replay_holds_picture 가 "영상 없음" 그림으로 덮지 않게 막는다). 짐벌 안전은 그대로다 (멈춘 중·건너뛰는 중에도 사람 없음).
- 재생을 한 번도 쓰지 않으면 이 파일은 아무것도 바꾸지 않는다 (cam_argv() 는 전과 같은 명령을 돌려준다, LIVE.replay 는 None).
- 정답과 비교 (11-12, replay_eval.py): 재생 카메라는 늘 프레임 머리(LWIR_CAM_STAMP=1)와 PNG 번호(--tag)를 붙여 띄운다 (spawn).
  카메라 쪽 설정(LWIR_CAM_STAMP)과 상관없이 재생 프로그램에만 그렇게 하고, 진짜 카메라를 띄우는 길은 바꾸지 않는다.
- 저장된 결과 (11-13, src/track/replay_backends.py): --replay-tracks-root 를 주면 <그 폴더>/<녹화 이름>/<key>.mot.txt 와 <key>.json 을 찾아
  그 녹화의 재생 추적기 'pre:<key>' 로 내놓는다. 고르면 서버 추적기의 상자 대신 파일의 상자와 번호를 그리고 채점한다. 옵션을 주지 않으면 아무것도 바뀌지 않는다.
"""
import os
import re
import sys
import threading
import time

from ..common.state import APP
from ..common.shared_values import HERE
from ..track.bytetrack import available_bytetrack_library
from ..track import replay_backends
from ..web.preview_worker import set_preview_producer
from . import camera_extra
from . import replay_eval

FPS_MIN, FPS_MAX, FPS_DEF = 5.0, 30.0, 25.0
LIST_TTL = 60.0
MAX_DEPTH = 4                     # root 아래 이 깊이까지 찾는다 (opi_sequences/seq_018 = 2)
NUM_PNG = re.compile(r'^\d+\.png$', re.I)
SEEK_GAP_S = 1.2                  # replay_cam.py 의 --gap 기본값과 같다 (참고용: 건너뛴 뒤 첫 프레임까지)
NOTE_8BIT = '8비트 녹화: 온도 Filter 가 맞지 않아 사람이 걸러질 수 있음. 보려면 온도 Filter 를 잠시 끄세요'
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
                it_['note'] = '측정 버튼 녹화(16비트 원본). 정답 라벨이 없어 채점하지 않음. 건너뛰기는 되는데, 먼 곳이나 뒤로 건너뛰면 파일을 앞에서부터 다시 풀어 몇 초 걸릴 수 있음'
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


NO_STORED = '이 녹화에는 저장된 결과가 없습니다'
_T = dict(t=-1e9, key=None, items=None, found={}, sigs={})
_T_LOCK = threading.Lock()   # 다시 읽는 동안 들어온 다른 요청이 같은 읽기를 겹쳐 시작하지 않게 한다


def tracks_root():   # 저장된 결과 폴더 (--replay-tracks-root). 주지 않았거나 폴더가 없으면 None = 이 기능은 아무것도 하지 않는다
    t_ = getattr(APP.args, 'replay_tracks_root', '') or ''
    return t_ if t_ and os.path.isdir(t_) else None


def _stored_sig(d_, im_, frames_):   # 한 녹화의 저장된 결과가 그대로인지 보는 값: 결과·설명 파일의 (이름, 크기, 바뀐 시각) + 녹화의 PNG 수 + images 폴더의 바뀐 시각. 읽지 못하면 None (= 다시 읽는다)
    try:
        with os.scandir(d_) as es_:
            fs_ = sorted((e_.name, e_.stat().st_size, e_.stat().st_mtime_ns) for e_ in es_ if e_.name.endswith((replay_backends.MOT_EXT, replay_backends.META_EXT)))
        return tuple(fs_), frames_, os.stat(im_).st_mtime_ns
    except OSError: return None


def stored_tracks(items_, refresh_=False):
    """{녹화 이름: (쓸 수 있는 것 [dict(id, key, label)], 못 쓰는 것 [dict(key, why)])}. 녹화 목록처럼 60초마다 다시 본다.

    다시 볼 때 파일을 처음부터 읽는 것(sha256, 모든 줄 풀기)은 _stored_sig 가 바뀐 녹화만 한다. 화면이 열려 있으면 카메라 추적 중에도
    60초마다 여기로 오기 때문이다. 새로 고침(refresh=1)은 모두 다시 읽는다. 크기와 바뀐 시각이 같은 채 내용만 바뀐 파일은 새로 고침 전까지
    목록에서 알아채지 못하지만, 재생을 시작할 때 _stored_target 과 replay_backends.make 가 언제나 처음부터 다시 읽고 확인한다."""
    t_ = tracks_root(); r_ = root()
    if t_ is None or r_ is None: return {}
    key_ = (t_, r_)
    with _T_LOCK:
        if refresh_ or _T['key'] != key_ or _T['items'] is not items_ or time.monotonic() - _T['t'] > LIST_TTL:
            found_ = {}; sigs_ = {}; old_ = _T['sigs'] if _T['key'] == key_ and not refresh_ else {}
            for it_ in items_:
                parts_ = it_['name'].split('/')
                if not it_.get('usable') or parts_[0] == MEAS: continue
                d_ = replay_backends.stored_dir(t_, it_['name'])
                if d_ is None or not os.path.isdir(d_): continue
                im_ = os.path.join(r_, *parts_, 'images')
                sig_ = _stored_sig(d_, im_, it_.get('frames')); prev_ = old_.get(it_['name'])
                res_ = prev_[1] if sig_ is not None and prev_ is not None and prev_[0] == sig_ else replay_backends.scan_stored(d_, im_, it_['name'])
                if sig_ is not None: sigs_[it_['name']] = (sig_, res_)
                if res_[0] or res_[1]: found_[it_['name']] = res_
            _T.update(t=time.monotonic(), key=key_, items=items_, found=found_, sigs=sigs_)
        return _T['found']


def _stored_target(it_, d_, key_):   # 재생을 시작할 때: 고른 녹화의 저장된 결과를 지금 다시 읽어 확인한다 -> (dict(key, mot, meta, label), None) 또는 (None, 이유)
    t_ = tracks_root()
    paths_ = replay_backends.stored_paths(t_, it_['name'], key_) if t_ is not None and it_['name'].split('/')[0] != MEAS else None
    if paths_ is None: return None, NO_STORED
    try: item_ = replay_backends.PrecomputedTracks(key_, paths_[0], paths_[1], replay_backends.png_numbers(d_), it_['name'])
    except replay_backends.Refused as e_: return None, '저장된 결과를 쓸 수 없습니다: %s' % e_
    except Exception as e_: return None, '저장된 결과를 쓸 수 없습니다 (%s)' % type(e_).__name__   # OSError(전부터) 와 미처 생각하지 못한 깨진 파일: 409 로 답한다
    return dict(key=key_, mot=paths_[0], meta=paths_[1], label=item_.label), None


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
    argv = [sys.executable, '-B', str(HERE / 'src' / 'camera' / 'replay_cam.py'), '--dir', r_['dir'], '--fps', '%g' % r_['fps'], '--loops', '0' if r_['loop'] else '1', '--tag', '--control', '--total', str(r_['frames'] or 0)]
    if replay_eval.lossless():
        argv.append('--no-skip')
    if 'pacing' in r_:
        argv.extend(['--pacing', r_['pacing']])
    return argv


class Ctl:
    """재생 프로그램의 표준 입력에 조종 줄을 쓴다. 쓰는 일은 이 스레드 하나만 한다 (웹 요청은 값만 맡기고 바로 돌아온다).
    같은 종류(SEEK / SPEED / PAUSE·PLAY)는 마지막 것만 남는다. 프로그램이 끝났거나 파이프가 끊기면 조용히 끝난다."""
    def __init__(self, p_):
        self.p, self.cv, self.pend = p_, threading.Condition(), {}
        self.dead = False; self.sent = 0; self.coalesced = 0; self.err = ''
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, kind_, val_=None):   # kind_: 'seek' | 'speed' | 'paused' | 'step'
        with self.cv:
            if self.dead: return False
            if kind_ == 'step':             # 한 장 넘기기는 모으지 않고 횟수를 더한다
                self.pend['step'] = self.pend.get('step', 0) + 1
            else:
                if kind_ in self.pend: self.coalesced += 1
                self.pend[kind_] = val_
            self.cv.notify()
            return True

    def _run(self):
        try:
            while True:
                with self.cv:
                    while not self.pend:
                        if self.p.poll() is not None: return
                        self.cv.wait(1.0)
                    take_ = dict(self.pend); self.pend.clear()
                lines_ = []
                if 'speed' in take_: lines_.append('SPEED %g' % take_['speed'])
                if 'seek' in take_:
                    pos, pause = take_['seek']
                    lines_.append('SEEK %d%s' % (pos, ' PAUSE' if pause else ''))
                if 'paused' in take_: lines_.append('PAUSE' if take_['paused'] else 'PLAY')
                if 'step' in take_: lines_ += ['STEP'] * take_['step']
                try: self.p.stdin.write(('\n'.join(lines_) + '\n').encode('ascii')); self.p.stdin.flush(); self.sent += len(lines_)
                except (BrokenPipeError, OSError, ValueError) as e_: self.err = type(e_).__name__; return
        finally:
            with self.cv: self.dead = True
            try: self.p.stdin.close()
            except Exception: pass


def spawn():   # lv_spawn 의 재생 판 (LIVE.replay 가 있을 때만): 머리와 PNG 번호를 늘 붙인다. 머리는 camera_extra 의 읽기(reader)가 읽는다
    env_ = dict(os.environ); env_['LWIR_CAM_STAMP'] = '1'; env_['LWIR_CAM_SHM'] = '0'
    p_ = APP.lv_sp.Popen(cam_argv(), stdin=APP.lv_sp.PIPE, stdout=APP.lv_sp.PIPE, stderr=APP.LIVE.log, bufsize=0, env=env_)
    p_.lwir_stamp = True; p_.lwir_eval = replay_eval.EV   # 이 프로그램의 프레임은 이 재생의 채점에만 들어간다
    p_.lwir_ctl = Ctl(p_)
    APP.LIVE.replay['paused'] = False   # 새로 띄운 프로그램은 처음부터 재생한다 (서버가 같은 녹화를 다시 띄운 경우도)
    camera_extra.install_producer(p_)
    camera_extra.log_start()
    APP.lv_th.Thread(target=APP.lv_reader, args=(p_,), daemon=True).start()


def top_text(r_):   # 화면 그림 맨 윗줄 앞에 붙는 글 (cv2 는 한글을 못 그려서 ASCII 만)
    t_ = 'REPLAY ' + r_['name'].encode('ascii', 'replace').decode('ascii')
    return t_ + ' | STORED ' + r_['tracks']['key'] if r_.get('tracks') else t_   # 저장된 결과를 그리는 중임을 그림에도 적는다


def _msg(t_):
    APP.LIVE.replay_msg = dict(text=t_, at=time.strftime('%H:%M:%S')); camera_extra.say('REPLAY ' + t_.encode('ascii', 'backslashreplace').decode('ascii'))


def _gimbal_stop():
    if APP.GB is not None:
        APP.GB.mover = None; APP.GB.state['answer'] = ''; APP.gb_send('H')


def async_preview(replay, worker):
    return bool(worker is not None and worker.async_replay and replay.get('preview', 'async') == 'async')


def state():
    L_ = APP.LIVE; r_ = getattr(L_, 'replay', None) if L_ is not None else None
    st_ = dict(on=r_ is not None, busy=bool(getattr(L_, 'replay_busy', False)), name=None, fps=None, loop=None, bits=None, frames=None, total=None, shown=None, pos=None, paused=False,
               seeking=False, seek_to=None, note=None, msg=getattr(L_, 'replay_msg', None)); st_['pass'] = None
    if r_ is not None:
        n_ = max(0, L_.frames_in - r_['fin0']) if r_.get('fin0') is not None else 0
        ev_ = replay_eval.EV
        worker = getattr(L_, 'preview_worker', None)
        if ev_ is not None and async_preview(r_, worker):
            ps_ = worker.replay_state(ev_.run_id)
        else:
            ps_ = ev_.position() if ev_ is not None else {}
        st_.update(name=r_['name'], fps=r_['fps'], loop=r_['loop'], bits=r_['bits'], frames=r_['frames'], total=r_['frames'], shown=n_, note=NOTE_8BIT if r_['bits'] == 8 else None,
                   pos=ps_.get('pos'), paused=bool(r_.get('paused')), seeking=ps_.get('seek_to') is not None, seek_to=ps_.get('seek_to'))   # pos = 0-based index of the frame on screen, from the frame header
        st_['pass'] = ps_.get('pass_')
        st_['backend'] = r_.get('backend', 'python')
        if r_.get('tracks'): st_['backend_label'] = r_['tracks']['label']
        st_['preview'] = 'async' if async_preview(r_, worker) else 'sync'
        st_['pacing'] = r_.get('pacing', 'legacy')
    return st_


def get_info(q_):
    r_ = on(); refresh_ = q_.get('refresh', ['0'])[0] == '1'
    why_ = None if r_ else ('녹화 폴더가 정해지지 않음 (--replay-root)' if not getattr(APP.args, 'replay_root', '') else '녹화 폴더가 없음')
    if r_ and APP.args.live_wait: why_ = '이 실행 방식에서는 쓸 수 없음 (--live-wait 0 필요)'
    backends = [dict(id='python', label='현재 추적기')]
    if available_bytetrack_library(APP.args) is not None:
        backends.append(dict(id='cpp_bytetrack', label='C++ ByteTrack'))
    items_ = listing(refresh_) if r_ else []
    stored_ = stored_tracks(items_, refresh_) if r_ else {}
    if stored_:   # 저장된 결과가 있는 녹화에만 tracks 를 붙인다 (화면은 고른 녹화의 것만 보여 준다). backends 에는 이름을 찾을 수 있게 모두 넣고 stored 표시를 한다
        labels_ = {}
        for name_ in sorted(stored_):
            for b_ in stored_[name_][0]: labels_.setdefault(b_['id'], b_['label'])
        backends += [dict(id=id_, label=label_, stored=True) for id_, label_ in labels_.items()]
        shown_ = []
        for it_ in items_:
            ok_, bad_ = stored_.get(it_['name'], ((), ()))
            if ok_ or bad_:
                it_ = dict(it_, tracks=[dict(id=b_['id'], label=b_['label']) for b_ in ok_])
                if bad_: it_['tracks_note'] = '저장된 결과 %s 못 씀: %s' % (bad_[0]['key'], bad_[0]['why']) + (' (외 %d개)' % (len(bad_) - 1) if len(bad_) > 1 else '')
            shown_.append(it_)
        items_ = shown_
    return dict(backends=backends, result_views=True, available=why_ is None, why=why_, fps_min=FPS_MIN, fps_max=FPS_MAX, fps_default=FPS_DEF, list=items_, state=state(), eval=replay_eval.state(), eval_last=replay_eval.last())


def start_request(q_):   # -> (http 코드, 글)
    try: fps_ = float(q_.get('fps', [str(FPS_DEF)])[0])
    except ValueError: return 400, 'fps 가 숫자가 아님'
    if not FPS_MIN <= fps_ <= FPS_MAX: return 400, 'fps 는 %g ~ %g' % (FPS_MIN, FPS_MAX)
    lp_ = q_.get('loop', ['0'])[0]
    if lp_ not in ('0', '1'): return 400, 'loop 는 0 또는 1'
    pacing_values = q_.get('pacing', [os.environ.get('LWIR_REPLAY_PACING', 'legacy')])
    if len(pacing_values) != 1 or pacing_values[0] not in ('legacy', 'deadline'):
        return 400, 'pacing 은 legacy 또는 deadline'
    pacing = pacing_values[0]
    backend = q_.get('backend', ['python'])[0]
    library = available_bytetrack_library(APP.args) if backend == 'cpp_bytetrack' else None
    stored_key = replay_backends.stored_key(backend) if tracks_root() is not None else None   # 'pre:<key>' (저장된 결과). 옵션이 없으면 전처럼 모르는 이름이다
    if backend not in ('python', 'cpp_bytetrack') and stored_key is None:
        return 400, '알 수 없는 재생 추적기'
    if backend == 'cpp_bytetrack' and library is None:
        return 409, 'C++ ByteTrack 라이브러리가 설정되지 않았습니다'
    it_, d_ = find(q_.get('seq', [''])[0])
    if it_ is None: return (409 if not on() else 400), d_
    tracks_ = None
    if stored_key is not None:
        tracks_, why_ = _stored_target(it_, d_, stored_key)
        if tracks_ is None: return 409, why_
    L_ = APP.LIVE
    worker = getattr(L_, 'preview_worker', None)
    can_async = worker is not None and worker.async_replay
    preview_modes = q_.get('preview', ['async' if can_async else 'sync'])
    if len(preview_modes) != 1 or preview_modes[0] not in ('sync', 'async'):
        return 400, 'preview 는 sync 또는 async'
    preview_mode = preview_modes[0]
    if preview_mode == 'async' and not can_async:
        return 409, '이 서버에서 비동기 재생 미리보기를 사용할 수 없습니다'
    with L_.sys_lock, L_.cam_lock:
        if L_.stopped or APP.args.live_wait: return 409, '이 실행 방식에서는 쓸 수 없음 (--live-wait 0 필요)'
        if L_.cam_hold: return 409, '카메라 초기화나 다른 재생 바꾸기가 진행 중. 잠시 뒤 다시 누르세요'
        L_.cam_hold = True; L_.replay_busy = True   # 이 순간부터 짐벌에 사람을 보내지 않는다
        set_preview_producer(L_, None, before=time.perf_counter())
    _gimbal_stop()
    tgt_ = dict(name=it_['name'], dir=d_, fps=fps_, loop=lp_ == '1', bits=it_['bits'], frames=it_['frames'], fin0=None, paused=False,
                backend=backend, tracker_library=library, preview=preview_mode, pacing=pacing)
    if tracks_ is not None: tgt_['tracks'] = tracks_
    APP.lv_th.Thread(target=_switch, args=(tgt_,), daemon=True).start()
    return 200, '재생 시작: %s' % it_['name']


def _ctl_target():   # -> (재생 dict, Ctl, 오류). 조종할 수 있을 때만 앞 둘이 있고, 아니면 오류 = (http 코드, 글)
    L_ = APP.LIVE; r_ = getattr(L_, 'replay', None)
    if r_ is None: return None, None, (409, '재생 중이 아님')
    if getattr(L_, 'replay_busy', False) or L_.cam_hold: return None, None, (409, '재생 시작·바꾸기가 진행 중. 잠시 뒤 다시 누르세요')
    ctl_ = getattr(L_.proc, 'lwir_ctl', None)
    if ctl_ is None or ctl_.dead or L_.proc.poll() is not None or L_.cam_lost: return None, None, (409, '재생 프로그램이 끝났거나 아직 시작하지 않음')
    return r_, ctl_, None


def seek_request(q_):   # POST /replay/seek?pos=<0 부터 센 프레임 번호>
    try: pos_ = int(q_.get('pos', [''])[0])
    except ValueError: return 400, 'pos 가 정수가 아님'
    r_, ctl_, e_ = _ctl_target()
    if r_ is None: return e_
    if not 0 <= pos_ < (r_['frames'] or 0): return 400, 'pos 는 0 ~ %d (0 부터 센 프레임 번호)' % ((r_['frames'] or 1) - 1)
    pause_ = q_.get('pause', ['0'])[0] == '1'
    ev_ = replay_eval.EV
    if ev_ is not None:
        before = time.perf_counter()
        previous_pass = ev_.want_seek(pos_, before)
        worker = getattr(APP.LIVE, 'preview_worker', None)
        if async_preview(r_, worker):
            worker.begin_replay_seek(ev_.run_id, pos_, before, previous_pass, replay=r_)
    if pause_:
        r_['paused'] = True
    ctl_.send('seek', (pos_, pause_))
    return 200, '%d 번 프레임으로 건너뜀 (%.1f 초 뒤 새 영상, 추적은 새로 시작)' % (pos_, SEEK_GAP_S)


def pause_request():
    r_, ctl_, e_ = _ctl_target()
    if r_ is None: return e_
    r_['paused'] = True; ctl_.send('paused', True)
    return 200, '멈춤 (같은 장면을 초당 5 장 되풀이, 채점하지 않음)'


def play_request():
    r_, ctl_, e_ = _ctl_target()
    if r_ is None: return e_
    r_['paused'] = False; ctl_.send('paused', False)
    return 200, '이어서 재생'


def speed_request(q_):   # POST /replay/speed?fps=<5~30>
    try: fps_ = float(q_.get('fps', [''])[0])
    except ValueError: return 400, 'fps 가 숫자가 아님'
    if not FPS_MIN <= fps_ <= FPS_MAX: return 400, 'fps 는 %g ~ %g' % (FPS_MIN, FPS_MAX)
    r_, ctl_, e_ = _ctl_target()
    if r_ is None: return e_
    r_['fps'] = fps_; ctl_.send('speed', fps_)
    return 200, '초당 %g 장' % fps_


def restart_request():
    target = getattr(APP.LIVE, 'replay', None) or replay_eval.last()
    if target is None:
        return 409, '다시 계산할 녹화가 없습니다'
    query = {'seq': [target['name']], 'fps': [str(target['fps'])], 'loop': ['0'],
             'backend': [target.get('backend', 'python')]}
    if 'preview' in target:
        query['preview'] = [target['preview']]
    if 'pacing' in target:
        query['pacing'] = [target['pacing']]
    return start_request(query)


def stop_request():
    L_ = APP.LIVE
    with L_.sys_lock, L_.cam_lock:
        if getattr(L_, 'replay', None) is None: return 409, '재생 중이 아님'
        if L_.cam_hold: return 409, '재생 바꾸기가 진행 중. 잠시 뒤 다시 누르세요'
        L_.cam_hold = True; L_.replay_busy = True
        set_preview_producer(L_, None, before=time.perf_counter())
    _gimbal_stop()
    APP.lv_th.Thread(target=_switch, args=(None,), daemon=True).start()
    return 200, '재생 멈춤: 카메라로 돌아가는 중 (영상은 10~60초 뒤)'

def step_request():   # POST /replay/step: 멈춘 상태에서 다음 한 장만 보내고 다시 멈춤 (추적이 이어짐)
    r_, ctl_, e_ = _ctl_target()
    if r_ is None: return e_
    if not r_.get('paused'): return 409, '멈춘 상태에서만 한 장씩 넘길 수 있습니다'
    ctl_.send('step')
    return 200, '다음 한 장'


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
    old_ = getattr(L_, 'replay', None)
    old_proc_ = L_.proc
    ev_ = None
    installed = False
    try:
        if tgt_ is not None:
            ev_ = replay_eval.prepare(tgt_)
            if ev_ is None and tgt_.get('backend') == 'cpp_bytetrack':
                _msg('C++ ByteTrack을 열지 못해 재생을 시작하지 않았습니다')
                return
            if ev_ is None and tgt_.get('tracks'):   # 저장된 결과를 다시 읽다가 실패: 서버 추적기의 상자를 대신 보여 주지 않는다
                _msg('저장된 결과를 열지 못해 재생을 시작하지 않았습니다')
                return
        if not _end_old(old_proc_, old_ is not None):
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
            replay_eval.activate(ev_, L_.replay_src_t, 'switched' if tgt_ is not None else 'stopped')
            installed = True   # 옛 재생의 채점은 파일로 쓰고 끝낸다
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
        if ev_ is not None and not installed:
            ev_.close('not started')
        if (not installed and old_ is not None and not L_.stopped
                and L_.proc is old_proc_ and getattr(L_, 'replay', None) is old_):
            set_preview_producer(L_, old_proc_, before=time.perf_counter())
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
    L_.replay_src_t = time.perf_counter()
    set_preview_producer(L_, None, before=L_.replay_src_t)
    L_.replay = None
    replay_eval.activate(None, None, 'ended' if rc_ == 0 else 'replay program ended with code %s' % rc_)
    with L_.cond: L_.q.clear()
    L_.cam_fail_n = 0; L_.cam_unrec_n = 0; L_.cam_next_spawn = None
    _gimbal_stop()
    _msg(('재생 끝: %s. 카메라로 돌아감 (영상은 10~60초 뒤)' % r_['name']) if rc_ == 0 else ('재생 프로그램이 코드 %s 로 끝남 (%s). 카메라로 돌아감' % (rc_, r_['name'])))
    APP.lv_spawn()
    return True
