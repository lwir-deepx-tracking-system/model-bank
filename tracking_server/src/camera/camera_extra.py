"""카메라 길에 덧붙인 세 가지 (2026-11-11). camera_link.py 가 부른다.

A. 지워지지 않는 카메라 기록 (기본으로 켜짐)
   카메라 프로그램(cam_pipe)이 내는 줄(1초마다 나오는 속도 줄 CAMFPS 는 빼고)과 서버의 "LIVE camera" / "CAMRESET" 줄을
   tracking_server/camera_events.log 에 이어 쓴다. 줄 앞에 날짜·시각과 부팅 뒤 지난 초(up=)가 붙는다.
   서버를 다시 시작해도 지워지지 않는다. 4 MB 가 넘으면 .1 .2 .3 으로 밀고 새로 쓴다 (가장 오래된 것은 버림).
   파일에 쓰는 일은 이 파일의 스레드 하나만 한다. 프레임을 읽는 쪽은 메모리의 줄 목록에 넣기만 하므로 디스크가 느리거나
   쓸 수 없어도 영상에는 영향이 없다 (쓰기 오류는 세기만 하고 버린다). cam_pipe 의 줄은 서버가 전부터 쓰던
   <결과 폴더>/cam_pipe.log 를 3 초마다 따라 읽어 옮긴다 (그 파일이 자라지 않았으면 읽지 않는다). cam_pipe 를 띄우는 방법은 바꾸지 않았다.
   파일 하나는 4 MB 를 넘지 않는다 (줄이 한꺼번에 밀려도 나눠 쓴다). 한 줄은 2000 글자까지만 남긴다.
   끄기: LWIR_CAM_LOG=0.  다른 자리에 쓰기: LWIR_CAM_LOG=<파일 경로>.

B. 프레임 시간표 (기본으로 꺼짐, LWIR_CAM_STAMP=1 일 때만)
   cam_pipe 가 프레임마다 48 바이트 머리(표시 글자 LWF1, 프레임 번호, 카메라에서 프레임을 받은 순간의 시각, 앞 프레임과의 간격)를
   붙인다. 서버는 머리를 확인하고, 어긋나면 다음 머리를 찾아 맞추고, 못 찾으면 cam_pipe 를 다시 시작한다.
   머리가 한 번도 안 맞으면(옛 cam_pipe) 시간표를 끄고 전 방식으로 다시 시작한다.
   그림 크기가 서버와 다르면 cam_pipe 를 멈추고 20 초, 60 초, 5 분, 15 분으로 늘려 가며 다시 띄운다. 읽는 쪽에서 예상하지 못한 오류가 나면
   cam_pipe 를 끝내고 파이프를 끝까지 읽은 뒤 다시 띄우고, 세 번 나면 시간표를 끄고 전 방식으로 다시 띄운다.
   숫자는 APP.LIVE.cam_stamp (dict) 와 APP.LIVE.cam_stamp_recent (최근 프레임의 (받은 시각 ta, 번호, 카메라 시각 초, 간격 us)) 에 있다.
   60 초마다 한 줄 요약을 카메라 기록에 쓴다.

C. 공유 메모리로 프레임 넘기기 (기본으로 꺼짐, LWIR_CAM_STAMP=1 이고 LWIR_CAM_SHM=1 일 때만)
   cam_pipe 가 그림을 /dev/shm/lwir_cam_ring_<uid> 의 칸 4 개짜리 고리에 쓰고, 파이프로는 48 바이트 머리(칸 번호 포함)만 보낸다.
   서버는 고리를 읽기 전용으로 붙여 그 칸을 복사한다. 복사 앞뒤로 칸의 프레임 번호를 확인해서, 읽는 사이에 덮어쓰인 프레임은 버리고 센다.
   서버가 고리를 붙이지 못하면 고리를 끄고 cam_pipe 를 다시 시작한다 (그림은 파이프로). cam_pipe 가 고리를 못 만들면 스스로 파이프로 보낸다.
   서버가 시작할 때, 강제 종료된 cam_pipe 가 남긴 고리 파일이 있으면 지운다 (그 cam_pipe 가 더는 없을 때만).

세 가지 모두 환경 변수가 없으면 B, C 의 코드는 실행되지 않는다. 머리와 고리의 모양은 cam_pipe.cpp (v2.6) 와 같아야 한다.
"""
import atexit
import collections
import math
import mmap
import os
import struct
import threading
import time

import numpy as np

from ..common.state import APP
from ..common.shared_values import HERE
from . import replay_eval   # 11-12: frames of the replay program carry their png number (flags bit 2)

# ---------------------------------------------------------------- A. 지워지지 않는 카메라 기록
LOG_NAME = 'camera_events.log'
LOG_MAX_BYTES = 4 * 1024 * 1024   # 이 크기를 넘으면 .1 로 밀고 새로 쓴다
LOG_KEEP = 3                      # 밀어 둔 옛 파일 수 (.1 .2 .3)
LINE_MAX = 2000                   # 한 줄에서 기록에 남기는 글자 수 (넘는 부분은 버린다)
POLL_S = 3.0                      # 기록 스레드가 깨는 간격 (2026-11-11 검토: 0.5 초일 때 가짜 카메라 시험에서 프레임을 0.2 % 잃었다). cam_pipe 줄의 시각은 이만큼 늦을 수 있다
SUMMARY_S = 60.0

_Q = collections.deque(maxlen=4000)   # (시각, 부팅 뒤 초, 누가, 글). 가득 차면 가장 오래된 줄이 빠진다
_S = dict(thread=None, path=None, tail=None, tail_name=None, tail_buf=b'', log_err=0, written=0, stamp_off='', shm_off='', t_sum=0.0, said=set(),
          wakes=0, idle_wakes=0, size_bad=0, reader_err=0, ring_removed='')
_SL = threading.Lock()                # 60 초 요약용 합계
_ACC = dict(n=0, n_age=0, age=0.0, age_max=0.0, gap=0.0, gap_max=0.0)


def note(who_, text_):   # 기록에 한 줄 (메모리에만 넣는다. 오류를 내지 않는다)
    try: _Q.append((time.time(), time.monotonic(), who_, str(text_)))
    except Exception: pass


def say(text_):   # 서버의 줄: 전과 똑같이 출력하고, 카메라 기록에도 넣는다
    print(text_, flush=True); note('server', text_)


def log_path():
    v_ = os.environ.get('LWIR_CAM_LOG', '')
    if v_ == '0': return None
    if v_: return v_ if os.path.isabs(v_) else str(HERE / v_)
    return str(HERE / LOG_NAME)


def _rotate(path_):   # 지금 파일을 .1 로 밀고 (.1 -> .2 -> .3, 가장 오래된 것은 버림) 새 파일로 시작한다
    for i_ in range(LOG_KEEP, 0, -1):
        src_ = path_ if i_ == 1 else '%s.%d' % (path_, i_ - 1)
        try:
            if os.path.exists(src_): os.replace(src_, '%s.%d' % (path_, i_))
        except OSError: pass


def _append(path_, parts_):
    if parts_:
        with open(path_, 'ab') as f_: f_.write(b''.join(parts_))
        _S['written'] += len(parts_)


def _flush():
    if not _Q: return
    rows_ = []
    while _Q:
        try: rows_.append(_Q.popleft())
        except IndexError: break
    path_ = _S['path']
    if not path_: return
    try:
        try: size_ = os.path.getsize(path_)
        except OSError: size_ = 0
        part_ = []; n_ = 0   # 한 번에 쓸 줄들. 파일 하나가 LOG_MAX_BYTES 를 넘지 않게 나눠 쓴다 (줄이 한꺼번에 많이 밀려도)
        for t_, up_, who_, text_ in rows_:
            b_ = ('%s.%03d up=%.1f [%s] %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_)), int(t_ * 1000) % 1000, up_, who_,
                                                   text_[:LINE_MAX].replace('\n', ' ').replace('\r', ' '))).encode('utf-8', 'replace')
            if size_ + n_ + len(b_) > LOG_MAX_BYTES and size_ + n_ > 0:
                _append(path_, part_); part_ = []; n_ = 0; _rotate(path_); size_ = 0
            part_.append(b_); n_ += len(b_)
        _append(path_, part_)
    except Exception: _S['log_err'] += 1   # 쓸 수 없음: 줄은 버린다


def _tail():   # cam_pipe 가 쓰는 <결과 폴더>/cam_pipe.log 에 새로 붙은 줄을 기록으로 옮긴다
    name_ = getattr(APP.LIVE.log, 'name', None)
    if not isinstance(name_, str): return
    if _S['tail_name'] != name_:
        if _S['tail'] is not None:
            try: _S['tail'].close()
            except Exception: pass
        _S['tail'] = open(name_, 'rb'); _S['tail_name'] = name_; _S['tail_buf'] = b''
    f_ = _S['tail']
    if os.fstat(f_.fileno()).st_size == f_.tell(): return False   # 파일이 자라지 않았다: 읽지 않는다
    for _ in range(8):   # 한 번에 최대 2 MB
        d_ = f_.read(262144)
        if not d_:
            if os.fstat(f_.fileno()).st_size < f_.tell(): f_.seek(0); _S['tail_buf'] = b''   # 파일이 다시 만들어짐
            break
        lines_ = (_S['tail_buf'] + d_).split(b'\n'); _S['tail_buf'] = lines_.pop()
        if len(_S['tail_buf']) > 4096: lines_.append(_S['tail_buf']); _S['tail_buf'] = b''
        for ln_ in lines_:
            t_ = ln_.decode('utf-8', 'replace').strip()
            if t_ and not t_.startswith('CAMFPS'): note('cam_pipe', t_)
    return True


def _summary():   # B: 60 초마다 한 줄
    now_ = time.monotonic()
    if now_ - _S['t_sum'] < SUMMARY_S: return
    _S['t_sum'] = now_
    cs_ = getattr(APP.LIVE, 'cam_stamp', None)
    if cs_ is not None: cs_['log_errors'] = _S['log_err']
    if not cs_ or not cs_.get('on'): return
    with _SL:
        a_ = dict(_ACC); _ACC.update(n=0, n_age=0, age=0.0, age_max=0.0, gap=0.0, gap_max=0.0)
    n_ = a_['n']; na_ = a_['n_age']
    last_ = dict(frames=n_, age_ms_avg=round(a_['age'] / na_ * 1000, 2) if na_ else None, age_ms_max=round(a_['age_max'] * 1000, 2) if na_ else None,
                 gap_ms_avg=round(a_['gap'] / n_ * 1000, 2) if n_ else None, gap_ms_max=round(a_['gap_max'] * 1000, 2) if n_ else None, at=time.strftime('%H:%M:%S'))
    cs_['last60'] = last_
    if n_:
        age_t_ = 'age avg %.1f ms max %.1f ms' % (last_['age_ms_avg'], last_['age_ms_max']) if na_ else 'age unknown (the two clocks differ)'
        note('server', 'STAMP 60 s: frames %d, %s, gap avg %.1f ms max %.1f ms, transport %s, since start: frames %d, numbers skipped %d, bad headers %d, resync %d, ring dropped %d (late %d, torn %d), clock odd %d'
             % (n_, age_t_, last_['gap_ms_avg'], last_['gap_ms_max'], cs_['transport'], cs_['frames'], cs_['missed'], cs_['hdr_bad'], cs_['resync'],
                cs_['ring_late'] + cs_['ring_torn'], cs_['ring_late'], cs_['ring_torn'], cs_['age_bad']))
    else: note('server', 'STAMP 60 s: no frame')


def _loop():   # POLL_S 초마다 깬다. cam_pipe.log 가 자라지 않았고 쓸 줄도 없으면 파일 크기 한 번 보는 것 말고는 아무것도 하지 않는다
    while True:
        time.sleep(POLL_S)
        _S['wakes'] += 1; grew_ = False
        try: grew_ = bool(_tail())
        except Exception: pass
        try: _summary()
        except Exception: pass
        if _Q:
            try: _flush()
            except Exception: pass
        elif not grew_: _S['idle_wakes'] += 1


def _last():   # 서버가 끝날 때 남은 줄을 쓴다
    for f_ in (_tail, _flush):
        try: f_()
        except Exception: pass


def _new_stats(on_):
    return dict(on=bool(on_), transport='pipe', seq=0, age_ms=None, gap_ms=None, frames=0, missed=0, hdr_bad=0, resync=0, ring_late=0, ring_torn=0, age_bad=0, last60=None,
                fallback='', log=_S['path'] or '', log_errors=0)


def log_start():   # lv_spawn 이 cam_pipe 를 띄운 뒤에 부른다. 오류를 내지 않는다
    try:
        if not hasattr(APP.LIVE, 'cam_stamp'):
            _S['path'] = log_path()
            APP.LIVE.cam_stamp = _new_stats(False); APP.LIVE.cam_stamp_recent = collections.deque(maxlen=64)
            _ring_once()
        if _S['thread'] is None and _S['path']:
            _S['t_sum'] = time.monotonic()
            note('server', 'server started (pid %d): camera log on, time stamps %s, shared memory %s' % (os.getpid(), 'on' if wanted() else 'off', 'on' if wanted() and os.environ.get('LWIR_CAM_SHM') == '1' else 'off'))
            if not wanted() and os.environ.get('LWIR_CAM_SHM') == '1': note('server', 'LWIR_CAM_SHM=1 is ignored: it needs LWIR_CAM_STAMP=1')
            _S['thread'] = threading.Thread(target=_loop, daemon=True); _S['thread'].start()
            atexit.register(_last)
        APP.LIVE.cam_stamp['log_errors'] = _S['log_err']
    except Exception: pass


# ---------------------------------------------------------------- B. 프레임 시간표, C. 공유 메모리 고리
MAGIC = b'LWF1'
HDR = struct.Struct('<4sHHQQIIIIHHI')   # magic, version, flags, seq, t_mono_ns, dt_us, slot, payload, ring_id, width, height, check
HDR_N = 48
_W12 = struct.Struct('<12I')
_Q8 = struct.Struct('<Q')
RING_MAGIC = b'LWR1'
RING_HDR = struct.Struct('<4sIIIIIIIQ')   # magic, version, ring_id, nslots, stride, payload, width, height, pid  (64 바이트 가운데 앞 40)
RING_HDR_N = 64
SLOT_HEAD = 16
SHM_DIR = '/dev/shm'
assert HDR.size == HDR_N and RING_HDR.size == 40


def wanted():   # 시간표를 쓰라고 했는가 (정확히 "1" 일 때만. cam_pipe 도 같은 규칙으로 읽는다)
    return os.environ.get('LWIR_CAM_STAMP') == '1'


def stamped(p_):   # 이 cam_pipe 는 머리를 붙여 보내도록 띄운 것인가
    return getattr(p_, 'lwir_stamp', False)


def ring_path():
    return '%s/lwir_cam_ring_%d' % (SHM_DIR, os.getuid() if hasattr(os, 'getuid') else 0)


def _pid_alive(pid_):   # 리눅스의 /proc 으로만 본다 (신호를 보내지 않는다). /proc 이 없는 곳에서는 "살아 있음"으로 답해 아무것도 지우지 않게 한다
    if not os.path.isdir('/proc/self'): return True
    return os.path.exists('/proc/%d' % pid_)


def ring_remove_stale():   # 남은 고리 파일을 지운다. 지웠으면 그 경로, 아니면 ''. 오류를 내지 않는다
    # 지우는 조건 (모두 맞아야 한다): 보통 파일이고(링크 아님), 내 것이고, 고리 머리(LWR1)가 맞고, 머리에 적힌 cam_pipe 프로세스가 더는 없다.
    # cam_pipe 는 공유 메모리를 켠 채 시작할 때 스스로 옛 고리를 갈아 치운다. 이 함수는 강제 종료된 뒤 공유 메모리를 끈 경우에 남는 파일(약 2.4 MB)을 치운다.
    try:
        path_ = ring_path()
        st_ = os.lstat(path_)
        import stat as st_mod
        if not st_mod.S_ISREG(st_.st_mode) or st_.st_size < RING_HDR_N: return ''
        if hasattr(os, 'getuid') and st_.st_uid != os.getuid(): return ''
        fd_ = os.open(path_, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
        try: head_ = os.read(fd_, RING_HDR.size)
        finally: os.close(fd_)
        if len(head_) < RING_HDR.size: return ''
        h_ = RING_HDR.unpack(head_)
        if h_[0] != RING_MAGIC or h_[8] <= 1 or _pid_alive(int(h_[8])): return ''
        os.unlink(path_); return path_
    except Exception: return ''


def _ring_once():   # 서버가 시작할 때 한 번. 시간표를 켠 경우에는 첫 cam_pipe 를 띄우기 전에 부른다 (새 cam_pipe 가 만들 고리와 엇갈리지 않게)
    if _S['ring_removed'] is None: return
    r_ = ring_remove_stale(); _S['ring_removed'] = None
    if r_: note('server', 'removed a shared memory ring left by a cam_pipe that is gone (%s)' % r_)


def parse_header(b_):   # 맞는 머리면 (flags, seq, t_ns, dt_us, slot, payload, ring_id, width, height), 아니면 None
    if bytes(b_[:4]) != MAGIC: return None
    w_ = _W12.unpack(b_); x_ = 0x5A17C0DE
    for v_ in w_[:11]: x_ ^= v_
    if x_ != w_[11]: return None
    h_ = HDR.unpack(b_)
    if h_[1] != 1: return None
    return h_[2:11]


class Ring:   # cam_pipe 가 만든 고리를 읽기 전용으로 붙인다. 안 맞으면 ValueError / OSError
    def __init__(self, ring_id_, payload_):
        path_ = ring_path()
        fd_ = os.open(path_, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
        try:
            st_ = os.fstat(fd_)
            if hasattr(os, 'getuid') and st_.st_uid != os.getuid(): raise ValueError('ring file belongs to another user')
            if st_.st_size < RING_HDR_N: raise ValueError('ring file too small (%d bytes)' % st_.st_size)
            self.mm = mmap.mmap(fd_, 0, access=mmap.ACCESS_READ)
        finally: os.close(fd_)
        self.mv = memoryview(self.mm)
        try:
            magic_, ver_, rid_, self.nslots, self.stride, pay_, w_, h_, pid_ = RING_HDR.unpack_from(self.mm, 0)
            if magic_ != RING_MAGIC or ver_ != 1: raise ValueError('not a ring of this version')
            if rid_ != ring_id_: raise ValueError('ring id %d, header says %d (ring of another cam_pipe)' % (rid_, ring_id_))
            if pay_ != payload_ or self.stride < SLOT_HEAD + pay_ or not 1 <= self.nslots <= 64: raise ValueError('ring sizes do not fit')
            if len(self.mm) < RING_HDR_N + self.nslots * self.stride: raise ValueError('ring file shorter than its header says')
        except Exception:
            self.close(); raise
        self.ring_id = rid_; self.payload = pay_

    def read(self, slot_, seq_, out_):   # '' = out_ 에 그림을 받음, 'late' = 읽기 전에 이미 덮어쓰임, 'torn' = 읽는 사이에 덮어쓰임
        if slot_ >= self.nslots: return 'late'
        off_ = RING_HDR_N + slot_ * self.stride
        if _Q8.unpack_from(self.mm, off_)[0] != seq_: return 'late'
        out_[:] = self.mv[off_ + SLOT_HEAD:off_ + SLOT_HEAD + self.payload]
        if _Q8.unpack_from(self.mm, off_)[0] != seq_: return 'torn'
        return ''

    def close(self):
        try: self.mv.release()
        except Exception: pass
        try: self.mm.close()
        except Exception: pass


def install_producer(process):
    """Swap readers under the queue lock used to accept their frames."""
    from ..web.preview_worker import set_preview_producer

    live = APP.LIVE
    with live.cond:
        set_preview_producer(live, process, before=time.perf_counter())
        live.proc = process
        live.proc_replay_src_t = getattr(live, 'replay_src_t', None)
        live.cam_lost = False
        live.proc_t0 = time.monotonic()
        live.cond.notify_all()


def spawn():   # lv_spawn 의 시간표 판: cam_pipe 에 같은 설정을 환경 변수로 넘겨 양쪽이 항상 같게 한다
    stamp_ = not _S['stamp_off']; shm_ = stamp_ and os.environ.get('LWIR_CAM_SHM') == '1' and not _S['shm_off']
    env_ = dict(os.environ); env_['LWIR_CAM_STAMP'] = '1' if stamp_ else '0'; env_['LWIR_CAM_SHM'] = '1' if shm_ else '0'
    _ring_once()
    from .replay_link import cam_argv   # 11-12: the same command as lv_spawn (the replay program while a recording is played)
    p_ = APP.lv_sp.Popen(cam_argv(), stdout=APP.lv_sp.PIPE, stderr=APP.LIVE.log, bufsize=0, env=env_)
    p_.lwir_stamp = stamp_
    install_producer(p_)
    log_start()
    cs_ = APP.LIVE.cam_stamp; cs_['on'] = stamp_; cs_['fallback'] = _S['stamp_off'] or _S['shm_off']
    if not stamp_: cs_['transport'] = 'pipe'
    key_ = (stamp_, shm_)
    if key_ not in _S['said']: _S['said'].add(key_); note('server', 'cam_pipe started with time stamps %s, shared memory %s' % ('on' if stamp_ else 'off', 'on' if shm_ else 'off'))
    APP.lv_th.Thread(target=APP.lv_reader, args=(APP.LIVE.proc,), daemon=True).start()


def _take(p_, pend_, mv_):   # mv_ 를 채운다: 먼저 미리 읽어 둔 것(pend_)에서, 다음에 파이프에서. False = 파이프가 끝남
    n_ = len(mv_); got_ = min(len(pend_), n_)
    if got_: mv_[:got_] = pend_[:got_]; del pend_[:got_]
    while got_ < n_:
        k_ = p_.stdout.readinto(mv_[got_:])
        if not k_: return False
        got_ += k_
    return True


def _resync(p_, pend_, hb_, limit_):   # 다음 맞는 머리를 찾는다. True = hb_ 에 넣음, False = 파이프가 끝남, None = limit_ 바이트 안에 없음
    pend_[:0] = hb_; del pend_[:1]; skipped_ = 1
    while True:
        i_ = pend_.find(MAGIC)
        if i_ < 0: d_ = max(0, len(pend_) - 3); del pend_[:d_]; skipped_ += d_
        else:
            del pend_[:i_]; skipped_ += i_
            if len(pend_) >= HDR_N:
                if parse_header(pend_[:HDR_N]) is not None: hb_[:] = pend_[:HDR_N]; del pend_[:HDR_N]; return True
                del pend_[:1]; skipped_ += 1; continue
        if skipped_ > limit_: return None
        d_ = p_.stdout.read(65536)
        if not d_: return False
        pend_ += d_


def _end(p_, ring_):   # 읽기 끝: 전 방식의 끝과 같다
    if ring_ is not None: ring_.close()
    with APP.LIVE.cond:
        if p_ is APP.LIVE.proc: APP.LIVE.cam_lost = True
        APP.LIVE.cond.notify_all()


def _stop_and_drain(p_, ring_):   # cam_pipe 를 끝내고(SIGTERM) 파이프가 닫힐 때까지 읽어 버린다. 그 뒤는 서버가 전처럼 다시 띄운다
    try: p_.terminate()
    except Exception: pass
    try:
        while p_.stdout.read(65536): pass
    except Exception: pass
    _end(p_, ring_)


SIZE_BAD_WAIT = (20, 60, 300, 900)   # 그림 크기가 안 맞는 cam_pipe 를 다시 띄우기까지 기다리는 초 (연달아 몇 번째인가에 따라 늘어난다)


def reader(p_):   # lv_reader 의 시간표 판. 예상하지 못한 오류가 나도 파이프를 끝까지 읽어 버려서 cam_pipe 가 막힌 파이프에 걸려 있지 않게 한다
    box_ = [None]   # 붙여 둔 고리 (오류가 났을 때 닫으려고)
    try: return _reader(p_, box_)
    except Exception as e_:
        try:
            _S['reader_err'] += 1
            if _S['reader_err'] >= 3 and not _S['stamp_off']: _S['stamp_off'] = 'time stamp reader failed %d times (%s)' % (_S['reader_err'], type(e_).__name__)
            say('LIVE camera: time stamp reader stopped on an unexpected error (%s: %s), restarting cam_pipe%s' % (type(e_).__name__, str(e_)[:120], ' the old way (time stamps off)' if _S['stamp_off'] else ''))
        except Exception: pass
        _stop_and_drain(p_, box_[0])


def _reader(p_, box_):   # 프레임을 줄에 넣는 부분은 lv_reader 와 같다
    live = APP.LIVE
    ev_ = getattr(p_, 'lwir_eval', None)
    lossless_ = replay_eval.lossless(p_)

    def current_source(tagged):
        # Replay switching can retire this source before installing the next process.
        return (not live.stopped and p_ is live.proc
                and (not tagged or live.proc_replay_src_t == getattr(live, 'replay_src_t', None)))

    n_ = APP.LIVE_W * APP.LIVE_H * 2; buf_ = bytearray(n_); cnt_ = 0; tl_ = time.monotonic()
    mv_ = memoryview(buf_); hb_ = bytearray(HDR_N); hv_ = memoryview(hb_); pend_ = bytearray(); ring_ = None; last_seq_ = 0; good_ = 0
    cs_ = APP.LIVE.cam_stamp; recent_ = APP.LIVE.cam_stamp_recent
    last_tag_t = None
    while True:
        if not _take(p_, pend_, hv_): return _end(p_, ring_)
        h_ = parse_header(hb_)
        if h_ is None:
            cs_['hdr_bad'] += 1
            r_ = _resync(p_, pend_, hb_, 2 * (n_ + HDR_N))
            if r_ is False: return _end(p_, ring_)
            if r_ is None:
                if good_ == 0:
                    _S['stamp_off'] = 'no time stamp header from cam_pipe (old build?)'
                    say('LIVE camera: cam_pipe sends no time stamp header (old build?): time stamps off, restarting it the old way')
                else: say('LIVE camera: time stamp header lost after %d frames and not found again, restarting cam_pipe' % good_)
                return _stop_and_drain(p_, ring_)
            cs_['resync'] += 1; h_ = parse_header(hb_)
            if cs_['resync'] <= 20: say('LIVE camera: time stamp header was out of step, found the next one')
        flags_, seq_, tns_, dtus_, slot_, nb_, rid_, w_, hh_ = h_
        if nb_ != n_ or w_ != APP.LIVE_W or hh_ != APP.LIVE_H:
            _S['size_bad'] += 1; wait_ = SIZE_BAD_WAIT[min(_S['size_bad'], len(SIZE_BAD_WAIT)) - 1]
            with APP.LIVE.cond:
                if p_ is APP.LIVE.proc: APP.LIVE.cam_next_spawn = time.monotonic() + wait_   # lv_cam_wait 는 이 시각 전에는 다시 띄우지 않는다
            say('LIVE camera: cam_pipe sends %dx%d pictures (%d bytes), the server expects %dx%d: cam_pipe stopped, next start in %d s (%d in a row)' % (w_, hh_, nb_, APP.LIVE_W, APP.LIVE_H, wait_, _S['size_bad']))
            return _stop_and_drain(p_, ring_)
        good_ += 1
        if _S['size_bad']: _S['size_bad'] = 0
        if last_seq_ and seq_ > last_seq_ + 1: cs_['missed'] += seq_ - last_seq_ - 1
        last_seq_ = seq_
        if flags_ & 1:
            if APP.LIVE.stopped: continue
            if ring_ is None or ring_.ring_id != rid_:
                if ring_ is not None: ring_.close(); ring_ = None; box_[0] = None
                try: ring_ = Ring(rid_, n_); box_[0] = ring_
                except Exception as e_:
                    _S['shm_off'] = 'shared memory ring could not be opened (%s)' % type(e_).__name__
                    say('LIVE camera: shared memory ring could not be opened (%s: %s): shared memory off, restarting cam_pipe with pictures through the pipe' % (type(e_).__name__, e_))
                    return _stop_and_drain(p_, None)
            bad_ = ring_.read(slot_, seq_, mv_)
            if bad_: cs_['ring_late' if bad_ == 'late' else 'ring_torn'] += 1; continue
            cs_['transport'] = 'shm'
        else:
            if not _take(p_, pend_, mv_): return _end(p_, ring_)
            if APP.LIVE.stopped: continue
            cs_['transport'] = 'pipe'
        age_ = time.monotonic() - tns_ * 1e-9; gap_ = dtus_ * 1e-6
        raw_ = np.frombuffer(buf_, np.uint16).reshape(APP.LIVE_H, APP.LIVE_W).copy()
        ta_ = time.perf_counter()
        if flags_ & 2:
            # Two frames can arrive in one clock tick after backpressure releases.
            if last_tag_t is not None and ta_ <= last_tag_t:
                ta_ = math.nextafter(last_tag_t, math.inf)
            last_tag_t = ta_
        tagged = bool(flags_ & 2)
        scored_replay = tagged and lossless_
        hold_frame = scored_replay and flags_ & 4
        enqueued = False
        while True:
            with live.cond:
                while current_source(tagged) and scored_replay and not hold_frame and len(live.q) == live.q.maxlen:
                    live.cond.wait(timeout=0.1)
                if not current_source(tagged):
                    break
                # The tag must exist before consumers see the frame. Never wait for scoring here.
                if not tagged or replay_eval.tag(p_, ta_, rid_, slot_, flags_, blocking=False):
                    if hold_frame:
                        # Pause keeps the same displayed frame without advancing tracker state.
                        live.last_frame_t = time.monotonic()
                        break
                    if len(live.q) == live.q.maxlen: live.q_drop += 1
                    live.q.append((ta_, raw_))
                    cnt_ += 1; now_ = time.monotonic(); live.last_frame_t = now_; live.cam_fail_n = 0; live.cam_unrec_n = 0; live.frames_in += 1
                    if now_ - tl_ >= 1.0: live.cam_fps = cnt_ / (now_ - tl_); cnt_ = 0; tl_ = now_
                    if scored_replay:
                        live.cond.notify_all()
                    else:
                        live.cond.notify()
                    enqueued = True
                    break
            # Wait outside the input condition. Release before retrying to preserve the lock order used by seek.
            if ev_.lock.acquire(timeout=0.1):
                ev_.lock.release()
        if not enqueued:
            continue
        rc_ = APP.LIVE.rec
        if rc_ is not None:
            with rc_['cond']:
                if len(rc_['q']) < 240: rc_['q'].append((ta_, raw_)); rc_['cond'].notify()
                else: rc_['dropped'] += 1
        recent_.append((ta_, seq_, tns_ * 1e-9, dtus_))
        cs_['seq'] = seq_; cs_['frames'] += 1; cs_['gap_ms'] = round(gap_ * 1000, 2)
        ok_ = -0.005 <= age_ < 3600.0   # 아니면 두 프로그램의 시계가 다르다 (같은 장치의 리눅스에서는 일어나지 않아야 한다)
        if ok_: cs_['age_ms'] = round(age_ * 1000, 2)
        else: cs_['age_bad'] += 1
        with _SL:
            _ACC['n'] += 1; _ACC['gap'] += gap_
            if gap_ > _ACC['gap_max']: _ACC['gap_max'] = gap_
            if ok_:
                _ACC['n_age'] += 1; _ACC['age'] += age_
                if age_ > _ACC['age_max']: _ACC['age_max'] = age_
