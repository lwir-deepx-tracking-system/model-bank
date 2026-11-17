"""서버가 잠깐 멈추는 까닭을 찾는 기록 (2026-11-13). 기본으로 꺼져 있고 LWIR_STALL_DIAG=1 (정확히 "1") 일 때만 켜진다.

왜 만들었나: 카메라 프로그램(cam_pipe)의 INPUT_GAP 줄을 보면, 15~30 초에 한 번 서버가 파이프에서 프레임을 130~260 ms 동안
꺼내 가지 않았다 (카메라는 정상). 서버 안의 무엇이 그동안 읽는 스레드를 세웠는지는 그 줄로 알 수 없다. 이 파일은 켰을 때만
아래 줄을 카메라 기록(camera_events.log, 서버 줄 "[server] ...")에 남겨서 INPUT_GAP 줄과 시각으로 맞춰 볼 수 있게 한다.

  READER_SLOW  카메라 프레임 한 장을 받는 데 100 ms 이상 걸림: readinto 횟수, 첫 번째와 가장 긴 readinto, 복사, LIVE.cond 를
               기다린 시간과 잡고 있던 시간, 그 프레임 동안 10 ms 타이머가 늦은 정도, 읽는 스레드가 CPU 코어를 기다린 시간.
               145 ms 이상이면 언제나 쓴다. 그보다 짧고 첫 readinto 만 길었으며 타이머도 코어 기다림도 30 ms 미만이면 쓰지 않고
               "카메라를 기다렸을 뿐" 으로 센다 (cam_pipe 는 같은 그림을 3 장까지 보내지 않으므로 133 ms 까지는 정상이다.
               셔터가 닫힌 동안도 그렇다). 줄은 타이머가 그 프레임 뒤에 한 번 깬 다음에 쓴다 (보통 한 프레임 뒤. 그래야
               그 멈춤 때문에 타이머가 늦은 정도가 줄에 들어간다).
  GIL_STALL    10 ms 마다 깨는 스레드가 80 ms 이상 늦게 깸. 그중 CPU 코어를 기다린 시간, 그동안 스레드마다 쓴 CPU (많이 쓴 3 개,
               파이썬이 만들지 않은 스레드도 본다), 스레드마다 멈춤이 끝난 뒤에 있는 자리 (CPU 를 많이 쓴 스레드부터).
  GC_SLOW      가비지 컬렉션 한 번이 CPU 를 30 ms 이상 씀: 세대, 쓴 CPU, 시계로 잰 시간, 돌린 스레드, 쌓아 둔 프레임 기록 수.
  LOCK_HELD    LIVE.cond 를 50 ms 이상 잡고 있었음: 누가, 어느 줄에서 잡고 어느 줄에서 놓았는지.
  SPAN_SLOW    따로 잰 몇 군데(프레임 기록 잘라내기, NPU 상태 묻기, 녹화 목록 다시 훑기)가 20 ms 이상 걸림.
  LOOP_SLOW    프레임 반복의 한 장에서 검출 결과를 기다리는 것 말고 한 부분이 80 ms 이상 걸림.
  STALL_DIAG   켜질 때 한 줄, 60 초마다 요약 한 줄 (기준에 못 미친 가장 큰 값들. 줄이 없을 때 "재고 있었다" 는 증거).

읽는 법 (2026-11-13 검토 뒤):
  - GIL_STALL 의 "CPU by thread" 에서 멈춘 시간만큼 CPU 를 쓴 스레드가 GIL 을 잡고 있던 스레드다. "threads that have ended" 가
    그만큼이면 잡고 있던 스레드는 타이머가 보기 전에 끝났다 (페이지 요청은 요청마다 스레드가 따로 생기고 곧 끝난다). 둘 다
    아니면 잡고 있던 스레드가 CPU 를 쓰지 않고 서 있었거나 (GIL 을 놓지 않는 기다림), 타이머가 코어를 기다린 것이다 (줄에 있다).
    NPU 같은 다른 라이브러리의 스레드는 GIL 없이도 CPU 를 쓰므로, "not in the thread list of Python" 인 것은 그것만으로 범인이 아니다
    (막 끝나는 파이썬 스레드도 잠깐 그렇게 보인다).
  - 스레드 자리는 멈춤이 끝난 "뒤" 의 것이다. 잡고 있던 스레드가 이미 몇 줄 더 갔을 수 있다. stall_diag.py 의 _gc_cb 안에 있는
    스레드가 보여도 컬렉션의 증거가 아니다 (콜백에 들어가면서 GIL 을 넘겨주고 기다리는 자리일 뿐이다).
  - GC_SLOW 는 "CPU" 가 멈춘 시간과 비슷할 때만 그 멈춤의 까닭이다. 시계 시간만 길고 CPU 가 짧으면 컬렉션을 돌린 스레드가
    앞뒤로 GIL 을 기다린 것이다 (다른 스레드가 잡고 있었다). 60 초 요약의 컬렉션 값도 CPU 다.
  - READER_SLOW: 타이머가 늦었다 = GIL. 코어를 기다린 시간이 크다 = 운영체제가 읽는 스레드를 돌리지 않았다. 둘 다 아니다 =
    cam_pipe 가 보내지 않았다 (까닭은 INPUT_GAP 줄에 있다).

멈춘 "동안" 의 스택은 남기지 않는다 (2026-11-13 에 해 보고 뺐다). 파이썬에서 GIL 없이 스택을 쓸 수 있는 것은 faulthandler 의
dump_traceback_later 뿐인데, (1) 타이머가 그것을 다시 맞출 때 쓰고 있는 덤프가 끝나기를 GIL 을 잡은 채 기다리고, 덤프는 옛 파이썬에서
글자 하나에 write 한 번이라 (3.8: 10 KB 에 write 6400 번) 느린 파일에서는 그만큼 멈춤이 더 길어지며 (PC 의 느린 폴더에서 1.8 초),
(2) 덤프하는 스레드는 다른 스레드가 도는 중에도 잠금 없이 파이썬 객체를 읽는다. 재는 것이 재려는 멈춤을 늘리거나 서버를 죽일 수
있어서, 스레드별 CPU 로 스레드를 가려내고 멈춤 뒤의 자리를 보는 데서 멈췄다.

줄 종류마다 1 초에 5 줄, 1 분에 30 줄까지만 쓴다 (넘친 수는 다음 줄 끝에 적는다). 줄은 메모리의 목록에 넣기만 하고 파일에는
camera_extra 의 기록 스레드가 쓴다. 이 파일은 잠금을 하나도 쓰지 않는다: 가비지 컬렉션 콜백이 어느 스레드의 어느 자리에서든
끼어들 수 있어서, 잠금을 쓰면 자기 자신을 기다리게 될 수 있다. 그래서 세는 값은 드물게 하나쯤 어긋날 수 있다.

꺼져 있으면: 스레드도 콜백도 만들지 않고 LIVE.cond 도 그대로다. 읽는 스레드는 프레임마다 "dg_ is not None" 한 번만 본다.
켜져 있으면 타이머 스레드가 10 ms 마다 스레드별 CPU 시계를 읽는다. PC 에서 스레드 25 개에 한 번에 0.03 ms, 다른 스레드가 바쁠 때는
0.15 ms 쯤 들었다 (타이머 스레드 전체가 코어 하나의 3 % 쯤). 보드에서 실제로 든 값은 60 초 요약의 "the timer thread of this log used" 에 나온다.
"""
import collections
import gc
import os
import sys
import threading
import time

from ..common.state import APP
from . import camera_extra

READER_MS = 100.0   # READER_SLOW: 앞 프레임을 줄에 넣은 뒤부터 이 프레임을 줄에 넣을 때까지 (보통 33 ms)
READER_SURE_MS = 145.0   # 이만큼이면 언제나 쓴다: cam_pipe 는 같은 그림을 3 장까지 보내지 않아 정상일 때 133 ms 까지 비고, 그 INPUT_GAP 줄은 150 ms 부터다
QUIET_S = 0.030     # 첫 readinto 만 길었던 프레임: 타이머도 코어 기다림도 이보다 짧으면 "카메라를 기다렸을 뿐"
WAIT_S = 0.5        # 느린 프레임의 줄은 타이머가 그 프레임 뒤에 한 번 깰 때까지 미룬다. 길어도 이만큼
GIL_MS = 80.0       # GIL_STALL: 10 ms 타이머가 이만큼 늦게 깸
GC_MS = 30.0        # GC_SLOW: 컬렉션을 돌린 스레드가 그동안 쓴 CPU (시계 시간이 아니다)
LOCK_MS = 50.0      # LOCK_HELD
SPAN_MS = 20.0      # SPAN_SLOW
LOOP_MS = 80.0      # LOOP_SLOW
TICK_S = 0.010
SUM_S = 60.0
RATE_S = 5          # 줄 종류마다 1 초에
RATE_MIN = 30       # 그리고 1 분에 쓰는 줄 수 (잘못된 상태가 이어져도 카메라 기록이 밀려 나가지 않게)
STACK_DEPTH = 3     # GIL_STALL: 스레드마다 안쪽에서부터 몇 단계
STACK_DEPTH_TOP = 8 # 그동안 CPU 를 가장 많이 쓴 파이썬 스레드는 더 깊이 (TOP_S 이상 썼을 때)
TOP_S = 0.020
CPU_TOP = 3         # GIL_STALL: CPU 를 많이 쓴 스레드 몇 개의 이름을 적는가
IDS_S = 1.0         # 스레드 목록(/proc/self/task)을 다시 읽는 간격
STACK_LINES = 4     # GIL_STALL 한 번에 쓰는 스레드 줄 수 (카메라 기록은 한 줄 2000 글자까지)
STACK_CHARS = 1700

ON = False          # start() 가 켠다. 호출하는 쪽의 얇은 갈고리는 이 값만 본다
_RL = {}            # 줄 종류 -> [1 초 창의 시작, 그 안의 줄 수, 1 분 창의 시작, 그 안의 줄 수, 쓰지 않은 줄 수]
_LATE = collections.deque(maxlen=512)   # (타이머가 깬 시각, 늦은 초). 5 ms 이상 늦은 것만
_S = dict(thread=None, stop=False, n_stall=0, gc_t=0.0, gc_c=0.0, waits=0, woke=0.0)   # waits: slow frames that only waited for the camera since the last READER_SLOW line. woke: when the timer last woke
_M0 = dict(late=0.0, gc_n=0, gc2_n=0, gc2=0.0, gc=0.0, rd_n=0, rd=0.0, cam_waits=0, lock=0.0, skipped=0)   # 60 초 요약에 쓰는 값 (gc, gc2 는 CPU 초)
_M = dict(_M0)
_SKIP = (threading.__file__, __file__)
_tcpu = getattr(time, 'thread_time', time.process_time)
_cg = getattr(time, 'clock_gettime', None)   # Linux: the CPU clock of any thread of this process is clock id ((~thread id) << 3) | 6


def wanted():   # 정확히 "1" 일 때만 (LWIR_CAM_STAMP 와 같은 규칙)
    return os.environ.get('LWIR_STALL_DIAG') == '1'


def _ok(kind_, now_):   # 이 종류의 줄을 지금 써도 되는가
    r_ = _RL.get(kind_)
    if r_ is None: r_ = _RL[kind_] = [now_, 0, now_, 0, 0]
    if now_ - r_[0] >= 1.0: r_[0] = now_; r_[1] = 0
    if now_ - r_[2] >= 60.0: r_[2] = now_; r_[3] = 0
    if r_[1] >= RATE_S or r_[3] >= RATE_MIN: r_[4] += 1; _M['skipped'] += 1; return False
    r_[1] += 1; r_[3] += 1; return True


def say(kind_, text_):   # 카메라 기록에 한 줄 (줄 수 제한을 넘으면 쓰지 않고 False). 오류를 내지 않는다
    try:
        if not _ok(kind_, time.monotonic()): return False
        r_ = _RL[kind_]; n_ = r_[4]; r_[4] = 0
        camera_extra.note('server', '%s %s%s' % (kind_, text_, ' (%d %s lines before this one were not written: rate limit)' % (n_, kind_) if n_ else ''))
        return True
    except Exception: return False


def _where(code_, line_):   # "camera/camera_link.py:29 lv_reader"
    return '%s:%d %s' % ('/'.join(code_.co_filename.replace('\\', '/').split('/')[-2:]), line_, code_.co_name)


def _site():   # the nearest caller outside this file and threading.py ("with cond:" goes through Condition.__enter__), as (code, line). Never raises: it runs while the lock is held
    try:
        f_ = sys._getframe(2)
        while f_ is not None and f_.f_code.co_filename in _SKIP: f_ = f_.f_back
        return (f_.f_code, f_.f_lineno) if f_ is not None else None
    except Exception: return None


def _late_since(t0_, t1_=None):   # the most the 10 ms timer was late in a sleep that overlaps t0_..t1_ (perf_counter; no t1_ = until now). None = the timer thread is not running
    if _S['thread'] is None: return None
    return max([l_ for w_, l_ in list(_LATE) if w_ >= t0_ and (t1_ is None or w_ - l_ - TICK_S <= t1_)], default=0.0)


def _sched_open():   # /proc/thread-self/schedstat of the calling thread ("ns on a CPU core, ns waiting for one, time slices"), kept open. None where there is no such file
    try: return os.open('/proc/thread-self/schedstat', os.O_RDONLY)
    except Exception: return None


def _sched(fd_):   # (seconds on a core, seconds waited for a core) of that thread so far, or None. A few microseconds
    if fd_ is None: return None
    try: a_ = os.pread(fd_, 80, 0).split(); return (int(a_[0]) * 1e-9, int(a_[1]) * 1e-9)
    except Exception: return None


def _task_ids(skip_=None):   # every thread of this process, also the ones Python did not start (NPU and camera libraries). [] where it cannot be asked
    if _cg is None: return []
    try: return [i_ for i_ in map(int, os.listdir('/proc/self/task')) if i_ != skip_]
    except Exception: return []


def _cpu_of(ids_):   # thread id -> CPU seconds it has used so far. A thread that has ended is left out
    try: return {i_: _cg(((~i_) << 3) | 6) for i_ in ids_}
    except Exception: pass
    out_ = {}
    for i_ in ids_:
        try: out_[i_] = _cg(((~i_) << 3) | 6)
        except Exception: pass
    return out_


# ---------------------------------------------------------------- LOCK_HELD: LIVE.cond 를 누가 오래 잡았나
class TimedLock:
    """The lock inside APP.LIVE.cond while the log is on: an RLock (what threading.Condition makes by itself) that notes when it
    was asked for, taken and let go. Condition picks up _release_save / _acquire_restore / _is_owned from the lock it is given,
    so wait() and notify() work as before; the time spent inside wait() is not counted as holding. The fields are written only by
    the thread that owns the lock. Logging happens after the lock is let go, so it never makes a hold longer."""

    def __init__(self, name_):
        self.name = name_; self.lk = threading.RLock(); self.depth = 0; self.t_ask = 0.0; self.t_got = 0.0; self.site = None
        self.marks = {}   # thread id -> [asked, taken, let go] of that thread's last hold (the camera reader asks for its own: READER_SLOW)

    def acquire(self, blocking=True, timeout=-1):
        ta_ = time.perf_counter()
        if not self.lk.acquire(blocking, timeout): return False
        self.depth += 1
        if self.depth == 1: self.t_ask = ta_; self.t_got = time.perf_counter(); self.site = _site()
        return True

    __enter__ = acquire

    def _out(self):   # the owner lets go for good (not a nested release): what to say afterwards
        t_ = time.perf_counter(); m_ = self.marks.get(threading.get_ident())
        if m_ is not None: m_[0] = self.t_ask; m_[1] = self.t_got; m_[2] = t_
        return (t_ - self.t_got, self.t_got - self.t_ask, self.site, _site()) if (t_ - self.t_got) * 1000.0 >= LOCK_MS or t_ - self.t_got > _M['lock'] else None

    def _tell(self, o_):
        try:
            if o_[0] > _M['lock']: _M['lock'] = o_[0]
            if o_[0] * 1000.0 < LOCK_MS: return
            say('LOCK_HELD', '%s was held for %.0f ms by thread %s: taken at %s, let go at %s (it had waited %.1f ms to get it)'
                % (self.name, o_[0] * 1000.0, threading.current_thread().name, _where(*o_[2]) if o_[2] else '?', _where(*o_[3]) if o_[3] else '?', o_[1] * 1000.0))
        except Exception: pass

    def release(self):
        if not self.lk._is_owned(): return self.lk.release()   # not the owner: RLock raises as it always did
        if self.depth > 1: self.depth -= 1; return self.lk.release()   # nested: the hold goes on
        try: o_ = self._out()
        except Exception: o_ = None   # whatever happens in the timing, the lock is let go
        self.depth = 0; self.lk.release()
        if o_ is not None: self._tell(o_)

    def __exit__(self, *a_): self.release()

    def _is_owned(self): return self.lk._is_owned()

    def _release_save(self):   # Condition.wait(): let go completely, however deep
        try: o_ = self._out()
        except Exception: o_ = None
        d_ = self.depth; self.depth = 0; st_ = self.lk._release_save()
        if o_ is not None: self._tell(o_)
        return (st_, d_)

    def _acquire_restore(self, st_):   # Condition.wait() comes back: a new hold starts here
        ta_ = time.perf_counter(); self.lk._acquire_restore(st_[0]); self.depth = st_[1]; self.t_ask = ta_; self.t_got = time.perf_counter(); self.site = _site()


# ---------------------------------------------------------------- READER_SLOW: 읽는 스레드가 프레임 한 장에 쓴 시간
class ReaderProbe:
    """One per camera reader thread, made in that thread. lv_reader calls fill() in place of its own readinto loop; that is its
    only hook. The times after the last byte (copy, waiting for LIVE.cond, holding it) come from the TimedLock, so a frame is
    measured when the next one starts. A slow one is then kept until the 10 ms timer has woken once after the frame ended and
    is judged and written at the next frame start after that, normally one frame later: only then has the timer noted how late
    it was (written at once, the line could say "0 ms late" for the very stall that made the frame slow). If the timer does
    not wake for WAIT_S, or the pipe ends, the line is written without it and says so.
    A frame of READER_SURE_MS or more is always written. A shorter slow frame whose only slow part was the first readinto is
    either the camera sending late (cam_pipe holds back up to 3 repeated frames, so up to 133 ms without a frame happens in normal
    running, also while the shutter is closed) or the reader not getting the GIL or a CPU core after the bytes were there: it is
    written if the timer was QUIET_S or more late in that frame or the reader waited that long for a core, otherwise it is only
    counted as "only waited for the camera"."""

    def __init__(self, lock_):
        self.lk = lock_ if isinstance(lock_, TimedLock) else None; self.m = [0.0, 0.0, 0.0]; self.cur = None; self.amb = []; self.t_end = None   # amb: slow frames not judged yet
        self.ss = _sched_open(); self.s0 = _sched(self.ss)   # this thread's own CPU-core numbers (the file belongs to the thread that opens it)
        if self.lk is not None: self.lk.marks[threading.get_ident()] = self.m

    def fill(self, f_, mv_, n_):   # read one whole frame into mv_; returns the bytes read (less than n_ = the pipe ended, the caller's loop sees that)
        t0_ = t_ = time.perf_counter()
        try: self._judge(t0_)
        except Exception: self.cur = None; self.amb = []
        got_ = 0; k_n_ = 0; first_ = 0.0; long_ = 0.0; li_ = 0
        while got_ < n_:
            k_ = f_.readinto(mv_[got_:]); t2_ = time.perf_counter()
            if not k_: self._end(t2_); return got_
            got_ += k_; k_n_ += 1; d_ = t2_ - t_; t_ = t2_   # d_ includes the few bytecodes since the read before: a thread switch there is also "could not run"
            if k_n_ == 1: first_ = d_
            if d_ > long_: long_ = d_; li_ = k_n_
        self.cur = (t0_, t_, k_n_, first_, long_, li_)
        return got_

    def _judge(self, now_):   # at the start of every frame: write or count the slow frames whose time has come, measure the frame just read
        s1_ = _sched(self.ss); s0_ = self.s0; self.s0 = s1_
        if self.amb: self._due(now_, False)
        if self.cur is None: return
        t0_, t1_, k_n_, first_, long_, li_ = self.cur; self.cur = None; m_ = self.m
        queued_ = self.lk is not None and m_[0] >= t1_
        if self.lk is not None and not queued_: self.t_end = now_; return   # thrown away before the lock (LIVE.stopped): nothing to say
        end_ = m_[2] if queued_ else now_   # without a timed lock the frame "ends" when the next read is asked for
        before_ = max(t0_ - self.t_end, 0.0) if self.t_end is not None else 0.0; self.t_end = end_; total_ = end_ - t0_
        _M['rd_n'] += 1
        if total_ > _M['rd']: _M['rd'] = total_
        if (before_ + total_) * 1000.0 < READER_MS: return
        self.amb.append((t0_ - before_, before_, total_, k_n_, first_, long_, li_, (m_[0] - t1_, m_[1] - m_[0], m_[2] - m_[1]) if queued_ else None, end_, end_ - t1_,
                         (s1_[0] - s0_[0], s1_[1] - s0_[1]) if s0_ is not None and s1_ is not None else None))

    def _due(self, now_, all_):   # a slow frame is judged once the timer has woken after it ended (or cannot be waited for any longer)
        a_ = self.amb; self.amb = []
        for r_ in a_:
            if all_ or _S['thread'] is None or _S['woke'] >= r_[8] or now_ - r_[8] >= WAIT_S: self._decide(r_, now_)
            else: self.amb.append(r_)

    def _decide(self, r_, now_):
        held_ = _late_since(r_[0], r_[8]); span_ = r_[1] + r_[2]; sc_ = r_[10]
        woke_ = held_ is None or _S['woke'] >= r_[8]   # False = the timer has not woken since the frame ended: how late it was in the frame is not known yet
        if span_ * 1000.0 < READER_SURE_MS and span_ - r_[4] < 0.050 and held_ is not None and woke_ and held_ < QUIET_S and (sc_ is None or sc_[1] < QUIET_S):
            _S['waits'] += 1; _M['cam_waits'] += 1; return
        self._tell(r_, held_, woke_, now_)

    def _tell(self, r_, held_, woke_, now_):
        w0_, before_, total_, k_n_, first_, long_, li_, lk_, end_, tail_, sc_ = r_
        say('READER_SLOW', 'camera frame took %.0f ms from the first readinto call until it was queued (and %.0f ms passed before that call since the frame before was queued): '
            '%d readinto calls, the first %.0f ms (it includes waiting for the camera to send), the longest %.0f ms (call %d); after the last byte: %s; '
            'the 10 ms timer %s; %sended %.0f ms ago; %d slow frames before this one only waited for the camera'
            % (total_ * 1000.0, before_ * 1000.0, k_n_, first_ * 1000.0, long_ * 1000.0, li_,
               'copy %.1f ms, waiting for LIVE.cond %.1f ms, holding it %.1f ms' % tuple(v_ * 1000.0 for v_ in lk_) if lk_ else 'copy and LIVE.cond %.1f ms together (the lock is not timed)' % (tail_ * 1000.0),
               'is not running' if held_ is None else 'was up to %.0f ms late meanwhile' % (held_ * 1000.0) if woke_ else 'has not woken since this frame ended (how late it was is not known yet)',
               'the reader thread was on a CPU core %.1f ms and waited %.1f ms for one (from this first readinto call to that of the next frame); ' % (sc_[0] * 1000.0, sc_[1] * 1000.0) if sc_ is not None else '',
               (now_ - end_) * 1000.0, _S['waits']))
        _S['waits'] = 0

    def _end(self, now_):   # the pipe ended (this cam_pipe is gone): write what is still waiting, give back what was taken
        try: self._due(now_, True)
        except Exception: self.amb = []
        try:
            if self.lk is not None: self.lk.marks.pop(threading.get_ident(), None)
            if self.ss is not None: os.close(self.ss)
        except Exception: pass
        self.ss = None


def reader_probe():   # lv_reader asks once when it starts. None unless the log is on
    if not ON: return None
    try: return ReaderProbe(getattr(getattr(APP.LIVE, 'cond', None), '_lock', None))
    except Exception: return None


# ---------------------------------------------------------------- GC_SLOW: 가비지 컬렉션
def _gc_cb(phase_, info_):   # gc.callbacks: runs in the thread whose allocation started the collection, with the GIL (the collection itself never gives it up)
    # A collection is judged by the CPU its thread used, not by the clock: on the way into this function ("start" and "stop"
    # alike) the thread hands the GIL to whoever is asking for it, and with 26 threads someone always is. If that thread then
    # keeps the GIL for 200 ms, the clock says the collection took 200 ms. Waiting for the GIL uses no CPU.
    try:
        if phase_ == 'start': _S['gc_t'] = time.perf_counter(); _S['gc_c'] = _tcpu(); return
        cpu_ = _tcpu() - _S['gc_c']; dt_ = time.perf_counter() - _S['gc_t']; gen_ = info_.get('generation', -1)
        _M['gc_n'] += 1
        if cpu_ > _M['gc']: _M['gc'] = cpu_
        if gen_ == 2:
            _M['gc2_n'] += 1
            if cpu_ > _M['gc2']: _M['gc2'] = cpu_
        if cpu_ * 1000.0 < GC_MS: return
        try: rows_ = len(APP.rows); box_ = len(APP.preds[-1]['xyxy']) if APP.preds else 0   # the per-frame records the frame loop keeps (APP reads raise NameError before the loop has made them)
        except Exception: rows_ = box_ = -1
        try:
            with open('/proc/self/statm') as f_: rss_ = int(f_.read().split()[1]) * 4096 // 1048576
        except Exception: rss_ = -1
        # len(gc.get_objects()) would be the direct size of the heap but is itself tens of ms with the GIL held: the kept records and the memory stand in for it
        say('GC_SLOW', 'generation %d collection used %.0f ms of CPU in thread %s (%.0f ms on the clock; the difference is that thread waiting for the GIL or a CPU core around the collection, not collecting): '
            'collected %d, uncollectable %d, counts now %s, collections so far %s; kept frame records %d, boxes in the last one %d, memory %d MB'
            % (gen_, cpu_ * 1000.0, threading.current_thread().name, dt_ * 1000.0, info_.get('collected', -1), info_.get('uncollectable', -1), '/'.join(str(v_) for v_ in gc.get_count()),
               '/'.join(str(s_.get('collections')) for s_ in gc.get_stats()), rows_, box_, rss_))
    except Exception: pass


# ---------------------------------------------------------------- GIL_STALL: 파이썬 전체가 섰는가
def _thread_cpu(c0_, c1_, skip_):   # [(CPU seconds between the two looks, thread id, started since the thread list was read)], largest first
    out_ = [(v_ - c0_[i_], i_, False) if i_ in c0_ else (v_, i_, True) for i_, v_ in c1_.items()]   # not in the look before = new in the thread list, which is read every IDS_S: all the CPU it has used since it started
    for i_ in _task_ids(skip_):
        if i_ not in c1_:   # not in the list yet (every page request gets a thread of its own): the same
            try: out_.append((_cg(((~i_) << 3) | 6), i_, True))
            except Exception: pass
    out_.sort(reverse=True); return out_


def _thread_name(i_, nat_):   # a thread Python knows by its Python name; any other by what the system calls it
    if i_ in nat_: return nat_[i_]
    try:
        with open('/proc/self/task/%d/comm' % i_) as f_: return 'thread "%s" (id %d, not in the thread list of Python)' % (f_.read().strip(), i_)   # made by another library (NPU, camera), or a Python thread that is just ending
    except Exception: return 'thread id %d (ended)' % i_


def _gil_stall(late_, cpu_, span_=None, c0_=None, c1_=None, wait_=None, skip_=None, own_=0.0):   # the timer thread gives all of them; (late_, cpu_) alone = without the CPU of single threads
    _S['n_stall'] += 1; k_ = _S['n_stall']
    # The holder of the GIL is named by CPU: the thread that used about as much CPU as the stall was long, in the time between
    # the timer's two wakes, was busy in one call that kept the GIL. Threads of other libraries (NPU) use CPU without the GIL,
    # so a thread that is not in the thread list of Python is not the holder for that alone. A thread that ended before this
    # look (a page request gets a thread of its own and it may finish right after its long call) cannot be asked any more:
    # what the process used beyond the threads still here is put down to "threads that have ended". No thread with that much
    # CPU and nothing ended = the holder was not computing (blocked with the GIL), or the timer waited for a CPU core (that
    # time is in the line).
    # The stacks are taken AFTER the stall, when this thread finally ran again: they show where every thread is now. For the
    # holder that is often still the line of its long call (it is switched out right after the call returns) but not always.
    # A thread shown inside _gc_cb is not proof of a collection: that is where a thread waits for the GIL on its way into the
    # callback. Only a GC_SLOW line whose CPU is about the stall length is.
    ths_ = threading.enumerate(); nat_ = {t_.native_id: t_.name for t_ in ths_}; by_ = {}; top_ = ''
    if c0_ is not None and c1_ is not None:
        used_ = _thread_cpu(c0_, c1_, skip_) if c1_ else []
        idn_ = {t_.native_id: t_.ident for t_ in ths_}; by_ = {idn_[i_]: v_ for v_, i_, _ in used_ if i_ in idn_}
        top_ = ', '.join('%s %.0f ms%s' % (_thread_name(i_, nat_), v_ * 1000.0, ' (started in the last second: all its CPU so far)' if new_ else '') for v_, i_, new_ in used_[:CPU_TOP] if v_ >= 0.0005)
        gone_ = cpu_ - own_ - sum(v_ for v_, _, _ in used_)   # process CPU that no thread still here accounts for (a thread new in the list is taken off with all it ever used, so this is a lower bound)
        top_ = 'CPU by thread in the %.0f ms between its two wakes: %s%s; ' % ((span_ or 0.0) * 1000.0, top_ or ('no thread that is still here used 1 ms' if used_ else 'cannot be read on this system'),
                                                                             ', threads that have ended in that time at least %.0f ms' % (gone_ * 1000.0) if used_ and gone_ >= 0.005 else '')
    if not say('GIL_STALL', '#%d: the 10 ms timer woke %.0f ms late%s; process CPU in that time %.0f ms; %swhere each thread is now, after the stall, follows'
               % (k_, late_ * 1000.0, ' (%.1f ms of that it waited for a CPU core, the rest for the GIL)' % (wait_ * 1000.0) if wait_ is not None else '', cpu_ * 1000.0, top_)): return
    names_ = {t_.ident: t_.name for t_ in ths_}; me_ = threading.get_ident(); groups_ = {}
    deep_ = max(by_, key=by_.get) if by_ else None
    if deep_ is not None and by_[deep_] < TOP_S: deep_ = None
    for id_, f_ in sys._current_frames().items():
        if id_ == me_: continue
        p_ = []; d_ = STACK_DEPTH_TOP if id_ == deep_ else STACK_DEPTH
        while f_ is not None and len(p_) < d_: p_.append(_where(f_.f_code, f_.f_lineno)); f_ = f_.f_back
        v_ = by_.get(id_, 0.0)
        groups_.setdefault(' < '.join(p_), []).append((-v_, names_.get(id_, 'thread %d' % id_) + (' [%.0f ms CPU]' % (v_ * 1000.0) if v_ >= 0.0005 else '')))   # threads at the same place (page requests, waiters) share one entry
    items_ = []
    for s_, n_ in sorted(groups_.items(), key=lambda g_: min(x_[0] for x_ in g_[1])):   # the place of the thread that used the most CPU first (sorted keeps the rest in their order)
        n_ = [x_[1] for x_ in sorted(n_, key=lambda x_: x_[0])]
        items_.append('%s%s: %s' % (', '.join(n_[:3]), ' (+%d more)' % (len(n_) - 3) if len(n_) > 3 else '', s_))
    line_ = ''; n_out_ = 0
    for i_, it_ in enumerate(items_):
        if line_ and len(line_) + len(it_) > STACK_CHARS:
            camera_extra.note('server', 'GIL_STALL #%d threads: %s' % (k_, line_)); n_out_ += 1; line_ = ''
            if n_out_ >= STACK_LINES: camera_extra.note('server', 'GIL_STALL #%d: %d more places not written' % (k_, len(items_) - i_)); return
        line_ += (' | ' if line_ else '') + it_[:STACK_CHARS]
    if line_: camera_extra.note('server', 'GIL_STALL #%d threads: %s' % (k_, line_))


def _summary(own_=None):
    m_ = dict(_M); _M.update(_M0)
    camera_extra.note('server', 'STALL_DIAG 60 s: the 10 ms timer was at most %.0f ms late; garbage collections %d (full ones %d, the longest full one used %.0f ms of CPU, the longest of any %.0f ms of CPU); '
                      'camera frames read %d, the slowest %.0f ms, %d slow ones only waited for the camera; LIVE.cond held at most %.0f ms; lines not written (rate limit) %d%s'
                      % (m_['late'] * 1000.0, m_['gc_n'], m_['gc2_n'], m_['gc2'] * 1000.0, m_['gc'] * 1000.0, m_['rd_n'], m_['rd'] * 1000.0, m_['cam_waits'], m_['lock'] * 1000.0, m_['skipped'],
                         '; the timer thread of this log used %.0f ms of CPU' % (own_ * 1000.0) if own_ is not None else ''))


def _watch():
    """The timer thread. It needs the GIL to come back from every 10 ms sleep and to go on after every call below that lets
    other threads run (reading a /proc file does). Its lateness is the time between two wakes that was neither the 10 ms sleep
    nor CPU used by this thread itself: waiting for the GIL, or for a CPU core (that part is measured separately). So a stall
    that begins while this thread is in the middle of its own work is counted in full, at the next wake."""
    me_ = threading.get_native_id(); ss_ = _sched_open(); ids_ = _task_ids(me_)
    tw_ = ts_ = tr_ = time.perf_counter(); ow_ = osum_ = _tcpu(); pc_ = time.process_time(); c0_ = _cpu_of(ids_); w0_ = _sched(ss_)   # all as they were at the last wake
    while not _S['stop']:
        time.sleep(TICK_S)
        now_ = time.perf_counter(); o_ = _tcpu(); p_ = time.process_time(); late_ = now_ - tw_ - TICK_S - (o_ - ow_)
        _S['woke'] = now_; c1_ = c0_; w1_ = w0_
        try:
            c1_ = _cpu_of(ids_); w1_ = _sched(ss_)
            if late_ >= 0.005: _LATE.append((now_, late_))
            if late_ > _M['late']: _M['late'] = late_
            if late_ * 1000.0 >= GIL_MS: _gil_stall(late_, p_ - pc_, now_ - tw_, c0_, c1_, w1_[1] - w0_[1] if w0_ is not None and w1_ is not None else None, me_, o_ - ow_)
            if now_ - ts_ >= SUM_S: ts_ = now_; _summary(o_ - osum_); osum_ = o_
            if now_ - tr_ >= IDS_S:   # threads that are new in the list are looked at now, so that from the next wake on their CPU is counted like that of the others
                tr_ = now_; ids_ = _task_ids(me_); c1_.update(_cpu_of([i_ for i_ in ids_ if i_ not in c1_]))
        except Exception: pass
        tw_ = now_; ow_ = o_; pc_ = p_; c0_ = c1_; w0_ = w1_
    try:
        if ss_ is not None: os.close(ss_)
    except Exception: pass


# ---------------------------------------------------------------- SPAN_SLOW, LOOP_SLOW: 따로 잰 자리
def mark():   # start of a timed piece: None unless the log is on (span() then does nothing)
    return time.perf_counter() if ON else None


def span(t_, what_):
    if t_ is None: return
    dt_ = (time.perf_counter() - t_) * 1000.0
    if dt_ >= SPAN_MS: say('SPAN_SLOW', '%s took %.0f ms in thread %s' % (what_, dt_, threading.current_thread().name))


def loop_frame(row_, t5_):   # the frame loop calls this at the end of a frame (only when ON). t5_ = perf_counter_ns when the row's times were taken
    try:
        tail_ = (time.perf_counter_ns() - t5_) / 1e6; dec_ = (row_.get('det_total_ms') or 0.0) - (row_.get('pre_model_ms') or 0.0); work_ = (row_.get('frame_total_ms') or 0.0) - (row_.get('det_total_ms') or 0.0)
        if max(dec_, work_, tail_) < LOOP_MS: return   # waiting for the detector (pre_model_ms) or for a camera frame is not counted: it is not work of this thread
        say('LOOP_SLOW', 'frame loop, frame %s: from the detector result to boxes (with reading and feeding the next frame) %.0f ms, tracker / re-ID / relabel part %.0f ms, after that (keeping the records, page picture, trim, NPU status) %.0f ms; '
            'the row: %s; boxes %s, tracks %s' % (row_.get('image_id'), dec_, work_, tail_, ', '.join('%s %.0f' % (k_[:-3], v_) for k_, v_ in row_.items() if k_.endswith('_ms') and isinstance(v_, (int, float)) and v_ >= 5.0) or 'nothing over 5 ms',
                                                 row_.get('n'), row_.get('n_tracks')))
    except Exception: pass


# ---------------------------------------------------------------- 켜기
def start(live_=None):
    """Called once by app.py right after APP.LIVE.cond is made, before any thread has it. Off (the default): returns False and
    changes nothing. On: live_.cond becomes a Condition around a TimedLock, the gc callback is registered, the timer thread starts."""
    global ON
    if ON or not wanted(): return False
    ON = True; _S['stop'] = False; _M.update(_M0)
    if live_ is not None: live_.cond = threading.Condition(TimedLock('LIVE.cond'))
    gc.callbacks.append(_gc_cb)
    _S['woke'] = time.perf_counter(); n_ = len(_task_ids())
    _S['thread'] = threading.Thread(target=_watch, name='stall-diag', daemon=True); _S['thread'].start()
    camera_extra.note('server', 'STALL_DIAG on (LWIR_STALL_DIAG=1): READER_SLOW from %.0f ms (always from %.0f ms), GIL_STALL from %.0f ms, GC_SLOW from %.0f ms of CPU, LOCK_HELD from %.0f ms, SPAN_SLOW from %.0f ms, LOOP_SLOW from %.0f ms; '
                      'at most %d lines a second and %d a minute of each kind; garbage collector thresholds %s; CPU by thread %s, waiting for a CPU core %s'
                      % (READER_MS, READER_SURE_MS, GIL_MS, GC_MS, LOCK_MS, SPAN_MS, LOOP_MS, RATE_S, RATE_MIN, gc.get_threshold(),
                         'can be read (%d threads now)' % n_ if n_ else 'cannot be read on this system', 'can be read' if os.path.exists('/proc/thread-self/schedstat') else 'cannot be read on this system'))
    return True


def stop():   # for the tests: the server never turns it off (a TimedLock already handed out stays, it works without the rest)
    global ON
    _S['stop'] = True
    th_ = _S['thread']; _S['thread'] = None
    if th_ is not None: th_.join(2.0)
    while _gc_cb in gc.callbacks: gc.callbacks.remove(_gc_cb)
    ON = False; _RL.clear(); _LATE.clear(); _M.update(_M0); _S['n_stall'] = 0; _S['waits'] = 0
