"""stall_diag.py (LWIR_STALL_DIAG=1 일 때만 켜지는 멈춤 기록) 시험. 카메라, NPU, 네트워크 없이 돈다.

꺼짐: 스레드도 가비지 컬렉션 콜백도 없고 LIVE.cond 도 그대로이며 갈고리는 아무 일도 하지 않는다.
켜짐: 줄 종류마다 흉내 낸 멈춤(잠금을 잡고 있기, GIL 을 놓지 않는 긴 호출, 느린 컬렉션, 조각 사이가 긴 파이프)으로 그 줄이 나온다.
스레드별 CPU 와 코어를 기다린 시간은 리눅스에서만 잴 수 있다 (없는 곳에서는 그 부분만 보지 않는다).
"""
from collections import deque
import gc
import os
import re
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from . import camera_extra, camera_link, stall_diag

W, H = 4, 2            # 시험용 프레임: 16 바이트
PIECE = 4              # 파이프가 한 번에 주는 바이트 수 (프레임 한 장 = readinto 4 번)


_PER = []   # sum(range(n)) 이 수 하나에 쓰는 CPU 초 (처음 한 번, 짧게 세 번 재서 가장 빠른 값. 시계가 아니라 CPU 로 재서 PC 가 바빠도 같다)


def hold_gil(s_):   # C 안에서만 도는 호출 하나: s_ 초쯤 (PC 가 바쁘면 더 오래) GIL 을 놓지 않는다. 다른 파이썬 스레드는 그동안 돌지 못한다
    if not _PER:
        b_ = 1.0
        for _ in range(3): t0_ = time.thread_time(); sum(range(200000)); b_ = min(b_, time.thread_time() - t0_)
        _PER.append(max(b_, 1e-4) / 200000)
    sum(range(int(s_ / _PER[0])))


def burn(s_):   # 이 스레드가 CPU 를 s_ 초 쓸 때까지 (자는 것은 CPU 를 쓰지 않는다)
    t0_ = time.thread_time()
    while time.thread_time() - t0_ < s_: sum(range(2000))


class Pipe:   # 프레임을 조각으로 나눠 주는 가짜 파이프. slow[(프레임, 조각)] = 그 조각을 주기 전에 쉬는 초, hold[(프레임, 조각)] = 그 조각을 주기 전에 GIL 을 잡고 있는 초
    def __init__(self, frames_, slow_=None, hold_=None):
        self.data = b''.join(f_.tobytes() for f_ in frames_); self.pos = 0; self.slow = slow_ or {}; self.hold = hold_ or {}

    def readinto(self, mv_):
        if self.pos >= len(self.data): return 0
        f_, off_ = divmod(self.pos, W * H * 2)
        time.sleep(self.slow.get((f_, off_ // PIECE), 0))
        if (f_, off_ // PIECE) in self.hold: hold_gil(self.hold[(f_, off_ // PIECE)])
        k_ = min(PIECE - off_ % PIECE, len(mv_)); mv_[:k_] = self.data[self.pos:self.pos + k_]; self.pos += k_
        return k_


def frames(n_):
    return [np.arange(W * H, dtype=np.uint16).reshape(H, W) + 100 * i_ for i_ in range(n_)]


def new_live():
    return NS(q=deque(maxlen=100), cond=threading.Condition(), stopped=False, cam_lost=False, proc=None, q_drop=0, frames_in=0, last_frame_t=0.0,
              cam_fail_n=0, cam_unrec_n=0, cam_fps=0.0, rec=None)


class Base(unittest.TestCase):
    def setUp(self):
        self.lines = []
        p_ = patch.object(camera_extra, 'note', lambda who_, text_: self.lines.append((who_, text_)))
        p_.start(); self.addCleanup(p_.stop)
        self.addCleanup(stall_diag.stop)
        self.live = new_live()
        p_ = patch.dict(vars(APP), dict(LIVE=self.live, LIVE_W=W, LIVE_H=H)); p_.start(); self.addCleanup(p_.stop)
        p_ = patch.object(camera_extra, 'stamped', return_value=False); p_.start(); self.addCleanup(p_.stop)

    def on(self, live_=None):
        with patch.dict(os.environ, {'LWIR_STALL_DIAG': '1'}):
            self.assertTrue(stall_diag.start(live_))

    def kind(self, k_):
        return [t_ for _, t_ in self.lines if t_.startswith(k_ + ' ')]

    def read(self, fr_, slow_=None, hold_=None):   # 읽는 스레드가 하는 일을 이 스레드에서 끝까지 (파이프가 끝나면 돌아온다)
        p_ = NS(stdout=Pipe(fr_, slow_, hold_)); self.live.proc = p_
        camera_link.lv_reader(p_)
        return [r_ for _, r_ in self.live.q]


class OffTests(Base):
    def test_off_starts_nothing_and_hooks_do_nothing(self):
        cond_ = self.live.cond; n_th_ = threading.active_count(); n_cb_ = len(gc.callbacks)
        for v_ in (None, '', '0', 'true', '1 ', '2'):   # 정확히 "1" 만 켠다
            env_ = {} if v_ is None else {'LWIR_STALL_DIAG': v_}
            with patch.dict(os.environ, env_):
                if v_ is None: os.environ.pop('LWIR_STALL_DIAG', None)
                self.assertFalse(stall_diag.wanted(), v_)
                self.assertFalse(stall_diag.start(self.live), v_)
        self.assertFalse(stall_diag.ON)
        self.assertIs(self.live.cond, cond_)
        self.assertNotIsInstance(cond_._lock, stall_diag.TimedLock)
        self.assertEqual(threading.active_count(), n_th_)
        self.assertNotIn('stall-diag', [t_.name for t_ in threading.enumerate()])
        self.assertEqual(len(gc.callbacks), n_cb_)
        self.assertNotIn(stall_diag._gc_cb, gc.callbacks)
        self.assertIsNone(stall_diag.reader_probe())
        self.assertIsNone(stall_diag.mark())
        time.sleep(0.03); stall_diag.span(stall_diag.mark(), 'a slow piece')
        gc.collect()
        self.assertEqual(self.lines, [])

    def test_off_reader_takes_its_old_path(self):
        fr_ = frames(3)
        with patch.object(stall_diag.ReaderProbe, 'fill') as fill_:
            got_ = self.read(fr_, {(1, 2): 0.12})
        fill_.assert_not_called()
        self.assertEqual(len(got_), 3)
        for a_, b_ in zip(got_, fr_): self.assertTrue(np.array_equal(a_, b_))
        self.assertEqual((self.live.frames_in, self.live.q_drop, self.live.cam_lost), (3, 0, True))
        self.assertEqual(self.lines, [])


class OnTests(Base):
    def test_start_once(self):
        cond_ = self.live.cond
        self.on(self.live)
        self.assertTrue(stall_diag.ON)
        self.assertIsNot(self.live.cond, cond_)
        self.assertIsInstance(self.live.cond, threading.Condition)
        self.assertIsInstance(self.live.cond._lock, stall_diag.TimedLock)
        with patch.dict(os.environ, {'LWIR_STALL_DIAG': '1'}):
            self.assertFalse(stall_diag.start(self.live))   # 두 번째는 아무것도 더 만들지 않는다
        self.assertEqual([t_.name for t_ in threading.enumerate()].count('stall-diag'), 1)
        self.assertEqual(gc.callbacks.count(stall_diag._gc_cb), 1)
        self.assertEqual(len(self.kind('STALL_DIAG')), 1)
        self.assertIn('LWIR_STALL_DIAG=1', self.kind('STALL_DIAG')[0])
        self.assertIn('READER_SLOW from 100 ms (always from 145 ms)', self.kind('STALL_DIAG')[0]); self.assertIn('GC_SLOW from 30 ms of CPU', self.kind('STALL_DIAG')[0])
        stall_diag.stop()
        self.assertNotIn('stall-diag', [t_.name for t_ in threading.enumerate()])
        self.assertNotIn(stall_diag._gc_cb, gc.callbacks)

    def test_lines_go_to_the_camera_log_as_server_lines(self):
        self.on()
        with patch.object(camera_extra, 'note', _REAL_NOTE):   # 이 시험만 진짜 note 로: 줄이 카메라 기록의 목록에 '[server]' 줄로 들어가는지
            n0_ = len(camera_extra._Q)
            self.assertTrue(stall_diag.say('SPAN_SLOW', 'a test line'))
            row_ = camera_extra._Q.pop()
        self.assertEqual(len(camera_extra._Q), n0_)
        self.assertEqual(row_[2:], ('server', 'SPAN_SLOW a test line'))

    # ---- LOCK_HELD
    def test_lock_held_names_the_holder(self):
        self.on(self.live)
        with self.live.cond: pass
        with self.live.cond:
            with self.live.cond: time.sleep(0.005)   # 겹쳐 잡은 것은 한 번으로 센다
        self.assertEqual(self.kind('LOCK_HELD'), [])
        with self.live.cond:
            time.sleep(0.07)
        l_ = self.kind('LOCK_HELD')
        self.assertEqual(len(l_), 1)
        self.assertRegex(l_[0], r'LIVE\.cond was held for \d+ ms by thread MainThread: taken at camera/test_stall_diag\.py:\d+ test_lock_held_names_the_holder, let go at camera/test_stall_diag\.py:\d+ test_lock_held')
        self.assertGreaterEqual(int(re.search(r'held for (\d+) ms', l_[0]).group(1)), 65)

    def test_condition_still_waits_and_notifies_and_waiting_is_not_holding(self):
        self.on(self.live)
        c_ = self.live.cond; got_ = []

        def waiter():
            with c_:
                got_.append(c_.wait(timeout=5.0))
                got_.append(c_._is_owned())
        th_ = threading.Thread(target=waiter); th_.start()
        time.sleep(0.12)                       # 기다리는 동안 잠금은 풀려 있다: 여기서 바로 잡힌다
        t0_ = time.perf_counter()
        with c_: c_.notify_all()
        self.assertLess(time.perf_counter() - t0_, 0.05)
        th_.join(5.0)
        self.assertEqual(got_, [True, True])
        self.assertEqual(self.kind('LOCK_HELD'), [])   # 0.12 초 기다린 것은 "잡고 있던 시간"이 아니다
        with self.assertRaises(RuntimeError): c_.release()   # 잡지 않은 스레드가 놓으면 전과 같은 오류
        with self.assertRaises(RuntimeError): c_.notify()

    # ---- READER_SLOW
    def test_reader_slow_piece_in_the_middle_of_a_frame(self):
        self.on(self.live); fr_ = frames(4)
        got_ = self.read(fr_, {(1, 2): 0.13})
        self.assertEqual(len(got_), 4)
        for a_, b_ in zip(got_, fr_): self.assertTrue(np.array_equal(a_, b_))   # 켜도 받는 그림은 같다
        self.assertEqual((self.live.frames_in, self.live.cam_lost), (4, True))
        l_ = self.kind('READER_SLOW')
        self.assertEqual(len(l_), 1)
        m_ = re.search(r'took (\d+) ms from the first readinto call .*: 4 readinto calls, the first (\d+) ms .*, the longest (\d+) ms \(call 3\); after the last byte: copy [\d.]+ ms, waiting for LIVE\.cond ([\d.]+) ms, holding it ([\d.]+) ms', l_[0])
        self.assertIsNotNone(m_, l_[0])
        self.assertGreaterEqual(int(m_.group(1)), 125); self.assertGreaterEqual(int(m_.group(3)), 125)
        self.assertLess(int(m_.group(2)), 50); self.assertLess(float(m_.group(4)), 50.0)

    def test_reader_slow_first_read_alone_is_the_camera_unless_the_timer_was_late(self):
        self.on(self.live); slow_ = {(1, 0): 0.13, (2, 0): 0.03}   # 133 ms 까지는 cam_pipe 가 같은 그림을 보내지 않는 정상 간격 (셔터가 닫힌 동안도). 다음 프레임은 카메라처럼 30 ms 뒤
        with patch.object(stall_diag, '_late_since', return_value=0.0):   # 타이머는 제때 깼다 = 파이썬은 서지 않았다 = 카메라가 늦게 보냈다
            self.read(frames(5), slow_)
        self.assertEqual(self.kind('READER_SLOW'), [])
        self.assertEqual(stall_diag._M['cam_waits'], 1)
        self.live.q.clear()
        with patch.object(stall_diag, '_late_since', return_value=0.2):   # 타이머도 그동안 늦었다 = 읽고 나서 GIL 을 못 받았다
            self.read(frames(5), slow_)
        l_ = self.kind('READER_SLOW')
        self.assertEqual(len(l_), 1)
        self.assertIn('the 10 ms timer was up to 200 ms late meanwhile', l_[0])
        self.assertIn('1 slow frames before this one only waited for the camera', l_[0])

    def test_reader_slow_first_read_alone_is_written_when_the_reader_waited_for_a_cpu_core(self):
        self.on(self.live)
        real_ = stall_diag._sched; me_ = threading.get_ident(); n_ = [0]

        def sched(fd_):   # 읽는 스레드(여기서는 이 스레드)의 값만 바꾼다: 프레임 1 을 읽는 동안 코어를 110 ms 기다렸다 (schedstat 의 둘째 숫자). 타이머 스레드는 진짜 값
            if threading.get_ident() != me_: return real_(fd_)
            n_[0] += 1; return (0.0, 0.01 * n_[0] if n_[0] < 4 else 0.1 + 0.01 * n_[0])   # 1 번째 = 만들 때, 2~ = 프레임 0, 1, 2 ... 를 읽기 시작할 때
        with patch.object(stall_diag, '_late_since', return_value=0.0), patch.object(stall_diag, '_sched', sched):   # 타이머는 제때 깼다
            self.read(frames(5), {(1, 0): 0.13, (2, 0): 0.03})
        l_ = self.kind('READER_SLOW')
        self.assertEqual(len(l_), 1, self.lines)
        self.assertIn('the reader thread was on a CPU core 0.0 ms and waited 110.0 ms for one', l_[0])
        self.assertEqual(stall_diag._M['cam_waits'], 0)

    def test_reader_slow_long_wait_is_always_written(self):   # 검토 3: 145 ms 이상이면 첫 readinto 만 길어도, 타이머가 제때 깼어도 쓴다 (INPUT_GAP 과 짝을 맞출 수 있게)
        self.on(self.live)
        with patch.object(stall_diag, '_late_since', return_value=0.0):
            self.read(frames(5), {(1, 0): 0.17, (2, 0): 0.03})
        l_ = self.kind('READER_SLOW')
        self.assertEqual(len(l_), 1, self.lines)
        m_ = re.search(r': 4 readinto calls, the first (\d+) ms .*; the 10 ms timer was up to 0 ms late meanwhile; (the reader thread was on a CPU core [\d.]+ ms and waited [\d.]+ ms for one \(from this first readinto call to that of the next frame\); )?ended \d+ ms ago', l_[0])
        self.assertIsNotNone(m_, l_[0]); self.assertGreaterEqual(int(m_.group(1)), 165)
        if os.path.exists('/proc/thread-self/schedstat'): self.assertIsNotNone(m_.group(2), l_[0])
        self.assertEqual(stall_diag._M['cam_waits'], 0)

    def test_reader_slow_line_has_the_timer_lateness_of_the_same_stall(self):   # 검토 4: 줄을 한 프레임 뒤에 쓰므로 그 멈춤 때문에 타이머가 늦은 것이 줄에 들어 있다
        self.on(self.live)
        time.sleep(0.05)
        self.read(frames(5), {(2, 0): 0.03}, {(1, 2): 0.2})   # 프레임 1 의 셋째 조각 앞에서 0.2 초 동안 GIL 을 잡는다
        l_ = self.kind('READER_SLOW')
        self.assertEqual(len(l_), 1, self.lines)
        m_ = re.search(r'the longest (\d+) ms \(call 3\); .*; the 10 ms timer was up to (\d+) ms late meanwhile', l_[0])
        self.assertIsNotNone(m_, l_[0])
        self.assertGreaterEqual(int(m_.group(2)), 100)
        self.assertLessEqual(int(m_.group(2)), int(m_.group(1)) + 20)

    def test_timer_lateness_is_taken_only_from_the_time_of_the_frame(self):
        self.on()
        with patch.object(stall_diag, '_LATE', deque([(10.0, 0.2), (12.0, 0.05)])):   # 타이머가 9.79~10.0 초와 11.94~12.0 초에 자고 있었다 (각각 200, 50 ms 늦게 깸)
            self.assertEqual(stall_diag._late_since(9.0, 9.5), 0.0)     # 프레임이 끝난 뒤에 시작된 멈춤은 그 프레임의 것이 아니다
            self.assertEqual(stall_diag._late_since(9.0, 9.8), 0.2)
            self.assertEqual(stall_diag._late_since(9.9, 9.95), 0.2)
            self.assertEqual(stall_diag._late_since(10.1, 11.0), 0.0)
            self.assertEqual(stall_diag._late_since(10.1, 11.95), 0.05)
            self.assertEqual(stall_diag._late_since(9.0), 0.2)
        stall_diag.stop()
        self.assertIsNone(stall_diag._late_since(0.0))   # 타이머 스레드가 없다

    def test_reader_slow_last_frame_is_written_when_the_pipe_ends(self):
        self.on(self.live)
        self.read(frames(3), {(2, 2): 0.13})   # 마지막 프레임이 느렸고 그 뒤로 프레임이 없다: 파이프가 끝날 때 쓴다
        self.assertEqual(self.live.frames_in, 3)
        l_ = self.kind('READER_SLOW')
        self.assertEqual(len(l_), 1, self.lines)
        self.assertRegex(l_[0], r'the longest \d+ ms \(call 3\)')
        self.assertEqual(self.live.cond._lock.marks, {})   # 끝난 읽는 스레드의 것은 남지 않는다

    def test_reader_slow_waiting_for_the_lock(self):
        self.on(self.live)
        p_ = NS(stdout=Pipe(frames(3))); self.live.proc = p_
        th_ = threading.Thread(target=camera_link.lv_reader, args=(p_,), name='reader-test')
        with self.live.cond:   # 다른 쪽이 줄의 잠금을 잡고 있다: 읽는 스레드는 첫 프레임을 다 읽고 여기서 기다린다
            th_.start(); time.sleep(0.14)
        th_.join(5.0)
        self.assertEqual(self.live.frames_in, 3)
        l_ = self.kind('READER_SLOW')
        self.assertEqual(len(l_), 1)
        m_ = re.search(r'waiting for LIVE\.cond ([\d.]+) ms', l_[0])
        self.assertGreaterEqual(float(m_.group(1)), 100.0)
        h_ = self.kind('LOCK_HELD')
        self.assertEqual(len(h_), 1); self.assertIn('by thread MainThread', h_[0])

    def test_reader_without_a_timed_lock_still_reports(self):
        self.on()   # LIVE.cond 는 보통 것 그대로
        self.read(frames(4), {(1, 1): 0.13})
        l_ = self.kind('READER_SLOW')
        self.assertEqual(len(l_), 1)
        self.assertIn('the lock is not timed', l_[0])

    # ---- GIL_STALL
    def test_gil_stall_real_hold(self):
        self.on()
        time.sleep(0.05)
        for _ in range(3):
            hold_gil(0.35)
            time.sleep(0.1)
            if self.kind('GIL_STALL'): break
        l_ = self.kind('GIL_STALL')
        self.assertGreaterEqual(len(l_), 2, self.lines)
        m_ = re.match(r'GIL_STALL #1: the 10 ms timer woke (\d+) ms late( \(([\d.]+) ms of that it waited for a CPU core, the rest for the GIL\))?; process CPU in that time (\d+) ms; (CPU by thread in the (\d+) ms between its two wakes: ([^;]*); )?where each thread is now, after the stall, follows', l_[0])
        self.assertIsNotNone(m_, l_[0]); self.assertGreaterEqual(int(m_.group(1)), 80)
        self.assertRegex(l_[1], r'GIL_STALL #1 threads: .*MainThread( \[\d+ ms CPU\])?: camera/test_stall_diag\.py:\d+ (hold_gil < camera/test_stall_diag\.py:\d+ )?test_gil_stall_real_hold')
        self.assertNotIn('stall_diag.py:', l_[1].replace('test_stall_diag.py', ''))   # 타이머 스레드 자신은 적지 않는다
        if stall_diag._task_ids():   # 리눅스: GIL 을 잡고 있던 스레드가 쓴 CPU 로 이름이 나온다
            self.assertIsNotNone(m_.group(5), l_[0])
            t_ = re.match(r'MainThread (\d+) ms', m_.group(7))
            self.assertIsNotNone(t_, l_[0])   # 이름이 맨 먼저 = 가장 많이 썼다. 한가한 PC 에서는 멈춘 시간과 거의 같다 (코어가 모자라면 잡고 있던 스레드도 코어를 기다리므로 더 적다)
            self.assertGreaterEqual(int(t_.group(1)), 30); self.assertLessEqual(int(t_.group(1)), int(m_.group(6)) + 5)
            self.assertRegex(l_[1], r'GIL_STALL #1 threads: MainThread \[\d+ ms CPU\]: ')   # 가장 많이 쓴 스레드가 맨 앞
            self.assertNotIn('stall-diag', m_.group(7))

    def test_gil_stall_names_another_thread_by_its_cpu(self):
        if not stall_diag._task_ids(): return   # 스레드별 CPU 시계가 없는 곳 (리눅스가 아님): 볼 것이 없다
        self.on(); go_ = threading.Event()

        def work(): go_.wait(10.0); hold_gil(0.3); done_.wait(10.0)   # 긴 호출 뒤에도 살아 있다 (타이머가 볼 때 끝난 스레드의 CPU 는 물을 수 없다)
        done_ = threading.Event(); th_ = threading.Thread(target=work, name='holder'); th_.start()
        stall_diag._S['thread'].join(stall_diag.IDS_S + 0.3)   # 타이머가 스레드 목록을 다시 읽을 때까지 (이 스레드는 자면서 기다린다)
        go_.set(); t0_ = time.perf_counter()
        while len(self.kind('GIL_STALL')) < 2 and time.perf_counter() - t0_ < 8.0: time.sleep(0.05)   # 타이머는 멈춤이 끝난 뒤에야 줄을 쓴다
        done_.set(); th_.join(10.0)
        l_ = self.kind('GIL_STALL')
        self.assertGreaterEqual(len(l_), 2, self.lines)
        m_ = re.search(r'the 10 ms timer woke \d+ ms late.*CPU by thread in the (\d+) ms between its two wakes: holder (\d+) ms[,;]', l_[0])
        self.assertIsNotNone(m_, l_[0])   # 자고 있던 MainThread 가 아니라 GIL 을 잡고 있던 스레드의 이름이 맨 먼저 나온다
        self.assertGreaterEqual(int(m_.group(2)), 30); self.assertLessEqual(int(m_.group(2)), int(m_.group(1)) + 5)
        self.assertRegex(l_[1], r'GIL_STALL #1 threads: holder \[\d+ ms CPU\]: ')

    def test_gil_stall_counts_a_thread_that_started_after_the_last_look(self):
        if not stall_diag._task_ids(): return
        ev_ = threading.Event()
        c0_ = stall_diag._cpu_of(stall_diag._task_ids())
        th_ = threading.Thread(target=lambda: (burn(0.06), ev_.wait(10.0)), name='late-comer'); th_.start()   # 목록을 읽은 뒤에 생겨서 CPU 를 60 ms 쓴 스레드 (페이지 요청 하나)
        try:
            time.sleep(0.15)
            stall_diag._gil_stall(0.150, 0.140, 0.160, c0_, stall_diag._cpu_of(list(c0_)), 0.0005, None)
        finally:
            ev_.set(); th_.join(5.0)
        l_ = self.kind('GIL_STALL')
        self.assertRegex(l_[0], r'#1: the 10 ms timer woke 150 ms late \(0\.5 ms of that it waited for a CPU core, the rest for the GIL\); process CPU in that time 140 ms; '
                                r'CPU by thread in the 160 ms between its two wakes: late-comer \d+ ms \(started in the last second: all its CPU so far\)')
        self.assertRegex(' '.join(l_[1:]), r'late-comer \[\d+ ms CPU\]: ')

    def test_gil_stall_says_what_threads_that_have_ended_used(self):
        if not stall_diag._task_ids(): return
        ids_ = stall_diag._task_ids(); c0_ = stall_diag._cpu_of(ids_); p0_ = time.process_time(); o0_ = time.thread_time()
        th_ = threading.Thread(target=burn, args=(0.08,), name='request'); th_.start(); th_.join(5.0); time.sleep(0.1)   # 긴 일을 하고 곧 끝난 스레드 (페이지 요청 하나): 타이머가 볼 때는 이미 없다
        stall_diag._gil_stall(0.090, time.process_time() - p0_, 0.100, c0_, stall_diag._cpu_of(ids_), 0.0, None, time.thread_time() - o0_)
        m_ = re.search(r'CPU by thread in the 100 ms between its two wakes: ([^;]*), threads that have ended in that time at least (\d+) ms; where each thread', self.kind('GIL_STALL')[0])
        self.assertIsNotNone(m_, self.kind('GIL_STALL')[0])
        self.assertNotIn('request', m_.group(1)); self.assertGreaterEqual(int(m_.group(2)), 60); self.assertLessEqual(int(m_.group(2)), 120)

    def test_gil_stall_lists_threads_and_groups_the_same_place(self):
        ev_ = threading.Event(); ths_ = []

        def parked(): ev_.wait(10.0)
        for i_ in range(5): ths_.append(threading.Thread(target=parked, name='parked-%d' % i_)); ths_[-1].start()
        try:
            time.sleep(0.05)
            th_ = threading.Thread(target=stall_diag._gil_stall, args=(0.150, 0.140)); th_.start(); th_.join(5.0)   # 타이머 스레드처럼 다른 스레드에서 (스레드별 CPU 없이)
        finally:
            ev_.set()
            for t_ in ths_: t_.join(5.0)
        l_ = self.kind('GIL_STALL')
        self.assertEqual(l_[0], 'GIL_STALL #1: the 10 ms timer woke 150 ms late; process CPU in that time 140 ms; where each thread is now, after the stall, follows')
        body_ = ' '.join(l_[1:])
        self.assertRegex(body_, r'parked-\d, parked-\d, parked-\d \(\+2 more\): [^|]*parked')
        self.assertIn('MainThread: ', body_)
        self.assertEqual(body_.count('parked-'), 3)
        for x_ in l_: self.assertLess(len(x_), camera_extra.LINE_MAX)

    # ---- GC_SLOW
    def test_gc_slow_fake_collection(self):
        stall_diag._gc_cb('start', dict(generation=0, collected=0, uncollectable=0))
        stall_diag._gc_cb('stop', dict(generation=0, collected=1, uncollectable=0))
        self.assertEqual(self.kind('GC_SLOW'), [])
        with patch.dict(vars(APP), dict(rows=[1, 2, 3], preds=[dict(xyxy=[[0, 0, 1, 1], [1, 1, 2, 2]])])):
            stall_diag._gc_cb('start', dict(generation=2, collected=0, uncollectable=0))
            burn(0.045)
            stall_diag._gc_cb('stop', dict(generation=2, collected=7, uncollectable=1))
        l_ = self.kind('GC_SLOW')
        self.assertEqual(len(l_), 1)
        m_ = re.match(r'GC_SLOW generation 2 collection used (\d+) ms of CPU in thread MainThread \((\d+) ms on the clock; the difference is that thread waiting for the GIL or a CPU core around the collection, not collecting\): '
                      r'collected 7, uncollectable 1, counts now \d+/\d+/\d+, collections so far \d+/\d+/\d+; kept frame records 3, boxes in the last one 2, memory -?\d+ MB', l_[0])
        self.assertIsNotNone(m_, l_[0]); self.assertGreaterEqual(int(m_.group(1)), 40); self.assertGreaterEqual(int(m_.group(2)), int(m_.group(1)) - 1)
        self.assertEqual((stall_diag._M['gc_n'], stall_diag._M['gc2_n']), (2, 1))
        self.assertGreaterEqual(stall_diag._M['gc2'], 0.040)

    def test_gc_is_judged_by_cpu_not_by_the_clock(self):   # 검토 1: 컬렉션을 돌린 스레드가 콜백 앞뒤로 GIL 을 기다린 시간은 컬렉션이 아니다 (기다림은 CPU 를 쓰지 않는다)
        stall_diag._gc_cb('start', dict(generation=0, collected=0, uncollectable=0))
        time.sleep(0.12)   # 시계로는 120 ms, CPU 는 거의 0: 다른 스레드가 GIL 을 잡고 있던 것과 같은 모습
        stall_diag._gc_cb('stop', dict(generation=0, collected=0, uncollectable=0))
        stall_diag._gc_cb('start', dict(generation=2, collected=0, uncollectable=0))
        time.sleep(0.12)
        stall_diag._gc_cb('stop', dict(generation=2, collected=0, uncollectable=0))
        self.assertEqual(self.kind('GC_SLOW'), [])
        self.assertLess(stall_diag._M['gc'], 0.020); self.assertLess(stall_diag._M['gc2'], 0.020)   # 60 초 요약의 가장 긴 값도 부풀지 않는다
        self.assertEqual((stall_diag._M['gc_n'], stall_diag._M['gc2_n']), (2, 1))

    def test_gc_slow_real_collection_through_the_callback(self):
        self.on()
        with patch.object(stall_diag, 'GC_MS', 0.0):
            gc.collect()
        l_ = self.kind('GC_SLOW')
        self.assertTrue(any(x_.startswith('GC_SLOW generation 2 collection used') for x_ in l_), self.lines)

    # ---- SPAN_SLOW, LOOP_SLOW
    def test_span_slow(self):
        self.on()
        stall_diag.span(stall_diag.mark(), 'a quick piece')
        self.assertEqual(self.kind('SPAN_SLOW'), [])
        t_ = stall_diag.mark(); time.sleep(0.03); stall_diag.span(t_, 'trim of the kept frame records')
        self.assertRegex(self.kind('SPAN_SLOW')[0], r'SPAN_SLOW trim of the kept frame records took \d+ ms in thread MainThread')

    def test_loop_slow(self):
        self.on()
        stall_diag.loop_frame(dict(image_id='live_000000001.png', det_total_ms=320.0, pre_model_ms=318.0, frame_total_ms=330.0, tracker_ms=1.5, n=40, n_tracks=3), time.perf_counter_ns())
        self.assertEqual(self.kind('LOOP_SLOW'), [])   # 검출기를 318 ms 기다린 것은 이 스레드의 일이 아니다
        stall_diag.loop_frame(dict(image_id='live_000000002.png', det_total_ms=30.0, pre_model_ms=28.0, frame_total_ms=150.0, tracker_ms=118.4, relabel_ms=1.0, n=40, n_tracks=3), time.perf_counter_ns())
        l_ = self.kind('LOOP_SLOW')
        self.assertEqual(len(l_), 1)
        self.assertIn('frame live_000000002.png', l_[0]); self.assertIn('tracker / re-ID / relabel part 120 ms', l_[0]); self.assertIn('tracker 118', l_[0]); self.assertIn('boxes 40, tracks 3', l_[0])
        stall_diag.loop_frame(dict(image_id='x', det_total_ms=30.0, pre_model_ms=28.0, frame_total_ms=40.0), time.perf_counter_ns() - 90 * 1000000)
        self.assertEqual(len(self.kind('LOOP_SLOW')), 2)   # 줄을 만든 뒤(기록 쌓기, 화면 그림, 잘라내기)가 90 ms

    # ---- 줄 수 제한, 요약
    def test_rate_limit_per_second_and_per_minute(self):
        ok_ = [stall_diag._ok('K', 100.0 + 0.01 * i_) for i_ in range(12)]
        self.assertEqual(ok_, [True] * 5 + [False] * 7)                       # 1 초에 5 줄
        self.assertTrue(stall_diag._ok('OTHER', 100.05))                      # 종류마다 따로 센다
        self.assertTrue(stall_diag._ok('K', 101.0))
        n_ = 6 + sum(stall_diag._ok('K', 102.0 + 0.5 * i_) for i_ in range(80))   # 0.5 초에 한 줄씩 40 초 동안
        self.assertEqual(n_, stall_diag.RATE_MIN)                             # 1 분에 30 줄
        self.assertFalse(stall_diag._ok('K', 159.9))
        self.assertTrue(stall_diag._ok('K', 160.1))                           # 1 분이 지나면 다시 쓴다

    def test_rate_limit_says_how_many_lines_were_left_out(self):
        r_ = [stall_diag.say('GC_SLOW', 'line %d' % i_) for i_ in range(9)]
        self.assertEqual(r_, [True] * 5 + [False] * 4)
        self.assertEqual(len(self.kind('GC_SLOW')), 5)
        stall_diag._RL['GC_SLOW'][0] -= 1.5   # 1.5 초 뒤
        self.assertTrue(stall_diag.say('GC_SLOW', 'later'))
        self.assertEqual(self.kind('GC_SLOW')[-1], 'GC_SLOW later (4 GC_SLOW lines before this one were not written: rate limit)')
        self.assertTrue(stall_diag.say('GC_SLOW', 'after that'))
        self.assertEqual(self.kind('GC_SLOW')[-1], 'GC_SLOW after that')

    def test_flood_of_slow_holds_is_limited(self):
        self.on(self.live)
        with patch.object(stall_diag, 'LOCK_MS', 0.0):
            for _ in range(40):
                with self.live.cond: pass
        self.assertEqual(len(self.kind('LOCK_HELD')), stall_diag.RATE_S)

    def test_summary_line_and_reset(self):
        stall_diag._M.update(late=0.062, gc_n=40, gc2_n=2, gc2=0.021, gc=0.021, rd_n=1500, rd=0.071, cam_waits=7, lock=0.004, skipped=3)
        stall_diag._summary()
        self.assertEqual(self.kind('STALL_DIAG'), ['STALL_DIAG 60 s: the 10 ms timer was at most 62 ms late; garbage collections 40 (full ones 2, the longest full one used 21 ms of CPU, the longest of any 21 ms of CPU); '
                                                   'camera frames read 1500, the slowest 71 ms, 7 slow ones only waited for the camera; LIVE.cond held at most 4 ms; lines not written (rate limit) 3'])
        self.assertEqual(stall_diag._M, stall_diag._M0)
        stall_diag._summary(0.150)   # 타이머 스레드가 부르면 자기가 쓴 CPU 가 붙는다
        self.assertTrue(self.kind('STALL_DIAG')[-1].endswith('lines not written (rate limit) 0; the timer thread of this log used 150 ms of CPU'), self.kind('STALL_DIAG')[-1])


_REAL_NOTE = camera_extra.note   # 이 파일을 읽을 때의 진짜 note (시험마다 가짜로 바꿔 끼운다)


if __name__ == '__main__':
    unittest.main()
