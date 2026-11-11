"""도착하는 대로 검출기에 넣기 (2026-11-11). 기본으로 꺼져 있다. 환경 변수 LWIR_NPU_FEED 가 정확히 arrival 일 때만 켜진다.

왜: 보통 방식에서는 프레임 반복이 그 자리에 왔을 때 프레임을 검출기(NPU)에 넣는다. CPU 가 앞 프레임 일로 바쁜 동안 도착한 프레임은
검출이 시작되기도 전에 기다린다. 이 방식에서는 작은 스레드(넣는 스레드) 하나가 프레임이 도착하자마자 검출기에 넣고, 프레임 반복은 끝난 검출을 도착 순서대로 받아 간다.

어떻게
  - 검출기 엔진을 하나 더 만든다: 같은 모델 파일, NPU 코어 1 (보통 방식의 엔진은 코어 0). 이 방식에서만 만든다.
  - 넣는 스레드: 카메라 읽기 스레드가 줄(APP.LIVE.q)에 넣은 프레임을 꺼내 16 비트 -> 8 비트, 흔들림 보정, 검출기 입력 만들기까지 하고(보통 방식의 live_next 뒷부분과
    submit_tiles 를 그대로 부른다) 비어 있는 엔진에 넣는다. 바로 앞 프레임이 쓴 엔진이 아닌 쪽을 먼저 고른다 (코어 0, 1 을 번갈아).
    엔진 하나에는 한 번에 한 장만 들어간다. "비었다"는 프레임 반복이 그 엔진의 결과를 받아 간 뒤다.
  - 프레임 반복: read_frame() 이 넣은 순서대로 한 장을 받아 오고, APP.a.wait() 가 그 장의 엔진에서 결과를 기다린다. 순서는 넣은 순서 그대로다.
    프레임 반복은 "다음 프레임 미리 읽기"가 없는 갈래(--pipeline 이 없을 때의 갈래)를 탄다. 미리 넣기는 넣는 스레드가 하기 때문이다. 그래서 start() 가 APP.args.pipeline 을 끈다.
  - 한도: 프레임 반복이 지금 다루는 프레임 말고, 검출기에 넣어 두고 아직 안 받아 간 프레임은 1 장까지(AHEAD), 그 뒤에서 아직 못 넣고 기다리는 프레임도 1 장까지.
    그래서 엔진 둘이 같이 도는 것은 "프레임 반복이 앞 프레임의 결과를 기다리는 중에 다음 프레임이 도착한 때"다.
    프레임 반복이 느려서 넣어 둔 프레임을 아직 안 받아 갔는데 프레임이 2 장 넘게 기다리게 되면, 기다리는 것 가운데 오래된 것부터 버리고 APP.LIVE.q_drop 에 센다
    (보통 방식에서 줄이 넘칠 때 세는 그 숫자, 같은 뜻). 넣어 둔 프레임을 더 쌓지 않는 까닭: 프레임 반복이 못 따라갈 때 쌓아 두면 낡은 프레임만 처리하게 된다.
  - 카메라 움직임 계산(--gmc-thread 의 작업 스레드)도 프레임이 도착했을 때 맡긴다 (보통 방식은 다음 프레임을 미리 읽을 때 맡긴다). 프레임 순서는 같다.
  - 검출 모델 바꾸기: 프레임 반복이 바꾸기 직전에 pause() (넣기를 멈추고, 옛 모델에 들어가 있던 프레임의 결과는 받아서 버리고, 둘째 엔진을 놓는다), 바꾼 뒤에 resume()
    (새 모델로 둘째 엔진을 다시 만든다. 못 만들면 코어 0 하나로 계속 도착하는 대로 넣는다). 옛 모델에 들어가 있던 프레임은 보통 방식처럼 새 모델에 다시 넣는다
    (버리지 않는다). 되돌리기(코드 4)도 같은 길이다.
  - Python 의 스레드 교대 간격을 5 ms 에서 1 ms 로 줄인다 (이 방식을 켰을 때만). 프레임 반복이 Python 계산을 하는 동안에도 카메라 읽기 스레드와 넣는 스레드가
    빨리 차례를 얻게 하려는 것이다. LWIR_NPU_FEED_SWITCH_MS=0 이면 건드리지 않는다.
  - 넣는 스레드에서 오류가 나면 이 방식을 끄고 그 자리에서 보통 읽기(live_next)로 넘어간다 (다음 프레임 미리 넣기는 없는 채로). 한 줄을 찍는다.

켤 수 없는 경우 (시작할 때 한 줄을 찍고 보통 방식 그대로 간다)
  실시간 실행이 아님, --pipeline 없음, --npu-depth 가 1 이 아님, --no-vit 없음(또는 --vit-budget / --vit-bench-n), --zoom, 검출 모델이 두 개짜리(타일),
  --mock, 둘째 엔진을 못 만듦(파일·NPU 메모리 등).

보통 방식과 달라지는 숫자의 뜻
  - 기록의 pre_model_ms(화면의 "NPU 대기"): 검출기에 넣은 때부터 프레임 반복이 결과를 받은 때까지. 결과가 먼저 나와 기다린 시간도 들어간다.
  - frame_total_ms(화면의 "처리 시간"): 검출기에 넣은 때(= 도착 직후)부터 그 프레임 일이 끝난 때까지.
  - 측정 기록·RESULT.json 의 pipeline 은 False 로 적힌다 (프레임 반복이 탄 갈래).
"""
import collections
import gc
import os
import sys
import threading
import time

import numpy as np

from ..common.state import APP
from ..track.stabilizer import STAB, stab_live, stab_off
from ..track import camera_motion
from ..camera import camera_extra
from . import detect_frame

SECOND = '#core1'   # 둘째 엔진의 이름 꼬리 (APP.pending 의 열쇠가 첫째 엔진과 겹치지 않게)
AHEAD = 1           # 검출기에 넣어 두고 프레임 반복이 아직 안 받아 간 프레임 수의 한도


def wanted():
    return os.environ.get('LWIR_NPU_FEED', '') == 'arrival'


def refuse_reason():
    a_ = APP.args
    if APP.LIVE is None: return 'it is for live runs only (--live-port)'
    if a_.mock: return 'not with --mock'
    if not a_.pipeline: return 'it needs --pipeline'
    if a_.npu_depth != 1: return 'it needs --npu-depth 1'
    if not a_.no_vit or a_.vit_budget or a_.vit_bench_n: return 'it needs --no-vit (no --vit-budget / --vit-bench-n)'
    if a_.zoom: return 'not with --zoom'
    if len(APP.det_names) != 1 or APP.b is not None: return 'the detector %s uses two models (cores 0 and 1 are both taken)' % a_.det
    return ''


def make_second():
    """같은 검출 모델을 NPU 코어 1 에 한 번 더 올리고 빈 그림으로 한 번 돌려 본다. 안 되면 오류."""
    e_ = APP.Engine(APP.det_names[0] + SECOND, 1, 1, model=APP.det_names[0])
    try:
        x_ = np.zeros(e_.in_shape, np.uint8)
        APP.pending[e_.name] = e_.b.run_async(x_, user_arg={'f': -1}, output_buffer=e_.packed); e_.wait()
    except BaseException:
        APP.pending.pop(e_.name, None); e_ = None; gc.collect(); raise
    return e_


def start():
    """app.py 가 프레임 반복 전에 한 번 부른다. 꺼져 있거나 켤 수 없으면 None (보통 방식)."""
    v_ = os.environ.get('LWIR_NPU_FEED', '')
    if v_ in ('', 'normal'): return None
    if v_ != 'arrival':
        camera_extra.say('FEED: LWIR_NPU_FEED=%s is not a known value (only "arrival"). The normal path is used.' % v_[:40]); return None
    why_ = refuse_reason(); second_ = None
    if not why_:
        try: second_ = make_second()
        except Exception as e_: why_ = 'the second detector engine on NPU core 1 could not be made (%s)' % repr(e_)[:160]
    if why_:
        camera_extra.say('FEED arrival refused: %s. The normal path is used.' % why_); return None
    f_ = Feed(second_)
    sw_ = _switch_interval()
    camera_extra.say('FEED arrival on: frames go to the detector as they arrive, model %s on NPU cores 0 and 1%s' % (APP.det_names[0], ', thread switch interval %g ms' % (sw_ * 1000) if sw_ else ''))
    return f_


def _switch_interval():
    """프레임 반복이 Python 계산으로 바쁜 동안 카메라 읽기 스레드와 넣는 스레드가 차례를 얻기까지 기다리는 시간을 줄인다 (Python 의 스레드 교대 간격, 기본 5 ms -> 1 ms).
    이 방식을 켰을 때만 바꾼다. LWIR_NPU_FEED_SWITCH_MS=<ms> 로 값을 바꾸고, 0 이면 건드리지 않는다. -> 바꾼 값(초) 또는 0"""
    try:
        ms_ = float(os.environ.get('LWIR_NPU_FEED_SWITCH_MS', '1') or 1)
        if not 0.05 <= ms_ < sys.getswitchinterval() * 1000: return 0
        sys.setswitchinterval(ms_ / 1000.0); return ms_ / 1000.0
    except Exception: return 0


class _Rec:
    __slots__ = ('seq', 't', 'raw', 'g8', 'gg', 'stab', 'e', 'job', 't_sub', 'gmc')


class Feed:
    def __init__(self, second_):
        self.eng = [APP.a, second_]; self.busy = [False, False]; self.last_e = 1
        self.out = collections.deque()   # 검출기에 넣은 프레임 (도착 순서). 프레임 반복이 아직 받아 가지 않은 것
        self.redo = collections.deque()  # 모델을 바꿀 때 옛 모델에 들어가 있던 프레임: 새 모델에 다시 넣을 것 (도착 순서)
        self.cur = None; self.paused = False; self.working = False; self.quit = False; self.direct = False; self.err = None; self.closed = False
        self.seq = 0; self.tgt = 0; self._job = None; self.prev_seq = -1
        self.st = dict(submitted=0, taken=0, core=[0, 0], overlap=0, drop_wait=0, redo=0, order_bad=0, max_ahead=0, swaps=0, one_core=0)
        APP.args.pipeline = False   # 프레임 반복은 "읽기 -> 넣기 -> 기다리기" 갈래를 탄다. 읽기는 read_frame(넣어 둔 프레임 받아 오기), 넣기는 submit_time(넣은 시각만 돌려줌)
        APP.a = self
        self.th = threading.Thread(target=self._run, daemon=True, name='npu-feed'); self.th.start()

    def __getattr__(self, name_):   # APP.a 를 엔진으로 보는 코드(예: APP.a.names)는 첫째 엔진의 값을 받는다
        if name_.startswith('__') or name_ in ('eng', 'direct'): raise AttributeError(name_)
        if not self.eng: raise AttributeError(name_)
        return getattr(self.eng[0], name_)

    # ---------------------------------------------------------------- 넣는 스레드
    def _free(self):
        n_ = len(self.eng)
        for k_ in range(1, n_ + 1):
            e_ = (self.last_e + k_) % n_
            if not self.busy[e_]: return e_
        return None

    def _run(self):
        L_ = APP.LIVE; e_ = None
        try:
            while True:
                with L_.cond:
                    while True:
                        if self.quit or L_.stopped: return
                        if not self.paused:
                            e_ = self._free() if len(self.out) < AHEAD else None
                            if e_ is None:
                                while len(L_.q) > 1: L_.q.popleft(); L_.q_drop += 1; self.st['drop_wait'] += 1   # 지금은 넣을 수 없다: 기다리는 것은 가장 새 프레임 1 장만
                            elif self.redo or L_.q: break
                        L_.cond.wait(timeout=0.5)
                    r_ = self.redo.popleft() if self.redo else None
                    if r_ is None: t_, raw_ = L_.q.popleft()
                    self.busy[e_] = True; self.working = True
                    if sum(self.busy) > 1: self.st['overlap'] += 1
                self._job = None
                if r_ is None:
                    g8_ = APP.lv_to8(raw_); gg_ = np.empty((512, 640), np.uint8); gg_[16:496] = g8_; gg_[:16] = g8_[0]; gg_[496:] = g8_[-1]   # live_next 와 같은 순서, 같은 계산
                    sc_ = stab_live(g8_, t_, gg_) if STAB['on'] else stab_off()
                    r_ = _Rec(); r_.seq = self.seq; self.seq += 1; r_.t = t_; r_.raw = raw_; r_.g8 = g8_; r_.gg = gg_; r_.stab = sc_
                    r_.gmc = self._gmc(gg_) if APP.args.gmc_thread else None
                self.tgt = e_; r_.t_sub = detect_frame.submit_tiles(r_.gg, r_.seq)   # APP.a 가 이 객체라서 submit() 으로 온다
                r_.e = e_; r_.job = self._job
                with L_.cond:
                    self.out.append(r_); self.last_e = e_; self.working = False
                    self.st['submitted'] += 1; self.st['core'][e_] += 1; self.st['max_ahead'] = max(self.st['max_ahead'], len(self.out))
                    e_ = None; L_.cond.notify_all()
        except BaseException as x_:
            with L_.cond:
                if e_ is not None and self._job is None and e_ < len(self.busy): self.busy[e_] = False
                self.err = repr(x_)[:200]; self.working = False; L_.cond.notify_all()

    def _gmc(self, gg_):   # camera_motion.gmc_submit 과 같은 일 (첫 프레임에서 계산 객체를 만들고, 작업 스레드에 순서대로 맡긴다). 프레임 번호는 받아 갈 때 붙는다
        if not APP.gmc_objs:
            o_ = camera_motion.GMC(APP.args.gmc_ds); o_.fp = dict(o_.fp, maxCorners=APP.args.gmc_corners, minDistance=APP.args.gmc_mindist); APP.gmc_objs.append(o_)
        return APP.gmc_pool.submit(APP.gmc_job, APP.gmc_objs[-1], gg_)

    def submit(self, x_, f_):   # detect_frame.submit_tiles 가 APP.a.submit 으로 부른다
        if self.direct: return self.eng[0].submit(x_, f_)
        e_ = self.eng[self.tgt]
        self._job = e_.b.run_async(x_, user_arg={'f': f_}, output_buffer=e_.packed)

    # ---------------------------------------------------------------- 프레임 반복 (app.py) 이 부르는 것
    def read_frame(self, k_):
        g_ = APP.live_next() if self.direct else self._next()
        if APP.args.gmc_thread:
            r_ = self.cur
            if r_ is not None and r_.gmc is not None: APP.gmc_futs[k_] = r_.gmc   # 도착했을 때 이미 맡겨 둔 계산
            else: camera_motion.gmc_submit(k_, g_)
        return g_, 0.0

    def submit_time(self, g_, k_):
        if self.direct or self.cur is None: return detect_frame.submit_tiles(g_, k_)
        return self.cur.t_sub

    def wait(self):
        if self.direct or self.cur is None: return self.eng[0].wait()
        r_ = self.cur; self.cur = None
        return self._collect(r_, True)

    def _collect(self, r_, keep_):
        e_ = self.eng[r_.e]
        try:
            APP.pending[e_.name] = r_.job; o_ = e_.wait()
            if keep_: o_ = [np.array(v_, copy=True) if np.may_share_memory(v_, e_.packed) else v_ for v_ in o_]   # 같은 엔진에 다음 프레임이 곧 들어간다
        finally:
            with APP.LIVE.cond: self.busy[r_.e] = False; APP.LIVE.cond.notify_all()
        return o_

    def _next(self):   # camera_link.live_next 와 같은 일을 한다. 다른 점: 줄 대신 "검출기에 넣어 둔 프레임"에서 꺼내고, 8 비트 바꾸기·흔들림 보정은 넣는 스레드가 이미 했다
        L_ = APP.LIVE
        while True:
            with L_.cond:
                while not self.out:
                    if self.err is not None or L_.stopped: break
                    if L_.cam_lost and APP.args.live_wait and not L_.q and not self.working: break
                    wait_s_ = (APP.args.live_wait or 10) if L_.last is None else 10
                    if L_.cond.wait(timeout=wait_s_): continue
                    if APP.args.live_wait: raise RuntimeError('no camera frame for %d s' % wait_s_)
                    break
                r_ = self.out.popleft() if self.out else None
            if r_ is not None:
                if L_.prev_t is not None and r_.t - L_.prev_t > 1.0:
                    L_.gap_n += 1
                    if APP.GB is not None: APP.gb_send('H')
                    camera_extra.say('LIVE input gap of %.1f s: selection dropped, tracks restart' % (r_.t - L_.prev_t))
                L_.prev_t = r_.t; break
            if self.err is not None:
                self._off('the feeding thread stopped on an error (%s)' % self.err); return APP.live_next()
            if L_.stopped or APP.args.live_wait:
                if L_.last is None: raise RuntimeError('camera delivered no frame')
                self._off(''); return L_.last
            with L_.cam_lock: APP.lv_cam_wait()
        if r_.seq <= self.prev_seq: self.st['order_bad'] += 1
        self.prev_seq = r_.seq
        APP.FRAME_DY[0] = 16.0
        L_.stabq.append(r_.stab); L_.arr.append(r_.t); L_.last = r_.gg
        if APP.GB is not None and APP.GB.gui: L_.raws.append(r_.raw if APP.args.gui_map == 'team' else (r_.g8.astype(np.uint16) * 257))
        L_.rawq.append(r_.raw)
        self.cur = r_; self.st['taken'] += 1
        return r_.gg

    # ---------------------------------------------------------------- 멈추기, 모델 바꾸기, 끝내기
    def _stop_thread(self):
        with APP.LIVE.cond: self.quit = True; APP.LIVE.cond.notify_all()
        self.th.join(timeout=10)

    def _drain(self):   # 검출기에 들어가 있는 프레임의 결과를 받아서 버린다 -> 그 프레임들 (도착 순서)
        rs_ = ([self.cur] if self.cur is not None else []) + list(self.out); self.out.clear(); self.cur = None
        for r_ in rs_:
            try: self._collect(r_, False)
            except Exception: pass
        return rs_

    def _off(self, why_):   # 이 방식을 끈다: 여기서부터 read_frame 은 보통 읽기(live_next), APP.a 는 첫째 엔진
        self._stop_thread(); self._drain()
        APP.a = self.eng[0]; second_ = self.eng[1:]; self.eng = self.eng[:1]; self.direct = True
        for e_ in second_:
            try: e_.b.dispose()
            except Exception: pass
        second_ = None; gc.collect()
        if why_: camera_extra.say('FEED arrival stopped: %s. From here frames are read the normal way, one at a time (no read-ahead).' % why_)

    def pause(self):   # 검출 모델을 바꾸기 직전 (프레임 반복, 두 프레임 사이)
        if self.direct: return
        L_ = APP.LIVE
        with L_.cond:
            self.paused = True
            while self.working and self.th.is_alive(): L_.cond.wait(0.2)
        rs_ = self._drain(); self.st['swaps'] += 1; self.st['redo'] += len(rs_)
        self.redo.extendleft(reversed(rs_))   # 옛 모델에 들어가 있던 프레임: 바꾼 뒤 새 모델에 다시 넣는다 (보통 방식과 같다)
        APP.a = self.eng[0]; self.eng = []; self.busy = []   # 엔진을 놓는다: 바꾸는 코드(do_model_swap)가 전과 똑같이 옛 엔진을 비우고 새 엔진을 만든다

    def resume(self, code_):   # 바꾼 뒤. code_ = do_model_swap 의 답 (5 = 되돌리기도 실패: 프레임 반복이 곧 오류로 끝난다)
        if self.direct or code_ == 5: return
        prim_ = APP.a; second_ = None
        try: second_ = make_second()
        except Exception as e_:
            self.st['one_core'] += 1
            camera_extra.say('FEED arrival: the second detector engine on NPU core 1 could not be made for %s (%s). Frames are still fed as they arrive, on core 0 only.' % (APP.det_names[0], repr(e_)[:160]))
        self.eng = [prim_] + ([second_] if second_ is not None else []); self.busy = [False] * len(self.eng); self.last_e = len(self.eng) - 1
        APP.a = self
        with APP.LIVE.cond: self.paused = False; APP.LIVE.cond.notify_all()

    def close(self):   # 프레임 반복이 끝난 뒤. 둘째 엔진을 닫는다 (첫째는 app.py 가 전처럼 닫는다)
        if self.closed: return
        self.closed = True
        if not self.direct:
            self._stop_thread(); self._drain()
            if self.eng: APP.a = self.eng[0]
            for e_ in self.eng[1:]:
                try: e_.b.dispose()
                except Exception: pass
            self.eng = self.eng[:1]; self.direct = True
        s_ = self.st
        camera_extra.say('FEED arrival end: fed %d, taken %d, core 0 %d, core 1 %d, started while the other core was still busy %d, dropped while waiting to be fed %d, '
                         'fed again after a model change %d (changes %d), most frames ahead of the loop %d, order errors %d'
                         % (s_['submitted'], s_['taken'], s_['core'][0], s_['core'][1], s_['overlap'], s_['drop_wait'], s_['redo'], s_['swaps'], s_['max_ahead'], s_['order_bad']))
