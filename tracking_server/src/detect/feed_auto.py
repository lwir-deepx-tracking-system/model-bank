"""카메라 주기 지키기: 주기 감시와 검출 투입 자동 선택 (2026-11-16). 장치에서 아직 안 돌려 봤다.

두 가지를 한다
  (1) 주기 감시 (넣는 스레드가 있으면 늘 잰다: LWIR_NPU_FEED=arrival / auto, 또는 LWIR_NPU_HOT_SWITCH=1). /rules 와 /healthz 의 npu_feed.auto.
      카메라 주기를 지키는지 말로 적는다(period.state, period.text): 지킴 / 주기는 지키지만 결과가 늦음 / 결과 간격이 고르지 않음 / 프레임을 버리는 중 /
      이 검출 모델로는 지킬 수 없음. 모델은 스스로 바꾸지 않는다. 알리기만 한다(화면 위 연결 표시와 고급의 검출 NPU 줄, 서버 기록 한 줄).
  (2) 검출 투입 자동 선택 (LWIR_NPU_FEED=auto 또는 화면의 '자동'. 기본은 꺼짐. 장치에서 권하는 설정은 지금처럼 arrival: LATENCY.md 맨 위 절).
      평소에는 검출기를 NPU 코어 0 하나로 돌리고, 검출 한 장이 카메라 주기를 못 따라가면 (예: NPU 가 뜨거워 클럭을 1000 -> 800 -> 600 MHz 로 낮춤,
      또는 검출이 33 ms 를 넘는 모델) 스스로 코어 0·1 교대로 바꾸고, 다시 빨라지면 코어 0 하나로 돌아온다.
      바꾸는 길은 손으로 바꿀 때와 같다 (frame_feed.Feed: 새로 넣기를 멈춤 -> 넣어 둔 프레임의 결과를 받음 -> 프레임 사이에서 바꿈).
      이것은 "초당 프레임 수"를 지키는 보험이다. 지연을 줄이지 않는다: 느려진 NPU 에서는 교대로 바꿔도 한 장 한 장이 그만큼 늦다. 고치는 길은 식히는 것이다.
      두 모델 상주(resident_det)가 켜져 있고 상주할 수 있으면 교대로 올라가 그대로 머문다(상주는 교대가 필요하다).

재는 것 (최근 WINDOW_S 초)
  - 검출 시간: 검출기에 넣은 때부터 프레임 반복이 그 결과를 받은 때까지 (화면의 "NPU 대기"와 같은 숫자), p50 / p90.
    프레임 반복이 결과를 EXACT_MS 넘게 기다린 프레임만 판단에 쓴다. 결과가 먼저 나와 있던 프레임의 숫자에는 프레임 반복이 늦은 시간이 섞여 있기 때문이다.
  - 카메라 주기: 넣는 스레드가 본 프레임(넣은 것, 버린 것)의 도착 간격 가운뎃값.
  - 결과가 나간 때: 그 프레임의 사람 상자를 짐벌에 보낸 때 (page_handler.live_publish 의 live_tracks.publish 바로 뒤, sent()).
    짐벌에 보내지 않은 프레임은 프레임 반복이 다음 프레임을 받으러 온 때(close()).
  - 늦음: 결과가 "도착 + 카메라 주기 한 번"보다 늦게 나간 프레임의 비율(period.late_pct). 지연의 잣대이고, 주기를 지키는지와는 따로 본다.
  - 결과 간격: 짐벌로 나간 결과 사이의 간격. 카메라 주기의 GAP(1.5)배를 넘는 간격의 비율(period.gap_pct)과 가장 긴 간격(period.gap_max_ms).
    프레임을 버리면 간격이 두 배가 되고, 두 코어의 결과가 짝지어 나오면(코어 2 일이 코어 1 검출을 기다릴 때 등) 짧은 간격 뒤에 긴 간격이 온다.
  - 버림: 넣지 못하고 버린 프레임(period.dropped_per_s). 코어가 모두 차 있고 프레임 반복이 그 가운데 하나를 기다리는 동안 버린 것(busy)과
    코어는 비었는데 프레임 반복이 앞 프레임을 아직 안 받아 가서 버린 것(프레임 반복이 늦음)을 따로 센다(frame_feed.Feed._core_bound).
    시작부터 센 수는 misses (frames, late, gaps, dropped, 그 가운데 코어가 차 있어서 버린 것 dropped_busy).
  - NPU 클럭(MHz)·온도(C): app.py 가 50 프레임마다 이미 묻는 DeviceStatus 에서 같이 읽는다(npu). 읽지 못하면 한 줄을 찍고 뺀다.

주기 상태 (period.state, 앞의 것이 먼저)
  - over: 검출 시간 p90 > 쓸 수 있는 코어 수 x UP x 주기 (두 코어: 2 x 0.90 x 33.3 = 60 ms), 또는 쓸 수 있는 코어를 다 쓰는데도 코어가 차서 버림,
    이 OVER_SAY_S(3초) 이어지면 "이 검출 모델로는 카메라 주기를 지킬 수 없음"이 되고 서버 기록에 한 줄.
    풀리는 것은 p90 < 코어 수 x OVER_DOWN(0.80) x 주기(53.3 ms)이고 코어가 차서 버린 프레임이 없는 상태가 OVER_END_S(10초) 이어질 때다(한 줄).
    그 사이(53.3~60 ms)에서는 있던 상태 그대로다: 클럭이 600 <-> 800 MHz 를 오갈 때 켜졌다 꺼졌다 하지 않게. 검출 모델이나 코어를 손으로 바꾸면 다시 잰다.
  - drop: 창 안에서 버린 프레임이 있음.  - uneven: 버리지는 않지만 주기의 1.5배를 넘는 결과 간격이 10% 를 넘음.
  - late: 주기는 지킴(버림 없음, 간격 고름) · 결과가 도착 + 주기 한 번보다 늦게 나간 프레임이 10% 를 넘음.  - ok: 지킴.
  - 없음: 재는 중, 또는 녹화 재생 중(판정하지 않음).

자동 선택의 규칙 (숫자의 근거는 LATENCY.md 의 "카메라 주기 지키기" 절)
  - 코어 0 하나 -> 교대: 검출 시간 p90 > UP x 주기 (0.90: 주기 33.3 ms 에서 30.0 ms. 도착 -> 넣기 4.0 ms 를 빼면 한 코어가 쓸 수 있는 시간은 29.3 ms)
    이거나 창 안에서 코어가 차 있어 버린 프레임이 DROPS 장 이상. 이 상태가 UP_HOLD_S(3초) 이어지면 바꾼다.
    3초인 까닭: 창이 2초라서 느린 프레임은 2초 동안 창에 남는다. 3초면 1초 남짓보다 짧은 한 번의 멈춤(카메라 끊김 등)으로는 바꾸지 않고,
    몇 분씩 가는 클럭 낮춤에는 3초 남짓 만에 바꾼다.
  - 교대 -> 코어 0 하나: 코어 0 의 검출 시간 p90 < DOWN x 주기 (0.85: 28.3 ms) 이고 코어가 차서 버린 프레임이 없는 상태가 DOWN_HOLD_S(10초) 이어지면 바꾼다.
    0.85 인 까닭: 0.80(26.7 ms)은 교대일 때 코어 0 의 한 장 p90 이 될 만한 값(장치 11-16 처리 시간 초당 평균 25.1 ms, p95 26.1 ms. 한 장씩은 더 퍼짐)에
    너무 가까워 교대에서 영영 안 돌아올 수 있었다(시뮬레이션 8번 가운데 5~8번). 0.85 는 한 코어가 쓸 수 있는 29.3 ms 아래다.
    돌아간 뒤 FAIL_S(60초) 안에 다시 교대로 가면 "잘못 돌아간 것"으로 보고 다음에는 두 배 오래(최대 HOLD_MAX_S 300초) 기다린다. 60초를 버티면 처음 값으로 돌아온다.
  - 마지막으로 바꾼 뒤 DWELL_S(5초) 안에는 어느 쪽으로도 바꾸지 않는다. 바꾸면 창을 비우고 새로 잰다.
  - 시작·전환·모델 바꾼 직후에는 창의 3/4(1.5초)만큼 새 프레임이 쌓일 때까지 판단하지 않는다 (처음 몇 장이 느린 것만으로 바꾸지 않게).
  - 창은 프레임마다가 아니라 EVAL_S(0.2초)마다 본다 (프레임 반복이 넣는 스레드의 잠금을 쥐고 하는 계산이라 짧게).
  - 바꿀 수 없는 때는 그 자리에 머물고 까닭을 적는다 (frame_feed.auto_conflict, 바꾸려는 때만): 검출 모델 바꾸는 중, 두 모델 상주가 교대를 씀,
    엔드투엔드 사용 중, 녹화 재생 중, 서버·카메라 작업 중, NPU 1 에 둘째 엔진 없음.
  - 화면에서 '자동'을 고르면 지금 쓰는 코어 그대로에서 시작한다(교대 중이었으면 교대, 규칙이 평소의 기다림으로 돌아온다). 이미 자동이면 아무것도 안 한다.

환경 변수 (서버를 시작할 때 읽는다)
  LWIR_NPU_FEED_AUTO_UP=0.90  LWIR_NPU_FEED_AUTO_DOWN=0.85  LWIR_NPU_FEED_AUTO_UP_HOLD_S=3  LWIR_NPU_FEED_AUTO_DOWN_HOLD_S=10
  LWIR_NPU_FEED_AUTO_DWELL_S=5  LWIR_NPU_FEED_AUTO_WINDOW_S=2  LWIR_NPU_FEED_AUTO_DROPS=3
  LWIR_NPU_FEED_AUTO_GIMBAL=0 이면 짐벌이 따라가거나 움직이는 동안에는 바꾸지 않는다 (기본 1: 바꾼다. 바꿀 때 비는 시간은 프레임 한 장 처리 시간 안쪽이다).

시험: test_frame_feed_auto.py (NPU 없이, 엔진은 흉내).
"""
import collections
import os
import time

MIN_N = 12          # samples needed in the window before a decision (2 s at 30 frames a second: 60 on one core, 30 on each of two)
SPAN = 0.75         # and the samples must reach back this share of the window (after a start, a switch or a model change: 1.5 s of new frames first)
EXACT_MS = 2.0      # the frame loop waited longer than this for the result: the service time is the detector's, not the loop's lateness
FAIL_S = 60.0       # back to core 0 only and alternating again within this time = a failed return: the next return waits twice as long
HOLD_MAX_S = 300.0  # the longest wait before a return
LATE_PCT = 10.0     # more late frames (or long result intervals) than this in the window = 'late' ('uneven') (the same tenth as the p90 of the rule)
GAP = 1.5           # an interval between two results sent to the gimbal longer than this many camera periods is a gap in the output
EVAL_S = 0.2        # the window is looked at this often, not every frame (one look is about 0.1 ms on a PC; it is under the feed lock)
OVER_SAY_S = 3.0    # 'cannot be kept with this model' for this long: shown (state 'over') and one server log line
OVER_DOWN = 0.80    # 'over' ends only below cores x this x period (53.3 ms on two cores; it starts above cores x UP x period = 60 ms) ...
OVER_END_S = 10.0   # ... with no frame dropped behind busy cores, for this long (one line). The gap between the two limits stops the flicker of 26x with the
                    # clock moving 600 <-> 800 MHz (rt_sim2: 1 start, 0 ends an hour with 10, 30 or 60 s; without the gap about 60 of each), LATENCY.md

NPU = {}            # clock_mhz / temp_c of the three NPU cores and when (npu_note)
_NPU_OK = [True]


def npu_note(status):
    """app.py, every 50 frames, with the DeviceStatus it already asked for (no extra query). Never raises: the first failure turns it off."""
    if not _NPU_OK[0]: return
    try:
        NPU.update(clock_mhz=[int(status.get_npu_clock(k_)) for k_ in range(3)], temp_c=[int(status.get_temperature(k_)) for k_ in range(3)], t=time.time())
    except Exception as e_:
        _NPU_OK[0] = False; NPU.clear()
        print('FEED: NPU clock / temperature cannot be read from DeviceStatus (%s), left out' % repr(e_)[:120], flush=True)


def _num(name, default, lo, hi):
    v_ = os.environ.get(name, '')
    if not v_: return default
    try:
        x_ = float(v_)
        if not lo <= x_ <= hi: raise ValueError(v_)
        return x_
    except ValueError:
        print('FEED auto: %s=%r ignored (allowed %g to %g), using %g' % (name, v_[:20], lo, hi, default), flush=True)
        return default


def _pct(sorted_, q):
    return sorted_[min(len(sorted_) - 1, int(len(sorted_) * q))] if sorted_ else None


def _r(v_):
    return None if v_ is None else round(v_, 1)


class Monitor:
    """One per frame_feed.Feed. arrive / drop: the feeding thread, under the feed lock. result / sent / close: the frame loop. step / switched / reset:
    the frame loop, under the feed lock. state: any thread (reads the snapshot of the last step and counters)."""

    def __init__(self):
        self.window_s = _num('LWIR_NPU_FEED_AUTO_WINDOW_S', 2.0, 0.5, 30.0)
        self.up = _num('LWIR_NPU_FEED_AUTO_UP', 0.90, 0.3, 2.0)
        self.down = _num('LWIR_NPU_FEED_AUTO_DOWN', 0.85, 0.2, 2.0)
        if self.down >= self.up:
            print('FEED auto: DOWN %g must be below UP %g, using 0.85 and 0.90' % (self.down, self.up), flush=True)
            self.up, self.down = 0.90, 0.85
        self.up_hold_s = _num('LWIR_NPU_FEED_AUTO_UP_HOLD_S', 3.0, 0.0, 60.0)
        self.down_hold_s = _num('LWIR_NPU_FEED_AUTO_DOWN_HOLD_S', 10.0, 0.0, HOLD_MAX_S)
        self.dwell_s = _num('LWIR_NPU_FEED_AUTO_DWELL_S', 5.0, 0.0, 3600.0)
        self.drops_up = int(_num('LWIR_NPU_FEED_AUTO_DROPS', 3, 1, 1000))
        self.gimbal_ok = os.environ.get('LWIR_NPU_FEED_AUTO_GIMBAL', '1') != '0'
        self.arr = collections.deque(maxlen=64)   # arrival times (s) of the frames the feeding thread saw, fed or dropped
        self.drops = collections.deque()          # (arrival s, True = dropped because no core was free)
        self.samp = collections.deque()           # [result s, core, service ms, exact, the result went out later than one period after arrival]
        self.outs = collections.deque()           # times (s) the results went out (to the gimbal), for the intervals between them
        self.open = None                          # (the sample of the frame the loop is working on, its arrival s): closed when its result goes out
        self.t_out = None                         # when the last result went out
        self.period = None                        # camera period (s), None = not known yet
        self.n_frames = self.n_late = self.n_inexact = self.n_drop = self.n_drop_busy = self.n_gap = 0
        self.n_up = self.n_down = 0
        self.last = None            # the last automatic switch
        self.hold_s = self.down_hold_s   # the wait before a return now (doubles after a failed return)
        self.t_switch = None; self.t_down = None; self.t_cond = None; self.cond_to = None
        self.log = ''               # the server log text of the switch that step() asked for
        self.note = ''              # a server log line for the feed to print (the period cannot be kept / can again)
        self.t_over = None; self.t_clear = None; self.over_on = False; self.over_text = ''; self.t_eval = None
        self.snap = dict(why='재는 중', period_ms=None, service_ms=dict(n=0, p50=None, p90=None), core0_ms=dict(n=0, p90=None), busy_drops=0,
                         period=dict(state=None, text='재는 중', late_pct=None, gap_pct=None, gap_max_ms=None, dropped_per_s=None, frames=0,
                                     one_core_ms=None, two_cores_ms=None))

    # ------------------------------------------------------------------ feeding thread (under the feed lock)
    def arrive(self, t_):
        self.arr.append(t_)

    def drop(self, t_, busy_):
        self.arr.append(t_); self.drops.append((t_, bool(busy_))); self.n_drop += 1
        if busy_: self.n_drop_busy += 1

    # ------------------------------------------------------------------ frame loop
    def result(self, core, t_arr, t_sub_ns, t_wait_ns, t_done_ns):
        """One detector result collected: submitted at t_sub_ns, the loop started to wait at t_wait_ns, had it at t_done_ns. t_arr = arrival (s)."""
        try:
            done_ = t_done_ns / 1e9; exact_ = (t_done_ns - t_wait_ns) / 1e6 >= EXACT_MS
            s_ = [done_, core, (t_done_ns - t_sub_ns) / 1e6, exact_, False]
            self.samp.append(s_); self.open = (s_, t_arr)
            self.n_frames += 1
            if not exact_: self.n_inexact += 1
        except Exception: pass   # a number of the wrong kind must not stop the frame loop

    def sent(self, t_arr, now):
        """page_handler.live_publish (frame loop), right after the frame's people went to the gimbal: its result is out now. t_arr = its arrival (s),
        the same number the feed gave the frame (LIVE.arr). Another frame's arrival (or none): nothing."""
        o_ = self.open
        if o_ is not None and o_[1] == t_arr: self._out(now)

    def close(self, now):
        """The frame loop comes for the next frame: a frame whose result was not sent (no live_publish) is finished now."""
        if self.open is not None: self._out(now)

    def _out(self, now):
        o_ = self.open; self.open = None
        try:
            p_ = self.period
            if p_ is not None and now - o_[1] > p_: o_[0][4] = True; self.n_late += 1
            if p_ is not None and self.t_out is not None and now - self.t_out > GAP * p_: self.n_gap += 1
            self.t_out = now; self.outs.append(now)
        except Exception: pass

    def _period(self):
        a_ = list(self.arr)
        d_ = sorted(b_ - c_ for b_, c_ in zip(a_[1:], a_[:-1]) if 0.0 < b_ - c_ < 0.5)
        return d_[len(d_) // 2] if len(d_) >= 8 else None

    def reset(self, over=True):
        """The cores in use were changed by someone else (page, model change): measure again from here. over=False (the automatic switch itself):
        a 'cannot be kept' statement stays (it is about the model on every usable core, not about the cores in use)."""
        self.samp.clear(); self.drops.clear(); self.outs.clear(); self.open = None; self.t_cond = None; self.cond_to = None
        if over:
            self.t_over = None; self.t_clear = None
            if self.over_on:
                self.over_on = False; self.note = 'FEED period: the detector model or the NPU cores in use changed; the camera period is judged again'

    def _kept(self, now, mode, cores, judge, pm_, n_exact, p90_, busy_):
        """Is the camera period kept? -> the period part of the snapshot. Also the server log line when 'cannot be kept' starts / ends (self.note)."""
        n_ = len(self.samp); late_ = sum(1 for s_ in self.samp if s_[4]); nd_ = len(self.drops)
        lp_ = 100.0 * late_ / n_ if n_ else None
        o_ = list(self.outs); iv_ = [b_ - a_ for a_, b_ in zip(o_[:-1], o_[1:])]
        gp_ = 100.0 * sum(1 for d_ in iv_ if d_ > GAP * pm_ / 1000.0) / len(iv_) if (iv_ and pm_) else None
        out_ = dict(state=None, text='재는 중', late_pct=_r(lp_), gap_pct=_r(gp_), gap_max_ms=_r(max(iv_) * 1000.0) if iv_ else None,
                    dropped_per_s=round(nd_ / self.window_s, 1), frames=n_,
                    one_core_ms=_r(self.up * pm_) if pm_ else None, two_cores_ms=_r(2 * self.up * pm_) if pm_ else None)
        raw_ = False; clear_ = False; evid_ = pm_ is not None and n_ >= MIN_N
        if not judge: out_['text'] = '녹화 재생 중에는 판정하지 않음'; self.t_over = None; self.t_clear = None; return out_
        if evid_:
            cap_ = cores * self.up * pm_; all_used_ = cores < 2 or mode == 'arrival'
            how_, full_ = ('NPU 두 개로도', 'NPU 두 개가') if cores >= 2 else ('NPU 하나로는 (둘째 엔진 없음)', 'NPU 하나가 (둘째 엔진 없음)')
            if n_exact >= MIN_N and p90_ > cap_:
                raw_ = True; text_ = '이 검출 모델로는 카메라 주기를 지킬 수 없음 · 검출 한 장 %.0f ms, %s %.0f ms 까지 (주기 %.1f ms) · 더 가벼운 모델을 고르세요' % (p90_, how_, cap_, pm_)
                log_ = 'detection p90 %.1f ms > %.1f ms (%d x %.2f x camera period %.1f ms)' % (p90_, cap_, cores, self.up, pm_)
            elif all_used_ and busy_ >= self.drops_up:
                raw_ = True; text_ = '이 검출 모델로는 카메라 주기를 지킬 수 없음 · %s 다 차서 %.0f초에 %d장 버림 · 더 가벼운 모델을 고르세요' % (full_, self.window_s, busy_)
                log_ = '%d frames dropped in %.1f s with every usable core busy (%d core%s, camera period %.1f ms)' % (busy_, self.window_s, cores, 's' if cores > 1 else '', pm_)
            else: clear_ = busy_ == 0 and n_exact >= MIN_N and p90_ < cores * OVER_DOWN * pm_
            if nd_:
                out_.update(state='drop', text='주기를 못 지킴 · 최근 %.0f초에 %d장 버림, 늦은 프레임 %.0f%%' % (self.window_s, nd_, lp_))
                if cores >= 2 and mode == 'normal' and n_exact >= MIN_N and p90_ > self.up * pm_: out_['text'] += ' · NPU 0·1 교대면 지킬 수 있음'
            elif gp_ is not None and gp_ > LATE_PCT:
                out_.update(state='uneven', text='주기를 못 지킴 · 결과 간격이 고르지 않음 (주기의 %.1f배 넘는 간격 %.0f%%, 가장 긴 간격 %.0f ms)' % (GAP, gp_, out_['gap_max_ms']))
            elif lp_ > LATE_PCT:
                out_.update(state='late', text='주기는 지킴 (버림 없음, 간격 고름) · 결과가 도착 + 주기 한 번보다 늦음 (늦은 프레임 %.0f%%%s)' % (lp_, ', 검출 p90 %.1f ms / 주기 %.1f ms' % (p90_, pm_) if p90_ is not None else ''))
            else: out_.update(state='ok', text='주기 지킴 (버림 없음, 간격 고름, 늦은 프레임 %.0f%%)' % lp_)
        if raw_:   # 'cannot be kept': shown once it has lasted OVER_SAY_S, then held until it is clearly over (OVER_DOWN, OVER_END_S)
            self.t_clear = None; self.over_text = text_
            if self.t_over is None: self.t_over = now
            if not self.over_on and now - self.t_over >= OVER_SAY_S:
                self.over_on = True; self.note = 'FEED period: cannot be kept with this detector model: %s. Frames are dropped or late; the model is not changed automatically.' % log_
        else:
            self.t_over = None
            if self.over_on and clear_:
                if self.t_clear is None: self.t_clear = now
                if now - self.t_clear >= OVER_END_S:
                    self.over_on = False; self.t_clear = None; self.note = 'FEED period: the detector model is no longer too slow for the camera period'
            elif evid_ and not clear_: self.t_clear = None   # a p90 between the two limits starts the clock again; measuring (too few samples) keeps it
        if self.over_on: out_.update(state='over', text=self.over_text)
        return out_

    def step(self, now, mode, on, hold=None, cores=2, judge=True):
        """Once per frame. mode = the cores in use now ('normal' = core 0 only, 'arrival' = cores 0 and 1), on = the setting is 'auto',
        hold() -> why the feed must not switch now (page text) or None, cores = detector engines there are (1 or 2), judge = False during a replay
        (the period state is not judged). -> 'arrival' / 'normal' when the feed should switch now, else None."""
        if self.t_eval is not None and 0.0 <= now - self.t_eval < EVAL_S: return None
        self.t_eval = now
        cut_ = now - self.window_s
        while self.samp and self.samp[0][0] < cut_: self.samp.popleft()
        while self.drops and self.drops[0][0] < cut_: self.drops.popleft()
        while self.outs and self.outs[0] < cut_: self.outs.popleft()
        p_ = self.period = self._period()
        all_ = sorted(s_[2] for s_ in self.samp if s_[3]); c0_ = sorted(s_[2] for s_ in self.samp if s_[3] and s_[1] == 0)
        busy_ = sum(1 for d_ in self.drops if d_[1])
        p90_ = _pct(all_, 0.9); p90c_ = _pct(c0_, 0.9); to_ = None; why_ = ''; log_ = ''
        full_ = bool(self.samp) and now - self.samp[0][0] >= SPAN * self.window_s   # the first frames after a start / switch alone decide nothing
        pm_ = p_ * 1000.0 if p_ is not None else None
        if not on: why_ = ''
        elif p_ is None: why_ = '카메라 주기를 재는 중'
        elif mode == 'normal':
            lim_ = self.up * pm_
            if len(all_) >= MIN_N and full_ and p90_ > lim_:
                to_ = 'arrival'; why_ = '검출이 느림 (p90 %.1f ms > %.1f ms)' % (p90_, lim_)
                log_ = 'detection p90 %.1f ms > %.1f ms (%.2f x camera period %.1f ms)' % (p90_, lim_, self.up, pm_)
            elif busy_ >= self.drops_up:
                to_ = 'arrival'; why_ = 'NPU 0이 차 있어 %d장 버림 (%.0f초 동안)' % (busy_, self.window_s)
                log_ = '%d frames dropped in %.1f s while core 0 was busy (camera period %.1f ms)' % (busy_, self.window_s, pm_)
            elif len(all_) < MIN_N or not full_: why_ = '재는 중'
            else: why_ = 'NPU 0만으로 충분 (검출 p90 %.1f ms, 한도 %.1f ms)' % (p90_, lim_)
        else:
            lim_ = self.down * pm_
            if len(c0_) < MIN_N or not full_: why_ = '재는 중'
            elif p90c_ < lim_ and busy_ == 0:
                to_ = 'normal'; why_ = '검출이 다시 빠름 (NPU 0 p90 %.1f ms < %.1f ms)' % (p90c_, lim_)
                log_ = 'detection p90 on core 0 %.1f ms < %.1f ms (%.2f x camera period %.1f ms)' % (p90c_, lim_, self.down, pm_)
            else: why_ = '검출이 아직 느림 (NPU 0 p90 %.1f ms, 돌아가는 기준 %.1f ms)' % (p90c_, lim_)
        r_ = hold() if (to_ is not None and hold is not None) else None   # asked only when a switch is wanted: otherwise the measured why stays
        if r_ is not None: to_ = None; why_ = '머무름 · %s · %s' % (r_, why_)
        if to_ is None or to_ != self.cond_to: self.t_cond = None
        self.cond_to = to_
        if to_ is not None:
            if self.t_cond is None: self.t_cond = now
            need_ = self.up_hold_s if to_ == 'arrival' else self.hold_s
            held_ = now - self.t_cond
            left_ = max(need_ - held_, self.dwell_s - (now - self.t_switch) if self.t_switch is not None else 0.0)
            if left_ > 0: why_ += ' · %.0f초 뒤 %s' % (left_ + 0.5, 'NPU 0·1 교대' if to_ == 'arrival' else 'NPU 0만'); to_ = None
            else: self.log = '%s for %.1f s' % (log_, held_)
        self.snap = dict(why=why_, period_ms=_r(pm_), service_ms=dict(n=len(all_), p50=_r(_pct(all_, 0.5)), p90=_r(p90_)),
                         core0_ms=dict(n=len(c0_), p90=_r(p90c_)), busy_drops=busy_,
                         period=self._kept(now, mode, cores, judge, pm_, len(all_), p90_, busy_))
        return to_

    def switched(self, to_, now):
        """The feed switched as step() asked. -> the server log line."""
        if to_ == 'arrival':
            self.n_up += 1
            failed_ = self.t_down is not None and now - self.t_down < FAIL_S
            self.hold_s = min(max(self.hold_s, 1.0) * 2.0, HOLD_MAX_S) if failed_ else self.down_hold_s
            text_ = 'core 0 only -> cores 0 and 1: %s%s' % (self.log, '; the next return waits %.0f s' % self.hold_s if failed_ else '')
        else:
            self.n_down += 1; self.t_down = now
            text_ = 'cores 0 and 1 -> core 0 only: %s' % self.log
        self.t_switch = now; self.last = dict(at=time.strftime('%H:%M:%S'), to=to_, why=text_)
        self.reset(over=False)
        return text_

    def held(self, why_):
        """The switch step() asked for could not be done after all (something started in between): nothing changed."""
        self.t_cond = None; self.cond_to = None
        self.snap = dict(self.snap, why='머무름 · ' + why_)

    # ------------------------------------------------------------------ any thread
    def state(self, on):
        npu_ = dict(NPU)
        if npu_: npu_['age_s'] = round(time.time() - npu_.pop('t'), 1)
        return dict(self.snap, on=bool(on), npu=npu_ or None,
                    limit=dict(up=self.up, down=self.down, up_hold_s=self.up_hold_s, down_hold_s=self.hold_s, dwell_s=self.dwell_s, window_s=self.window_s, drops=self.drops_up),
                    misses=dict(frames=self.n_frames, late=self.n_late, gaps=self.n_gap, dropped=self.n_drop, dropped_busy=self.n_drop_busy), late_collect=self.n_inexact,
                    switches=dict(up=self.n_up, down=self.n_down, last=self.last))
