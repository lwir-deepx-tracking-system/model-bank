"""프레임 한 장이 서버 안의 단계마다 언제 지나갔는지 재기 (2026-11-11). 기본으로 꺼져 있다. 환경 변수 LWIR_STAGE_TIMING 이 정확히 1 일 때만 켜진다.

꺼져 있으면 app.py 의 프레임 반복에서 "APP.TIMING is not None" 을 두 번 보는 것 말고는 아무 일도 하지 않는다 (이 파일의 코드는 start() 한 번만 실행된다).
실시간 실행(--live-port)에서만 켜진다.

프레임마다 적는 시각 (초, 서버의 프레임 시계 = time.perf_counter. 장치의 리눅스에서는 CLOCK_MONOTONIC 과 같은 시계다)
  cam         카메라 프로그램이 프레임을 받은 시각 (LWIR_CAM_STAMP=1 일 때만, 아니면 비어 있음)
  arrival     프레임이 서버에 다 들어온 시각 (카메라 읽기 스레드)
  submit      검출기(NPU)에 넣은 시각
  det_ready   검출기 결과를 받은 시각
  decoded     상자로 풀고 겹침 정리(NMS)·작은 사람 거름까지 끝난 시각
  track_start 추적기에 넣기 직전 (카메라 움직임 보정, 온도 거름이 끝난 뒤)
  tracked     추적기가 끝난 시각 (BoT-SORT 의 외형 비교 조각도 여기에 든다)
  identity    번호 잇기(외형 모델, NPU 코어 2)까지 끝난 시각. 번호 잇기를 끄면 tracked 와 같다
  published   결과를 내보낸 시각 (짐벌 프로그램에 보내고 화면용 그림을 만든 뒤)
그리고 프레임 번호(frame), 그때까지 버린 프레임 수(q_drop, 처음부터 센 값), 카메라 프로그램이 붙인 프레임 번호(cam_seq, LWIR_CAM_STAMP=1 일 때만. 번호가 건너뛰면 그 사이 프레임은 서버가 받지 않았거나 버린 것이다).

주의: 보통 방식(--pipeline)에서는 det_ready 와 decoded 사이에 "다음 프레임을 읽어 검출기에 넣는 일"이 들어 있다 (다음 프레임이 아직 안 왔으면 기다리는 시간도).
도착하는 대로 넣는 방식(LWIR_NPU_FEED=arrival)에서는 submit 이 프레임이 도착한 직후이고, det_ready 는 "프레임 반복이 그 결과를 받아 간 시각"이다.

내는 것
  - 60 초마다 한 줄 (STAGE 60 s: ...): 지난 60 초의 프레임 수, 구간마다 평균/p50/p95 ms, 도착→내보냄 평균/p50/p95, 버린 프레임 수. 화면 출력과 카메라 기록(camera_events.log)에.
  - 끝날 때 <결과 폴더>/stage_timing.csv (마지막 N 천 프레임, 기본 3 천. LWIR_STAGE_TIMING_N=<천 단위 수> 로 바꾼다) 와 stage_timing_summary.json
    (처음부터의 프레임 수·구간 평균, 그리고 마지막 N 천 프레임의 구간별 개수·평균·p50·p95).
    N 은 한 주기의 프레임 수(60 초 x 초당 프레임 수)보다 커야 60 초 줄이 그 60 초를 다 본다 (기본 3 천이면 된다).
  - 도는 중에 CSV 가 필요하면 <결과 폴더>에 stage_timing.request 라는 빈 파일을 만든다. 2~3 초 안에 stage_timing_<시분초>.csv 가 생기고 그 파일은 지워진다.
켰을 때 프레임마다 하는 일: 시계 두 번 읽기, 숫자 12 개를 미리 잡아 둔 표의 한 줄에 쓰기. 통계 계산과 파일 쓰기는 따로 도는 스레드가 한다.
"""
import atexit
import json
import os
import threading
import time

import numpy as np

from .state import APP

COLS = ('frame', 'q_drop', 'cam_seq', 'cam', 'arrival', 'submit', 'det_ready', 'decoded', 'track_start', 'tracked', 'identity', 'published')
C = {n: i for i, n in enumerate(COLS)}
STAGES = (('cam>arrival', 'cam', 'arrival'), ('arrival>submit', 'arrival', 'submit'), ('submit>det_ready', 'submit', 'det_ready'), ('det_ready>decoded', 'det_ready', 'decoded'),
          ('decoded>track_start', 'decoded', 'track_start'), ('track_start>tracked', 'track_start', 'tracked'), ('tracked>identity', 'tracked', 'identity'),
          ('identity>published', 'identity', 'published'), ('arrival>published', 'arrival', 'published'))
STAGES_KO = {'cam>arrival': '카메라 프로그램이 받음 → 서버에 도착', 'arrival>submit': '도착 → 검출기에 넣음', 'submit>det_ready': '검출기에 넣음 → 결과 받음',
             'det_ready>decoded': '결과 받음 → 상자로 풂 (보통 방식에서는 다음 프레임 읽고 넣기 포함)', 'decoded>track_start': '상자로 풂 → 추적기 직전 (움직임 보정, 온도 거름)',
             'track_start>tracked': '추적기', 'tracked>identity': '번호 잇기 (외형 모델)', 'identity>published': '기록 정리 → 내보냄 (짐벌, 화면 그림)', 'arrival>published': '도착 → 내보냄 (전체)'}


def wanted():
    return os.environ.get('LWIR_STAGE_TIMING') == '1'


def start():
    """app.py 가 프레임 반복 전에 한 번 부른다. 꺼져 있으면 None."""
    if not wanted(): return None
    if APP.LIVE is None:
        print('STAGE timing is for live runs only (--live-port): off', flush=True); return None
    try:
        n_ = int(float(os.environ.get('LWIR_STAGE_TIMING_N', '3') or 3) * 1000); every_ = float(os.environ.get('LWIR_STAGE_TIMING_S', '60') or 60)
        t_ = StageTiming(str(APP.out), min(max(n_, 100), 200000), max(every_, 1.0))
    except Exception as e_:
        print('STAGE timing could not start (%s): off' % repr(e_)[:120], flush=True); return None
    atexit.register(t_.close)
    _say('STAGE timing on: last %d frames kept, one line every %.0f s, files in the output folder at the end' % (t_.n, t_.every))
    return t_


def _say(text_):
    try:
        from ..camera import camera_extra
        camera_extra.say(text_)
    except Exception:
        try: print(text_, flush=True)
        except Exception: pass


def intervals(rows_):
    """rows_ [k, len(COLS)] 초 -> {구간 이름: ms 값 배열 (값이 있는 프레임만)}"""
    out_ = {}
    for name_, a_, b_ in STAGES:
        d_ = (rows_[:, C[b_]] - rows_[:, C[a_]]) * 1000.0
        out_[name_] = d_[np.isfinite(d_)]
    return out_


def stats(rows_):
    out_ = {}
    for name_, v_ in intervals(rows_).items():
        out_[name_] = dict(n=int(len(v_)), mean=round(float(v_.mean()), 3), p50=round(float(np.percentile(v_, 50)), 3), p95=round(float(np.percentile(v_, 95)), 3)) if len(v_) else dict(n=0, mean=None, p50=None, p95=None)
    return out_


class StageTiming:
    def __init__(self, out_dir_, n_, every_):
        self.out_dir = out_dir_; self.n = n_; self.every = every_
        self.buf = np.full((n_, len(COLS)), np.nan, np.float64); self.count = 0
        self.tf = 0; self.ta = None
        self.t_sum = time.monotonic(); self.sum_from = 0; self.drop_from = 0; self.closed = False
        self.tot_n = {s_[0]: 0 for s_ in STAGES}; self.tot_ms = {s_[0]: 0.0 for s_ in STAGES}; self.lock = threading.Lock()
        self.req = os.path.join(out_dir_, 'stage_timing.request')

    def mark(self):   # 추적기에 넣기 직전
        self.tf = time.perf_counter_ns()
        a_ = APP.LIVE.arr; self.ta = a_[0] if a_ else None

    def frame(self, i_, t0_, t1_, t2_, t4_, rl_ms_, t5_):   # 결과를 내보낸 직후. t0.. 는 프레임 반복이 이미 재 둔 perf_counter_ns 값
        tp_ = time.perf_counter(); ta_ = self.ta
        if ta_ is None: return
        cam_ = seq_ = np.nan
        cs_ = getattr(APP.LIVE, 'cam_stamp', None)
        if cs_ is not None and cs_['on']:
            for r_ in reversed(APP.LIVE.cam_stamp_recent):
                if r_[0] == ta_: cam_ = r_[2]; seq_ = r_[1]; break
                if r_[0] < ta_: break
        tk_ = t4_ * 1e-9
        self.buf[self.count % self.n] = (i_, APP.LIVE.q_drop, seq_, cam_, ta_, t0_ * 1e-9, t1_ * 1e-9, t2_ * 1e-9, self.tf * 1e-9, tk_, tk_ + rl_ms_ * 1e-3, tp_)
        self.count += 1
        if self.count & 63 == 0:
            now_ = time.monotonic(); due_ = now_ - self.t_sum >= self.every
            if due_ or self.count - self.sum_from > self.n - 128:   # 주기가 됐거나, 아직 셈에 넣지 않은 프레임이 표에서 밀려나기 직전 (N 을 작게 잡았을 때): 그때는 줄 없이 셈에만 넣는다
                if due_: self.t_sum = now_
                rows_ = self._rows(self.sum_from); self.sum_from = self.count
                threading.Thread(target=self._summary, args=(rows_, self.every, due_), daemon=True).start()
            if os.path.exists(self.req):
                try: os.remove(self.req)
                except OSError: pass
                threading.Thread(target=self._write_csv, args=(self._rows(0), 'stage_timing_%s.csv' % time.strftime('%H%M%S')), daemon=True).start()

    def _rows(self, from_):   # from_ 번째 프레임부터 지금까지 (표에 남아 있는 것만), 복사본
        k_ = min(self.count - from_, self.n, self.count)
        if k_ <= 0: return np.empty((0, len(COLS)))
        idx_ = np.arange(self.count - k_, self.count) % self.n
        return self.buf[idx_]

    def _summary(self, rows_, secs_, say_=True):
        try:
            if not len(rows_): return
            iv_ = intervals(rows_)
            with self.lock:
                for name_, v_ in iv_.items(): self.tot_n[name_] += int(len(v_)); self.tot_ms[name_] += float(v_.sum())
            if not say_: return
            parts_ = []
            for name_, v_ in iv_.items():
                if len(v_): parts_.append('%s %.1f/%.1f/%.1f' % (name_, v_.mean(), np.percentile(v_, 50), np.percentile(v_, 95)))
            d1_ = int(rows_[-1, C['q_drop']]); d0_ = self.drop_from; self.drop_from = d1_
            _say('STAGE %.0f s: frames %d, dropped %d (since start %d), ms mean/p50/p95: %s' % (secs_, len(rows_), d1_ - d0_, d1_, ' | '.join(parts_)))
        except Exception as e_:
            _say('STAGE summary failed: %s' % repr(e_)[:120])

    def _write_csv(self, rows_, name_):
        try:
            p_ = os.path.join(self.out_dir, name_)
            with open(p_ + '.tmp', 'w', encoding='utf-8', newline='\n') as f_:
                f_.write(','.join(COLS) + '\n')
                for r_ in rows_:
                    f_.write('%d,%d,%s,' % (r_[0], r_[1], '%d' % r_[2] if np.isfinite(r_[2]) else '') + ','.join('' if not np.isfinite(v_) else '%.6f' % v_ for v_ in r_[3:]) + '\n')
            os.replace(p_ + '.tmp', p_)
            _say('STAGE csv written: %s (%d frames)' % (name_, len(rows_)))
        except Exception as e_:
            _say('STAGE csv failed: %s' % repr(e_)[:120])

    def close(self):   # 서버가 끝날 때 (app.py 가 부르고, 못 불렀으면 Python 이 끝나면서 부른다). 두 번 불러도 된다
        if self.closed: return
        self.closed = True
        try:
            self._summary(self._rows(self.sum_from), 0, say_=False); self.sum_from = self.count
            rows_ = self._rows(0)
            self._write_csv(rows_, 'stage_timing.csv')
            with self.lock:
                all_ = {n_: dict(n=self.tot_n[n_], mean=round(self.tot_ms[n_] / self.tot_n[n_], 3) if self.tot_n[n_] else None) for n_ in self.tot_n}
            res_ = dict(frames=self.count, dropped=int(APP.LIVE.q_drop), kept_frames=int(len(rows_)), since_start=all_, last_kept_frames=stats(rows_), unit='ms',
                        feed=os.environ.get('LWIR_NPU_FEED', '') or 'normal', stages_ko=STAGES_KO, columns=list(COLS))
            with open(os.path.join(self.out_dir, 'stage_timing_summary.json'), 'w', encoding='utf-8', newline='\n') as f_: json.dump(res_, f_, ensure_ascii=False, indent=1)
            lk_ = res_['last_kept_frames']['arrival>published']
            _say('STAGE end: frames %d, dropped %d, arrival>published of the last %d frames: mean %s p50 %s p95 %s ms' % (self.count, res_['dropped'], len(rows_), lk_['mean'], lk_['p50'], lk_['p95']))
        except Exception as e_:
            _say('STAGE end failed: %s' % repr(e_)[:120])
