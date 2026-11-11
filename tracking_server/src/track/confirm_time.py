"""새 사람 확인 규칙을 시간(ms)으로 적기 (2026-11-11, --confirm-ms).

추적기는 "새 궤적이 N 장 연속으로 짝이 맞아야 화면에 낸다"(--n-init) 를 장 수로 센다. 초당 장 수가 바뀌면 같은 N 장이 다른 시간이 된다
(3 장 = 22 fps 에서 136 ms, 30 fps 에서 100 ms). 이 파일은 규칙을 ms 로 받아, 실제로 잰 초당 장 수로 장 수를 다시 계산한다.

    장 수 = round(ms x fps / 1000)          예: 130 ms -> 22~27 fps 에서 3 장, 30 fps 에서 4 장

- fps = 추적기가 받은 프레임의 도착 시각 간격(최근 WINDOW 개)의 중간값. 1 초가 넘는 간격(카메라 끊김)은 세지 않는다.
- 경계에서 3 장과 4 장 사이를 오가지 않도록 여유(MARGIN)를 둔다: 장 수는 x = ms x fps / 1000 이 지금 값에서 0.5 + MARGIN 넘게 벗어나야 바뀐다.
- 추적기마다 원래 장 수(--n-init 로 받은 값)보다 작아지지는 않는다(frames_for). 느려졌다고 확인을 느슨하게 하지 않기 위해서다.
- 시작할 때는 NOMINAL_FPS(카메라 25 fps 모드) 기준의 장 수로 시작한다.
- ms = 0 은 끔: 추적기는 원래 장 수(--n-init)로 돌아간다. 실행 중에 set_ms 로 바꿀 수 있다(화면의 Filter 값, /rules/set?confirm_ms=).
"""
import statistics
from collections import deque

NOMINAL_FPS = 25.0   # 카메라 25 fps 모드. 잰 값이 모이기 전의 시작 값
WINDOW = 50          # 중간값을 내는 간격 개수 (약 2 초)
MARGIN = 0.15        # 경계 여유 (장 수 단위)
GAP_S = 1.0          # 이보다 긴 간격은 끊김으로 보고 세지 않음
MS_MAX = 1000.0      # 받는 값의 위쪽 한계 (ms)


class ConfirmByTime:
    def __init__(self, ms, nominal_fps=NOMINAL_FPS, window=WINDOW, margin=MARGIN):
        if ms < 0: raise ValueError('confirm ms must be >= 0')
        self.margin = float(margin); self.window = int(window); self.nominal = float(nominal_fps)
        self.iv = deque(maxlen=self.window); self.last = None; self.fps = None; self.changes = 0
        self.set_ms(ms)

    def set_ms(self, ms):
        """실행 중에 값을 바꾼다. 0 = 끔 (장 수 0 -> frames_for 가 추적기의 원래 장 수를 돌려준다). 잰 속도가 있으면 그 속도로, 없으면 NOMINAL_FPS 로 바로 계산한다."""
        self.ms = max(0.0, min(MS_MAX, float(ms)))
        self.n = max(1, int(self.ms * (self.fps or self.nominal) / 1000.0 + 0.5)) if self.ms > 0 else 0

    def tick(self, t):
        """t = 이번 프레임의 도착 시각(초, 단조 증가 시계). 돌려주는 값 = 지금의 장 수."""
        if self.last is not None:
            dt = t - self.last
            if 0.0 < dt <= GAP_S: self.iv.append(dt)
        self.last = t
        if len(self.iv) >= self.window:
            self.fps = 1.0 / statistics.median(self.iv)
            x = self.ms * self.fps / 1000.0
            if self.ms > 0 and (x > self.n + 0.5 + self.margin or x < self.n - 0.5 - self.margin):
                n = max(1, int(x + 0.5))
                if n != self.n: self.n = n; self.changes += 1
        return self.n

    def frames_for(self, base):
        """추적기의 원래 장 수(base)보다 작아지지 않게 한 장 수."""
        return max(int(base), self.n)

    def state(self, tracker=None):
        base = getattr(tracker, 'n_init_base', None) if tracker is not None else None
        return dict(ms=self.ms, frames=(self.frames_for(base) if base is not None else (self.n or None)), fps=(round(self.fps, 1) if self.fps else None), changes=self.changes)


def apply(rule, tracker, t):
    """매 프레임, 추적기 update 전에 부른다. 추적기의 n_init 을 지금 장 수로 맞추고, 바뀌었으면 (이전, 새 값) 을 돌려준다(아니면 None).
    이미 화면에 나온 궤적은 그대로다. 아직 확인 전인 궤적만 새 장 수를 따른다."""
    n = rule.tick(t)
    if not hasattr(tracker, 'n_init_base'): tracker.n_init_base = int(tracker.n_init)
    want = max(tracker.n_init_base, n)
    if want == tracker.n_init: return None
    old = tracker.n_init; tracker.n_init = want
    return old, want
