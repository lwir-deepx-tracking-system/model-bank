"""Virtual camera following for recordings. State is independent of live control."""
from collections import deque
from math import exp, isfinite
from pathlib import Path
from threading import Lock

import numpy as np

from ..track.position_memory import PosGRU


MAX_AGE = 10.0
HISTORY = np.linspace(-2.0, 0., 11)


class CentrePredictor:
    def __init__(self, weight_path):
        with np.load(weight_path, allow_pickle=False) as weights:
            self.w = {key: np.array(weights[key], dtype=np.float64) for key in (
                'history_weight_ih_l0', 'history_weight_hh_l0', 'history_bias_ih_l0',
                'history_bias_hh_l0', 'head_0_weight', 'head_0_bias',
                'head_2_weight', 'head_2_bias',
            )}
        w = self.w
        self.gru = PosGRU(dict(w_ih=w['history_weight_ih_l0'], w_hh=w['history_weight_hh_l0'],
                             b_ih=w['history_bias_ih_l0'], b_hh=w['history_bias_hh_l0'],
                             w_o=np.eye(24), b_o=np.zeros(24)))
        if not np.isfinite(self.centre(self.gru.init(), .5)).all():
            raise ValueError('invalid centre predictor')

    def history_state(self, records):
        if len(records) < 2:
            return None
        times, boxes = zip(*records)
        t = np.asarray(times, dtype=float)
        x = np.asarray(boxes, dtype=float)
        targets = t[-1] + HISTORY
        if targets[0] < t[0] - 1e-7 or max(np.diff(t)) > .45:
            return None
        resampled = np.column_stack([np.interp(targets, t, col) for col in x.T])
        last = resampled[-1]
        feature = np.c_[(resampled[:, :2] - last[:2]) / max(last[3], 1.),
                        np.log(np.maximum(resampled[:, 2:], 1.) / last[2:]), HISTORY / 2.]
        state = self.gru.init()
        for row in feature:
            _, state = self.gru.step(row, state)
        return state

    def centre(self, state, age):
        if not .5 <= age <= MAX_AGE:
            raise ValueError('time outside trained GRU interval')
        w = self.w
        x = np.r_[state, np.log1p(age) / np.log1p(MAX_AGE)]
        return w['head_2_weight'] @ np.tanh(w['head_0_weight'] @ x + w['head_0_bias']) + w['head_2_bias']


class ReplayFollow:
    def __init__(self):
        self.lock = Lock()
        self._clear()
        self.model = None
        self.attempted = False
        self.frames = deque(maxlen=48)
        self.zoom = 2.0

    def _clear(self):
        self.epoch = self.pos = self.selected = self.anchor = self.centre = self.state = None
        self.history = deque(maxlen=180)

    def clear(self):
        with self.lock:
            if self.epoch is not None:
                self._clear()
                self.frames.clear()

    def _select(self, person):
        self.selected = person
        self.anchor = self.centre = self.state = None

    def _try_model(self):
        if not self.attempted:
            self.attempted = True
            try:
                self.model = CentrePredictor(Path(__file__).resolve().parents[2] / 'models/track/pos_gru_elapsed_10s.npz')
            except (OSError, ValueError, KeyError) as error:
                print('REPLAY GRU:', type(error).__name__, str(error)[:100], flush=True)

    def click(self, stamp, epoch_id, x, y):
        with self.lock:
            shown = next((row for row in reversed(self.frames) if row[0] == stamp), None)
            if shown is None or shown[1] != epoch_id or not 0 <= x <= 1 or not 0 <= y <= 1:
                return False, '최신 녹화 화면에서 다시 누르세요'
            _, _, boxes, crop = shown
            left, top, width, height = crop
            x, y = left + x * width, top + y * height
            hits = [tid for tid, (a, b), (c, d) in boxes if a <= x <= c and b <= y <= d]
            if len(hits) > 1:
                return False, '상자가 겹칩니다. 한 사람만 있는 위치를 누르세요'
            self._select(hits[0] if hits else None)
            return True, ('%d번 확대 추적' % hits[0]) if hits else '전체 화면'

    def set_zoom(self, value):
        if not 1.0 <= value <= 4.0:
            return False, '확대 배율은 1.0~4.0배입니다'
        with self.lock:
            self.zoom = value
        return True, '확대 %.1f배' % value

    def update(self, stamp, receipt, fps, boxes, width, height):
        # Draw rows are already aligned with the preview image. No physical
        # camera control and no use of labels or future tracking results.
        run, pos, round_no, _ = receipt
        epoch = run, round_no, float(fps)
        now = pos / float(fps)
        rows = tuple((int(tid), xy1, xy2) for xy1, xy2, _, tid in boxes)
        with self.lock:
            if self.epoch != epoch or self.pos is not None and (pos < self.pos or pos - self.pos > max(fps*.45, 1)):
                self._clear()
                self.frames.clear()
            self.epoch = epoch
            if self.pos != pos:
                self.history.append((now, rows))
                while self.history and now - self.history[0][0] > 2.55:
                    self.history.popleft()
            dt = max(0., (pos - self.pos) / fps) if self.pos is not None else 0.
            self.pos = pos
            crop = (0, 0, width, height)
            age = None
            predicted_box = None
            if self.selected is not None:
                match = next((self._centre_box(a,b) for tid,a,b in rows if tid == self.selected), None)
                if match is not None:
                    self.anchor = now, match
                    self.state = None
                    estimate = match
                elif self.anchor is not None and now - self.anchor[0] <= MAX_AGE + 1e-8:
                    age = min(MAX_AGE, now - self.anchor[0])
                    t0, last = self.anchor
                    self._try_model()
                    if self.model is not None and self.state is None:
                        history = [(t, box) for t, frame in self.history if t <= t0
                                   for tid,a,b in frame if tid == self.selected
                                   and (box := self._centre_box(a,b)) is not None]
                        self.state = self.model.history_state(history)
                    estimate = last
                    if self.state is not None and age >= .5:
                        x, y = self.model.centre(self.state, age) * last[3] + last[:2]
                        if isfinite(x) and isfinite(y):
                            estimate = (x, y, last[2], last[3])
                            predicted_box = estimate
                else:
                    self._select(None)
                    estimate = None
                if estimate is not None:
                    target = np.array([estimate[0], estimate[1] - .15 * estimate[3]])
                    alpha = 1. - exp(-dt / .13)
                    self.centre = target if self.centre is None else self.centre + alpha * (target - self.centre)
                    w, h = max(1, round(width / self.zoom)), max(1, round(height / self.zoom))
                    left = int(np.clip(self.centre[0] - w*.5, 0, width-w))
                    top = int(np.clip(self.centre[1] - h*.43, 0, height-h))
                    crop = left, top, w, h
            self.frames.append((stamp, run, rows, crop))
            return crop, self.selected, age, predicted_box, self.zoom

    @staticmethod
    def _centre_box(a, b):
        w, h = b[0] - a[0], b[1] - a[1]
        return ((a[0]+b[0])*.5, (a[1]+b[1])*.5, float(w), float(h)) if min(w, h) > 1 else None


VIRTUAL = ReplayFollow()
