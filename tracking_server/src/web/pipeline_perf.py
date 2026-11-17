"""Live camera, decoded boxes and final track output, independent of JPEG delivery."""

from collections import deque
from math import floor, isfinite
from threading import Lock
import time


_HISTORY_S = 60
_MAX_SAMPLES = 8192  # comfortably over 60 s of frames at the supported camera rates


def _delay_ms(received, finished):
    if isinstance(received, (int, float)) and isfinite(received) and 0 < received <= finished:
        return (finished - received) * 1000
    return None


def _p95(values):
    if not values:
        return None
    values.sort()
    pos = 0.95 * (len(values) - 1)
    lo = int(pos)
    return round(values[lo] + (pos - lo) * (values[min(lo + 1, len(values) - 1)] - values[lo]), 2)


class PipelinePerf:
    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._start = clock()
        self._lock = Lock()
        self._counts = {}
        self._latest_bin = 0
        self._samples = [deque(maxlen=_MAX_SAMPLES) for _ in range(3)]

    def camera(self, arrived, transport_ms=None):
        self._record(0, arrived, transport_ms)

    def detected(self, arrived, finished):
        delay = _delay_ms(arrived, finished)
        if delay is not None:
            self._record(1, finished, delay)

    def tracked(self, arrived, finished):
        delay = _delay_ms(arrived, finished)
        if delay is not None:
            self._record(2, finished, delay)

    def _record(self, stage, finished, delay):
        bucket = floor(finished - self._start)
        with self._lock:
            if bucket > self._latest_bin:
                self._latest_bin = bucket
                for old in tuple(self._counts):
                    if old < bucket - 2:
                        del self._counts[old]
            if max(0, self._latest_bin - 2) <= bucket:
                counts = self._counts.get(bucket)
                if counts is None:
                    counts = self._counts[bucket] = [0, 0, 0]
                counts[stage] += 1
            if delay is not None:
                samples = self._samples[stage]
                while samples and samples[0][0] <= finished - _HISTORY_S:
                    samples.popleft()
                samples.append((finished, delay))

    def snapshot(self):
        with self._lock:
            now = self._clock()
            last_complete = floor(now - self._start) - 1
            camera, det, track = self._counts.get(last_complete, (0, 0, 0) if last_complete >= 0 else (None, None, None))
            c_ms, d_ms, t_ms = ([ms for t, ms in rows if now - _HISTORY_S < t <= now] for rows in self._samples)
        return dict(camera_fps=camera, det_fps=det, track_fps=track,
                    camera_p95_ms=_p95(c_ms), det_p95_ms=_p95(d_ms), track_p95_ms=_p95(t_ms),
                    camera_count=len(c_ms), det_count=len(d_ms), sample_count=len(t_ms), goal_ms=33)
