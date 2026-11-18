"""Live camera, decoded boxes and final track output, independent of JPEG delivery."""

from collections import deque
from math import floor, isfinite
from threading import Lock
import time


_HISTORY_S = 60
_RATE_WINDOW_S = 5
_MAX_SAMPLES = 8192  # comfortably over 60 s of frames at the supported camera rates
_SCHEDULE_SIZE = 3
_SCHEDULE_FRESH_S = 2.5
_PENDING_LIMIT = 128


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
        self._pending = {}
        self._completed = []

    def camera(self, arrived, transport_ms=None, *, dropped=False):
        self._record(0, arrived, transport_ms, dropped=dropped)

    def dropped_input(self):
        self._record(3, self._clock(), None)

    def detected(self, arrived, finished, cores=()):
        delay = _delay_ms(arrived, finished)
        if delay is not None:
            self._record(1, finished, delay, arrived, cores)

    def tracked(self, arrived, finished):
        delay = _delay_ms(arrived, finished)
        if delay is not None:
            self._record(2, finished, delay, arrived)

    def _record(self, stage, finished, delay, arrived=None, cores=(), *, dropped=False):
        bucket = floor(finished - self._start)
        with self._lock:
            if bucket > self._latest_bin:
                self._latest_bin = bucket
                for old in tuple(self._counts):
                    if old < bucket - _RATE_WINDOW_S:
                        del self._counts[old]
            if max(0, self._latest_bin - _RATE_WINDOW_S) <= bucket:
                counts = self._counts.get(bucket)
                if counts is None:
                    counts = self._counts[bucket] = [0, 0, 0, 0]
                counts[stage] += 1
                if dropped:
                    counts[3] += 1
            if delay is not None:
                samples = self._samples[stage]
                while samples and samples[0][0] <= finished - _HISTORY_S:
                    samples.popleft()
                samples.append((finished, delay))
            if stage < 3:
                self._pair(stage, finished, delay, arrived, cores)

    def _pair(self, stage, finished, delay, arrived, cores):
        if stage == 0:
            # A transport sample marks camera-stamped live input. Replay
            # packets and the fallback without camera stamps supply no sample.
            if delay is None:
                if self._pending or self._completed:
                    self._pending.clear()
                    self._completed.clear()
                return
            self._pending[finished] = None
            if len(self._pending) > _PENDING_LIMIT:
                self._pending.pop(next(iter(self._pending)))
        elif stage == 1:
            if arrived in self._pending and self._pending[arrived] is None:
                self._pending[arrived] = (finished, cores)
        else:
            detected = self._pending.pop(arrived, None)
            if detected is not None and detected[0] <= finished:
                det_time, det_cores = detected
                rows = self._completed
                rows.append((arrived, det_time, finished, det_cores))
                if len(rows) > 1 and rows[-2][0] > arrived:
                    rows.sort()
                if len(rows) > _SCHEDULE_SIZE:
                    rows.pop(0)

    def snapshot(self):
        with self._lock:
            now = self._clock()
            last_complete = floor(now - self._start) - 1
            counts = self._counts.get(last_complete, (0, 0, 0, 0))
            camera, det, track = counts[:3] if last_complete >= 0 else (None, None, None)
            rate_fps = (None, None, None)
            drops = None
            output_times = []
            if last_complete >= _RATE_WINDOW_S - 1:
                end = self._start + last_complete + 1
                start = end - _RATE_WINDOW_S
                recent_buckets = range(last_complete + 1 - _RATE_WINDOW_S, last_complete + 1)
                stage_counts = [self._counts.get(b, (0, 0, 0, 0)) for b in recent_buckets]
                cam_count, det_count, track_count, drops = map(sum, zip(*stage_counts))
                rate_fps = tuple(count / _RATE_WINDOW_S for count in (cam_count, det_count, track_count))
                output_times = [t for t, _ in self._samples[2] if start <= t < end]
            c_ms, d_ms, t_ms = ([ms for t, ms in rows if now - _HISTORY_S < t <= now] for rows in self._samples)
            frames = [row for row in self._completed if now - _SCHEDULE_FRESH_S < row[0] and row[2] <= now]
        first = frames[0][0] if frames else 0
        schedule = [[round((stamp - first) * 1000, 3) for stamp in row[:3]] for row in frames]
        intervals = [(after - before) * 1000 for before, after in zip(output_times, output_times[1:])]
        return dict(camera_fps=camera, det_fps=det, track_fps=track,
                    camera_fps_5s=rate_fps[0], det_fps_5s=rate_fps[1], track_fps_5s=rate_fps[2],
                    output_interval_p95_ms=_p95(intervals), queue_drops_5s=drops,
                    camera_p95_ms=_p95(c_ms), det_p95_ms=_p95(d_ms), track_p95_ms=_p95(t_ms),
                    camera_count=len(c_ms), det_count=len(d_ms), sample_count=len(t_ms), goal_ms=33,
                    frame_schedule_ms=schedule, frame_schedule_npu=[list(row[3]) for row in frames])
