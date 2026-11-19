"""Schedule full cyclic collections around live input, with bounded deferral."""

import gc
import os
import sys
import threading
import time


_MIB = 1024 * 1024
_SAMPLE_INTERVAL = 2.0
_WARMUP_FRAMES = 300
_FULL_THRESHOLD = (1 << 31) - 1
_MAX_DEFER_SECONDS = 600.0
_RSS_GROWTH_LIMIT = 64 * _MIB
_RSS_LIMIT = 1024 * _MIB


def _rss():
    with open('/proc/self/statm', 'rb') as stats:
        return int(stats.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def for_live():
    if (sys.platform != 'linux' or sys.implementation.name != 'cpython'
            or sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12))
            or os.environ.get('LWIR_LIVE_GC_DEFER', '1') != '1'):
        return None
    if not gc.isenabled() or gc.get_threshold()[0] <= 0:
        return None
    return LiveGC()


class LiveGC:
    def __init__(self):
        self._lock = threading.Lock()
        self._defaults = gc.get_threshold()
        self._slow = (*self._defaults[:2], _FULL_THRESHOLD)
        self._base_rss = None
        self._start = None
        self._next_sample = 0.0
        self._safe = 0
        self._due = False
        self._fallbacks = 0
        self._rss_peak = 0
        self._last_ms = None
        self._reason = 'warmup'
        self._closed = False

    def frame(self, frame_index):
        """Check the memory and time limits outside the published frame."""
        if frame_index < _WARMUP_FRAMES or self._closed:
            return
        now = time.monotonic()
        if now < self._next_sample:
            return
        with self._lock:
            if self._closed or now < self._next_sample:
                return
            self._next_sample = now + _SAMPLE_INTERVAL
            if self._reason not in ('warmup', 'deferring'):
                return
            rss = self._sample_rss()
            if self._reason == 'warmup':
                self._arm(now, rss)
                return
            if rss is None:
                self._fallback('rss_unavailable')
            elif rss >= _RSS_LIMIT:
                self._fallback('rss_limit')
            elif rss - self._base_rss >= _RSS_GROWTH_LIMIT:
                self._fallback('rss_growth')
            elif now - self._start >= _MAX_DEFER_SECONDS:
                # A healthy 30 FPS source has no guaranteed window for a full
                # stop-the-world scan. Keep the RSS guards armed while waiting
                # for a source boundary, which is the next safe collection point.
                self._due = True

    def between_sources(self):
        """Collect after old input has ended and before the next producer starts."""
        with self._lock:
            if self._closed or self._reason == 'warmup':
                return
            self._last_ms = self._collect()
            self._safe += 1
            self._arm(time.monotonic(), self._sample_rss())

    def snapshot(self):
        # Collection and callbacks may reenter Python, so the health check uses
        # read-only counters without waiting for the maintenance lock.
        return dict(mode=self._reason, active=(gc.get_threshold() == self._slow),
                    full_collections=gc.get_stats()[2]['collections'],
                    between_sources=self._safe, fallbacks=self._fallbacks, maintenance_due=self._due,
                    peak_rss_mb=round(self._rss_peak / _MIB, 1),
                    last_maintenance_ms=self._last_ms)

    def close(self):
        with self._lock:
            self._closed = True
            if gc.get_threshold() == self._slow:
                gc.set_threshold(*self._defaults)
            self._reason = 'closed'

    def _sample_rss(self):
        try:
            rss = _rss()
        except (OSError, ValueError, IndexError):
            return None
        self._rss_peak = max(self._rss_peak, rss)
        return rss

    def _arm(self, now, rss):
        if rss is None or rss >= _RSS_LIMIT:
            if gc.get_threshold() == self._slow:
                gc.set_threshold(*self._defaults)
            self._reason = 'rss_unavailable' if rss is None else 'rss_limit'
            return
        self._base_rss = rss
        self._due = False
        self._start = now
        gc.set_threshold(*self._slow)
        self._reason = 'deferring'

    def _fallback(self, reason):
        gc.set_threshold(*self._defaults)
        self._reason = reason
        self._due = False
        self._fallbacks += 1
        self._last_ms = self._collect()

    @staticmethod
    def _collect():
        start = time.perf_counter()
        gc.collect(2)
        return round((time.perf_counter() - start) * 1000, 3)
