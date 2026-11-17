"""Selectable RAW16 to detector-gray mapping for the live server."""
import sys
import threading
import time

import numpy as np

MODES = ('pct', 'fixed', 'clahe', 'fieldscale2-video')
FIELD = 'fieldscale2-video'
FRAME_SHAPE = (480, 640)
REQUIREMENTS = 'Python 3.10+, NumPy >= 1.23, OpenCV >= 4.5'


def _field_processor():
    if sys.version_info < (3, 10):
        raise RuntimeError(REQUIREMENTS)
    import cv2
    if (np.lib.NumpyVersion(np.__version__) < '1.23.0'
            or tuple(map(int, cv2.__version__.split('.')[:2])) < (4, 5)):
        raise RuntimeError(REQUIREMENTS)
    from ..vendor.fieldscale2 import MultiKnotPercentileFieldscale
    return MultiKnotPercentileFieldscale(video=True)


class ThermalMapping:
    def __init__(self, mode='pct'):
        self._lock = threading.Lock()
        self.mode = 'pct'
        self._field = None
        self._clahe = None
        self._source = None
        self._time = None
        self._position = None
        self._output = None
        self.frames = self.resets = 0
        self.last_ms = self.error = None
        self._status = {}
        self.set_mode(mode)
        self._publish_status()

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError('unknown thermal mapping mode')
        field = _field_processor() if mode == FIELD and self._field is None else None
        with self._lock:
            if mode == self.mode:
                return
            if field is not None:
                self._field = field
            if mode == 'clahe' and self._clahe is None:
                import cv2
                self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.mode = mode
            self._reset()
            self.error = None
            self._publish_status()

    def _reset(self):
        if self._field is not None:
            self._field.reset_temporal_state()
        self._source = self._time = self._position = self._output = None
        self.resets += 1

    def status(self):
        return self._status.copy()

    def _publish_status(self):
        self._status = dict(mode=self.mode, frames=self.frames, resets=self.resets,
                            last_ms=self.last_ms, error=self.error,
                            fieldscale2_requirement=REQUIREMENTS)

    def process(self, raw, fallback, source):
        if raw.shape != FRAME_SHAPE or raw.dtype != np.uint16:
            raise ValueError('thermal mapping requires 640x480 uint16 grayscale')
        producer, replay_pass, position, repeat, time_in = source
        with self._lock:
            mode = self.mode
            if mode != FIELD:
                start = time.perf_counter()
                gray = fallback(raw, 'fixed' if mode == 'fixed' else 'pct')
                if mode == 'clahe':
                    gray = self._clahe.apply(gray)
                self.last_ms = round((time.perf_counter() - start) * 1000, 2)
                self.frames += 1
                self._publish_status()
                return gray
            try:
                source_key = (producer, replay_pass)
                gap = replay_pass is None and self._time is not None and time_in - self._time > 1.0
                rewind = (replay_pass is not None and self._position is not None
                          and position is not None and position < self._position)
                if source_key != self._source or gap or rewind:
                    self._reset()
                    self._source = source_key
                if repeat and self._output is not None and position == self._position:
                    return self._output.copy()
                start = time.perf_counter()
                gray = self._field.process(raw)
                if gray.shape != FRAME_SHAPE or gray.dtype != np.uint8:
                    raise ValueError('FieldScale returned an invalid frame')
                self.last_ms = round((time.perf_counter() - start) * 1000, 2)
                self._time = time_in
                self._position = position
                self._output = gray.copy() if replay_pass is not None else None
                self.frames += 1
                self._publish_status()
                return gray
            except Exception as exc:
                self.error = '%s: %s' % (type(exc).__name__, str(exc)[:160])
                self._publish_status()
                raise RuntimeError('FieldScale preprocessing failed: ' + self.error) from exc


def queued_frame(mapper, time_in, raw, producer, *, replay_pass=None, position=None, repeat=False):
    if mapper is None or mapper.mode != FIELD:
        return time_in, raw
    return time_in, raw, (producer, replay_pass, position, repeat, time_in)


def detector_gray(mapper, fallback, queued, producer):
    if mapper is None:
        return fallback(queued[1])
    source = queued[2] if len(queued) == 3 else (producer, None, None, False, queued[0])
    return mapper.process(queued[1], fallback, source)
