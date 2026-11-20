"""Bounded, in-memory identity tracing. No I/O or control side effects."""
from collections import OrderedDict
import json
import os
import time
from threading import Lock
from typing import NamedTuple

FRAME_LIMIT = 180
ROW_LIMIT = 16
VALUE_LIMIT = 2**32-1
MAX_WINDOW_NS = 600_000_000_000
READ_GRACE_NS = 60_000_000_000
ACTUATOR_COMMANDS = ("STOP", "SPEED", "MANUAL_STOP", "MANUAL_SPEED")
ACTUATOR_RESULTS = ("serial_write_ok", "serial_write_error")


class Frame(NamedTuple):
    captured_ns: int
    person_detections: int
    owner_rows: tuple
    judge_rows: tuple
    bridge_seq: int
    selected_public: int
    observation_healthy: bool
    reported_auto: bool
    reported_lost: bool
    reported_moving: bool
    bridge_flags: int
    truncated: bool
    actuator: tuple = ()
    camera_key: int = 0


def small_int(x):
    return min(VALUE_LIMIT, max(0, int(x)))


def owner_row(row):
    raw, public, detection_index = row
    # -1 preserves the explicit "no observation this frame" sentinel.
    return small_int(raw), small_int(public), min(VALUE_LIMIT, max(-1, int(detection_index)))


def judge_row(row):
    birth, candidate, public_before, score_10000, eligibility_code, accepted = row
    return (small_int(birth), small_int(candidate), small_int(public_before),
            min(10000, max(0, int(score_10000))), small_int(eligibility_code), bool(accepted))


def bounded(rows, cast):
    result = []
    for row in rows:
        if len(result) == ROW_LIMIT:
            return tuple(result), True
        result.append(cast(row))
    return tuple(result), False


class IdentityTrace:
    def __init__(self, *, enabled=False, deadline_ns=None):
        self.enabled = bool(enabled)
        self._deadline_ns = deadline_ns
        self._frames = OrderedDict()
        self._group = None
        self._source = None
        self._epoch = 0
        self._bridge_watermark = 0
        self._lock = Lock()
        self.dropped = 0
        self.errors = 0

    @property
    def recording(self):
        return self.enabled and (self._deadline_ns is None or time.monotonic_ns() < self._deadline_ns)

    @property
    def readable(self):
        return self.enabled and (self._deadline_ns is None or time.monotonic_ns() < self._deadline_ns + READ_GRACE_NS)

    @classmethod
    def from_env(cls):
        if os.environ.get("LWIR_IDENTITY_TRACE", "0") != "1":
            return cls()
        text = os.environ.get("LWIR_IDENTITY_TRACE_DEADLINE_NS")
        if text is None:
            return cls(enabled=True)  # preserve the existing explicit diagnostic switch
        now = time.monotonic_ns()
        if not text.isascii() or not text.isdigit() or len(text) > 19:
            return cls()
        deadline = int(text)
        return cls(enabled=True, deadline_ns=deadline) if now < deadline <= now + MAX_WINDOW_NS else cls()

    def record(self, *, group, captured_ns, person_detections, owners, candidates,
               bridge_seq, selected_public, healthy, auto, lost, moving, flags, camera_key=0, bridge_namespace=None, reset=False):
        """Called on tracking thread, after the existing frame result is queued.

        group is an integer server/model/tracker epoch. No pixel data, floats,
        filenames, pose coordinates, features, free-form strings, or commands.
        """
        if not self.recording:
            return False
        if not self._lock.acquire(blocking=False):
            self.dropped += 1
            return False
        try:
            frame_time, seq = int(captured_ns), small_int(bridge_seq)
            if not (0 < frame_time <= 2**63-1 and seq):
                return False
            if bridge_namespace is not None:
                if frame_time <= self._bridge_watermark:
                    self.dropped += 1
                    return False
                self._bridge_watermark = frame_time
                if reset or self._source != bridge_namespace:
                    self._epoch += 1
                    self._source = bridge_namespace
                group = self._epoch
            group = int(group)
            if not 0 <= group <= 2**63-1:
                return False
            if self._group != group:
                self._frames.clear()
                self._group = group
            key = (seq, frame_time)
            if key in self._frames or (self._frames and next(reversed(self._frames))[1] >= frame_time):
                self.dropped += 1
                return False
            owner_data, a = bounded(owners, owner_row)
            judge_data, b = bounded(candidates, judge_row)
            self._frames[key] = Frame(
                frame_time, small_int(person_detections), owner_data, judge_data,
                seq, small_int(selected_public), bool(healthy), bool(auto), bool(lost),
                bool(moving), min(15,small_int(flags)), a or b, camera_key=int(camera_key))
            if len(self._frames) > FRAME_LIMIT:
                self._frames.popitem(last=False)
            return True
        except Exception:
            self.errors += 1
            return False
        finally:
            self._lock.release()

    def note_serial_write(self, *, group, bridge_seq, captured_ns, result,
                          kind, x, y):
        """An optional bridge receipt for the specified exact frame.

        Existing controller telemetry currently exposes none. An observation
        offer or the latest reported movement must never call this method.
        """
        if not self.recording or result not in ACTUATOR_RESULTS or kind not in ACTUATOR_COMMANDS:
            return False
        if not self._lock.acquire(blocking=False):
            self.dropped += 1
            return False
        try:
            key = (int(bridge_seq), int(captured_ns))
            frame = self._frames.get(key) if self._group == group else None
            if frame is None:
                return False
            write = (kind, result, max(-32768,min(32767,int(x))), max(-32768,min(32767,int(y))))
            if len(frame.actuator) >= 4:
                self.dropped += 1
                return False
            self._frames[key] = frame._replace(actuator=(*frame.actuator, write))
            return True
        except Exception:
            self.errors += 1
            return False
        finally:
            self._lock.release()

    def snapshot(self, limit=30, *, ceiling=60):
        if not self.readable or not self._lock.acquire(blocking=False):
            return ()
        try:
            cap = min(FRAME_LIMIT, ceiling, max(0, int(limit)))
            if cap == 0:
                return ()
            latest = tuple(self._frames.values())
            return latest[-cap:]
        except (TypeError, ValueError, OverflowError):
            return ()
        finally:
            self._lock.release()



def record_bridge_observation(details, source, sequence, state, captured_ns, selected, healthy, text):
    """Copy only allowlisted diagnostics after a normal observation was queued."""
    app_rows, detections, frame_key = details
    trace, raw, published, candidates, reset, detector = app_rows
    try:
        auto, lost, moving = state
        flags = sum(1 << i for i, flag in enumerate((" OBS ", " PRED ", " LEAD ", " SEEK ")) if flag in text)
        owners = ((a[4], b[4], a[7]) for a, b in zip(raw, published, strict=True) if int(a[6]) == 0)
        trace.record(group=0, captured_ns=captured_ns, camera_key=frame_key,
                     person_detections=sum(int(d[5]) == 0 for d in detections),
                     owners=owners, candidates=candidates, bridge_seq=sequence,
                     selected_public=selected, healthy=healthy, auto=auto, lost=lost,
                     moving=moving, flags=flags, bridge_namespace=(source, detector), reset=reset)
    except Exception:
        trace.errors += 1


def diagnostic_payload(trace, peer, host, limit=30):
    """Loopback-only, one bounded non-streaming response when opt-in is enabled."""
    loopback_hosts = {"127.0.0.1", "127.0.0.1:8090", "[::1]", "[::1]:8090", "localhost", "localhost:8090"}
    if trace is None or not trace.readable or peer not in ("127.0.0.1", "::1") or host not in loopback_hosts:
        return None
    rows = [row._asdict() for row in trace.snapshot(limit, ceiling=FRAME_LIMIT)]
    return json.dumps(dict(version=1, epoch=trace._group, dropped=trace.dropped,
                           errors=trace.errors, rows=rows),separators=(",",":")).encode()
