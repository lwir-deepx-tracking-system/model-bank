"""Smoke-test and compare team C++ vs the current Python tracker on an exact-input cache."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .bytetrack import ByteTrackTracker, NativeByteTrack


def smoke(library):
    with NativeByteTrack(library) as tracker, NativeByteTrack(library) as independent:
        det = [[90, 10, 91, 11, .04, 3], [20, 30, 40, 80, .95, 0], [20, 30, 40, 80, .8, 2]]
        first = {t[7]: t for t in tracker.update(det)}
        assert set(first) == {1, 2}, first
        assert first[1][6] == 0 and first[2][6] == 2
        assert len({t[4] for t in first.values()}) == 2
        again = tracker.update(det[::-1])
        assert {t[7] for t in again} == {0, 1}
        assert all(t[6] == det[::-1][t[7]][5] for t in again)
        assert not tracker.update([])
        assert not tracker.update(np.empty((0, 6)))
        assert independent.update(det) == list(first.values())
        tracker.reset()
        assert tracker.update(det) == list(first.values())
        try:
            tracker.update([[0, 1, 2, 3, float("nan"), 0]])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid detector input was accepted")


def compare(library, cache):
    totals = Counter()
    native_ms, python_ms = [], []
    with NativeByteTrack(library) as native, cache.open(encoding="utf-8") as stream:
        python = None
        for line in stream:
            frame = json.loads(line)
            if frame["tracker_reset"] or python is None:
                native.reset()
                python = ByteTrackTracker("bytetrack")
                totals["reset"] += 1
            arr = np.asarray(frame["tracker_input_xyxy_conf_class"], np.float32).reshape(-1, 6)
            begin = time.perf_counter_ns()
            a = native.update(arr)
            middle = time.perf_counter_ns()
            b = python.update(arr)
            finish = time.perf_counter_ns()
            native_ms.append((middle - begin) / 1e6)
            python_ms.append((finish - middle) / 1e6)
            slots_a, slots_b = {r[7] for r in a}, {r[7] for r in b}
            assert len(slots_a) == len(a), "two C++ tracks reference the same detection row"
            assert len({r[4] for r in a}) == len(a), "duplicate C++ track ID"
            assert all(0 <= r[7] < len(arr) and np.isfinite(r[:4]).all()
                       and r[6] == arr[r[7], 5] and abs(r[5] - arr[r[7], 4]) < 1e-6 for r in a)
            totals["frames"] += 1
            totals["native_rows"] += len(a)
            totals["python_rows"] += len(b)
            totals["same_detection_set_frames"] += slots_a == slots_b
            totals["native_only_rows"] += len(slots_a - slots_b)
            totals["python_only_rows"] += len(slots_b - slots_a)
            totals["native_person_rows"] += sum(r[6] == 0 for r in a)
            totals["python_person_rows"] += sum(r[6] == 0 for r in b)
    return dict(totals, native_p95_ms=float(np.percentile(native_ms, 95)),
                python_p95_ms=float(np.percentile(python_ms, 95)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    ap.add_argument("--cache", type=Path)
    args = ap.parse_args()
    smoke(args.library)
    print("PASS: exact detection row, duplicate boxes, independent instances, empty frames and reset")
    if args.cache:
        print(json.dumps(compare(args.library, args.cache), indent=2))
