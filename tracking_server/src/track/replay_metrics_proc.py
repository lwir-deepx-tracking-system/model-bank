"""replay_metrics.compute in a process of its own (2026-11-15).

Why: the server called replay_metrics.compute in a thread (HOTA / AP at the end of a full pass, and the scores up to a
slider position). That is Python work, so the thread held the interpreter lock in turn with the frame loop: on the board
the frame loop fell from 25 to 6 frames a second and the picture came 0.9 s late for about 20 s after a recording ended
(stall log, 11-15 22:38: 'GIL_STALL ... replay-metrics ... replay_metrics.py _ap_for_iou'). A recording started in that
time played slowly. A child process has its own interpreter lock, so the frame loop does not wait for it.

How: compute(frames) starts `python -B replay_metrics_proc.py`, sends the frames (pickle) on its standard input and reads
the result (pickle) from its standard output. The child loads replay_metrics.py from this folder and nothing else of the
server (no camera, no NPU, no page). The numbers are the same objects the thread would have made (same code, same
input). If the child cannot be started, fails, answers something else or takes longer than TIMEOUT_S, the result is
computed here, in the calling thread, as before (one line is printed).

Off switch: LWIR_REPLAY_METRICS_PROC=0 computes in the calling thread, as before 11-15.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time

TIMEOUT_S = 300.0
KEYS = ('hota', 'ap50', 'map')
STATS = dict(child=0, inline=0, failed=0, last_s=None, last_why='')


def enabled() -> bool:
    return os.environ.get('LWIR_REPLAY_METRICS_PROC', '1') != '0'


def _inline(frames):
    from . import replay_metrics
    return replay_metrics.compute(frames)


def _plain(frames):
    """The frames as lists and tuples of Python numbers, as replay_metrics.compute reads them."""
    return [(list(gt), list(tracks), None if detections is None else list(detections))
            for gt, tracks, detections in frames]


def _child_result(frames, timeout):
    data = pickle.dumps(_plain(frames), protocol=4)
    done = subprocess.run([sys.executable, '-B', os.path.abspath(__file__)], input=data, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError('exit %d: %s' % (done.returncode, done.stderr.decode('utf-8', 'replace').strip()[-200:]))
    kind, value = pickle.loads(done.stdout)
    if kind == 'error':
        # The same input fails the same way in this process: let the caller see the real exception.
        return None
    if kind != 'ok' or not isinstance(value, dict) or tuple(sorted(value)) != tuple(sorted(KEYS)):
        raise RuntimeError('unexpected answer')
    if any(v is not None and not isinstance(v, float) for v in value.values()):
        raise RuntimeError('unexpected value')
    return value


def compute(frames, *, timeout=TIMEOUT_S):
    """Same result as replay_metrics.compute(frames); the work is done in a child process when possible."""
    if not enabled():
        STATS['inline'] += 1
        return _inline(frames)
    started = time.perf_counter()
    try:
        value = _child_result(frames, timeout)
    except Exception as exc:   # no child, a killed child, a timeout: the old way
        STATS['failed'] += 1
        STATS['last_why'] = '%s: %s' % (type(exc).__name__, str(exc)[:160])
        print('REPLAY METRICS child process failed (%s); computing in the server' % STATS['last_why'], flush=True)
        value = None
    if value is None:
        STATS['inline'] += 1
        return _inline(frames)
    STATS['child'] += 1
    STATS['last_s'] = round(time.perf_counter() - started, 3)
    return value


def _child_main():
    here = os.path.dirname(os.path.abspath(__file__))
    # Only replay_metrics.py is wanted from this folder: its other files (sort.py, kalman.py ...) must not be found by name.
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != here]
    try:
        os.nice(10)   # the frame loop comes first
    except (AttributeError, OSError):
        pass
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    frames = pickle.loads(stdin.read())
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('lwir_replay_metrics', os.path.join(here, 'replay_metrics.py'))
        replay_metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay_metrics)
        answer = ('ok', replay_metrics.compute(frames))
    except Exception as exc:
        answer = ('error', '%s: %s' % (type(exc).__name__, str(exc)[:200]))
    stdout.write(pickle.dumps(answer, protocol=4))
    stdout.flush()


if __name__ == '__main__':
    _child_main()
