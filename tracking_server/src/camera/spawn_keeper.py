"""Start the camera program from a thread that lives as long as the server (2026-11-15).

Why: cam_pipe asks the kernel to send it SIGTERM when its parent ends (prctl PR_SET_PDEATHSIG, so the camera is closed
when the server dies). Linux ties that to the THREAD that started the program, not to the process. The server started
cam_pipe from short-lived threads in two places: the thread that switches from a recording back to the camera
(replay_link._switch) and the camera reset button's thread (camera_link.cam_reset_work). Each thread ended right after
the start, so the new cam_pipe got SIGTERM while it was opening the camera: it closed the camera again (exit code 0),
the server noticed 20 s later and opened the camera a second time. Seen on the board 11-15 22:44: two opens and two
shutter calibrations within 25 s, and 25 s without a picture after 'back to the camera'. In camera_events.log every
'cam_pipe ended with code 0 after 20 s without a frame' since 11-12 follows such a start.

How: popen(make, ...) calls make(...) in the caller when the caller is the main thread (the frame loop: it lives as
long as the server, and that is the path of the server start and of the automatic restarts), otherwise in one keeper
thread that never ends, and hands the result (or the exception) back. Only the start of the process moves; everything
else of lv_spawn runs where it ran before.
"""
import queue
import threading

_LOCK = threading.Lock()
_KEEPER = dict(thread=None, jobs=None, calls=0)


def _run(jobs):
    while True:
        make, args, kwargs, answer, done = jobs.get()
        try:
            answer.append((True, make(*args, **kwargs)))
        except BaseException as exc:   # handed to the caller; the keeper itself must never end
            answer.append((False, exc))
        done.set()


def _keeper_jobs():
    with _LOCK:
        thread = _KEEPER['thread']
        if thread is None or not thread.is_alive():
            _KEEPER['jobs'] = queue.Queue()
            thread = threading.Thread(target=_run, args=(_KEEPER['jobs'],), name='lwir-camera-start', daemon=True)
            thread.start()
            _KEEPER['thread'] = thread
        _KEEPER['calls'] += 1
        return _KEEPER['jobs']


def popen(make, *args, **kwargs):
    """make(*args, **kwargs) (a subprocess.Popen call) run by a thread that outlives the caller. -> its result."""
    if threading.current_thread() is threading.main_thread():
        return make(*args, **kwargs)
    answer, done = [], threading.Event()
    _keeper_jobs().put((make, args, kwargs, answer, done))
    done.wait()
    ok, value = answer[0]
    if ok:
        return value
    raise value
