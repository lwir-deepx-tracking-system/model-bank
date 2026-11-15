#!/usr/bin/env python3
"""v27 --coast end to end through a pseudo terminal (nothing moves): option parsing, COAST_CONTROL line, serial output after a
loss with and without the server's PRED suffix.
Usage: python3 test_v27_coast.py <path of gimbal_bridge_v27>   (test_gimbal_bridge.py must be next to this file)"""
import subprocess, sys
sys.argv = sys.argv[:2]
import test_gimbal_bridge as t


def rig(*options, obs=True, pred=''):
    """obs: frames carry OBS 1 (validated live frames); pred: the PRED suffix of the frames without id 7."""
    r = t.Rig('--deadband', '0.06', '--speed-max', '30', '--speed-rate', '1000', '--reacquire-ms', '4000', *options)
    assert r.serial(1.0) == t.STOP, 'the first command must be STOP'

    def frame(*people, age_ms=0, r=r):
        r.frame_id += 1
        ps = ' '.join('%d %.1f %.1f 40 80 0.900' % (i, x - 20, y - 40) for i, x, y in people)
        seen = any(i == 7 for i, _, _ in people)
        r.send('F %d %d %d %s%s%s' % (r.frame_id, t.time.monotonic_ns() - age_ms * 1000000, len(people), ps,
                                      ' OBS 1' if obs else '', '' if seen else pred))
    r.frame = frame
    return r


def lines_until(r, prefix, seconds):
    got, end = [], t.time.monotonic() + seconds
    while t.time.monotonic() < end:
        a = r.line(.05)
        if a is not None:
            got.append(a)
            if a.startswith(prefix): break
    return got


def follow_and_lose(r, after=0.6):
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    moves = [c for c in r.feed(.4, (7, 600, 240)) if c['T'] == 141]
    assert moves and moves[-1]['X'] == 1, moves
    return r.feed(after)   # nobody in the picture, heartbeats go on


def main():
    r = rig(pred=' PRED 7 600 200 40 80')
    try:
        cmds = follow_and_lose(r)
        assert cmds and cmds[0] == t.STOP and all(c == t.STOP for c in cmds), cmds
    finally:
        r.close()
    print('ok   default: PRED is accepted but ignored, a loss STOPs (v25 rule)')

    r = rig('--coast', '1', '--coast-ms', '400', pred=' PRED 7 600 200 40 80')
    try:
        first = lines_until(r, 'COAST_CONTROL', 1.0)
        assert 'COAST_CONTROL 1' in first, first
        cmds = follow_and_lose(r, 0.9)
        assert t.STOP in cmds, cmds
        k = cmds.index(t.STOP)
        assert k > 0 and all(c['T'] == 141 and c['X'] == 1 for c in cmds[:k]), cmds
        assert all(c == t.STOP for c in cmds[k:]), cmds
    finally:
        r.close()
    print('ok   --coast 1 with PRED: no STOP at the loss, turning on toward the prediction, then one STOP and still')

    r = rig('--coast', '1', '--coast-ms', '400')
    try:
        cmds = follow_and_lose(r)
        assert cmds and cmds[0] == t.STOP and all(c == t.STOP for c in cmds), cmds
    finally:
        r.close()
    print('ok   --coast 1 without PRED: STOP at once')

    r = rig('--coast', '1', '--coast-ms', '400', obs=False, pred=' PRED 7 600 200 40 80')
    try:
        cmds = follow_and_lose(r)
        assert cmds and cmds[0] == t.STOP and all(c == t.STOP for c in cmds), cmds
    finally:
        r.close()
    print('ok   --coast 1 without validated frames (no OBS 1): STOP at once')

    for bad in (('--coast', '2'), ('--coast', '1', '--coast-ms', '0'), ('--coast', '1', '--coast-ms', '9000')):
        p = subprocess.run([t.BRIDGE, '--device', '/dev/null', *bad], stdin=subprocess.DEVNULL, capture_output=True, timeout=5)
        assert p.returncode != 0, bad
    print('ok   bad --coast settings are refused')


if __name__ == '__main__':
    main()
