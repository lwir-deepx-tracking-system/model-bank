#!/usr/bin/env python3
"""v27 --seek end to end through a pseudo terminal (nothing moves): option parsing, SEEK_CONTROL and SEARCH lines, serial
output after a loss with and without the server's SEEK suffix (speed cap, turn through STOP, time limit).
Usage: python3 test_v27_seek.py <path of gimbal_bridge_v27>   (test_gimbal_bridge.py must be next to this file)"""
import subprocess, sys
sys.argv = sys.argv[:2]
import test_gimbal_bridge as t


def rig(*options, obs=True, seek=''):
    """obs: frames carry OBS 1 (validated live frames); seek: the suffix of the frames without id 7 (a list entry per call
    of feed is not needed: r.suffix can be changed between feeds)."""
    r = t.Rig('--deadband', '0.06', '--speed-max', '30', '--speed-rate', '1000', '--reacquire-ms', '4000', *options)
    assert r.serial(1.0) == t.STOP, 'the first command must be STOP'
    r.suffix = seek

    def frame(*people, age_ms=0, r=r):
        r.frame_id += 1
        ps = ' '.join('%d %.1f %.1f 40 80 0.900' % (i, x - 20, y - 40) for i, x, y in people)
        seen = any(i == 7 for i, _, _ in people)
        r.send('F %d %d %d %s%s%s' % (r.frame_id, t.time.monotonic_ns() - age_ms * 1000000, len(people), ps,
                                      ' OBS 1' if obs else '', '' if seen else r.suffix))
    r.frame = frame
    return r


def lines_for(r, seconds):
    got, end = [], t.time.monotonic() + seconds
    while t.time.monotonic() < end:
        a = r.line(.05)
        if a is not None: got.append(a)
    return got


def follow_and_lose(r, after=0.6):
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    moves = [c for c in r.feed(.4, (7, 600, 240)) if c['T'] == 141]
    assert moves and moves[-1]['X'] == 1, moves
    return r.feed(after)   # nobody in the picture, heartbeats go on


RIGHT, LEFT = ' SEEK 7 590 200 40 80', ' SEEK 7 10 200 40 80'   # box centres 610 and 30: 290 px right / left of the centre


def main():
    r = rig(seek=RIGHT)
    try:
        cmds = follow_and_lose(r)
        assert cmds and cmds[0] == t.STOP and all(c == t.STOP for c in cmds), cmds
    finally:
        r.close()
    print('ok   default: SEEK is accepted but ignored, a loss STOPs (v25 rule)')

    r = rig('--seek', '1', '--seek-ms', '500', '--seek-speed', '6', seek=RIGHT)
    try:
        first = lines_for(r, 0.5)
        assert 'SEEK_CONTROL 1' in first, first
        cmds = follow_and_lose(r, 1.0)
        moves = [c for c in cmds if c['T'] == 141]
        assert moves and all(c['X'] == 1 and c['Y'] == 0 for c in moves), cmds
        assert moves[-1]['SPD'] == 6 and max(c['SPD'] for c in moves) <= 6, moves   # the PD demand is 27; the move under way (SPD 26) is cut to the cap at once
        assert cmds[-1] == t.STOP, cmds
    finally:
        r.close()
    print('ok   --seek 1 with SEEK: turning toward the SEEK box at no more than --seek-speed, STOP after --seek-ms')

    r = rig('--seek', '1', '--seek-ms', '3000', '--seek-speed', '8', seek=RIGHT)
    try:
        follow_and_lose(r, 0.3)
        r.suffix = LEFT   # the server's next turn
        cmds = r.feed(0.4)
        assert cmds and cmds[0] == t.STOP, cmds
        moves = [c for c in cmds if c['T'] == 141]
        assert moves and all(c['X'] == -1 for c in moves) and moves[0]['SPD'] == 1 and max(c['SPD'] for c in moves) <= 8, cmds
        r.suffix = ''     # the server ended the search
        cmds = r.feed(0.3)
        assert cmds and cmds[0] == t.STOP and all(c == t.STOP for c in cmds), cmds
        r.suffix = RIGHT
        r.feed(0.2)
        cmds = r.feed(0.3, (7, 330, 240))   # the same id is back at the centre: following again (inside the deadband: STOP)
        assert cmds and cmds[-1] == t.STOP, cmds
    finally:
        r.close()
    print('ok   --seek 1: the other side goes through STOP and starts at SPD 1; no SEEK = STOP at once; the id back ends it')

    r = rig('--seek', '1', '--seek-ms', '3000')
    try:
        cmds = follow_and_lose(r)
        assert cmds and cmds[0] == t.STOP and all(c == t.STOP for c in cmds), cmds
    finally:
        r.close()
    print('ok   --seek 1 without SEEK: STOP at once')

    r = rig('--seek', '1', '--seek-ms', '3000', obs=False, seek=RIGHT)
    try:
        cmds = follow_and_lose(r)
        assert cmds and cmds[0] == t.STOP and all(c == t.STOP for c in cmds), cmds
    finally:
        r.close()
    print('ok   --seek 1 without validated frames (no OBS 1): STOP at once')

    r = rig('--seek', '1', '--seek-ms', '3000', seek=' SEEK 7 590 200 40')
    try:
        cmds = follow_and_lose(r)
        assert cmds and cmds[0] == t.STOP and all(c == t.STOP for c in cmds), cmds
    finally:
        r.close()
    print('ok   --seek 1 with a damaged SEEK suffix: the frame is not trusted, STOP at once')

    for bad in (('--seek', '2'), ('--seek', '1', '--seek-ms', '0'), ('--seek', '1', '--seek-ms', '20000'), ('--seek', '1', '--seek-speed', '0'),
                ('--seek', '1', '--seek-speed', '31'), ('--seek', '1', '--search', '1')):
        p = subprocess.run([t.BRIDGE, '--device', '/dev/null', '--speed-max', '30', *bad], stdin=subprocess.DEVNULL, capture_output=True, timeout=5)
        assert p.returncode != 0, bad
    print('ok   bad --seek settings (and --seek with --search) are refused')


if __name__ == '__main__':
    main()
