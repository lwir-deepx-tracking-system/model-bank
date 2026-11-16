#!/usr/bin/env python3
"""v27 --search end to end through a pseudo terminal (nothing moves): option parsing, SEARCH lines, serial output after a loss.
Usage: python3 test_v27_search.py <path of gimbal_bridge_v27>   (test_gimbal_bridge.py must be next to this file)"""
import subprocess, sys
sys.argv = sys.argv[:2]
import test_gimbal_bridge as t

FAST = ('--search', '1', '--search-grace-ms', '100', '--search-follow-ms', '200', '--search-speed', '3',
        '--search-sweep-px', '40', '--search-max-px', '60', '--search-timeout-ms', '2500', '--reacquire-ms', '4000')


def rig(*options, obs=True, vel=''):
    """obs: frames carry OBS 1 (a validated live frame, as the server sends them); the search moves only on those."""
    r = t.Rig('--deadband', '0.06', '--speed-max', '30', '--speed-rate', '1000', *options)
    assert r.serial(1.0) == t.STOP, 'the first command must be STOP'
    if obs:
        def frame(*people, age_ms=0, r=r):
            r.frame_id += 1
            ps = ' '.join('%d %.1f %.1f 40 80 0.900' % (i, x - 20, y - 40) for i, x, y in people)
            r.send('F %d %d %d %s OBS 1%s' % (r.frame_id, t.time.monotonic_ns() - age_ms * 1000000, len(people), ps,
                                               vel if any(i == 7 for i, _, _ in people) else ''))
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


def follow_and_lose(r):
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    moves = [c for c in r.feed(.4, (7, 600, 240)) if c['T'] == 141]
    assert moves and moves[-1]['X'] == 1, moves
    return r.feed(3.0)   # nobody in the picture, heartbeats go on


def main():
    r = rig()
    try:
        cmds = follow_and_lose(r)
        assert cmds and cmds[0] == t.STOP and all(c == t.STOP for c in cmds), cmds
    finally:
        r.close()
    print('ok   default: a loss STOPs and nothing moves again (v25 rule)')

    r = rig(*FAST)
    try:
        first = lines_until(r, 'SEARCH_CONTROL', 1.0)
        assert 'SEARCH_CONTROL 1' in first, first
        cmds = follow_and_lose(r)
        moves = [c for c in cmds if c['T'] == 141]
        assert cmds[0] == t.STOP, cmds
        dirs = []
        for c in moves:
            if not dirs or dirs[-1] != c['X']: dirs.append(c['X'])
        assert dirs[:3] == [1, -1, 1] and all(abs(c['SPD']) <= 10 and c['Y'] == 0 for c in moves), cmds
        assert cmds[-1] == t.STOP, cmds
        seen = []
        while True:
            a = r.line(.05)
            if a is None: break
            if a.startswith('SEARCH '): seen.append(int(a.split()[1]))
        assert seen and seen[-1] == 4 and 3 in seen, seen
    finally:
        r.close()
    print('ok   --search 1: STOP, follow-through right, sweep left/right, STOP and hold; SEARCH lines')

    r = rig(*FAST, vel=' VEL 7 -250 0')   # following right, but the predictive velocity says the person walks left
    try:
        cmds = follow_and_lose(r)
        moves = [c for c in cmds if c['T'] == 141]
        assert cmds[0] == t.STOP and moves and moves[0]['X'] == -1 and max(c['SPD'] for c in moves[:3]) <= 8, cmds
    finally:
        r.close()
    print('ok   --search 1 with VEL: the follow-through goes where the predictive velocity says')

    r = rig(*FAST, obs=False)
    try:
        cmds = follow_and_lose(r)
        assert cmds and all(c == t.STOP for c in cmds), cmds
    finally:
        r.close()
    print('ok   --search 1 without validated frames (no OBS 1): STOP and no search motion')

    for bad in (('--search', '2'), ('--search', '1', '--search-speed', '0'), ('--search', '1', '--search-widen', '9'),
                ('--search', '1', '--search-timeout-ms', '100', '--search-grace-ms', '300')):
        p = subprocess.run([t.BRIDGE, '--device', '/dev/null', *bad], stdin=subprocess.DEVNULL, capture_output=True, timeout=5)
        assert p.returncode != 0, bad
    print('ok   bad --search settings are refused')


if __name__ == '__main__':
    main()
