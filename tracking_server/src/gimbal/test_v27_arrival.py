#!/usr/bin/env python3
"""v27 arrival control, end to end through gimbal_bridge_v27 without a gimbal (pseudo terminal, nothing moves):
the time from a people line on stdin to the gimbal command on the serial line with --arrival-control 1 (the default of
gimbal_bridge_v27) and 0 (wait for the 25 ms period, as v25), the SPD ramp in real time with both, and the option values.
Usage: python3 test_v27_arrival.py <path of gimbal_bridge_v27>   (test_gimbal_bridge.py must be next to this file)
Prints the measured times; exit code 0 only when every check passes."""
import statistics, subprocess, sys, time
sys.argv = sys.argv[:2]
import test_gimbal_bridge as t

FRAME = 0.0334   # a tracking result every 33.4 ms, like the camera: every phase of the 25 ms period is met
failures = []


def check(ok, text):
    print(('ok   ' if ok else 'FAIL ') + text, flush=True)
    if not ok: failures.append(text)


def rig(*options):
    r = t.Rig('--deadband', '0.06', '--speed-max', '100', *options)
    assert r.serial(1.0) == t.STOP, 'the first command must be STOP'
    return r


def select(r, at, drain=True):
    for _ in range(3):
        r.send('P'); r.frame((7, *at)); time.sleep(FRAME)
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    while drain and r.serial(.005) is not None: pass


def latency(*options, frames=120):
    """A person who is alternately 100 and 200 px right of the centre (P control: SPD 9 and 19), and a ramp fast enough that
    every result changes the command. -> ms from writing the people line to reading the matching command, one per frame."""
    r = rig('--speed-rate', '100000', *options)
    try:
        select(r, (420, 240))
        got = []; start = time.monotonic() + .05
        for i in range(frames):
            due = start + i * FRAME
            while time.monotonic() < due: time.sleep(.0005)
            if i % 6 == 0: r.send('P')
            x, want = (520, 19) if i % 2 else (420, 9)
            sent = time.monotonic()
            r.frame((7, x, 240))
            while True:
                c = r.serial(max(sent + .031 - time.monotonic(), 0))
                if c is None: break
                if c.get('T') == 141 and c.get('SPD') == want:
                    got.append((time.monotonic() - sent) * 1e3); break
        return got
    finally:
        r.close()


def ramp(*options):
    """Follow a person 280 px right of the centre (demand SPD 26) with --speed-rate 150. -> [(ms since the first move, SPD)]"""
    r = rig('--speed-rate', '150', *options)
    try:
        select(r, (600, 240), drain=False)   # nothing moves before the selection: the first command read below is the first move
        got = []; first = None; start = time.monotonic(); i = 0
        while time.monotonic() - start < .5:
            due = start + i * FRAME; i += 1
            if i % 6 == 0: r.send('P')
            r.frame((7, 600, 240))
            while True:
                c = r.serial(max(due + FRAME - time.monotonic(), 0))
                if c is None: break
                now = time.monotonic()
                if c.get('T') != 141: continue
                if first is None: first = now
                got.append(((now - first) * 1e3, c['SPD']))
        return got
    finally:
        r.close()


def main():
    print('bridge', t.BRIDGE, flush=True)
    on, off, default = latency('--arrival-control', '1'), latency('--arrival-control', '0'), latency()
    for name, d in (('--arrival-control 1', on), ('--arrival-control 0', off), ('no option', default)):
        d = sorted(d) or [float('nan')]
        print('     people line -> serial command, %-19s: %3d of 120 frames, mean %5.2f ms, median %5.2f, 90 %% %5.2f, worst %5.2f'
              % (name, len(d), statistics.mean(d), d[len(d) // 2], d[len(d) * 9 // 10], d[-1]), flush=True)
    check(len(on) >= 110 and len(off) >= 110 and len(default) >= 110, 'every result changes the gimbal command (with and without the option)')
    if on and off and default:
        check(statistics.median(on) < 2.0 and sorted(on)[len(on) * 9 // 10] < 3.0, '--arrival-control 1: the command is on the serial line within 2 ms of the people line (median), 3 ms (90 %)')
        check(6.0 < statistics.mean(off) < 19.0 and max(off) < 30.0, '--arrival-control 0: the command waits for the 25 ms period (mean 6..19 ms)')
        check(statistics.median(default) < 2.0, 'no option: gimbal_bridge_v27 acts on arrival (the default)')

    for name, options in (('--arrival-control 1', ()), ('--arrival-control 0', ('--arrival-control', '0'))):
        moves = ramp(*options)
        top = [ms for ms, spd in moves if spd >= 26]
        # 1 + 150 SPD/s x time, capped at the demand 26.25; 1.5 SPD = rounding + 5 ms of reading jitter
        law = bool(moves) and moves[0][1] == 1 and all(abs(spd - min(26.25, 1 + .150 * ms)) <= 1.5 for ms, spd in moves)
        print('     ramp, %s: %d commands, SPD 26 after %s ms (163 ms + one step): %s'
              % (name, len(moves), '%.0f' % top[0] if top else '-', ' '.join('%d' % spd for _, spd in moves)), flush=True)
        check(law, '%s: the ramp is 150 SPD per second of real time from SPD 1' % name)
        check(bool(top) and 150 <= top[0] <= 215, '%s: the demand SPD 26 is reached 150..215 ms after the first move' % name)

    for option in ('--arrival-control', '--gui-video-always'):
        for value in ('2', 'yes', '-1'):
            result = subprocess.run([t.BRIDGE, '--device', '/dev/null', option, value], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=3)
            check(result.returncode != 0 and option.lstrip('-') in result.stderr, '%s %s is refused' % (option, value))
    print('FAILED %d' % len(failures) if failures else 'ALL OK', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
