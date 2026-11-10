#!/usr/bin/env python3
"""v25: checks the --controller / --ki / --kd options of gimbal_bridge without a gimbal (pseudo terminal, nothing moves).
Usage: python3 test_pid_options.py [path of gimbal_bridge]   (test_gimbal_bridge.py must be next to this file)"""
import subprocess, sys
sys.argv = sys.argv[:2]
import test_gimbal_bridge as t

def run(*options, x=371, seconds=2.5):
    """one person held at a steady offset right of the center; returns the speeds sent while following"""
    r = t.Rig(*options)
    try:
        assert r.serial(1.0) == t.STOP
        r.feed(.1, (7, x, 240)); r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
        got = [c for c in r.feed(seconds, (7, x, 240)) if c['T'] == 141]
        assert got and all(c['X'] == 1 for c in got), got
        return [c['SPD'] for c in got]
    finally:
        r.close()

p = run()
p2 = run('--controller', 'p', '--ki', '20')            # P selected: the I gain must be ignored
pi = run('--controller', 'pi', '--ki', '20')
pid = run('--controller', 'pid', '--ki', '20', '--kd', '2')
pi_lim = run('--controller', 'pi', '--ki', '20', '--integral-limit', '2')
print('P        last', p[-1], 'max', max(p))
print('P (+ki)  last', p2[-1], 'max', max(p2))
print('PI       last', pi[-1], 'max', max(pi))
print('PID      last', pid[-1], 'max', max(pid))
print('PI lim 2 last', pi_lim[-1], 'max', max(pi_lim))
assert max(p) == max(p2), 'P must ignore ki'
assert pi[-1] > p[-1], 'the I term must raise the speed at a steady offset'
assert p[-1] < pi_lim[-1] <= p[-1] + 2 + 1, 'the integral limit must cap the I part'
assert all(1 <= s <= 30 for s in p + pi + pid + pi_lim), 'speed limits'
for bad in (['--controller', 'xyz'], ['--ki', '-1'], ['--kd', '-1'], ['--derivative-filter-ms', '0']):
    q = subprocess.run([t.BRIDGE, '--device', '/dev/null', *bad], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    assert q.returncode != 0, bad
print('ALL OK')
