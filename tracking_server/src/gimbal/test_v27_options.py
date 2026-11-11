#!/usr/bin/env python3
"""v27: checks the motion-smoothing options of gimbal_bridge_v27 end to end without a gimbal (pseudo terminal, nothing moves):
--deadband-exit, --stop-rate, --axis-reversal, --reversal-join-speed, --reversal-hold-ms, and that leaving them out keeps the v25 behaviour.
Usage: python3 test_v27_options.py <path of gimbal_bridge_v27>   (test_gimbal_bridge.py must be next to this file)
Also run the v25 suite against v27 (all its cases must still pass): python3 test_gimbal_bridge.py <path of gimbal_bridge_v27>"""
import subprocess, sys
sys.argv = sys.argv[:2]
import test_gimbal_bridge as t


def follow(r, at, seconds=.6):
    r.feed(.1, (7, *at))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    moves = [c for c in r.feed(seconds, (7, *at)) if c['T'] == 141]
    assert moves, 'no move while following'
    return moves


def rig(*options):
    r = t.Rig('--deadband', '0.06', *options)
    assert r.serial(1.0) == t.STOP, 'the first command must be STOP'
    return r


def hysteresis(exit_option):
    """centre the target, then put it 25 px off the centre (error 0.078: outside deadband 0.06, inside exit 0.10). True when the gimbal moved again"""
    r = rig(*exit_option)
    try:
        follow(r, (600, 240), .3)
        assert t.STOP in r.feed(.3, (7, 322, 240)), 'no STOP at the centre'
        return any(c['T'] == 141 for c in r.feed(.4, (7, 345, 240)))
    finally:
        r.close()


def reversal(options):
    """follow right+up, then the target jumps below the centre (Y reverses, X keeps). Returns the commands after the jump up to the first STOP"""
    r = rig('--speed-max', '30', *options)
    try:
        moves = follow(r, (600, 60), .5)
        assert all(c['X'] == 1 and c['Y'] == 1 for c in moves), moves
        got = r.feed(.5, (7, 600, 440))
        while got and got[0] != t.STOP and got[0]['X'] == 1 and got[0]['Y'] == 1: got.pop(0)   # speed updates already on the line
        return got[:got.index(t.STOP) + 1] if t.STOP in got else got
    finally:
        r.close()


def deadband_entry(options):
    """follow far right until the speed has ramped, then centre the target. Returns the commands from the centring up to the first STOP"""
    r = rig('--speed-max', '30', *options)
    try:
        follow(r, (630, 240), 1.0)
        got = r.feed(.6, (7, 321, 240))
        assert t.STOP in got, got
        return got[:got.index(t.STOP) + 1]
    finally:
        r.close()


# 1. hysteresis: v25 rule moves again at 0.078, --deadband-exit 0.10 holds
assert hysteresis(()) is True, 'v25 rule: 0.078 > deadband must move again'
assert hysteresis(('--deadband-exit', '0.10')) is False, '--deadband-exit 0.10: 0.078 must stay held'
print('ok   deadband hysteresis')

# 2. axis reversal: v25 stops both axes at once; v27 halts Y alone and keeps X moving
v25 = reversal(())
assert v25 and v25[0] == t.STOP, 'v25 rule: a reversal is an immediate STOP of both axes: %s' % v25
v27 = reversal(('--axis-reversal', '1'))
assert v27 and v27[0]['T'] == 141 and v27[0]['X'] == 1 and v27[0]['Y'] == 0, '--axis-reversal 1: Y halts alone, X keeps moving: %s' % v27
print('ok   per-axis reversal')

# 3. soft deadband stop: v25 stops at once; --stop-rate gives decreasing speeds first
v25 = deadband_entry(())
assert len([c for c in v25 if c['T'] == 141]) <= 1, 'v25 rule: STOP right away (at most one speed update already on the line): %s' % v25
soft = deadband_entry(('--stop-rate', '100'))
spd = [c['SPD'] for c in soft if c['T'] == 141]
assert len(spd) >= 3 and spd[-3:] == sorted(spd[-3:], reverse=True) and spd[-1] < spd[0], '--stop-rate 100: speeds must step down before STOP: %s' % soft
print('ok   soft deadband stop', spd)

# 4. bad values are refused
for bad in (['--axis-reversal', '2'], ['--stop-rate', '-1'], ['--deadband-exit', '-1'], ['--reversal-join-speed', '-1'], ['--reversal-hold-ms', '-1']):
    q = subprocess.run([t.BRIDGE, '--device', '/dev/null', *bad], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    assert q.returncode != 0, bad
print('ok   bad values refused')
print('ALL OK')
