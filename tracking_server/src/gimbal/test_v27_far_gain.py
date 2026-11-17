#!/usr/bin/env python3
"""v27: --far-gain and --brake-px-per-spd must also work with --axis-reversal 1 (the executor used to take the shared SPD
from the unshaped PID values there, so both options did nothing on the live command line). End to end through
gimbal_bridge_v27 without a gimbal (pseudo terminal, nothing moves).
Usage: python3 test_v27_far_gain.py <path of gimbal_bridge_v27>   (test_gimbal_bridge.py must be next to this file)"""
import sys
sys.argv = sys.argv[:2]
import test_gimbal_bridge as t


def steady_speed(*options, at=(448, 240)):
    """follow a person standing 128 px right of the centre; the SPD the bridge settles at"""
    r = t.Rig('--deadband', '0.06', '--speed-max', '100', '--speed-rate', '150', *options)
    try:
        assert r.serial(1.0) == t.STOP, 'the first command must be STOP'
        r.feed(.1, (7, *at))
        r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
        moves = [c for c in r.feed(.8, (7, *at)) if c['T'] == 141]
        assert moves and all(c['X'] == 1 and c['Y'] == 0 for c in moves), moves
        return moves[-1]['SPD']
    finally:
        r.close()


AXIS = ('--axis-reversal', '1')
FAR = ('--far-gain', '2', '--far-start-px', '64', '--far-full-px', '160')

# 1. defaults: Kp 30 at 128 px (error 0.4) = SPD 12, with and without axis reversal
plain, plain_axis = steady_speed(), steady_speed(*AXIS)
assert plain == 12 and plain_axis == 12, (plain, plain_axis)
print('ok   far gain off: SPD 12 at 128 px with and without --axis-reversal 1')

# 2. far gain 2 over 64..160 px is x 1.667 at 128 px = SPD 20, the same with axis reversal
far, far_axis = steady_speed(*FAR), steady_speed(*AXIS, *FAR)
assert far == 20, far
assert far_axis == 20, '--axis-reversal 1 dropped the far gain: SPD %s' % far_axis
print('ok   far gain 2: SPD 20 at 128 px with and without --axis-reversal 1')

# 3. below far-start nothing changes (48 px: SPD 4 or 5)
near, near_far = steady_speed(*AXIS, at=(368, 240)), steady_speed(*AXIS, *FAR, at=(368, 240))
assert near == near_far and 4 <= near <= 5, (near, near_far)
print('ok   far gain 2: no change below --far-start-px')

# 4. the near brake caps the speed also with axis reversal: (128 - 19.2) / 20 = SPD 5
brake, brake_axis = steady_speed('--brake-px-per-spd', '20'), steady_speed(*AXIS, '--brake-px-per-spd', '20')
assert brake == 5 and brake_axis == 5, (brake, brake_axis)
print('ok   brake 20 px/SPD: SPD 5 at 128 px with and without --axis-reversal 1')
print('ALL OK')
