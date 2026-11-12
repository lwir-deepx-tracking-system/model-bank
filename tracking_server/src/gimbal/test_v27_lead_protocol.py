#!/usr/bin/env python3
"""Actual F/G parsing and serial commands via a pseudo-terminal; no motor or camera.

Usage: python3 test_v27_lead_protocol.py ./gimbal_bridge_v27
"""
import time
import test_gimbal_bridge as t


def feed(r, mode, suffix='', seconds=.24, x=384, y=240):
    commands = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        r.send('P')
        r.frame_id += 1
        header = f'{mode} {r.frame_id} {time.monotonic_ns()}'
        if mode == 'G':
            header += ' 640 480'
        header += f' 1 7 {x-20} {y-40} 40 80 0.9{suffix}\n'
        r.proc.stdin.write(header.encode())
        if mode == 'G':
            r.proc.stdin.write(bytes(640*480*2))
        command = r.serial(.025)
        if command is not None:
            commands.append(command)
    return [command for command in commands if command['T'] == 141]


def select(r):
    r.send('A 7')
    assert r.wait_line('R A') == 'R A OK'
    return int(r.wait_line('CONTROL_EPOCH ').split()[1])


def protocol(mode):
    r = t.Rig('--deadband', '0.06', '--controller', 'p', '--kp', '30', '--aim-y', '.5')
    try:
        assert r.serial(1) == t.STOP
        assert r.wait_line('LEAD_CONTROL ') == 'LEAD_CONTROL 1'
        feed(r, mode, seconds=.1)
        epoch = select(r)
        moves = feed(r, mode, f' LEAD 7 60 0 {epoch}', seconds=.4)
        assert moves and max(c['SPD'] for c in moves) >= 11, moves
        previous_epoch = epoch
        epoch = select(r)
        assert epoch != previous_epoch
        assert r.until_stop()
        stale = feed(r, mode, f' LEAD 7 60 0 {previous_epoch}')
        assert stale and max(c['SPD'] for c in stale) <= 6, stale
        moves = feed(r, mode, f' LEAD 7 60 0 {epoch}', seconds=.4)
        assert moves and max(c['SPD'] for c in moves) >= 11, moves
        print(f'ok {mode}: bounded forecast, same-ID reselect and stale-epoch rejection')
    finally:
        r.close()


def invalid_suffixes():
    for suffix in (' LEAD 7 60', ' LEAD 7 nan 0 1', ' LEAD 7 61 0 1',
                   ' LEAD 7 10 20 1 trailing', ' LEAD 99 10 20 1'):
        r = t.Rig('--deadband', '.06', '--aim-y', '.5')
        try:
            assert r.serial(1) == t.STOP
            feed(r, 'G', suffix, seconds=.06)
            select(r)  # bad optional suffix must not discard the observed person
            moves = feed(r, 'G', suffix, seconds=.2)
            assert moves and all(c['X'] == 1 and c['Y'] == 0 for c in moves), moves
        finally:
            r.close()
    print('ok malformed, excessive, wrong-ID forecast preserves G picture framing and observations')


if __name__ == '__main__':
    protocol('F')
    protocol('G')
    invalid_suffixes()
    print('ALL OK')
