#!/usr/bin/env python3
"""Check the actual v27 F/G parser and serial output through a PTY, without a motor.

Usage: python3 test_v27_observation_grace.py ./gimbal_bridge_v27
"""
import subprocess
import time

import test_gimbal_bridge as bridge


def frame(rig, mode, rows=(), suffix=' OBS 1', stamp=None):
    rig.frame_id += 1
    captured = time.monotonic_ns() if stamp is None else stamp
    dimensions = ' 64 48' if mode == 'G' else ''
    header = f'{mode} {rig.frame_id} {captured}{dimensions} {len(rows)}'
    wire_rows = ''.join(f' {row}' for row in rows)
    payload = bytes(64 * 48 * 2) if mode == 'G' else b''
    rig.proc.stdin.write((header + wire_rows + suffix + '\n').encode() + payload)


def row(mode, person=7, confidence='.9'):
    # Same normalized position and size for F and G.
    geometry = '36.4 20 4 8' if mode == 'G' else '364 200 40 80'
    return f'{person} {geometry} {confidence}'


def start(mode):
    rig = bridge.Rig('--deadband', '.06', '--controller', 'p', '--kp', '30',
                     '--aim-y', '.5', '--speed-min', '6', '--input-timeout-ms', '250',
                     '--observation-grace-ms', '100')
    try:
        assert rig.serial(1) == bridge.STOP
        assert rig.wait_line('OBSERVATION_STATUS ') == 'OBSERVATION_STATUS 1'
        rig.send('P')
        frame(rig, mode, (row(mode),), stamp=time.monotonic_ns() - 70_000_000)
        rig.send('A 7')
        assert rig.wait_line('R A') == 'R A OK'
        epoch = int(rig.wait_line('CONTROL_EPOCH ').split()[1])
        frame(rig, mode, (row(mode),))
        command = rig.serial(.1)
        assert command and command['T'] == 141 and command['X'] == 1 and command['SPD'] == 6, command
        rig.wait_state(dict(auto=1, lost=0, moving=1, id=7))
        time.sleep(.03)
        return rig, epoch
    except BaseException:
        rig.close()
        raise


def one_gap(mode):
    rig, epoch = start(mode)
    try:
        lead = f' LEAD 7 6 0 {epoch}'
        suffix = ' OBS 1' + lead if mode == 'F' else lead + ' OBS 1'
        frame(rig, mode, (row(mode, 8),), suffix)
        rig.wait_state(dict(auto=1, lost=1, moving=1, id=7), .065)
        assert rig.serial(.015) is None, 'grace emitted a replacement command'
        frame(rig, mode, (row(mode),), suffix)
        rig.wait_state(dict(auto=1, lost=0, moving=1, id=7), .065)
        assert rig.serial(.015) is None, 'same-ID return issued STOP or reused old LEAD'

        time.sleep(.035)
        frame(rig, mode, (row(mode),), suffix)
        # The next frame must be consumed individually to test consecutive misses.
        time.sleep(.015)
        frame(rig, mode, (), suffix)
        rig.wait_state(dict(auto=1, lost=1, moving=1, id=7), .065)
        frame(rig, mode, (row(mode, 8),), suffix)
        assert rig.until_stop(.07), 'second distinct miss failed to stop'
        rig.wait_state(dict(auto=1, lost=1, moving=0, id=7), .075)
    finally:
        rig.close()
    print(f'ok {mode}: suffix order, one-frame hold, same-ID recovery and second-miss STOP')


def unsafe_suffixes():
    cases = ('', ' OBS 0', ' OBS 1 OBS 1', ' OBS 1 OBS 0', ' OBS 01',
             ' OBS 1x', ' OBS', ' OBS 1 LEAD 7 6', ' OBS 1junk')
    for mode in ('F', 'G'):
        for suffix in cases:
            rig, _ = start(mode)
            try:
                frame(rig, mode, (), suffix)
                assert rig.until_stop(.075), (mode, suffix)
                rig.wait_state(dict(auto=1, lost=1, moving=0, id=7), .075)
            finally:
                rig.close()
    print('ok F/G: absent, unsafe, duplicate and malformed suffixes cannot hold')


def invalid_selected_rows():
    for rows in ((row('G', confidence='nan'),), (row('G', confidence='.01'),),
                 (row('G'), row('G'))):
        rig, _ = start('G')
        try:
            frame(rig, 'G', rows)
            assert rig.until_stop(.075), rows
            rig.wait_state(dict(auto=1, lost=1, moving=0, id=7), .075)
            frame(rig, 'G', (row('G'),))
            command = rig.serial(.15)
            assert command and command['T'] == 141, command
        finally:
            rig.close()
    print('ok invalid and duplicate selected rows stop, and malformed G still consumes its picture')


def invalid_headers():
    for mode in ('F', 'G'):
        rig, _ = start(mode)
        try:
            rig.frame_id += 1
            dimensions = ' 64 48' if mode == 'G' else ''
            bad_header = f'{mode} invalid {time.monotonic_ns()}{dimensions} 0 OBS 1\n'.encode()
            payload = bytes(64 * 48 * 2) if mode == 'G' else b''
            rig.proc.stdin.write(bad_header + payload)
            assert rig.until_stop(.075), mode
            frame(rig, mode, (row(mode),))
            command = rig.serial(.15)
            assert command and command['T'] == 141, command
        finally:
            rig.close()
    print('ok malformed mandatory F/G fields stop and G keeps framing when dimensions are valid')


def option_limits():
    for timeout, value in ((250, '-1'), (250, '101'), (250, '50junk'), (50, '50'), (50, '100')):
        result = subprocess.run([bridge.BRIDGE, '--device', '/dev/null',
                                 '--input-timeout-ms', str(timeout), '--observation-grace-ms', value],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=2)
        assert result.returncode != 0 and 'observation-grace-ms' in result.stderr, result
    print('ok CLI rejects values outside 0..100, non-integers and values at or above the input timeout')


if __name__ == '__main__':
    option_limits()
    one_gap('F')
    one_gap('G')
    unsafe_suffixes()
    invalid_selected_rows()
    invalid_headers()
    print('ALL OK')
