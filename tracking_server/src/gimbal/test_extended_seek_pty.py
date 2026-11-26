"""Validate the extended deadline on a pseudo-terminal, without a gimbal device."""
import os
import subprocess
import sys
from pathlib import Path

OPTIONS = ('--seek', '1', '--bounded-seek', '1', '--seek-speed', '2',
        '--seek-ms', '3600', '--coast', '1', '--coast-ms', '800',
        '--reacquire-ms', '10000')


def verify_extended_seek(t, *, wide=False):
    options = ('--seek-ms', '6500') if wide else ()
    rig = t.Rig(*OPTIONS, *options)
    command = ''
    try:
        assert rig.serial(1) == t.STOP
        assert rig.wait_line('BOUNDED_SEEK_CONTROL') == 'BOUNDED_SEEK_CONTROL 1'

        def frame(*people, age_ms=0):
            rig.frame_id += 1
            rows = ''.join(' %d %.1f %.1f 40 80 0.9' % (i, x - 20, y - 40) for i, x, y in people)
            rig.send(f'F {rig.frame_id} {t.time.monotonic_ns() - age_ms * 1000000} {len(people)}{rows} OBS 1{command}')

        rig.frame = frame
        rig.feed(.12, (7, 320, 240))
        rig.send('A 7')
        assert rig.wait_line('R A') == 'R A OK'
        rig.feed(.12, (7, 320, 240))
        rig.feed(.82)
        command = ' SEEK 7 348.4 230 20 20'
        if wide:
            assert {'T': 141, 'X': 1, 'Y': 0, 'SPD': 2} in rig.feed(1.12, (8, 419, 240))
            command = ''
            assert t.STOP in rig.feed(.15, (8, 419, 240))
            command = ' SEEK 7 271.6 230 20 20'
            assert {'T': 141, 'X': -1, 'Y': 0, 'SPD': 2} in rig.feed(2.32, (8, 419, 240))
            rig.wait_state({'id': 7, 'lost': 1})
            command = ''
            assert t.STOP in rig.feed(.15, (8, 419, 240))
            command = ' SEEK 7 348.4 230 20 20'
            assert {'T': 141, 'X': 1, 'Y': 0, 'SPD': 2} in rig.feed(.5, (8, 419, 240))
            assert t.STOP in rig.feed(1.65, (8, 419, 240))
            assert not any(c['T'] == 141 for c in rig.feed(.20, (8, 419, 240)))
            print('PASS wide left/right and return with a bystander, STOP and deadline, original ID retained')
            return
        assert {'T': 141, 'X': 1, 'Y': 0, 'SPD': 2} in rig.feed(.63)
        command = ''
        assert t.STOP in rig.feed(.15)
        command = ' SEEK 7 271.6 230 20 20'
        assert {'T': 141, 'X': -1, 'Y': 0, 'SPD': 2} in rig.feed(.68)
        command = ''
        assert t.STOP in rig.feed(.15)
        command = ' SEEK 7 348.4 230 20 20'
        assert {'T': 141, 'X': 1, 'Y': 0, 'SPD': 2} in rig.feed(.50)
        assert t.STOP in rig.feed(.80)
        assert not any(c['T'] == 141 for c in rig.feed(.20))
        rig.wait_state({'id': 7, 'lost': 1})
        print('PASS low-speed reverse and return after 1.45 s, STOP at pauses, bridge time limit and original ID retained')
    finally:
        rig.close()


def verify_axis_switch(t):
    rig = t.Rig(*OPTIONS, '--seek-ms', '6500')
    command = ''
    try:
        assert rig.serial(1) == t.STOP
        assert rig.wait_line('BOUNDED_SEEK_CONTROL') == 'BOUNDED_SEEK_CONTROL 1'

        def frame(*people, age_ms=0):
            rig.frame_id += 1
            rows = ''.join(' %d %.1f %.1f 40 80 0.9' % (i, x - 20, y - 40) for i, x, y in people)
            rig.send(f'F {rig.frame_id} {t.time.monotonic_ns() - age_ms * 1000000} {len(people)}{rows} OBS 1{command}')

        rig.frame = frame
        rig.feed(.12, (7, 320, 240))
        rig.send('A 7')
        assert rig.wait_line('R A') == 'R A OK'
        rig.feed(.12, (7, 320, 240))
        rig.feed(.83, (8, 410, 240))
        command = ' SEEK 7 348.4 230 20 20'
        assert {'T': 141, 'X': 1, 'Y': 0, 'SPD': 2} in rig.feed(.30, (8, 410, 240))
        for x, y, direction in ((310, 201.2, (0, 1)), (310, 258.8, (0, -1)), (271.6, 230, (-1, 0))):
            command = f' SEEK 7 {x} {y} 20 20'
            received = rig.feed(.30, (8, 410, 240))
            move = {'T': 141, 'X': direction[0], 'Y': direction[1], 'SPD': 2}
            assert received.index(t.STOP) < received.index(move), received
        received = rig.feed(.35, (7, 320, 240), (8, 410, 240))
        assert t.STOP in received
        rig.wait_state({'id': 7, 'lost': 0})
        assert not any(packet['T'] == 141 for packet in rig.feed(.20, (7, 320, 240)))
        print('PASS pan/tilt switches, STOP before each new command, same-ID recovery stops search')
    finally:
        rig.close()


if __name__ == '__main__':
    sys.argv[1] = bridge = str(Path(sys.argv[1]).resolve())
    from test_v27_coast import t

    bad = subprocess.run([bridge, '--device', os.devnull, '--seek', '1', '--bounded-seek', '1',
                          '--seek-speed', '2', '--seek-ms', '6501'],
                         capture_output=True, text=True, timeout=5)
    assert bad.returncode == 2 and '--bounded-seek requires' in bad.stderr
    verify_extended_seek(t)
    verify_extended_seek(t, wide=True)
    verify_axis_switch(t)
