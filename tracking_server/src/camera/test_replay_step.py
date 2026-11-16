"""Drive replay's real control parser and packet loop without a camera."""
from contextlib import ExitStack
import io
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import replay_cam
from .test_replay_pacing import Clock


class ReplayStepTest(unittest.TestCase):
    def run_packets(self, prepare, pacing, commands, expected):
        clock, packets = Clock(), []
        control = None

        class Source:
            total, bad, kind = 4, 0, 'synthetic raw'

            def __init__(self, *_):
                pass

            def read(self, pos):
                return bytes([pos])

            def png(self, pos):
                return 100 + pos

        class Control(replay_cam.Control):
            def start(self):
                nonlocal control
                control = self
                self.ev.wait = self.wait

            def wait(self, timeout):
                if not self.ev.is_set():
                    clock.sleep(timeout)
                return self.ev.is_set()

        def emit(_fd, data):
            head = replay_cam.HDR.unpack(data[:replay_cam.HDR.size])
            pos, repeat, pass_no = head[6], bool(head[2] & replay_cam.FLAG_REPEAT), head[8] >> 20
            self.assertEqual(head[3], len(packets) + 1)
            self.assertEqual(head[8] & 0xFFFFF, 100 + pos)
            self.assertEqual(data[replay_cam.HDR.size:], bytes([pos]))
            self.assertLess(len(packets), len(expected), 'unexpected extra packet')
            self.assertEqual((pos, repeat, pass_no), expected[len(packets)])
            packets.append((pos, repeat, pass_no))
            lines = commands.get(len(packets))
            if lines:
                with patch.object(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(lines))):
                    control._run()
                self.assertEqual(control.bad, 0)
            if len(packets) == len(expected):
                replay_cam._on_signal(None, None)

        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', [
                'replay_cam', '--dir', 'synthetic.raw', '--fps', '30', '--rep-fps', '10', '--gap', '.05',
                '--delay', '0', '--loops', '1', '--tag', '--control', '--no-skip', '--pacing', pacing]))
            stack.enter_context(patch.object(sys, 'stderr', io.StringIO()))
            stack.enter_context(patch.dict(os.environ, {
                'LWIR_CAM_STAMP': '1', 'LWIR_CAM_SHM': '0', 'LWIR_REPLAY_PREPARE_FRAME': str(int(prepare)),
            }))
            stack.enter_context(patch.object(replay_cam.os.path, 'isfile', return_value=True))
            stack.enter_context(patch.object(replay_cam, 'RawSource', Source))
            stack.enter_context(patch.object(replay_cam, 'Control', Control))
            stack.enter_context(patch.object(replay_cam, 'write_all', emit))
            stack.enter_context(patch.object(replay_cam.signal, 'signal'))
            stack.enter_context(patch.object(replay_cam.time, 'perf_counter', clock.now))
            stack.enter_context(patch.object(replay_cam.time, 'monotonic_ns', lambda: int(clock.t * 1e9)))
            stack.enter_context(patch.object(replay_cam.time, 'sleep', clock.sleep))
            self.assertEqual(replay_cam.main(), 0)
        self.assertEqual(packets, expected)

    def check_modes(self, commands, expected):
        for prepare in (False, True):
            for pacing in ('legacy', 'deadline'):
                with self.subTest(prepare=prepare, pacing=pacing):
                    self.run_packets(prepare, pacing, commands, expected)

    def test_pause_two_steps_and_play_keep_every_frame_once(self):
        self.check_modes({1: b'PAUSE\n', 2: b'STEP\nSTEP\n', 5: b'PLAY\n', 7: b'PAUSE\nSTEP\nSTEP\n'}, [
            (0, False, 0), (0, True, 0), (1, False, 0), (2, False, 0),
            (2, True, 0), (3, False, 0), (3, True, 0), (3, True, 0),
        ])

    def test_play_and_seek_clear_steps_and_playing_step_does_not_leak(self):
        self.check_modes({
            1: b'STEP\nPAUSE\n',
            2: b'STEP\nSTEP\nPLAY\nPAUSE\n',
            3: b'STEP\nSTEP\n',
            4: b'STEP\nSEEK 2 PAUSE\n',
            6: b'PLAY\n',
        }, [
            (0, False, 0), (0, True, 0), (0, True, 0), (1, False, 0),
            (2, False, 1), (2, True, 1), (3, False, 1), (3, True, 1),
        ])


if __name__ == '__main__':
    unittest.main()
