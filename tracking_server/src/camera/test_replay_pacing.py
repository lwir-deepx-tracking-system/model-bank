"""Drive the real replay loop with a timed pipe and deterministic source.

The clock is simulated, so these are scheduling regressions, not device
performance measurements. No detector, physical camera or motor is used.
"""
import contextlib
import io
import os
import struct
import sys
import unittest
from unittest.mock import patch

from . import replay_cam


class Clock:
    def __init__(self):
        self.t = 100.0

    def now(self):
        self.t += 0.000001  # allow the real spin-wait to finish
        return self.t

    def sleep(self, delay):
        self.t += max(0.0, delay)


class ReplayPacingTest(unittest.TestCase):
    def run_loop(self, pacing, *, frames=212, stall_at=None, fps=30):
        clock = Clock()
        rows = []

        class Source:
            total, bad, kind = frames, 0, 'test raw'

            def __init__(self, *_):
                pass

            def read(self, pos):
                clock.sleep(.0005)
                return struct.pack('<I', pos) + bytes(replay_cam.FRAME_BYTES - 4)

            def png(self, pos):
                return pos + 1

        def write(_fd, data):
            head = replay_cam.HDR.unpack(data[:replay_cam.HDR.size])
            body = data[replay_cam.HDR.size:]
            self.assertEqual(len(body), replay_cam.FRAME_BYTES)
            rows.append((clock.t, head, struct.unpack('<I', body[:4])[0]))
            clock.sleep(.080 if len(rows) - 1 == stall_at else .0015)

        # RawSource is virtual, but main(), its packet header and scheduling
        # loop are the production implementation.
        args = ['replay_cam', '--dir', 'pacing.raw', '--fps', str(fps),
                '--loops', '1', '--no-skip', '--tag', '--pacing', pacing]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', args))
            stack.enter_context(patch.object(sys, 'stderr', io.StringIO()))
            stack.enter_context(patch.dict(os.environ, {
                'LWIR_CAM_STAMP': '1', 'LWIR_CAM_SHM': '0',
                'LWIR_REPLAY_CONTROL': '0', 'LWIR_REPLAY_DELAY': '0',
            }))
            stack.enter_context(patch.object(replay_cam.os.path, 'isfile', return_value=True))
            stack.enter_context(patch.object(replay_cam, 'RawSource', Source))
            stack.enter_context(patch.object(replay_cam, 'write_all', write))
            stack.enter_context(patch.object(replay_cam.signal, 'signal'))
            stack.enter_context(patch.object(replay_cam.time, 'perf_counter', clock.now))
            stack.enter_context(patch.object(replay_cam.time, 'monotonic_ns', lambda: int(clock.t * 1e9)))
            stack.enter_context(patch.object(replay_cam.time, 'sleep', clock.sleep))
            self.assertEqual(replay_cam.main(), 0)
        self.assertEqual([row[1][6] for row in rows], list(range(frames)))
        self.assertEqual([row[2] for row in rows], list(range(frames)))
        self.assertEqual([row[1][3] for row in rows], list(range(1, frames + 1)))
        self.assertTrue(all(not row[1][2] & replay_cam.FLAG_REPEAT for row in rows))
        return [row[0] for row in rows]

    def test_read_and_pipe_cost_do_not_accumulate_in_deadline_mode(self):
        legacy = self.run_loop('legacy')
        deadline = self.run_loop('deadline')
        self.assertAlmostEqual((legacy[-1] - legacy[0]) / 211, 1 / 30 + .002, delta=.00002)
        self.assertAlmostEqual((deadline[-1] - deadline[0]) / 211, 1 / 30, delta=.00002)

    def test_slow_pipe_rebases_time_without_skipping_or_unbounded_catchup(self):
        times = self.run_loop('deadline', stall_at=70)
        self.assertGreaterEqual(times[71] - times[70], .080 + 1 / 30)
        for a, b in zip(times[71:], times[72:]):
            self.assertAlmostEqual(b - a, 1 / 30, delta=.00002)

    def test_lower_playback_rate_keeps_its_requested_period(self):
        times = self.run_loop('deadline', frames=15, fps=5)
        self.assertAlmostEqual((times[-1] - times[0]) / 14, .2, delta=.00002)

    def test_skip_mode_schedule_is_unchanged(self):
        for mode in ('legacy', 'deadline'):
            self.assertEqual(replay_cam.next_deadline(1., 5., .1, no_skip=False, pacing=mode), 1.1)

    def test_invalid_environment_cannot_enable_candidate_silently(self):
        with (patch.object(sys, 'argv', ['replay_cam', '--dir', 'unused.raw']),
              patch.dict(os.environ, {'LWIR_REPLAY_PACING': 'legcay'}),
              patch.object(sys, 'stderr', io.StringIO()),
              self.assertRaises(SystemExit) as error):
            replay_cam.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
