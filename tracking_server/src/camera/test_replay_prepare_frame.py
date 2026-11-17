"""Deterministic replay cache misses and controls through the real packet loop."""
from contextlib import ExitStack
import io
import os
import sys
from threading import Event, Thread
import unittest
from unittest.mock import patch

from . import replay_cam, test_replay_pacing


class ReplayPrepareFrameTest(unittest.TestCase):
    def run_loop(self, pacing, *, enabled=True, command=None, no_skip=True):
        clock = test_replay_pacing.Clock()
        rows, reads = [], []
        control = None

        class Source:
            total, bad, kind = 4, 0, 'test PNG'

            def __init__(self, _):
                pass

            def read(self, pos):
                reads.append((pos, clock.t))
                if pos == 1 and len(reads) == 2:
                    clock.sleep(.025)
                    if command == 'pause':
                        control.request(paused=True)
                    elif command == 'seek':
                        control.request(seek=2, paused=True)
                    elif command == 'stop':
                        replay_cam._on_signal(None, None)
                return bytes([pos + 1]) * replay_cam.FRAME_BYTES

            def png(self, pos):
                return 100 + pos

        class EmptyAhead(replay_cam.Ahead):
            # Deliberately miss the existing PNG cache without a worker race.
            def start(self):
                pass

            def join(self, _):
                pass

        class Control(replay_cam.Control):
            def start(self):
                nonlocal control
                control = self
                self.ev.wait = self.wait

            def wait(self, timeout):
                if command == 'seek_on_wait' and len(reads) == 2 and len(rows) == 1:
                    self.request(seek=2, paused=True)
                if not self.ev.is_set():
                    clock.sleep(timeout)
                return self.ev.is_set()

            def request(self, *, seek=None, paused=None):
                with self.lk:
                    self.seek, self.paused = seek, paused
                    self.ev.set()

        def write(_fd, data):
            head = replay_cam.HDR.unpack(data[:replay_cam.HDR.size])
            repeat = bool(head[2] & replay_cam.FLAG_REPEAT)
            pos, pass_no, png = head[6], head[8] >> 20, head[8] & 0xFFFFF
            self.assertEqual(head[3], len(rows) + 1)
            self.assertEqual(png, pos + 100)
            self.assertEqual(data[replay_cam.HDR.size:], bytes([pos + 1]) * replay_cam.FRAME_BYTES)
            rows.append((pos, repeat, pass_no, clock.t))
            if repeat:
                if pos == 3:
                    replay_cam._on_signal(None, None)
                control.request(paused=False)
            clock.sleep(.001)

        args = ['replay_cam', '--dir', 'virtual_pngs', '--fps', '30', '--rep-fps', '10',
                '--gap', '.05', '--delay', '0', '--loops', '1', '--tag', '--pacing', pacing]
        if no_skip:
            args.append('--no-skip')
        if command is not None:
            args.append('--control')
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', args))
            stack.enter_context(patch.object(sys, 'stderr', io.StringIO()))
            stack.enter_context(patch.dict(os.environ, {
                'LWIR_CAM_STAMP': '1', 'LWIR_CAM_SHM': '0', 'LWIR_REPLAY_CONTROL': '0',
                'LWIR_REPLAY_PREPARE_FRAME': '1' if enabled else '0',
            }))
            stack.enter_context(patch.dict(sys.modules, {'cv2': object(), 'numpy': object()}))
            stack.enter_context(patch.object(replay_cam.os.path, 'isfile', return_value=False))
            stack.enter_context(patch.object(replay_cam.os.path, 'isdir', return_value=True))
            stack.enter_context(patch.object(replay_cam, 'PngSource', Source))
            stack.enter_context(patch.object(replay_cam, 'Ahead', EmptyAhead))
            stack.enter_context(patch.object(replay_cam, 'Control', Control))
            stack.enter_context(patch.object(replay_cam, 'write_all', write))
            stack.enter_context(patch.object(replay_cam, 'SPIN_S', .0005))
            stack.enter_context(patch.object(replay_cam.signal, 'signal'))
            stack.enter_context(patch.object(replay_cam.time, 'perf_counter', clock.now))
            stack.enter_context(patch.object(replay_cam.time, 'monotonic_ns', lambda: int(clock.t * 1e9)))
            stack.enter_context(patch.object(replay_cam.time, 'sleep', clock.sleep))
            self.assertEqual(replay_cam.main(), 0)
        return rows, reads

    def test_cache_miss_work_fits_inside_the_period(self):
        for pacing in ('legacy', 'deadline'):
            with self.subTest(pacing=pacing):
                baseline, _ = self.run_loop(pacing, enabled=False)
                rows, reads = self.run_loop(pacing)
                self.assertEqual([r[:3] for r in rows], [r[:3] for r in baseline])
                self.assertEqual([p for p, _ in reads], [0, 1, 2, 3])
                self.assertGreater(baseline[1][3] - baseline[0][3], .057)
                period = 1 / 30 + (.001 if pacing == 'legacy' else 0)
                self.assertAlmostEqual(rows[1][3] - rows[0][3], period, delta=.00002)
                self.assertLess(reads[1][1] + .025, rows[1][3])

    def test_pause_holds_last_frame_and_play_reuses_the_prepared_frame(self):
        for pacing in ('legacy', 'deadline'):
            with self.subTest(pacing=pacing):
                rows, reads = self.run_loop(pacing, command='pause')
                self.assertEqual([r[:3] for r in rows], [
                    (0, False, 0), (0, True, 0), (1, False, 0),
                    (2, False, 0), (3, False, 0), (3, True, 0),
                ])
                self.assertEqual([p for p, _ in reads], [0, 1, 2, 3])

    def test_seek_with_pause_discards_the_old_frame_and_keeps_the_gap(self):
        for pacing in ('legacy', 'deadline'):
            with self.subTest(pacing=pacing):
                rows, reads = self.run_loop(pacing, command='seek')
                self.assertEqual([r[:3] for r in rows], [
                    (0, False, 0), (2, False, 1), (2, True, 1), (3, False, 1), (3, True, 1),
                ])
                self.assertEqual([p for p, _ in reads], [0, 1, 2, 3])
                self.assertGreaterEqual(rows[1][3] - reads[1][1], .075)

    def test_signal_during_prepare_stops_without_emitting_the_frame(self):
        for pacing in ('legacy', 'deadline'):
            with self.subTest(pacing=pacing):
                rows, reads = self.run_loop(pacing, command='stop')
                self.assertEqual([r[:3] for r in rows], [(0, False, 0)])
                self.assertEqual([p for p, _ in reads], [0, 1])

    def test_seek_during_release_wait_discards_the_prepared_frame(self):
        for pacing in ('legacy', 'deadline'):
            with self.subTest(pacing=pacing):
                rows, reads = self.run_loop(pacing, command='seek_on_wait')
                self.assertEqual([r[:3] for r in rows[:3]], [
                    (0, False, 0), (2, False, 1), (2, True, 1),
                ])
                self.assertEqual([p for p, _ in reads], [0, 1, 2, 3])

    def test_skip_mode_ignores_preparation_option(self):
        for pacing in ('legacy', 'deadline'):
            with self.subTest(pacing=pacing):
                self.assertEqual(self.run_loop(pacing, no_skip=False),
                                 self.run_loop(pacing, no_skip=False, enabled=False))

    def test_preparation_reuses_in_flight_png_read(self):
        entered, release, waiting = Event(), Event(), Event()
        reads, results = [], []

        class Source:
            def read(self, pos):
                reads.append(pos)
                entered.set()
                if not release.wait(2):
                    raise TimeoutError('PNG read was not released')
                return b'png'

        ahead = replay_cam.Ahead(Source())
        wait_for = ahead.cv.wait_for

        def observe_wait(predicate):
            self.assertFalse(predicate())
            waiting.set()
            return wait_for(predicate)

        prepare = Thread(target=lambda: results.append(ahead.prepare(1)), daemon=True)
        ahead.start()
        try:
            ahead.set_want([1])
            self.assertTrue(entered.wait(2))
            with patch.object(ahead.cv, 'wait_for', observe_wait):
                prepare.start()
                self.assertTrue(waiting.wait(2))
                release.set()
                prepare.join(2)
        finally:
            release.set()
            with ahead.cv:
                ahead.done = True
                ahead.cv.notify_all()
            ahead.join(2)
        self.assertEqual(reads, [1])
        self.assertEqual(results, [b'png'])


if __name__ == '__main__':
    unittest.main()
