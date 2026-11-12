import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import predictive_brake_ab_hil as A


class BrakingHarnessTest(unittest.TestCase):
    def test_profile_boundaries_and_jitter(self):
        self.assertEqual(A.stimulus(0), ('static', 0.))
        self.assertEqual(A.stimulus(650_000_000), ('outward', 30.))
        self.assertEqual(A.stimulus(800_000_000), ('dwell', 60.))
        self.assertEqual(A.stimulus(2_400_000_000), ('return', 30.))
        self.assertEqual(A.stimulus(2_700_000_000), ('settle', 0.))
        self.assertEqual(A.stimulus(4_080_000_000), ('jitter', 4.))
        with self.assertRaises(ValueError):
            A.stimulus(A.END_NS)

    def test_publication_over_budget_records_stages_without_sending(self):
        bridge = mock.Mock()
        stamp = 1_000_000_000
        row = dict(sdk_recv_ns=stamp, arrival_ns=stamp + 3_000_000,
                   frame_id=197, template_px=[0., 0.])
        with mock.patch.object(A.time, 'monotonic_ns', return_value=stamp + 105_000_000):
            with self.assertRaisesRegex(A.H.Abort, 'scheduling tolerance'):
                A.publish(bridge, row, 0., 'settle', 3_000_000_000, 0.)
        bridge.send.assert_not_called()
        self.assertEqual(row['publish_attempt_ns'], stamp + 105_000_000)
        self.assertEqual(row['publish_deadline_checked_ns'], stamp + 105_000_000)

    def test_profile_affinity_validates_and_restores_caller_on_error(self):
        requested = A.parse_cpus('4,5')
        with self.assertRaises(A.argparse.ArgumentTypeError):
            A.parse_cpus('4,4')
        current = {0, 1, 4, 5}

        def set_affinity(thread, cpus):
            self.assertEqual(thread, 0)
            current.clear()
            current.update(cpus)

        record = {}
        with (mock.patch.object(A.os, 'sched_getaffinity', side_effect=lambda _: set(current), create=True),
              mock.patch.object(A.os, 'sched_setaffinity', side_effect=set_affinity, create=True) as setter,
              self.assertRaisesRegex(RuntimeError, 'failure inside profile')):
            with A.profile_affinity(requested, record):
                self.assertEqual(current, {4, 5})
                raise RuntimeError('failure inside profile')
        self.assertEqual(setter.call_args_list, [mock.call(0, requested), mock.call(0, {0, 1, 4, 5})])
        self.assertEqual(record['profile_cpu_affinity'],
                         dict(requested=[4, 5], caller_original=[0, 1, 4, 5],
                              caller_profile=[4, 5], caller_restored=[0, 1, 4, 5]))

    def test_wire_rejects_pitch_excess_duplicate_fields_and_unexpected_commands(self):
        self.assertEqual(A.parse_serial(b'{"T":135}'), (0, 0, 0))
        self.assertEqual(A.parse_serial(b'{"T":141,"X":-1,"Y":0,"SPD":7}'), (-1, 0, 7))
        for line in (b'{"T":141,"X":1,"Y":1,"SPD":7}', b'{"T":141,"X":1,"Y":0,"SPD":8}',
                     b'{"T":141,"X":1,"Y":0,"SPD":true}', b'{"T":141,"X":1,"X":-1,"Y":0,"SPD":1}',
                     b'{"T":135,"SPD":1}', b'{"T":133}', b'[]'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                A.parse_serial(line)

    def test_guard_blocks_stale_and_latches_against_restarting(self):
        tap = A.GuardedTap(None)
        tap.active_block = 1
        tap.last_sample_ns = time.monotonic_ns() - A.F.FRAME_STALE_NS - 1_000_000
        tap.forward(b'{"T":141,"X":1,"Y":0,"SPD":7}', 1)
        self.assertIn('recent unique', tap.fault)
        self.assertEqual([r['reason'] for r in tap.rows], ['abort_stop'] * 3)
        tap.last_sample_ns = time.monotonic_ns()
        tap.forward(b'{"T":141,"X":1,"Y":0,"SPD":7}', 1)
        self.assertEqual(tap.rows[-1]['reason'], 'bridge_blocked')
        self.assertTrue(all(r['command'][0] == 0 for r in tap.rows if r.get('ok')))

    def test_independent_watchdog_stops_without_bridge_or_more_frames(self):
        tap = A.GuardedTap(None)
        tap.connect()
        try:
            start = time.monotonic_ns()
            tap.last_sample_ns = start
            tap.arm(1)
            tap.forward(b'{"T":141,"X":1,"Y":0,"SPD":7}', 1)
            self.assertEqual(tap.current, (1, 0, 7))
            while time.monotonic_ns() - start < 500_000_000:
                with tap.lk:
                    stop = next((r for r in tap.rows if r['reason'] == 'abort_stop'), None)
                if stop:
                    break
                time.sleep(.005)
            self.assertIsNotNone(stop)
            self.assertIn('no recent accepted registration', tap.fault)
            self.assertLessEqual(stop['write_start_ns'] - start, A.F.FRAME_STALE_NS + 35_000_000)
        finally:
            tap.close()

    def test_guard_checks_common_scene_outward_with_both_registration_methods(self):
        tap = A.GuardedTap(None)
        tap.active_block = 1
        tap.last_sample_ns = time.monotonic_ns()
        tap.scene(dict(template_px=[-93., 0.], lk_px=[-96., 0.]))
        tap.forward(b'{"T":141,"X":-1,"Y":0,"SPD":7}', 1)
        self.assertEqual(tap.current, (-1, 0, 7))
        tap.forward(b'{"T":141,"X":1,"Y":0,"SPD":7}', 1)
        self.assertIn('common scene limit', tap.fault)
        self.assertEqual([r['reason'] for r in tap.rows], ['bridge_motion', *(['abort_stop'] * 3)])

    def test_metrics_distinguish_net_pan_from_unnecessary_path(self):
        samples = []
        for phase, start, end, *_ in A.PROFILE:
            for t in range(start, end, 50):
                x = 1. if t == start else (3. if t == start + 50 else 2.)
                samples.append(dict(phase=phase, elapsed_ns=t * 1_000_000,
                                    template_px=[x, 0], error_x_px=2., sent_ns=40_000_000, observation_ns=0))
        result = A.metrics(samples, [(1, 0, 3), (0, 0, 0), (-1, 0, 5)])
        self.assertEqual(result['hold_path']['settle'],
                         dict(total_path_px=3., net_path_px=1., excess_path_px=2.))
        self.assertEqual(result['stationary_hold_excess_path_px'], 6.)
        self.assertEqual(result['stop_restarts'], 1)
        self.assertEqual(result['reversals'], 1)
        self.assertEqual(result['center_deadband_occupancy'], 1.)

    def test_trace_requires_matching_sample_and_forwarded_stop(self):
        sample = dict(frame_id=11, observation_ns=1000, sent_ns=40_000_000)
        data = dict(event='predictive_brake', frame_id=11, captured_ns=1000,
                    track_id=A.H.TID, reserve_px=4., sent=True)
        stop = dict(write_start_ns=41_000_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bridge.log'
            path.write_text('logger\n' + json.dumps(data) + '\n', encoding='utf-8')
            self.assertEqual(A.read_trace(path, [sample], 1, [stop])[0]['downstream_stop_write_ns'],
                             stop['write_start_ns'])
            for mode, stops in ((0, [stop]), (1, [])):
                with self.subTest(mode=mode, stops=stops), self.assertRaises(A.H.Abort):
                    A.read_trace(path, [sample], mode, stops)
            path.write_text((json.dumps(data) + '\n') * 2, encoding='utf-8')
            with self.assertRaises(A.H.Abort):
                A.read_trace(path, [sample], 1, [stop])

    def test_abba_verdict_noop_and_hold_tradeoff(self):
        def block(index, stops, excess, restarts=0):
            return dict(block=index, predictive_stops=stops,
                        metrics=dict(center_deadband_occupancy=.9, final_settle_abs_error_px=3.,
                                     outside_110px_count=0, stop_restarts=restarts,
                                     stationary_hold_excess_path_px=excess))
        blocks = [block(1, 0, 3), block(2, 0, 2), block(3, 0, 2), block(4, 0, 3)]
        self.assertEqual(A.compare(blocks)['verdict'], 'candidate_no_op')
        blocks[1] = block(2, 1, 2)
        self.assertEqual(A.compare(blocks)['verdict'], 'provisional_improvement')
        blocks[2] = block(3, 1, 2, restarts=1)
        self.assertEqual(A.compare(blocks)['verdict'], 'provisional_improvement_not_established')


if __name__ == '__main__':
    unittest.main()
