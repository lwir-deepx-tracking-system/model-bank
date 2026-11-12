#!/usr/bin/env python3
"""CPU-only checks; neither real serial nor a camera program is opened."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import fixed_pulse_hil as h


def header(**changes):
    row = dict(magic=b'LWF1', version=1, flags=0, seq=9, sdk_ns=1000, dt=30_000,
               slot=0xffffffff, payload=h.PIXEL_BYTES, ring=0, width=h.H.W, height=h.H.H, crc=0)
    row.update(changes)
    raw = h.HDR.pack(*row.values())
    check = 0x5A17C0DE
    for word in struct.unpack('<11I', raw[:-4]):
        check ^= word
    return raw[:-4] + struct.pack('<I', check)


class FixedPulseTests(unittest.TestCase):
    def wait_until(self, pred, timeout=1):
        start = time.monotonic()
        while not pred() and time.monotonic() - start < timeout:
            time.sleep(.005)
        self.assertTrue(pred())

    def make_tap(self, write=None):
        tap = h.PulseTap(write=write)
        tap.connect()
        self.addCleanup(tap.close)
        time.sleep(h.BASELINE_NS / h.S + .03)
        tap.last_sample_ns = time.monotonic_ns()
        return tap

    def test_stamped_camera_header_strict(self):
        self.assertEqual(h.unpack_header(header()), (9, 1000))
        for raw in (header(version=2), header(flags=1), header(slot=0), header(width=320),
                    header(payload=42), header()[:22], header()[:-1] + b'\0'):
            with self.subTest(raw=raw.hex()), self.assertRaises(h.H.Abort):
                h.unpack_header(raw)

    def test_recent_arrival_cannot_refresh_a_stale_sdk_frame(self):
        arrival = time.monotonic_ns()
        sdk = arrival - h.FRAME_STALE_NS - 1
        raw = np.zeros((h.H.H, h.H.W), dtype=np.uint16)
        camera = SimpleNamespace(get=lambda *_: (1, sdk, arrival, raw))
        tap = SimpleNamespace(last_sample_ns=0)
        capture = h.Capture(camera, tap, True, h.BACKGROUND_BOTTOM)
        with self.assertRaisesRegex(h.H.Abort, 'stale camera SDK'):
            capture.next(recovery=True)
        self.assertEqual(tap.last_sample_ns, 0)
        self.assertEqual(capture.rows[0]['sdk_recv_ns'], sdk)
        self.assertEqual(capture.rows[0]['arrival_ns'], arrival)
        self.assertFalse(capture.rows[0]['valid'])

    def test_watchdog_uses_sdk_stamp_for_real_registration(self):
        arrival = time.monotonic_ns()
        sdk = arrival - 50_000_000
        raw = np.zeros((h.H.H, h.H.W), dtype=np.uint16)
        camera = SimpleNamespace(get=lambda *_: (1, sdk, arrival, raw))
        tap = SimpleNamespace(last_sample_ns=0)
        capture = h.Capture(camera, tap, True, h.BACKGROUND_BOTTOM)
        capture.registration = SimpleNamespace(update=lambda _: ([0., 0.], [0., 0.], 1.))
        self.assertTrue(capture.next(recovery=True)['valid'])
        self.assertEqual(tap.last_sample_ns, sdk)

    def duplicate_camera(self, samples):
        clock = [10 * h.S]
        tap = SimpleNamespace(last_sample_ns=0)
        seen = []

        def get(last, timeout):
            seen.append(tap.last_sample_ns)
            expiry = clock[0] + round(timeout * h.S)
            if last >= len(samples):
                clock[0] = expiry
                return None
            offset_ms, raw = samples[last]
            sdk = 10 * h.S + offset_ms * 1_000_000
            arrival = sdk + 2_000_000
            if arrival > expiry:
                clock[0] = expiry
                return None
            clock[0] = max(clock[0], arrival)
            return last + 1, sdk, arrival, raw

        capture = h.Capture(SimpleNamespace(get=get), tap, True, h.BACKGROUND_BOTTOM)
        registered = []

        def register(raw):
            registered.append(raw)
            return [0., 0.], [0., 0.], 1.

        capture.registration = SimpleNamespace(update=register)
        return clock, tap, seen, registered, capture

    def test_fresh_stamp_on_exact_duplicate_does_not_pet_watchdog(self):
        first = np.zeros((h.H.H, h.H.W), dtype=np.uint16)
        second = np.ones_like(first)
        clock, tap, seen, registered, capture = self.duplicate_camera(
            [(0, first), (40, first.copy()), (80, second)])
        with patch.object(h.time, 'monotonic_ns', lambda: clock[0]):
            self.assertEqual(capture.next(recovery=True)['frame_id'], 1)
            self.assertEqual(capture.next(recovery=True)['frame_id'], 3)
        self.assertEqual([r['valid'] for r in capture.rows], [True, False, True])
        self.assertTrue(capture.rows[1]['duplicate_full_raw16'])
        self.assertEqual(capture.rows[1]['sdk_recv_ns'], 10 * h.S + 40_000_000)
        self.assertEqual(capture.rows[1]['arrival_ns'], 10 * h.S + 42_000_000)
        self.assertEqual([f[0] for f in capture.raw], [1, 2, 3])
        self.assertEqual(seen, [0, 10 * h.S, 10 * h.S])
        self.assertEqual(tap.last_sample_ns, 10 * h.S + 80_000_000)
        self.assertEqual(len(registered), 2)

    def test_continuous_duplicate_frames_cannot_extend_180ms_deadline(self):
        first = np.zeros((h.H.H, h.H.W), dtype=np.uint16)
        clock, tap, seen, registered, capture = self.duplicate_camera(
            [(offset, first.copy()) for offset in range(0, 361, 30)])
        with patch.object(h.time, 'monotonic_ns', lambda: clock[0]):
            capture.next(recovery=True)
            with self.assertRaisesRegex(h.H.Abort, 'no fresh unique'):
                capture.next(timeout=10, recovery=True)
        self.assertEqual(clock[0], 10 * h.S + h.FRAME_STALE_NS)
        self.assertEqual(tap.last_sample_ns, 10 * h.S)
        self.assertEqual(len(registered), 1)
        self.assertEqual(len(capture.rows), 6)
        self.assertTrue(all(r['duplicate_full_raw16'] and not r['valid'] for r in capture.rows[1:]))
        self.assertTrue(all(ns == 10 * h.S for ns in seen[1:]))

    def test_background_only_repeat_keeps_the_existing_freeze_guard(self):
        rng = np.random.default_rng(8)
        first = rng.integers(7000, 10000, (h.H.H, h.H.W), dtype=np.uint16)
        second = first.copy()
        second[-1, -1] += 1
        clock, _, _, _, capture = self.duplicate_camera([(0, first), (40, second)])
        capture.registration = h.Registration()
        with patch.object(h.time, 'monotonic_ns', lambda: clock[0]):
            capture.next(recovery=True)
            with self.assertRaisesRegex(h.H.Abort, 'frozen RAW16 background'):
                capture.next(recovery=True)
        self.assertFalse(capture.rows[-1]['duplicate_full_raw16'])
        self.assertFalse(capture.rows[-1]['valid'])

    def test_post_stop_recovery_restarts_only_the_view_reference(self):
        rng = np.random.default_rng(8)
        first = rng.integers(7000, 10000, (h.H.H, h.H.W), dtype=np.uint16)
        second = np.roll(first, 1, axis=1)
        clock, tap, _, _, capture = self.duplicate_camera([(0, first), (400, second)])
        capture.registration = h.Registration()
        capture.camera.cv = contextlib.nullcontext()
        capture.camera.frames = [first]
        tap.fault = 'latched fault after STOP'
        with patch.object(h.time, 'monotonic_ns', lambda: clock[0]):
            capture.next(recovery=True)
            clock[0] += 300_000_000
            capture.reset_registration_for_stop_view()
            self.assertEqual(capture.last_observed_ns, 0)
            self.assertIsNone(capture.last_raw)
            self.assertEqual(capture.next(recovery=True)['frame_id'], 2)
        self.assertEqual(tap.fault, 'latched fault after STOP')
        self.assertEqual([r['frame_id'] for r in capture.rows], [1, 2])

    def test_nominal_stop_does_not_need_main_loop(self):
        tap = self.make_tap()
        tx = tap.start(1, 1, 250, 1)
        time.sleep(.13)
        tap.last_sample_ns = time.monotonic_ns()
        self.wait_until(lambda: any(r.get('reason') == 'pulse_stop' for r in tap.rows))
        stop = next(r for r in tap.rows if r.get('reason') == 'pulse_stop')
        self.assertFalse(tap.fault)
        self.assertTrue(stop['ok'])
        self.assertEqual(stop['command'], (0, 0, 0))
        self.assertLessEqual(stop['due_ns'], stop['write_start_ns'])
        self.assertLessEqual(stop['write_start_ns'] - tx['write_start_ns'], 280_000_000)
        self.assertIsNone(tap.pulse)

    def test_watchdog_stops_on_stale_registration(self):
        tap = self.make_tap()
        tap.start(-1, 7, 350, 1)
        self.wait_until(lambda: tap.fault is not None)
        self.wait_until(lambda: any(r.get('reason') == 'abort_stop' for r in tap.rows))
        self.assertIn('registration', tap.fault)
        self.assertFalse(any(r.get('reason') == 'pulse_stop' for r in tap.rows))
        self.assertIsNone(tap.pulse)

    def test_rejects_invalid_command_before_write(self):
        sent = []
        tap = self.make_tap(lambda payload: sent.append(payload) or len(payload))
        with self.assertRaises(h.H.Abort):
            tap.start(1, 100, 350, 1)
        self.assertTrue(tap.fault)
        self.assertTrue(all(payload == h.STOP_WIRE for payload in sent))

    def test_short_motion_and_stop_writes_leave_a_standalone_stop(self):
        sent = []
        wire = bytearray()
        moving = False
        remaining_short_stops = 2

        def write(payload):
            nonlocal moving, remaining_short_stops
            sent.append(payload)
            size = len(payload)
            if b'141' in payload:
                moving = True
                size -= 4
            elif moving and remaining_short_stops:
                remaining_short_stops -= 1
                size -= 4
            wire.extend(payload[:size])
            return size

        tap = self.make_tap(write)
        wire.clear()
        with self.assertRaises(h.H.Abort):
            tap.start(1, 1, 250, 1)
        with self.assertRaises(h.H.Abort):
            tap.start(-1, 1, 250, 2)
        self.assertTrue(tap.write_failed)
        self.assertEqual(sum(b'141' in payload for payload in sent), 1)
        self.assertTrue(all(payload == h.STOP_WIRE for payload in sent[-3:]))
        recovered = [line for line in wire.splitlines() if line]
        self.assertEqual(len(recovered), 4)
        for partial in recovered[:3]:
            with self.assertRaises(json.JSONDecodeError):
                json.loads(partial)
        self.assertEqual(json.loads(recovered[-1]), {'T': 135})
        self.assertEqual([row['ok'] for row in tap.rows if row['reason'] == 'abort_stop'],
                         [False, False, True])

    def test_onset_requires_quiet_baseline_and_agreeing_signed_brackets(self):
        def row(ns, template, lk):
            return dict(arrival_ns=ns, template_px=[template, 0.], lk_px=[lk, 0.])

        motion = dict(write_start_ns=100)
        stop = dict(write_start_ns=220, write_end_ns=221)
        clean = [row(20, 0, 0), row(40, 0, 0), row(60, 0, 0)]
        good = [row(120, .4, .5), row(150, 2, 2), row(180, 2, 2), row(240, 3, 3)]
        result = h.displacement_evidence(clean + good, clean, good[-1:], motion, stop)
        self.assertTrue(all(v['onset_status'] == 'resolved' for v in result.values()))
        cases = [
            (clean + [row(80, 1.1, 0)], good),
            (clean, [row(120, .4, -.4), row(150, 2, -2), row(180, 2, -2), row(240, 3, -3)]),
            (clean, [row(120, 2, 0), row(150, 2, 0), row(180, 2, 2), row(240, 3, 3)]),
        ]
        for baseline, response in cases:
            result = h.displacement_evidence(baseline + response, baseline, response[-1:], motion, stop)
            self.assertTrue(all(v['onset_status'] == 'unresolved' and v['onset_arrival_ms'] is None
                                for v in result.values()))

    def test_freeze_and_bad_contrast_are_rejected(self):
        cv2.setNumThreads(1)
        rng = np.random.default_rng(8)
        frame = rng.integers(7000, 10000, (h.H.H, h.H.W), dtype=np.uint16)
        reg = h.Registration()
        reg.update(frame)
        with self.assertRaisesRegex(h.H.Abort, 'frozen'):
            reg.update(frame.copy())
        with self.assertRaisesRegex(h.H.Abort, 'contrast'):
            h.Registration().update(np.full_like(frame, 8000))

    def test_moving_foreground_outside_background_roi_is_ignored(self):
        cv2.setNumThreads(1)
        rng = np.random.default_rng(3)
        frame = rng.integers(7000, 10000, (h.H.H, h.H.W), dtype=np.uint16)
        reg = h.Registration(180)
        reg.update(frame)
        moved = np.roll(frame, 4, axis=1)
        moved[180:] = rng.integers(0, 65535, moved[180:].shape, dtype=np.uint16)
        template, flow, _ = reg.update(moved)
        np.testing.assert_allclose(template, [4, 0], atol=.5)
        np.testing.assert_allclose(flow, [4, 0], atol=.5)

    def test_default_dry_evidence(self):
        root = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / 'evidence'
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = h.main(['--repo', str(root), '--out', str(out)])
            self.assertEqual(rc, 0, stdout.getvalue())
            result = json.loads((out / 'result.json').read_text())
            self.assertTrue(result['dry'] and result['stop_view_confirmed'])
            self.assertEqual(result['background_roi_xyxy'], [16, 0, 624, 180])
            self.assertEqual(result['raw_frames'], result['raw_saved'])
            frames = [json.loads(line) for line in (out / 'frames.jsonl').read_text().splitlines()]
            self.assertEqual(len(frames), result['raw_saved'])
            self.assertTrue(all(r['sdk_recv_ns'] is None and r['exposure_ns'] is None for r in frames))
            self.assertTrue(all(a['arrival_ns'] < b['arrival_ns'] for a, b in zip(frames, frames[1:])))
            raw_file = next((out / 'raw16').glob('*.png'))
            raw = cv2.imdecode(np.frombuffer(raw_file.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            self.assertEqual((raw.dtype, raw.shape), (np.uint16, (480, 640)))
            serial = [json.loads(line) for line in (out / 'serial.jsonl').read_text().splitlines()]
            key = [r for r in serial if r['reason'] in ('fixed_pulse', 'pulse_stop')]
            self.assertEqual(len(key), 8)
            for (direction, speed, duration), motion, stop, trial in zip(h.PULSES, key[::2], key[1::2],
                                                                         result['trials']):
                self.assertEqual(motion['command'], [direction, 0, speed])
                self.assertEqual(stop['command'], [0, 0, 0])
                self.assertGreaterEqual(trial['post_stop_capture_ms'], 1200)
                self.assertGreaterEqual(stop['write_start_ns'] - motion['write_start_ns'],
                                        (duration - 1) * 1_000_000)
                for metric in trial['image_response'].values():
                    self.assertGreater(-direction * metric['total_dx_px'], 2)
                    self.assertIsNotNone(metric['onset_arrival_ms'])


if __name__ == '__main__':
    unittest.main()
