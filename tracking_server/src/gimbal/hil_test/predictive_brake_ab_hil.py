#!/usr/bin/env python3
"""Bounded camera-only braking A/B using the actual bridge; default is PTY + synthetic RAW16."""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

import fixed_pulse_hil as F

H = F.H
ORDER = (0, 1, 1, 0)
CAP = 7
RUN_SECONDS = 45
CAMERA_SECONDS = 50
MAX_FRAMES = 1400
LATENCY_NS = 40_000_000
MAX_PUBLISH_AGE_NS = 65_000_000
DEADBAND_PX = 19.2
SCENE_BRAKE_PX = 95.
DRY_RATE_CURVE = ((1, 112.), (7, 212.))
DRY_DELAY_S = .105
DRY_TAU_S = .027
TARGET_LIMIT_PX = 110.
PROFILE = (
    ('static', 0, 500, 0., 0.),
    ('outward', 500, 800, 0., 60.),
    ('dwell', 800, 2100, 60., 60.),
    ('return', 2100, 2700, 60., 0.),
    ('settle', 2700, 4000, 0., 0.),
    ('jitter', 4000, 4600, 0., 0.),
)
END_NS = PROFILE[-1][2] * 1_000_000
BRIDGE_OPTIONS = (
    '--manual-speed', '7', '--speed-max', '7', '--x-sign', '1', '--y-sign', '1',
    '--aim-y', str(H.AIM), '--heartbeat-ms', '600', '--input-timeout-ms', '250',
    *F.gap.PD_OPTIONS, '--catchup-speed', '0', '--observation-grace-ms', '0',
)


def stimulus(ns):
    if not 0 <= ns < END_NS:
        raise ValueError('sample outside the fixed profile')
    for phase, start, end, first, last in PROFILE:
        if ns < end * 1_000_000:
            elapsed = ns - start * 1_000_000
            if phase == 'jitter':
                return phase, (0., 4., 0., -4.)[(elapsed // 80_000_000) % 4]
            return phase, first + (last - first) * elapsed / ((end - start) * 1_000_000)


def parse_serial(raw):
    command = json.loads(raw, object_pairs_hook=unique_fields)
    if type(command) is not dict or any(type(v) is not int for v in command.values()):
        raise ValueError('unexpected serial schema')
    if command == {'T': 135}:
        return (0, 0, 0)
    if set(command) == {'T', 'X', 'Y', 'SPD'} and command['T'] == 141:
        x, y, speed = (command[key] for key in ('X', 'Y', 'SPD'))
        if x in (-1, 1) and y == 0 and 1 <= speed <= CAP:
            return x, y, speed
    raise ValueError('serial command violates horizontal-only SPD<=7 guard')


def unique_fields(fields):
    result = dict(fields)
    if len(result) != len(fields):
        raise ValueError('duplicate serial command field')
    return result


class TimedRegistration(F.Registration):
    def __init__(self, capture, background_bottom):
        super().__init__(background_bottom)
        self.capture = capture

    def update(self, raw):
        row = self.capture.rows[-1]
        row['registration_started_ns'] = time.monotonic_ns()
        cpu_start = time.thread_time_ns()
        try:
            return super().update(raw)
        finally:
            row['registration_finished_ns'] = time.monotonic_ns()
            row['registration_thread_cpu_ns'] = time.thread_time_ns() - cpu_start


class TimedCapture(F.Capture):
    def __init__(self, camera, tap, stamped, background_bottom):
        super().__init__(camera, tap, stamped, background_bottom)
        self.registration = TimedRegistration(self, background_bottom)

    def record_frame(self, frame, register):
        dequeued = time.monotonic_ns()
        row, raw = super().record_frame(frame, register)
        row['capture_dequeued_ns'] = dequeued
        row['raw_checks_finished_ns'] = time.monotonic_ns()
        return row, raw

    def reset_registration_for_stop_view(self):
        super().reset_registration_for_stop_view()
        self.registration = TimedRegistration(self, self.registration.bottom)


class SyntheticCamera(F.SyntheticCamera):
    def __init__(self, tap):
        self.frames = []
        self.error = None
        H.FakeCam.__init__(self, tap, K=0., L=DRY_DELAY_S, fps=20,
                          curve=DRY_RATE_CURVE, tau=DRY_TAU_S)


class GuardedTap(F.PulseTap):
    """Relay only validated motion. STOP always has a direct, independent route."""
    def __init__(self, device):
        super().__init__(device, seconds=RUN_SECONDS)
        self.active_block = None
        self.scene_x = (0., 0.)
        self.current = (0, 0, 0)

    def arm(self, block):
        with self.lk:
            if self.fault or self.write_failed or not self.th.is_alive():
                raise H.Abort(self.fault or 'serial guard unavailable')
            if not 0 <= time.monotonic_ns() - self.last_sample_ns <= F.FRAME_STALE_NS:
                raise H.Abort('cannot arm without a fresh registration')
            self.active_block = block

    def stop_and_disarm(self, reason):
        with self.lk:
            self.active_block = None
            self.current = (0, 0, 0)
            if not self.stop(reason):
                raise H.Abort('direct STOP failed; use physical stop')

    def check_outward(self, direction):
        if direction and max(-direction * v for v in self.scene_x) >= SCENE_BRAKE_PX:
            raise H.Abort('outward motion approached the common scene limit')

    def scene(self, row):
        row['scene_check_started_ns'] = time.monotonic_ns()
        with self.lk:
            row['scene_lock_acquired_ns'] = time.monotonic_ns()
            self.scene_x = (row['template_px'][0], row['lk_px'][0])
            try:
                self.check_outward(self.current[0])
            except H.Abort as exc:
                self.fail(str(exc))
                raise
            finally:
                row['scene_check_finished_ns'] = time.monotonic_ns()

    def forward(self, raw, block):
        try:
            command = parse_serial(raw)
        except (ValueError, UnicodeError) as exc:
            with self.lk:
                self.rows.append(dict(ns=time.monotonic_ns(), block=block, rejected=raw.hex(), error=str(exc)))
            self.fail(str(exc))
            return
        with self.lk:
            if self.fault or self.active_block != block:
                self.rows.append(dict(ns=time.monotonic_ns(), block=block, command=command,
                                      reason='bridge_blocked'))
                return
            if command[0]:
                if not 0 <= time.monotonic_ns() - self.last_sample_ns <= F.FRAME_STALE_NS:
                    self.fail('refused motion without a recent unique registered frame')
                    return
                try:
                    self.check_outward(command[0])
                except H.Abort as exc:
                    self.fail(str(exc))
                    return
            payload = raw + b'\n' if command[0] else F.STOP_WIRE
            row = self.tx(payload, 'bridge_motion' if command[0] else 'bridge_stop', command, block=block)
            if not row['ok']:
                self.fail('bridge command serial write failed')
            else:
                self.current = command
                # With deadline == the watchdog deadline, PulseTap.watch checks
                # freshness but cannot take its timed nominal-STOP branch.
                self.pulse = (block, self.deadline) if command[0] else None
                if not command[0]:
                    self.last_stop_ns = row['write_end_ns']
                self.wake.set()


class Bridge:
    def __init__(self, binary, tap, block, mode, log):
        import pty
        import tty

        self.tap, self.block = tap, block
        self.protocol = []
        self.ready = threading.Event()
        self.observation = threading.Event()
        self.selection = threading.Event()
        self.arm_reply = None
        self.armed = False
        self.closing = threading.Event()
        self.master, self.slave = pty.openpty()
        tty.setraw(self.slave)
        options = (*BRIDGE_OPTIONS, '--predictive-brake', str(mode))
        self.options = options
        try:
            self.child = subprocess.Popen(
                [str(binary), '--device', os.ttyname(self.slave), *options],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, bufsize=0,
                env=dict(os.environ, LWIR_CONTROL_TRACE='1'),
            )
        except BaseException:
            os.close(self.master)
            os.close(self.slave)
            raise
        self.reader = threading.Thread(target=self.read_protocol, daemon=True)
        self.relay = threading.Thread(target=self.relay_serial, daemon=True)
        self.reader.start()
        self.relay.start()

    def note(self, direction, line, ns=None):
        self.protocol.append(dict(ns=time.monotonic_ns() if ns is None else ns,
                                  block=self.block, direction=direction, line=line))

    def send(self, line):
        self.check()
        payload = (line + '\n').encode('ascii')
        ns = time.monotonic_ns()
        if self.child.stdin.write(payload) != len(payload):
            raise H.Abort('short bridge input write')
        self.note('send', line, ns)
        return ns

    def check(self):
        if self.tap.fault or self.child.poll() is not None or not self.relay.is_alive():
            raise H.Abort(self.tap.fault or 'bridge or serial relay stopped')

    def read_protocol(self):
        try:
            for raw in iter(self.child.stdout.readline, b''):
                line = raw.decode('ascii').strip()
                self.note('receive', line)
                if line == 'READY':
                    self.ready.set()
                elif line == 'OBSERVATION_STATUS 1':
                    self.observation.set()
                elif line.startswith('R A '):
                    self.arm_reply = line
                    self.armed = line == 'R A OK'
                    self.selection.set()
                elif line.startswith('S ') and not self.closing.is_set():
                    automatic, lost, _, error, track = map(int, line.split()[1:6])
                    if error or (self.armed and (not automatic or lost or track != H.TID)):
                        self.tap.fail('bridge reported lost, wrong ID, disabled, or error')
        except (OSError, ValueError) as exc:
            if not self.closing.is_set():
                self.tap.fail('bridge status read failed: ' + str(exc))
        finally:
            if not self.closing.is_set():
                self.tap.fail('unexpected bridge protocol EOF')

    def relay_serial(self):
        pending = b''
        try:
            while not self.closing.is_set():
                ready, _, _ = select.select([self.master], [], [], .005)
                if not ready:
                    continue
                pending += os.read(self.master, 4096)
                while b'\n' in pending:
                    row, pending = pending.split(b'\n', 1)
                    if len(row) > 1024:
                        raise ValueError('oversized bridge serial line')
                    if row.strip():
                        self.tap.forward(row, self.block)
                if len(pending) > 1024:
                    raise ValueError('oversized partial bridge serial line')
        except (OSError, ValueError) as exc:
            if not self.closing.is_set():
                self.tap.fail('serial relay failed: ' + str(exc))

    def close(self):
        self.closing.set()
        self.child.stdin.close()
        try:
            self.child.wait(timeout=.8)
        except subprocess.TimeoutExpired:
            self.child.terminate()
            try:
                self.child.wait(timeout=.8)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=.8)
        self.reader.join(.5)
        self.relay.join(.5)
        self.child.stdout.close()
        os.close(self.master)
        os.close(self.slave)
        if self.child.returncode != 0 or self.reader.is_alive() or self.relay.is_alive():
            raise H.Abort('bridge failed to close cleanly')


def verify_camera_idle(camera_path, own_pid=None):
    for entry in Path('/proc').glob('[0-9]*/exe'):
        try:
            name = Path(os.readlink(entry).removesuffix(' (deleted)')).name
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(entry.parent.name) != own_pid and (name == camera_path.name or name.startswith('cam_pipe')):
            raise H.Abort('another camera process is running')


def total_variation(values):
    return float(sum(abs(b - a) for a, b in zip(values, values[1:])))


def command_counts(commands):
    starts = stops = reversals = restarts = 0
    prior = last_motion = 0
    max_speed = 0
    for x, _, speed in commands:
        max_speed = max(max_speed, speed)
        if not x and prior:
            stops += 1
        if x and not prior:
            starts += 1
            restarts += int(bool(last_motion))
        if x and last_motion and x != last_motion:
            reversals += 1
        if x:
            last_motion = x
        prior = x
    return dict(motion_starts=starts, stop_transitions=stops, reversals=reversals,
                stop_restarts=restarts, maximum_speed=max_speed)


def weighted_percentile(values, weights, fraction):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, fraction * cumulative[-1], side='left')
    return float(values[order[index]])


def metrics(samples, commands):
    by_phase = {phase: [r for r in samples if r['phase'] == phase] for phase, *_ in PROFILE}
    if any(len(rows) < 2 for rows in by_phase.values()):
        raise H.Abort('profile phase missing enough unique observations')
    times = np.asarray([r['elapsed_ns'] for r in samples] + [END_NS], dtype=np.int64)
    weights = np.diff(times)
    if times[0] != 0 or np.any(weights <= 0) or np.max(weights) > F.FRAME_STALE_NS:
        raise H.Abort('profile timing coverage is incomplete or nonmonotonic')
    errors = np.abs(np.asarray([row['error_x_px'] for row in samples]))
    # Zero-order hold, truncated exactly at the profile and settle boundaries.
    settle_weights = np.maximum(0, np.minimum(times[1:], 4_000_000_000) -
                                np.maximum(times[:-1], 3_700_000_000))
    in_settle = settle_weights > 0
    if np.count_nonzero(in_settle) < 2:
        raise H.Abort('insufficient final settle observations')
    holds = {}
    for name in ('static', 'dwell', 'settle', 'jitter'):
        values = [r['template_px'][0] for r in by_phase[name]]
        tv = total_variation(values)
        net = abs(values[-1] - values[0])
        holds[name] = dict(total_path_px=tv, net_path_px=net, excess_path_px=max(0., tv - net))
    return dict(
        sample_count=len(samples), center_abs_mae_px=float(np.average(errors, weights=weights)),
        center_abs_p95_px=weighted_percentile(errors, weights, .95),
        center_deadband_occupancy=float(np.average(errors <= DEADBAND_PX, weights=weights)),
        final_settle_abs_error_px=weighted_percentile(errors[in_settle], settle_weights[in_settle], .5),
        peak_abs_error_px=float(np.max(errors)), outside_110px_count=int(np.count_nonzero(errors > TARGET_LIMIT_PX)),
        outside_110px_duration_ms=float(weights[errors > TARGET_LIMIT_PX].sum() / 1e6),
        total_image_path_px=total_variation([row['template_px'][0] for row in samples]),
        hold_path=holds, stationary_hold_excess_path_px=sum(holds[p]['excess_path_px'] for p in ('static', 'dwell', 'settle')),
        publish_age_ms_p95=float(np.percentile([(row['sent_ns'] - row['observation_ns']) / 1e6 for row in samples], 95)),
        **command_counts(commands),
    )


def read_trace(path, published, mode, stop_rows):
    trace = []
    matched_stops = set()
    samples = {row['frame_id']: row for row in published}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        if row.get('event') != 'predictive_brake':
            continue
        obs = samples.get(row.get('frame_id'))
        if (not mode or obs is None or row.get('captured_ns') != obs['observation_ns']
                or row.get('track_id') != H.TID or not 0 < row.get('reserve_px', 0) <= 4
                or type(row.get('sent')) is not bool):
            raise H.Abort('predictive brake trace did not match a valid published observation')
        downstream = [stop for stop in stop_rows
                      if stop['write_start_ns'] not in matched_stops
                      and 0 <= stop['write_start_ns'] - obs['sent_ns'] <= 100_000_000]
        if row['sent'] and not downstream:
            raise H.Abort('predictive STOP trace lacks a nearby forwarded STOP')
        verified_stop = downstream[0]['write_start_ns'] if row['sent'] else None
        if verified_stop is not None:
            matched_stops.add(verified_stop)
        trace.append(dict(row, downstream_stop_write_ns=verified_stop))
    return trace


def compare(blocks):
    pairs = []
    for a, b in ((blocks[0], blocks[1]), (blocks[3], blocks[2])):
        base, candidate = a['metrics'], b['metrics']
        center_ok = (candidate['center_deadband_occupancy'] >= base['center_deadband_occupancy']
                     and candidate['final_settle_abs_error_px'] <= base['final_settle_abs_error_px'] + 1.)
        safety_ok = (candidate['outside_110px_count'] <= base['outside_110px_count']
                     and candidate['stop_restarts'] <= base['stop_restarts'])
        less_excess = candidate['stationary_hold_excess_path_px'] < base['stationary_hold_excess_path_px']
        pairs.append(dict(blocks_in_time_order=sorted((a['block'], b['block'])),
                          center_ok=center_ok, safety_ok=safety_ok, lower_excess=less_excess,
                          predictive_stops=b['predictive_stops']))
    active = any(pair['predictive_stops'] for pair in pairs)
    if not active:
        verdict = 'candidate_no_op'
    elif all(p['center_ok'] and p['safety_ok'] and p['lower_excess'] for p in pairs):
        verdict = 'provisional_improvement'
    else:
        verdict = 'provisional_improvement_not_established'
    return dict(verdict=verdict, chronological_pairs=pairs)


def camera_stamp(row):
    return row['sdk_recv_ns'] if row['sdk_recv_ns'] is not None else row['arrival_ns']


def publish(bridge, row, origin, phase, elapsed_ns, offset):
    row['publish_attempt_ns'] = time.monotonic_ns()
    stamp = camera_stamp(row)
    error = offset + row['template_px'][0] - origin
    if abs(error) > TARGET_LIMIT_PX:
        raise H.Abort('virtual target exceeded 110 px framing limit')
    x1 = H.W / 2 + error - H.BW / 2
    y1 = H.H / 2 - H.AIM * H.BH
    remaining = stamp + LATENCY_NS - time.monotonic_ns()
    if remaining > 0:
        time.sleep(remaining / F.S)
    row['publish_deadline_checked_ns'] = time.monotonic_ns()
    if not LATENCY_NS <= row['publish_deadline_checked_ns'] - stamp <= MAX_PUBLISH_AGE_NS:
        raise H.Abort('40 ms observation pipeline exceeded its scheduling tolerance')
    sent = bridge.send(f"F {row['frame_id']} {stamp} 1 {H.TID} {x1:.4f} {y1:.4f} {H.BW:.4f} {H.BH:.4f} .900 OBS 1")
    if not LATENCY_NS <= sent - stamp <= MAX_PUBLISH_AGE_NS:
        raise H.Abort('publication timestamp outside observation pipeline budget')
    row.update(control_phase=phase, observation_ns=stamp, sent_ns=sent, error_x_px=error,
               elapsed_ns=elapsed_ns, target_offset_px=offset)
    return dict(row, phase=phase)


def parse_cpus(text):
    try:
        parts = text.split(',')
        if not all(part.isascii() and part.isdecimal() for part in parts):
            raise ValueError
        cpus = frozenset(map(int, parts))
        if len(cpus) != len(parts):
            raise ValueError
        return cpus
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected unique comma-separated CPU numbers') from exc


@contextmanager
def profile_affinity(cpus, record):
    original = os.sched_getaffinity(0)
    settings = dict(requested=sorted(cpus) if cpus is not None else None,
                    caller_original=sorted(original))
    record['profile_cpu_affinity'] = settings
    if cpus is not None:
        os.sched_setaffinity(0, cpus)
    try:
        actual = os.sched_getaffinity(0)
        settings['caller_profile'] = sorted(actual)
        if cpus is not None and actual != cpus:
            raise H.Abort('profile thread CPU affinity did not match the requested CPUs')
        yield
    finally:
        if cpus is not None:
            os.sched_setaffinity(0, original)
        settings['caller_restored'] = sorted(os.sched_getaffinity(0))
        if settings['caller_restored'] != settings['caller_original']:
            raise H.Abort('could not restore profile thread CPU affinity')


def run_block(args, capture, tap, block, mode, output, record):
    trace_path = output / f'bridge_{block}.log'
    capture.trial, capture.phase = block, 'block_origin'
    origin_row = capture.next()
    tap.scene(origin_row)
    if any(np.any(np.abs(origin_row[key]) >= F.START_FOV) for key in ('template_px', 'lk_px')):
        raise H.Abort('block origin outside common 65 px horizontal / 35 px vertical limit')
    origin = origin_row['template_px'][0]
    record.update(origin_frame_id=origin_row['frame_id'], origin_template_px=origin_row['template_px'],
                  origin_lk_px=origin_row['lk_px'])
    published, scored = [], []
    with trace_path.open('wb') as log, ExitStack() as affinity:
        bridge = Bridge(args.bridge, tap, block, mode, log)
        try:
            affinity.enter_context(profile_affinity(args.profile_cpus, record))
            record['bridge_options'] = bridge.options
            # Wait for the local process while preserving sequential camera reads.
            while not bridge.ready.is_set() or not bridge.observation.is_set():
                bridge.check()
                row = capture.next()
                tap.scene(row)
                if row['arrival_ns'] - origin_row['arrival_ns'] > 1_000_000_000:
                    raise H.Abort('bridge did not advertise READY and OBS 1')
            bridge.send('P')
            capture.phase = 'preselect'
            for _ in range(2):
                row = capture.next()
                tap.scene(row)
                published.append(publish(bridge, row, origin, 'preselect', 0, 0.))
            tap.arm(block)
            bridge.send(f'A {H.TID}')
            for _ in range(5):
                if bridge.selection.is_set():
                    break
                row = capture.next()
                tap.scene(row)
                published.append(publish(bridge, row, origin, 'preselect', 0, 0.))
            if bridge.arm_reply != 'R A OK':
                raise H.Abort('single virtual ID selection was not accepted')
            capture.phase = 'profile'
            row = capture.next()
            start = camera_stamp(row)
            last_heartbeat = time.monotonic_ns()
            while camera_stamp(row) - start < END_NS:
                tap.scene(row)
                elapsed = camera_stamp(row) - start
                if time.monotonic_ns() - last_heartbeat > 150_000_000:
                    last_heartbeat = bridge.send('P')
                    row['heartbeat_send_ns'] = last_heartbeat
                    row['heartbeat_write_finished_ns'] = time.monotonic_ns()
                sample = publish(bridge, row, origin, *phase_args(elapsed))
                published.append(sample)
                scored.append(sample)
                row = capture.next()
            tap.scene(row)
            tap.stop_and_disarm('block_stop')
            stop_ns = tap.last_stop_ns
            record['block_stop_write_end_ns'] = stop_ns
            bridge.closing.set()
            affinity.close()
            dwell = capture.window(stop_ns, F.DWELL_NS, 'post_stop')
            record['post_stop_span_px'] = F.stable([r for r in dwell if r['arrival_ns'] >= stop_ns + 650_000_000])
            record['post_stop_verified'] = True
        except BaseException as exc:
            tap.fail(str(exc))
            raise
        finally:
            try:
                if tap.active_block == block:
                    tap.fail('block aborted before verified STOP')
                    tap.stop_and_disarm('abort_direct_stop')
            finally:
                try:
                    try:
                        affinity.close()
                    finally:
                        bridge.close()
                finally:
                    write_rows(output / f'protocol_{block}.jsonl', bridge.protocol)
    with tap.lk:
        serial = [r for r in tap.rows if r.get('block') == block and r.get('ok')]
    if any(r['reason'] == 'bridge_motion' and r['write_start_ns'] < scored[0]['sent_ns'] for r in serial):
        raise H.Abort('motion occurred before the common stimulus')
    stop_rows = [r for r in serial if r['reason'] == 'bridge_stop']
    record['predictive_brake_trace'] = read_trace(trace_path, published, mode, stop_rows)
    record['predictive_stops'] = sum(r['sent'] for r in record['predictive_brake_trace'])
    record['metrics'] = metrics(scored, [r['command'] for r in serial])


def phase_args(elapsed):
    phase, offset = stimulus(elapsed)
    return phase, elapsed, offset


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row) + '\n')


def save_synthetic_unprocessed(capture):
    if capture is None or capture.stamped:
        return
    processed = {r['frame_id'] for r in capture.rows}
    for fid, arrival, raw in capture.camera.frames:
        if fid not in processed:
            capture.raw.append((fid, arrival, raw))
            capture.rows.append(dict(frame_id=fid, sdk_recv_ns=None, arrival_ns=arrival, exposure_ns=None,
                                     phase='unprocessed', valid=False, duplicate_full_raw16=None))


def restore_fixed_limits(limits):
    F.CONTROL_SECONDS, F.MAX_FRAMES = limits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--bridge', required=True, type=Path)
    parser.add_argument('--camera', type=Path)
    parser.add_argument('--background-bottom', type=int, default=360)
    parser.add_argument('--profile-cpus', type=parse_cpus,
                        help='optional CPU list for the profile processing thread only')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--attended', action='store_true')
    args = parser.parse_args(argv)
    args.bridge, out = args.bridge.resolve(), args.out.resolve()
    repo = Path(__file__).resolve().parents[4]
    if sys.platform != 'linux' or not args.bridge.is_file() or not os.access(args.bridge, os.X_OK):
        parser.error('requires Linux and a named executable bridge binary')
    if args.profile_cpus is not None and not args.profile_cpus <= os.sched_getaffinity(0):
        parser.error('--profile-cpus must be a nonempty subset of current allowed CPUs')
    if args.execute != args.attended or args.execute != (args.camera is not None):
        parser.error('physical mode requires --execute --attended and a named --camera; all three are physical-only')
    if out.exists() or out.is_relative_to(repo):
        parser.error('use a new private output directory outside the repository')
    if not 180 <= args.background_bottom <= 360 or args.background_bottom % 2:
        parser.error('background bottom must be even and within 180..360')
    if args.execute:
        args.camera = args.camera.resolve()
        H.CAMP = str(args.camera)
        F.gap.real_preflight(parser, H)
        verify_camera_idle(args.camera)
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True)
    if shutil.disk_usage(out).free < MAX_FRAMES * (F.PIXEL_BYTES + 16384):
        parser.error('insufficient space for bounded private RAW16 evidence')
    cv2.setNumThreads(1)
    result = dict(completed=False, dry=not args.execute, order=ORDER, profile_ms=PROFILE,
                  camera_observation='SDK receipt when stamped, otherwise synthetic completed frame',
                  bridge_sha256=hashlib.sha256(args.bridge.read_bytes()).hexdigest(),
                  bridge_path=str(args.bridge), input_latency_target_ms=40, max_publication_age_ms=65,
                  background_bottom=args.background_bottom, control_deadline_s=RUN_SECONDS, blocks=[])
    if not args.execute:
        result['synthetic_plant'] = dict(rate_curve_px_s=DRY_RATE_CURVE, delay_s=DRY_DELAY_S, tau_s=DRY_TAU_S)
    if args.execute:
        result.update(camera_path=str(args.camera), camera_sha256=hashlib.sha256(args.camera.read_bytes()).hexdigest())
    tap = camera = capture = None

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'signal {signum}')

    with ExitStack() as cleanup:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            cleanup.callback(signal.signal, sig, signal.signal(sig, interrupted))
        cleanup.callback(restore_fixed_limits, (F.CONTROL_SECONDS, F.MAX_FRAMES))
        F.CONTROL_SECONDS, F.MAX_FRAMES = CAMERA_SECONDS, MAX_FRAMES
        with (out / 'camera.log').open('wb') as log:
            try:
                tap = GuardedTap(H.DEV if args.execute else None)
                tap.connect()
                camera = F.StampedCamera(args.camera, log) if args.execute else SyntheticCamera(tap)
                capture = TimedCapture(camera, tap, args.execute, args.background_bottom)
                first = capture.next(timeout=2., register=False)['arrival_ns']
                while capture.last_arrival < first + F.WARMUP_NS:
                    capture.next(timeout=.30, register=False)
                baseline = capture.window(time.monotonic_ns(), F.BASELINE_NS, 'baseline')
                result['baseline_span_px'] = F.stable(baseline)
                for block, variant in enumerate(ORDER, 1):
                    if time.monotonic_ns() + END_NS + F.DWELL_NS + 2 * F.S >= tap.deadline:
                        raise H.Abort('insufficient watchdog time for a full block plus STOP')
                    if args.execute:
                        F.verify_idle()
                        verify_camera_idle(args.camera, camera.p.pid)
                    record = dict(block=block, predictive_brake=variant)
                    result['blocks'].append(record)
                    run_block(args, capture, tap, block, variant, out, record)
                result['comparison'] = compare(result['blocks'])
            except BaseException as exc:
                result['error'] = repr(exc)
            finally:
                if tap is not None and tap.th.ident is not None:
                    try:
                        tap.stop_and_disarm('final_stop')
                        if capture is None:
                            raise H.Abort('camera unavailable to verify final STOP')
                        stop_ns = tap.last_stop_ns
                        capture.trial = None
                        if result.get('error') or tap.fault:
                            capture.reset_registration_for_stop_view()
                        dwell = capture.window(stop_ns, F.DWELL_NS, 'final_stop', recovery=True)
                        result['final_stop_span_px'] = F.stable(
                            [r for r in dwell if r['arrival_ns'] >= stop_ns + 650_000_000])
                        result['final_stop_verified'] = True
                    except BaseException as exc:
                        result['final_stop_error'] = repr(exc)
                    finally:
                        result.update(watchdog_fault=tap.fault, serial_write_failed=tap.write_failed)
                if tap is not None:
                    tap.close()
                if camera is not None:
                    try:
                        result['camera_exit'] = camera.close()
                        if result['camera_exit'] != 0:
                            raise H.Abort('camera returned a nonzero exit status')
                    except BaseException as exc:
                        result['camera_error'] = repr(exc)
        result['completed'] = bool(
            len(result['blocks']) == len(ORDER) and all(r.get('post_stop_verified') for r in result['blocks'])
            and result.get('final_stop_verified') and 'comparison' in result
            and not any(result.get(k) for k in ('error', 'final_stop_error', 'watchdog_fault',
                                                'serial_write_failed', 'camera_error')))
        try:
            save_synthetic_unprocessed(capture)
            F.save_output(out, capture, tap, result)
        except Exception as exc:
            result.update(completed=False, save_error=repr(exc))
            (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['completed'] else 3


if __name__ == '__main__':
    sys.exit(main())
