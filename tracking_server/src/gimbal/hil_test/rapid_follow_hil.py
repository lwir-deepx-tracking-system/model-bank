#!/usr/bin/env python3
"""Camera-framing A/B: v27 catch-up 0 vs 7, no LEAD, one virtual track.

The default uses the actual bridge with a PTY and synthetic RAW16. Use a new
private output folder. Real mode requires --execute --attended on the board.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import sys
import threading
import time

import cv2
import numpy as np

import observation_gap_hil as gap
import predictive_ab_hil as ref

H = ref.H
MAX_SECONDS = 90.
CAP = 10
LATENCY_NS = 40_000_000
ORIGIN_LIMIT = 22.
ORIGIN_WARN = 4.
SCENE_LIMIT = np.array([130., 65.])
BRAKE_AT = np.array([95., 35.])
DEADBAND_X_PX, DEADBAND_Y_PX = 19.2, 14.4
DRY_PLANTS = {
    'legacy': dict(delay_s=.08, tau_s=.05, px_s_per_spd=6., speed_curve=None),
    'reference': dict(delay_s=.12, tau_s=.05, px_s_per_spd=6.,
                      speed_curve=((1, 156.), (2, 161.), (3, 160.), (5, 172.), (7, 237.), (10, 356.))),
}
PROFILE_NAMES = ('rapid-reversal100', 'rapid-pulse60')


@dataclass(frozen=True)
class Phase:
    name: str
    start: float
    end: float
    start_px: float
    end_px: float
    rapid: bool


@dataclass(frozen=True)
class Profile:
    name: str
    velocity: int
    amplitude: float
    phases: tuple[Phase, ...]

    @property
    def end(self):
        return self.phases[-1].end


def profile_for(name, velocity=None):
    if name == 'rapid-pulse60':
        if velocity not in (None, 400):
            raise ValueError('rapid-pulse60 has a fixed 400 px/s outward ramp')
        velocity, amplitude = 400, 60.
        movement = [('ramp', .15, 60., True), ('stop_dwell', 1., 60., False),
                    ('slow_return', .60, 0., False)]
    else:
        velocity, amplitude = velocity or 240, 100.
        ramp = amplitude / velocity
        movement = [('ramp', ramp, 100., True), ('stop_dwell', .55, 100., False),
                    ('reversal', 2 * ramp, -100., True), ('reverse_dwell', .55, -100., False),
                    ('return', ramp, 0., True)]
    phases = []
    start, offset = 0., 0.
    for phase_name, duration, end_px, rapid in [
        ('static', .60, 0., False), *movement, ('settle', .80, 0., False), ('jitter', 1.10, 0., False),
    ]:
        end = start + duration
        phases.append(Phase(phase_name, start, end, offset, end_px, rapid))
        start, offset = end, end_px
    return Profile(name, velocity, amplitude, tuple(phases))


def stimulus(t, p):
    for phase in p.phases:
        if t < phase.end:
            elapsed = t - phase.start
            if phase.name == 'jitter':
                return phase.name, (0., 4., 0., -4.)[int(elapsed / .08) % 4]
            return phase.name, phase.start_px + (phase.end_px - phase.start_px) * elapsed / (phase.end - phase.start)
    raise ValueError('stimulus time exceeds profile')


class GuardedTap(ref.StrictTap):
    """Keep the existing STOP and watchdog, reject unsafe frames before write."""
    def trip_once(self, reason):
        with self.lk:
            if self.trip is None:
                self.trip = reason
                self.estop()

    def forward(self, raw):
        ns = time.monotonic_ns()
        error = None
        try:
            cmd = json.loads(raw)
            if type(cmd) is not dict or any(type(v) is not int for v in cmd.values()):
                raise ValueError('unexpected serial schema')
            typ = cmd.get('T')
            if typ == 135 and set(cmd) == {'T'}:
                motion = (0, 0, 0)
            elif typ == 141 and set(cmd) == {'T', 'X', 'Y', 'SPD'}:
                motion = (cmd['X'], cmd['Y'], cmd['SPD'])
                x, y, speed = motion
                if x not in (-1, 0, 1) or y not in (-1, 0, 1) or not (x or y) or not 1 <= speed <= CAP:
                    raise ValueError('serial motion violates direction/speed guard')
            else:
                raise ValueError('unexpected serial command')
        except (ValueError, UnicodeError) as exc:
            self.note(dict(ns=ns, rejected=raw.decode('latin1'), reason=str(exc)))
            self.trip_once(str(exc))
            return
        with self.lk:
            blocked = self.block or bool(self.trip)
            if not blocked:
                try:
                    if self.fd is not None and os.write(self.fd, raw + b'\n') != len(raw) + 1:
                        raise OSError('short serial write')
                    self.last = motion
                    self.maxspd = max(self.maxspd, motion[2])
                    self.nstop += int(typ == 135)
                except OSError as exc:
                    error = repr(exc)
                    self.trip_once('device write failed: ' + error)
        self.note(dict(ns=ns, cmd=cmd, blocked=blocked, werr=error))

    def run(self):
        try:
            while not self.done:
                device = self.fd
                try:
                    ready, _, _ = select.select([self.master] + ([] if device is None else [device]), [], [], .025)
                except (OSError, ValueError):
                    ready = []
                self.watch()
                if self.master in ready:
                    self.buf += os.read(self.master, 4096)
                    while b'\n' in self.buf:
                        line, self.buf = self.buf.split(b'\n', 1)
                        if line.strip(): self.forward(line)
                    if len(self.buf) > 1024:
                        self.trip_once('oversized serial command')
                        self.buf = b''
                if device is not None and device in ready and self.fd == device:
                    try:
                        data = os.read(device, 4096)
                        if not data: raise OSError('serial line hung up')
                        self.note(dict(ns=time.monotonic_ns(), rx=data.decode('latin1')))
                    except BlockingIOError:
                        pass
        except (OSError, ValueError) as exc:
            self.trip_once('serial tap failed: ' + repr(exc))

    def close(self):
        already_closed = self.done
        super().close()
        if not already_closed:
            for fd in (self.master, self.slave):
                os.close(fd)

    def reopen(self):
        # A failed emergency STOP may reopen the same stable serial identity,
        # never another device that happens to match the model glob.
        if not self.real: return None
        if self.fd is not None:
            try: os.close(self.fd)
            except OSError: pass
        self.fd = None
        try:
            self.fd = H.open_device(self.dev)
        except OSError:
            return None
        self.nreopen += 1
        return self.dev


class Bridge(H.Bridge):
    def __init__(self, tap, opts, log, audit):
        self.tap, self.audit = tap, audit
        self.armed = False
        self.obs = threading.Event()
        super().__init__(tap, opts, log, 'v27')

    def run(self):
        try:
            for raw in iter(self.p.stdout.readline, b''):
                line = raw.decode().strip()
                self.audit.append(dict(ns=time.monotonic_ns(), direction='receive', line=line))
                if line == 'READY': self.ready.set()
                elif line == 'OBSERVATION_STATUS 1': self.obs.set()
                elif line.startswith('R '):
                    if line == 'R A OK': self.armed = True
                    self.ans.append(line)
                elif line.startswith('S '):
                    self.state = tuple(map(int, line.split()[1:6]))
                    automatic, lost, _, error, track = self.state
                    if error or (self.armed and (not automatic or lost or track != H.TID)):
                        self.tap.trip_once('unexpected bridge disabled/lost/error/wrong-ID state')
        except (OSError, ValueError) as exc:
            self.tap.trip_once('unreadable bridge status: ' + repr(exc))

    def send(self, text):
        ns = time.monotonic_ns()
        self.audit.append(dict(ns=ns, direction='send', line=text))
        payload = (text + '\n').encode()
        if self.p.stdin.write(payload) != len(payload):
            raise H.Abort('short bridge input write')
        return ns


def no_other_camera():
    own = os.getpid()
    for path in Path('/proc').glob('[0-9]*/comm'):
        if int(path.parent.name) == own:
            continue
        try:
            if path.read_text().strip() == 'cam_pipe':
                raise H.Abort('another cam_pipe is running')
        except (FileNotFoundError, ProcessLookupError):
            pass


def verify_idle():
    if H.other_servers() or gap.tracking_port_in_use():
        raise H.Abort('tracking server running or port 8090 occupied')


def require_exclusive_serial():
    import fcntl
    opener = H.open_device

    def open_exclusive(device):
        fd = opener(device)
        try:
            fcntl.ioctl(fd, H.termios.TIOCEXCL)
        except BaseException:
            os.close(fd)
            raise
        return fd

    H.open_device = open_exclusive


# Reuse OBS harness output setup, two-image-estimator check and bounded close.
class Session(gap.make_session(ref)):
    def start_camera(self):
        # Supply the selected dry plant only during the parent's camera setup.
        original_tap, original_camera = ref.StrictTap, H.FakeCam

        def dry_camera(tap, _speed, _delay, *, tau):
            settings = DRY_PLANTS[self.a.dry_plant]
            return original_camera(tap, settings['px_s_per_spd'], settings['delay_s'],
                                   curve=settings['speed_curve'], tau=settings['tau_s'])

        try:
            ref.StrictTap = GuardedTap
            if self.a.dry: H.FakeCam = dry_camera
            super().start_camera()
        finally:
            ref.StrictTap, H.FakeCam = original_tap, original_camera

    def frame(self):
        frame = super().frame()
        # A second measured coordinate cannot overrun the scene limit either.
        # Anticipate the historical braking distance while already moving
        # outward; the supervisor's excursion limits still apply at all times.
        shift = frame[2]
        flow = self.check.total
        if np.any(np.abs(flow) > SCENE_LIMIT):
            raise H.Abort('optical-flow excursion limit')
        serial_x, serial_y, _ = self.tap.last
        # The scene moves left for a rightward camera command and down for an
        # upward command. These signs also match the existing FakeCam model.
        scene_direction = np.array([-serial_x, serial_y])
        if np.any(np.maximum(scene_direction * shift, scene_direction * flow) >= BRAKE_AT):
            raise H.Abort('approached scene limit while moving outward; stopped for braking margin')
        return frame

    def stationary_after_stop(self, tag):
        self.post_stop_attempted = True
        if not self.tap.estop():
            raise H.Abort('STOP write failed')
        begin = time.monotonic()
        late, previous = [], None
        while time.monotonic() - begin < 1.6:
            raw = self.cam.get(self.last, .30)
            if raw is None: raise H.Abort('no image after STOP')
            self.last = raw[0]
            if not 0 <= time.monotonic_ns() - raw[1] <= 300_000_000:
                raise H.Abort('stale post-STOP image')
            if previous is not None and (raw[1] <= previous[1] or np.array_equal(raw[2], previous[2])):
                raise H.Abort('nonmonotonic or frozen post-STOP image')
            previous = raw
            shift, quality = self.est.update(raw[2])
            flow = self.check.update(raw[2])
            self.S = shift.copy()
            disagreement = float(np.linalg.norm(shift - flow))
            self.audit.append(dict(event='post_stop_image', tag=tag, ns=raw[1], fid=raw[0],
                                   template=shift.tolist(), optical_flow=flow.tolist(),
                                   disagreement_px=disagreement, response=float(quality)))
            if (self.est.bad or disagreement > 8 or
                    np.any(np.maximum(abs(shift), abs(flow)) > SCENE_LIMIT)):
                raise H.Abort('post-STOP registration/disagreement/excursion fault')
            if time.monotonic() - begin >= .8: late.append((shift.copy(), flow.copy()))
        if len(late) < 5: raise H.Abort('too few post-STOP frames')
        span = np.ptp(np.asarray(late), axis=0)
        self.audit.append(dict(event='post_stop', tag=tag, ns=time.monotonic_ns(),
                               span_px=span.tolist(), frames=len(late)))
        if np.max(span) > 4 or self.tap.stop_ever_failed:
            raise H.Abort('post-STOP view moved or STOP write failed earlier')
        self.end_view_checked = True
        return span.tolist()

    def block(self, catchup, initial):
        profile = self.a.profile
        if time.monotonic() + profile.end + 12 >= self.deadline:
            raise H.Abort('insufficient time for next block and verified STOP')
        if not self.a.dry: verify_idle()
        self.name = 'catchup_%d' % catchup
        self.end_view_checked = False
        self.post_stop_attempted = False
        record = dict(catchup_speed=catchup, name=self.name, samples=[])
        self.blocks.append(record)
        options = [*H.BASE, *gap.PD_OPTIONS, '--observation-grace-ms', '100',
                   '--catchup-speed', str(catchup)]
        self.elog.write('--- %s: %s\n' % (self.name, ' '.join(options)))
        self.elog.flush()
        self.br = Bridge(self.tap, options, self.elog, self.audit)
        self.tap.child = self.br.p
        if not self.br.obs.wait(.5): raise H.Abort('bridge did not advertise OBS support')
        self.br.send('P')
        self.lastp = time.monotonic()
        first = self.frame()
        origin = first[2]
        delta = origin - initial
        record.update(origin_px=origin.tolist(), initial_delta_px=delta.tolist(),
                      origin_fid=first[0], origin_ns=first[1])
        if np.max(abs(delta)) > ORIGIN_LIMIT: raise H.Abort('origin mismatch >22 px')
        if np.max(abs(delta)) > ORIGIN_WARN:
            record['origin_warning'] = 'physical origin differs from reference by >4 px'
            print('WARN %s origin delta %s px' % (self.name, delta.tolist()), flush=True)
        if not self.tap.open_line(): raise H.Abort('serial line cannot open after STOP fault')
        previous_fid, previous_ns, previous_raw = first[0], first[1], self.raw_frames[-1][2]

        def next_frame():
            nonlocal previous_fid, previous_ns, previous_raw
            item = self.frame()
            fid, ns, *_ = item
            raw = self.raw_frames[-1][2]
            if fid != previous_fid + 1 or ns <= previous_ns or np.array_equal(raw, previous_raw):
                raise H.Abort('missing/nonmonotonic/frozen RAW16 during publication')
            previous_fid, previous_ns, previous_raw = fid, ns, raw
            return item

        def publish(item, t, phase, offset):
            fid, ns, shift, response = item
            world = np.array([offset, 0.]) - origin
            ex, ey = (world + shift).tolist()
            x, y = 320. + ex, 240. + ey
            box = self.box(x, y)
            if box is None or abs(ex) > 110 or abs(ey) > 55:
                raise H.Abort('virtual target exceeded its error limit')
            x1, y1, x2, y2 = box
            wait_ns = ns + LATENCY_NS - time.monotonic_ns()
            if wait_ns > 0: time.sleep(wait_ns / 1e9)
            if self.tap.trip or self.tap.stop_ever_failed:
                raise H.Abort(self.tap.trip or 'STOP write failed')
            sent = self.br.send('F %d %d 1 %d %.4f %.4f %.4f %.4f .900 OBS 1' %
                               (fid, ns, H.TID, x1, y1, x2 - x1, y2 - y1))
            self.row(phase, t, fid, ns, shift, response, world, 1, (x, y), (ex/320, ey/240))
            record['samples'].append(dict(fid=fid, capture_ns=ns, sent_ns=sent, t=t, phase=phase,
                                          offset_px=offset, ex_px=ex, ey_px=ey,
                                          scene_px=shift.tolist(), flow_px=self.check.total.tolist()))

        for _ in range(3): publish(next_frame(), 0., 'select', 0.)
        self.br.send('A %d' % H.TID)
        for _ in range(7):
            if self.br.ans:
                if self.br.ans.popleft() != 'R A OK': raise H.Abort('single selection refused')
                break
            publish(next_frame(), 0., 'select', 0.)
        else:
            raise H.Abort('no acknowledgement for the single selection')
        sample = next_frame()
        start_ns = sample[1]
        while True:
            t = (sample[1] - start_ns) / 1e9
            if t >= profile.end: break
            publish(sample, t, *stimulus(t, profile))
            sample = next_frame()
        record['stop_cut_requested_ns'] = time.monotonic_ns()
        self.close_bridge()
        record['post_stop_span_px'] = self.stationary_after_stop(self.name)
        record['post_stop_verified'] = self.end_view_checked
        self.f.flush()
        self.slog.flush()
        self.elog.flush()
        return origin

    def compare(self):
        self.deadline = self.t_begin + MAX_SECONDS
        self.post_stop_attempted = False
        self.start_camera()
        warm_until = time.monotonic() + 5
        while time.monotonic() < warm_until:
            frame = self.cam.get(self.last, .30)
            if frame is None: raise H.Abort('camera stalled during warm-up')
            self.last = frame[0]
        stable_until = time.monotonic() + 1
        stable = []
        while time.monotonic() < stable_until:
            stable.append((self.frame()[2], self.check.total.copy()))
        if len(stable) < 5 or np.max(np.ptp(np.asarray(stable), axis=0)) > 4:
            raise H.Abort('starting view is not stationary or has too few frames')
        origin = self.S.copy()
        order = (7, 0) if self.a.candidate_first else (0, 7)
        for self.step_i, speed in enumerate(order, 1):
            actual = self.block(speed, origin)
            if self.step_i == 1: origin = actual


def stats(values):
    if not len(values): return None
    return dict(n=len(values), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(np.max(values)))


def read_evidence(output, blocks, profile):
    import bisect
    from collections import Counter

    commands = []
    with (output / 'serial.jsonl').open(encoding='utf-8') as log:
        for line in log:
            row = json.loads(line)
            if 'cmd' in row and not row.get('blocked') and not row.get('werr'):
                commands.append(row)
    traces, outputs = [], []
    with (output / 'bridge_err.log').open(encoding='utf-8') as log:
        for line in log:
            if line.startswith('{'):
                row = json.loads(line)
                if row.get('event') == 'control_trace': traces.append(row)
                if row.get('event') == 'control_output': outputs.append(row)

    for block in blocks:
        samples = block['samples']
        if not samples: continue
        for index, item in enumerate(samples[1:], 1):
            prev = samples[index - 1]
            dt = (item['capture_ns'] - prev['capture_ns']) / 1e9
            item['scene_x_velocity_px_s'] = (item['flow_px'][0] - prev['flow_px'][0]) / dt
        sent_times = [x['sent_ns'] for x in samples]
        fid_index = {item['fid']: index for index, item in enumerate(samples)}
        mapped = [[] for _ in samples]
        for command in commands:
            if not sent_times[0] <= command['ns'] < block.get('stop_cut_requested_ns', sent_times[-1] + 250_000_000):
                continue
            mapped[bisect.bisect_right(sent_times, command['ns']) - 1].append(command)
        metrics = {}
        phase_starts = {phase.name: phase.start for phase in profile.phases}
        for phase in ('select', *phase_starts):
            which = [i for i, x in enumerate(samples) if x['phase'] == phase]
            seq = [samples[i] for i in which]
            serial = [r for i in which for r in mapped[i]]
            ex = [x['ex_px'] for x in seq]
            sx = [x['scene_px'][0] for x in seq]
            series = [abs(x.get('scene_x_velocity_px_s', 0.)) for x in seq]
            metrics[phase] = dict(samples=len(seq), abs_error_x_px=stats(np.abs(ex)),
                                  error_x_mae_px=float(np.mean(np.abs(ex))) if ex else None,
                                  error_x_rmse_px=float(np.sqrt(np.mean(np.square(ex)))) if ex else None,
                                  abs_error_y_px=stats([abs(x['ey_px']) for x in seq]),
                                  abs_camera_x_velocity_px_s=stats(series),
                                  camera_travel_x_px=float(np.abs(np.diff(sx)).sum()),
                                  speed_histogram=dict(Counter(r['cmd'].get('SPD', 0) for r in serial)),
                                  moving_commands=sum(r['cmd']['T'] == 141 for r in serial),
                                  stop_commands=sum(r['cmd']['T'] == 135 for r in serial))
            if phase in ('stop_dwell', 'reverse_dwell', 'settle', 'jitter'):
                inside = [abs(v) <= DEADBAND_X_PX for v in ex]
                first = next((i for i in range(len(inside) - 2) if all(inside[i:])), None)
                metrics[phase]['settle_to_19_2_px_s'] = (None if first is None else
                                                         seq[first]['t'] - phase_starts[phase])
                if phase == 'stop_dwell': metrics[phase]['overshoot_px'] = max([0., *(-v for v in ex)])
                if phase == 'reverse_dwell': metrics[phase]['overshoot_px'] = max([0., *ex])
        block['phase_metrics'] = metrics
        block['publication_age_ms'] = stats([(x['sent_ns'] - x['capture_ns']) / 1e6 for x in samples])
        block['camera_gap_ms'] = stats(np.diff([x['capture_ns'] for x in samples]) / 1e6)
        paired = []
        relevant_outputs = [row for row in outputs if row['frame_id'] in fid_index]
        for mark in relevant_outputs:
            if not mark['catchup_applied'] or not mark['sent']: continue
            index = fid_index.get(mark['frame_id'])
            item = samples[index]
            matches = [r for r in mapped[index] if r['cmd'].get('T') == 141 and
                       (r['cmd']['X'], r['cmd']['Y'], r['cmd']['SPD']) == (mark['x'], mark['y'], mark['final_speed'])]
            physical_speeds = [-mark['x'] * x['scene_x_velocity_px_s'] for x in samples[index + 1:]
                               if item['sent_ns'] <= x['capture_ns'] <= item['sent_ns'] + 250_000_000]
            # A capture just after the end of a ramp still compares against
            # the preceding capture. Preserve its actual dwell phase label.
            rapid_interval = index > 0 and any(
                phase.rapid and max(phase.start, samples[index - 1]['t']) < min(phase.end, item['t'])
                for phase in profile.phases)
            paired.append(dict(mark, phase=item['phase'], serial_verified=bool(matches),
                               rapid_observation_interval=rapid_interval,
                               serial_ns=[r['ns'] for r in matches],
                               following_direction_image_velocity_px_s=stats(physical_speeds)))
        trace_rows = [row for row in traces if row['frame_id'] in fid_index]
        block['control_trace_rows'] = len(trace_rows)
        block['control_output_rows'] = len(relevant_outputs)
        block['catchup_eligible_frame_ids'] = sorted({r['frame_id'] for r in relevant_outputs if r['catchup_eligible']})
        block['unexpected_lead_rows'] = sum(row['lead_valid'] or any(row['forecast_used'].values())
                                            for row in trace_rows)
        block['catchup_events'] = paired
        block['confirmed_rapid_boost'] = any(m['serial_verified'] and m['track_id'] == H.TID and
                                             m['final_speed'] >= 7 and m['rapid_observation_interval']
                                             for m in paired)
        motion = [r for group in mapped for r in group if r['cmd']['T'] == 141]
        block['first_motion'] = motion[0] if motion else None
        block['first_spd_at_least_7'] = next((r for r in motion if r['cmd']['SPD'] >= 7), None)
        block['serial_motion_directions'] = sorted({r['cmd']['X'] for r in motion})
        horizontal_signs = [r['cmd']['X'] for r in motion if r['cmd']['X']]
        block['serial_x_reversals'] = sum(a != b for a, b in zip(horizontal_signs, horizontal_signs[1:]))
        block.update(deadband_timing(samples, block['first_motion'], block['first_spd_at_least_7']))


def deadband_timing(samples, first_motion, first_at_least_7):
    seq = [row for row in samples if row['phase'] != 'select']
    outside_x = [abs(row['ex_px']) > DEADBAND_X_PX for row in seq]
    outside_y = [abs(row['ey_px']) > DEADBAND_Y_PX for row in seq]
    first_outside = next((row for row, x, y in zip(seq, outside_x, outside_y) if x or y), None)
    # Sum measured inter-frame intervals classified by their first sample.
    # Do not estimate time beyond the last published frame of a partial block.
    intervals = [(nxt['capture_ns'] - row['capture_ns']) / 1e9 for row, nxt in zip(seq, seq[1:])]

    def elapsed(command):
        if command is None or first_outside is None or command['ns'] < first_outside['sent_ns']:
            return None
        return (command['ns'] - first_outside['sent_ns']) / 1e6

    return dict(deadband_measured_duration_s=sum(intervals),
                time_outside_x_deadband_s=sum(dt for dt, x in zip(intervals, outside_x) if x),
                time_outside_either_deadband_s=sum(dt for dt, x, y in zip(intervals, outside_x, outside_y) if x or y),
                first_deadband_exit_frame_id=first_outside['fid'] if first_outside is not None else None,
                deadband_exit_to_first_motion_ms=elapsed(first_motion),
                deadband_exit_to_first_spd_at_least_7_ms=elapsed(first_at_least_7),
                first_motion_to_spd_at_least_7_ms=(None if first_motion is None or first_at_least_7 is None else
                                                   (first_at_least_7['ns'] - first_motion['ns']) / 1e6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry', action='store_true', help='actual bridge with synthetic camera/PTY (default)')
    mode.add_argument('--execute', action='store_true', help='actual camera and gimbal')
    parser.add_argument('--attended', action='store_true')
    parser.add_argument('--candidate-first', action='store_true')
    parser.add_argument('--profile', choices=PROFILE_NAMES, default=PROFILE_NAMES[0])
    parser.add_argument('--rapid-velocity', type=int, choices=(240, 400),
                        help='virtual scene velocity in px/s (default 240 for rapid-reversal100, fixed 400 for rapid-pulse60)')
    parser.add_argument('--dry-plant', choices=tuple(DRY_PLANTS), default='reference',
                        help='synthetic camera motion model (dry mode only)')
    a = parser.parse_args()
    try:
        a.profile = profile_for(a.profile, a.rapid_velocity)
    except ValueError as exc:
        parser.error(str(exc))
    a.dry = not a.execute
    a.repo, a.bridge, a.out = (p.resolve() for p in (a.repo, a.bridge, a.out))
    if not sys.platform.startswith('linux') or H.pty is None:
        parser.error('the actual bridge comparison requires Linux and a POSIX PTY')
    if a.execute and not a.attended: parser.error('real run requires --execute --attended')
    if a.out.exists() or a.out.is_relative_to(a.repo): parser.error('use a new private directory outside the repository')
    if not a.bridge.is_file() or not os.access(a.bridge, os.X_OK): parser.error('bridge executable missing')
    gap.load_harness(a.repo, a.bridge)
    a.max_seconds, (a.limx, a.limy) = MAX_SECONDS, SCENE_LIMIT.tolist()
    cv2.setNumThreads(1)
    os.umask(0o077)
    os.environ.update(LWIR_CAM_STAMP='0', LWIR_CAM_SHM='0', LWIR_CONTROL_TRACE='1', CAM_NO_RECOVER='1')
    os.environ.pop('CAM_PIPE_TEST_STALL_AT', None)
    a.out.mkdir(mode=0o700, parents=True)
    result = dict(completed=False, dry=a.dry, test='virtual camera framing only',
                  settings=dict(catchup_speeds=[7, 0] if a.candidate_first else [0, 7],
                                bridge_sha256=hashlib.sha256(a.bridge.read_bytes()).hexdigest(),
                                speed_cap=CAP, pd=list(gap.PD_OPTIONS), lead=False, latency_ms=40,
                                observation_grace_ms=100, deadline_s=MAX_SECONDS,
                                scene_limit_px=SCENE_LIMIT.tolist(), outward_brake_px=BRAKE_AT.tolist(),
                                origin_limit_px=ORIGIN_LIMIT, ramp_px=a.profile.amplitude,
                                ramp_velocity_px_s=a.profile.velocity, profile=asdict(a.profile),
                                raw16_timestamp='host monotonic complete frame arrival'))
    if a.dry: result['settings']['dry_plant'] = dict(name=a.dry_plant, **DRY_PLANTS[a.dry_plant])

    def stop_signal(signum, _):
        for tap in H.TAPS:
            if not tap.done: tap.estop()
        raise KeyboardInterrupt('signal %d' % signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, stop_signal)
    session = None
    try:
        if a.execute:
            gap.real_preflight(parser, H)
            no_other_camera()
            require_exclusive_serial()
        session = Session(a, time.monotonic(), a.out)
        session.compare()
        result['completed'] = True
    except BaseException as exc:
        if session is not None and session.tap is not None: session.tap.estop()
        result['error'] = repr(exc)
        print('ABORT', repr(exc), flush=True)
    finally:
        if session is not None:
            if session.tap is not None:
                session.tap.estop()
                if (session.cam is not None and not session.end_view_checked and
                        not getattr(session, 'post_stop_attempted', False)):
                    try:
                        session.close_bridge()
                        session.stationary_after_stop('abort')
                    except BaseException as exc:
                        result['post_stop_error'] = repr(exc)
            try:
                result['camera_exit'] = session.close()
            except BaseException as exc:
                result['close_error'] = repr(exc)
                result['completed'] = False
            result['end_view_checked'] = session.end_view_checked
            if session.tap is not None:
                result['max_speed'] = session.tap.maxspd
                result['stop_ever_failed'] = session.tap.stop_ever_failed
                result['watchdog_trip'] = session.tap.trip
                if session.tap.trip or session.tap.stop_ever_failed: result['completed'] = False
            raw = a.out / 'raw16'
            raw.mkdir()
            saved = 0
            for fid, ns, frame in session.raw_frames:
                try:
                    if not cv2.imwrite(str(raw / ('%06d_%d.png' % (fid, ns))), frame):
                        raise OSError('cv2.imwrite returned false')
                    saved += 1
                except (OSError, cv2.error) as exc:
                    result.setdefault('raw_save_errors', []).append(dict(fid=fid, error=repr(exc)))
            result.update(raw_frames_captured=len(session.raw_frames), raw_frames_saved=saved, blocks=session.blocks)
            with (a.out / 'protocol.jsonl').open('w', encoding='utf-8') as log:
                for row in session.audit: log.write(json.dumps(row) + '\n')
            try:
                read_evidence(a.out, session.blocks, a.profile)
            except (OSError, ValueError, KeyError) as exc:
                result['evidence_error'] = repr(exc)
                result['completed'] = False
            result['completed'] &= bool(saved == len(session.raw_frames) and session.end_view_checked and
                                        len(session.blocks) == 2 and all(
                b.get('post_stop_verified') and b.get('control_trace_rows') and b.get('control_output_rows')
                and not b.get('unexpected_lead_rows')
                and (b.get('serial_motion_directions') == [-1, 1] if a.profile.name == 'rapid-reversal100'
                     else 1 in b.get('serial_motion_directions', []))
                and b.get('first_motion', {}).get('cmd', {}).get('SPD') == 1
                and all(b['phase_metrics'][phase.name]['samples'] for phase in a.profile.phases)
                for b in session.blocks))
            candidate = next((b for b in session.blocks if b['catchup_speed'] == 7), {})
            result['confirmed_candidate_rapid_serial_boost'] = bool(candidate.get('confirmed_rapid_boost'))
            result['interpretation'] = ('matched samples; compare physical metrics before any conclusion' if
                result['completed'] and result['confirmed_candidate_rapid_serial_boost'] and not a.dry
                else 'no physical efficacy claim from this run')
        (a.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'blocks'}, indent=2), flush=True)
    return 0 if result['completed'] else 3


if __name__ == '__main__':
    sys.exit(main())
