#!/usr/bin/env python3
"""Compare grace 0 and 100 ms using a virtual track and a checked camera view.

Dry mode uses a synthetic camera and an actual v27 bridge on a PTY. Attended
mode drives the serial gimbal with the same virtual track and records real
RAW16. Camera frame IDs/times describe frames received by the camera reader.
The test measures commands and image movement, not detection or human tracking.
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

RUN_SECONDS = 75
SPEED_CAP = 10
GAP_NS = 100_000_000
# S telemetry is polled at 50 ms in the bridge. This allowance is for receiving
# its confirmation only, not for extending the controller's physical grace.
STATUS_ALLOWANCE_NS = 75_000_000
PD_OPTIONS = ('--kp', '30.0', '--controller', 'pid', '--ki', '0', '--kd', '6', '--deadband', '0.06')


def load_harness(repo, bridge_path):
    server = repo / 'tracking_server'
    sys.path.insert(0, str(server / 'src' / 'gimbal' / 'hil_test'))
    import predictive_ab_hil as reference

    reference.H.ROOT = str(server)
    reference.H.CAMP = str(server / 'src' / 'camera' / 'cam_pipe')
    reference.H.BRIDGES['v27'] = str(bridge_path)
    return reference


def tracking_port_in_use():
    try:
        for table in ('tcp', 'tcp6'):
            path = Path('/proc/net') / table
            if table == 'tcp6' and not path.exists():
                continue
            for row in path.read_text(encoding='ascii').splitlines()[1:]:
                fields = row.split()
                if fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) == 8090:
                    return True
    except (OSError, IndexError, ValueError):
        raise SystemExit('could not verify that port 8090 is idle')
    return False


def real_preflight(parser, harness):
    if harness.other_servers() or tracking_port_in_use():
        parser.error('a tracking server is running or port 8090 is occupied')
    devices = sorted(glob.glob(harness.DEV_GLOB))
    if len(devices) != 1 or not os.path.exists(devices[0]):
        parser.error('expected exactly one connected stable gimbal serial identity')
    if shutil.which('fuser') is None:
        parser.error('fuser is required to verify serial ownership')
    try:
        checked = subprocess.run(['fuser', devices[0]], capture_output=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        parser.error('could not verify serial ownership')
    if checked.returncode == 0:
        parser.error('gimbal serial device is occupied')
    if checked.returncode != 1 or checked.stdout.strip() or checked.stderr.strip():
        parser.error('serial ownership check failed')
    if not os.access(harness.CAMP, os.X_OK):
        parser.error('camera program is missing or cannot be executed')
    harness.DEV = devices[0]


def make_session(reference):
    H = reference.H

    class Bridge(H.Bridge):
        def __init__(self, tap, options, log, audit):
            self.tap = tap
            self.audit = audit
            self.armed = False
            self.observation_capable = threading.Event()
            self.gap_lock = threading.Lock()
            self.gap_deadline_ns = 0
            self.gap_recovery_sent = False
            self.gap_open = False
            self.gap_timer = None
            self.gap_publication_timer = None
            super().__init__(tap, options, log, 'v27')

        def trip(self, reason):
            if self.tap.trip is None:
                self.tap.trip = reason
                self.tap.estop()

        def run(self):
            try:
                for raw in iter(self.p.stdout.readline, b''):
                    line = raw.decode().strip()
                    now = time.monotonic_ns()
                    self.audit.append(dict(ns=now, direction='receive', line=line))
                    if line == 'READY':
                        self.ready.set()
                    elif line == 'OBSERVATION_STATUS 1':
                        self.observation_capable.set()
                    elif line.startswith('S '):
                        state = tuple(map(int, line.split()[1:6]))
                        automatic, lost, _, error, track = state
                        with self.gap_lock:
                            previously_lost = bool(self.state[1])
                            self.state = state
                            loss_allowed = self.gap_open and (
                                now < self.gap_deadline_ns or
                                (previously_lost and now < self.gap_deadline_ns + STATUS_ALLOWANCE_NS))
                            if self.gap_open and self.gap_recovery_sent and not lost:
                                self.gap_open = False
                            violation = self.armed and (
                                not automatic or error or track != H.TID or (lost and not loss_allowed))
                        if violation:
                            self.trip('unexpected lost/disabled/error/wrong-ID bridge status')
                    elif line.startswith('R '):
                        if line == 'R A OK':
                            self.armed = True
                        self.ans.append(line)
            except (OSError, UnicodeError, ValueError) as exc:
                self.trip('unreadable bridge status: %s' % exc)

        def send(self, text, *, gap=False, recovery=False):
            data = (text + '\n').encode()
            now = time.monotonic_ns()
            if gap:
                with self.gap_lock:
                    self.gap_deadline_ns = now + GAP_NS
                    self.gap_open = True
                self.gap_timer = threading.Timer((GAP_NS + STATUS_ALLOWANCE_NS) / 1e9, self.expire_gap)
                self.gap_timer.daemon = True
                self.gap_timer.start()
                self.gap_publication_timer = threading.Timer(GAP_NS / 1e9, self.expire_publication)
                self.gap_publication_timer.daemon = True
                self.gap_publication_timer.start()
            if recovery:
                with self.gap_lock:
                    if now >= self.gap_deadline_ns:
                        raise H.Abort('same-ID recovery publication missed the 100 ms window')
                    self.gap_recovery_sent = True
            self.audit.append(dict(ns=now, direction='send', line=text))
            if self.p.stdin.write(data) != len(data):
                raise H.Abort('bridge input write was incomplete')
            return now

        def expire_publication(self):
            with self.gap_lock:
                overdue = self.armed and not self.gap_recovery_sent
            if overdue:
                self.trip('same-ID recovery was not published within 100 ms')

        def expire_gap(self):
            with self.gap_lock:
                overdue = self.armed and (not self.gap_recovery_sent or self.state[1])
                self.gap_open = False
            if overdue:
                self.trip('no state confirmation within the 175 ms telemetry window')

        def close(self):
            with self.gap_lock:
                self.armed = False
            if self.gap_timer is not None:
                self.gap_timer.cancel()
            if self.gap_publication_timer is not None:
                self.gap_publication_timer.cancel()
            super().close()

    class Session(reference.Session):
        def __init__(self, args, start, output):
            self.a = args
            self.output = output
            self.t_begin = start
            self.deadline = start + RUN_SECONDS
            self.audit = []
            self.raw_frames = []
            self.blocks = []
            self.end_view_checked = False
            self.check = reference.MotionCheck()
            self.est = H.Shift()
            self.last = 0
            self.br = None
            self.lastp = 0.0
            self.S = np.zeros(2)
            self.step_i = 0
            self.name = ''
            self.same = self.nsame0 = 0
            self.prev_m = self.hook = self.meas_t = None
            self.bw, self.bh = H.BW, H.BH
            self.proc = 0.0
            self.slog = open(output / 'serial.jsonl', 'w', encoding='utf-8')
            self.elog = open(output / 'bridge_err.log', 'w', encoding='utf-8')
            self.f = open(output / 'frames.csv', 'w', newline='', encoding='utf-8')
            self.w = csv.writer(self.f)
            self.w.writerow(H.FIELDS)
            self.cam_log = None
            self.cam = None
            self.tap = None

        def start_camera(self):
            self.tap = reference.StrictTap(self.a.dry, self.slog)
            self.tap.guard = self.tap.guard_y = SPEED_CAP
            self.tap.deadline = self.deadline
            if self.a.dry:
                self.cam = H.FakeCam(self.tap, 6., .08, tau=.05)
            else:
                self.cam_log = open(self.output / 'cam_pipe.log', 'w', encoding='utf-8')
                self.cam = H.Cam(self.cam_log)
            original_get = self.cam.get

            def capture(last, timeout):
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise H.Abort('75 s session deadline reached')
                frame = original_get(last, min(timeout, remaining))
                if frame is not None and (not self.raw_frames or frame[0] != self.raw_frames[-1][0]):
                    self.raw_frames.append((frame[0], frame[1], frame[2].copy()))
                return frame

            self.cam.get = capture
            first = self.cam.get(0, 45.0)
            if first is None:
                raise H.Abort('no camera frame within the startup budget')
            self.last = first[0]

        def frame(self):
            result = super().frame()
            if self.tap.trip or self.tap.stop_ever_failed:
                raise H.Abort(self.tap.trip or 'STOP write failed')
            return result

        def open_bridge(self, grace):
            options = [*H.BASE, *PD_OPTIONS, '--observation-grace-ms', str(grace)]
            self.elog.write('--- %s: %s\n' % (self.name, ' '.join(options)))
            self.elog.flush()
            self.br = Bridge(self.tap, options, self.elog, self.audit)
            self.tap.child = self.br.p
            if not self.br.observation_capable.wait(.5):
                raise H.Abort('v27 observation capability was not advertised')
            self.br.send('P')
            self.lastp = time.monotonic()

        def publish(self, frame, base, offset, phase, start):
            fid, ns, shift, response = frame
            x, y = (np.array([320., 240.]) + base + np.array([offset, 0.]) + shift).tolist()
            ex, ey = x - 320, y - 240
            box = self.box(x, y)
            if box is None or abs(ex) > 110 or abs(ey) > 55:
                raise H.Abort('virtual track exceeded its image error limit')
            visible = phase != 'gap'
            if visible:
                x1, y1, x2, y2 = box
                line = ('F %d %d 1 %d %.4f %.4f %.4f %.4f .900 OBS 1'
                        % (fid, ns, H.TID, x1, y1, x2 - x1, y2 - y1))
            else:
                line = 'F %d %d 0 OBS 1' % (fid, ns)
            sent = self.br.send(line, gap=phase == 'gap', recovery=phase == 'recover')
            self.row(phase, time.monotonic() - start, fid, ns, shift, response,
                     base + [offset, 0], int(visible), (x, y), (ex / 320, ey / 240))
            return sent, ex

        def block(self, grace, initial_scene):
            self.name = 'grace_%d' % grace
            self.end_view_checked = False
            result = dict(grace_ms=grace, observed_id=H.TID, camera_ids=[], image_x_error_px={})
            self.blocks.append(result)
            if time.monotonic() + 7 >= self.deadline:
                raise H.Abort('insufficient session time for another block and verified STOP')
            if not self.a.dry and (H.other_servers() or tracking_port_in_use()):
                raise H.Abort('a tracking server started during the session')
            frame = self.frame()
            if np.max(np.abs(frame[2] - initial_scene)) > 22:
                raise H.Abort('starting view differs by more than 22 px')
            base = -frame[2]
            self.open_bridge(grace)
            if not self.tap.open_line():
                raise H.Abort('serial line cannot reopen after a STOP or watchdog fault')
            start = time.monotonic()
            previous_fid = previous_ns = None
            previous_raw = None

            def next_frame():
                nonlocal previous_fid, previous_ns, previous_raw
                if time.monotonic() - start > 2.5:
                    raise H.Abort('block did not obtain all witnesses within 2.5 s')
                sample = self.frame()
                fid, ns = sample[:2]
                raw = self.raw_frames[-1][2][::4, ::4]
                if previous_fid is not None and (
                        fid != previous_fid + 1 or ns <= previous_ns or np.array_equal(raw, previous_raw)):
                    raise H.Abort('nonconsecutive, nonmonotonic, or frozen camera input during publication')
                previous_fid, previous_ns, previous_raw = fid, ns, raw
                result['camera_ids'].append(fid)
                return sample

            for _ in range(3):
                self.publish(next_frame(), base, 0, 'select', start)
            self.br.send('A %d' % H.TID)
            for _ in range(6):
                if self.br.ans:
                    answer = self.br.ans.popleft()
                    if answer != 'R A OK':
                        raise H.Abort('bridge refused the one selection: %s' % answer)
                    break
                self.publish(next_frame(), base, 0, 'select', start)
            else:
                raise H.Abort('bridge did not acknowledge selection')

            # Use one fixed offset. At least two adjacent observations must have
            # already produced forward output, with measurable camera movement.
            right, left = 64., -42.
            prior_forward = 0
            prior_x = []
            while True:
                frame = next_frame()
                command = self.tap.last
                if (prior_forward >= 2 and command[0] == 1 and command[1] == 0 and command[2]
                        and frame[2][0] <= -base[0] - 1.5
                        and prior_x[-2] - frame[2][0] >= .75
                        and 320 + base[0] + right + frame[2][0] > 344):
                    result['pre_gap_command'] = list(command)
                    result['pre_gap_scene_shift_px'] = (frame[2] + base).tolist()
                    result['pre_gap_recent_horizontal_motion_px'] = prior_x[-2] - frame[2][0]
                    result['gap_ns'], result['image_x_error_px']['gap_virtual'] = self.publish(
                        frame, base, right, 'gap', start)
                    break
                self.publish(frame, base, right, 'forward', start)
                prior_x.append(frame[2][0])
                prior_forward = prior_forward + 1 if command[0] == 1 and command[1] == 0 and command[2] else 0

            result['recovery_ns'], result['image_x_error_px']['recovery'] = self.publish(
                next_frame(), base, right, 'recover', start)
            frame = next_frame()
            _, right_error = self.publish(frame, base, right, 'after_recovery', start)
            frame = next_frame()
            left_error = base[0] + left + frame[2][0]
            if right_error <= 19.2 or left_error >= -19.2 or self.tap.last[0] != 1 or not self.tap.last[2]:
                raise H.Abort('did not witness same-direction recovery before a measured reversal')
            result['reversal_ns'], result['image_x_error_px']['reversal'] = self.publish(
                frame, base, left, 'reverse', start)
            result['image_x_error_px']['before_reversal'] = right_error
            until = time.monotonic() + .40
            while self.tap.last[0] != -1 or not self.tap.last[2]:
                if time.monotonic() >= until:
                    raise H.Abort('no opposite command within 400 ms of reversal')
                self.publish(next_frame(), base, left, 'reverse', start)
            result['opposite_seen_ns'] = time.monotonic_ns()
            self.close_bridge()
            self.stationary_after_stop(self.name)
            self.f.flush()
            self.slog.flush()
            result['post_stop_verified'] = self.end_view_checked

        def compare(self):
            self.start_camera()
            # Allow the raw stream to settle with the serial line cut, then use
            # both registration methods before any bridge can issue movement.
            warm_until = time.monotonic() + 5
            while time.monotonic() < warm_until:
                frame = self.cam.get(self.last, .3)
                if frame is None:
                    raise H.Abort('camera stalled during warm-up')
                self.last = frame[0]
            start = time.monotonic()
            while time.monotonic() - start < 1:
                self.frame()
            initial_scene = self.S.copy()
            order = ((self.a.single_condition,) if self.a.single_condition is not None else
                     ((100, 0) if self.a.candidate_first else (0, 100)))
            for index, grace in enumerate(order, 1):
                self.step_i = index
                self.block(grace, initial_scene)

        def close(self):
            if self.tap is None:
                for handle in (self.f, self.slog, self.elog):
                    handle.close()
                return None
            self.tap.estop()
            try:
                self.close_bridge()
                if self.cam is None:
                    return None
                if self.a.dry:
                    return self.cam.close()
                self.cam.p.terminate()
                try:
                    return self.cam.p.wait(timeout=max(.05, min(5., self.deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    raise H.Abort('camera child did not finish after SIGTERM (pid %s)' % self.cam.p.pid)
            finally:
                self.tap.estop()
                self.tap.close()
                for handle in (self.f, self.slog, self.elog, self.cam_log):
                    if handle is not None:
                        handle.close()

    return Session


def serial_comparison(path, blocks):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    commands = [row for row in rows if 'cmd' in row and not row.get('blocked') and not row.get('werr')]
    for block in blocks:
        if not all(key in block for key in ('gap_ns', 'recovery_ns', 'reversal_ns', 'opposite_seen_ns')):
            continue
        gap = [row for row in commands if block['gap_ns'] <= row['ns'] < block['reversal_ns']]
        stop = [row for row in gap if row['cmd'].get('T') == 135]
        forward = [row for row in gap if row['ns'] >= block['recovery_ns']
                   and row['cmd'].get('T') == 141 and row['cmd'].get('X') == 1]
        wrong = [row for row in gap if row['cmd'].get('T') == 141 and (
            row['cmd'].get('X') != 1 or row['cmd'].get('Y') != 0)]
        opposite = [row for row in commands if block['reversal_ns'] <= row['ns'] <= block['opposite_seen_ns']
                    and row['cmd'].get('T') == 141 and row['cmd'].get('X') == -1
                    and row['cmd'].get('Y') == 0 and row['cmd'].get('SPD', 0) > 0]
        block['gap_serial'] = gap
        block['gap_stop_ms'] = [(row['ns'] - block['gap_ns']) / 1e6 for row in stop]
        block['same_direction_restart_ms'] = [(row['ns'] - block['recovery_ns']) / 1e6 for row in forward]
        block['reversal_command_ms'] = [(row['ns'] - block['reversal_ns']) / 1e6 for row in opposite]
        if block['grace_ms'] == 0:
            expected = bool(forward and any(row['ns'] < block['recovery_ns'] for row in stop))
        else:
            expected = not stop and not any(row['ns'] < block['recovery_ns'] for row in gap)
        block['expected_witness'] = bool(expected and opposite and not wrong and block.get('post_stop_verified'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='repository root containing tracking_server')
    parser.add_argument('--bridge', type=Path, required=True, help='v27 executable with OBS support')
    parser.add_argument('--out', type=Path, required=True, help='new private output directory outside the repository')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry', action='store_true', help='use the synthetic camera and PTY (default)')
    mode.add_argument('--execute', action='store_true', help='use real hardware with the operator present')
    parser.add_argument('--attended', action='store_true')
    parser.add_argument('--candidate-first', action='store_true')
    parser.add_argument('--single-condition', type=int, choices=(0,100),
                        help='Single serial-behavior diagnostic; not a matched physical A/B comparison')
    args = parser.parse_args()
    args.dry = not args.execute
    if args.execute and not args.attended:
        parser.error('real execution requires --execute --attended')
    args.repo = args.repo.resolve()
    args.bridge = args.bridge.resolve()
    args.out = args.out.resolve()
    if args.out.exists() or args.out.is_relative_to(args.repo):
        parser.error('output must be a new private directory outside the repository')
    if not os.access(args.bridge, os.X_OK):
        parser.error('v27 executable is missing or cannot be executed')
    reference = load_harness(args.repo, args.bridge)
    H = reference.H
    if H.pty is None:
        parser.error('a POSIX pseudo-terminal is required to test the actual bridge')
    if args.execute:
        real_preflight(parser, H)
    cv2.setNumThreads(1)
    os.umask(0o077)
    os.environ['LWIR_CAM_STAMP'] = '0'
    os.environ['LWIR_CAM_SHM'] = '0'
    os.environ['LWIR_CONTROL_TRACE'] = '1'
    args.out.mkdir(mode=0o700, parents=True)
    args.max_seconds, args.limx, args.limy = RUN_SECONDS, 130., 65.

    def stop_signal(signum, _):
        for tap in H.TAPS:
            if not tap.done:
                tap.estop()
        raise KeyboardInterrupt('signal %d' % signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop_signal)
    session = make_session(reference)(args, time.monotonic(), args.out)
    result = dict(success=False, dry=args.dry, test='virtual-track control only',
                  settings=dict(speed_cap=SPEED_CAP, pd=list(PD_OPTIONS), lead=False,
                                max_seconds=RUN_SECONDS, candidate_first=args.candidate_first,
                                single_condition=args.single_condition,
                                recovery_publication_limit_ms=GAP_NS/1e6,
                                state_confirmation_limit_ms=(GAP_NS+STATUS_ALLOWANCE_NS)/1e6))
    try:
        session.compare()
        result['success'] = True
    except BaseException as exc:
        if session.tap is not None:
            session.tap.estop()
        result['error'] = repr(exc)
        print('ABORT', repr(exc), flush=True)
    finally:
        if session.tap is not None:
            session.tap.estop()
            if session.cam is not None and not session.end_view_checked:
                try:
                    session.close_bridge()
                    session.stationary_after_stop('abort')
                except BaseException as exc:
                    result['post_stop_error'] = repr(exc)
        try:
            result['camera_exit'] = session.close()
        except BaseException as exc:
            result['close_error'] = repr(exc)
            result['success'] = False
        result['end_view_checked'] = session.end_view_checked
        if session.tap is not None:
            result['max_speed'] = session.tap.maxspd
            result['stop_ever_failed'] = session.tap.stop_ever_failed
            result['watchdog_trip'] = session.tap.trip
            if session.tap.trip or session.tap.stop_ever_failed:
                result['success'] = False
        try:
            serial_comparison(args.out / 'serial.jsonl', session.blocks)
        except (OSError, ValueError) as exc:
            result['serial_evidence_error'] = repr(exc)
            result['success'] = False
        result['blocks'] = session.blocks
        expected_blocks = 1 if args.single_condition is not None else 2
        result['success'] &= len(session.blocks) == expected_blocks and all(block.get('expected_witness') for block in session.blocks)
        raw = args.out / 'raw16'
        raw.mkdir()
        for fid, ns, frame in session.raw_frames:
            if not cv2.imwrite(str(raw / ('%06d_%d.png' % (fid, ns))), frame):
                result['success'] = False
                result['save_error'] = 'could not save RAW16 frame %d' % fid
                break
        with (args.out / 'protocol.jsonl').open('w', encoding='utf-8') as log:
            for event in session.audit:
                log.write(json.dumps(event) + '\n')
        result['raw_frames_captured'] = len(session.raw_frames)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['success'] and result['end_view_checked'] else 3


if __name__ == '__main__':
    sys.exit(main())
