#!/usr/bin/env python3
"""Bounded horizontal camera-only response recording; default is synthetic."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

import observation_gap_hil as gap
import pid_hil as H
from rapid_follow_hil import no_other_camera, verify_idle

PULSES = ((1, 1, 250), (-1, 1, 250), (-1, 7, 350), (1, 7, 350))
FOV = np.array([130., 65.])
START_FOV = np.array([65., 35.])
BACKGROUND_BOTTOM = 180
S = 1_000_000_000
FRAME_STALE_NS = 180_000_000
STOP_WIRE = b'\n' + H.STOP_MSG
DWELL_NS = 1_200_000_000
BASELINE_NS = 600_000_000
WARMUP_NS = 5 * S
CONTROL_SECONDS = 30
MAX_FRAMES = 720
PIXEL_BYTES = H.W * H.H * 2
HDR = struct.Struct('<4sHHQQIIIIHHI')
EVIDENCE_VERSION = 2


def unpack_header(raw):
    if len(raw) != HDR.size:
        raise H.Abort('short camera header')
    magic, version, flags, seq, recv, _, slot, payload, ring, width, height, check = HDR.unpack(raw)
    expected = 0x5A17C0DE
    for word in struct.unpack('<11I', raw[:-4]):
        expected ^= word
    if ((magic, version, flags, width, height, payload, ring) !=
            (b'LWF1', 1, 0, H.W, H.H, PIXEL_BYTES, 0) or slot != 0xffffffff
            or check != expected or not seq or not recv):
        raise H.Abort('bad stamped RAW16 header (requires current cam_pipe without SHM)')
    return seq, recv


def read_exactly(fd, size):
    payload = bytearray()
    while len(payload) < size:
        chunk = os.read(fd, size - len(payload))
        if not chunk:
            raise H.Abort('camera ended before the next complete frame')
        payload.extend(chunk)
    return payload


class StampedCamera:
    """Sequential, bounded camera reader, with separate SDK-return and pipe-arrival stamps."""
    def __init__(self, path, log):
        env = dict(os.environ, LWIR_CAM_STAMP='1', LWIR_CAM_SHM='0', CAM_NO_RECOVER='1')
        for key in ('CAM_PIPE_TEST_STALL_AT', 'CAM_TELEMETRY'):
            env.pop(key, None)
        self.p = subprocess.Popen([str(path), '-1', str(CONTROL_SECONDS)], cwd=path.parent,
                                  stdout=subprocess.PIPE, stderr=log, env=env, bufsize=0)
        self.cv = threading.Condition()
        self.frames = []
        self.error = None
        self.th = threading.Thread(target=self.read, daemon=True)
        self.th.start()

    def read(self):
        fd = self.p.stdout.fileno()
        try:
            while len(self.frames) < MAX_FRAMES:
                fid, recv = unpack_header(read_exactly(fd, HDR.size))
                data = read_exactly(fd, PIXEL_BYTES)
                now = time.monotonic_ns()
                raw = np.frombuffer(data, '<u2').reshape(H.H, H.W)
                with self.cv:
                    if recv > now or fid != len(self.frames) + 1 or (
                            self.frames and recv <= self.frames[-1][1]):
                        raise H.Abort('camera frame gap or nonmonotonic camera stamp')
                    self.frames.append((fid, recv, now, raw))
                    self.cv.notify_all()
            raise H.Abort('camera frame budget exhausted')
        except Exception as exc:
            with self.cv:
                self.error = str(exc)
                self.cv.notify_all()

    def get(self, last, timeout):
        with self.cv:
            self.cv.wait_for(lambda: len(self.frames) > last or self.error, timeout)
            if self.error:
                raise H.Abort(self.error)
            return self.frames[last] if len(self.frames) > last else None

    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
        try:
            self.p.wait(timeout=6)
        except subprocess.TimeoutExpired as exc:
            raise H.Abort('own camera process did not stop after SIGTERM') from exc
        self.th.join(1)
        self.p.stdout.close()
        return self.p.returncode


class SyntheticCamera(H.FakeCam):
    """The existing synthetic plant with sequential frames and no burst catch-up."""
    def __init__(self, tap):
        self.frames = []
        self.error = None
        super().__init__(tap, K=6., L=.08, fps=20, vmin=45., tau=.06)

    def run(self):
        while not self.done:
            time.sleep(1 / self.fps)
            x, y = self.pos
            matrix = np.float32([[1, 0, x - 430], [0, 1, y - 310]])
            frame = cv2.warpAffine(self.big, matrix, (H.W, H.H)).astype(np.uint16)
            self.fid += 1
            frame[0, 20] += self.fid % 101  # non-frozen background, without scene motion at rest
            with self.cv:
                self.frames.append((self.fid, time.monotonic_ns(), frame))
                self.cv.notify_all()

    get = StampedCamera.get

    def close(self):
        self.done = True
        self.th.join(1)
        self.ph.join(1)
        return 0


def exclusive_open(device):
    import fcntl
    before = os.stat(device)
    fd = H.open_device(device)
    try:
        if os.fstat(fd).st_rdev != before.st_rdev:
            raise H.Abort('serial identity changed while opening')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.ioctl(fd, H.termios.TIOCEXCL)
    except BaseException:
        os.close(fd)
        raise
    return fd


class PulseTap:
    """Fixed-command tap, with timed STOP in a thread independent of image processing."""
    def __init__(self, device=None, write=None, seconds=CONTROL_SECONDS):
        self.device = device
        self.fd = None
        self.write = write
        self.rows = []
        self.lh = collections.deque()
        self.lk = threading.RLock()
        self.wake = threading.Event()
        self.done = False
        self.fault = None
        self.write_failed = False
        self.pulse = None
        self.last_sample_ns = 0
        self.last_stop_ns = 0
        self.deadline = time.monotonic_ns() + int(seconds * S)
        self.th = threading.Thread(target=self.watch, daemon=True)

    def connect(self):
        if self.device is not None:
            self.fd = exclusive_open(self.device)
            seen = subprocess.run(['fuser', self.device], capture_output=True, timeout=3)
            if seen.returncode != 0 or seen.stdout.split() != [str(os.getpid()).encode()]:
                raise H.Abort('serial ownership changed while opening')
        self.th.start()
        if not self.stop('initial') or self.write_failed:
            raise H.Abort('initial STOP failed')

    def tx(self, payload, reason, command, **extra):
        row = dict(write_start_ns=time.monotonic_ns(), reason=reason, command=command, **extra)
        try:
            if self.write is not None:
                size = self.write(payload)
            elif self.fd is not None:
                size = os.write(self.fd, payload)
            elif self.device is None:
                size = len(payload)
            else:
                raise OSError('serial is closed')
            if size != len(payload):
                raise OSError('short serial write')
            row['ok'] = True
            self.lh.append((row['write_start_ns'] / S, command))
        except OSError as exc:
            self.write_failed = True
            row.update(ok=False, error=str(exc))
        row['write_end_ns'] = time.monotonic_ns()
        self.rows.append(row)
        return row

    def start(self, direction, speed, duration_ms, trial):
        with self.lk:
            now = time.monotonic_ns()
            if ((direction, speed, duration_ms) not in PULSES or type(direction) is not int
                    or type(speed) is not int or type(duration_ms) is not int):
                self.fail('unexpected fixed pulse command')
                raise H.Abort(self.fault)
            if (self.fault or self.write_failed or self.done or not self.th.is_alive() or
                    self.pulse or not 0 <= now - self.last_sample_ns <= FRAME_STALE_NS or
                    now + duration_ms * 1_000_000 + DWELL_NS >= self.deadline or
                    now - self.last_stop_ns < BASELINE_NS):
                self.fail('pulse cannot be armed')
                raise H.Abort(self.fault)
            cmd = (direction, 0, speed)
            payload = json.dumps(dict(T=141, X=direction, Y=0, SPD=speed),
                                 separators=(',', ':')).encode() + b'\n'
            # Arm before attempting the nonblocking write. An interrupted write is stopped in finally.
            self.pulse = (trial, now + duration_ms * 1_000_000)
            row = self.tx(payload, 'fixed_pulse', cmd, trial=trial, nominal_ms=duration_ms)
            self.wake.set()
            if not row['ok']:
                self.fail('fixed pulse serial write failed')
                raise H.Abort(self.fault)
            return row

    def stop(self, reason, repeat=3):
        with self.lk:
            self.pulse = None
            written = False
            for i in range(repeat):
                row = self.tx(STOP_WIRE, reason, (0, 0, 0), attempt=i + 1)
                if row['ok']:
                    self.last_stop_ns = row['write_end_ns']
                    written = True
                if i + 1 < repeat:
                    time.sleep(.005)
            if not written and self.device is not None:
                try:
                    if self.fd is not None:
                        os.close(self.fd)
                    self.fd = None
                    self.fd = exclusive_open(self.device)
                    row = self.tx(STOP_WIRE, reason, (0, 0, 0), reopened=True)
                    if row['ok']:
                        self.last_stop_ns = row['write_end_ns']
                        written = True
                except (H.Abort, OSError) as exc:
                    self.rows.append(dict(ns=time.monotonic_ns(), reason='serial_reopen_failed',
                                          error=str(exc)))
            return written

    def fail(self, reason):
        with self.lk:
            if not self.fault:
                self.fault = reason
                self.stop('abort_stop')

    def watch(self):
        try:
            while not self.done:
                self.wake.wait(.005)
                self.wake.clear()
                with self.lk:
                    if self.done:
                        return
                    now = time.monotonic_ns()
                    if not self.fault:
                        if now >= self.deadline:
                            self.fail('control watchdog deadline')
                        elif self.pulse is not None:
                            trial, end = self.pulse
                            if now - self.last_sample_ns > FRAME_STALE_NS:
                                self.fail('no recent accepted registration while moving')
                            elif now >= end:
                                self.pulse = None
                                row = self.tx(STOP_WIRE, 'pulse_stop', (0, 0, 0),
                                              trial=trial, due_ns=end)
                                if row['ok']:
                                    self.last_stop_ns = row['write_end_ns']
                                if not row['ok'] or row['write_start_ns'] - end > 30_000_000:
                                    self.fail('nominal STOP late or failed')
                        if self.fd is not None and not self.fault:
                            try:
                                data = os.read(self.fd, 4096)
                                if not data:
                                    raise OSError('serial hung up')
                                self.rows.append(dict(rx_ns=time.monotonic_ns(), rx_hex=data.hex()))
                                if len(self.rows) > 4000:
                                    self.fail('serial evidence budget exceeded')
                            except BlockingIOError:
                                pass
                            except OSError:
                                self.fail('serial read failed')
        except Exception as exc:
            self.fail('serial watchdog failed: ' + str(exc))

    def close(self):
        with self.lk:
            self.done = True
            self.wake.set()
        if self.th.ident is not None:
            self.th.join(1)
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class Registration:
    """Fixed initial background template and independent consecutive-frame LK."""
    def __init__(self, background_bottom=BACKGROUND_BOTTOM):
        self.bottom = background_bottom
        self.key = None
        self.prev = None
        self.previous_gray = None
        self.flow = np.zeros(2)
        self.previous_template = np.zeros(2)

    def update(self, raw):
        roi = raw[:self.bottom, 16:-16]
        if self.prev is not None and np.array_equal(self.prev, roi):
            raise H.Abort('frozen RAW16 background (identical successive images)')
        self.prev = roi
        small = cv2.resize(roi, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA).astype(np.float32)
        lo, hi = np.percentile(small, (1, 99))
        if hi - lo < 4:
            raise H.Abort('insufficient image contrast in background ROI')
        gray = np.clip((small - lo) * (255 / (hi - lo)), 0, 255).astype(np.uint8)
        band = cv2.GaussianBlur(small, (0, 0), 1) - cv2.GaussianBlur(small, (0, 0), 12)
        mx, my = 75, 35  # half-scale search extends past the full-resolution FOV limits
        if self.key is None:
            self.key = band[my:-my, mx:-mx].copy()
            if self.key.std() < 2:
                raise H.Abort('insufficient texture in background template')
            self.previous_gray = gray
            return [0., 0.], [0., 0.], 1.

        response = cv2.matchTemplate(band, self.key, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(response)
        if not (score >= .55 and 0 < x < 2 * mx and 0 < y < 2 * my):
            raise H.Abort('weak or clipped background template match')

        def subpixel(a, b, c):
            q = a - 2 * b + c
            return float(.5 * (a - c) / q) if q < -1e-9 else 0.

        template = 2 * np.array([
            x - mx + subpixel(*response[y, x - 1:x + 2]),
            y - my + subpixel(*response[y - 1:y + 2, x]),
        ])
        response[max(0, y - 5):y + 6, max(0, x - 5):x + 6] = -1.
        if float(response.max()) >= score - .035 or np.max(np.abs(template - self.previous_template)) > 32:
            raise H.Abort('ambiguous or discontinuous background template match')

        pts = cv2.goodFeaturesToTrack(self.previous_gray, 160, .02, 8)
        if pts is None or len(pts) < 16:
            raise H.Abort('insufficient background corners')
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, pts, None)
        if nxt is None or ok is None:
            raise H.Abort('background optical flow unavailable')
        back, backok, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, nxt, None)
        if back is None or backok is None:
            raise H.Abort('background optical flow reverse check unavailable')
        mask = (ok.ravel() != 0) & (backok.ravel() != 0) & (
            np.linalg.norm(back - pts, axis=2).ravel() < .8)
        if mask.sum() < max(12, len(pts) // 2):
            raise H.Abort('background optical-flow consistency lost')
        delta = (nxt - pts).reshape(-1, 2)[mask] * 2
        median = np.median(delta, axis=0)
        if np.median(np.linalg.norm(delta - median, axis=1)) > 3:
            raise H.Abort('nonrigid background motion')
        self.flow += median
        self.previous_gray = gray
        self.previous_template = template
        if (not np.isfinite([*template, *self.flow, score]).all()
                or np.max(np.abs(template - self.flow)) > 7):
            raise H.Abort('bad registration or template/LK disagreement')
        if np.any(np.abs(template) >= FOV) or np.any(np.abs(self.flow) >= FOV):
            raise H.Abort('view limit reached by template or LK')
        return template.tolist(), self.flow.tolist(), float(score)


class Capture:
    def __init__(self, camera, tap, stamped, background_bottom):
        self.camera, self.tap, self.stamped = camera, tap, stamped
        self.registration = Registration(background_bottom)
        self.raw = []
        self.rows = []
        self.fid = 0
        self.last_arrival = 0
        self.last_observed_ns = 0
        self.last_raw = None
        self.phase = 'warmup'
        self.trial = None

    def next(self, timeout=.18, register=True, recovery=False):
        budget_ns = int(timeout * S)
        if register:
            budget_ns = min(budget_ns, FRAME_STALE_NS)
        deadline = time.monotonic_ns() + budget_ns
        if register and self.last_observed_ns:
            deadline = min(deadline, self.last_observed_ns + FRAME_STALE_NS)

        while True:
            if not recovery and (self.tap.fault or not self.tap.th.is_alive()):
                raise H.Abort(self.tap.fault or 'serial watchdog ended')
            remaining_ns = deadline - time.monotonic_ns()
            if remaining_ns <= 0:
                raise H.Abort('no fresh unique registered camera frame within budget')
            frame = self.camera.get(self.fid, remaining_ns / S)
            if frame is None:
                raise H.Abort('camera silent or no fresh unique frame within budget')
            row, raw = self.record_frame(frame, register)
            if not register:
                return row
            if time.monotonic_ns() >= deadline:
                raise H.Abort('no fresh unique registered camera frame within budget')
            if row['duplicate_full_raw16']:
                continue

            template, flow, score = self.registration.update(raw)
            stamp = row['sdk_recv_ns'] if self.stamped else row['arrival_ns']
            finished = time.monotonic_ns()
            if finished >= deadline or finished - stamp > FRAME_STALE_NS:
                raise H.Abort('camera sample became stale during registration')
            if not recovery and self.tap.fault:
                raise H.Abort(self.tap.fault)
            row.update(template_px=template, lk_px=flow, template_score=score, valid=True)
            self.last_observed_ns = stamp
            self.tap.last_sample_ns = stamp
            return row

    def record_frame(self, frame, register):
        fid, sdk, arrival, raw = frame if self.stamped else (frame[0], None, frame[1], frame[2])
        duplicate = self.last_raw is not None and np.array_equal(self.last_raw, raw)
        row = dict(frame_id=fid, sdk_recv_ns=sdk, arrival_ns=arrival, exposure_ns=None,
                   trial=self.trial, phase=self.phase, valid=False, duplicate_full_raw16=duplicate)
        self.rows.append(row)
        self.raw.append((fid, arrival, raw))
        now = time.monotonic_ns()
        if not 0 <= now - arrival <= FRAME_STALE_NS or arrival <= self.last_arrival:
            raise H.Abort('stale or nonmonotonic complete camera arrival')
        if register and self.fid and fid != self.fid + 1:
            raise H.Abort('frame missing during registered sequence')
        if register and self.stamped and not (0 <= arrival - sdk <= FRAME_STALE_NS and
                                             0 <= now - sdk <= FRAME_STALE_NS):
            raise H.Abort('stale camera SDK receipt or buffered frame')
        self.fid = fid
        self.last_arrival = arrival
        self.last_raw = raw
        return row, raw

    def reset_registration_for_stop_view(self):
        self.registration = Registration(self.registration.bottom)
        with self.camera.cv:
            self.fid = len(self.camera.frames)
        self.last_arrival = 0
        self.last_observed_ns = 0
        self.last_raw = None

    def window(self, start_ns, duration_ns, phase, recovery=False):
        self.phase = phase
        rows = []
        while not rows or rows[-1]['arrival_ns'] - start_ns < duration_ns:
            row = self.next(recovery=recovery)
            if row['arrival_ns'] >= start_ns:
                rows.append(row)
        return rows


def stable(rows, bound=1.5):
    if len(rows) < 6 or rows[-1]['arrival_ns'] - rows[0]['arrival_ns'] < 350_000_000:
        raise H.Abort('too few frames for stationary image verification')
    spans = {key: np.ptp(np.array([r[key] for r in rows]), axis=0).tolist()
             for key in ('template_px', 'lk_px')}
    if any(max(span) > bound for span in spans.values()):
        raise H.Abort('post-STOP or baseline image still moving')
    return spans


def displacement_evidence(rows, baseline, late, motion, stop):
    evidence = {}
    keys = ('template_px', 'lk_px')
    after = [r for r in rows if r['arrival_ns'] >= motion['write_start_ns']]
    before_stop = next((r for r in reversed(rows) if r['arrival_ns'] < stop['write_start_ns']), None)
    after_stop = next((r for r in rows if r['arrival_ns'] >= stop['write_end_ns']), None)
    for key in keys:
        baseline_x = np.array([r[key][0] for r in baseline])
        origin = float(np.median(baseline_x))
        rest = float(np.median([r[key][0] for r in late]))
        signed_dx = rest - origin
        sign = np.sign(signed_dx)
        crossing = None
        if np.max(np.abs(baseline_x - origin)) < 1 and abs(signed_dx) >= 1:
            crossing = next((i for i in range(len(after) - 1)
                             if all(sign * (r[key][0] - origin) >= 1 for r in after[i:i + 2])), None)
        lo = (after[crossing - 1] if crossing is not None and crossing > 0 else
              next((r for r in reversed(rows) if r['arrival_ns'] < motion['write_start_ns']), None))
        hi = after[crossing] if crossing is not None else None
        evidence[key] = dict(
            total_dx_px=signed_dx,
            onset_arrival_ms=([(r['arrival_ns'] - motion['write_start_ns']) / 1e6 for r in (lo, hi)]
                              if lo is not None and hi is not None else None),
            plateau_dx_from_last_pre_stop_arrival_px=(rest - before_stop[key][0] if before_stop else None),
            plateau_dx_from_first_post_stop_arrival_px=(rest - after_stop[key][0] if after_stop else None))
    brackets = [evidence[key]['onset_arrival_ms'] for key in keys]
    consistent = (all(brackets) and evidence[keys[0]]['total_dx_px'] * evidence[keys[1]]['total_dx_px'] > 0
                  and max(bounds[0] for bounds in brackets) <= min(bounds[1] for bounds in brackets))
    for metric in evidence.values():
        metric['onset_status'] = 'resolved' if consistent else 'unresolved'
        if not consistent:
            metric['onset_arrival_ms'] = None
    return evidence


def measure(capture, tap, verify, result):
    warm_from = capture.next(timeout=10, register=False)['arrival_ns']
    while capture.last_arrival - warm_from < WARMUP_NS:
        capture.next(timeout=1, register=False)
    for trial, (direction, speed, nominal_ms) in enumerate(PULSES, 1):
        capture.trial = trial
        baseline = capture.window(time.monotonic_ns(), BASELINE_NS, 'baseline')
        stable(baseline)
        if any(np.any(np.abs(baseline[-1][k]) >= START_FOV) for k in ('template_px', 'lk_px')):
            raise H.Abort('starting view is too far from the run origin for another pulse')
        verify()
        tx = tap.start(direction, speed, nominal_ms, trial)
        entry = dict(trial=trial, direction=direction, speed=speed, nominal_ms=nominal_ms,
                     motion_write_ns=tx['write_start_ns'])
        result['trials'].append(entry)
        capture.phase = 'pulse_or_settling'
        stop = None
        while True:
            capture.next()
            stop = next((r for r in reversed(tap.rows)
                         if r.get('reason') == 'pulse_stop' and r.get('trial') == trial), None)
            if stop and stop['ok'] and capture.last_arrival >= stop['write_end_ns'] + DWELL_NS:
                break
        if tap.fault:
            raise H.Abort(tap.fault)
        late = [r for r in capture.rows if r['trial'] == trial and r['valid'] and
                r['arrival_ns'] >= stop['write_end_ns'] + DWELL_NS - 500_000_000]
        trial_rows = [r for r in capture.rows if r['trial'] == trial and r['valid']]
        entry.update(stop_write_ns=stop['write_start_ns'], stop_write_end_ns=stop['write_end_ns'],
                     observed_write_interval_ms=(stop['write_start_ns'] - tx['write_start_ns']) / 1e6,
                     post_stop_capture_ms=(capture.last_arrival - stop['write_end_ns']) / 1e6,
                     late_span_px=stable(late),
                     image_response=displacement_evidence(trial_rows, baseline, late, tx, stop),
                     completed=True)


def save_output(out, capture, tap, result):
    def rows(name, records):
        with (out / name).open('w', encoding='utf-8') as log:
            for record in records:
                log.write(json.dumps(record) + '\n')
    if tap:
        rows('serial.jsonl', tap.rows)
    if capture:
        # A stamped frame that arrived ahead of processing can still be preserved, e.g. on abort.
        raw = capture.raw
        if capture.stamped:
            by_id = {r['frame_id'] for r in capture.rows}
            for fid, sdk, arrival, frame in capture.camera.frames:
                if fid not in by_id:
                    raw.append((fid, arrival, frame))
                    capture.rows.append(dict(frame_id=fid, sdk_recv_ns=sdk, arrival_ns=arrival,
                                             exposure_ns=None, phase='unprocessed', valid=False,
                                             duplicate_full_raw16=None))
        capture.rows.sort(key=lambda row: row['frame_id'])
        rows('frames.jsonl', capture.rows)
        target = out / 'raw16'
        target.mkdir()
        result['raw_frames'] = len(raw)
        result['raw_saved'] = 0
        result['duplicate_full_raw16_frames'] = sum(
            row['duplicate_full_raw16'] is True for row in capture.rows)
        for fid, ns, frame in raw:
            path = target / ('%06d_%d.png' % (fid, ns))
            # imencode + Path handles non-ASCII Windows output names.
            ok, encoded = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if not ok:
                raise OSError('could not encode RAW16 PNG')
            path.write_bytes(encoded.tobytes())
            result['raw_saved'] += 1
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--camera', type=Path, help='optional current stamped cam_pipe executable for real mode')
    parser.add_argument('--background-bottom', type=int, default=BACKGROUND_BOTTOM,
                        help='exclusive bottom row of the upper background ROI (even, 180–360)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--attended', action='store_true')
    args = parser.parse_args(argv)
    repo, out = args.repo.resolve(), args.out.resolve()
    if args.execute and (not args.attended or sys.platform != 'linux'):
        parser.error('real run requires Linux and --execute --attended')
    if not args.execute and args.camera is not None:
        parser.error('--camera requires --execute')
    if not 180 <= args.background_bottom <= 360 or args.background_bottom % 2:
        parser.error('--background-bottom must be even and between 180 and 360')
    if out.exists() or out.is_relative_to(repo):
        parser.error('use a new private directory outside the repository')
    if not (repo / 'tracking_server' / 'src' / 'camera').is_dir():
        parser.error('--repo must contain the tracking server source tree')
    if args.execute:
        H.CAMP = str(args.camera.resolve() if args.camera else repo / 'tracking_server/src/camera/cam_pipe')
        gap.real_preflight(parser, H)
        no_other_camera()
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True)
    if shutil.disk_usage(out).free < MAX_FRAMES * (PIXEL_BYTES + 16384):
        parser.error('insufficient free space for the bounded RAW16 evidence')
    cv2.setNumThreads(1)
    result = dict(evidence_version=EVIDENCE_VERSION, completed=False, dry=not args.execute, trials=[],
                  stop_view_confirmed=False,
                  sdk_recv_time='host CLOCK_MONOTONIC after RecvImage' if args.execute else None,
                  arrival_time='host monotonic_ns after complete frame read',
                  exposure_time=None, serial_time='host monotonic_ns around write; no motor ACK time',
                  limits_px=FOV.tolist(), control_deadline_s=CONTROL_SECONDS,
                  max_camera_sample_age_ms=FRAME_STALE_NS / 1e6,
                  background_roi_xyxy=[16, 0, H.W - 16, args.background_bottom])
    if args.execute:
        result['camera_sha256'] = hashlib.sha256(Path(H.CAMP).read_bytes()).hexdigest()
    camera = tap = capture = None
    old_signals = {}

    def stop_signal(signum, _frame):
        raise KeyboardInterrupt('signal %d' % signum)

    for sig in (signal.SIGINT, signal.SIGTERM, *([signal.SIGHUP] if hasattr(signal, 'SIGHUP') else [])):
        old_signals[sig] = signal.signal(sig, stop_signal)
    with (out / 'camera.log').open('wb') as cam_log:
        try:
            tap = PulseTap(H.DEV if args.execute else None)
            tap.connect()
            camera = StampedCamera(Path(H.CAMP), cam_log) if args.execute else SyntheticCamera(tap)
            capture = Capture(camera, tap, args.execute, args.background_bottom)
            measure(capture, tap, verify_idle if args.execute else lambda: None, result)
        except BaseException as exc:
            result['error'] = repr(exc)
        finally:
            if tap is not None and tap.th.ident is not None:
                try:
                    if not tap.stop('final_stop'):
                        raise H.Abort('final STOP failed; use physical stop')
                    if capture is None:
                        raise H.Abort('no camera available for final STOP verification')
                    stop_ns = tap.last_stop_ns
                    capture.trial = None
                    if result.get('error') or tap.fault:
                        # Check stillness with a new reference even when the run reference was lost.
                        capture.reset_registration_for_stop_view()
                    rows = capture.window(stop_ns, DWELL_NS, 'final_stop', recovery=True)
                    result['final_span_px'] = stable([r for r in rows if r['arrival_ns'] >= stop_ns + 650_000_000])
                    result['stop_view_confirmed'] = True
                except BaseException as exc:
                    result['final_stop_error'] = repr(exc)
                result.update(watchdog_fault=tap.fault, serial_write_failed=tap.write_failed)
                tap.close()
            elif tap is not None:
                tap.close()
            if camera is not None:
                try:
                    result['camera_exit'] = camera.close()
                    if result['camera_exit'] != 0:
                        result['camera_error'] = 'camera exited with nonzero status'
                except BaseException as exc:
                    result['camera_error'] = repr(exc)
            for sig, previous in old_signals.items():
                signal.signal(sig, previous)
    result['completed'] = bool(
        not any(result.get(key) for key in ('error', 'final_stop_error', 'watchdog_fault',
                                            'serial_write_failed', 'camera_error'))
        and result['stop_view_confirmed'] and len(result['trials']) == len(PULSES)
        and all(t.get('completed') for t in result['trials']))
    try:
        save_output(out, capture, tap, result)
    except Exception as exc:
        result.update(completed=False, save_error=repr(exc))
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['completed'] else 3


if __name__ == '__main__':
    sys.exit(main())
