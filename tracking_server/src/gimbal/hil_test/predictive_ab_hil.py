#!/usr/bin/env python3
"""One bounded new-LEAD comparison. Default: synthetic camera + PTY only.

Real mode needs --execute --attended, an idle serial device, and no live server.
No gain sweep, homing, retry, or restart of another process is performed here.
Raw16 frames stay in the private output directory. Camera times are ARRIVAL
times, not sensor exposure times. This isolates control, not person detection.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import pid_hil as H

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictive_lead import estimate


class StrictTap(H.Tap):
    def __init__(self, *args, **kwargs):
        self.stop_ever_failed = False
        super().__init__(*args, **kwargs)
        # Do not wait for camera startup while a previous motor command persists.
        if not self.estop() or self.stop_ever_failed:
            self.close()
            raise H.Abort('initial STOP before camera startup failed')

    def stops(self, n=3, gap=.03):
        count, error = super().stops(n, gap)
        if count != n or error is not None: self.stop_ever_failed = True
        return count, error

    def open_line(self):
        if self.stop_ever_failed: return False
        return super().open_line()


class Bridge(H.Bridge):
    def __init__(self, *args, audit, **kwargs):
        self.audit = audit
        self.tap = args[0]
        self.armed = False
        self.epoch = None
        self.lead_capable = False
        super().__init__(*args, **kwargs)

    def run(self):
        for line in iter(self.p.stdout.readline, b''):
            line = line.decode().strip()
            self.audit.append(dict(ns=time.monotonic_ns(), direction='receive', line=line))
            if line == 'READY': self.ready.set()
            elif line == 'LEAD_CONTROL 1': self.lead_capable = True
            elif line.startswith('CONTROL_EPOCH '): self.epoch = int(line.split()[1])
            elif line.startswith('S '):
                self.state = tuple(map(int, line.split()[1:6]))
                if self.armed and (not self.state[0] or self.state[1] or self.state[3] or self.state[4] != H.TID):
                    self.tap.trip = 'bridge lost/disabled/error/wrong target after selection'
                    self.tap.estop()
            elif line.startswith('R '):
                if line == 'R A OK': self.armed = True
                self.ans.append(line)

    def send(self, text):
        self.audit.append(dict(ns=time.monotonic_ns(), direction='send', line=text))
        self.p.stdin.write((text + '\n').encode())


class MotionCheck:
    """Independent image estimator; not an encoder or actual angle measurement."""
    def __init__(self):
        self.previous = None
        self.total = np.zeros(2)

    def update(self, raw):
        small = cv2.resize(raw, (320, 240), interpolation=cv2.INTER_AREA)
        lo, hi = np.percentile(small, (1, 99))
        if hi - lo < 4: raise H.Abort('insufficient image contrast')
        gray = np.clip((small.astype(float) - lo) * 255 / (hi - lo), 0, 255).astype('uint8')
        if self.previous is None:
            self.previous = gray
            return self.total.copy()
        pts = cv2.goodFeaturesToTrack(self.previous, 160, .02, 8)
        if pts is None or len(pts) < 16: raise H.Abort('insufficient background corners')
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, pts, None)
        back, backok, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, nxt, None)
        mask = (ok.ravel() != 0) & (backok.ravel() != 0) & (np.linalg.norm(back - pts, axis=2).ravel() < .8)
        if mask.sum() < 12: raise H.Abort('optical-flow consistency lost')
        delta = (nxt - pts).reshape(-1, 2)[mask] * 2
        med = np.median(delta, axis=0)
        if np.median(np.linalg.norm(delta - med, axis=1)) > 3:
            raise H.Abort('nonrigid scene: virtual-target measurement invalid')
        self.total += med
        self.previous = gray
        return self.total.copy()


class Session(H.Session):
    def __init__(self, a):
        self.raw_frames = []
        self.audit = []
        self.check = MotionCheck()
        self.previous_sample = None
        self.previous_scene = None
        self.last_epoch = None
        self.lead_ms = 0
        self.lead_space = 'scene'
        self.end_view_checked = False
        super().__init__(a)
        original_get = self.cam.get
        def capture(*args, **kwargs):
            fr = original_get(*args, **kwargs)
            if fr is not None and (not self.raw_frames or fr[0] != self.raw_frames[-1][0]):
                self.raw_frames.append((fr[0], fr[1], fr[2].copy()))
            return fr
        self.cam.get = capture

    def frame(self):
        try:
            result = super().frame()
            if self.est.bad: raise H.Abort('first failed template registration')
            lk = self.check.update(self.raw_frames[-1][2])
            disagreement = float(np.linalg.norm(lk - result[2]))
            self.audit.append(dict(ns=result[1], event='image_check', fid=result[0],
                                   template=result[2].tolist(), optical_flow=lk.tolist(), disagreement_px=disagreement))
            if disagreement > 8: raise H.Abort('image estimators disagree by %.2f px' % disagreement)
            return result
        except BaseException:
            self.tap.estop()
            raise

    def open_bridge(self, opts, key='v27'):
        self.previous_sample = self.previous_scene = self.last_epoch = None
        self.br = Bridge(self.tap, H.BASE + opts, self.elog, 'v27', audit=self.audit)
        self.tap.child = self.br.p
        self.br.send('P')
        self.lastp = time.monotonic()
        until = time.monotonic() + 1
        while time.monotonic() < until and (not self.br.lead_capable or self.br.epoch is None):
            self.tap.pet = time.monotonic()
            time.sleep(.01)
        if not self.br.lead_capable or self.br.epoch is None:
            raise H.Abort('new LEAD capability/epoch handshake absent')

    def close_bridge(self):
        # Cut motor commands BEFORE closing the child; never reopen automatically.
        self.tap.estop()
        self.tap.child = None
        if self.br is not None:
            self.br.armed = False
            self.br.close()
            self.br = None

    def feed(self, fid, ns, pw, vis):
        xi, yi = 320 + pw[0] + self.S[0], 240 + pw[1] + self.S[1]
        box = self.box(xi, yi) if vis and 0 <= xi < 640 and 0 <= yi < 480 else None
        if box is None: raise H.Abort('virtual target left image')
        epoch = self.br.epoch
        if epoch != self.last_epoch:
            self.previous_sample = self.previous_scene = None
            self.last_epoch = epoch
        pair = None
        if self.previous_scene is not None:
            prev_time, prev_scene = self.previous_scene
            delta = self.S - prev_scene
            pair = (prev_time, ns / 1e9, np.array([[1, 0, delta[0]], [0, 1, delta[1]]]))
        row = (H.TID, *box, .9)
        self.previous_sample, lead = estimate(self.previous_sample, H.TID, row, ns / 1e9, pair, self.lead_ms, space=self.lead_space)
        self.previous_scene = (ns / 1e9, self.S.copy())
        wait = ns / 1e9 + self.a.latency_ms / 1000 - time.monotonic()
        if wait > 0: time.sleep(wait)
        message = 'F %d %d 1 %d %.4f %.4f %.4f %.4f .9 LEAD %d %.6f %.6f %d' % (
            fid, ns, H.TID, box[0], box[1], box[2]-box[0], box[3]-box[1], H.TID, *lead, epoch)
        self.br.send(message)
        return (xi, yi), ((xi - 320) / 320, (yi - 240) / 240), 1

    def stationary_after_stop(self, tag):
        if not self.tap.estop() or self.tap.stop_ever_failed: raise H.Abort('STOP write failed at least once')
        begin = time.monotonic()
        reference = H.Shift()
        independent = MotionCheck()
        late = []
        prev = None
        repeated = 0
        while time.monotonic() - begin < 1.6:
            fr = self.cam.get(self.last, .30)
            if fr is None: raise H.Abort('no image after STOP')
            self.last = fr[0]
            self.tap.pet = time.monotonic()
            if (time.monotonic_ns() - fr[1]) / 1e9 > .30: raise H.Abort('stale post-STOP frame')
            if prev is not None:
                if fr[1] <= prev[1]: raise H.Abort('nonmonotonic post-STOP camera timestamp')
                repeated = repeated + 1 if np.array_equal(fr[2], prev[2]) else 0
                if repeated >= 2: raise H.Abort('frozen post-STOP raw image')
            prev = fr
            if time.monotonic() - begin < .8: continue
            xy, quality = reference.update(fr[2])
            if reference.bad: raise H.Abort('cannot verify image stability after STOP')
            lk = independent.update(fr[2])
            if np.linalg.norm(lk - xy) > 4: raise H.Abort('post-STOP image estimators disagree')
            late.append(xy.copy())
        if len(late) < 5: raise H.Abort('too few post-STOP frames')
        span = np.ptp(np.array(late), axis=0)
        self.audit.append(dict(event='post_stop', ns=time.monotonic_ns(), tag=tag, span_px=span.tolist(), frames=len(late)))
        if np.max(span) > 4: raise H.Abort('view still moves after STOP: %s' % span)
        if self.tap.stop_failed: raise H.Abort('STOP write failed')
        self.end_view_checked = True

    def compare(self):
        self.tap.guard = self.tap.guard_y = 10
        if not self.tap.estop() or self.tap.stop_ever_failed: raise H.Abort('initial STOP write failed')
        # Warm camera while the serial line stays blocked, then validate texture.
        self.static(dict(T=5, blind=True))
        self.static(dict(T=2))
        initial_scene = self.S.copy()
        horizons = (self.a.lead_ms, 0) if self.a.candidate_first else (0, self.a.lead_ms)
        conditions = [(lead, 'scene') for lead in horizons]
        if self.a.reference_comparison: conditions = [(60, 'image'), (60, 'scene')]
        for index, (lead, space) in enumerate(conditions):
            self.step_i = index + 1
            self.lead_ms = lead
            self.lead_space = space
            self.end_view_checked = False
            self.frame()
            if np.max(np.abs(self.S - initial_scene)) > 22:
                raise H.Abort('starting view differs by more than the common deadband tolerance')
            if not self.tap.open_line(): raise H.Abort('serial line cannot reopen after STOP fault')
            name = '%s_%d' % (space, lead)
            print('BLOCK %s cap=10' % name, flush=True)
            self.track(dict(name=name, bridge='v27', spd_guard=10,
                            opts=H.HOME_OPTS, dur=8, errlim=[110, 55],
                            wp=[[0,0,0],[1,0,0],[2,60,0],[3,60,0],[4,0,0],[8,0,0]]))
            self.stationary_after_stop(name)
            self.f.flush()
            self.slog.flush()
        self.elog.flush()
        records = [json.loads(line) for line in Path(self.elog.name).read_text().splitlines() if line.startswith('{"event":"control_trace"')]
        if not any(row['lead_valid'] and any(row['forecast_used'].values()) for row in records):
            raise H.Abort('no nonzero forecast was applied by the controller')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bridge', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--attended', action='store_true')
    parser.add_argument('--lead-ms', type=int, choices=(60, 75, 150), default=150)
    parser.add_argument('--candidate-first', action='store_true')
    parser.add_argument('--reference-comparison', action='store_true', help='image60 then scene60; isolated experimental comparison')
    a = parser.parse_args()
    if a.execute and not a.attended: parser.error('real test requires --attended')
    if Path(a.out).exists(): parser.error('output already exists; preserve previous evidence')
    if not os.access(a.bridge, os.X_OK): parser.error('bridge executable missing')
    a.dry = not a.execute
    a.simK, a.simL, a.simMin, a.simTau, a.curve = 6., .08, 0., .05, None
    a.max_seconds, a.limx, a.limy, a.spd_abort, a.latency_ms = 75., 130., 65., 10, 40.
    H.BRIDGES['v27'] = str(Path(a.bridge).resolve())
    H.Tap = StrictTap
    cv2.setNumThreads(1)
    if a.execute:
        if H.other_servers(): parser.error('live server still running')
        # USB reconnect can change ttyUSB0 to ttyUSB1. Use the same stable
        # CP2102N identity as the operational launcher, not a numbered fallback.
        devices = sorted(glob.glob(H.DEV_GLOB))
        if len(devices) != 1: parser.error('expected exactly one stable gimbal serial identity')
        H.DEV = devices[0]
        if not os.path.exists(H.DEV): parser.error('gimbal serial link is absent')
        if subprocess.run(['fuser', H.DEV], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            parser.error('gimbal serial device busy')
        if not os.access(H.CAMP, os.X_OK): parser.error('camera program missing')
    os.environ['LWIR_CONTROL_TRACE'] = '1'
    def stop_signal(signum, frame):
        for tap in H.TAPS: tap.estop()
        raise KeyboardInterrupt('signal %d' % signum)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, stop_signal)
    session = None
    result = dict(success=False, dry=a.dry, settings=vars(a), test='virtual-target control only')
    try:
        session = Session(a)
        session.compare()
        result['success'] = True
    except BaseException as exc:
        result['error'] = repr(exc)
        print('ABORT', repr(exc), flush=True)
    finally:
        if session is not None:
            session.tap.estop()
            if not session.end_view_checked:
                try: session.stationary_after_stop('abort')
                except BaseException as exc: result['post_stop_error'] = repr(exc)
            result['end_view_checked'] = session.end_view_checked
            result['max_speed'] = session.tap.maxspd
            try:
                result['camera_exit'] = session.close()
            except BaseException as exc:
                session.tap.estop()
                result['success'] = False
                result['close_error'] = repr(exc)
            result['stop_failed'] = session.tap.stop_failed
            result['stop_ever_failed'] = session.tap.stop_ever_failed
            result['watchdog_trip'] = session.tap.trip
            if session.tap.trip or session.tap.stop_ever_failed: result['success'] = False
            rawdir = Path(a.out) / 'raw16'
            rawdir.mkdir()
            for fid, ns, raw in session.raw_frames:
                if not cv2.imwrite(str(rawdir / ('%06d_%d.png' % (fid, ns))), raw):
                    result['success'] = False
                    result['save_error'] = 'RAW16 frame %d was not saved' % fid
            with open(Path(a.out) / 'protocol.jsonl', 'w') as output:
                for event in session.audit: output.write(json.dumps(event) + '\n')
            result['raw_frames'] = len(session.raw_frames)
        Path(a.out).mkdir(exist_ok=True, parents=True)
        (Path(a.out) / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['success'] and result.get('end_view_checked') and not result.get('stop_ever_failed') else 3


if __name__ == '__main__': sys.exit(main())
