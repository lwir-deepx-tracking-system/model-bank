#!/usr/bin/env python3
"""Real gimbal, virtual target (2026-11-10). Tunes the PID gains of gimbal_bridge_v25 without a person in front of the camera.

The camera image shift between frames (phase correlation, Kuglin & Hines 1975) tells how far the gimbal has turned, in image pixels.
A virtual person is "painted" on the scene: image position = world position + scene displacement. Its box goes to the real bridge
(F lines), so the team's unchanged controller closes the loop on the real gimbal. A pseudo terminal sits between the bridge and
/dev/ttyUSB0: every command is logged with its time, and this program can cut the line and send STOP itself.

--dry: no camera, no gimbal. A synthetic scene moves as  d(scene)/dt = K * SPD  after a delay L (integrator + dead time).
Plan file: JSON list of steps  {"type":"static","T":3} | {"type":"ident","axis":"x","speed":5,"T":1.5,"maxpx":120}
           | {"type":"track","name":"..","opts":["--controller","pid",...],"dur":8,"wp":[[t,x,y],...],"drop":[[t0,t1],...]}
After every ident / track step the gimbal is driven back to the start pose with the plain P controller (home).

v27 test copy (2026-11-11, every changed line carries "v27"; list and reasons in CHANGES_FROM_M1_TOOLS_KO.md):
new server folder; a track step may name its bridge ("bridge": "v25" | "v27") and extra options ("extra": [...]); the gimbal is stopped
when 3 measurements in a row are identical or the picture is older than STALE_S; vertical hold band in the home step; a watchdog in the
serial tap thread; excursion limits, speed guard and total run time limit end the run; the end position is written to end_state.json.

Second round, same day (lines marked "v27b": the safety review's findings and the three stages; list in CHANGES_FROM_M1_TOOLS_KO.md):
a bridge program that ends by itself is noticed by the tap thread and by the main loop, STOP goes straight to the device; 3 failed image
matches in a row while a velocity command is active stop the gimbal first; the emergency stop records whether its write succeeded and
opens the gimbal device again by its fixed name when it did not; identical measurements while the command is zero are only counted;
speed guard per step; virtual person's box size per step, aim height 0.08 (head) as in live use; a block whose error passes its limit
ends as "person lost" (STOP first) instead of ending the run; timed pulses for the stage 1 identification (the tap itself writes the STOP).

Third round, 2026-11-12 (lines marked "v27c": the second safety review, ../review2; rules written in PLAN_V27_HILS_20261111_KO.md first):
up-down guard in the tap thread for following blocks (a command that moves the up-down axis with SPD above GUARD_Y: STOP at once, the block ends
as "person lost"); while the live command is above FAST_SPD the picture-age limit is STALE_FAST_S and the watchdog limit WD_FAST_S; the main loop
notices a dead tap thread (STOP, the run ends); STOP when a velocity command is still alive after a bridge was closed; a following block's speed
guard is never above FOLLOW_GUARD; a view that still moves after the last STOP ends with exit code 7 (found in lead_hil.py where())."""
import argparse, collections, csv, glob, json, os, select, signal, subprocess, sys, threading, time   # v27b: + glob
try: import pty, termios, tty                     # v27: these do not exist on a Windows PC (dry run there uses a plain file instead of a pseudo terminal)
except ImportError: pty = termios = tty = None    # v27
import numpy as np, cv2

ROOT = os.environ.get('HIL_ROOT', os.path.expanduser('~/workspace/lwir_model_bank/tracking_server'))   # v27: new server folder (HIL_ROOT: only for a dry run on a PC)
BRIDGES = {'v25': ROOT + '/src/gimbal/gimbal_bridge_v25', 'v27': ROOT + '/src/gimbal/gimbal_bridge_v27'}   # v27: a track step names its bridge program
BRIDGE, CAMP, DEV = BRIDGES['v25'], ROOT + '/src/camera/cam_pipe', '/dev/ttyUSB0'   # v27: new locations
DEV_GLOB = '/dev/serial/by-id/usb-Silicon_Labs_CP2102N_*'   # v27b: the gimbal's fixed name; the emergency stop opens it again when a write to the open device fails (USB dropped and came back under another ttyUSB number)
W, H, BW, BH, AIM, TID = 640, 480, 40.0, 100.0, 0.08, 7   # v27b: aim height 0.08 of the box from its top = the head, as in live use since 11-11 (was 0.3)
LIMX, LIMY, RESP_MIN = 230.0, 100.0, 0.45       # hard excursion limits (pixels of scene displacement), weakest accepted correlation peak   # v27: was 260 / 200
STALE_S, SAME_N, WD_S, SPD_ABORT = 0.30, 3, 0.45, 40   # v27: oldest accepted picture (s), identical measurements in a row = frozen picture, main-loop silence the watchdog accepts while a velocity command is active (s), highest accepted SPD
BAD_MOVING_N, GATE_PX, HOME_GUARD, MAX_ERRLIM = 3, 36.0, 10, 3   # v27b: failed image matches in a row that stop the gimbal while a velocity command is active; largest accepted jump between two frames (px, as before); speed guard of the home step (its bridge runs with --speed-max 10); blocks per run that may end as "person lost"
FAST_WINDOW = 0.3                               # v27d (third review, minor 2): "fast" = the highest SPD that reached the gimbal line in the last FAST_WINDOW s is above FAST_SPD (the gimbal still runs on a command of 0.1..0.19 s ago)
STALE_FAST_S, WD_FAST_S, FAST_SPD = 0.15, 0.25, 10   # v27c (second review): while the live speed command is above FAST_SPD: oldest accepted picture / longest wait for a new one (s), main-loop silence the watchdog accepts (s). At 800 px/s every 0.1 s is 80 px
GUARD_Y, FOLLOW_GUARD = 10, 40                  # v27c (second review): following blocks only: highest SPD of a command that moves the up-down axis (the team's control code uses ONE speed for both axes; 10 = the cap of the home step); highest speed guard a following block can get, whatever --spd-abort says
HOME_DBY, HOME_DBY_OUT = 24.0, 34.0             # v27: home step, vertical hold band (px): inside it no vertical command is asked for (this morning the home step swung +-22 px up and down for 7 s)
DRY_BRIDGE = None                               # v27: dry run on a PC without the real bridge programs: a stand-in script (--dry-bridge)
BASE = ['--manual-speed', '10', '--speed-max', '10', '--aim-y', str(AIM), '--x-sign', '1', '--y-sign', '1']
HOME_OPTS = ['--kp', '30.0', '--controller', 'pid', '--ki', '0', '--kd', '6', '--deadband', '0.06']   # v27b: the home step uses the PD values and the deadband tuned on 11-10 (was plain P with the default deadband 0.04: the combination the 11-10 record found swinging to and fro; this morning each home step swung for most of its 14 s). Speed cap 10 as before
HOME_X = 22.0                                   # v27b: home step, "back at the start view" left-right (px): just outside the deadband 0.06 = 19.2 px (was 14, inside it: the step could not end before its time-out)
STOP_MSG = b'{"T":135}\n'                        # v27b


class Abort(Exception): pass


class ErrLimit(Abort): pass                        # v27b: the virtual person got further from the picture centre than the block allows = "person lost" for that block (the gimbal is stopped first)


TAPS = []                                          # v27: for the signal handler: STOP goes out before anything else happens
SIG = []                                           # v27: signals received


def open_device(dev):                              # v27b: the device set-up that was inside Tap.__init__, now also used when the device is opened again
    fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK); tty.setraw(fd); a = termios.tcgetattr(fd)
    a[4] = a[5] = termios.B115200; a[2] |= termios.CLOCAL | termios.CREAD; a[2] &= ~(termios.PARENB | termios.CSTOPB | termios.CRTSCTS)
    termios.tcsetattr(fd, termios.TCSANOW, a); return fd


def only_stops(d):                                 # v27b: is this chunk from the bridge nothing but STOP commands?
    try: return all(json.loads(a).get('T') == 135 for a in d.decode('latin1').splitlines() if a.strip())
    except Exception: return False


class Tap:
    """pseudo terminal for the bridge; forwards to the real device (not in dry mode) and logs every command"""
    last = property(lambda s: s._last, lambda s, v: (setattr(s, '_last', v), s.lh.append((time.monotonic(), v)), setattr(s, 'fast_t', time.monotonic()) if v[2] > FAST_SPD else None)[0])   # v27b: the gimbal's command as this tap knows it. Every change is also kept with its time (the last 4000), which only the pretend gimbal of the dry run reads   # v27d: + fast_t, the last time a command above FAST_SPD reached the line (kept here, not in lh: the dry run's pretend gimbal empties lh)

    def fast(s):                                       # v27d (third review, minor 2): was "s.last[2] > FAST_SPD" (the command at this instant only)
        return s._last[2] > FAST_SPD or time.monotonic() - s.fast_t <= FAST_WINDOW

    def __init__(s, dry, log):
        s.lh = collections.deque(maxlen=4000); s.fast_t = -1e9   # v27b   # v27d: + fast_t
        if pty is not None: s.master, s.slave = pty.openpty(); tty.setraw(s.master); s.name = os.ttyname(s.slave)   # v27: only where pseudo terminals exist
        else: s.master = None; s.name = os.path.abspath(log.name) + '.link'; open(s.name, 'wb').close(); s.rd = open(s.name, 'rb')   # v27: Windows PC, dry only: the stand-in bridge appends its commands to this file
        s.fd = None; s.block = False; s.last = (0, 0, 0); s.buf = b''; s.log = log; s.done = False; s.n = 0; s.fb = 0
        s.pet = time.monotonic(); s.deadline = None; s.trip = None; s.maxspd = 0; s.nstop = 0; s.lk = threading.RLock(); TAPS.append(s)   # v27: watchdog state; lk: no command can slip onto the device line after a STOP
        s.guard = SPD_ABORT; s.child = None; s.stop_failed = False; s.nfail = 0; s.nreopen = 0; s.dry_fail = 0; s.dev = None   # v27b: speed guard of the current step; the bridge process to watch; result of the emergency stops; dry_fail: forced "write failed" (dry mode only)
        s.arm = None; s.pulse_end = None; s.mute = False; s.t_pvel = s.t_pstop = None; s.npulse = 0   # v27b: timed pulse (stage 1): armed length (s), end time, "velocity commands held back after the pulse", times of the pulse on the line (ns)
        s.real = not dry                               # v27b
        s.guard_y = None; s.ykick = 0                  # v27c: up-down guard of the current step (None = none: home step, stage 1 moves); the SPD of the command that tripped it (0 = not tripped), read and cleared by the main loop
        if not dry: s.fd = open_device(DEV); s.dev = DEV   # v27b: was written out here (now open_device)
        s.th = threading.Thread(target=s.run, daemon=True); s.th.start()

    def watch(s):
        """v27: runs in this tap thread, so it works even when the main loop hangs. A velocity command with a silent main loop, or the end of the allowed run time -> STOP and the line cut.
        v27b: also a bridge program that ended by itself (it sends no STOP then: its last velocity command would stay active), and the end of a timed pulse."""
        now = time.monotonic()
        if s.pulse_end is not None and now >= s.pulse_end:   # v27b: stage 1: the pulse length is kept here, at the device line, not by the main loop
            with s.lk:
                s.pulse_end = None; s.mute = True; n, e = s.stops(1, 0); s.last = (0, 0, 0); s.t_pstop = time.monotonic_ns()   # the time of the move's end = right after the first STOP write
                time.sleep(0.005); n2, e2 = s.stops(1, 0); n += n2; e = e2 or e                                     # a second STOP 5 ms later
            s.note(dict(ns=s.t_pstop, pulse_stop=1, ok=int(n > 0), n=n, **({'err': e} if e else {})))
            if n == 0 and s.trip is None: s.trip = 'the STOP that ends a timed pulse could not be written (%s)' % e; s.estop()
        c = s.child                                    # v27b
        if s.trip is None and c is not None and c.poll() is not None: s.trip = 'bridge program ended by itself (exit code %s)' % c.returncode; s.estop()   # v27b
        wd = WD_FAST_S if s.fast() else WD_S             # v27c: a shorter limit while the gimbal runs fast   # v27d: s.fast() (was s.last[2] > FAST_SPD)
        if s.trip is None and s.last[2] and now - s.pet > wd: s.trip = 'main loop silent for %.2f s while a velocity command was active (SPD %d, limit %.2f s)' % (now - s.pet, s.last[2], wd); s.estop()   # v27c: wd (was WD_S)
        if s.trip is None and s.deadline is not None and now > s.deadline: s.trip = 'total run time limit (watchdog)'; s.estop()

    def note(s, d):                                    # v27b: a closed log must never hide what follows
        try: s.log.write(json.dumps(d) + '\n'); s.log.flush()
        except ValueError: pass

    def run(s):
        while not s.done:
            if s.master is None: time.sleep(0.004); d0 = s.rd.read(4096); r = [None] if d0 else []   # v27: Windows PC, dry only
            else:                                      # v27: the two lines below are unchanged, only indented
                fds = [s.master] + ([s.fd] if s.fd is not None else [])
                pe = s.pulse_end                       # v27b
                try: r, _, _ = select.select(fds, [], [], 0.05 if pe is None else min(0.05, max(0.001, pe - time.monotonic())))   # v27b: wakes in time for the end of a pulse; a device that was just closed and opened again must not end this thread
                except (OSError, ValueError): r = []   # v27b
            s.watch()                                  # v27
            if s.master in r:
                d = os.read(s.master, 4096) if s.master is not None else d0; ns = time.monotonic_ns()   # v27: d0 on a Windows PC
                werr = None                            # v27b
                with s.lk:                             # v27: checked and written under the lock that estop() takes (the two lines below are unchanged except for blk)
                    blk = s.block or (s.mute and not only_stops(d))   # v27: was this chunk passed on to the device?   # v27b: after a timed pulse velocity commands are held back until the main loop opens the line again; a STOP always passes
                    if s.fd is not None and not blk:
                        try: os.write(s.fd, d)
                        except OSError as e: werr = repr(e); s.log.write(json.dumps(dict(ns=ns, err=str(e))) + '\n')   # v27b: + werr
                if werr and s.trip is None: s.trip = 'a command could not be written to the gimbal device (%s)' % werr; s.estop()   # v27b: was only logged. The gimbal keeps its last command then: STOP, with the device opened again by its fixed name
                s.buf += d
                while b'\n' in s.buf:
                    a, s.buf = s.buf.split(b'\n', 1)
                    try: c = json.loads(a.decode().strip())
                    except Exception: c = dict(raw=a.decode('latin1'))
                    if c.get('T') == 141 and not blk and not s.block and not werr:  # v27: a command that was not passed on, or that a STOP has overtaken, does not count as the gimbal's state
                        s.last = (int(c.get('X', 0)), int(c.get('Y', 0)), int(c.get('SPD', 0))); s.maxspd = max(s.maxspd, s.last[2])   # v27: + highest SPD seen
                        if s.last[2] > s.guard and s.trip is None: s.trip = 'speed command %d above the guard %d' % (s.last[2], s.guard); s.estop()   # v27: speed guard   # v27b: the guard of the current step
                        elif s.guard_y is not None and s.last[1] and s.last[2] > s.guard_y: yk = s.last; s.ykick = yk[2]; s.note(dict(ns=ns, yguard=1, X=yk[0], Y=yk[1], SPD=yk[2], guard_y=s.guard_y)); s.estop()   # v27c (second review): up-down guard. STOP at once and the line cut; no trip: the main loop ends this block as "person lost" and the run goes on
                        elif s.arm is not None and s.last[2] and (s.last[0] or s.last[1]): s.pulse_end = time.monotonic() + s.arm; s.arm = None; s.t_pvel = ns; s.t_pstop = None; s.npulse += 1   # v27b: a timed pulse starts when its command reaches the line
                    elif c.get('T') == 135 and not blk and not werr: s.last = (0, 0, 0); s.nstop += 1   # v27: + count
                    s.n += 1; s.log.write(json.dumps(dict(ns=ns, cmd=c, **({'blocked': 1} if blk else {}), **({'werr': 1} if werr else {}))) + '\n')   # v27: + blocked mark   # v27b: + werr mark (the write to the device failed)
            if s.fd is not None and s.fd in r:
                gone = None                                # v27b
                try:                                       # device answers are logged, not passed on (the bridge never needs them)
                    d = os.read(s.fd, 4096); ns = time.monotonic_ns(); s.fb += len(d); s.rbuf = getattr(s, 'rbuf', b'') + d
                    if not d: gone = 'its line was hung up'   # v27b
                    while b'\n' in s.rbuf:
                        a, s.rbuf = s.rbuf.split(b'\n', 1); s.log.write(json.dumps(dict(ns=ns, rx=a.decode('latin1').strip())) + '\n')
                except (BlockingIOError, TypeError, ValueError): pass   # v27b: nothing to read after all, or the device was just opened again
                except OSError as e: gone = repr(e)        # v27b: was "pass". A device that cannot be read any more has gone (USB dropped): the gimbal keeps its last command
                if gone and s.trip is None: s.trip = 'the gimbal device is gone (%s)' % gone; s.estop()   # v27b: STOP, with the device opened again by its fixed name
                elif gone: time.sleep(0.02)                # v27b

    def stops(s, n=3, gap=0.03):
        """v27b: writes STOP n times to the open device. Returns (complete writes, last error). Dry mode: nothing to write = counted as written, unless a failure is forced."""
        if s.fd is None: return (0, 'forced failure (dry mode only)' if s.dry_fail else 'gimbal device not open') if (s.dry_fail or s.real) else (n, None)
        ok = 0; err = None
        for i in range(n):
            try:
                if os.write(s.fd, STOP_MSG) == len(STOP_MSG): ok += 1
                else: err = 'short write'
            except OSError as e: err = repr(e)
            if i + 1 < n: time.sleep(gap)
        return ok, err

    def reopen(s):
        """v27b: the open device took no STOP: close it and open the gimbal again by its fixed name (the first by-id entry; the start-up name only when there is none)."""
        if not s.real: return None
        try:
            if s.fd is not None: os.close(s.fd)
        except OSError: pass
        s.fd = None; c = sorted(glob.glob(DEV_GLOB)) + [s.dev]
        for dev in c:
            try: s.fd = open_device(dev); s.nreopen += 1; return dev
            except Exception: continue
        return None

    def estop(s):
        """STOP straight to the device and the command line cut. v27b: returns whether a STOP was written; a failed write is recorded and tried again on the device opened by its fixed name."""
        with s.lk:
            s.block = True; s.arm = None; s.pulse_end = None   # v27: under the lock: from here on nothing is passed on
            real = s.real; n, err = s.stops(); re = None
            if n == 0 and real:
                re = s.reopen() or 'FAILED'
                if s.fd is not None: n, err2 = s.stops(); err = '%s | after opening %s again: %s' % (err, re, err2 or 'written')
        s.last = (0, 0, 0); ok = n > 0
        if not ok: s.stop_failed = True; s.nfail += 1
        else: s.stop_failed = False
        s.note(dict(ns=time.monotonic_ns(), estop=1, ok=int(ok), n=n, why=s.trip or '', **({'err': err} if err else {}), **({'reopened': re} if re else {}), **({} if real or s.dry_fail else {'dry': 1})))   # v27: + reason   # v27b: + ok, n, err, reopened
        return ok

    def open_line(s):                                  # v27b: the only way the line is opened again after a STOP; never after a watchdog stop or a STOP that could not be written
        if s.trip or s.stop_failed: return False
        with s.lk: s.block = False; s.mute = False
        return True

    def close(s):
        s.done = True; s.th.join(1.0)
        if s.fd is not None: fd, s.fd = s.fd, None; os.close(fd)   # v27: safe to call twice (was: os.close(s.fd))
        if s.master is None: s.rd.close()              # v27


class Cam:
    def __init__(s, log):
        s.p = subprocess.Popen([CAMP], stdout=subprocess.PIPE, stderr=log, bufsize=0, cwd=ROOT + '/src/camera')   # v27: new folder
        s.cv = threading.Condition(); s.cur = None; s.fid = 0; s.eof = False
        s.th = threading.Thread(target=s.run, daemon=True); s.th.start()

    def run(s):
        n = W * H * 2; fd = s.p.stdout.fileno()
        while True:                                   # read to EOF, always (cam_pipe must never block on a full pipe)
            b = bytearray()
            while len(b) < n:
                d = os.read(fd, n - len(b))
                if not d: break
                b += d
            if len(b) < n: break
            ns = time.monotonic_ns()
            with s.cv: s.fid += 1; s.cur = (s.fid, ns, np.frombuffer(bytes(b), np.uint16).reshape(H, W)); s.cv.notify_all()
        with s.cv: s.eof = True; s.cv.notify_all()

    def get(s, last, timeout):
        end = time.monotonic() + timeout
        with s.cv:
            while (s.cur is None or s.cur[0] <= last) and not s.eof:
                left = end - time.monotonic()
                if left <= 0: return None
                s.cv.wait(left)
            return None if (s.cur is None or s.cur[0] <= last) else s.cur

    def close(s):
        s.p.send_signal(signal.SIGTERM)
        try: s.p.wait(45)
        except Exception: s.p.kill()
        return s.p.returncode


class FakeCam:
    """dry mode: the scene moves K pixels/s per SPD unit, L seconds after the command"""
    def __init__(s, tap, K, L, fps=24.0, vmin=0.0, curve=None, tau=0.0):   # v27: + vmin   # v27b: + curve (px/s per SPD, a table), tau (the pretend gimbal speeds up and slows down with this time constant)
        rng = np.random.default_rng(1); big = cv2.GaussianBlur(rng.normal(0, 1, (1100, 1500)).astype(np.float32), (0, 0), 6)
        big = big / big.std() * 300
        for _ in range(60):
            x, y, w, h = int(rng.integers(0, 1400)), int(rng.integers(0, 1000)), int(rng.integers(15, 90)), int(rng.integers(15, 90)); big[y:y + h, x:x + w] += float(rng.normal(0, 500))
        s.big = big + 8000; s.tap, s.K, s.L, s.fps = tap, K, L, fps; s.pos = np.zeros(2); s.vmin = vmin; s.curve = curve; s.tau = tau; s.v = np.zeros(2)   # v27: + vmin   # v27b
        s.cv = threading.Condition(); s.cur = None; s.fid = 0; s.eof = False; s.done = False
        s.noise_until = 0.0                            # v27b: forced "pictures that match nothing"
        s.freeze_until = s.stall_until = s.old_until = 0.0   # v27: forced faults for the dry run (the pretend gimbal keeps moving meanwhile, as the real one did)
        s.th = threading.Thread(target=s.run, daemon=True); s.th.start()
        s.ph = threading.Thread(target=s.phys, daemon=True); s.ph.start()   # v27b

    def speed(s, spd):                                 # v27b: px/s of the pretend gimbal for one SPD value
        if not spd: return 0.0
        if s.curve: return float(np.interp(spd, [p[0] for p in s.curve], [p[1] for p in s.curve]))
        return max(s.K * spd, s.vmin)                  # v27: --simMin = the real gimbal's lowest speed (about 160 px/s whatever low SPD is asked, measured 11-10); 0 = as before

    def phys(s):
        """v27b: the pretend gimbal's motion, in its own thread (was: once per picture, too coarse for pulses of 50 ms). A command acts L seconds after the time
        at which it reached the line (the tap's own time stamp, so a late wake-up of this thread does not lengthen a move); so does a STOP."""
        pend = collections.deque(); cmd = (0, 0, 0); t0 = time.monotonic(); pos = s.pos.copy(); v = s.v.copy()
        def go(dt):
            nonlocal pos, v
            if dt <= 0: return
            tgt = np.array([-cmd[0], cmd[1]], float) * s.speed(cmd[2])
            if s.tau <= 0: v = tgt; pos = pos + tgt * dt
            else: k = float(np.exp(-dt / s.tau)); pos = pos + tgt * dt + (v - tgt) * s.tau * (1.0 - k); v = tgt + (v - tgt) * k
        while not s.done:
            time.sleep(0.004); now = time.monotonic()
            while s.tap.lh: pend.append(s.tap.lh.popleft())
            while pend and pend[0][0] + s.L <= now: te = max(pend[0][0] + s.L, t0); go(te - t0); t0 = te; cmd = pend.popleft()[1]
            go(now - t0); t0 = now; s.pos = pos.copy(); s.v = v.copy()

    def run(s):
        nxt = time.monotonic(); rng = np.random.default_rng(2)
        while not s.done:
            nxt += 1 / s.fps; time.sleep(max(0, nxt - time.monotonic())); now = time.monotonic(); pos = s.pos.copy()   # v27b: the position comes from phys()
            if now < s.stall_until: continue               # v27: forced "no picture at all"
            if now < s.freeze_until and s.cur is not None:  # v27: forced frozen picture: new frame number and time, the same picture
                with s.cv: s.fid += 1; s.cur = (s.fid, time.monotonic_ns(), s.cur[2]); s.cv.notify_all()
                continue
            M = np.float32([[1, 0, pos[0] - 430], [0, 1, pos[1] - 310]])
            img = cv2.warpAffine(s.big, M, (W, H), flags=cv2.INTER_LINEAR) + rng.normal(0, 8, (H, W)).astype(np.float32)
            if now < s.noise_until: img = 8000 + rng.normal(0, 300, (H, W)).astype(np.float32)   # v27b
            with s.cv: s.fid += 1; s.cur = (s.fid, time.monotonic_ns() - (500000000 if now < s.old_until else 0), np.clip(img, 0, 65535).astype(np.uint16)); s.cv.notify_all()   # v27: forced "picture 0.5 s old"

    get = Cam.get

    def close(s): s.done = True; return 0


class Shift:
    """scene displacement since the first frame, in full-resolution pixels (+x: the scene moved right in the image).
    The centre patch of a key frame is searched in the new frame by normalized cross-correlation (Lewis 1995) at half resolution."""
    MX, MY = 60, 45                                   # search range, half-resolution pixels

    def __init__(s): s.key = None; s.off = np.zeros(2); s.S = np.zeros(2); s.bad = 0; s.gate = GATE_PX   # v27b: + gate (the largest accepted jump between two frames; raised only for the stage 1 pulses)

    def prep(s, raw):
        f = cv2.resize(raw, (W // 2, H // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
        return cv2.GaussianBlur(f, (0, 0), 1.0) - cv2.GaussianBlur(f, (0, 0), 12.0)

    def update(s, raw):
        f = s.prep(raw)
        if s.key is None: s.key = f; return s.S, 1.0
        r = cv2.matchTemplate(f, s.key[s.MY:H // 2 - s.MY, s.MX:W // 2 - s.MX], cv2.TM_CCOEFF_NORMED); _, mx, _, (px, py) = cv2.minMaxLoc(r)
        dx, dy = float(px), float(py)
        if 0 < px < r.shape[1] - 1:
            a, b, c = r[py, px - 1], r[py, px], r[py, px + 1]; q = a - 2 * b + c; dx += 0.5 * (a - c) / q if q < -1e-9 else 0.0
        if 0 < py < r.shape[0] - 1:
            a, b, c = r[py - 1, px], r[py, px], r[py + 1, px]; q = a - 2 * b + c; dy += 0.5 * (a - c) / q if q < -1e-9 else 0.0
        d = np.array([dx - s.MX, dy - s.MY]) * 2.0; new = s.off + d
        if mx < RESP_MIN or np.abs(new - s.S).max() > s.gate * (s.bad + 1): s.bad += 1; return s.S, float(mx)   # v27b: s.gate (was the number 36.0)
        s.bad = 0; s.S = new
        if abs(d[0]) > 56 or abs(d[1]) > 40: s.key = f; s.off = s.S.copy()
        return s.S, float(mx)


class Bridge:
    def __init__(s, tap, opts, errlog, key='v25'):   # v27: + which bridge program
        if key not in BRIDGES: raise Abort('unknown bridge "%s" in the plan (v25 or v27)' % key)   # v27
        prog = [sys.executable, DRY_BRIDGE, '--standin', key] if DRY_BRIDGE else [BRIDGES[key]]   # v27
        s.p = subprocess.Popen([*prog, '--device', tap.name, '--heartbeat-ms', '600', *opts], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errlog, bufsize=0)   # v27: prog instead of BRIDGE
        s.lines = collections.deque(); s.state = (0, 0, 0, 0, 0); s.ready = threading.Event(); s.ans = collections.deque()
        s.th = threading.Thread(target=s.run, daemon=True); s.th.start()
        if not s.ready.wait(3.0): s.close(); raise Abort('bridge gave no READY with options %s' % ' '.join(opts))

    def run(s):
        for a in iter(s.p.stdout.readline, b''):
            a = a.decode().strip()
            if a == 'READY': s.ready.set()
            elif a.startswith('S '):
                f = a.split(); s.state = tuple(int(v) for v in f[1:6])
            elif a.startswith('R '): s.ans.append(a)

    def send(s, text):
        try: s.p.stdin.write((text + '\n').encode())
        except Exception: pass

    def close(s):
        try: s.p.stdin.close()
        except Exception: pass
        try: s.p.wait(3)
        except Exception:
            s.p.terminate()
            try: s.p.wait(3)
            except Exception: s.p.kill()               # v27: a bridge that ignores SIGTERM must not break the shutdown (the caller sends STOP itself)
        try: s.p.wait(2)                               # v27b: no child is left behind unreaped
        except Exception: pass


FIELDS = 'step name phase t fid ns sx sy resp bad pwx pwy vis ximg yimg ex ey X Y SPD auto lost moving proc_ms'.split()


class Session:
    def __init__(s, a):
        os.makedirs(a.out, exist_ok=True); s.a = a
        s.slog = open(os.path.join(a.out, 'serial.jsonl'), 'w'); s.elog = open(os.path.join(a.out, 'bridge_err.log'), 'w')
        s.tap = Tap(a.dry, s.slog)
        s.cam = FakeCam(s.tap, a.simK, a.simL, vmin=a.simMin, curve=a.curve, tau=a.simTau) if a.dry else Cam(open(os.path.join(a.out, 'cam_pipe.log'), 'w'))   # v27b: + curve, tau
        s.est = Shift(); s.last = 0; s.f = open(os.path.join(a.out, 'frames.csv'), 'w', newline=''); s.w = csv.writer(s.f); s.w.writerow(FIELDS)
        s.step_i = 0; s.name = ''; s.br = None; s.lastp = 0.0; s.S = np.zeros(2)
        s.t_begin = time.monotonic(); s.tap.deadline = s.t_begin + a.max_seconds + 3.0; s.same = 0; s.prev_m = None; s.hy = False; s.why = ''; s.meas_t = None; s.proc = 0.0; s.hook = None   # v27
        s.nsame0 = 0; s.bw, s.bh = BW, BH; s.nerrlim = 0   # v27b: identical-measurement streaks while the command was zero (counted only); the virtual person's box of the current step; blocks ended as "person lost"
        try:                                           # v27: a start without a camera picture must also close the camera program and send STOP
            fr = s.cam.get(0, 90.0)
            if fr is None: raise Abort('no camera frame in 90 s')
            cv2.imwrite(os.path.join(a.out, 'first_frame.png'), cv2.normalize(fr[2], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        except BaseException as e:                     # v27
            print('START FAILED: %r' % (e,), flush=True); Session.close(s); raise

    def frame(s):
        if s.hook: s.hook()                            # v27: forced faults, dry mode only (set by lead_hil.py)
        dead = not s.tap.th.is_alive()                 # v27c (second review): without the tap thread no bridge command (also no bridge STOP) reaches the gimbal and there is no watchdog: handled like a dead bridge program
        if dead: s.tap.estop()                         # v27c: written by this thread, straight to the device
        if s.tap.stop_failed: raise Abort('EMERGENCY STOP NOT WRITTEN to the gimbal device: the run ends, the gimbal may still be moving')   # v27b
        if dead: raise Abort('the serial tap thread has ended (no watchdog, no command reaches the gimbal): gimbal stopped first, the run ends')   # v27c
        if s.tap.trip: raise Abort('watchdog: ' + s.tap.trip + ' (STOP already sent)')   # v27
        if s.tap.ykick: k = s.tap.ykick; s.tap.ykick = 0; raise ErrLimit('person lost: up-down guard: a command with SPD %d moved the up-down axis while following (limit %d; gimbal stopped first)' % (k, GUARD_Y))   # v27c (second review): the tap thread has already written STOP and cut the line; the block ends like one in which the person was lost
        if s.br is not None and s.br.p.poll() is not None: s.tap.estop(); raise Abort('bridge program ended by itself (exit code %s): gimbal stopped first' % s.br.p.returncode)   # v27b (review): a dead bridge sends no STOP; its last velocity command would stay active
        if time.monotonic() - s.t_begin > s.a.max_seconds: s.tap.estop(); raise Abort('total run time limit of %d s' % s.a.max_seconds)   # v27
        fr = s.cam.get(s.last, STALE_FAST_S); lim = STALE_FAST_S   # v27: was 2.0 s   # v27c: first the short limit, which is the whole limit while the live command is above FAST_SPD
        if fr is None and not s.tap.fast(): fr = s.cam.get(s.last, STALE_S - STALE_FAST_S); lim = STALE_S   # v27d: not s.tap.fast() (was s.tap.last[2] <= FAST_SPD)   # v27c: at lower commands the rest of STALE_S, as before
        if fr is None:                                 # v27: no new picture for STALE_S: STOP first, then see whether the camera is alive at all
            s.tap.estop(); fr = s.cam.get(s.last, 2.0 - lim)   # v27c: lim (was STALE_S)
            if fr is None: raise Abort('camera silent for 2 s')
            s.last = fr[0]; raise Abort('image shift lost: no new picture for %.2f s (gimbal stopped first)' % lim)   # v27   # v27c: lim
        s.tap.pet = time.monotonic()                   # v27: tells the watchdog that the main loop is alive
        age = (time.monotonic_ns() - fr[1]) / 1e9      # v27
        lim = STALE_FAST_S if s.tap.fast() else STALE_S   # v27c   # v27d: s.tap.fast()
        if age > lim: s.tap.estop(); s.last = fr[0]; raise Abort('image shift lost: picture %.2f s old (limit %.2f s; gimbal stopped first)' % (age, lim))   # v27   # v27c: lim (was STALE_S)
        t0 = time.monotonic(); s.last = fr[0]; S, resp = s.est.update(fr[2]); s.S = S.copy(); s.proc = (time.monotonic() - t0) * 1e3
        m = (float(S[0]), float(S[1]), float(resp)); s.same = s.same + 1 if m == s.prev_m else 0; s.prev_m = m   # v27: frozen picture = the measurement repeats exactly
        if s.same >= SAME_N - 1:                       # v27b: only while a velocity command is active this ends the block; with the command at zero it is counted (the streak goes on, so a command that starts on a still frozen picture is stopped at the next frame)
            if s.tap.last[2] != 0: s.tap.estop(); s.same = 0; s.prev_m = None; raise Abort('image shift lost: picture frozen (%d identical measurements in a row, gimbal stopped first)' % SAME_N)   # v27
            if s.same == SAME_N - 1: s.nsame0 += 1     # v27b
        if s.est.bad >= BAD_MOVING_N and s.tap.last[2] != 0: s.tap.estop(); raise Abort('image shift lost: image matching failed for %d frames while a velocity command was active (gimbal stopped first)' % s.est.bad)   # v27b (review): was 12 frames (about 0.5 s) with the command still active
        if s.est.bad > 12: s.tap.estop(); raise Abort('image shift lost (correlation below %.2f for 12 frames)' % RESP_MIN)   # v27b: + STOP first here too
        if abs(S[0]) > s.a.limx or abs(S[1]) > s.a.limy: s.tap.estop(); raise Abort('excursion limit: scene displacement %.0f, %.0f px (limit %.0f, %.0f)' % (S[0], S[1], s.a.limx, s.a.limy))   # v27: limits from the command line, STOP at once
        if s.est.bad == 0: s.meas_t = time.monotonic() # v27: time of the last good measurement (for end_state.json)
        if s.br is not None and time.monotonic() - s.lastp > 0.2: s.br.send('P'); s.lastp = time.monotonic()
        return fr[0], fr[1], S.copy(), resp

    def row(s, phase, t, fid, ns, S, resp, pw=None, vis=0, img=None, err=None):
        X, Y, SPD = s.tap.last; st = s.br.state if s.br is not None else (0, 0, 0, 0, 0)
        s.w.writerow([s.step_i, s.name, phase, '%.4f' % t, fid, ns, '%.2f' % S[0], '%.2f' % S[1], '%.3f' % resp, s.est.bad,
                      *(('%.2f' % pw[0], '%.2f' % pw[1]) if pw is not None else ('', '')), vis,
                      *(('%.2f' % img[0], '%.2f' % img[1]) if img is not None else ('', '')),
                      *(('%.4f' % err[0], '%.4f' % err[1]) if err is not None else ('', '')), X, Y, SPD, st[0], st[1], st[2], '%.1f' % s.proc])

    def open_bridge(s, opts, key='v25'):             # v27: + which bridge program
        s.elog.write('--- step %d %s bridge %s guard %d opts %s\n' % (s.step_i, s.name, key, s.tap.guard, ' '.join(BASE + opts))); s.elog.flush()   # v27: the exact options of every bridge start, for the record   # v27b: + guard
        s.br = Bridge(s.tap, BASE + opts, s.elog, key); s.tap.child = s.br.p; s.br.send('P'); s.lastp = time.monotonic()   # v27: + key   # v27b: the tap thread watches this process

    def close_bridge(s):
        s.tap.child = None                             # v27b: from here on the end of the bridge is wanted
        if s.br is not None: s.br.send('H'); time.sleep(0.1); s.br.close(); s.br = None
        if s.tap.last[2]: s.tap.estop(); s.tap.open_line()   # v27c (second review): the STOP at the end of a block is the one the bridge sends on 'H'. A bridge that died at that very moment sent none and nothing watches it any more: STOP straight to the device (the line opens again only when that STOP was written and no watchdog stop stands)

    def box(s, xi, yi):
        """v27b: the virtual person's box (x1, y1, x2, y2) for the aim point xi, yi, cut at the picture's edge as a detector's box is. None: nothing left inside."""
        x1, y1 = xi - s.bw / 2, yi - AIM * s.bh; x2, y2 = x1 + s.bw, y1 + s.bh
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
        return (x1, y1, x2, y2) if x2 - x1 >= 2 and y2 - y1 >= 2 else None

    def feed(s, fid, ns, pw, vis):
        """virtual person at world position pw (aim point). Returns image position and the controller's normalized error"""
        xi, yi = W / 2 + pw[0] + s.S[0], H / 2 + pw[1] + s.S[1]
        b = s.box(xi, yi) if vis and 0 <= xi < W and 0 <= yi < H else None; inside = b is not None   # v27b: box size per step, cut at the picture's edge
        lat = ns + int(s.a.latency_ms * 1e6) - time.monotonic_ns()
        if lat > 0: time.sleep(lat / 1e9)
        if inside: s.br.send('F %d %d 1 %d %.2f %.2f %.2f %.2f 0.900' % (fid, ns, TID, b[0], b[1], b[2] - b[0], b[3] - b[1]))   # v27b
        else: s.br.send('F %d %d 0' % (fid, ns))
        return (xi, yi), ((xi - W / 2) / (W / 2), (yi - H / 2) / (H / 2)), int(bool(inside))

    def static(s, st):
        s.name = 'static'; t0 = time.monotonic()
        while time.monotonic() - t0 < st.get('T', 3):
            if st.get('blind'):                          # camera warm-up check: no shift measurement, one picture every 3 s
                fr = s.cam.get(s.last, 2.0); s.last = fr[0]; k = int((time.monotonic() - t0) / 3)
                if k != getattr(s, '_k', -1): s._k = k; cv2.imwrite(os.path.join(s.a.out, 'warm_%02d.png' % (k * 3)), cv2.normalize(fr[2], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)); print('  t %d raw percentiles %s' % (k * 3, np.percentile(fr[2], [1, 50, 99]).round().tolist()), flush=True)
                continue
            fid, ns, S, resp = s.frame(); s.row('static', time.monotonic() - t0, fid, ns, S, resp)
        cv2.imwrite(os.path.join(s.a.out, 'static_end_%d.png' % s.step_i), cv2.normalize(s.cam.cur[2], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    def ident(s, st):
        ax = st.get('axis', 'x'); sp = int(st['speed']); T = float(st.get('T', 1.5)); mp = float(st.get('maxpx', 120)); k = 0 if ax == 'x' else 1
        s.tap.guard_y = None                           # v27c: no up-down guard in an identification step (its own guard is the step's speed)
        s.name = 'ident_%s_%d' % (ax, sp); s.tap.guard = min(sp, s.a.spd_abort); s.open_bridge(['--kp', '30.0'] + ['--manual-speed', str(sp)]); t0 = time.monotonic()   # v27b: guard = this step's own speed
        def hold(phase, cmd, dur, stop_px=None):
            if cmd: s.br.send(cmd)
            a = time.monotonic(); S0 = s.S[k]
            while time.monotonic() - a < dur:
                fid, ns, S, resp = s.frame(); s.row(phase, time.monotonic() - t0, fid, ns, S, resp)
                if stop_px is not None and abs(S[k] - S0) > stop_px: break
            return time.monotonic() - a
        hold('rest', None, 0.5)
        S0 = s.S[k]; d = hold('fwd', 'V 1 0' if ax == 'x' else 'V 0 1', T, mp); hold('coast', 'V 0 0', 1.2)
        if sp >= 5 and abs(s.S[k] - S0) < 3: raise Abort('the image did not move with speed %d on %s: servo power off?' % (sp, ax))
        hold('back', 'V -1 0' if ax == 'x' else 'V 0 -1', d); hold('coast2', 'V 0 0', 1.2)
        s.close_bridge()

    def track(s, st, home=False):
        s.name = st['name']; s.hy = False; s.bw, s.bh = [float(v) for v in st.get('box', (BW, BH))]; s.tap.guard = HOME_GUARD if home else min(FOLLOW_GUARD, s.a.spd_abort, int(st.get('spd_guard', s.a.spd_abort)))   # v27b: box and speed guard of this step   # v27c: never above FOLLOW_GUARD in a following block
        s.tap.guard_y = None if home else GUARD_Y      # v27c (second review): up-down guard in following blocks only (the home step's bridge runs with --speed-max 10 anyway)
        s.open_bridge(st['opts'] + st.get('extra', []), st.get('bridge', 'v25')); wp = st.get('wp', [[0, 0, 0]]); drop = st.get('drop', []); dur = float(st.get('dur', wp[-1][0]))   # v27: bridge + extra options per step
        base = np.zeros(2) if home else -s.S.copy()          # home: the start pose itself; else: wherever the gimbal looks now
        t0 = time.monotonic(); sel = False; n = 0; okt = None
        while True:                                          # acquire: target at the image centre, select it
            fid, ns, S, resp = s.frame(); n += 1; img, err, vis = s.feed(fid, ns, base if not home else base, 1); s.row('acq', time.monotonic() - t0, fid, ns, S, resp, base, vis, img, err)
            if n == 4: s.br.send('A %d' % TID)
            if s.br.ans:
                a = s.br.ans.popleft()
                if a == 'R A OK': sel = True; break
                if n > 30: raise Abort('bridge refused the selection: ' + a)
                s.br.send('A %d' % TID)
            if time.monotonic() - t0 > 4: raise Abort('no answer to the selection')
        limx = (abs(s.S[0]) + 60) if home else max(abs(p[1]) for p in wp) + 90; limy = (abs(s.S[1]) + 60) if home else max(abs(p[2]) for p in wp) + 70
        if not home and st.get('errlim'): limx, limy = min(float(st['errlim'][0]), 240.0), min(float(st['errlim'][1]), 70.0)   # v27b: a block may name its error limit (never above 240 / 70 px, this morning's values for a walk of +-150 px). The rule "walk range + 90 px" would make short fast walks end as "person lost" with the person still well inside the picture
        t0 = time.monotonic()
        while True:
            fid, ns, S, resp = s.frame(); t = time.monotonic() - t0
            if home:
                if abs(S[1]) < HOME_DBY: s.hy = True       # v27: vertical hold band of the home step (with hysteresis)
                elif abs(S[1]) > HOME_DBY_OUT: s.hy = False   # v27
                pw = np.array([0.0, -S[1]]) if s.hy else base; vis = 1   # v27: inside the band the target is put at the picture's own height = no vertical command asked for
                if abs(S[0]) < HOME_X and s.hy and s.tap.last[2] == 0:   # v27: was abs(S[1]) < 11   # v27b: HOME_X (was 14)
                    okt = okt or t
                    if t - okt > 0.6: break
                else: okt = None
            else:
                x = np.interp(t, [p[0] for p in wp], [p[1] for p in wp]); y = np.interp(t, [p[0] for p in wp], [p[2] for p in wp])
                pw = base + np.array([x, y]); vis = 0 if any(a <= t < b for a, b in drop) else 1
            img, err, vis = s.feed(fid, ns, pw, vis); s.row('home' if home else 'run', t, fid, ns, S, resp, pw, vis, img, err)
            if abs(err[0]) * 320 > limx or abs(err[1]) * 240 > limy:
                s.tap.estop()                          # v27b: STOP first, in both cases
                msg = 'error grew to %.0f, %.0f px (limit %.0f, %.0f)' % (err[0] * 320, err[1] * 240, limx, limy)
                if home: raise Abort(msg + ': wrong direction or runaway')
                raise ErrLimit('person lost: ' + msg)  # v27b: a tracking block ends as "person lost"; the caller decides whether the run goes on (lead_hil.py: at most MAX_ERRLIM blocks per run)
            if t > dur: break
        if s.tap.ykick: k = s.tap.ykick; s.tap.ykick = 0; raise ErrLimit('person lost: up-down guard: a command with SPD %d moved the up-down axis while following (limit %d; gimbal stopped first)' % (k, GUARD_Y))   # v27c: the guard tripped after the last picture of this block: the block still counts as lost
        s.close_bridge()

    def peek(s, st):
        """blind look-around when the image has no texture: timed manual move, a frame saved before, after, and after the return. No shift measurement."""
        ax = st.get('axis', 'x'); sp = int(st.get('speed', 10)); T = float(st.get('T', 2.0)); s.name = 'peek_%s' % ax; s.tap.guard = min(sp, s.a.spd_abort); s.tap.guard_y = None   # v27b: guard   # v27c: no up-down guard here
        s.open_bridge(['--kp', '30.0', '--manual-speed', str(sp)])
        def hold(cmd, dur, tag):
            if cmd: s.br.send(cmd)
            a = time.monotonic(); fr = None
            while time.monotonic() - a < dur:
                fr = s.cam.get(s.last, 2.0)
                if fr is None: raise Abort('camera silent for 2 s')
                s.last = fr[0]; s.tap.pet = time.monotonic()   # v27: watchdog
                if time.monotonic() - s.lastp > 0.2: s.br.send('P'); s.lastp = time.monotonic()
            r = fr[2]; q = np.percentile(r, [0, 1, 50, 99, 100]); print('  peek %s %s dtype %s raw percentiles %s' % (ax, tag, r.dtype, np.round(q, 1).tolist()), flush=True)
            cv2.imwrite(os.path.join(s.a.out, 'peek_%d_%s.png' % (s.step_i, tag)), cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        try:
            hold(None, 0.5, 'a_before'); hold('V 1 0' if ax == 'x' else 'V 0 1', T, 'b_moving'); hold('V 0 0', 0.8, 'c_moved')
            hold('V -1 0' if ax == 'x' else 'V 0 -1', T, 'd_returning'); hold('V 0 0', 0.8, 'e_back')
        finally: s.close_bridge()

    def home(s):
        if abs(s.S[0]) < HOME_X and abs(s.S[1]) < HOME_DBY: return   # v27: was 11   # v27b: HOME_X (was 14)
        s.track(dict(name='home', opts=HOME_OPTS, dur=14), home=True)   # always gimbal_bridge_v25 with the plain P controller and speed limit 10, for both programs under test

    def run(s, plan):
        for i, st in enumerate(plan):
            s.step_i = i; print('STEP %d %s' % (i, json.dumps(st)[:160]), flush=True)
            if st['type'] == 'static': s.static(st)
            elif st['type'] == 'ident': s.ident(st); s.home()
            elif st['type'] == 'track':
                s.track(st)
                if not st.get('nohome'): s.home()
            elif st['type'] == 'peek': s.peek(st)
            elif st['type'] == 'pulse': raise Abort('pulse steps (stage 1) need lead_hil.py')   # v27b
            s.f.flush(); s.slog.flush(); print('  end: scene displacement %.1f, %.1f px, serial commands %d' % (s.S[0], s.S[1], s.tap.n), flush=True)

    def close(s):
        s.tap.estop()                                  # v27: STOP straight to the device and the line cut BEFORE anything else (was after closing the bridge)
        try: s.close_bridge()
        except Exception: pass
        s.tap.estop(); rc = s.cam.close(); s.tap.close(); s.f.close(); s.slog.close(); s.elog.close(); return rc


def other_servers():
    """v27b: every process that looks like the live server by a LOOSE match (either file name anywhere in its command line), with its command line. The test must not start while one is alive."""
    out = []
    for p in sorted(glob.glob('/proc/[0-9]*')):
        try: c = open(p + '/cmdline', 'rb').read().replace(b'\0', b' ').decode('latin1').strip()
        except OSError: continue
        if int(os.path.basename(p)) == os.getpid(): continue
        f = c.split()
        if any(os.path.basename(w) in ('start_tracking_server.py', 'lwir_tracker_m1.py') for w in f[:6]) and f and ('python' in os.path.basename(f[0])): out.append('%s: %s' % (os.path.basename(p), c[:100]))
    return out


def main():
    global DRY_BRIDGE, SPD_ABORT                       # v27
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); ap.add_argument('--out', required=True); ap.add_argument('--dry', action='store_true')
    ap.add_argument('--latency-ms', type=float, default=40.0, help='time from camera frame to the box reaching the bridge (stands for detector + tracker)')
    ap.add_argument('--simK', type=float, default=6.0); ap.add_argument('--simL', type=float, default=0.08)
    ap.add_argument('--simMin', type=float, default=0.0, help='v27: dry mode: lowest speed of the pretend gimbal in px/s (0 = none, as before)')   # v27
    ap.add_argument('--simCurve', default='', help='v27b: dry mode: speed of the pretend gimbal as a table "SPD:px/s,SPD:px/s,..." (replaces --simK / --simMin)')   # v27b
    ap.add_argument('--simTau', type=float, default=0.0, help='v27b: dry mode: time constant of the pretend gimbal speeding up and slowing down (s)')   # v27b
    ap.add_argument('--limx', type=float, default=LIMX, help='v27: the run ends when the view is this many px left or right of the start view')   # v27
    ap.add_argument('--limy', type=float, default=LIMY, help='v27: the same, up or down')   # v27
    ap.add_argument('--spd-abort', type=int, default=SPD_ABORT, help='v27: the run ends when a speed command above this reaches the gimbal line')   # v27
    ap.add_argument('--max-seconds', type=float, default=1080.0, help='v27: the run ends itself after this time, whatever it is doing')   # v27
    ap.add_argument('--dry-bridge', default=None, help='v27: dry mode on a PC only: stand-in script used instead of the real bridge programs'); a = ap.parse_args()   # v27
    if a.dry_bridge and not a.dry: sys.exit('--dry-bridge is for --dry only')   # v27
    if pty is None and not (a.dry and a.dry_bridge): sys.exit('no pseudo terminal on this system: only --dry with --dry-bridge works here')   # v27
    DRY_BRIDGE = a.dry_bridge; SPD_ABORT = a.spd_abort   # v27
    a.curve = [(float(p.split(':')[0]), float(p.split(':')[1])) for p in a.simCurve.split(',') if p] or None   # v27b
    if a.limx > LIMX or a.limy > LIMY: sys.exit('--limx / --limy can only be made smaller than %.0f / %.0f' % (LIMX, LIMY))   # v27b: the view limits are the same for every stage
    if not a.dry:
        o = other_servers()                            # v27b (review): any server-like process, however it was started and from whatever folder
        if o or subprocess.run(['pgrep', '-f', r'start_tracking_server\.py|lwir_tracker_m1\.py'], stdout=subprocess.DEVNULL).returncode == 0: sys.exit('live server is running: stop it first %s' % o)   # v27: new server name (the old one too)
        if subprocess.run(['fuser', DEV], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0: sys.exit('gimbal device busy')
        for p in [CAMP] + sorted(BRIDGES.values()):    # v27: every program of the plan must exist before anything starts
            if not os.access(p, os.X_OK): sys.exit('missing program: ' + p)
    plan = json.load(open(a.plan)); s = None; code = 0; why = 'PLAN DONE'   # v27: + why
    os.makedirs(a.out, exist_ok=True); json.dump(plan, open(os.path.join(a.out, 'plan_used.json'), 'w'))   # v27b: the plan that was run stays with its record (the analysis reads the candidates from it)
    for st in plan:                                    # v27: a wrong bridge name must end the run before the gimbal moves, not in the middle
        if st.get('type') == 'track' and st.get('bridge', 'v25') not in BRIDGES: sys.exit('unknown bridge "%s" in step %s' % (st.get('bridge'), st.get('name')))
        if st.get('type') == 'meta' and st.get('assumed') and not a.dry: sys.exit('this plan was made from ASSUMED numbers (for dry checks only): it must not run on the real gimbal')   # v27b
        if st.get('type') in ('ident', 'peek', 'pulse') and int(st.get('speed', 0)) > a.spd_abort: sys.exit('step with speed %s above --spd-abort %d' % (st.get('speed'), a.spd_abort))   # v27b
    def on_signal(n, _):                               # v27: was a SIGTERM-only lambda. STOP is written inside the handler itself, then the program is ended
        SIG.append(n)
        for t in TAPS: t.estop()
        raise KeyboardInterrupt('signal %d' % n)
    for n in ('SIGTERM', 'SIGHUP', 'SIGQUIT', 'SIGBREAK', 'SIGINT'):   # v27: a closed terminal (SIGHUP) or Ctrl-C must also end with STOP
        if hasattr(signal, n): signal.signal(getattr(signal, n), on_signal)
    try:
        s = Session(a); s.run(plan); print('PLAN DONE', flush=True)
    except Abort as e: why = 'ABORT: %s' % e; print(why, flush=True); code = 3   # v27: + why
    except KeyboardInterrupt as e: why = 'INTERRUPTED %s' % e; print(why, flush=True); code = 4   # v27: + why
    except BaseException as e:                         # v27: any other error: recorded, STOP in finally, then the traceback (a signal that surfaced as another error counts as the signal)
        why = ('INTERRUPTED signal %d (surfaced as %r)' % (SIG[0], e)) if SIG else 'ERROR: %r' % (e,); print(why, flush=True); code = 4 if SIG else 5
        if not SIG: raise
    finally:
        if s is not None:
            s.why = why                                # v27: for end_state.json
            for k in range(3):                         # v27: a second signal during the shutdown must not skip the STOP
                try: print('camera exit code %s, device answer bytes %d' % (s.close(), s.tap.fb), flush=True); break
                except KeyboardInterrupt: s.tap.estop(); print('signal during the shutdown: STOP sent again, closing again', flush=True)
                except Exception as e: s.tap.estop(); print('error during the shutdown (%r): STOP sent again' % (e,), flush=True); break
            print('EXIT %s | stops passed to the device line %d, highest SPD %d | emergency stops not written %d, device opened again %d, identical measurements while the command was zero %d, blocks ended as person lost %d' % (
                why, s.tap.nstop, s.tap.maxspd, s.tap.nfail, s.tap.nreopen, s.nsame0, s.nerrlim), flush=True)   # v27   # v27b: + the four counts
            if s.tap.stop_failed: print('WARNING: THE LAST EMERGENCY STOP WAS NOT WRITTEN TO THE GIMBAL DEVICE. Go to the gimbal.', flush=True); code = 7   # v27b
            if getattr(s, 'moving_after_stop', None): print('WARNING: THE VIEW IS STILL MOVING AFTER THE LAST STOP (two views %.0f px apart, more than 0.8 s after the STOP). STOP was written again. Go to the gimbal.' % s.moving_after_stop, flush=True); code = 7   # v27c (second review): the STOP was written but the gimbal did not obey it
            if getattr(s, 'end_view_checked', True) is False: print('WARNING: END VIEW NOT CHECKED - LOOK AT THE GIMBAL (the view after the last STOP was never matched to the start view).', flush=True); code = 7   # v27d (third review, minor 1)
    sys.exit(code)


if __name__ == '__main__': main()
