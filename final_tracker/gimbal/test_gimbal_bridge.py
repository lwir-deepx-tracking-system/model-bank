#!/usr/bin/env python3
"""Tests gimbal_bridge without a gimbal: its device is a pseudo terminal, so every command the team's code would send to the gimbal is read here and nothing moves. Needs only the Python standard library.
Usage (from the folder that holds gimbal_bridge):  python3 test_gimbal_bridge.py [path of gimbal_bridge]
Each case starts a new bridge process."""
import json, os, pty, select, signal, subprocess, sys, termios, time, tty
from pathlib import Path

BRIDGE = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / 'gimbal_bridge')
STOP = {'T': 135}


class Rig:
    def __init__(self, *options):
        self.master, slave = pty.openpty()
        tty.setraw(self.master)
        self.name = os.ttyname(slave)
        self.slave = slave     # stays open so the bridge never sees a hang-up
        self.proc = subprocess.Popen([BRIDGE, '--device', self.name, '--heartbeat-ms', '600', *options], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.buf = b''; self.out = b''; self.frame_id = 0
        assert self.line(2.0) == 'READY', 'no READY'

    def send(self, text):
        self.proc.stdin.write((text + '\n').encode())

    def line(self, timeout=1.0):
        """next stdout line of the bridge, None on timeout"""
        end = time.monotonic() + timeout
        while b'\n' not in self.out:
            left = end - time.monotonic()
            if left <= 0: return None
            r, _, _ = select.select([self.proc.stdout], [], [], left)
            if not r: return None
            d = os.read(self.proc.stdout.fileno(), 4096)
            if not d: return None
            self.out += d
        a, self.out = self.out.split(b'\n', 1)
        return a.decode()

    def wait_line(self, prefix, timeout=1.5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            a = self.line(max(end - time.monotonic(), .01))
            if a is not None and a.startswith(prefix): return a
        raise AssertionError('no line starting %r' % prefix)

    def wait_state(self, want, timeout=1.5):
        """want: dict of fields of the S line"""
        end = time.monotonic() + timeout
        last = None
        while time.monotonic() < end:
            a = self.line(max(end - time.monotonic(), .01))
            if a is None or not a.startswith('S '): continue
            f = a.split(); last = dict(auto=int(f[1]), lost=int(f[2]), moving=int(f[3]), error=int(f[4]), id=int(f[5]))
            if all(last[k] == v for k, v in want.items()): return last
        raise AssertionError('state %s never seen, last %s' % (want, last))

    def serial(self, timeout=.5):
        """next JSON command on the serial line, None on timeout"""
        end = time.monotonic() + timeout
        while b'\n' not in self.buf:
            left = end - time.monotonic()
            if left <= 0: return None
            r, _, _ = select.select([self.master], [], [], left)
            if not r: return None
            self.buf += os.read(self.master, 4096)
        a, self.buf = self.buf.split(b'\n', 1)
        return json.loads(a.decode().strip())

    def until_stop(self, timeout=.6):
        """reads commands up to the next STOP. Speed updates already on the line before it are fine; anything else is not, and a STOP must come in time"""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            c = self.serial(max(end - time.monotonic(), .01))
            if c == STOP: return True
            assert c is None or c['T'] == 141, c
        return False

    def frame(self, *people, age_ms=0):
        """people: (id, center x, center y) in camera pixels, box 40x80"""
        self.frame_id += 1
        ps = ' '.join('%d %.1f %.1f 40 80 0.900' % (i, x - 20, y - 40) for i, x, y in people)
        self.send('F %d %d %d %s' % (self.frame_id, time.monotonic_ns() - age_ms * 1000000, len(people), ps))

    def feed(self, seconds, *people, ping=True):
        """frames at about 30 fps with heartbeats, for this many seconds; returns the serial commands seen"""
        got = []; end = time.monotonic() + seconds; next_ping = 0.
        while time.monotonic() < end:
            now = time.monotonic()
            if ping and now >= next_ping: self.send('P'); next_ping = now + .2
            self.frame(*people)
            c = self.serial(.033)
            if c is not None: got.append(c)
        return got

    def close(self):
        try: self.proc.stdin.close()
        except OSError: pass
        try: self.proc.wait(3)
        except subprocess.TimeoutExpired: self.proc.kill()
        os.close(self.master); os.close(self.slave)


def case(fn):
    r = Rig(*getattr(fn, 'options', ()))
    try:
        assert r.serial(1.0) == STOP, 'the first command must be STOP'
        fn(r)
    finally:
        r.close()
    print('ok  ', fn.__name__, '-', fn.__doc__, flush=True)


def no_move_without_a_click(r):
    """frames and heartbeats alone never move the gimbal"""
    got = r.feed(.6, (7, 500, 100))
    assert not got, got


def click_needs_a_viewer(r):
    """without a heartbeat a selection is refused"""
    r.frame((7, 500, 100))
    r.send('A 7'); assert r.wait_line('R A') == 'R A NOT_READY'
    assert r.serial(.2) is None


def unknown_id_refused(r):
    """an ID that is not on screen is refused, nothing moves"""
    r.feed(.1, (7, 500, 100))
    r.send('A 99'); assert r.wait_line('R A') == 'R A INVALID'
    r.send('A 0'); assert r.wait_line('R A') == 'R A INVALID'
    assert not r.feed(.2, (7, 500, 100))


def follows_right_and_up(r):
    """target right of and above the center: X=+1, Y=+1, speed starts at the minimum and never passes the limit"""
    r.feed(.1, (7, 600, 60))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    got = r.feed(.8, (7, 600, 60))
    moves = [c for c in got if c['T'] == 141]
    assert moves and all(c['X'] == 1 and c['Y'] == 1 for c in moves), got
    assert moves[0]['SPD'] == 1 and all(1 <= c['SPD'] <= 30 for c in moves), moves
    assert [c['SPD'] for c in moves] == sorted(c['SPD'] for c in moves), moves


def follows_left_and_down(r):
    """target left of and below the center: X=-1, Y=-1"""
    r.feed(.1, (7, 40, 440))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    moves = [c for c in r.feed(.3, (7, 40, 440)) if c['T'] == 141]
    assert moves and all(c['X'] == -1 and c['Y'] == -1 for c in moves), moves


def axis_signs(r):
    """--x-sign -1 flips the pan command only"""
    r.feed(.1, (7, 600, 60))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    moves = [c for c in r.feed(.3, (7, 600, 60)) if c['T'] == 141]
    assert moves and all(c['X'] == -1 and c['Y'] == 1 for c in moves), moves
axis_signs.options = ('--x-sign', '-1')


def centered_stops(r):
    """target reaches the center: STOP"""
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    assert any(c['T'] == 141 for c in r.feed(.2, (7, 600, 240)))
    assert STOP in r.feed(.2, (7, 322, 240))
    assert not r.feed(.2, (7, 322, 240))
    r.wait_state(dict(auto=1, moving=0, id=7))


def lost_stops_and_drops(r):
    """target leaves: STOP at once, a return within the wait resumes, after the wait the selection is gone and a late return does not resume"""
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    assert any(c['T'] == 141 for c in r.feed(.2, (7, 600, 240)))
    got = r.feed(.2, (8, 100, 100))
    assert STOP in got, got
    r.wait_state(dict(auto=1, lost=1, id=7))
    assert any(c['T'] == 141 for c in r.feed(.2, (7, 600, 240))), 'no resume'
    assert STOP in r.feed(.2, (8, 100, 100))
    got = r.feed(1.2, (8, 100, 100))
    assert not [c for c in got if c['T'] == 141], got
    assert not r.feed(.4, (7, 600, 240)), 'a late return must not resume'


def frames_stop_stops(r):
    """the tracker stalls while following: STOP without any new frame"""
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    assert any(c['T'] == 141 for c in r.feed(.2, (7, 600, 240)))
    r.send('P')
    assert r.until_stop()


def old_frame_not_used(r):
    """a frame older than the input limit cannot start following"""
    r.send('P'); r.frame((7, 600, 240), age_ms=400)
    r.send('A 7'); assert r.wait_line('R A') == 'R A INVALID'
    assert r.serial(.2) is None


def viewer_gone_stops(r):
    """the page closes (no heartbeat): STOP and the selection is dropped, new heartbeats do not bring it back"""
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    assert any(c['T'] == 141 for c in r.feed(.2, (7, 600, 240)))
    got = r.feed(1.0, (7, 600, 240), ping=False)
    assert STOP in got, got
    assert not r.feed(.3, (7, 600, 240)), 'no return after the viewer came back'


def manual_keys(r):
    """arrow key: following is cancelled, STOP, manual move at the manual speed; key released: STOP"""
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    assert any(c['T'] == 141 for c in r.feed(.2, (7, 600, 240)))
    r.send('P'); r.send('V 0 1')
    assert r.until_stop()
    assert r.serial(.5) == {'T': 141, 'X': 0, 'Y': 1, 'SPD': 30}
    assert not r.feed(.3, (7, 600, 240)), 'new frames must not revive following'
    r.send('V 0 0')
    assert r.serial(.5) == STOP


def halt_stops(r):
    """the stop button: STOP and the selection is dropped"""
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    assert any(c['T'] == 141 for c in r.feed(.2, (7, 600, 240)))
    r.send('H')
    assert r.until_stop()
    r.wait_state(dict(auto=0, moving=0, id=0))
    assert not r.feed(.3, (7, 600, 240))


def input_closed_stops(r):
    """the tracker ends: STOP, the bridge exits by itself"""
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    assert any(c['T'] == 141 for c in r.feed(.2, (7, 600, 240)))
    r.proc.stdin.close()
    assert r.until_stop()
    assert r.proc.wait(3) == 0


def sigterm_stops(r):
    """SIGTERM: STOP, then exit"""
    r.feed(.1, (7, 600, 240))
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'
    assert any(c['T'] == 141 for c in r.feed(.2, (7, 600, 240)))
    r.proc.send_signal(signal.SIGTERM)
    assert r.until_stop()
    assert r.proc.wait(3) == 0


def damaged_lines_ignored(r):
    """damaged input is dropped, nothing moves, and the bridge keeps working"""
    for bad in ('F', 'F x y z', 'F 1 2 3 4', 'A', 'A x', 'V', 'V a b', 'Q 1 2', 'F 5 0 1 7 1 2 3', ''):
        r.send(bad)
    assert r.wait_line('R A') == 'R A INVALID' and r.wait_line('R A') == 'R A INVALID'   # the two damaged A lines
    r.feed(.1, (7, 600, 240))
    assert r.serial(.1) is None
    r.send('A 7'); assert r.wait_line('R A') == 'R A OK'


def device_missing():
    """the device does not exist: the bridge ends with an error instead of hanging"""
    p = subprocess.run([BRIDGE, '--device', '/dev/does-not-exist'], capture_output=True, timeout=5)
    assert p.returncode == 2 and b'does-not-exist' in p.stderr, p
    print('ok   device_missing -', device_missing.__doc__, flush=True)


if __name__ == '__main__':
    for f in (no_move_without_a_click, click_needs_a_viewer, unknown_id_refused, follows_right_and_up, follows_left_and_down, axis_signs, centered_stops, lost_stops_and_drops, frames_stop_stops, old_frame_not_used,
              viewer_gone_stops, manual_keys, halt_stops, input_closed_stops, sigterm_stops, damaged_lines_ignored):
        case(f)
    device_missing()
    print('ALL OK')
