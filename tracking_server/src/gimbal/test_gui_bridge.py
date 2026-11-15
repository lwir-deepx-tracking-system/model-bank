#!/usr/bin/env python3
"""Tests the PC GUI side of gimbal_bridge without a gimbal and without the PC: the gimbal device is a pseudo terminal (nothing moves), and this
script plays the PC: it receives the video RTP packets (UDP), decodes the track metadata datagrams (UDP) with the wire format of
team/docs/pc_communication.md, and talks the TCP command protocol (PING, AUTO, MANUAL, MOVE, HOME). Software H.264 (x264enc) so it also runs
without the board's hardware encoder. Needs only the Python standard library; the video decode check runs when GStreamer's Python binding is there.
11-15: the picture goes to the video encoder and the PC only while a PC GUI is connected on the command port (stdout GUI_PICTURE 0|1); the
people of a G line still reach the executor. --gui-video-always 1 or --gui-command-port 0 = always, as before.
Usage (from the folder that holds gimbal_bridge):  python3 test_gui_bridge.py [path of gimbal_bridge] [encoder: x264enc (default) or mpph264enc]
Exit code 0 only when every check passes."""
import json, os, pty, select, socket, struct, subprocess, sys, threading, time, tty
from pathlib import Path

BRIDGE = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / 'gimbal_bridge')
ENCODER = sys.argv[2] if len(sys.argv) > 2 else 'x264enc'   # mpph264enc on the board
VIDEO, META, CMD = 15000, 15001, 15002
W, H = 640, 480
MAGIC = 0x4c574952
STOP = {'T': 135}
failures = []


def check(ok, text):
    print(('ok   ' if ok else 'FAIL ') + text, flush=True)
    if not ok: failures.append(text)


class Rig:
    def __init__(self, *extra, command_port=CMD):
        self.master, slave = pty.openpty()
        tty.setraw(self.master)
        self.name = os.ttyname(slave)
        self.slave = slave
        self.video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.video.bind(('127.0.0.1', VIDEO)); self.video.setblocking(False)
        self.meta = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.meta.bind(('127.0.0.1', META)); self.meta.setblocking(False)
        self.proc = subprocess.Popen([BRIDGE, '--device', self.name, '--heartbeat-ms', '1500', '--gui-host', '127.0.0.1', '--gui-bind', '127.0.0.1',
                                      '--gui-video-port', str(VIDEO), '--gui-metadata-port', str(META), '--gui-command-port', str(command_port),
                                      '--gui-encoder', ENCODER, '--gui-fps', '30', '--gui-bitrate', '600', *extra],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open('test_gui_bridge.stderr.log', 'wb'), bufsize=0)
        self.out = b''; self.lines = []; self.serial_buf = b''; self.serial_seen = []
        self.frame_id = 0; self.people = [(7, 100.0, 100.0, 40.0, 80.0, 0.9), (12, 420.0, 300.0, 50.0, 90.0, 0.8)]
        self.lock = threading.Lock(); self.feeding = False
        assert self.line(3.0) == 'READY', 'no READY'

    # ---- bridge stdin / stdout
    def send(self, text):
        with self.lock: self.proc.stdin.write((text + '\n').encode())

    def line(self, timeout=1.0):
        end = time.monotonic() + timeout
        while b'\n' not in self.out:
            left = end - time.monotonic()
            if left <= 0: return None
            r, _, _ = select.select([self.proc.stdout], [], [], left)
            if not r: return None
            d = os.read(self.proc.stdout.fileno(), 65536)
            if not d: return None
            self.out += d
        a, self.out = self.out.split(b'\n', 1)
        a = a.decode(); self.lines.append(a)
        return a

    def drain(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end: self.line(max(end - time.monotonic(), .01))

    def wait_line(self, prefix, timeout=2.0, where=None):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            a = self.line(max(end - time.monotonic(), .01))
            if a is not None and a.startswith(prefix) and (where is None or where(a)): return a
        return None

    def state(self, a):
        f = a.split(); return dict(auto=int(f[1]), lost=int(f[2]), moving=int(f[3]), error=int(f[4]), id=int(f[5]))

    def wait_state(self, want, timeout=2.0):
        a = self.wait_line('S ', timeout, lambda s: all(self.state(s)[k] == v for k, v in want.items()))
        return self.state(a) if a else None

    # ---- frames with pictures
    def picture(self):
        row = (list(range(0, 65536, 65536 // W))[:W])
        pic = bytearray()
        line = struct.pack('<%dH' % W, *row)
        for y in range(H): pic += line
        return bytes(pic)

    def frame(self, people=None):
        people = self.people if people is None else people
        self.frame_id += 1
        head = 'G %d %d %d %d %d' % (self.frame_id, time.monotonic_ns(), W, H, len(people)) + ''.join(' %d %.1f %.1f %.1f %.1f %.3f' % p for p in people)
        with self.lock: self.proc.stdin.write(head.encode() + b'\n' + self.pic)
        return self.frame_id

    def feed(self):
        self.pic = self.picture()
        def run():
            while self.feeding:
                self.frame(); time.sleep(1 / 30)
        self.feeding = True; self.thread = threading.Thread(target=run, daemon=True); self.thread.start()

    # ---- serial (what the gimbal would receive)
    def serial(self, timeout=.5):
        end = time.monotonic() + timeout
        while b'\n' not in self.serial_buf:
            left = end - time.monotonic()
            if left <= 0: return None
            r, _, _ = select.select([self.master], [], [], left)
            if not r: return None
            self.serial_buf += os.read(self.master, 4096)
        a, self.serial_buf = self.serial_buf.split(b'\n', 1)
        c = json.loads(a.decode().strip()); self.serial_seen.append(c)
        return c

    def serial_until(self, want, timeout=2.0):
        """want(cmd) -> bool; returns the first matching command, None on timeout"""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            c = self.serial(max(end - time.monotonic(), .01))
            if c is not None and want(c): return c
        return None

    # ---- UDP from the bridge
    def datagrams(self, sock, seconds):
        end = time.monotonic() + seconds; got = []
        while time.monotonic() < end:
            r, _, _ = select.select([sock], [], [], max(end - time.monotonic(), .01))
            if not r: continue
            try:
                while True: got.append(sock.recv(65535))
            except BlockingIOError: pass
        return got

    def first_datagram(self, sock, timeout):
        """seconds until the next datagram on sock, None on timeout"""
        start = time.monotonic()
        r, _, _ = select.select([sock], [], [], timeout)
        return time.monotonic() - start if r else None

    def close(self):
        self.feeding = False
        if getattr(self, 'thread', None): self.thread.join(1)
        try: self.proc.stdin.close()
        except OSError: pass
        try: self.proc.wait(5)
        except subprocess.TimeoutExpired: self.proc.kill()
        if self.video: self.video.close()
        self.meta.close()


def decode_metadata(p):
    magic, frame_id, rtp_ts, started_us, count = struct.unpack('>IQIQI', p[:28])
    assert magic == MAGIC, hex(magic)
    tracks = []
    for i in range(count):
        o = 28 + 25 * i
        tid, x, y, w, h, cls, conf = struct.unpack('>IffffBf', p[o:o + 25])
        tracks.append(dict(id=tid, x=x, y=y, w=w, h=h, cls=cls, conf=conf))
    assert len(p) == 28 + 25 * count, (len(p), count)
    return dict(frame_id=frame_id, rtp_ts=rtp_ts, started_us=started_us, tracks=tracks)


class Pc:
    """the PC's control connection"""
    def __init__(self):
        self.sock = socket.create_connection(('127.0.0.1', CMD), timeout=3); self.buf = b''

    def send(self, category, command, payload=b''):
        self.sock.sendall(struct.pack('>IBBH', MAGIC, category, command, len(payload)) + payload)

    def message(self, timeout=2.0):
        self.sock.settimeout(timeout)
        try:
            while True:
                if len(self.buf) >= 8:
                    magic, cat, cmd, n = struct.unpack('>IBBH', self.buf[:8]); assert magic == MAGIC
                    if len(self.buf) >= 8 + n:
                        payload = self.buf[8:8 + n]; self.buf = self.buf[8 + n:]
                        return cat, cmd, payload
                d = self.sock.recv(4096)
                if not d: return None
                self.buf += d
        except socket.timeout: return None

    def wait(self, cat, cmd, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            m = self.message(max(end - time.monotonic(), .05))
            if m is None: continue
            if m[0] == cat and m[1] == cmd: return m
        return None

    def close(self): self.sock.close()


def decode_video_check(rig, seconds=3.0):
    """decodes the RTP/H.264 stream with GStreamer's Python binding; returns the number of decoded frames and their size, or None when the binding is missing"""
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
    except Exception as e:
        print('     (video decode check skipped: %s)' % e); return None
    rig.video.close(); rig.video = None   # one socket per port: udpsrc gets the packets from now on
    Gst.init(None)
    p = Gst.parse_launch('udpsrc port=%d caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false max-buffers=4 drop=true' % VIDEO)
    sink = p.get_by_name('sink'); p.set_state(Gst.State.PLAYING)
    n = 0; size = None; end = time.monotonic() + seconds
    while time.monotonic() < end:
        s = sink.emit('try-pull-sample', 200 * Gst.MSECOND)
        if s is None: continue
        caps = s.get_caps().get_structure(0); size = (caps.get_value('width'), caps.get_value('height')); n += 1
    p.set_state(Gst.State.NULL)
    return n, size


def always_on(name, *extra, command_port=CMD):
    """the behaviour before 11-15: video and metadata without a connected PC GUI"""
    rig = Rig(*extra, command_port=command_port)
    try:
        check(rig.wait_line('GUI_PICTURE ', 2.0) == 'GUI_PICTURE 1', '%s: the bridge asks for the picture from the start (GUI_PICTURE 1)' % name)
        rig.send('P'); rig.feed()
        check(rig.first_datagram(rig.video, 3.0) is not None, '%s: RTP packets without a connected PC GUI' % name)
        check(len(rig.datagrams(rig.meta, .5)) >= 5, '%s: track metadata without a connected PC GUI' % name)
    finally:
        rig.close()


def main():
    print('bridge', BRIDGE, flush=True)
    rig = Rig()
    try:
        check(rig.serial(2.0) == STOP, 'the executor sends one STOP when it opens the device')
        # ---- no PC GUI connected: nothing is encoded or sent, the people still reach the executor
        check(rig.wait_line('GUI_PICTURE ', 2.0) == 'GUI_PICTURE 0', 'at start the bridge says GUI_PICTURE 0: no PC GUI, people lines are enough')
        rig.send('P')
        rig.feed(); time.sleep(.6)
        quiet_rtp, quiet_meta = rig.datagrams(rig.video, .7), rig.datagrams(rig.meta, .3)
        check(not quiet_rtp and not quiet_meta, 'no PC GUI: no RTP packet and no metadata datagram although pictures arrive (%d, %d)' % (len(quiet_rtp), len(quiet_meta)))
        t = rig.wait_line('T ', 2.5)
        check(t is not None and t.split()[1:6] == ['0', '0', '0', '0', '0'], 'no PC GUI: status line reports no client and nothing given, sent or encoded: %s' % t)
        rig.send('P'); rig.send('A 7')
        check(rig.wait_line('R A', 2.0) == 'R A OK', 'no PC GUI: the people of the G lines reach the executor (the web page can select 7)')
        check(rig.wait_state(dict(auto=1, id=7), 2.0) is not None, 'no PC GUI: following 7')
        mv = rig.serial_until(lambda c: c.get('T') == 141, 2.5)
        check(mv is not None, 'no PC GUI: following sends gimbal moves (T=141): %s' % mv)
        rig.send('H'); check(rig.wait_state(dict(auto=0), 2.0) is not None, 'no PC GUI: web page stop')
        check(not rig.datagrams(rig.video, .3), 'no PC GUI: still no RTP packet after following')
        # ---- the PC GUI connects: the picture is wanted, video and metadata start
        pc = Pc(); connected = time.monotonic()
        first_rtp = rig.first_datagram(rig.video, 3.0)
        first_meta = rig.first_datagram(rig.meta, 3.0)
        first_meta = None if first_meta is None else time.monotonic() - connected
        check(first_rtp is not None and first_rtp < 1.0, 'PC GUI connected: the first RTP packet arrives within 1 s (%s s)' % ('%.3f' % first_rtp if first_rtp is not None else None))
        check(first_meta is not None and first_meta < 1.0, 'PC GUI connected: the first metadata datagram arrives within 1 s (%s s)' % ('%.3f' % first_meta if first_meta is not None else None))
        check(rig.wait_line('GUI_PICTURE ', 2.0) == 'GUI_PICTURE 1', 'PC GUI connected: the bridge says GUI_PICTURE 1')
        st = pc.wait(1, 3)   # System / STATUS on connection
        check(st is not None and len(st[2]) == 8 and st[2][0] == 0 and st[2][1] == 0, 'STATUS on connection: Manual / None / id 0: %s' % (st[2].hex() if st else None))
        time.sleep(.5)
        # metadata
        rig.meta_pkts = rig.datagrams(rig.meta, 1.0)
        check(len(rig.meta_pkts) >= 15, 'track metadata datagrams arrive on UDP %d (%d in 1 s)' % (META, len(rig.meta_pkts)))
        if rig.meta_pkts:
            m = decode_metadata(rig.meta_pkts[-1])
            check(m['frame_id'] > 0 and m['frame_id'] <= rig.frame_id, 'metadata carries the capture frame id (%d)' % m['frame_id'])
            check(len(m['tracks']) == 2 and m['tracks'][0]['id'] == 7 and m['tracks'][1]['id'] == 12, 'metadata carries both people with their ids: %s' % [t['id'] for t in m['tracks']])
            t = m['tracks'][0]
            check((t['x'], t['y'], t['w'], t['h'], t['cls']) == (100.0, 100.0, 40.0, 80.0, 0) and abs(t['conf'] - 0.9) < 1e-6, 'box, class 0 and confidence of person 7 are as sent: %s' % t)
            check(m['started_us'] > 1.6e15, 'gui_started_us is epoch microseconds')
        # video packets
        rtp = rig.datagrams(rig.video, 1.0)
        check(len(rtp) >= 10, 'RTP packets arrive on UDP %d (%d in 1 s)' % (VIDEO, len(rtp)))
        if rtp:
            v = rtp[0][0] >> 6; pt = rtp[0][1] & 0x7f
            check(v == 2 and pt == 96, 'RTP version 2, payload type 96 (H.264): v=%d pt=%d' % (v, pt))
        # T status
        t = rig.wait_line('T ', 3.5, lambda s: s.split()[1] == '1' and int(s.split()[3]) > 0 and int(s.split()[4]) > 0)
        check(t is not None, 'status line reports the GUI client and frames sent and encoded: %s' % t)
        # decoded video
        dv = decode_video_check(rig, 4.0)
        if dv is not None: check(dv[0] >= 10 and dv[1] == (W, H), 'the RTP stream decodes to %dx%d frames (%d in 3 s)' % (dv[1][0] if dv[1] else 0, dv[1][1] if dv[1] else 0, dv[0]))
        if rig.video is None:   # the decode check took the port; take it back for the checks after the GUI leaves
            rig.video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for attempt in range(20):
                try: rig.video.bind(('127.0.0.1', VIDEO)); break
                except OSError: time.sleep(.1)
            rig.video.setblocking(False)
        # ---- TCP commands
        pc.send(1, 4); a = pc.wait(1, 1)
        check(a is not None and a[2] == bytes([1, 4, 0]), 'PING -> ACK OK: %s' % (a[2].hex() if a else None))
        pc.send(2, 1, struct.pack('>I', 7)); a = pc.wait(1, 1)
        check(a is not None and a[2] == bytes([2, 1, 0]), 'AUTO 7 -> ACK OK (the person is in the fresh frames): %s' % (a[2].hex() if a else None))
        s = rig.wait_state(dict(auto=1, id=7), 2.0)
        check(s is not None, 'the executor follows 7 (web page sees S auto=1 id=7): %s' % s)
        st = pc.wait(1, 3, 2.0)
        check(st is not None and st[2][0] == 1 and st[2][1] == 1 and struct.unpack('>I', st[2][4:8])[0] == 7, 'STATUS to the GUI: Auto / Tracking / id 7: %s' % (st[2].hex() if st else None))
        mv = rig.serial_until(lambda c: c.get('T') == 141, 2.5)
        check(mv is not None, 'following sends gimbal moves (T=141) on the serial line: %s' % mv)
        pc.send(2, 1, struct.pack('>I', 999)); a = pc.wait(1, 1)
        check(a is not None and a[2] == bytes([2, 1, 4]), 'AUTO 999 (unknown) -> ACK INVALID_TARGET: %s' % (a[2].hex() if a else None))
        pc.send(2, 2); a = pc.wait(1, 1)
        check(a is not None and a[2] == bytes([2, 2, 0]), 'MANUAL -> ACK OK: %s' % (a[2].hex() if a else None))
        s = rig.wait_state(dict(auto=0), 2.0)
        check(s is not None, 'the executor left AUTO: %s' % s)
        check(rig.serial_until(lambda c: c == STOP, 2.0) is not None, 'leaving AUTO sends STOP')
        pc.send(3, 1, bytes([1, 0]))   # MOVE right, no ACK
        mv = rig.serial_until(lambda c: c.get('T') == 141, 2.0)
        check(mv is not None, 'MOVE (1,0) -> gimbal move on the serial line: %s' % mv)
        s = rig.wait_state(dict(moving=1), 2.0); check(s is not None, 'state shows manual moving: %s' % s)
        pc.send(3, 1, bytes([0, 0]))
        check(rig.serial_until(lambda c: c == STOP, 2.0) is not None, 'MOVE (0,0) -> STOP')
        pc.send(3, 3); a = pc.wait(1, 1)
        check(a is not None and a[2] == bytes([3, 3, 5]), 'HOME -> ACK NOT_READY (team: HOME completion unimplemented): %s' % (a[2].hex() if a else None))
        # select again, then the GUI leaves: the team drops the selection, the web page can select again after its next heartbeat
        pc.send(2, 1, struct.pack('>I', 12)); a = pc.wait(1, 1)
        check(a is not None and a[2] == bytes([2, 1, 0]), 'AUTO 12 -> ACK OK')
        check(rig.wait_state(dict(auto=1, id=12), 2.0) is not None, 'following 12')
        pc.close()
        s = rig.wait_state(dict(auto=0, id=0), 3.0)
        check(s is not None, 'GUI disconnect drops the selection (team behaviour): %s' % s)
        t = rig.wait_line('T ', 2.5, lambda s: s.split()[1] == '0'); check(t is not None, 'status reports the GUI client left')
        check('GUI_PICTURE 0' in rig.lines[-80:], 'the GUI left: the bridge says GUI_PICTURE 0')
        rig.datagrams(rig.video, .4)   # packets of frames that were already in the encoder
        late = rig.datagrams(rig.video, .7)
        check(not late, 'the GUI left: the video stops (%d RTP packets in 0.7 s)' % len(late))
        rig.send('P'); rig.send('A 7')
        r = rig.wait_line('R A', 2.0)
        check(r == 'R A OK', 'the web page re-opens its session with the next heartbeat and can select: %s' % r)
        check(rig.wait_state(dict(auto=1, id=7), 2.0) is not None, 'following 7 for the web page')
        rig.send('H'); check(rig.wait_state(dict(auto=0), 2.0) is not None, 'web page stop')
        # a second GUI connection works (listener stays), and its video starts again
        pc = Pc(); again = rig.first_datagram(rig.video, 3.0)
        st = pc.wait(1, 3); check(st is not None, 'a second GUI connection gets STATUS')
        check(again is not None and again < 1.0, 'a second GUI connection gets video again within 1 s (%s s)' % ('%.3f' % again if again is not None else None))
        pc.close()
        rig.feeding = False; time.sleep(.2)
    finally:
        rig.close()
        tail = [l for l in rig.lines if l.startswith('S ')][-1:]
        print('last state', tail, 'serial commands', len(rig.serial_seen), flush=True)
    always_on('--gui-video-always 1', '--gui-video-always', '1')
    always_on('--gui-command-port 0', command_port=0)
    print('FAILED %d' % len(failures) if failures else 'ALL OK', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
