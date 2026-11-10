#!/usr/bin/env python3
"""Checks the live server with the PC GUI link on the M1 without a gimbal and without the PC: the server runs on the fake camera (FLIR frames) with a
pseudo terminal as its gimbal device and --gui-host 127.0.0.1 (software H.264). This script plays the PC: it reads the track metadata (UDP 5001),
decodes the video (UDP 5000, GStreamer Python binding) and sends GUI commands (TCP 5002), and compares with what the web page reports.
Usage: test_gui_server.py <tracker folder> <fake camera script> <out folder name>
Leaves the server stopped. Exit code 0 only when every check passes."""
import json, os, pty, select, signal, socket, struct, subprocess, sys, time, tty, urllib.error, urllib.request

folder, fake_cam, name = sys.argv[1:4]
PORT = 8091
BASE = 'http://127.0.0.1:%d' % PORT
VIDEO, META, CMD = 5000, 5001, 5002
MAGIC = 0x4c574952
STOP = {'T': 135}
master, slave = pty.openpty()
tty.setraw(master)
device = os.ttyname(slave)
os.close(slave)   # the server refuses a device another process holds open; reads below tolerate the moments when nobody holds the other side
wrapper = os.path.join(folder, name + '_cam.sh')
with open(wrapper, 'w') as f: f.write('#!/bin/bash\nexec python3 -B %s\n' % fake_cam)
os.chmod(wrapper, 0o755)
meta = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); meta.bind(('127.0.0.1', META)); meta.setblocking(False)
log = open(os.path.join(folder, name + '.log'), 'w')
server = subprocess.Popen(['python3', '-B', 'lwir_tracker_m1.py', '--tracker', 'bytetrack', '--no-vit', '--det', 'w480', '--class-gate', '--gmc', '--gmc-corners', '300', '--gmc-mindist', '5', '--n-init', '3', '--pipeline',
                           '--relabel', 'RVGGA0_REID2C256', '--mamba', '../mamba', '--mamba-crops', '3', '--mamba-duplicates', '--confirm-gate', 'gate_gru32_frozen.json', '--gimbal', device, '--live-cam', wrapper,
                           '--gui-host', '127.0.0.1', '--gui-bind', '127.0.0.1', '--gui-encoder', 'x264enc', '--gui-bitrate', '800',
                           '--live-port', str(PORT), '--live-bind', '127.0.0.1', '--live-seconds', '600', '--out', name], cwd=folder, stdout=log, stderr=subprocess.STDOUT)
buf = b''; seen = []; failures = []


def serial(timeout):
    global buf
    end = time.monotonic() + timeout
    while b'\n' not in buf:
        left = end - time.monotonic()
        if left <= 0: return None
        r, _, _ = select.select([master], [], [], left)
        if not r: return None
        try: buf += os.read(master, 4096)
        except OSError: time.sleep(.05)
    a, buf = buf.split(b'\n', 1)
    c = json.loads(a.decode().strip()); seen.append(c)
    return c


def serial_until(want, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        c = serial(max(end - time.monotonic(), .01))
        if c is not None and want(c): return c
    return None


def post(path):
    req = urllib.request.Request(BASE + path, method='POST', headers={'X-Gimbal': '1'})
    with urllib.request.urlopen(req, timeout=3) as r: return json.loads(r.read().decode())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=3) as r: return r.read()


def people():
    return json.loads(get('/gimbal/people').decode())


def check(ok, text):
    print(('ok   ' if ok else 'FAIL ') + text, flush=True)
    if not ok: failures.append(text)


def datagrams(seconds):
    end = time.monotonic() + seconds; got = []
    while time.monotonic() < end:
        r, _, _ = select.select([meta], [], [], max(end - time.monotonic(), .01))
        if not r: continue
        try:
            while True: got.append(meta.recv(65535))
        except BlockingIOError: pass
    return got


def decode_metadata(p):
    magic, frame_id, rtp_ts, started_us, count = struct.unpack('>IQIQI', p[:28]); assert magic == MAGIC
    tracks = [dict(zip(('id', 'x', 'y', 'w', 'h', 'cls', 'conf'), struct.unpack('>IffffBf', p[28 + 25 * i:53 + 25 * i]))) for i in range(count)]
    assert len(p) == 28 + 25 * count
    return dict(frame_id=frame_id, tracks=tracks)


class Pc:
    def __init__(self):
        self.sock = socket.create_connection(('127.0.0.1', CMD), timeout=3); self.buf = b''
    def send(self, category, command, payload=b''):
        self.sock.sendall(struct.pack('>IBBH', MAGIC, category, command, len(payload)) + payload)
    def wait(self, cat, cmd, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.sock.settimeout(max(end - time.monotonic(), .05))
            try:
                if len(self.buf) >= 8:
                    magic, c, m, n = struct.unpack('>IBBH', self.buf[:8]); assert magic == MAGIC
                    if len(self.buf) >= 8 + n:
                        payload = self.buf[8:8 + n]; self.buf = self.buf[8 + n:]
                        if c == cat and m == cmd: return payload
                        continue
                d = self.sock.recv(4096)
                if not d: return None
                self.buf += d
            except socket.timeout: return None
        return None
    def close(self): self.sock.close()


def decoded_frames(seconds):
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
    except Exception as e:
        print('     (video decode check skipped: %s)' % e); return None
    Gst.init(None)
    p = Gst.parse_launch('udpsrc port=%d caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false max-buffers=4 drop=true' % VIDEO)
    sink = p.get_by_name('sink'); p.set_state(Gst.State.PLAYING)
    n = 0; size = None; end = time.monotonic() + seconds
    while time.monotonic() < end:
        s = sink.emit('try-pull-sample', 200 * Gst.MSECOND)
        if s is None: continue
        c = s.get_caps().get_structure(0); size = (c.get_value('width'), c.get_value('height')); n += 1
    p.set_state(Gst.State.NULL)
    return n, size


code = 1
try:
    end = time.monotonic() + 150
    while time.monotonic() < end:
        try:
            if len(get('/snap.jpg')) > 1000: break
        except (urllib.error.URLError, OSError): pass
        check(server.poll() is None, 'server still running while waiting for the first image')
        time.sleep(1)
    else: check(False, 'server did not start')
    check(serial(3.) == STOP, 'the first gimbal command is STOP')
    post('/gimbal/ping')
    # metadata against the web page
    pk = datagrams(2.0)
    check(len(pk) >= 20, 'track metadata datagrams arrive on UDP %d (%d in 2 s)' % (META, len(pk)))
    web_ids = set()
    for _ in range(10):
        web_ids |= {q['id'] for q in people()['people']}; time.sleep(.1)
    wire = [decode_metadata(p) for p in pk[-20:]]
    wire_ids = {t['id'] for m in wire for t in m['tracks']}
    check(bool(wire_ids), 'the datagrams carry people (ids %s)' % sorted(wire_ids)[:8])
    check(bool(wire_ids & web_ids), 'the ids on the wire are the ids the web page shows (wire %s, web %s)' % (sorted(wire_ids)[:6], sorted(web_ids)[:6]))
    check(all(0 <= t['x'] < 640 and 0 <= t['y'] < 480 and t['w'] > 0 and t['h'] > 0 and t['cls'] == 0 for m in wire for t in m['tracks']), 'boxes are inside 640x480 camera pixels, class 0')
    ids = [m['frame_id'] for m in wire]
    check(ids == sorted(ids) and len(set(ids)) == len(ids), 'frame ids increase')
    st = people()['state']
    check(st['gui']['on'] and st['gui']['sent'] > 20 and st['gui']['failed'] == 0, 'the web page reports the GUI link: %s' % st['gui'])
    dv = decoded_frames(4.0)
    if dv is not None: check(dv[0] >= 20 and dv[1] == (640, 480), 'the video decodes to 640x480 (%d frames in 4 s)' % dv[0])
    # GUI commands
    pc = Pc()
    s = pc.wait(1, 3); check(s is not None and len(s) == 8, 'STATUS on connection: %s' % (s.hex() if s else None))
    end = time.monotonic() + 3
    while time.monotonic() < end and not people()['state']['gui']['client']: post('/gimbal/ping'); time.sleep(.2)
    check(people()['state']['gui']['client'] == 1, 'the web page reports the GUI client')
    a = None
    for attempt in range(5):   # a person that is in the frames right now
        pk = datagrams(.3)
        cur = [decode_metadata(p)['tracks'] for p in pk[-1:]]
        if not cur or not cur[0]: continue
        tid = cur[0][0]['id']
        pc.send(2, 1, struct.pack('>I', tid)); a = pc.wait(1, 1)
        if a == bytes([2, 1, 0]): break
    check(a == bytes([2, 1, 0]), 'AUTO <id from the wire> -> ACK OK: %s' % (a.hex() if a else None))
    end = time.monotonic() + 3; st = None
    while time.monotonic() < end:
        st = people()['state']
        if st['auto'] and st['id'] == tid: break
        time.sleep(.1)
    check(st is not None and st['auto'] and st['id'] == tid, 'the web page sees the GUI selection: auto=%s id=%s' % (st['auto'], st['id']))
    mv = serial_until(lambda c: c.get('T') == 141, 3.0)
    check(mv is not None, 'following sends gimbal moves (T=141): %s' % mv)
    pc.send(2, 2); a = pc.wait(1, 1); check(a == bytes([2, 2, 0]), 'MANUAL -> ACK OK')
    check(serial_until(lambda c: c == STOP, 3.0) is not None, 'leaving AUTO sends STOP')
    pc.close()
    time.sleep(1.0)
    # the web page works after the GUI left
    post('/gimbal/ping'); time.sleep(.3); post('/gimbal/ping')
    p = people()
    q = next((q for q in p['people'] if q['y2'] - q['y1'] > 8), None)
    if q is not None:
        st = post('/gimbal/click?x=%.5f&y=%.5f' % ((q['x1'] + q['x2']) / 2 / p['width'], (q['y1'] + q['y2']) / 2 / p['height']))
        end = time.monotonic() + 3
        while time.monotonic() < end and not st['auto']: st = post('/gimbal/ping'); time.sleep(.1)
        check(st['auto'], 'after the GUI left, a click on the web page follows again: %s' % st)
        post('/gimbal/stop')
    else: check(False, 'no person on the web page to click')
    server.send_signal(signal.SIGINT)
    check(serial_until(lambda c: c == STOP, 8.0) is not None, 'stopping the server sends STOP')
    code = 0 if not failures else 1
    print('ALL CHECKS PASSED' if not failures else 'FAILED %d' % len(failures), flush=True)
finally:
    if server.poll() is None:
        try: server.wait(10)
        except subprocess.TimeoutExpired: server.terminate()
        try: server.wait(20)
        except subprocess.TimeoutExpired: server.kill()
    time.sleep(1)
    subprocess.run("for p in $(ps -eo pid,args | grep -E 'gimbal_bridge --device /dev/pts' | grep -v grep | awk '{print $1}'); do kill $p; done", shell=True)
    print('commands seen', len(seen), seen[:6], flush=True)
sys.exit(code)
