"""PC tests of the /stream sender (stream_sender.py) with a fake slow link: no server, camera or network.

python -m unittest src.web.test_stream_sender   (from tracking_server/)
"""
import socket
import sys
import threading
import time
import unittest

import cv2
import numpy as np

from . import stream_sender as ss

BOUNDARY = ss.FIRST_PART


def picture(value, size=(480, 640), quality=70):
    rng = np.random.default_rng(value)
    img = np.clip(rng.normal(value, 25, size), 0, 255).astype(np.uint8)   # noisy like the thermal picture
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


class FakeLive:
    def __init__(self):
        self.jcond = threading.Condition()
        self.seq, self.jpeg, self.jpeg_frame_t = 0, b'', None
        self.stopped = False
        self.viewers = 0
        self.published = {}   # seq -> perf_counter when published

    def publish(self, jpeg, frame_t=None, drawn=None):
        with self.jcond:
            self.jpeg, self.jpeg_frame_t = jpeg, frame_t
            self.jpeg_drawn = (jpeg, drawn) if drawn is not None else None
            self.seq += 1
            self.published[self.seq] = time.perf_counter()
            self.jcond.notify_all()

    def stop(self):
        with self.jcond:
            self.stopped = True
            self.jcond.notify_all()


class SlowLink:
    """A link that delivers `rate` bytes/s. state() = (bytes written and not yet delivered, as TIOCOUTQ, connected).
    buffer: with no queue wait (limit None), write blocks while more than this many bytes are queued (a socket buffer)."""

    def __init__(self, live, rate, buffer=1 << 30):
        self.live, self.rate, self.buffer = live, float(rate), buffer
        self.lock = threading.Lock()
        self.backlog, self.t = 0.0, time.perf_counter()
        self.writes = []   # (perf time of the write, backlog before it, bytes, live.seq at the write)
        self.data = bytearray()
        self.fail = None
        self.closed = False   # the viewer closed the connection (with bytes still on the way)

    def _drain(self):
        now = time.perf_counter()
        self.backlog = max(0.0, self.backlog - self.rate * (now - self.t))
        self.t = now
        return now

    def state(self, _conn=None):
        with self.lock:
            self._drain()
            return int(self.backlog), not self.closed

    def write(self, b):
        if self.fail is not None:
            raise self.fail
        while True:
            with self.lock:
                now = self._drain()
                if self.backlog + len(b) <= self.buffer or self.backlog == 0:
                    self.writes.append((now, self.backlog, len(b), self.live.seq))
                    self.backlog += len(b)
                    self.data += b
                    return len(b)
            time.sleep(0.002)

    def delivered_at(self, i):
        """Perf time the last byte of write i reaches the viewer (with the next part's headers: drawable then)."""
        t, before, n, _ = self.writes[i]
        return t + (before + n) / self.rate


def run(live, link, query='', limit=4096, logs=None, light=None):
    out = {}
    th = threading.Thread(target=lambda: out.setdefault('why', ss.serve(
        live, None, link.write, query, 'test', limit=limit, state=link.state, light=light,
        log=(lambda *a, **k: logs.append(a[0])) if logs is not None else (lambda *a, **k: None))), daemon=True)
    th.start()
    return th, out


def publish_for(live, pics, seconds, fps=30.0, drawn=None):
    t_next = time.perf_counter()
    t_end = t_next + seconds
    k = 0
    while time.perf_counter() < t_end:
        live.publish(pics[k % len(pics)], time.perf_counter(), drawn)
        k += 1
        t_next += 1.0 / fps   # on a schedule: Windows sleeps overshoot by up to ~15 ms
        time.sleep(max(0.0, t_next - time.perf_counter()))
    return k


def parts(data):
    assert data.startswith(BOUNDARY), data[:60]
    chunks = bytes(data).split(b'\r\n' + BOUNDARY)
    chunks[0] = chunks[0][len(BOUNDARY):]
    return [c for c in chunks if c]


class SenderTests(unittest.TestCase):
    def setUp(self):
        self.pics = [picture(v) for v in (60, 90, 120)]

    def finish(self, live, th):
        live.stop()
        th.join(3)
        self.assertFalse(th.is_alive())
        self.assertEqual(live.viewers, 0)

    def test_slow_viewer_gets_only_new_pictures_and_never_a_queue(self):
        live = FakeLive()
        size = len(self.pics[0])
        link = SlowLink(live, rate=10 * size)   # about 10 pictures/s get through
        th, _ = run(live, link)
        published = publish_for(live, self.pics, 1.5)
        self.finish(live, th)
        sent = link.writes[1:]   # [0] = the first part's boundary and headers
        self.assertGreater(published, 20)   # about 45 (fewer on a busy PC)
        self.assertTrue(8 <= len(sent) <= 20, len(sent))   # about the link's rate (10/s)
        self.assertLess(len(sent), 0.75 * published)   # fewer pictures than were made
        for t, before, n, seq_at_write in sent:
            self.assertLessEqual(before, 4096)   # the previous picture had left the socket
        ages = []
        for i in range(1, len(link.writes)):
            info = ss.read_comment(parts(link.data)[i - 1])
            self.assertGreaterEqual(info['n'], link.writes[i][3] - 1)   # the newest picture when it was written
            ages.append(link.delivered_at(i) - live.published[int(info['n'])])
        self.assertLess(max(ages), 0.25, ages)   # one picture's transfer (0.1 s) + up to one frame of waiting

    def test_without_the_wait_a_slow_viewer_lags(self):   # the old behaviour (and the fallback off Linux), for contrast
        def median_age(limit):
            live = FakeLive()
            size = len(self.pics[0])
            link = SlowLink(live, rate=5 * size, buffer=2 * size + 4096)   # a socket buffer of about two pictures
            th, _ = run(live, link, limit=limit)
            publish_for(live, self.pics, 2.0)
            self.finish(live, th)
            got = parts(link.data)
            ages = sorted(link.delivered_at(i) - live.published[int(ss.read_comment(got[i - 1])['n'])]
                          for i in range(2, len(link.writes)))
            return ages[len(ages) // 2]
        old, new = median_age(None), median_age(4096)
        self.assertGreater(old, 0.45)   # two pictures queued ahead + the one blocked in write
        self.assertLess(new, 0.3)   # one picture's transfer (0.2 s) + up to one frame
        self.assertGreater(old, 2 * new)

    def test_fast_viewer_gets_every_picture(self):
        live = FakeLive()
        link = SlowLink(live, rate=50e6)
        th, _ = run(live, link)
        published = publish_for(live, self.pics, 1.0)
        self.finish(live, th)
        self.assertGreaterEqual(len(link.writes) - 1, published - 2)

    def test_framing_and_capture_time_comment(self):
        live = FakeLive()
        link = SlowLink(live, rate=50e6)
        th, _ = run(live, link)
        t_unix0, t_perf0 = time.time(), time.perf_counter()
        publish_for(live, self.pics, 0.5)
        self.finish(live, th)
        got = parts(link.data)
        self.assertGreaterEqual(len(got), 5)
        self.assertTrue(link.data.endswith(ss.NEXT_PART))   # each picture is followed at once by the next part's headers
        for part in got:
            info = ss.read_comment(part)
            k = int(info['n'])
            original = self.pics[(k - 1) % 3]
            self.assertEqual(cv2.imdecode(np.frombuffer(part, np.uint8), 0).tolist(),
                             cv2.imdecode(np.frombuffer(original, np.uint8), 0).tolist())   # same picture
            capture_unix = t_unix0 + (live.published[k] - t_perf0)
            self.assertAlmostEqual(info['t'], capture_unix, delta=0.02)
            self.assertGreaterEqual(info['s'] + 0.002, info['t'])

    def test_newer_stream_of_the_same_page_replaces_the_older(self):
        live = FakeLive()
        logs = []
        a, b, c = SlowLink(live, 50e6), SlowLink(live, 50e6), SlowLink(live, 50e6)
        th_a, out_a = run(live, a, 'p=abc&n=1', logs=logs)
        th_c, out_c = run(live, c, 'p=other&n=1', logs=logs)
        publish_for(live, self.pics, 0.3)
        th_b, out_b = run(live, b, 'p=abc&n=2', logs=logs)
        publish_for(live, self.pics, 0.3)
        th_a.join(1)
        self.assertFalse(th_a.is_alive())
        self.assertEqual(out_a['why'], 'replaced by stream 2 of this page')
        self.assertTrue(a.data.endswith(ss.NEXT_PART))   # it ended after a whole picture
        self.assertTrue(th_b.is_alive() and th_c.is_alive())
        self.assertEqual(live.viewers, 2)
        self.assertIs(live.stream_pages['abc'].n, 2)
        live.stop()
        for th in (th_b, th_c):
            th.join(3)
        self.assertEqual(live.viewers, 0)
        self.assertEqual(live.stream_pages, {})
        self.assertTrue(any('replaces stream 1' in line for line in logs), logs)

    def test_a_late_older_request_of_a_page_ends_at_once(self):
        live = FakeLive()
        new, old = SlowLink(live, 50e6), SlowLink(live, 50e6)
        th_new, _ = run(live, new, 'p=abc&n=5')
        live.publish(self.pics[0], time.perf_counter())
        time.sleep(0.1)
        th_old, out_old = run(live, old, 'p=abc&n=4')
        th_old.join(1)
        self.assertEqual(out_old['why'], 'older than stream 5 of this page')
        self.assertEqual(bytes(old.data), ss.FIRST_PART)
        self.assertTrue(th_new.is_alive())
        self.finish(live, th_new)

    def test_paused_picture_is_sent_again_after_a_quarter_second(self):
        live = FakeLive()
        link = SlowLink(live, 50e6)
        live.publish(self.pics[0], time.perf_counter())
        th, _ = run(live, link)
        time.sleep(0.6)
        self.finish(live, th)
        times = [w[0] - link.writes[1][0] for w in link.writes[1:]]
        self.assertEqual(len(times), 2, times)   # the held picture once more, 0.25 s after it (then every 2 s)
        self.assertAlmostEqual(times[1], 0.25, delta=0.1)

    def test_repeat_waits_for_the_queue_too(self):
        live = FakeLive()
        big = picture(77, size=(960, 1280), quality=95)
        link = SlowLink(live, rate=len(big) / 1.0)   # one picture takes 1 s
        live.publish(big, time.perf_counter())
        th, _ = run(live, link)
        time.sleep(0.7)
        self.finish(live, th)
        self.assertEqual(len(link.writes), 2)   # no repeat while the first copy is still on its way

    def test_viewer_gone_ends_the_stream(self):
        live = FakeLive()
        link = SlowLink(live, 50e6)
        th, out = run(live, link, 'p=abc&n=1')
        live.publish(self.pics[0], time.perf_counter())
        time.sleep(0.1)
        link.fail = BrokenPipeError()
        live.publish(self.pics[1], time.perf_counter())
        th.join(2)
        self.assertEqual(out['why'], 'BrokenPipeError')
        self.assertEqual(live.viewers, 0)
        self.assertEqual(live.stream_pages, {})

    def test_viewer_that_closes_while_its_picture_is_on_the_way_ends_at_once(self):
        live = FakeLive()
        big = picture(77, size=(960, 1280), quality=95)
        link = SlowLink(live, rate=len(big) / 10.0)   # one picture takes 10 s
        th, out = run(live, link, 'p=abc&n=1')
        live.publish(big, time.perf_counter())
        time.sleep(0.1)
        live.publish(big, time.perf_counter())   # waits for the queue
        time.sleep(0.1)
        t = time.perf_counter()
        link.closed = True   # TIOCOUTQ keeps counting the unsent bytes of a reset connection
        th.join(2)
        self.assertFalse(th.is_alive())
        self.assertLess(time.perf_counter() - t, 0.5)
        self.assertTrue(out['why'].startswith('ConnectionResetError'), out['why'])
        self.assertEqual(live.viewers, 0)

    def test_viewer_that_takes_nothing_is_closed(self):
        live = FakeLive()
        link = SlowLink(live, rate=1.0)   # takes (almost) nothing
        old = ss.STALL_S
        ss.STALL_S = 0.3
        try:
            th, out = run(live, link)
            live.publish(self.pics[0], time.perf_counter())
            time.sleep(0.05)
            live.publish(self.pics[1], time.perf_counter())
            th.join(2)
        finally:
            ss.STALL_S = old
        self.assertFalse(th.is_alive())
        self.assertTrue(out['why'].startswith('TimeoutError'), out['why'])
        self.assertEqual(len(link.writes), 2)   # the first part's headers and one picture, nothing queued behind it

    @unittest.skipUnless(sys.platform.startswith('linux'), 'TIOCOUTQ and TCP_INFO are Linux only')
    def test_real_socket_viewer_that_resets_ends_at_once(self):   # what the device does when a page is closed
        live = FakeLive()
        srv = socket.socket()
        srv.bind(('127.0.0.1', 0))
        srv.listen(1)
        cli = socket.socket()
        cli.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)   # takes little, then nothing (it never reads)
        cli.connect(srv.getsockname())
        conn, _ = srv.accept()
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)
        self.addCleanup(srv.close)
        self.addCleanup(conn.close)
        out = {}
        th = threading.Thread(target=lambda: out.setdefault('why', ss.serve(
            live, conn, conn.sendall, 'p=abc&n=1', 'test', limit=4096, log=lambda *a, **k: None)), daemon=True)
        th.start()
        for _ in range(8):   # fill the socket: the stream then waits for the queue
            live.publish(self.pics[0], time.perf_counter())
            time.sleep(0.05)
        self.assertGreater(ss.link_state(conn)[0], 4096)
        self.assertTrue(th.is_alive())
        t = time.perf_counter()
        cli.close()   # unread bytes -> the kernel resets the connection; TIOCOUTQ keeps the unsent count
        th.join(2)
        self.assertFalse(th.is_alive())
        self.assertLess(time.perf_counter() - t, 0.5)
        self.assertTrue(out['why'].startswith('ConnectionResetError'), out['why'])
        self.assertEqual(live.viewers, 0)

    def test_comment_helpers(self):
        jpeg = self.pics[0]
        self.assertEqual(ss.read_comment(jpeg), {})
        marked = ss.with_comment(jpeg, ss.picture_comment(3, None, 0.0, 1000.0))
        self.assertEqual(ss.read_comment(marked), {'n': 3.0, 's': 1000.0})
        self.assertEqual(ss.with_comment(b'not a jpeg', b'LWIR n=1'), b'not a jpeg')
        no_app0 = b'\xff\xd8' + jpeg[2 + 2 + int.from_bytes(jpeg[4:6], 'big'):]   # the same JPEG without its APP0 segment
        self.assertEqual(ss.read_comment(ss.with_comment(no_app0, b'LWIR n=9')), {'n': 9.0})
        self.assertEqual(ss.page_of('p=ab-c_9&n=12'), ('ab-c_9', 12))
        self.assertEqual(ss.page_of('7'), (None, 0))   # the old page's '/stream?7'
        self.assertEqual(ss.page_of('p=../x&n=1'), (None, 0))


class LighterPictureTests(unittest.TestCase):
    def test_slow_full_pictures_switch_to_lighter_ones(self):
        logs = []
        q = ss.Quality(logs.append)
        kinds, t = [], 0.0
        for _ in range(8):
            k = q.pick(t, waited=True)   # each picture had to wait for the one before: it took 150 ms
            kinds.append(k)
            q.sent(t, k)
            t += 0.15
        self.assertEqual(kinds, ['full'] * 4 + ['light'] * 4)
        self.assertIn('lighter pictures (a full picture took 150 ms)', logs)

    def test_full_pictures_that_get_through_in_time_stay_full(self):
        q = ss.Quality(lambda text: None)
        t = 0.0
        for _ in range(30):
            k = q.pick(t, waited=True)   # 40 ms each: under LIGHT_ABOVE
            self.assertEqual(k, 'full')
            q.sent(t, k)
            t += 0.04

    def test_test_picture_decides_the_way_back(self):
        q = ss.Quality(lambda text: None)
        t = 0.0
        for _ in range(5):   # into lighter pictures
            q.sent(t, q.pick(t, waited=True))
            t += 0.15
        self.assertTrue(q.light)
        t = q.probe_at
        self.assertEqual(q.pick(t, waited=False), 'probe')
        q.sent(t, 'probe')
        self.assertEqual(q.pick(t + 0.2, waited=True), 'light')   # the full test picture took 200 ms: stay
        q.sent(t + 0.2, 'light')
        self.assertEqual(q.probe_gap, 2 * ss.PROBE_S)
        self.assertAlmostEqual(q.probe_at, t + 0.2 + 2 * ss.PROBE_S)
        t = q.probe_at
        self.assertEqual(q.pick(t, waited=False), 'probe')
        q.sent(t, 'probe')
        self.assertEqual(q.pick(t + 0.03, waited=False), 'full')   # it had gone before the next picture: full again
        q.sent(t + 0.03, 'full')
        self.assertFalse(q.light)
        t += 0.03
        for _ in range(4):   # but full pictures are slow again at once: lighter, and the next test waits twice as long
            t += 0.15
            q.sent(t, q.pick(t, waited=True))
        self.assertTrue(q.light)
        self.assertEqual(q.probe_gap, 4 * ss.PROBE_S)
        self.assertAlmostEqual(q.probe_at, t + 4 * ss.PROBE_S)
        t = q.probe_at
        q.sent(t, q.pick(t, waited=False))
        t += 0.03
        q.sent(t, q.pick(t, waited=False))   # full again, and this time it lasts
        self.assertFalse(q.light)
        t += ss.STEADY_S + 1
        for _ in range(4):
            t += 0.15
            q.sent(t, q.pick(t, waited=True))
        self.assertTrue(q.light)
        self.assertEqual(q.probe_gap, ss.PROBE_S)   # full pictures had lasted: back to the short wait

    def test_slow_viewer_gets_lighter_pictures_end_to_end(self):
        live = FakeLive()
        full = picture(60)
        drawn = np.zeros((480, 640, 3), np.uint8)
        small = picture(61, size=(240, 320))
        calls = []
        light = lambda jpeg, d: calls.append(jpeg) or small   # noqa: E731
        link = SlowLink(live, rate=len(full) / 0.2)   # a full picture takes 200 ms, a lighter one about 50 ms
        logs = []
        th, _ = run(live, link, logs=logs, light=light)
        publish_for(live, [full], 3.0, drawn=drawn)
        live.stop()
        th.join(3)
        got = [ss.read_comment(p) for p in parts(link.data)]
        lighter = [g for g in got if g.get('l') == 1.0]
        self.assertGreater(len(lighter), 20, len(lighter))   # most pictures after the first second are lighter
        self.assertTrue(all(g.get('l') is None for g in got[:4]))   # the first ones are full
        self.assertTrue(any('lighter pictures' in line for line in logs), logs)
        self.assertTrue(all(c is full for c in calls))
        self.assertLess(len(link.data), len(got) * len(full) * 0.6)   # far fewer bytes than full pictures

    def test_no_drawn_picture_means_full_pictures(self):
        live = FakeLive()
        full = picture(60)
        link = SlowLink(live, rate=len(full) / 0.2)
        th, _ = run(live, link, light=lambda jpeg, d: b'never used')
        publish_for(live, [full], 2.0)   # placeholders and inline pictures have no drawn picture
        live.stop()
        th.join(3)
        self.assertTrue(all(ss.read_comment(p).get('l') is None for p in parts(link.data)))

    def test_light_jpeg_is_made_once_per_picture(self):
        drawn = np.zeros((480, 640, 3), np.uint8)
        cv2.rectangle(drawn, (100, 100), (200, 300), (0, 220, 255), 4)
        a, b = picture(10), picture(20)
        la = ss.light_jpeg(cv2, a, drawn, (0.5, 60))
        self.assertIs(ss.light_jpeg(cv2, a, drawn, (0.5, 60)), la)   # the same picture: made once, shared
        self.assertEqual(cv2.imdecode(np.frombuffer(la, np.uint8), 1).shape, (240, 320, 3))
        self.assertIsNot(ss.light_jpeg(cv2, b, drawn, (0.5, 60)), la)
        self.assertIsNone(ss.light_jpeg(cv2, picture(30), None, (0.5, 60)))   # cannot be made: None, full one is sent
        self.assertEqual(ss._light_setting('0'), None)
        self.assertEqual(ss._light_setting('0.5,60'), (0.5, 60))
        self.assertEqual(ss._light_setting('1.0,60'), None)


if __name__ == '__main__':
    unittest.main()
