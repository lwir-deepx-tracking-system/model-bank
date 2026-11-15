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

    def __init__(self, live, rate, buffer=1 << 30, rtt=0.0):
        self.live, self.rate, self.buffer, self.rtt = live, float(rate), buffer, rtt
        self.total = 0
        self.lock = threading.Lock()
        self.backlog, self.t = 0.0, time.perf_counter()
        self.writes = []   # (perf time of the write, backlog before it, bytes, live.seq at the write)
        self.data = bytearray()
        self.fail = None
        self.closed = False   # the viewer closed the connection (with bytes still on the way)
        self.arrivals, self.acked = [], 0   # with rtt: (time the write's last byte is acknowledged - rtt, total bytes then)

    def _drain(self):
        now = time.perf_counter()
        self.backlog = max(0.0, self.backlog - self.rate * (now - self.t))
        self.t = now
        return now

    def _acked(self, now):
        while self.arrivals and self.arrivals[0][0] + self.rtt <= now:
            self.acked = self.arrivals.pop(0)[1]
        return self.acked

    def state(self, _conn=None):
        with self.lock:
            now = self._drain()
            return int(self.total - self._acked(now) if self.rtt else self.backlog), not self.closed

    def acks(self, _conn=None):
        """(bytes acknowledged, shortest round trip) as TCP_INFO. With rtt: a write's bytes are acknowledged when its last
        byte has arrived (write + queue + rtt/2) plus the way back (rtt/2); state() then counts unacknowledged bytes."""
        with self.lock:
            now = self._drain()
            return (self._acked(now), self.rtt) if self.rtt else (int(self.total - self.backlog), 0.0)

    def write(self, b):
        if self.fail is not None:
            raise self.fail
        while True:
            with self.lock:
                now = self._drain()
                if self.backlog + len(b) <= self.buffer or self.backlog == 0:
                    self.writes.append((now, self.backlog, len(b), self.live.seq))
                    self.backlog += len(b)
                    self.total += len(b)
                    self.arrivals.append((now + self.backlog / self.rate, self.total))   # acknowledged one round trip after it left the queue
                    self.data += b
                    return len(b)
            time.sleep(0.002)

    def delivered_at(self, i):
        """Perf time the last byte of write i reaches the viewer (with the next part's headers: drawable then)."""
        t, before, n, _ = self.writes[i]
        return t + (before + n) / self.rate


def run(live, link, query='', limit=4096, logs=None, light=None, inflight=0):
    out = {}
    th = threading.Thread(target=lambda: out.setdefault('why', ss.serve(
        live, None, link.write, query, 'test', limit=limit, state=link.state, acks=link.acks, light=light, inflight=inflight,
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
    def quality(self, logs=None):
        q = ss.Quality((logs if logs is not None else []).append, levels=2, guess=(0.4, 0.2))
        q.size(0, 24000)
        return q

    def test_slow_link_steps_down_to_the_level_that_fits(self):
        logs = []
        q = self.quality(logs)
        t = 0.0
        self.assertEqual(q.pick(t), 0)
        for _ in range(3):   # a full picture needs 40 ms on a link of 600 KB/s: the middle level (9.6 KB = 16 ms) fits
            t += 0.2
            q.sample(t, 0, 0.040, 600e3)
        self.assertEqual(q.pick(t), 1)
        self.assertIn('level 0 -> 1', logs[-1])
        q2 = self.quality()
        for _ in range(3):   # 150 KB/s: the middle level would need 64 ms, so straight to the lightest
            t += 0.2
            q2.sample(t, 0, 0.160, 150e3)
        self.assertEqual(q2.pick(t), 2)

    def test_fast_link_keeps_full_pictures(self):
        q = self.quality()
        t = 0.0
        for _ in range(20):
            t += 0.2
            q.sample(t, 0, 0.012, 2e6)
            self.assertEqual(q.pick(t), 0)

    def test_back_up_when_the_rate_carries_the_heavier_level_and_back_off_when_it_does_not_last(self):
        q = self.quality()
        t = 0.0
        for _ in range(3):
            t += 0.2
            q.sample(t, 0, 0.080, 300e3)
        self.assertEqual(q.pick(t), 2)   # the middle level (9.6 KB) would need 32 ms, the lightest (4.8 KB) 16 ms
        t_down = t
        for _ in range(3):   # the link got fast, but not before the wait is over
            t += 0.2
            q.sample(t, 2, 0.003, 1.2e6)
        self.assertEqual(q.pick(t), 2)
        t = t_down + ss.UP_WAIT_S
        for k in range(5):   # one of the pictures waited behind another viewer's: the slowest one counts
            t += 0.2
            q.sample(t, 2, 0.003, 1.2e6 if k else 0.3e6)
        self.assertEqual(q.pick(t), 2)
        for _ in range(5):
            t += 0.2
            q.sample(t, 2, 0.003, 1.2e6)
        self.assertEqual(q.pick(t), 1)   # 9.6 KB at 1.2 MB/s = 8 ms
        for _ in range(3):   # the middle level is too slow after all: down again, and the next try waits twice as long
            t += 0.2
            q.sample(t, 1, 0.050, 190e3)
        self.assertEqual(q.pick(t), 2)
        self.assertEqual(q.up_wait, 2 * ss.UP_WAIT_S)
        self.assertAlmostEqual(q.up_at, t + 2 * ss.UP_WAIT_S)

    def test_old_samples_and_other_levels_do_not_count(self):
        q = self.quality()
        for k in range(3):
            q.sample(0.1 * k, 0, 0.080, 300e3)
        self.assertEqual(q.pick(10.0), 0)   # 10 s old
        q.sample(10.0, 1, 0.080, 300e3)
        q.sample(10.1, 1, 0.080, 300e3)
        q.sample(10.2, 1, 0.080, 300e3)
        self.assertEqual(q.pick(10.3), 0)   # timed on another level

    def test_slow_viewer_gets_lighter_pictures_end_to_end_and_its_page_gets_the_numbers(self):
        live = FakeLive()
        full = picture(60)
        drawn = np.zeros((480, 640, 3), np.uint8)
        small = {1: picture(61, size=(360, 480)), 2: picture(62, size=(240, 320))}
        calls = []

        def light(jpeg, d, level):
            calls.append((jpeg, level))
            return small[level]
        light.levels, light.guess = 2, (len(small[1]) / len(full), len(small[2]) / len(full))
        link = SlowLink(live, rate=len(full) / 0.2, rtt=0.03)   # a full picture needs 200 ms on the link, the lightest about 50
        logs = []
        th, _ = run(live, link, 'p=slowpage&n=1', logs=logs, light=light)
        publish_for(live, [full], 3.0, drawn=drawn)
        st = ss.page_stats(live, 'slowpage')
        live.stop()
        th.join(3)
        got = [ss.read_comment(p) for p in parts(link.data)]
        lighter = [g for g in got if g.get('l') == 2.0]
        self.assertGreater(len(lighter), 15, len(lighter))
        self.assertTrue(all(g.get('l') is None for g in got[:2]))   # the first ones are full
        self.assertTrue(any('lighter pictures: level 0 -> 2' in line for line in logs), logs)
        self.assertTrue(all(c[0] is full for c in calls))
        # the page's numbers: the lightest picture needs len/rate on the link + half the round trip to arrive
        expect = (len(small[2]) / link.rate + 0.015) * 1000
        self.assertEqual(st['level'], 2)
        self.assertAlmostEqual(st['net_ms'], expect, delta=25)
        self.assertAlmostEqual(st['rtt_ms'], 30, delta=1)
        self.assertGreater(st['fps'], 8)
        self.assertIsNotNone(st['est_ms'])
        self.assertIsNone(ss.page_stats(live, 'slowpage'))   # the stream has ended
        self.assertIsNone(ss.page_stats(live, '../x'))

    def test_arrival_estimate_matches_the_link(self):
        live = FakeLive()
        full = picture(60)
        for rate_kb, rtt in ((2000, 0.0), (600, 0.04), (300, 0.08)):
            link = SlowLink(live, rate=rate_kb * 1000.0, rtt=rtt)
            th, _ = run(live, link, 'p=est&n=1')
            publish_for(live, [full], 2.0)
            st = ss.page_stats(live, 'est')
            # what really happened on the link: write -> last byte at the viewer, median of the pictures
            real = sorted(link.delivered_at(i) + rtt / 2 - link.writes[i][0] for i in range(2, len(link.writes)))
            real_ms = real[len(real) // 2] * 1000
            live.stopped = True
            with live.jcond:
                live.jcond.notify_all()
            th.join(3)
            live.stopped = False
            delta = 8 if sys.platform.startswith('linux') else 22   # Windows sleeps in steps of up to ~15 ms
            self.assertAlmostEqual(st['net_ms'], real_ms, delta=delta, msg='%d KB/s rtt %.0f ms' % (rate_kb, rtt * 1000))

    def test_far_viewer_is_not_held_to_one_picture_per_round_trip(self):
        small = picture(61, size=(120, 160))
        self.assertGreater(len(small), 4096)
        rate = 100.0 * len(small)   # a picture needs 10 ms on the link; the round trip is 80 ms

        def fps(inflight):
            live = FakeLive()
            link = SlowLink(live, rate=rate, rtt=0.08)
            th, _ = run(live, link, inflight=inflight)
            publish_for(live, [small], 3.0)
            live.stop()
            th.join(3)
            late = [w for w in link.writes[1:] if w[0] - link.writes[1][0] > 1.5]
            return len(late) / 1.5, max(w[1] for w in late)
        held, _ = fps(0)
        free, queued = fps(32768)
        self.assertLess(held, 16)        # one picture per round trip (about 11/s)
        self.assertGreater(free, 22)     # nearly every picture
        self.assertLess(queued, 3 * len(small))   # still (almost) nothing waiting at the link itself

    def test_no_drawn_picture_means_full_pictures(self):
        live = FakeLive()
        full = picture(60)
        link = SlowLink(live, rate=len(full) / 0.2)
        light = lambda jpeg, d, level: b'never used'   # noqa: E731
        th, _ = run(live, link, light=light)
        publish_for(live, [full], 2.0)   # placeholders and inline pictures have no drawn picture
        live.stop()
        th.join(3)
        self.assertTrue(all(ss.read_comment(p).get('l') is None for p in parts(link.data)))

    def test_light_jpeg_is_made_once_per_picture_and_level(self):
        drawn = np.zeros((480, 640, 3), np.uint8)
        cv2.rectangle(drawn, (100, 100), (200, 300), (0, 220, 255), 4)
        a, b = picture(10), picture(20)
        two = ((0.75, 60), (0.5, 60))
        la = ss.light_jpeg(cv2, a, drawn, 2, two)
        self.assertIs(ss.light_jpeg(cv2, a, drawn, 2, two), la)   # the same picture and level: made once, shared
        self.assertEqual(cv2.imdecode(np.frombuffer(la, np.uint8), 1).shape, (240, 320, 3))
        self.assertEqual(cv2.imdecode(np.frombuffer(ss.light_jpeg(cv2, a, drawn, 1, two), np.uint8), 1).shape, (360, 480, 3))
        self.assertIsNot(ss.light_jpeg(cv2, b, drawn, 2, two), la)
        self.assertIsNone(ss.light_jpeg(cv2, picture(30), None, 1, two))   # cannot be made: None, the full one is sent
        self.assertEqual(ss._light_setting('0'), ())
        self.assertEqual(ss._light_setting('0.5,60'), ((0.5, 60),))
        self.assertEqual(ss._light_setting('0.75,60;0.5,60'), two)
        self.assertEqual(ss._light_setting('1.0,60'), ())


if __name__ == '__main__':
    unittest.main()
