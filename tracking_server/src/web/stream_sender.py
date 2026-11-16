"""The page's picture stream (GET /stream): one thread per viewer (11-15, user: 화면 전송 지연).

- Before each picture it waits until this viewer's socket has sent the previous one (Linux TIOCOUTQ: bytes not yet
  acknowledged by the viewer, at most LWIR_STREAM_OUTQ, plus what the link carries in one round trip, see below) and
  then sends the NEWEST picture, so a slow viewer gets fewer pictures, never a queue of old ones.
- Each picture is followed at once by the next part's boundary and headers: browsers draw a multipart part only when
  the next part's headers arrive (Chromium, WebKit and Firefox checked 11-15). So the parts carry no Content-Length.
- Each picture carries a JPEG comment 'LWIR n=<picture number> t=<camera frame arrival> s=<sent> [l=<level>]' (unix
  seconds of the server; t is missing on placeholder pictures). Decoders ignore comments; a page or a test client
  that reads the bytes gets the picture's age from it.
- The page opens its stream as /stream?p=<page id>&n=<count>. A newer stream of the same page replaces the older one:
  a browser keeps the old stream running until the new one has drawn its first picture, so every reopen of the page
  (network hiccup, tab shown again) ran two streams over the viewer's slow link. The replaced stream ends after the
  picture it is sending, so the browser never gets half a picture.
- Measured per viewer (11-15, for pages that cannot read the picture bytes, e.g. iPhone): about five pictures a second
  are timed from the write until the viewer has acknowledged all but their last ~1.5 KB (TCP_INFO bytes_acked). That
  time minus half the connection's shortest round trip is when the picture arrived at the viewer; minus the whole
  round trip it is the time the picture's bytes needed on the link. page_stats() gives the page its numbers
  (GET /time?p=<page id>): pictures per second, picture level, server time (frame in -> written), transfer time.
- Picture levels: a viewer whose picture needs more than DOWN_S on the link gets the next lighter level (the same drawn
  picture smaller: LWIR_STREAM_LIGHT, default 0.75,60;0.5,60 = 480x360 and 320x240 at JPEG quality 60), made once per
  picture for all such viewers and only when one exists. It goes back up when the link's measured rate (the slowest of
  the last timed pictures) would carry the heavier level within UP_S; the wait before going up doubles (at most 60 s)
  when that did not last 30 s.
  LWIR_STREAM_LIGHT=0 = full pictures only.
- In flight: a viewer far away (round trip longer than a frame) would otherwise get a picture only once per round
  trip. The queue limit is therefore at least the bytes the link carries in one shortest round trip (measured rate x
  round trip, at most LWIR_STREAM_INFLIGHT bytes, default 32768; 0 = off): those are on their way, not waiting.
"""
from collections import deque
import os
import re
import threading
import time
from urllib.parse import parse_qs

try:   # bytes still queued in a viewer's socket (Linux TIOCOUTQ = sent or not, not yet acknowledged)
    import fcntl as _fcntl
    import socket as _socket
    import struct as _struct
    import termios as _termios
    _TCP_INFO = getattr(_socket, 'TCP_INFO', None)

    def link_state(sock):
        """(bytes queued in the socket, still connected). A viewer that closed or reset the connection leaves its unsent
        bytes counted in TIOCOUTQ for good, so the TCP state is read too (TCP_INFO, 1 = established)."""
        try:
            queued = _struct.unpack('i', _fcntl.ioctl(sock.fileno(), _termios.TIOCOUTQ, b'\0\0\0\0'))[0]
        except (OSError, ValueError, _struct.error):
            return 0, True   # unknown: no wait; a dead connection then fails on the write
        try:
            alive = _TCP_INFO is None or sock.getsockopt(_socket.IPPROTO_TCP, _TCP_INFO, 1)[0] == 1
        except (OSError, IndexError):
            alive = True
        return queued, alive

    def link_acks(sock):
        """(bytes the viewer has acknowledged on this connection, its shortest round trip in s), or None if the kernel
        does not say (struct tcp_info: tcpi_bytes_acked at 120, tcpi_min_rtt in us at 148; Linux 4.6+)."""
        try:
            info = sock.getsockopt(_socket.IPPROTO_TCP, _TCP_INFO, 152)
            if len(info) < 152:
                return None
            min_rtt = _struct.unpack_from('I', info, 148)[0]
            return _struct.unpack_from('Q', info, 120)[0], (min_rtt / 1e6 if min_rtt < 0xFFFFFFFF else 0.0)
        except (OSError, TypeError, _struct.error):
            return None
    OUTQ_LIMIT = int(os.environ.get('LWIR_STREAM_OUTQ', '4096'))   # wait while more than this many bytes are queued; < 0 turns the wait off
    if OUTQ_LIMIT < 0:
        OUTQ_LIMIT = None
except ImportError:   # not Linux: no wait (old behaviour), nothing measured
    OUTQ_LIMIT = None

    def link_state(sock):
        return 0, True

    def link_acks(sock):
        return None

FIRST_PART = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
NEXT_PART = b'\r\n' + FIRST_PART
PAGE_ID = re.compile(r'[A-Za-z0-9_-]{1,32}\Z')


def _light_setting(text):
    """'scale,quality;scale,quality' -> ((scale, quality), ...) from heavier to lighter; '0' or bad -> ()."""
    out = []
    for part in text.split(';'):
        try:
            scale, quality = (float(v) for v in part.split(','))
        except ValueError:
            return ()
        if not (0.1 <= scale < 1.0 and 20 <= quality <= 95):
            return ()
        out.append((scale, int(quality)))
    return tuple(out)


LIGHT = _light_setting(os.environ.get('LWIR_STREAM_LIGHT', '0.75,60;0.5,60'))
try:
    INFLIGHT_MAX = max(0, int(os.environ.get('LWIR_STREAM_INFLIGHT', '32768')))
except ValueError:
    INFLIGHT_MAX = 32768
# 11-15 (user: low delay first, picture quality second). Times a picture's bytes need on the link (round trip excluded)
def _env_ms(name, default):
    try:
        return max(0.001, float(os.environ.get(name, default)) / 1000.0)
    except ValueError:
        return default / 1000.0


# 12 / 8 ms (was 25 / 15): on a 10 Mbit/s link the 480x360 picture arrives 7 ms sooner than the full one (20 vs 28 ms, harness)
DOWN_S = _env_ms('LWIR_STREAM_DOWN_MS', 12)   # more than this -> the next lighter level
UP_S = _env_ms('LWIR_STREAM_UP_MS', 8)        # the heavier level would need at most this at the measured rate -> back up
UP_WAIT_S = 5.0      # s on a lighter level before going up is considered (doubles, at most UP_WAIT_MAX, when it did not last)
UP_WAIT_MAX = 60.0
STEADY_S = 30.0      # s: a heavier level that lasts this long resets the wait to UP_WAIT_S
SAMPLE_GAP_S = 0.18  # s between timed pictures (about five a second)
SAMPLE_TAIL = 1500   # a timed picture counts as arrived when all but this many bytes are acknowledged (one packet: its ACK may be delayed)
SAMPLE_GIVE_UP_S = 1.0
POLL_S = 0.002
_LIGHT_CACHE = {'of': None, 'by': {}}
_LIGHT_LOCK = threading.Lock()
_JPEG_OPTIMIZE = 3   # cv2.IMWRITE_JPEG_OPTIMIZE: Huffman tables for this picture, about 12 % fewer bytes for under 1 ms


def light_jpeg(cv2, jpeg, drawn, level=1, settings=None):
    """The lighter picture (level 1..) of one published picture (shared by all viewers that want it, made at most once).
    jpeg: the published full picture; drawn: its picture before encoding. None if it cannot be made."""
    settings = settings or LIGHT
    with _LIGHT_LOCK:
        if _LIGHT_CACHE['of'] is not jpeg:
            _LIGHT_CACHE['of'], _LIGHT_CACHE['by'] = jpeg, {}
        by = _LIGHT_CACHE['by']
        if level not in by:
            try:
                scale, quality = settings[level - 1]
                small = cv2.resize(drawn, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, quality, _JPEG_OPTIMIZE, 1])
                by[level] = buf.tobytes() if ok and buf is not None else None
            except Exception:   # noqa: BLE001 (a bad picture must not end the stream; the full one is sent)
                by[level] = None
        return by[level]


def light_maker(cv2):
    """serve(light=...) for this server: (full jpeg, drawn picture, level) -> lighter jpeg or None; None when switched off."""
    if not LIGHT or cv2 is None:
        return None

    def make(jpeg, drawn, level):
        return light_jpeg(cv2, jpeg, drawn, level)
    make.levels = len(LIGHT)
    make.guess = tuple(scale * scale * 0.65 + 0.03 for scale, _q in LIGHT)   # share of the full picture's bytes, until one was made
    return make


def _median(values):
    values = sorted(values)
    return values[len(values) // 2]


class Quality:
    """Per viewer: which picture level (0 = full, 1.. = lighter), from the timed pictures (see the module notes).
    sample() for each timed picture, size() for each picture sent, pick() before each picture. Times are clock() seconds."""

    def __init__(self, log, levels=1, guess=(0.25,)):
        self.level = 0
        self.levels = levels
        self.guess = (1.0,) + tuple(guess)
        self.log = log
        self.sizes = [None] * (levels + 1)
        self.samples = deque(maxlen=8)   # (time, level, s on the link, bytes/s)
        self.up_at = 0.0
        self.up_wait = UP_WAIT_S
        self.up_since = None   # when it last went up a level

    def size(self, level, nbytes):
        old = self.sizes[level]
        self.sizes[level] = float(nbytes) if old is None else 0.8 * old + 0.2 * nbytes

    def _bytes(self, level):
        if self.sizes[level] is not None:
            return self.sizes[level]
        return (self.sizes[0] or 24000.0) * self.guess[level]

    def sample(self, now, level, link_s, rate):
        self.samples.append((now, level, link_s, rate))

    def pick(self, now):
        cur = [(s, r) for t, lv, s, r in self.samples if lv == self.level and now - t < 3.0]
        if len(cur) < 3:
            return self.level
        link_s, rate = _median([s for s, _r in cur]), max(_median([r for _s, r in cur]), 1.0)
        if link_s > DOWN_S and self.level < self.levels:
            new = self.level + 1
            while new < self.levels and self._bytes(new) / rate > DOWN_S:
                new += 1
            if self.up_since is not None:   # it had gone up: wait longer if that did not last
                self.up_wait = UP_WAIT_S if now - self.up_since >= STEADY_S else min(self.up_wait * 2, UP_WAIT_MAX)
            self.up_at = now + self.up_wait
            self.log('lighter pictures: level %d -> %d (a picture needed %.0f ms on the link)' % (self.level, new, link_s * 1000))
            self.level = new
            self.samples.clear()
        elif self.level > 0 and now >= self.up_at and len(cur) >= 5 and self._bytes(self.level - 1) / min(r for _s, r in cur) <= UP_S:
            # the slowest of the timed pictures: with other viewers on the same link some pictures wait behind theirs
            self.log('heavier pictures: level %d -> %d (the link carries %.0f KB/s)' % (self.level, self.level - 1, min(r for _s, r in cur) / 1000))
            self.level -= 1
            self.up_since = now
            self.samples.clear()
        return self.level


class Stats:
    """What one viewer's stream measured, for its page (page_stats)."""

    def __init__(self):
        self.sent = deque(maxlen=120)     # (time, level) of the pictures written
        self.server = deque(maxlen=30)    # s from camera frame arrival to the write
        self.net = deque(maxlen=10)       # (time, s from the write to the arrival at the viewer, shortest round trip s)

    def snapshot(self, now):
        sent = [(t, lv) for t, lv in list(self.sent) if now - t <= 2.0]
        server = list(self.server)
        net = [(a, r) for t, a, r in list(self.net) if now - t <= 5.0]
        out = dict(fps=round(len(sent) / 2.0, 1), level=sent[-1][1] if sent else 0, levels=len(LIGHT),   # levels: how many lighter levels there are
                   server_ms=round(_median(server) * 1000, 1) if server else None,
                   net_ms=round(_median([a for a, _r in net]) * 1000, 1) if net else None,
                   rtt_ms=round(net[-1][1] * 1000, 1) if net else None)
        out['est_ms'] = (round(out['server_ms'] + out['net_ms'], 1)
                         if sent and out['server_ms'] is not None and out['net_ms'] is not None else None)
        return out


class Viewer:
    """One open /stream connection. closed: None while it runs, else why it ends."""

    def __init__(self, page, n, peer):
        self.page, self.n, self.peer = page, n, peer
        self.closed = None
        self.stats = Stats()


def page_stats(live, page, clock=time.monotonic):
    """The numbers of this page's running stream (Stats.snapshot), or None: fps, level, server_ms (camera frame in ->
    written), net_ms (written -> arrived at the viewer), est_ms = their sum, rtt_ms."""
    if live is None or not page or not PAGE_ID.match(page):
        return None
    with live.jcond:
        viewer = (getattr(live, 'stream_pages', None) or {}).get(page)
    return viewer.stats.snapshot(clock()) if viewer is not None else None


def page_of(query):
    """(page id, stream number) from 'p=<id>&n=<count>'; (None, 0) for a stream without a page id."""
    q = parse_qs(query or '')
    page = (q.get('p') or [''])[0]
    if not PAGE_ID.match(page):
        return None, 0
    try:
        return page, int((q.get('n') or ['0'])[0])
    except ValueError:
        return page, 0


def register_locked(live, viewer):
    """Under live.jcond. Returns the stream this one replaces (now marked closed), or None."""
    pages = getattr(live, 'stream_pages', None)
    if pages is None:
        pages = live.stream_pages = {}
    if viewer.page is None:
        return None
    old = pages.get(viewer.page)
    if old is not None and old.n > viewer.n:   # a late request of this page: the page already has a newer stream
        viewer.closed = 'older than stream %d of this page' % old.n
        return None
    pages[viewer.page] = viewer
    if old is None:
        return None
    old.closed = 'replaced by stream %d of this page' % viewer.n
    live.jcond.notify_all()
    return old


def unregister_locked(live, viewer):
    pages = getattr(live, 'stream_pages', None)
    if pages is not None and viewer.page is not None and pages.get(viewer.page) is viewer:
        del pages[viewer.page]


def _segments(jpeg):
    """(marker, start, end) of the header segments of a JPEG, up to its first scan."""
    pos = 2
    while pos + 4 <= len(jpeg) and jpeg[pos] == 0xFF:
        marker = jpeg[pos + 1]
        if marker == 0xDA:   # start of scan: entropy-coded data follows
            return
        end = pos + 2 + int.from_bytes(jpeg[pos + 2:pos + 4], 'big')
        if end > len(jpeg):
            return
        yield marker, pos, end
        pos = end


def with_comment(jpeg, text):
    """The JPEG with a comment segment (FF FE) after its APP0 segment (or right after the start marker)."""
    if len(jpeg) < 4 or jpeg[:2] != b'\xff\xd8' or len(text) > 60000:
        return jpeg
    pos = 2
    for marker, _start, end in _segments(jpeg):
        if marker == 0xE0:
            pos = end
        break
    return jpeg[:pos] + b'\xff\xfe' + (len(text) + 2).to_bytes(2, 'big') + text + jpeg[pos:]


def picture_comment(seq, frame_t, now_perf, now_unix, level=0):
    """'LWIR n=<seq> t=<frame arrival> s=<now>' in unix seconds (+ ' l=<level>' on a lighter picture); frame_t is a
    time.perf_counter() value or None."""
    text = b'LWIR n=%d s=%.3f' % (seq, now_unix) if frame_t is None else \
        b'LWIR n=%d t=%.3f s=%.3f' % (seq, now_unix - (now_perf - frame_t), now_unix)
    return text + b' l=%d' % level if level else text


def read_comment(jpeg):
    """The numbers of a picture_comment() in a JPEG, e.g. {'n': 12.0, 't': ..., 's': ...}; {} if there is none."""
    for marker, start, end in _segments(jpeg):
        if marker == 0xFE and jpeg[start + 4:start + 9] == b'LWIR ':
            out = {}
            for kv in jpeg[start + 9:end].split():
                k, _, v = kv.partition(b'=')
                try:
                    out[k.decode()] = float(v)
                except ValueError:
                    pass
            return out
    return {}


STALL_S = 30.0   # a viewer that takes nothing of its picture for this long is closed (the page opens the stream again)


def serve(live, conn, write, query='', peer='?', *, limit=..., state=link_state, acks=link_acks, light=None,
          inflight=..., clock=time.monotonic, sleep=time.sleep, perf=time.perf_counter, unix=time.time, log=print):
    """Send pictures to one viewer until it goes away, the server stops or a newer stream of the same page opens.
    live: APP.LIVE (jcond, seq, jpeg, jpeg_frame_t, jpeg_drawn, stopped, viewers). write: the connection's blocking write.
    limit: bytes allowed in the socket queue before the next picture (default LWIR_STREAM_OUTQ; None = no wait).
    state(conn) -> (queued bytes, connected). acks(conn) -> (bytes acknowledged, shortest round trip s) or None.
    light: (jpeg, drawn picture, level) -> lighter jpeg or None (light_maker); None = full pictures only (also when
    there is no queue wait or nothing can be timed). inflight: most bytes added to the limit for a far viewer
    (default LWIR_STREAM_INFLIGHT). Returns why it ended."""
    limit = OUTQ_LIMIT if limit is ... else limit
    inflight = INFLIGHT_MAX if inflight is ... else inflight
    page, n = page_of(query)
    me = Viewer(page, n, peer)
    stats = me.stats
    t0 = clock()
    pictures = lights = sent = 0
    why = 'server stopped'
    write(FIRST_PART)
    first = acks(conn) if limit is not None else None
    written = first[0] + state(conn)[0] if first is not None else None   # bytes written on this connection so far
    quality = None
    if light is not None and written is not None:
        quality = Quality(lambda text: log('STREAM %s page %s stream %d: %s' % (peer, page or '-', n, text), flush=True),
                          getattr(light, 'levels', 1), getattr(light, 'guess', (0.25,)))
    with live.jcond:
        live.viewers += 1   # 11-12: pictures are made for every frame while this is above 0
        old = register_locked(live, me)
        viewers = live.viewers
    log('STREAM %s page %s stream %d: open, %d viewers%s'
        % (peer, page or '-', n, viewers, ', replaces stream %d' % old.n if old is not None else ''), flush=True)
    pending = None    # the timed picture: (clock at its write, acknowledged bytes that mean it arrived, level, its bytes)
    last_sample = -1e9
    rates = deque(maxlen=5)   # (time, bytes/s, shortest round trip s) of the last timed pictures
    allow = [0]       # bytes added to the limit: what the link carries in one round trip

    def newest_locked():
        drawn = getattr(live, 'jpeg_drawn', None)
        return (live.seq, live.jpeg, getattr(live, 'jpeg_frame_t', None),
                drawn[1] if drawn is not None and drawn[0] is live.jpeg else None)

    def check_pending():
        t_w, target, level, nbytes = pending
        got = acks(conn)
        if got is None:
            return None
        took = clock() - t_w
        if got[0] < target and took < SAMPLE_GIVE_UP_S:
            return pending
        min_rtt = min(got[1], took)
        link_s = max(took - min_rtt, 0.0)            # the bytes' time on the link: serialisation, and waiting behind others
        rate = nbytes / (link_s + 0.001)             # so this is at most the link's rate (less while it waits behind others)
        now = clock()
        stats.net.append((now, max(took - min_rtt / 2, min_rtt / 2), min_rtt))   # write -> arrival at the viewer
        rates.append((now, rate, min_rtt))
        recent = [(r, m) for t, r, m in rates if now - t < 5.0]
        allow[0] = int(min(inflight, _median([r for r, _m in recent]) * min(m for _r, m in recent))) if inflight and len(recent) >= 3 else 0
        if quality is not None:
            quality.sample(now, level, link_s, rate)
        return None
    try:
        last = 0
        fresh = False
        while True:
            # 11-13: nothing new (paused or held replay, single step, stalled camera): send the current picture again,
            # once 0.25 s after a new picture and then every 2 s, so a page opened during a pause gets the held
            # frame. With live video the short wait never expires (a new picture comes within about 40 ms).
            deadline = clock() + (0.25 if fresh else 2)
            while True:
                with live.jcond:
                    left = max(0.0, deadline - clock())
                    live.jcond.wait_for(lambda: live.seq != last or live.stopped or me.closed is not None,
                                        timeout=min(left, POLL_S) if pending is not None else left)
                    ready = live.seq != last or live.stopped or me.closed is not None
                if pending is not None:
                    pending = check_pending()
                if ready or clock() >= deadline:
                    break
            with live.jcond:
                if me.closed is not None:
                    why = me.closed
                    break
                if live.stopped:
                    break
                if live.seq == last and not live.jpeg:
                    fresh = False
                    continue
                fresh = live.seq != last
                last, jb, frame_t, drawn = newest_locked()
            queued = 0
            if limit is not None:
                # 11-15: on a slow link the old ~128 KB socket queue held about 1 s of pictures per viewer. Wait until the
                # previous picture has left this socket, then send the NEWEST picture.
                t_w = clock()
                while not live.stopped and me.closed is None:
                    queued, connected = state(conn)
                    if not connected:
                        raise ConnectionResetError('the viewer closed the connection')
                    if pending is not None:
                        pending = check_pending()
                    if queued <= max(limit, allow[0] + SAMPLE_TAIL):
                        break
                    if clock() - t_w > STALL_S:
                        raise TimeoutError('the viewer took nothing for %.0f s' % STALL_S)
                    sleep(POLL_S if pending is not None else 0.01)
                with live.jcond:
                    if me.closed is not None:
                        why = me.closed
                        break
                    if live.stopped:
                        break
                    if live.jpeg:
                        last, jb, frame_t, drawn = newest_locked()
            payload, level = jb, 0
            if quality is not None:
                quality.size(0, len(jb))
                level = quality.pick(clock())
                if level:
                    small = light(jb, drawn, level) if drawn is not None else None
                    if small:
                        payload = small
                        quality.size(level, len(small))
                    else:   # no drawn picture (a placeholder) or it could not be made: the full one
                        level = 0
            now_perf, t_send = perf(), clock()
            part = with_comment(payload, picture_comment(last, frame_t, now_perf, unix(), level)) + NEXT_PART
            write(part)
            pictures += 1
            lights += level > 0
            sent += len(payload)
            stats.sent.append((t_send, level))
            if frame_t is not None and fresh:
                stats.server.append(now_perf - frame_t)
            if written is not None:
                written += len(part)
                if pending is None and t_send - last_sample >= SAMPLE_GAP_S:
                    pending = (t_send, written - SAMPLE_TAIL, level, max(len(part) - SAMPLE_TAIL, 1))
                    last_sample = t_send
    except OSError as e:   # the viewer went away (or took nothing for the socket timeout / STALL_S)
        why = '%s%s' % (type(e).__name__, ': %s' % e if isinstance(e, (ConnectionResetError, TimeoutError)) and e.args else '')
    finally:
        with live.jcond:
            live.viewers -= 1
            unregister_locked(live, me)
        log('STREAM %s page %s stream %d: ended after %.1f s, %d pictures (%d lighter), %.1f MB (%s)'
            % (peer, page or '-', n, clock() - t0, pictures, lights, sent / 1e6, why), flush=True)
    return why
