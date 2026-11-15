"""The page's picture stream (GET /stream): one thread per viewer (11-15, user: 화면 전송 지연).

- Before each picture it waits until this viewer's socket has sent the previous one (Linux TIOCOUTQ: bytes not yet
  acknowledged by the viewer, at most LWIR_STREAM_OUTQ) and then sends the NEWEST picture, so a slow viewer gets fewer
  pictures, never a queue of old ones.
- Each picture is followed at once by the next part's boundary and headers: browsers draw a multipart part only when
  the next part's headers arrive (Chromium, WebKit and Firefox checked 11-15). So the parts carry no Content-Length.
- Each picture carries a JPEG comment 'LWIR n=<picture number> t=<camera frame arrival> s=<sent>' (unix seconds of the
  server; t is missing on placeholder pictures). Decoders ignore comments; a test client reads the picture's age.
- The page opens its stream as /stream?p=<page id>&n=<count>. A newer stream of the same page replaces the older one:
  a browser keeps the old stream running until the new one has drawn its first picture, so every reopen of the page
  (network hiccup, tab shown again) ran two streams over the viewer's slow link. The replaced stream ends after the
  picture it is sending, so the browser never gets half a picture.
- A viewer whose full picture takes more than LIGHT_ABOVE to get through gets a lighter picture (the same drawn
  picture at LWIR_STREAM_LIGHT scale and JPEG quality, default 0.5 = 320x240 and 60, about a quarter of the bytes),
  made once per picture for all such viewers and only when one exists. Every PROBE_S it sends one full picture; if
  that gets through within FULL_BELOW the viewer is back on full pictures. The wait doubles (at most 60 s) after a
  test picture that was too slow and after full pictures that did not last 30 s. LWIR_STREAM_LIGHT=0 = full only.
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
    OUTQ_LIMIT = int(os.environ.get('LWIR_STREAM_OUTQ', '4096'))   # wait while more than this many bytes are queued; < 0 turns the wait off
    if OUTQ_LIMIT < 0:
        OUTQ_LIMIT = None
except ImportError:   # not Linux: no wait (old behaviour)
    OUTQ_LIMIT = None

    def link_state(sock):
        return 0, True

FIRST_PART = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
NEXT_PART = b'\r\n' + FIRST_PART
PAGE_ID = re.compile(r'[A-Za-z0-9_-]{1,32}\Z')


def _light_setting(text):
    """'scale,quality' -> (scale, quality); '0' or bad -> None (no lighter picture)."""
    try:
        scale, quality = (float(v) for v in text.split(','))
    except ValueError:
        return None
    return (scale, int(quality)) if 0.1 <= scale < 1.0 and 20 <= quality <= 95 else None


LIGHT = _light_setting(os.environ.get('LWIR_STREAM_LIGHT', '0.5,60'))
LIGHT_ABOVE = 0.08   # s: a full picture that takes longer than this to get through -> lighter pictures
FULL_BELOW = 0.05    # s: a full test picture that gets through within this -> full pictures again
PROBE_S = 8.0        # s between full test pictures while on lighter ones (doubles after each miss, at most PROBE_MAX)
PROBE_MAX = 60.0
STEADY_S = 30.0      # s: full pictures that last this long reset the wait between test pictures to PROBE_S
_LIGHT_CACHE = {'of': None, 'jpeg': None}
_LIGHT_LOCK = threading.Lock()


def light_jpeg(cv2, jpeg, drawn, setting=None):
    """The lighter picture of one published picture (shared by all viewers that want it, made at most once).
    jpeg: the published full picture; drawn: its picture before encoding. None if it cannot be made."""
    scale, quality = setting or LIGHT
    with _LIGHT_LOCK:
        if _LIGHT_CACHE['of'] is not jpeg:
            try:
                small = cv2.resize(drawn, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, quality])
                out = buf.tobytes() if ok and buf is not None else None
            except Exception:   # noqa: BLE001 (a bad picture must not end the stream; the full one is sent)
                out = None
            _LIGHT_CACHE['of'], _LIGHT_CACHE['jpeg'] = jpeg, out
        return _LIGHT_CACHE['jpeg']


def light_maker(cv2):
    """serve(light=...) for this server: (full jpeg, drawn picture) -> lighter jpeg or None; None when switched off."""
    if LIGHT is None or cv2 is None:
        return None
    return lambda jpeg, drawn: light_jpeg(cv2, jpeg, drawn)


class Quality:
    """Per viewer: full or lighter pictures, from how long its pictures take to get through (see the module notes).
    pick() before each picture, sent() after it. All times are clock() seconds."""

    def __init__(self, log):
        self.light = False
        self.log = log
        self.samples = deque(maxlen=5)   # (time, s) of full pictures the next picture had to wait for
        self.probe_at = 0.0
        self.probe_gap = PROBE_S
        self.full_since = None   # when a test picture last brought the viewer back to full pictures
        self.prev = None   # (time the previous picture was written, it was full, it was a test picture)

    def pick(self, now, waited):
        """waited: the previous picture was still on its way when this one was ready. Returns 'full', 'light' or 'probe'."""
        if self.prev is not None:
            t_prev, was_full, was_probe = self.prev
            took = now - t_prev   # exact when waited, else an upper bound (it had already gone)
            if was_full and waited:
                self.samples.append((now, took))
            if was_probe:
                if not waited or took <= FULL_BELOW:
                    self.light = False
                    self.full_since = now
                    self.samples.clear()
                    self.log('full pictures again (a full picture took %.0f ms%s)' % (took * 1000, '' if waited else ' or less'))
                else:
                    self.probe_gap = min(self.probe_gap * 2, PROBE_MAX)
                    self.probe_at = now + self.probe_gap
        if not self.light:
            recent = sorted(s for t, s in self.samples if now - t < 3.0)
            if len(recent) >= 4 and recent[len(recent) // 2] > LIGHT_ABOVE:
                self.light = True
                if self.full_since is not None:   # a test picture had brought it back: wait longer if that did not last
                    self.probe_gap = PROBE_S if now - self.full_since >= STEADY_S else min(self.probe_gap * 2, PROBE_MAX)
                self.probe_at = now + self.probe_gap
                self.samples.clear()
                self.log('lighter pictures (a full picture took %.0f ms)' % (recent[len(recent) // 2] * 1000))
        if not self.light:
            return 'full'
        return 'probe' if now >= self.probe_at else 'light'

    def sent(self, now, kind):
        self.prev = (now, kind != 'light', kind == 'probe')


class Viewer:
    """One open /stream connection. closed: None while it runs, else why it ends."""

    def __init__(self, page, n, peer):
        self.page, self.n, self.peer = page, n, peer
        self.closed = None


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


def picture_comment(seq, frame_t, now_perf, now_unix, light=False):
    """'LWIR n=<seq> t=<frame arrival> s=<now>' in unix seconds (+ ' l=1' on a lighter picture); frame_t is a
    time.perf_counter() value or None."""
    text = b'LWIR n=%d s=%.3f' % (seq, now_unix) if frame_t is None else \
        b'LWIR n=%d t=%.3f s=%.3f' % (seq, now_unix - (now_perf - frame_t), now_unix)
    return text + b' l=1' if light else text


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


def serve(live, conn, write, query='', peer='?', *, limit=..., state=link_state, light=None, clock=time.monotonic,
          sleep=time.sleep, perf=time.perf_counter, unix=time.time, log=print):
    """Send pictures to one viewer until it goes away, the server stops or a newer stream of the same page opens.
    live: APP.LIVE (jcond, seq, jpeg, jpeg_frame_t, jpeg_drawn, stopped, viewers). write: the connection's blocking write.
    limit: bytes allowed in the socket queue before the next picture (default LWIR_STREAM_OUTQ; None = no wait).
    state(conn) -> (queued bytes, connected). light: (jpeg, drawn picture) -> lighter jpeg or None (light_maker);
    None = full pictures only (also when there is no queue wait). Returns why it ended."""
    limit = OUTQ_LIMIT if limit is ... else limit
    page, n = page_of(query)
    me = Viewer(page, n, peer)
    t0 = clock()
    pictures = lights = sent = 0
    why = 'server stopped'
    quality = None
    if light is not None and limit is not None:
        quality = Quality(lambda text: log('STREAM %s page %s stream %d: %s' % (peer, page or '-', n, text), flush=True))
    write(FIRST_PART)
    with live.jcond:
        live.viewers += 1   # 11-12: pictures are made for every frame while this is above 0
        old = register_locked(live, me)
        viewers = live.viewers
    log('STREAM %s page %s stream %d: open, %d viewers%s'
        % (peer, page or '-', n, viewers, ', replaces stream %d' % old.n if old is not None else ''), flush=True)

    def newest_locked():
        drawn = getattr(live, 'jpeg_drawn', None)
        return (live.seq, live.jpeg, getattr(live, 'jpeg_frame_t', None),
                drawn[1] if drawn is not None and drawn[0] is live.jpeg else None)
    try:
        last = 0
        fresh = False
        while True:
            with live.jcond:
                live.jcond.wait_for(lambda: live.seq != last or live.stopped or me.closed is not None,
                                    timeout=0.25 if fresh else 2)
                # 11-13: nothing new (paused or held replay, single step, stalled camera): send the current picture again,
                # once 0.25 s after a new picture and then every 2 s, so a page opened during a pause gets the held
                # frame. With live video the short wait never expires (a new picture comes within about 40 ms).
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
            waited = False
            if limit is not None:
                # 11-15: on a slow link the old ~128 KB socket queue held about 1 s of pictures per viewer. Wait until the
                # previous picture has left this socket, then send the NEWEST picture.
                t_w = clock()
                while not live.stopped and me.closed is None:
                    queued, connected = state(conn)
                    if not connected:
                        raise ConnectionResetError('the viewer closed the connection')
                    if queued <= limit:
                        break
                    if clock() - t_w > STALL_S:
                        raise TimeoutError('the viewer took nothing for %.0f s' % STALL_S)
                    waited = True
                    sleep(0.01)
                with live.jcond:
                    if me.closed is not None:
                        why = me.closed
                        break
                    if live.stopped:
                        break
                    if live.jpeg:
                        last, jb, frame_t, drawn = newest_locked()
            payload, kind = jb, 'full'
            if quality is not None:
                kind = quality.pick(clock(), waited)
                if kind == 'light':
                    small = light(jb, drawn) if drawn is not None else None
                    if small:
                        payload = small
                    else:   # no drawn picture (a placeholder) or it could not be made: the full one
                        kind = 'full'
            t_send = clock()
            write(with_comment(payload, picture_comment(last, frame_t, perf(), unix(), kind == 'light')) + NEXT_PART)
            if quality is not None:
                quality.sent(t_send, kind)
            pictures += 1
            lights += kind == 'light'
            sent += len(payload)
    except OSError as e:   # the viewer went away (or took nothing for the socket timeout / STALL_S)
        why = '%s%s' % (type(e).__name__, ': %s' % e if isinstance(e, (ConnectionResetError, TimeoutError)) and e.args else '')
    finally:
        with live.jcond:
            live.viewers -= 1
            unregister_locked(live, me)
        log('STREAM %s page %s stream %d: ended after %.1f s, %d pictures (%d lighter), %.1f MB (%s)'
            % (peer, page or '-', n, clock() - t0, pictures, lights, sent / 1e6, why), flush=True)
    return why
