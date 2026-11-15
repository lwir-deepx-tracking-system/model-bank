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
"""
import os
import re
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


def picture_comment(seq, frame_t, now_perf, now_unix):
    """'LWIR n=<seq> t=<frame arrival> s=<now>' in unix seconds; frame_t is a time.perf_counter() value or None."""
    if frame_t is None:
        return b'LWIR n=%d s=%.3f' % (seq, now_unix)
    return b'LWIR n=%d t=%.3f s=%.3f' % (seq, now_unix - (now_perf - frame_t), now_unix)


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


def serve(live, conn, write, query='', peer='?', *, limit=..., state=link_state, clock=time.monotonic,
          sleep=time.sleep, perf=time.perf_counter, unix=time.time, log=print):
    """Send pictures to one viewer until it goes away, the server stops or a newer stream of the same page opens.
    live: APP.LIVE (jcond, seq, jpeg, jpeg_frame_t, stopped, viewers). write: the connection's blocking write.
    limit: bytes allowed in the socket queue before the next picture (default LWIR_STREAM_OUTQ; None = no wait).
    state(conn) -> (queued bytes, connected). Returns why it ended."""
    limit = OUTQ_LIMIT if limit is ... else limit
    page, n = page_of(query)
    me = Viewer(page, n, peer)
    t0 = clock()
    pictures = sent = 0
    why = 'server stopped'
    write(FIRST_PART)
    with live.jcond:
        live.viewers += 1   # 11-12: pictures are made for every frame while this is above 0
        old = register_locked(live, me)
        viewers = live.viewers
    log('STREAM %s page %s stream %d: open, %d viewers%s'
        % (peer, page or '-', n, viewers, ', replaces stream %d' % old.n if old is not None else ''), flush=True)

    def newest_locked():
        return live.seq, live.jpeg, getattr(live, 'jpeg_frame_t', None)
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
                last, jb, frame_t = newest_locked()
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
                    sleep(0.01)
                with live.jcond:
                    if me.closed is not None:
                        why = me.closed
                        break
                    if live.stopped:
                        break
                    if live.jpeg:
                        last, jb, frame_t = newest_locked()
            write(with_comment(jb, picture_comment(last, frame_t, perf(), unix())) + NEXT_PART)
            pictures += 1
            sent += len(jb)
    except OSError as e:   # the viewer went away (or took nothing for the socket timeout / STALL_S)
        why = '%s%s' % (type(e).__name__, ': %s' % e if isinstance(e, (ConnectionResetError, TimeoutError)) and e.args else '')
    finally:
        with live.jcond:
            live.viewers -= 1
            unregister_locked(live, me)
        log('STREAM %s page %s stream %d: ended after %.1f s, %d pictures, %.1f MB (%s)'
            % (peer, page or '-', n, clock() - t0, pictures, sent / 1e6, why), flush=True)
    return why
