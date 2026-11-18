"""Direct one-PC Ethernet link for the model-bank live process.

The wire carries length-prefixed JSON requests and a JSON response header plus
optional binary JPEG. No HTTP port or web page is served in this mode. Existing
model-bank command validation is reused in memory.
"""
from __future__ import annotations

import io
import json
import socketserver
import struct
import threading
import time
from email.message import Message
from urllib.parse import urlsplit

from .common.state import APP
from .web import page_handler, stream_sender, picture_first
from .gimbal import live_tracks
from .camera import replay_link

MAX_REQUEST = 16384
MAX_RESPONSE = 8000000
GET_PATHS = {'/time', '/gimbal/people', '/rules', '/rules/auto_det_log', '/perf',
             '/healthz', '/model', '/measure', '/replay', '/replay/eval'}
POST_PATHS = {'/model/set', '/system/restart', '/system/npu-feed', '/system/reboot',
              '/camera/reset', '/tracker/camera', '/rules/set', '/measure/start',
              '/measure/stop', '/measure/note', '/replay/start', '/replay/stop',
              '/replay/seek', '/replay/pause', '/replay/step', '/replay/play',
              '/replay/speed', '/replay/eval/reset', '/replay/eval/draw',
              '/gimbal/click', '/gimbal/move', '/gimbal/stop', '/gimbal/home',
              '/gimbal/ping'}


def read_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        block = sock.recv(count - len(data))
        if not block:
            raise ConnectionError('direct LAN connection ended')
        data.extend(block)
    return bytes(data)


def read_request(sock):
    size = struct.unpack('!I', read_exact(sock, 4))[0]
    if not 0 < size <= MAX_REQUEST:
        raise ValueError('request too large')
    request = json.loads(read_exact(sock, size))
    if not isinstance(request, dict):
        raise ValueError('request must be an object')
    return request


def write_response(sock, status, data=b'', **meta):
    if not isinstance(data, bytes) or len(data) > MAX_RESPONSE:
        raise ValueError('response too large')
    header = json.dumps({'status': status, 'length': len(data), **meta},
                        ensure_ascii=False, separators=(',', ':')).encode()
    sock.sendall(struct.pack('!I', len(header)) + header + data)


def call_existing(method, path, body=b'', peer='127.0.0.1'):
    """Run existing model-bank routes against memory streams, without HTTP I/O."""
    if not isinstance(path, str) or len(path) > 2048:
        return 400, b'{"error":"invalid path"}'
    route = urlsplit(path).path
    allowed = GET_PATHS if method == 'GET' else POST_PATHS if method == 'POST' else set()
    if route not in allowed and not (method == 'GET' and route.startswith('/measure/')):
        return 404, b'{"error":"unknown command"}'
    if not isinstance(body, bytes) or len(body) > 8192:
        return 400, b'{"error":"invalid body"}'
    handler_type = page_handler.make_lv_Handler()
    handler = handler_type.__new__(handler_type)
    handler.path = path
    handler.command = method
    handler.request_version = 'HTTP/1.1'
    handler.requestline = method + ' ' + path + ' HTTP/1.1'
    handler.client_address = (peer, 0)
    handler.close_connection = True
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler._headers_buffer = []
    headers = Message()
    headers['Host'] = 'direct-lan'
    headers['Origin'] = 'http://direct-lan'
    headers['X-Gimbal'] = '1'
    headers['Content-Length'] = str(len(body))
    handler.headers = headers
    if method == 'GET':
        handler.do_GET()
    else:
        handler.do_POST()
    head, sep, payload = handler.wfile.getvalue().partition(b'\r\n\r\n')
    if not sep:
        return 500, b'{"error":"empty response"}'
    try:
        status = int(head.split(b' ', 2)[1])
    except (IndexError, ValueError):
        return 500, b'{"error":"invalid response"}'
    return status, payload


def snapshot(quality='default'):
    live = APP.LIVE
    with live.jcond:
        live.snap_t = time.perf_counter()
        if live.stopped:
            return 503, b'', {}
        jpeg = live.jpeg
        seq = live.seq
        token = '%s:%s' % (seq, vars(live).get('overlay_seq', 0) if picture_first.active(live) else 0)
        sent_perf = time.perf_counter()
        sent_unix = time.time()
        drawn = vars(live).get('jpeg_high_drawn') or vars(live).get('jpeg_drawn')
        frame_t = vars(live).get('jpeg_frame_t')
        people = vars(live).get('jpeg_people')
        ticket = vars(live).get('jpeg_tracks_ticket')
        owners = vars(live).get('jpeg_click_owners') or {}
        overlay = vars(live).get('overlay')
        try:
            overlay_data = json.loads(overlay) if overlay else None
        except (ValueError, TypeError):
            overlay_data = None
        frame_key = stream_sender.frame_key(frame_t) if frame_t is not None else None
        click_t = frame_t
        if picture_first.active(live):
            people, ticket, owners = None, None, {}
            if overlay_data and overlay_data.get('pf') == 1:
                record = next((item for item in reversed(vars(live).get('overlay_clicks') or ())
                               if item[0] == overlay_data.get('k')), None)
                if record:
                    _, click_t, people, ticket, owners = record
                else:
                    overlay_data = None
            else:
                overlay_data = None
        if click_t is not None and people is not None and ticket is not None:
            APP.direct_link.remember(token, click_t, people, ticket, owners)
    if quality == 'high' and drawn is not None and drawn[0] is jpeg:
        make_high = stream_sender.high_maker(APP.cv2)
        if make_high is not None:
            jpeg = make_high(jpeg, drawn[1]) or jpeg
    return 200, jpeg, {'seq': seq, 'token': token, 'overlay': overlay_data,
                       'capture_unix': (sent_unix - (sent_perf - frame_t)) if frame_t is not None else None,
                       'server_ms': (sent_perf - frame_t) * 1000 if frame_t is not None else None,
                       'frame_key': frame_key, 'width': APP.LIVE_W, 'height': APP.LIVE_H}


def click_displayed(token, x, y, peer):
    live = APP.LIVE
    if APP.GB is None:
        return 409, {'answer': '짐벌이 연결되지 않았습니다', 'refused': True}
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return 400, {'answer': '영상 밖 좌표입니다', 'refused': True}
    with live.jcond:
        entry = APP.direct_link.recall(token)
        if entry is None:
            return 409, {'answer': '표시된 프레임이 오래됐습니다', 'refused': True}
        frame_t, people, ticket, owners = entry
        if (time.perf_counter() - frame_t >= 0.5 or live.stopped or
                getattr(live, 'cam_hold', False) or getattr(live, 'cam_lost', False) or
                replay_link.frame_blocked(frame_t)):
            return 409, {'answer': '최신 카메라 화면에서 다시 누르세요', 'refused': True}
        px, py = x * APP.LIVE_W, y * APP.LIVE_H
        hit = live_tracks.unique_hit_id(people, px, py)
        if hit is live_tracks.AMBIGUOUS_CLICK:
            return 409, {'answer': '사람 상자가 겹칩니다', 'refused': True}
        APP.GB.mover_t[peer] = time.monotonic()
        APP.gb_send('P')
        APP.GB.mover = None
        ok, answer = live_tracks.click(px, py, expected_id=hit,
                                      expected_ticket=ticket, shown_frame_t=frame_t,
                                      expected_owner=owners.get(hit) if hit is not None else None,
                                      viewer_ip=peer)
    APP.GB.state['answer'] = answer
    return (200 if ok else 409), dict(APP.GB.state, refused=not ok)


class DirectLink:
    def __init__(self, bind, port, peer):
        self.peer = peer
        self._frames = {}
        self._lock = threading.Lock()
        link = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(5)
                while True:
                    try:
                        request = read_request(self.request)
                        op = request.get('op')
                        if op == 'snapshot':
                            quality = request.get('quality', 'default')
                            if quality not in ('default', 'high'):
                                raise ValueError('invalid picture quality')
                            status, data, meta = snapshot(quality)
                            write_response(self.request, status, data, **meta)
                        elif op == 'click':
                            status, answer = click_displayed(str(request['token']),
                                                             float(request['x']), float(request['y']),
                                                             self.client_address[0])
                            write_response(self.request, status,
                                           json.dumps(answer, ensure_ascii=False).encode())
                        elif op == 'route':
                            body = request.get('body') or {}
                            if not isinstance(body, dict):
                                raise ValueError('body must be an object')
                            status, data = call_existing(request.get('method'),
                                                         request.get('path', ''),
                                                         json.dumps(body, ensure_ascii=False).encode() if body else b'',
                                                         self.client_address[0])
                            write_response(self.request, status, data)
                        else:
                            write_response(self.request, 400, b'{"error":"unknown operation"}')
                    except (ValueError, KeyError, TypeError) as exc:
                        try:
                            write_response(self.request, 400,
                                           json.dumps({'error': str(exc)}, ensure_ascii=False).encode())
                        except OSError: break
                    except (OSError, ConnectionError):
                        break

        class Listener(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
            request_queue_size = 8

            def server_bind(self):
                # The Orange Pi may start before the cable gets link and the
                # dedicated address becomes locally assigned.
                try: self.socket.setsockopt(0, 15, 1)  # Linux IP_FREEBIND
                except OSError: pass
                super().server_bind()

            def verify_request(self, request, client_address):
                return client_address[0] == link.peer

        self.listener = Listener((bind, port), Handler)
        self.thread = threading.Thread(target=self.listener.serve_forever,
                                       name='modelbank-direct-lan', daemon=True)

    def remember(self, seq, frame_t, people, ticket, owners=None):
        with self._lock:
            self._frames[seq] = (frame_t, people, ticket, owners or {})
            for key in list(self._frames):
                if key != seq and (len(self._frames) > 16 or
                                   time.perf_counter() - self._frames[key][0] > 0.5):
                    del self._frames[key]

    def recall(self, seq):
        with self._lock:
            return self._frames.get(seq)

    def start(self):
        self.thread.start()

    def close(self):
        if self.thread.is_alive():
            self.listener.shutdown()
        self.listener.server_close()
