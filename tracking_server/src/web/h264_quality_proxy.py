#!/usr/bin/env python3
"""Opt-in H.264 preview proxy. Does not control camera, tracker, or gimbal."""
import http.server
import os
import queue
import re
import threading
import time
import urllib.request
import cv2
import gi
import numpy as np
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)
SLOT = threading.BoundedSemaphore(2)
BIND = os.environ.get('LWIR_H264_BIND', '127.0.0.1')
PORT = int(os.environ.get('LWIR_H264_PORT', '18773'))
LIVE_PORT = int(os.environ.get('LWIR_LIVE_PORT', '8090'))
ORIGIN = f'http://{BIND}:{LIVE_PORT}'
FRAME_ORIGIN = 'X-LWIR-Frame-Origin'
_FRAME_KEY = re.compile(rb'(?:^| )k=([0-9]+)(?: |$)')


def jpeg_frame_key(jpeg):
    # The stream inserts its LWIR COM before the first JPEG scan.
    pos = 2
    while pos + 4 <= len(jpeg) and jpeg[pos] == 0xff:
        marker = jpeg[pos + 1]
        if marker in (0xda, 0xd9):
            break
        count = int.from_bytes(jpeg[pos + 2:pos + 4], 'big')
        end = pos + count + 2
        if count < 2 or end > len(jpeg):
            break
        if marker == 0xfe and jpeg[pos + 4:pos + 9] == b'LWIR ':
            match = _FRAME_KEY.search(jpeg[pos + 9:end])
            return int(match[1]) if match else None
        pos = end
    return None



class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/video':
            self.send_error(404); return
        if not SLOT.acquire(False):
            self.send_response(503)
            origin = self.headers.get('Origin', '')
            if origin == ORIGIN:
                self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        done = threading.Event()
        chunks = queue.Queue(maxsize=64)
        first_clock = queue.Queue(maxsize=1)
        try:
            pipeline = Gst.parse_launch(
                'appsrc name=source is-live=true format=time block=false '
                'caps="video/x-raw,format=BGR,width=640,height=480,framerate=30/1" '
                '! queue leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 '
                '! videoconvert ! video/x-raw,format=NV12 '
                '! mpph264enc gop=15 bps=4000000 ! h264parse config-interval=-1 '
                '! mp4mux streamable=true fragment-duration=34 trak-timescale=1000000 '
                '! appsink name=sink emit-signals=true sync=false max-buffers=8 drop=false')
            src = pipeline.get_by_name('source')
            sink = pipeline.get_by_name('sink')
            def on_sample(s):
                sample = s.emit('pull-sample')
                if not sample: return Gst.FlowReturn.ERROR
                b = sample.get_buffer()
                data = b.extract_dup(0, b.get_size())
                try: chunks.put(data, timeout=1)
                except queue.Full: done.set(); return Gst.FlowReturn.ERROR
                return Gst.FlowReturn.OK
            sink.connect('new-sample', on_sample)
            pipeline.set_state(Gst.State.PLAYING)

            def feed():
                start = time.monotonic()
                last = -1
                origin_key = None
                buf = bytearray()
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{LIVE_PORT}/stream', timeout=5) as source:
                        while not done.is_set():
                            block = source.read1(4096)
                            if not block: break
                            buf.extend(block)
                            while True:
                                a = buf.find(b'\xff\xd8'); b = buf.find(b'\xff\xd9', a+2) if a >= 0 else -1
                                if b < 0:
                                    if a > 0: del buf[:a]
                                    break
                                jpeg = bytes(buf[a:b+2])
                                del buf[:b+2]
                                img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                if img is None: continue
                                key = jpeg_frame_key(jpeg)
                                first = last < 0
                                if first:
                                    origin_key = key
                                if origin_key is None:
                                    pts = 0 if first else max(last + 1000, int((time.monotonic() - start) * Gst.SECOND))
                                elif key is None:
                                    continue   # a placeholder cannot have a selectable media timestamp
                                else:
                                    pts = (key - origin_key) * 1000
                                    if pts <= last:
                                        continue
                                if img.shape[:2] != (480, 640): img = cv2.resize(img, (640, 480))
                                raw = img.tobytes()
                                frame = Gst.Buffer.new_allocate(None, len(raw), None)
                                frame.fill(0, raw)
                                frame.pts = frame.dts = pts
                                frame.duration = Gst.SECOND // 30
                                if src.emit('push-buffer', frame) != Gst.FlowReturn.OK: return
                                last = pts
                                if first:
                                    first_clock.put_nowait(origin_key)
                except Exception as exc:
                    print('preview feed:', repr(exc), flush=True)
                finally: done.set()
            threading.Thread(target=feed, daemon=True).start()
            try:
                origin_key = first_clock.get(timeout=5)
            except queue.Empty:
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            origin = self.headers.get('Origin', '')
            if origin == ORIGIN:
                self.send_header('Access-Control-Allow-Origin', origin)
            if origin_key is not None:
                self.send_header(FRAME_ORIGIN, str(origin_key))
                self.send_header('Access-Control-Expose-Headers', FRAME_ORIGIN)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            while not done.is_set():
                try: data = chunks.get(timeout=2)
                except queue.Empty: break
                self.wfile.write(data); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print('preview proxy:', repr(exc), flush=True)
        finally:
            done.set()
            if 'pipeline' in locals(): pipeline.set_state(Gst.State.NULL)
            SLOT.release()


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


if __name__ == '__main__':
    Server((BIND, PORT), Handler).serve_forever()
