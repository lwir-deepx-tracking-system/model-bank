#!/usr/bin/env python3
"""Opt-in H.264 preview proxy. Does not control camera, tracker, or gimbal."""
import http.server
import os
import queue
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
        try:
            pipeline = Gst.parse_launch(
                'appsrc name=source is-live=true format=time block=false '
                'caps="video/x-raw,format=BGR,width=640,height=480,framerate=30/1" '
                '! queue leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 '
                '! videoconvert ! video/x-raw,format=NV12 '
                '! mpph264enc gop=15 bps=4000000 ! h264parse config-interval=-1 '
                '! mp4mux streamable=true fragment-duration=34 '
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
                start = time.monotonic(); last = 0; buf = bytearray()
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{LIVE_PORT}/stream', timeout=5) as source:
                        while not done.is_set():
                            block = source.read(4096)
                            if not block: break
                            buf.extend(block)
                            while True:
                                a = buf.find(b'\xff\xd8'); b = buf.find(b'\xff\xd9', a+2) if a >= 0 else -1
                                if b < 0:
                                    if a > 0: del buf[:a]
                                    break
                                img = cv2.imdecode(np.frombuffer(bytes(buf[a:b+2]), dtype=np.uint8), cv2.IMREAD_COLOR)
                                del buf[:b+2]
                                if img is None: continue
                                if img.shape[:2] != (480, 640): img = cv2.resize(img, (640, 480))
                                raw = img.tobytes(); frame = Gst.Buffer.new_allocate(None, len(raw), None); frame.fill(0, raw)
                                pts = max(last+1, int((time.monotonic()-start)*Gst.SECOND)); last=pts
                                frame.pts=frame.dts=pts; frame.duration=Gst.SECOND//30
                                if src.emit('push-buffer', frame) != Gst.FlowReturn.OK: return
                except Exception as exc:
                    print('preview feed:', repr(exc), flush=True)
                finally: done.set()
            threading.Thread(target=feed, daemon=True).start()
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            origin = self.headers.get('Origin', '')
            if origin == ORIGIN:
                self.send_header('Access-Control-Allow-Origin', origin)
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
