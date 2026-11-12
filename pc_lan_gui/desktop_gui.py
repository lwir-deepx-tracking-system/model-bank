"""Desktop host for the unchanged model-bank page, using direct LAN only.

Run on the PC with --board BOARD_IP. No HTTP/MJPEG service or Python process is
started on the Orange Pi. H.264 and metadata enter through the existing UDP
ports; commands use the existing Runtime TCP port.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("WebKit2", "4.0")
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, Gst, GstVideo, Gtk, WebKit2
from PIL import Image, ImageDraw

from desktop_routes import DesktopRoutes
from lan_protocol import ControlLink, ProtocolError, decode_metadata


FETCH_SHIM = r"""
(function () {
  const nativeFetch = window.fetch.bind(window), pending = new Map();
  let nextId = 0;
  window.__lwirReply = function (id, data) {
    const resolve = pending.get(id); if (!resolve) return;
    pending.delete(id);
    resolve(new Response(JSON.stringify(data),
      {status: 200, headers: {'Content-Type': 'application/json'}}));
  };
  window.fetch = function (url, options) {
    if (typeof url !== 'string' || !url.startsWith('/'))
      return nativeFetch(url, options);
    const id = ++nextId;
    return new Promise((resolve, reject) => {
      const signal = options && options.signal;
      if (signal && signal.aborted) {
        reject(new DOMException('Aborted', 'AbortError')); return;
      }
      const onAbort = () => {
        pending.delete(id);
        reject(new DOMException('Aborted', 'AbortError'));
      };
      if (signal) signal.addEventListener('abort', onAbort, {once: true});
      pending.set(id, response => {
        if (signal) signal.removeEventListener('abort', onAbort);
        resolve(response);
      });
      window.webkit.messageHandlers.lwir.postMessage(JSON.stringify({
        id: id, url: url, method: (options && options.method) || 'GET',
        body: (options && options.body) || null
      }));
    });
  };
})();
"""


class DesktopApp:
    def __init__(self, board: str, video_port: int, metadata_port: int,
                 command_port: int):
        self.stop_event = threading.Event()
        self.latest_lock = threading.Lock()
        self.latest_packet: dict = {}
        self.received = 0
        self.video_frames = 0
        self.video_received_at = 0.0
        self.web_ready = False
        self.pending_frame = False
        self.workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lwir-ui")
        self.work_slots = threading.BoundedSemaphore(4)
        self.control = ControlLink(board, command_port)
        self.routes = DesktopRoutes(self.control, self.latest)
        self.metadata_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.metadata_socket.bind(("0.0.0.0", metadata_port))
        self.metadata_socket.settimeout(0.5)

        Gst.init(None)
        description = (
            f'udpsrc port={video_port} caps="application/x-rtp,media=video,'
            'encoding-name=H264,payload=96,clock-rate=90000" '
            '! rtpjitterbuffer latency=100 ! rtph264depay ! h264parse '
            '! avdec_h264 ! videoconvert ! video/x-raw,format=RGB '
            '! appsink name=frames emit-signals=true sync=false max-buffers=1 drop=true'
        )
        self.pipeline = Gst.parse_launch(description)
        sink = self.pipeline.get_by_name("frames")
        sink.connect("new-sample", self._on_sample)

        manager = WebKit2.UserContentManager()
        manager.register_script_message_handler("lwir")
        manager.connect("script-message-received::lwir", self._on_message)
        manager.add_script(WebKit2.UserScript.new(
            FETCH_SHIM, WebKit2.UserContentInjectedFrames.ALL_FRAMES,
            WebKit2.UserScriptInjectionTime.START, None, None))
        self.web = WebKit2.WebView.new_with_user_content_manager(manager)
        self.web.connect("load-changed", self._on_load)
        self.window = Gtk.Window(title="LWIR 추적")
        self.window.set_default_size(1440, 900)
        self.window.add(self.web)
        self.window.connect("destroy", self._close)
        self.web.load_uri((Path(__file__).resolve().parent / "page" /
                           "live_page.html").as_uri())

    def latest(self) -> dict:
        with self.latest_lock:
            return dict(self.latest_packet, received=self.received,
                        video_frames=self.video_frames,
                        video_received_at=self.video_received_at)

    def _receive_metadata(self):
        while not self.stop_event.is_set():
            try:
                packet, _ = self.metadata_socket.recvfrom(65535)
                decoded = decode_metadata(packet)
            except socket.timeout:
                continue
            except (OSError, ProtocolError):
                if self.stop_event.is_set():
                    break
                continue
            with self.latest_lock:
                decoded["received_at"] = time.monotonic()
                self.latest_packet = decoded
                self.received += 1

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample or self.pending_frame or not self.web_ready:
            return Gst.FlowReturn.OK
        caps = sample.get_caps()
        ok, info = GstVideo.VideoInfo().from_caps(caps)
        if not ok:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            raw = bytes(mapped.data)
            image = Image.frombytes("RGB", (info.width, info.height), raw,
                                    "raw", "RGB", info.stride[0])
        finally:
            buffer.unmap(mapped)
        # The page's original MJPEG path drew boxes on the stream. Keep that
        # presentation, using the newest metadata packet available on the PC.
        metadata = self.latest()
        if metadata.get("tracks"):
            draw = ImageDraw.Draw(image)
            for track in metadata["tracks"]:
                box = (track["x"], track["y"],
                       track["x"] + track["width"],
                       track["y"] + track["height"])
                draw.rectangle(box, outline="#23df72", width=2)
                draw.text((track["x"], max(0, track["y"] - 13)),
                          str(track["id"]), fill="#23df72")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=80)
        url = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()
        self.pending_frame = True
        GLib.idle_add(self._display_frame, url)
        return Gst.FlowReturn.OK

    def _display_frame(self, url):
        self.web.run_javascript(
            "if (window.openStream) window.openStream=function(){};"
            "var img=document.getElementById('v');"
            "if(img) img.src=" + json.dumps(url) + ";", None, None, None)
        with self.latest_lock:
            self.video_received_at = time.monotonic()
            self.video_frames += 1
        self.pending_frame = False
        return False

    def _on_load(self, _view, event):
        if event == WebKit2.LoadEvent.FINISHED:
            self.web_ready = True
            self.web.run_javascript(
                "if (window.openStream) window.openStream=function(){};",
                None, None, None)

    def _on_message(self, _manager, result):
        try:
            message = json.loads(result.get_js_value().to_string())
        except (ValueError, AttributeError):
            return

        if not self.work_slots.acquire(blocking=False):
            self._reply(message["id"], dict(ok=False, text="Runtime 명령 처리 중입니다"))
            return

        def work():
            try:
                answer = self.routes.handle(message["url"], message["method"],
                                            message.get("body"))
                if os.environ.get("LWIR_GUI_TRACE"):
                    print("GUI route", message["url"], answer, flush=True)
            except Exception as exc:
                print("GUI route error", message.get("url"), repr(exc), flush=True)
                answer = dict(ok=False, text="GUI 요청 처리 오류")
            finally:
                self.work_slots.release()
            GLib.idle_add(self._reply, message["id"], answer)

        self.workers.submit(work)

    def _reply(self, ident, answer):
        self.web.run_javascript(
            "window.__lwirReply(" + str(int(ident)) + "," +
            json.dumps(answer, ensure_ascii=False) + ");", None, None, None)
        return False

    def _close(self, *_args):
        self.stop_event.set()
        self.metadata_socket.close()
        self.pipeline.set_state(Gst.State.NULL)
        self.control.close()
        self.workers.shutdown(wait=False, cancel_futures=True)
        Gtk.main_quit()

    def run(self):
        threading.Thread(target=self._receive_metadata, daemon=True).start()
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("H.264 RTP receiver could not start")
        self.window.show_all()
        Gtk.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True, help="Runtime LAN IPv4 address")
    parser.add_argument("--video-port", type=int, default=5000)
    parser.add_argument("--metadata-port", type=int, default=5001)
    parser.add_argument("--command-port", type=int, default=5002)
    options = parser.parse_args()
    DesktopApp(options.board, options.video_port, options.metadata_port,
               options.command_port).run()


if __name__ == "__main__":
    main()
