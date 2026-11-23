"""Stand-in for the tracking server's page in the namespace test: the real src/web/direct_listeners.py (listeners, netinfo, PC download)
with a tiny handler. Run from tracking_server/:  python3 -B scripts/net/netns/page_stub.py"""
import http.server
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src.web import direct_listeners as dl   # noqa: E402


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/netinfo':
            body, ctype, status, name = dl.netinfo_body(self.client_address[0]), 'application/json', 200, None
        elif path == '/healthz':
            body, ctype, status, name = json.dumps(dict(up=5)).encode(), 'application/json', 200, None
        elif path.startswith('/pc/'):
            status, ctype, name, body = dl.pc_download(path, self.client_address[0])
        else:
            body, ctype, status, name = b'<html>page</html>', 'text/html', 200, None
        if status != 200:
            self.send_error(status)
            return
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        if name:
            self.send_header('Content-Disposition', 'attachment; filename="%s"' % name)
        self.end_headers()
        self.wfile.write(body)


started = dl.start(Handler, int(os.environ.get('PAGE_PORT', '8090')), os.environ.get('SPECS'), log=lambda m: print(m, flush=True))
print('started', started, flush=True)
while True:
    time.sleep(1)
