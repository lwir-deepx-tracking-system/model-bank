"""Captive-portal responder for the direct cable port (HTTP, stdlib only).

The DHCP/DNS side (scripts/net/direct-dnsmasq.conf) answers the connectivity-check host names of the operating systems with
the board's direct-link address. Those checks arrive here (port 80 is redirected to this listener by an nft rule that only
matches the direct interface). What a check gets back decides what the PC does:

  not yet shown      -> HTTP 302 to the page (http://<board address>:8090/): the OS calls the network a captive portal and
                        opens (or offers to open) a browser on the page
  shown, < SILENCE_S -> the normal success answer of that check: the OS stops asking for 10 minutes
  Apple              -> never the success answer by default: the Captive Network Assistant window closes by itself on success,
                        it would vanish at once
  switch off         -> the file ~/.lwir_portal_off (or LWIR_PORTAL_MODE=off): every check gets its success answer, no PC is sent
                        to the page by its operating system (DHCP, the fixed address, mDNS and the PC program still work)

Any other request (the browser Windows opens at www.msftconnecttest.com/redirect, a typed bare address, ...) is a visit:
it is answered with the 302 and the client counts as 'shown'. The table is cleared when the cable is re-plugged and for a
client whose MAC address changed.
"""
import http.server
import json
import threading
import time

from .netutil import make_freebind_server

APPLE_SUCCESS = b'<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>'
TEXT = 'text/plain'
HTML = 'text/html'

# host -> {path: (status, body, content type)} ; None as path = any path
CHECKS = {
    'www.msftconnecttest.com': {'/connecttest.txt': (200, b'Microsoft Connect Test', TEXT)},
    'www.msftncsi.com': {'/ncsi.txt': (200, b'Microsoft NCSI', TEXT)},
    'captive.apple.com': {'/hotspot-detect.html': (200, APPLE_SUCCESS, HTML), '/library/test/success.html': (200, APPLE_SUCCESS, HTML)},
    'connectivitycheck.gstatic.com': {'/generate_204': (204, b'', TEXT)},
    'connectivitycheck.android.com': {'/generate_204': (204, b'', TEXT)},
    'clients3.google.com': {'/generate_204': (204, b'', TEXT)},
    'edge-http.microsoft.com': {'/captiveportal/generate_204': (204, b'', TEXT)},
    'detectportal.firefox.com': {'/success.txt': (200, b'success\n', TEXT), '/canonical.html': (200, b'success\n', TEXT)},
    'nmcheck.gnome.org': {'/check_network_status.txt': (200, b'NetworkManager is online\n', TEXT)},
    'connectivity-check.ubuntu.com': {'/': (204, b'', TEXT)},
}
APPLE_HOSTS = ('captive.apple.com',)


def lookup_check(host, path):
    """(status, body, content type) of the success answer for a known check, else None."""
    entry = CHECKS.get((host or '').split(':')[0].lower())
    if entry is None:
        return None
    return entry.get(path.split('?', 1)[0])


class ClientTable:
    """Which clients have been shown the page, and when (by IP; MAC remembered to notice a different PC on the same IP)."""

    def __init__(self, silence_s=600.0, clock=time.monotonic, limit=512):
        self.silence_s, self.clock, self.limit = silence_s, clock, limit
        self.shown_at, self.macs = {}, {}
        self.lock = threading.Lock()

    def silenced(self, ip):
        with self.lock:
            then = self.shown_at.get(ip)
            return then is not None and self.clock() - then < self.silence_s

    def mark_shown(self, ip):
        with self.lock:
            if len(self.shown_at) >= self.limit and ip not in self.shown_at:
                self.shown_at.pop(min(self.shown_at, key=self.shown_at.get))
            self.shown_at[ip] = self.clock()

    def reset(self, ip=None):
        with self.lock:
            if ip is None:
                self.shown_at.clear()
                self.macs.clear()
            else:
                self.shown_at.pop(ip, None)
                self.macs.pop(ip, None)

    def neighbours(self, arp):
        """Feed {ip: mac}: a client whose MAC changed is a different machine: show the page again."""
        changed = []
        with self.lock:
            for ip, mac in arp.items():
                if ip in self.shown_at and self.macs.get(ip, mac) != mac:
                    self.shown_at.pop(ip, None)
                    changed.append(ip)
                self.macs[ip] = mac
        return changed

    def snapshot(self):
        with self.lock:
            now = self.clock()
            return {ip: round(now - then) for ip, then in self.shown_at.items()}


def make_handler(table, page_port=8090, apple_silence=False, log=None, is_off=None):
    class PortalHandler(http.server.BaseHTTPRequestHandler):
        server_version = 'lwir-portal'
        sys_version = ''
        protocol_version = 'HTTP/1.0'

        def log_message(self, fmt, *args):
            if log is not None:
                log('portal %s %s' % (self.client_address[0], fmt % args))

        def _answer(self, status, body, ctype, location=None, head_only=False):
            self.send_response(status)
            if location:
                self.send_header('Location', location)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            if body and not head_only:
                self.wfile.write(body)

        def _handle(self, head_only=False):
            ip = self.client_address[0]
            host = (self.headers.get('Host') or '').split(':')[0].lower()
            path = self.path
            if '://' in path:   # absolute-form request line (proxy style)
                path = '/' + path.split('://', 1)[1].partition('/')[2]
            if path.split('?', 1)[0] == '/lwir-status':   # for the PC connect watcher: has the OS already shown this PC the page?
                body = json.dumps(dict(lwir=1, shown=table.silenced(ip))).encode()
                return self._answer(200, body, 'application/json', head_only=head_only)
            known = lookup_check(host, path)
            if known is None:
                table.mark_shown(ip)   # a visit (browser opened by the OS, bare address typed): count it as shown
            elif (is_off is not None and is_off()) or ((host not in APPLE_HOSTS or apple_silence) and table.silenced(ip)):   # is_off: the switch, probes always get the success answer
                status, body, ctype = known
                return self._answer(status, body, ctype, head_only=head_only)
            # the redirect goes to the address this client used to reach us
            here = self.connection.getsockname()[0]
            target = 'http://%s:%d/' % (here, page_port)
            body = ('<html><head><meta http-equiv="refresh" content="0;url=%s"></head><body><a href="%s">LWIR</a></body></html>' % (target, target)).encode()
            return self._answer(302, body, HTML, location=target, head_only=head_only)

        def do_GET(self):
            self._handle()

        def do_HEAD(self):
            self._handle(head_only=True)

        def do_POST(self):
            self._handle()
    return PortalHandler


def start_portal(address, port, table, page_port=8090, allowed_net=None, apple_silence=False, log=None, is_off=None):
    """Start the responder on address:port in a daemon thread. Bound with IP_FREEBIND: works before the address exists."""
    server_class = make_freebind_server(http.server.ThreadingHTTPServer, allowed_net)
    server = server_class((address, port), make_handler(table, page_port, apple_silence, log, is_off))
    thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=0.5), daemon=True, name='portal')
    thread.start()
    return server, thread


def describe():
    """The probe answers as JSON (for the docs and tests)."""
    return json.dumps({h: sorted(p) for h, p in CHECKS.items()}, indent=1)
