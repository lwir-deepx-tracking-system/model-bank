# -*- coding: utf-8 -*-
"""HTTP 요청을 받아 주소별로 나눠 주는 부분.

GET   /                목록 화면
      /edit            편집 화면
      /api/list        시퀀스 목록 (마지막으로 훑은 지 20초가 지났으면 라벨 폴더를 다시 훑는다)
      /api/meta        한 시퀀스의 상태. 이름을 함께 보내면 "지금 열어 둔 사람"으로 90초 동안 기억한다
      /data/<시퀀스>/  RESULT.json, images/000001.png, tracks_mot.txt, tracks_edited_mot.txt, review_state.json
POST  /api/save, /api/purge, /api/seq_trash, /api/seq_rename   (하는 일은 file_ops.py)
접속 키: 주소에 ?k=<키> 를 붙여 한 번 열면 쿠키가 심기고, 그 뒤로는 키 없이 열린다.
"""
import json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

from .state import IMG_RE, LOCK, PRESENCE, READ_NAMES, RESCAN, SEQS
from .sequences import scan, seq_info
from .list_page import INDEX
from .editor_page import build_editor
from .file_ops import FileOps


class H(FileOps, BaseHTTPRequestHandler):
    server_version = 'lwirlabel/1'
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *a):
        pass

    def send(self, code, body, ctype='application/json; charset=utf-8', extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def authed(self, q):
        tok = self.server.token
        if not tok:
            return True, None
        ck = self.headers.get('Cookie', '')
        if ('lwirk=' + tok) in ck:
            return True, None
        if q.get('k', [''])[0] == tok:
            return True, 'lwirk=%s; Path=/; Max-Age=2592000; HttpOnly; SameSite=Lax' % tok
        return False, None

    def seq_of(self, q):
        rel = q.get('seq', [''])[0]
        return rel if rel in SEQS else None

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        ok, cookie = self.authed(q)
        if not ok:
            return self.send(403, '접근 키가 필요합니다. 받은 주소 전체(?k=... 포함)로 다시 여세요.', 'text/plain; charset=utf-8')
        if cookie:
            self.send_response(302)
            self.send_header('Set-Cookie', cookie)
            self.send_header('Location', '/')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        p = unquote(u.path)
        if p == '/':
            return self.send(200, INDEX, 'text/html; charset=utf-8', {'Cache-Control': 'no-store'})
        if p == '/edit':
            try:
                return self.send(200, build_editor(self.server.root), 'text/html; charset=utf-8', {'Cache-Control': 'no-store'})
            except Exception as e:
                return self.send(500, {'error': str(e)})
        if p == '/api/list':
            with LOCK:
                if time.time() - RESCAN['t'] > 20:      # pick up sequences added after start
                    RESCAN['t'] = time.time()
                    try:
                        new = {r: v for r, v in scan(self.server.root).items() if r.startswith(RESCAN['only'])}
                        SEQS.clear(); SEQS.update(new)
                    except Exception:
                        pass
                return self.send(200, [seq_info(r) for r in SEQS], extra={'Cache-Control': 'no-store'})
        if p == '/api/meta':
            rel = self.seq_of(q)
            if not rel:
                return self.send(404, {'error': 'unknown sequence'})
            user = q.get('user', [''])[0][:20]
            with LOCK:
                if user:
                    PRESENCE.setdefault(rel, {})[user] = time.time()
                return self.send(200, seq_info(rel), extra={'Cache-Control': 'no-store'})
        if p.startswith('/data/'):
            rest = p[len('/data/'):]
            for rel, s in SEQS.items():
                if rest.startswith(rel + '/'):
                    name = rest[len(rel) + 1:]
                    if name == 'RESULT.json':
                        return self.send(200, {'frames': s['frames']}, extra={'Cache-Control': 'no-store'})
                    if IMG_RE.match(name):
                        fp = os.path.join(s['dir'], 'images', name[7:])
                        if os.path.isfile(fp):
                            return self.send(200, open(fp, 'rb').read(), 'image/png', {'Cache-Control': 'private, max-age=86400'})
                    elif name in READ_NAMES:
                        fp = os.path.join(s['dir'], name)
                        if os.path.isfile(fp):
                            return self.send(200, open(fp, 'rb').read(), 'text/plain; charset=utf-8', {'Cache-Control': 'no-store'})
                    break
            return self.send(404, {'error': 'not found'})
        return self.send(404, {'error': 'not found'})

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        ok, _ = self.authed(q)
        if not ok:
            return self.send(403, {'error': 'no key'})
        if u.path == '/api/purge':
            return self.purge(q)
        if u.path in ('/api/seq_trash', '/api/seq_rename'):
            return self.seq_admin(u.path, q)
        if u.path != '/api/save':
            return self.send(404, {'error': 'not found'})
        return self.save(q)


class Srv(ThreadingHTTPServer):
    daemon_threads = True
