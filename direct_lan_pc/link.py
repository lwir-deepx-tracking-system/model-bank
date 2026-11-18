"""Client for model-bank's direct one-PC Ethernet link (no HTTP)."""
from __future__ import annotations

import json
import socket
import struct
from urllib.parse import urlencode

MAX_RESPONSE = 8000000


class LinkError(Exception):
    pass


def _read(sock, count):
    data = bytearray()
    while len(data) < count:
        block = sock.recv(count - len(data))
        if not block:
            raise LinkError('Orange Pi 연결이 끊겼습니다')
        data.extend(block)
    return bytes(data)


class Link:
    def __init__(self, board='192.168.77.2', port=8765, timeout=3):
        self.board, self.port, self.timeout = board, port, timeout

    @staticmethod
    def exchange(sock, packet):
        raw = json.dumps(packet, ensure_ascii=False, separators=(',', ':')).encode()
        if len(raw) > 16384:
            raise LinkError('요청이 너무 깁니다')
        sock.sendall(struct.pack('!I', len(raw)) + raw)
        size = struct.unpack('!I', _read(sock, 4))[0]
        if not 0 < size <= 16384:
            raise LinkError('응답 머리 크기 오류')
        header = json.loads(_read(sock, size))
        length = header.get('length')
        if not isinstance(length, int) or not 0 <= length <= MAX_RESPONSE:
            raise LinkError('응답 길이 오류')
        body = _read(sock, length)
        return header, body

    def request(self, packet, timeout=None):
        limit = self.timeout if timeout is None else timeout
        with socket.create_connection((self.board, self.port), limit) as sock:
            sock.settimeout(limit)
            return self.exchange(sock, packet)

    def session(self):
        sock = socket.create_connection((self.board, self.port), self.timeout)
        sock.settimeout(self.timeout)
        return sock

    def route(self, method, path, body=None, *, allow_error=False):
        header, raw = self.request({'op': 'route', 'method': method, 'path': path,
                                    'body': body or {}}, timeout=8)
        try:
            result = json.loads(raw)
        except ValueError:
            result = {'error': raw.decode('utf-8', 'replace')[:200]}
        if header['status'] >= 400 and not allow_error:
            raise LinkError(result.get('text') or result.get('answer') or
                            result.get('error') or f"Orange Pi 응답 {header['status']}")
        return result

    def get(self, path):
        return self.route('GET', path)

    def post(self, path, body=None):
        return self.route('POST', path, body)

    def query(self, path, **kwargs):
        return self.post(path + '?' + urlencode(kwargs))

    def snapshot(self, quality='default', session=None):
        packet = {'op': 'snapshot', 'quality': quality}
        return self.exchange(session, packet) if session is not None else self.request(packet)

    def click(self, token, x, y):
        header, raw = self.request({'op': 'click', 'token': token, 'x': x, 'y': y})
        data = json.loads(raw)
        if header['status'] >= 400:
            raise LinkError(data.get('answer') or '사람 선택 실패')
        return data
