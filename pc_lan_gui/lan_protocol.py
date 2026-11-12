"""Direct LAN client for the existing LWIR TCP/UDP wire contract.

This module does not start a listener or a web service. The board remains the
only command endpoint. Calls are serialized because the wire format has no
version or per-session multiplexing outside Pipeline request IDs.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
import math
from dataclasses import dataclass

MAGIC = b"LWIR"
HEADER = struct.Struct(">4sBBH")
STATE = struct.Struct(">BBBBI")
TRACK = struct.Struct(">IffffBf")
METADATA = struct.Struct(">4sQIQI")
TELEMETRY = struct.Struct(">4sQQQQIfII18f")
TRACKER_POLICY = struct.Struct(">IIIIfIII")
CAT_SYSTEM = 1
CAT_MODE = 2
CAT_GIMBAL = 3
CAT_PIPELINE = 4


class ProtocolError(RuntimeError):
    pass


def encode(category: int, command: int, payload: bytes = b"") -> bytes:
    if len(payload) > 65535:
        raise ValueError("payload too large")
    return HEADER.pack(MAGIC, category, command, len(payload)) + payload


def decode_metadata(packet: bytes) -> dict:
    if len(packet) < METADATA.size:
        raise ProtocolError("short metadata")
    magic, frame_id, rtp, gui_us, count = METADATA.unpack_from(packet)
    base = METADATA.size + count * TRACK.size
    if magic != MAGIC or frame_id == 0 or len(packet) not in (base, base + TELEMETRY.size):
        raise ProtocolError("invalid metadata header")
    telemetry = None
    if len(packet) == base + TELEMETRY.size:
        fields = TELEMETRY.unpack_from(packet, base)
        if fields[0] != b"LWTM" or not fields[1] or not 0 < fields[4] < frame_id or \
                not fields[7] or not fields[8] or fields[5] & ~((1 << 18) - 1) or \
                not math.isfinite(fields[6]) or fields[6] < 0 or \
                any(not math.isfinite(value) or value < 0 for index, value in
                    enumerate(fields[9:]) if fields[5] & (1 << index)):
            raise ProtocolError("invalid telemetry footer")
        telemetry = dict(session_id=fields[1], model_generation=fields[2],
                         tracker_epoch=fields[3], frame_id=fields[4],
                         valid_mask=fields[5], sample_age_ms=fields[6],
                         model_id=fields[7], tracker_id=fields[8], values=fields[9:])
    tracks = []
    for i in range(count):
        ident, x, y, width, height, class_id, confidence = TRACK.unpack_from(
            packet, METADATA.size + i * TRACK.size
        )
        if ident and width > 0 and height > 0 and math.isfinite(confidence) and \
                0 <= confidence <= 1 and all(math.isfinite(v) for v in (x, y, width, height)):
            tracks.append(dict(id=ident, x=x, y=y, width=width, height=height,
                               class_id=class_id, confidence=confidence))
    return dict(frame_id=frame_id, rtp_timestamp=rtp, gui_started_us=gui_us,
                track_count=count, tracks=tracks, telemetry=telemetry)


@dataclass
class BoardState:
    mode: int = 0
    target: int = 0
    gimbal: int = 0
    error: int = 0
    selected_id: int = 0


class ControlLink:
    def __init__(self, host: str, port: int = 5002, timeout: float = 2.0):
        self.host, self.port, self.timeout = host, port, timeout
        self._socket: socket.socket | None = None
        self._lock = threading.RLock()
        self._request_id = 0
        self.state = BoardState()
        self._catalog: dict[str, tuple[float, list[dict]]] = {}

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            self._catalog.clear()

    def _connect(self) -> socket.socket:
        if self._socket is None:
            self._socket = socket.create_connection((self.host, self.port), self.timeout)
            self._socket.settimeout(self.timeout)
            self._catalog.clear()
        return self._socket

    def _read_exact(self, length: int) -> bytes:
        result = bytearray()
        while len(result) < length:
            part = self._connect().recv(length - len(result))
            if not part:
                raise ConnectionError("board closed the control connection")
            result.extend(part)
        return bytes(result)

    def _read(self) -> tuple[int, int, bytes]:
        magic, category, command, length = HEADER.unpack(self._read_exact(HEADER.size))
        if magic != MAGIC or length > 4096:
            raise ProtocolError("invalid control frame")
        return category, command, self._read_exact(length)

    def _next_id(self) -> int:
        self._request_id = (self._request_id + 1) & 0xFFFFFFFF or 1
        return self._request_id

    def _wait_for(self, category: int, command: int, request_id: int | None = None,
                  timeout: float | None = None):
        end = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < end:
            remaining = max(0.01, end - time.monotonic())
            self._connect().settimeout(remaining)
            try:
                cat, cmd, payload = self._read()
            except socket.timeout:
                break
            if cat == CAT_SYSTEM and cmd in (2, 3) and len(payload) == STATE.size:
                self.state = BoardState(*STATE.unpack(payload))
                continue
            if cat != category or cmd != command:
                raise ProtocolError(f"unexpected response {cat}/{cmd}")
            if request_id is not None and (len(payload) < 4 or
                                            struct.unpack_from(">I", payload)[0] != request_id):
                raise ProtocolError("request ID mismatch")
            return payload
        raise TimeoutError(f"board did not answer {category}/{command}")

    def _request(self, category: int, command: int, payload: bytes,
                 response_command: int, request_id: int | None = None) -> bytes:
        with self._lock:
            try:
                self._connect().sendall(encode(category, command, payload))
                return self._wait_for(category if category == CAT_PIPELINE else CAT_SYSTEM,
                                      response_command, request_id)
            except (OSError, ProtocolError, TimeoutError):
                self.close()
                raise

    def ping(self) -> BoardState:
        payload = self._request(CAT_SYSTEM, 4, b"", 1)
        if payload != bytes((CAT_SYSTEM, 4, 0)):
            raise ProtocolError("PING rejected")
        return self.state

    def catalog(self, kind: str, *, refresh: bool = False) -> list[dict]:
        if kind not in ("model", "tracker"):
            raise ValueError(kind)
        with self._lock:
            cached = self._catalog.get(kind)
            if cached and not refresh and time.monotonic() - cached[0] < 2:
                return cached[1]
            count_cmd, info_cmd = (1, 5) if kind == "model" else (2, 6)
            request_id = self._next_id()
            answer = self._request(CAT_PIPELINE, count_cmd,
                                   struct.pack(">I", request_id), count_cmd | 0x80, request_id)
            count = struct.unpack_from(">I", answer, 4)[0]
            if count > 256:
                raise ProtocolError("catalog count is unreasonable")
            items = []
            for ident in range(1, count + 1):
                request_id = self._next_id()
                answer = self._request(CAT_PIPELINE, info_cmd,
                                       struct.pack(">II", request_id, ident), info_cmd | 0x80,
                                       request_id)
                if len(answer) != 76:
                    raise ProtocolError("invalid catalog item")
                item_id, flags = struct.unpack_from(">II", answer, 4)
                name = answer[12:76].split(b"\0", 1)[0].decode("utf-8", "replace")
                items.append(dict(id=item_id, name=name, available=bool(flags & 1),
                                  active=bool(flags & 2)))
            self._catalog[kind] = time.monotonic(), items
            return items

    def change(self, kind: str, ident: int) -> int:
        if kind not in ("model", "tracker") or ident < 1:
            raise ValueError("invalid catalog request")
        command = 3 if kind == "model" else 4
        with self._lock:
            try:
                request_id = self._next_id()
                self._connect().sendall(encode(CAT_PIPELINE, command,
                                               struct.pack(">II", request_id, ident)))
                while True:
                    payload = self._wait_for(CAT_PIPELINE, command | 0x80, request_id,
                                             timeout=30.0)
                    if len(payload) != 6:
                        raise ProtocolError("invalid change result")
                    _, phase, result = struct.unpack(">IBB", payload)
                    if phase == 2:
                        self._catalog.pop(kind, None)
                        return result
                    if phase != 1 or result:
                        raise ProtocolError("invalid accepted result")
            except (OSError, ProtocolError, TimeoutError):
                self.close()
                raise

    def tracker_policy(self) -> dict:
        """Read the active Runtime policy; this command cannot change settings."""
        with self._lock:
            request_id = self._next_id()
            answer = self._request(CAT_PIPELINE, 7, struct.pack(">I", request_id),
                                   0x87, request_id)
            if len(answer) != TRACKER_POLICY.size:
                raise ProtocolError("invalid tracker policy")
            _, tracker_id, warmth_min, warmth_max, edge_birth, frames, ms, flags = \
                TRACKER_POLICY.unpack(answer)
            if flags & ~7 or not math.isfinite(edge_birth):
                raise ProtocolError("invalid tracker policy fields")
            return dict(tracker_id=tracker_id, warmth_min=warmth_min,
                        warmth_max=warmth_max, edge_birth=edge_birth,
                        confirmation_frames=frames, confirmation_ms=ms,
                        class_gate=bool(flags & 1), gmc=bool(flags & 2),
                        post_reid=bool(flags & 4))

    def change_policy(self, **values) -> int:
        """Replace the active tracker after the Runtime completes a policy change."""
        bits = dict(warmth_min=1, warmth_max=2, edge_birth=4,
                    confirmation_ms=8, gmc=16, class_gate=32)
        if not values or set(values) - set(bits):
            raise ValueError("invalid policy fields")
        mask = sum(bits[key] for key in values)
        minimum = int(values.get("warmth_min", 0))
        maximum = int(values.get("warmth_max", 0))
        edge = float(values.get("edge_birth", 0))
        confirmation = int(values.get("confirmation_ms", 0))
        if not (0 <= minimum <= 65535 and 0 <= maximum <= 65535 and
                0 <= edge <= 1 and math.isfinite(edge) and
                0 <= confirmation <= 1000):
            raise ValueError("policy value out of range")
        flags = int(bool(values.get("gmc", False))) | \
                (int(bool(values.get("class_gate", False))) << 1)
        with self._lock:
            try:
                request_id = self._next_id()
                payload = struct.pack(">IIIIfII", request_id, mask, minimum,
                                      maximum, edge, confirmation, flags)
                self._connect().sendall(encode(CAT_PIPELINE, 8, payload))
                while True:
                    answer = self._wait_for(CAT_PIPELINE, 0x88, request_id,
                                            timeout=30.0)
                    if len(answer) != 6:
                        raise ProtocolError("invalid policy result")
                    _, phase, result = struct.unpack(">IBB", answer)
                    if phase == 2:
                        return result
                    if phase != 1 or result:
                        raise ProtocolError("invalid policy acceptance")
            except (OSError, ProtocolError, TimeoutError):
                self.close()
                raise

    def mode(self, target_id: int | None) -> int:
        command = 1 if target_id is not None else 2
        payload = struct.pack(">I", target_id) if target_id is not None else b""
        answer = self._request(CAT_MODE, command, payload, 1)
        if len(answer) != 3 or answer[:2] != bytes((CAT_MODE, command)):
            raise ProtocolError("invalid mode ACK")
        return answer[2]

    def move(self, x: int, y: int) -> None:
        if x not in (-1, 0, 1) or y not in (-1, 0, 1):
            raise ValueError("manual direction out of range")
        with self._lock:
            self._connect().sendall(encode(CAT_GIMBAL, 1, struct.pack("bb", x, y)))

    def home(self) -> int:
        answer = self._request(CAT_GIMBAL, 3, b"", 1)
        if len(answer) != 3 or answer[:2] != bytes((CAT_GIMBAL, 3)):
            raise ProtocolError("invalid HOME ACK")
        return answer[2]

    def position(self) -> dict | None:
        """Display-only command estimate; None means no fresh live position."""
        with self._lock:
            try:
                self._connect().sendall(encode(CAT_GIMBAL, 4))
                answer = self._wait_for(CAT_GIMBAL, 0x84)
                if len(answer) != 9:
                    raise ProtocolError("invalid gimbal position")
                valid, pan, tilt = struct.unpack(">Bff", answer)
                if valid not in (0, 1) or not all(math.isfinite(v) and 0 <= v <= 100
                                                  for v in (pan, tilt)):
                    raise ProtocolError("invalid gimbal position fields")
                return dict(pan=pan, tilt=tilt) if valid else None
            except (OSError, ProtocolError, TimeoutError):
                self.close()
                raise
