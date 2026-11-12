"""Map the unchanged model-bank page's local calls to the Runtime LAN protocol.

No HTTP endpoint is started. Unsupported page actions fail explicitly.
"""

from __future__ import annotations

import time
from collections import deque
from urllib.parse import parse_qs, urlsplit


class DesktopRoutes:
    def __init__(self, control, latest):
        self.control = control
        self.latest = latest
        self.started = time.monotonic()
        self.previous_warmth = (0, 0)
        self.previous_edge = 0.0
        self.perf_rows = deque(maxlen=900)
        self.perf_last = None

    def _state(self, answer: str = "") -> dict:
        state = self.control.ping()
        latest = self.latest()
        now = time.monotonic()
        live = (now - latest.get("received_at", 0) < 1.5 and
                now - latest.get("video_received_at", 0) < 1.5)
        return dict(running=True, ready=state.error == 0, error=state.error,
                    auto=state.mode == 1, moving=state.gimbal == 1,
                    lost=state.target == 2, id=state.selected_id,
                    pose=None,
                    position_estimate=self.control.position() if live else None,
                    answer=answer)

    def _model(self) -> dict:
        catalog = self.control.catalog("model")
        return dict(catalog=[dict(id=x["id"], det=x["name"], name=x["name"],
                                  available=x["available"]) for x in catalog],
                    current=next((x["id"] for x in catalog if x["active"]), 0),
                    busy=False)

    def _rules(self) -> dict:
        catalog = self.control.catalog("tracker")
        policy = self.control.tracker_policy()
        if policy["warmth_min"] or policy["warmth_max"]:
            self.previous_warmth = (policy["warmth_min"], policy["warmth_max"])
        if policy["edge_birth"] > 0:
            self.previous_edge = policy["edge_birth"]
        can_relabel = policy["tracker_id"] in (1, 3) and any(
            x["id"] == 3 and x["available"] for x in catalog)
        return dict(presets=[dict(key=str(x["id"]), name=x["name"],
                                  available=x["available"]) for x in catalog],
                    tracker=str(policy["tracker_id"]), relabel=policy["post_reid"],
                    relabel_available=can_relabel, stab=policy["gmc"],
                    warm_min=policy["warmth_min"], warm_max=policy["warmth_max"],
                    warm_on=bool(policy["warmth_min"] or policy["warmth_max"]),
                    warm_prev=list(self.previous_warmth),
                    edge_birth=policy["edge_birth"], edge_prev=self.previous_edge,
                    confirm=dict(ms=policy["confirmation_ms"],
                                 frames=policy["confirmation_frames"], fps=0),
                    predictors=[dict(key="kalman", available=True)],
                    predictor="kalman", apps=[], reids=[], reid_hot=False,
                    pending=None)

    def _perf(self, since: float) -> dict:
        """Observed host receive rates; Runtime stage latency is unavailable on wire."""
        now = time.time()
        stamp = time.monotonic()
        latest = self.latest()
        counts = (int(latest.get("received", 0)), int(latest.get("video_frames", 0)))
        previous = self.perf_last
        if previous is None:
            self.perf_last = (stamp, counts)
        elif stamp - previous[0] >= 0.8:
            elapsed = stamp - previous[0]
            track_fps = max(0.0, (counts[0] - previous[1][0]) / elapsed)
            video_fps = max(0.0, (counts[1] - previous[1][1]) / elapsed)
            self.perf_rows.append([int(now), track_fps, video_fps, None, None, None])
            self.perf_last = (stamp, counts)
        return dict(now=now, rows=[row for row in self.perf_rows if row[0] > since],
                    source="PC 수신율", latency_available=False)

    def _people(self) -> dict:
        metadata = self.latest()
        state = self._state()
        tracks = metadata.get("tracks", [])
        return dict(width=640, height=480,
                    state=dict(id=state["id"]),
                    pose=state["pose"], position_estimate=state["position_estimate"],
                    people=[dict(id=t["id"], x1=t["x"], y1=t["y"],
                                 x2=t["x"] + t["width"],
                                 y2=t["y"] + t["height"]) for t in tracks])

    def _click(self, query: dict) -> dict:
        x, y = float(query["x"][0]) * 640, float(query["y"][0]) * 480
        if not (0 <= x <= 640 and 0 <= y <= 480):
            return dict(ok=False, text="영상 밖을 선택했습니다")
        tracks = self.latest().get("tracks", [])
        matches = [t for t in tracks if t["x"] <= x <= t["x"] + t["width"] and
                   t["y"] <= y <= t["y"] + t["height"]]
        if not matches:
            result = self.control.mode(None)
            self.control.move(0, 0)
            return self._state() if result == 0 else dict(ok=False, text=f"STOP 실패 ({result})")
        chosen = min(matches, key=lambda t: t["width"] * t["height"])
        result = self.control.mode(chosen["id"])
        if result:
            return dict(ok=False, text=f"AUTO 요청 실패 ({result})")
        return self._state()

    def handle(self, url: str, method: str = "GET", body=None) -> dict:
        parsed = urlsplit(url)
        path, query = parsed.path, parse_qs(parsed.query)
        try:
            if path == "/model" and method == "GET":
                return self._model()
            if path == "/model/set" and method == "POST":
                result = self.control.change("model", int(query["id"][0]))
                if result:
                    return dict(ok=False, text=f"모델 변경 실패 ({result})")
                return self._model()
            if path == "/rules" and method == "GET":
                return self._rules()
            if path == "/rules/set" and method == "POST":
                if set(query) == {"tracker"}:
                    result = self.control.change("tracker", int(query["tracker"][0]))
                elif set(query) == {"relabel"}:
                    active = self.control.tracker_policy()["tracker_id"]
                    requested = query["relabel"][0]
                    if active not in (1, 3) or requested not in ("0", "1"):
                        return dict(ok=False, text="KeepID는 ByteTrack 모드에서만 바꿀 수 있습니다")
                    result = self.control.change("tracker", 3 if requested == "1" else 1)
                else:
                    keys = {"warm_min", "warm_max", "edge_birth", "confirm_ms", "warm_on", "edge_on", "stab"}
                    if not query or set(query) - keys or (any(k in query for k in ("warm_on", "edge_on", "stab")) and len(query) != 1):
                        return dict(ok=False, text="이 설정은 Runtime LAN 명령으로 아직 지원하지 않습니다")
                    if "stab" in query:
                        if query["stab"][0] not in ("0", "1"):
                            return dict(ok=False, text="흔들림 보정 값은 0 또는 1이어야 합니다")
                        values = dict(gmc=query["stab"][0] == "1")
                    elif "edge_on" in query:
                        requested = query["edge_on"][0]
                        if requested not in ("0", "1"):
                            return dict(ok=False, text="가장자리 Filter 값은 0 또는 1이어야 합니다")
                        if requested == "1" and self.previous_edge <= 0:
                            return dict(ok=False, text="가장자리 점수를 아래 Filter 값에 먼저 입력하세요")
                        values = dict(edge_birth=self.previous_edge if requested == "1" else 0.0)
                    elif "warm_on" in query:
                        if query["warm_on"][0] == "0":
                            values = dict(warmth_min=0, warmth_max=0)
                        elif query["warm_on"][0] == "1" and any(self.previous_warmth):
                            values = dict(zip(("warmth_min", "warmth_max"), self.previous_warmth))
                        else:
                            return dict(ok=False, text="온도 범위를 아래 Filter 값에 먼저 입력하세요")
                    else:
                        names = dict(warm_min="warmth_min", warm_max="warmth_max",
                                     edge_birth="edge_birth", confirm_ms="confirmation_ms")
                        values = {names[key]: float(item[0]) if key == "edge_birth" else int(item[0])
                                  for key, item in query.items()}
                    result = self.control.change_policy(**values)
                if result:
                    return dict(ok=False, text=f"Tracker 정책 변경 실패 ({result})")
                return self._rules()
            if path == "/gimbal/ping" and method == "POST":
                return self._state()
            if path == "/gimbal/people" and method == "GET":
                return self._people()
            if path == "/gimbal/click" and method == "POST":
                return self._click(query)
            if path == "/gimbal/stop" and method == "POST":
                result = self.control.mode(None)
                self.control.move(0, 0)
                return self._state() if result == 0 else dict(ok=False, text=f"STOP 실패 ({result})")
            if path == "/gimbal/move" and method == "POST":
                x, y = int(query["x"][0]), int(query["y"][0])
                result = self.control.mode(None)
                if result:
                    return dict(ok=False, text=f"MANUAL 실패 ({result})")
                self.control.move(x, y)
                return self._state()
            if path == "/perf" and method == "GET":
                return self._perf(float(query.get("since", ["0"])[0]))
            if path == "/measure" and method == "GET":
                return dict(available=False, on=False, last=None, history=[],
                            text="측정 제어는 Runtime LAN 명령으로 아직 지원하지 않습니다")
            if path == "/replay" and method == "GET":
                return dict(available=False, why="녹화 재생은 Runtime LAN 명령으로 아직 지원하지 않습니다")
            if path == "/healthz" and method == "GET":
                metadata = self.latest()
                return dict(up=time.monotonic() - self.started,
                            frames_in=metadata.get("received", 0))
        except (ValueError, KeyError, IndexError) as exc:
            return dict(ok=False, text=f"잘못된 요청: {exc}")
        except (OSError, RuntimeError, TimeoutError) as exc:
            return dict(ok=False, text=f"Runtime 연결 오류: {exc}")
        return dict(ok=False, text="이 기능은 Runtime LAN 명령으로 아직 지원하지 않습니다")
