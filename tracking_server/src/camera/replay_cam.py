#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""저장해 둔 프레임을 카메라 프로그램(cam_pipe)과 똑같은 형식으로 표준 출력에 내보내는 재생 카메라. 시험용이다 (진짜 카메라를 건드리지 않는다).

서버의 --live-cam 옵션에 이 파일(또는 이 파일을 부르는 두 줄짜리 스크립트)을 주면, 서버는 진짜 카메라 대신 저장된 프레임을 받는다.

내보내는 형식 (cam_pipe.cpp v2.6 과 같다)
  - 프레임 하나 = 640x480 화소, 화소마다 16 비트(작은 자리 먼저), 위 줄부터 = 614,400 바이트. 프레임 사이에 아무것도 없다.
  - 환경 변수 LWIR_CAM_STAMP 가 정확히 1 이면 프레임마다 앞에 48 바이트 머리가 붙는다: 표시 글자 LWF1, 판 번호 1, 표시값 0, 프레임 번호(1 부터),
    내보내기 직전의 시각(CLOCK_MONOTONIC, ns), 앞 프레임과의 간격(us), 칸 번호 0xFFFFFFFF, 그림 바이트 수, 고리 번호 0, 폭, 높이, 확인값.
    공유 메모리(LWIR_CAM_SHM)는 쓰지 않는다. 그림은 언제나 파이프로 간다.

읽는 것 (11-12 부터 필요한 프레임만 그때그때 읽는다: 길이 제한 없음, 메모리는 수 MB)
  - 폴더: 그 안의 *.png 를 이름 순서로. 640x480 의 8 비트 또는 16 비트 흑백 PNG. 크기가 다른 파일은 건너뛰고 센다 (시작할 때 PNG 머리 33 바이트만 읽어 본다).
    프레임은 필요할 때 풀고, 뒤 스레드가 다음 6 장을 미리 풀어 둔다 (보내는 쪽의 시간표가 PNG 풀기 때문에 흔들리지 않게).
  - 8 비트 PNG 는 화소값에 257 을 곱해 16 비트로 만든다 (0 -> 0, 255 -> 65535). 서버를 --live-map fixed 로 띄우면 서버가 다시 8 비트로 바꾼 그림이
    저장된 PNG 와 화소 하나까지 같다. 기본값(--live-map pct)에서는 서버가 밝기 범위를 다시 잡으므로 조금 달라진다.
    서버의 온도 거름(--warm-min)은 16 비트 값을 보므로, 8 비트 PNG 를 틀 때는 --warm-min 0 으로 끄는 것이 맞다.
  - 16 비트 PNG 는 값을 그대로 보낸다. 색이 있는 PNG(3 채널)는 흑백으로 바꾼 뒤 위 규칙을 따른다.
  - 파일 하나(이름이 .raw 또는 .raw.gz 로 끝남): 서버의 측정 녹화(frames.raw.gz)처럼 16 비트 프레임이 줄줄이 든 파일. 그대로 보낸다.
    .raw.gz 는 gzip 을 풀면서 읽는다. 앞으로 건너뛰기 = 그 장까지 풀어서 버림, 뒤로 건너뛰기 = 파일을 다시 열어 앞에서부터 풂 (먼 곳일수록 오래 걸린다).
    전체 장 수 = --total 로 받은 값, 없으면 gzip 끝의 원래 크기(4 GB 아래)로 센다.

속도
  - 초당 프레임 수는 --fps (기본 25, 30 fps 는 --fps 30). 다음 프레임을 "그 앞 프레임의 시각 + 1 / fps" 에 보낸다. 앞 프레임에서 얼마나 잤는지를 더해 가지 않으므로 밀리지 않는다.
  - 받는 쪽이 느려서 한 프레임 간격 넘게 늦어지면 그 사이의 프레임은 보내지 않고 건너뛴다 (진짜 카메라도 그 순간은 잃는다). 건너뛴 수는 끝날 때 적는다.
  - 끝까지 가면 처음부터 다시 돈다 (--loops 0 = 끝없이, 기본). SIGTERM / SIGINT 를 받으면 바로 깨끗이 끝난다 (종료 코드 0). 받는 쪽이 닫혀도 0 으로 끝난다.

조종 (11-12, --control 또는 LWIR_REPLAY_CONTROL=1 일 때만 표준 입력을 읽는다. 서버가 줄 단위로 쓴다)
  SEEK <번호> [PAUSE]    PAUSE가 붙으면 이동한 장에서 멈춘다. 0 부터 센 프레임 번호로 건너뜀. 새 "바퀴"(pass)를 시작하고, 첫 프레임을 보내기 전에 --gap (기본 1.2 초) 동안 아무것도 보내지 않는다
                 (서버의 "입력이 1 초 넘게 끊기면 추적을 새로 시작" 규칙이 그 틈으로 추적을 비운다). 그 프레임은 멈춘 중이어도 한 번은 새 프레임으로 나간다.
  PAUSE          지금 프레임에서 멈춤. 멈춘 동안 같은 프레임을 "되풀이" 표시를 달아 초당 --rep-fps (기본 5) 장 보낸다 (서버가 카메라가 죽었다고 보지 않게, 그림도 계속 갱신되게. 서버는 되풀이를 채점하지 않는다).
  PLAY           이어서 재생 (멈춘 프레임의 다음 프레임부터)
  웹 재생은 완주하면 마지막 장을 유지한다. PLAY는 처음부터 재생하고, SEEK는 원하는 장으로 이동한다.
  SPEED <fps>    초당 프레임 수 바꾸기
  줄은 서버가 모아서 마지막 것만 보낸다. 모르는 줄과 범위 밖의 번호는 (끝 쪽으로 맞춰) 조용히 처리한다. 표준 입력이 닫혀도 재생은 계속된다.

옵션 (서버의 --live-cam 은 인자를 넘기지 못하므로 환경 변수로도 받는다. 옵션이 있으면 옵션이 이긴다)
  --dir <폴더 또는 .raw/.raw.gz 파일>   LWIR_REPLAY_DIR
  --fps <숫자>                          LWIR_REPLAY_FPS   (기본 25)
  --loops <횟수>                        LWIR_REPLAY_LOOPS (기본 0 = 끝없이)
  --max-mb <MB>                         LWIR_REPLAY_MAX_MB  (더는 쓰지 않음: 길이 제한이 없어졌다. 옛 실행 스크립트가 줘도 받아만 둔다)
  --delay <초>                          LWIR_REPLAY_DELAY (기본 0)  첫 프레임을 보내기 전에 기다리는 시간 (서버가 준비를 마친 뒤 첫 프레임부터 받게 할 때)
  --total <장 수>                       LWIR_REPLAY_TOTAL (.raw.gz 의 전체 장 수. 서버가 알려 준다)
  --gap <초>                            LWIR_REPLAY_GAP   (기본 1.2)  SEEK 뒤 첫 프레임 전의 빈 시간
  --rep-fps <숫자>                      LWIR_REPLAY_REP_FPS (기본 5)  멈춘 동안 되풀이 프레임의 초당 장 수
  --control                             LWIR_REPLAY_CONTROL=1  표준 입력의 조종 줄을 읽는다
  --prepare-frame                       LWIR_REPLAY_PREPARE_FRAME=1  --no-skip 재생에서 다음 한 장을 내보내기 전에 읽는다
  --tag                                 LWIR_REPLAY_TAG=1  (정답 비교용) 머리를 붙일 때(LWIR_CAM_STAMP=1) 어느 프레임인지 적는다:
                                        flags 2 번 비트(값 2)를 켠다. 멈춘 동안 되풀이하는 프레임은 flags 값 4 도 켠다. slot = 녹화 안의 프레임 번호(0 부터, 폴더도 .raw 도 같다),
                                        ring_id = PNG 번호(파일 이름의 숫자, 아래 20 비트) + 바퀴 번호(0 부터, 위 12 비트; 4095 에서 멈춘다),
                                        이름에 숫자가 없거나 .raw 파일이면 PNG 번호 0 (모름). 바퀴 = 끝에서 처음으로 돌 때마다, 그리고 SEEK 마다 하나씩 늘어난다.
                                        머리가 꺼져 있으면 아무것도 하지 않는다. 서버(src/camera/replay_eval.py)가 읽는다.
                                        (11-12 이전의 slot 은 시간표 칸 번호였다. 서버가 건너뛴 수를 세는 방법이 바뀌었다: 같은 바퀴 안에서 번호가 1 보다 크게 뛴 만큼.)
표준 오류에는 시작 줄(CAMERA, REPLAY ...), 조종 줄(REPLAY CMD ...)과 끝 줄(FRAMES ...)만 쓴다. Python 3.10, numpy, cv2 만 쓴다 (.raw 파일만 틀 때는 둘 다 없어도 된다).
"""
import argparse
import gzip
import os
import signal
import struct
import sys
import threading
import time

W, H = 640, 480
FRAME_BYTES = W * H * 2
HDR = struct.Struct('<4sHHQQIIIIHHI')   # cam_pipe.cpp 의 FrameHead, camera_extra.py 의 HDR 과 같다
_W11 = struct.Struct('<11I')
SPIN_S = 0.0005 if os.name == 'posix' else 0.017   # 이 시간보다 가까워지면 자지 않고 시계만 본다 (Windows 의 Python 3.10 은 잠이 16 ms 단위라 시험용으로 넉넉히 잡는다)
AHEAD = 6                                           # 미리 풀어 두는 PNG 장 수 (약 3.7 MB)
FLAG_TAG, FLAG_REPEAT = 2, 4
PASS_MAX = 0xFFF
TAIL_REPEATS = 4                                    # 한 번만 재생(--loops)이 끝날 때 마지막 프레임을 되풀이로 몇 장 더 보낸다 (--control 일 때만)


class _Stop(BaseException):   # 신호를 받으면 막힌 쓰기에서도 바로 빠져나오게 한다
    pass


def _on_signal(signum, frame):
    raise _Stop()


def header(seq, t_ns, dt_us, flags=0, slot=0xFFFFFFFF, ring_id=0):
    """48 바이트 머리. 확인값 = 앞의 32 비트 낱말 11 개와 0x5A17C0DE 의 xor."""
    b = HDR.pack(b'LWF1', 1, flags, seq, t_ns, min(dt_us, 0xFFFFFFFF), slot & 0xFFFFFFFF, FRAME_BYTES, ring_id & 0xFFFFFFFF, W, H, 0)
    x = 0x5A17C0DE
    for v in _W11.unpack(b[:44]):
        x ^= v
    return b[:44] + struct.pack('<I', x)


def to16(img):
    """읽은 그림 -> 480x640 uint16, 맞지 않으면 None."""
    import numpy as np
    if img is None:
        return None
    if img.ndim == 3:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if img.ndim != 2 or img.shape != (H, W):
        return None
    if img.dtype == np.uint8:
        return img.astype(np.uint16) * 257
    if img.dtype == np.uint16:
        return img
    return None


def png_number(name):
    """000123.png -> 123. 숫자가 아니거나 20 비트를 넘으면 0 (모름)"""
    s = os.path.splitext(name)[0]
    return int(s) if s.isdigit() and 0 < int(s) <= 0xFFFFF else 0


def png_head(path):
    """(비트, 폭, 높이) 또는 None. PNG 머리(IHDR)만 읽는다"""
    try:
        with open(path, 'rb') as f:
            b = f.read(33)
    except OSError:
        return None
    if len(b) < 33 or b[:8] != b'\x89PNG\r\n\x1a\n' or b[12:16] != b'IHDR':
        return None
    return b[24], int.from_bytes(b[16:20], 'big'), int.from_bytes(b[20:24], 'big')


class PngSource:
    """PNG 폴더. read(i) = i 번째 파일을 풀어 614,400 바이트로 (못 풀면 None). 시작할 때 머리만 보고 쓸 수 있는 파일 목록을 만든다."""
    def __init__(self, folder):
        self.folder = folder
        self.paths, self.nums, self.bad = [], [], 0
        kinds = set()
        for n in sorted(x for x in os.listdir(folder) if x.lower().endswith('.png')):
            p = os.path.join(folder, n)
            h = png_head(p)
            if h is None or h[0] not in (8, 16) or (h[1], h[2]) != (W, H):
                self.bad += 1
                continue
            kinds.add('8-bit x257' if h[0] == 8 else '16-bit')
            self.paths.append(p)
            self.nums.append(png_number(n))
        self.total = len(self.paths)
        self.kind = ' + '.join(sorted(kinds)) or 'none'

    def read(self, i):
        import cv2
        img = to16(cv2.imread(self.paths[i], cv2.IMREAD_UNCHANGED))
        return None if img is None else img.astype('<u2', copy=False).tobytes()

    def png(self, i):
        return self.nums[i]


class RawSource:
    """16 비트 프레임이 줄줄이 든 .raw / .raw.gz. read(i): 바로 다음 장이면 그냥 읽고, 아니면 건너뛴다 (gz: 앞은 풀어서 버림, 뒤는 다시 열기)."""
    def __init__(self, path, total_hint=0):
        self.path = path
        self.gz = path.lower().endswith('.gz')
        if self.gz:
            with open(path, 'rb') as f:
                f.seek(-4, os.SEEK_END)
                isize = struct.unpack('<I', f.read(4))[0]
            self.total = total_hint if total_hint > 0 else isize // FRAME_BYTES
        else:
            self.total = total_hint if total_hint > 0 else os.path.getsize(path) // FRAME_BYTES
        self.f = None
        self.at = 0          # 지금 파일 위치 = 이 번호 장의 앞
        self.bad = 0
        self.kind = 'raw 16-bit'
        self.rewinds = 0

    def _open(self):
        if self.f is not None:
            self.f.close()
        self.f = gzip.open(self.path, 'rb') if self.gz else open(self.path, 'rb')
        self.at = 0

    def read(self, i):
        if self.f is None or i < self.at:
            if self.f is not None:
                self.rewinds += 1
            self._open()
        if i > self.at:
            if self.gz:
                left = (i - self.at) * FRAME_BYTES
                while left > 0:
                    d = self.f.read(min(left, 1 << 20))
                    if not d:
                        return None
                    left -= len(d)
            else:
                self.f.seek(i * FRAME_BYTES)
            self.at = i
        b = self.f.read(FRAME_BYTES)
        if len(b) < FRAME_BYTES:
            return None
        self.at = i + 1
        return b

    def png(self, i):
        return 0


class Ahead(threading.Thread):
    """뒤에서 다음 장들을 미리 읽어 둔다 (PNG 만). 못 읽은 장은 get() 이 그 자리에서 읽는다. 메모리 = AHEAD 장."""
    def __init__(self, src):
        super().__init__(daemon=True)
        self.src, self.cv, self.want, self.cache, self.done = src, threading.Condition(), [], {}, False
        self.reading = None

    def run(self):
        while True:
            with self.cv:
                while True:
                    if self.done:
                        return
                    todo = next((i for i in self.want if i not in self.cache), None)
                    if todo is not None:
                        break
                    self.cv.wait()
                self.reading = todo
            try:
                b = self.src.read(todo)
            except Exception:
                b = None
            with self.cv:
                self.reading = None
                if todo in self.want:
                    self.cache[todo] = b
                for k in [k for k in self.cache if k not in self.want]:
                    del self.cache[k]
                self.cv.notify_all()

    def set_want(self, lst):
        with self.cv:
            self.want = lst
            for k in [k for k in self.cache if k not in lst]:
                del self.cache[k]
            self.cv.notify_all()

    def get(self, i):
        with self.cv:
            if i in self.cache:
                return self.cache.pop(i)
        return self.src.read(i)

    def prepare(self, i):
        with self.cv:
            # Reuse an in-flight read of this frame. If another frame is in
            # flight, claim this one so the worker won't also decode it.
            self.cv.wait_for(lambda: self.reading != i)
            self.want = [wanted for wanted in self.want if wanted != i]
            if i in self.cache:
                return self.cache.pop(i)
        return self.src.read(i)


class Control:
    """표준 입력의 조종 줄. 읽는 스레드는 값만 덮어쓴다 (모으기: 같은 종류는 마지막 것만 남는다)."""
    def __init__(self, say):
        self.lk, self.ev, self.say = threading.Lock(), threading.Event(), say
        self.seek = self.paused = self.speed = None
        self.step = 0
        self.clear_steps = False
        self.step_paused = False
        self.n = self.bad = 0

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            for raw in iter(sys.stdin.buffer.readline, b''):
                f = raw.decode('ascii', 'replace').split()
                if not f:
                    continue
                k = f[0].upper()
                try:
                    with self.lk:
                        if k == 'SEEK' and (len(f) == 2 or (len(f) == 3 and f[2].upper() == 'PAUSE')):
                            self.seek = int(f[1])
                            self.step = 0
                            self.clear_steps = True
                            if len(f) == 3:
                                self.paused = self.step_paused = True
                        elif k == 'SPEED' and len(f) == 2:
                            self.speed = float(f[1])
                        elif k in ('PAUSE', 'PLAY') and len(f) == 1:
                            self.paused = self.step_paused = k == 'PAUSE'
                            if not self.paused:
                                self.step = 0
                                self.clear_steps = True
                        elif k == 'STEP' and len(f) == 1:
                            if self.step_paused:
                                self.step += 1
                        else:
                            self.bad += 1
                            continue
                        self.n += 1
                        self.ev.set()
                except ValueError:
                    self.bad += 1
                    continue
        except (OSError, ValueError, AttributeError):   # 표준 입력이 없거나 닫힘
            pass

    def take(self):
        with self.lk:
            c = (self.seek, self.paused, self.speed, self.step, self.clear_steps)
            self.seek = self.paused = self.speed = None
            self.step = 0
            self.clear_steps = False
            self.ev.clear()
        return c


def write_all(fd, data):
    mv = memoryview(data)
    while len(mv):
        n = os.write(fd, mv)
        mv = mv[n:]


def next_deadline(due, completed, period, *, no_skip, pacing):
    """Pace lossless replay without adding every write time to its period.

    Deadline pacing preserves all frame positions. If a whole next deadline
    was missed, abandon the timing debt and wait a fresh period; do not burst
    through a backlog. Sub-period lateness can shorten the following interval.
    The legacy mode remains available for same-input comparisons.
    """
    nominal = due + period
    if not no_skip:
        return nominal
    if pacing != 'deadline':
        return max(due, completed) + period
    return completed + period if nominal <= completed else nominal


def main():
    ap = argparse.ArgumentParser(description='replay stored frames in the cam_pipe output format')
    ap.add_argument('--dir', default=os.environ.get('LWIR_REPLAY_DIR', ''))
    ap.add_argument('--fps', type=float, default=float(os.environ.get('LWIR_REPLAY_FPS', '25') or 25))
    ap.add_argument('--loops', type=int, default=int(os.environ.get('LWIR_REPLAY_LOOPS', '0') or 0))
    ap.add_argument('--max-mb', type=float, default=float(os.environ.get('LWIR_REPLAY_MAX_MB', '400') or 400))   # no longer used
    ap.add_argument('--delay', type=float, default=float(os.environ.get('LWIR_REPLAY_DELAY', '0') or 0))
    ap.add_argument('--total', type=int, default=int(os.environ.get('LWIR_REPLAY_TOTAL', '0') or 0))
    ap.add_argument('--gap', type=float, default=float(os.environ.get('LWIR_REPLAY_GAP', '1.2') or 1.2))
    ap.add_argument('--rep-fps', type=float, default=float(os.environ.get('LWIR_REPLAY_REP_FPS', '5') or 5))
    ap.add_argument('--control', action='store_true', default=os.environ.get('LWIR_REPLAY_CONTROL') == '1')
    ap.add_argument('--tag', action='store_true', default=os.environ.get('LWIR_REPLAY_TAG') == '1')
    ap.add_argument('--no-skip', action='store_true', help='preserve sequential frames under backpressure')
    ap.add_argument('--pacing', choices=['legacy', 'deadline'],
                    default=os.environ.get('LWIR_REPLAY_PACING', 'legacy'),
                    help='lossless replay timing; deadline includes read/write work in the frame period')
    ap.add_argument('--prepare-frame', action='store_true',
                    default=os.environ.get('LWIR_REPLAY_PREPARE_FRAME') == '1',
                    help='read one frame before its scheduled release during lossless replay')
    a = ap.parse_args()
    if a.pacing not in ('legacy', 'deadline'):
        ap.error('LWIR_REPLAY_PACING must be legacy or deadline')
    err = sys.stderr

    def say(t):
        try:
            err.write(t + '\n')
            err.flush()
        except (OSError, ValueError):
            pass
    if not a.dir:
        raise SystemExit('replay_cam: give --dir <folder of png frames> or set LWIR_REPLAY_DIR')
    if not 0.1 <= a.fps <= 240:
        raise SystemExit('replay_cam: --fps must be between 0.1 and 240')
    if not 0.5 <= a.rep_fps <= 30:
        raise SystemExit('replay_cam: --rep-fps must be between 0.5 and 30')
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    stamp = os.environ.get('LWIR_CAM_STAMP') == '1'
    prepare_frame = a.prepare_frame and a.no_skip
    ctl = Control(say) if a.control else None
    hold_at_end = ctl is not None and stamp and a.tag and a.loops == 1
    if ctl is not None and not (os.path.isfile(a.dir) and a.dir.lower().endswith(('.raw', '.raw.gz'))):
        import cv2   # noqa: F401  (Windows: loading a DLL while another thread waits on the stdin pipe hangs, so cv2 / numpy are loaded before the control thread starts)
        import numpy   # noqa: F401
    sent = skipped = repeats = seeks = 0
    t_begin = None
    code = 0
    ahead = None
    try:
        src = None
        if os.path.isfile(a.dir) and a.dir.lower().endswith(('.raw', '.raw.gz')):
            src = RawSource(a.dir, a.total)
        elif os.path.isdir(a.dir):
            src = PngSource(a.dir)
        else:
            raise SystemExit('replay_cam: no such folder or .raw/.raw.gz file: %s' % os.path.basename(a.dir))
        if ctl is not None:
            ctl.start()
        total = src.total
        if total <= 0:
            say('REPLAY no usable frame (640x480 png, 8-bit or 16-bit) in the given folder; %d files skipped' % src.bad)
            return 2
        say('CAMERA %d %d' % (W, H))
        tag = stamp and a.tag
        say('REPLAY %d frames (%s), %d skipped files, %g fps, loops %s, time stamps %s, png numbers %s, control %s' % (total, src.kind, src.bad, a.fps, a.loops or 'endless', 'on' if stamp else 'off', 'on' if tag else 'off', 'on' if ctl else 'off'))
        say('REPLAY pacing %s, lossless %s' % (a.pacing, 'on' if a.no_skip else 'off'))
        if prepare_frame:
            say('REPLAY frame preparation on')
        if stamp and os.environ.get('LWIR_CAM_SHM') == '1':
            say('SHM_RING not used by replay_cam: pictures go through the pipe')
        fd = sys.stdout.fileno()
        if os.name == 'nt':
            import msvcrt
            msvcrt.setmode(fd, os.O_BINARY)
        if isinstance(src, PngSource):
            ahead = Ahead(src)
            ahead.start()
        get = src.read
        if ahead is not None:
            get = ahead.prepare if prepare_frame else ahead.get
        clock = time.perf_counter
        fps = a.fps
        period = 1.0 / fps
        rep_period = 1.0 / a.rep_fps
        if a.delay > 0:
            time.sleep(a.delay)
        t_begin = clock()
        pos = 0                    # 다음에 보낼 프레임 번호
        pass_no = 0                # 바퀴 번호 (끝에서 처음으로 돌 때, SEEK 할 때마다 하나 늘어난다)
        wraps = 0
        due = t_begin              # 다음 프레임을 보낼 시각
        first = True               # 새 프레임을 하나 보낼 차례 (멈춘 중이어도: 시작과 SEEK 뒤 첫 프레임)
        seek_wait = False          # SEEK 를 받았고 그 첫 프레임은 아직 안 보냄 (그 사이에 또 SEEK 가 오면 같은 구간으로 센다)
        paused = False
        at_end = False
        cur = None                 # 마지막으로 보낸 새 프레임 (멈춘 동안 되풀이)
        prepared = None            # 아직 보내지 않은 (위치, 이미지). SEEK 이 오면 버린다.
        cur_pos = cur_png = 0
        next_rep = 0.0
        seq = 0
        t_prev_ns = 0
        steps_left = 0
        stepping = False

        def emit(b, p, png, repeat):
            nonlocal seq, t_prev_ns
            if stamp:
                seq += 1
                t_ns = time.monotonic_ns()
                dt = (t_ns - t_prev_ns) // 1000 if t_prev_ns else 0
                if tag or repeat:   # 되풀이는 머리에 늘 표시를 단다 (서버가 채점하지 않으려고)
                    h = header(seq, t_ns, dt, FLAG_TAG | (FLAG_REPEAT if repeat else 0), p, png | (min(pass_no, PASS_MAX) << 20))
                else:
                    h = header(seq, t_ns, dt)
                t_prev_ns = t_ns
                write_all(fd, h + b)
            else:
                write_all(fd, b)

        while True:
            if ctl is not None:
                c_seek, c_paused, c_speed, c_step, c_clear_steps = ctl.take()
                if c_clear_steps:
                    steps_left = 0
                    if stepping:
                        first = stepping = False
                if c_speed is not None and 0.1 <= c_speed <= 240:
                    fps = c_speed
                    period = 1.0 / fps
                    say('REPLAY CMD speed %g' % fps)
                if at_end and c_paused is False and c_seek is None:
                    c_seek = 0   # Play after a completed pass uses the same gap/reset as an explicit seek.
                    wraps = 0
                if c_seek is not None:
                    at_end = False
                    pos = min(max(c_seek, 0), total - 1)
                    prepared = None
                    if not seek_wait:
                        pass_no += 1
                    seek_wait = True
                    seeks += 1
                    due = clock() + a.gap
                    first = True
                    if ahead is not None:
                        ahead.set_want([pos])
                    say('REPLAY CMD seek %d (pass %d, first frame after %.1f s)' % (pos, pass_no + 1, a.gap))
                if c_paused is not None and c_paused != paused:
                    paused = c_paused
                    if paused:
                        next_rep = clock() + rep_period
                    elif not first:
                        due = clock()
                    say('REPLAY CMD %s at %d' % ('pause' if paused else 'play', cur_pos))
                if paused and not seek_wait and not at_end:
                    steps_left += c_step
            if not paused or at_end or seek_wait or pos >= total:
                steps_left = 0
                if stepping:
                    first = stepping = False
            elif not first and steps_left > 0:
                steps_left -= 1
                first = stepping = True  # 다음 한 장만 보통 프레임으로 보낸다 (되풀이 표시 없음).
                due = clock()
                say('REPLAY CMD step %d -> %d' % (cur_pos, pos))
            if (paused or at_end) and not first:   # 멈춤: 같은 프레임을 되풀이로 (조종 줄이 오면 바로 깬다)
                d = next_rep - clock()
                if d > 0:
                    if ctl is not None:
                        ctl.ev.wait(d)
                    else:
                        time.sleep(d)
                    continue
                if cur is not None:
                    emit(cur, cur_pos, cur_png, True)
                    repeats += 1
                next_rep = max(next_rep + rep_period, clock() + rep_period * 0.5)
                continue
            if prepare_frame and (pos < total or a.loops <= 0 or wraps + 1 < a.loops):
                prepare_pos = pos % total
                if prepared is None or prepared[0] != prepare_pos:
                    prepared = (prepare_pos, get(prepare_pos))
                    continue  # Recheck controls before committing the frame or wrap.
            d = due - clock()
            if not a.no_skip and not first and d < -period:   # 한 칸 넘게 늦었다: 지나간 칸은 건너뛴다 (몰아서 보내지 않는다)
                jump = int(-d / period)
                if hold_at_end and pos < total:
                    jump = min(jump, total - 1 - pos)   # Never skip the frame we will hold.
                if jump:
                    pos += jump
                    skipped += jump
                    due += jump * period
                    continue
                due = clock()
                d = 0
            if d > SPIN_S:
                if ctl is not None:
                    if ctl.ev.wait(d - SPIN_S):
                        continue
                else:
                    time.sleep(d - SPIN_S)
            while clock() < due:
                pass
            if prepare_frame and ctl is not None and ctl.ev.is_set():
                continue
            while pos >= total:   # 끝에서 처음으로
                wraps += 1
                if a.loops > 0 and wraps >= a.loops:
                    pos = -1
                    break
                pos -= total
                pass_no += 1
            if pos < 0:
                if ctl is not None and cur is not None:   # 끝: 서버는 다음 프레임이 와야 앞 프레임을 화면에 낸다 (미리 읽기). 마지막 프레임이 나오도록 되풀이를 몇 장 더 보낸다
                    for _ in range(TAIL_REPEATS):
                        time.sleep(rep_period)
                        emit(cur, cur_pos, cur_png, True)
                        repeats += 1
                break
            if prepare_frame:
                _, b = prepared
                prepared = None
            else:
                b = get(pos)
            if b is None:   # 못 읽은 프레임은 건너뛴다 (시간은 쓰지 않는다)
                src.bad += 1
                pos += 1
                continue
            emit(b, pos, src.png(pos), False)
            cur, cur_pos, cur_png = b, pos, src.png(pos)
            seek_wait = False
            sent += 1
            pos += 1
            due = next_deadline(due, clock(), period, no_skip=a.no_skip, pacing=a.pacing)
            first = stepping = False
            if hold_at_end and pos == total:
                at_end = True
                say('REPLAY END: holding final frame')
            if paused or at_end:
                next_rep = clock() + rep_period
            if ahead is not None:
                ahead.set_want([(pos + k) % total for k in range(AHEAD)] if not (paused or at_end) else [])
    except _Stop:
        pass
    except (BrokenPipeError, ConnectionError):
        pass
    except OSError as e:
        if getattr(e, 'errno', None) not in (22, 32):   # 받는 쪽이 닫힘 (Windows 는 22, 리눅스는 32)
            say('REPLAY write error: %s' % type(e).__name__)
            code = 1
    finally:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (OSError, ValueError):
            pass
        if ahead is not None:   # 미리 읽는 스레드가 PNG 를 풀다가 파이썬이 끝나면 죽을 수 있으니 끝나기를 기다린다
            ahead.done = True
            with ahead.cv:
                ahead.cv.notify_all()
            ahead.join(5)
    secs = (time.perf_counter() - t_begin) if t_begin is not None else 0.0
    say('FRAMES %d in %.2f s = %.2f fps, skipped %d (receiver was slow), repeats %d, seeks %d' % (sent, secs, sent / secs if secs > 0 else 0.0, skipped, repeats, seeks))
    try:
        sys.stdout.flush()
    except (OSError, ValueError):
        pass
    return code


if __name__ == '__main__':
    try:
        rc = main()
    except _Stop:
        rc = 0
    # 표준 출력이 이미 닫혔어도 Python 이 끝나면서 오류 글을 찍지 않게 한다. 표준 입력을 읽고 있는 스레드가 있어도 바로 끝나도록 정리 없이 나간다 (쓸 것은 모두 내보낸 뒤)
    try:
        sys.stderr.flush()
    except (OSError, ValueError):
        pass
    os._exit(rc)
