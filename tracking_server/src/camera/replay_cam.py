#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""저장해 둔 프레임을 카메라 프로그램(cam_pipe)과 똑같은 형식으로 표준 출력에 내보내는 재생 카메라. 시험용이다 (진짜 카메라를 건드리지 않는다).

서버의 --live-cam 옵션에 이 파일(또는 이 파일을 부르는 두 줄짜리 스크립트)을 주면, 서버는 진짜 카메라 대신 저장된 프레임을 받는다.

내보내는 형식 (cam_pipe.cpp v2.6 과 같다)
  - 프레임 하나 = 640x480 화소, 화소마다 16 비트(작은 자리 먼저), 위 줄부터 = 614,400 바이트. 프레임 사이에 아무것도 없다.
  - 환경 변수 LWIR_CAM_STAMP 가 정확히 1 이면 프레임마다 앞에 48 바이트 머리가 붙는다: 표시 글자 LWF1, 판 번호 1, 표시값 0, 프레임 번호(1 부터),
    내보내기 직전의 시각(CLOCK_MONOTONIC, ns), 앞 프레임과의 간격(us), 칸 번호 0xFFFFFFFF, 그림 바이트 수, 고리 번호 0, 폭, 높이, 확인값.
    공유 메모리(LWIR_CAM_SHM)는 쓰지 않는다. 그림은 언제나 파이프로 간다.

읽는 것
  - 폴더: 그 안의 *.png 를 이름 순서로. 640x480 의 8 비트 또는 16 비트 흑백 PNG. 크기가 다른 파일은 건너뛰고 센다.
  - 8 비트 PNG 는 화소값에 257 을 곱해 16 비트로 만든다 (0 -> 0, 255 -> 65535). 서버를 --live-map fixed 로 띄우면 서버가 다시 8 비트로 바꾼 그림이
    저장된 PNG 와 화소 하나까지 같다. 기본값(--live-map pct)에서는 서버가 밝기 범위를 다시 잡으므로 조금 달라진다.
    서버의 온도 거름(--warm-min)은 16 비트 값을 보므로, 8 비트 PNG 를 틀 때는 --warm-min 0 으로 끄는 것이 맞다.
  - 16 비트 PNG 는 값을 그대로 보낸다. 색이 있는 PNG(3 채널)는 흑백으로 바꾼 뒤 위 규칙을 따른다.
  - 파일 하나(이름이 .raw 또는 .raw.gz 로 끝남): 서버의 측정 녹화(frames.raw.gz)처럼 16 비트 프레임이 줄줄이 든 파일. 그대로 보낸다.

속도
  - 초당 프레임 수는 --fps (기본 25, 30 fps 는 --fps 30). k 번째 프레임을 "시작 시각 + k / fps" 에 보낸다. 앞 프레임에서 얼마나 잤는지를 더해 가지 않으므로 밀리지 않는다.
  - 받는 쪽이 느려서 한 프레임 간격 넘게 늦어지면 그 사이의 프레임은 보내지 않고 건너뛴다 (진짜 카메라도 그 순간은 잃는다). 건너뛴 수는 끝날 때 적는다.
  - 프레임은 시작할 때 모두 메모리에 올린다 (보내는 동안 디스크·PNG 풀기로 서버의 CPU 를 뺏지 않게). --max-mb (기본 400 MB, 약 650 장) 를 넘는 뒤쪽 프레임은 쓰지 않는다.
  - 끝까지 가면 처음부터 다시 돈다 (--loops 0 = 끝없이, 기본). SIGTERM / SIGINT 를 받으면 바로 깨끗이 끝난다 (종료 코드 0). 받는 쪽이 닫혀도 0 으로 끝난다.

옵션 (서버의 --live-cam 은 인자를 넘기지 못하므로 환경 변수로도 받는다. 옵션이 있으면 옵션이 이긴다)
  --dir <폴더 또는 .raw/.raw.gz 파일>   LWIR_REPLAY_DIR
  --fps <숫자>                          LWIR_REPLAY_FPS   (기본 25)
  --loops <횟수>                        LWIR_REPLAY_LOOPS (기본 0 = 끝없이)
  --max-mb <MB>                         LWIR_REPLAY_MAX_MB (기본 400)
  --delay <초>                          LWIR_REPLAY_DELAY (기본 0)  첫 프레임을 보내기 전에 기다리는 시간 (서버가 준비를 마친 뒤 첫 프레임부터 받게 할 때)
표준 오류에는 시작 줄(CAMERA, REPLAY ...)과 끝 줄(FRAMES ...)만 쓴다. Python 3.10, numpy, cv2 만 쓴다 (.raw 파일만 틀 때는 둘 다 없어도 된다).
"""
import argparse
import gzip
import os
import signal
import struct
import sys
import time

W, H = 640, 480
FRAME_BYTES = W * H * 2
HDR = struct.Struct('<4sHHQQIIIIHHI')   # cam_pipe.cpp 의 FrameHead, camera_extra.py 의 HDR 과 같다
_W11 = struct.Struct('<11I')
SPIN_S = 0.0005 if os.name == 'posix' else 0.017   # 이 시간보다 가까워지면 자지 않고 시계만 본다 (Windows 의 Python 3.10 은 잠이 16 ms 단위라 시험용으로 넉넉히 잡는다)


class _Stop(BaseException):   # 신호를 받으면 막힌 쓰기에서도 바로 빠져나오게 한다
    pass


def _on_signal(signum, frame):
    raise _Stop()


def header(seq, t_ns, dt_us):
    """48 바이트 머리. 확인값 = 앞의 32 비트 낱말 11 개와 0x5A17C0DE 의 xor."""
    b = HDR.pack(b'LWF1', 1, 0, seq, t_ns, min(dt_us, 0xFFFFFFFF), 0xFFFFFFFF, FRAME_BYTES, 0, W, H, 0)
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


def load(src, max_mb, say):
    """-> 프레임 목록 (각각 614,400 바이트의 bytes)"""
    limit = max(1, int(max_mb * 1024 * 1024 // FRAME_BYTES))
    frames = []
    if os.path.isfile(src) and src.lower().endswith(('.raw', '.raw.gz')):
        f = gzip.open(src, 'rb') if src.lower().endswith('.gz') else open(src, 'rb')
        with f:
            while len(frames) < limit:
                b = f.read(FRAME_BYTES)
                if len(b) < FRAME_BYTES:
                    break
                frames.append(b)
            more = len(frames) == limit and bool(f.read(1))
        if more:
            say('REPLAY note: only the first %d frames are used (--max-mb %g)' % (limit, max_mb))
        return frames, 'raw 16-bit', 0
    if not os.path.isdir(src):
        raise SystemExit('replay_cam: no such folder or .raw/.raw.gz file: %s' % os.path.basename(src))
    import cv2
    names = sorted(n for n in os.listdir(src) if n.lower().endswith('.png'))
    bad = 0
    kinds = set()
    for n in names:
        if len(frames) >= limit:
            say('REPLAY note: only the first %d of %d png files are used (--max-mb %g)' % (limit, len(names), max_mb))
            break
        raw = cv2.imread(os.path.join(src, n), cv2.IMREAD_UNCHANGED)
        img = to16(raw)
        if img is None:
            bad += 1
            continue
        kinds.add('8-bit x257' if raw.dtype.itemsize == 1 else '16-bit')
        frames.append(img.astype('<u2', copy=False).tobytes())
    return frames, ' + '.join(sorted(kinds)) or 'none', bad


def write_all(fd, data):
    mv = memoryview(data)
    while len(mv):
        n = os.write(fd, mv)
        mv = mv[n:]


def main():
    ap = argparse.ArgumentParser(description='replay stored frames in the cam_pipe output format')
    ap.add_argument('--dir', default=os.environ.get('LWIR_REPLAY_DIR', ''))
    ap.add_argument('--fps', type=float, default=float(os.environ.get('LWIR_REPLAY_FPS', '25') or 25))
    ap.add_argument('--loops', type=int, default=int(os.environ.get('LWIR_REPLAY_LOOPS', '0') or 0))
    ap.add_argument('--max-mb', type=float, default=float(os.environ.get('LWIR_REPLAY_MAX_MB', '400') or 400))
    ap.add_argument('--delay', type=float, default=float(os.environ.get('LWIR_REPLAY_DELAY', '0') or 0))
    a = ap.parse_args()
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
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    stamp = os.environ.get('LWIR_CAM_STAMP') == '1'
    sent = skipped = 0
    t_begin = None
    code = 0
    try:
        frames, kind, bad = load(a.dir, a.max_mb, say)
        if not frames:
            say('REPLAY no usable frame (640x480 png, 8-bit or 16-bit) in the given folder; %d files skipped' % bad)
            return 2
        say('CAMERA %d %d' % (W, H))
        say('REPLAY %d frames (%s), %d skipped files, %g fps, loops %s, time stamps %s' % (len(frames), kind, bad, a.fps, a.loops or 'endless', 'on' if stamp else 'off'))
        if stamp and os.environ.get('LWIR_CAM_SHM') == '1':
            say('SHM_RING not used by replay_cam: pictures go through the pipe')
        fd = sys.stdout.fileno()
        if os.name == 'nt':
            import msvcrt
            msvcrt.setmode(fd, os.O_BINARY)
        period = 1.0 / a.fps
        total = a.loops * len(frames) if a.loops > 0 else None
        clock = time.perf_counter
        if a.delay > 0:
            time.sleep(a.delay)
        t_begin = clock()
        k = 0                  # 시간표의 칸 번호: k 번째 칸은 t_begin + k * period 에 나간다
        seq = 0
        t_prev_ns = 0
        while total is None or k < total:
            due = t_begin + k * period
            d = due - clock()
            if d < -period:   # 한 칸 넘게 늦었다: 지나간 칸은 건너뛴다 (몰아서 보내지 않는다)
                jump = int(-d / period)
                k += jump
                skipped += jump
                continue
            if d > SPIN_S:
                time.sleep(d - SPIN_S)
            while clock() < due:
                pass
            b = frames[k % len(frames)]
            if stamp:
                seq += 1
                t_ns = time.monotonic_ns()
                b = header(seq, t_ns, (t_ns - t_prev_ns) // 1000 if t_prev_ns else 0) + b
                t_prev_ns = t_ns
            write_all(fd, b)
            sent += 1
            k += 1
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
    secs = (time.perf_counter() - t_begin) if t_begin is not None else 0.0
    say('FRAMES %d in %.2f s = %.2f fps, skipped %d (receiver was slow)' % (sent, secs, sent / secs if secs > 0 else 0.0, skipped))
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
    # 표준 출력이 이미 닫혔어도 Python 이 끝나면서 오류 글을 찍지 않게 한다
    try:
        sys.stdout = open(os.devnull, 'w')
    except OSError:
        pass
    sys.exit(rc)
