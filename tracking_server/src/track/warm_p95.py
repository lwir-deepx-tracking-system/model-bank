"""온도 Filter 의 숫자(상자 안 16 비트 값의 95 % 지점)를 np.percentile 과 같은 값으로, 더 가볍게 구한다 (2026-11-12).

np.percentile(v, 95) 의 기본(linear) 방식은 정렬했을 때 (n-1) x 0.95 번째 자리 앞뒤 두 값 사이를 잇는다.
여기서는 그 두 자리만 np.partition 으로 골라 같은 식으로 잇는다.
불러올 때 여러 크기의 시험 배열로 np.percentile 과 비트까지 같은지 스스로 확인하고, 하나라도 다르면 np.percentile 을 그대로 쓴다
(numpy 판에 따라 잇는 식의 반올림이 다를 수 있어서다). 어느 쪽인지는 MODE 에 남는다.
"""
import numpy as np

_Q = np.true_divide(95, 100)


def _fast(v):
    a = np.asarray(v).reshape(-1)
    n = a.size
    vi = (n - 1) * _Q
    lo = int(np.floor(vi)); hi = min(lo + 1, n - 1); g = vi - lo
    part = np.partition(a, (lo, hi)) if hi != lo else np.partition(a, lo)
    x0 = part[lo]; x1 = part[hi]
    d = np.subtract(x1, x0) * 1.0
    return float(x0 + d * g) if g < 0.5 else float(x1 - d * (1.0 - g))


def _slow(v):
    return float(np.percentile(v, 95))


def _selfcheck():
    rng = np.random.default_rng(12345)
    sizes = list(range(1, 64)) + [97, 128, 255, 500, 999, 1140, 2048, 3001, 4500, 9000, 15000]
    for n in sizes:
        for kind in range(3):
            if kind == 0: a = rng.integers(0, 65536, n).astype(np.uint16)
            elif kind == 1: a = rng.integers(40000, 40008, n).astype(np.uint16)
            else: a = np.full(n, 47749, np.uint16)
            if _fast(a) != _slow(a): return False
            if n >= 8:
                b = a.reshape(1, -1)[:, ::2]
                if _fast(b) != _slow(b): return False
    return True


try: _OK = _selfcheck()
except Exception: _OK = False
p95 = _fast if _OK else _slow
MODE = 'partition' if _OK else 'np.percentile (self-check failed)'
