"""실행이 끝난 뒤 결과 파일에 적는 요약 숫자(개수, 평균, p50, p95 ...) 계산."""
import numpy as np
from ..common.state import APP


def summ(key):
    v = np.array([r[key] for r in APP.rows if r.get(key) is not None])
    if not len(v): return None
    return dict(n=int(len(v)), mean=float(v.mean()), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)), max=float(v.max()), over_33ms_pct=float(100 * (v > 33.333).mean()))


def dist_z(v):
    v = np.asarray(v, np.float64)
    return dict(n=int(len(v)), mean=float(v.mean()), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)), max=float(v.max()), hist={int(x): int((v == x).sum()) for x in np.unique(v)}) if len(v) else None


def dist_(v):
    v = np.asarray(v, np.float64)
    return dict(n=int(len(v)), mean=float(v.mean()), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)), max=float(v.max())) if len(v) else None


def dist_r(v):
    v = np.asarray(v, np.float64)
    return dict(n=int(len(v)), mean=float(v.mean()), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)), max=float(v.max())) if len(v) else None


def dist(v):
    v = np.asarray(v, np.float64)
    return dict(n=int(len(v)), mean=float(v.mean()), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)), max=float(v.max())) if len(v) else None
