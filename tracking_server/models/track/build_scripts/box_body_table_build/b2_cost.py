"""body_v2/scripts/b2_cost.py - PC cost of box_refine (server code of the worktree) per frame for 10 / 20 / 30 people, one thread.
apply() with the server glue (rows + matched detections + a camera warp, as live_publish calls it) for off / size / body (pose detector key, plain
detector key), and body with every person inside a detector-switch transition. 400 frames, median of frames 50.., best of 3 runs.
Usage: python b2_cost.py <tracking_server dir> <out.json>"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import json
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, sys.argv[1])
from src.track import box_refine as BR   # noqa: E402
from src.common.state import APP           # noqa: E402

POSE = 'bank:yolo26m_pose_otp2cocomotllvip_20261115_bodybox_farc2_kpt_640x480'
PLAIN = 'bank:yolo26x_flir_nms_640x480'


def one(mode, n, det, switch=False, frames=400, seed=0):
    rng = np.random.default_rng(seed); BR.BR.update(mode='off', blend_s=0.0); BR.BR.pop('obj', None)
    APP.args = SimpleNamespace(det=det); APP.CTL = dict(lead_motion=None); APP.tracker = object(); BR.BR['dir'] = os.path.join(sys.argv[1], 'models', 'track')
    if mode != 'off': ok, txt, ap = BR.parse_request({'box_refine': [mode]}); assert ok, txt; ap()
    cx = rng.uniform(40, 600, n); cy = rng.uniform(150, 400, n); h = np.exp(rng.uniform(np.log(25), np.log(300), n)); ar = np.where(h < 90, rng.uniform(0.33, 0.45, n), 0.62); ts = []
    for k in range(frames):
        cx += rng.normal(0, 1, n); hk = h * (1 + rng.normal(0, 0.03, n)); w = ar * hk
        rows = [[cx[i] - w[i] / 2, cy[i] - hk[i] / 2, cx[i] + w[i] / 2, cy[i] + hk[i] / 2, i + 1, 0.9, 0, i] for i in range(n)]
        dets = np.array([[r[0] + 1, r[1] + 1, r[2] + 1, r[3] + 2, 0.9, 0.] for r in rows])
        APP.CTL['lead_motion'] = (0, 0, np.array([[1., 0, 0.5], [0, 1., 0]]))
        if switch: APP.args.det = POSE if (k // 10) % 2 == 0 else PLAIN   # every 10 frames: every person starts a transition
        t0 = time.perf_counter(); out = BR.apply(rows, dets, k / 25.); ts.append(time.perf_counter() - t0)
        assert len(out) == n
    assert BR.BR['err'] == '', BR.BR['err']
    return float(np.median(ts[50:]) * 1000.0), float(np.percentile(ts[50:], 99) * 1000.0)


def main():
    res = {}
    for name, mode, det, sw in (('off', 'off', POSE, False), ('size', 'size', POSE, False), ('body_pose', 'body', POSE, False), ('body_plain', 'body', PLAIN, False), ('body_switching', 'body', POSE, True)):
        for n in (10, 20, 30):
            best = min((one(mode, n, det, sw, seed=s) for s in range(3)), key=lambda x: x[0]); res['%s|%d' % (name, n)] = dict(median_ms=round(best[0], 3), p99_ms=round(best[1], 3), per_person_ms=round(best[0] / n, 4))
            print(name, n, res['%s|%d' % (name, n)], flush=True)
    import platform
    res['pc'] = dict(cpu=platform.processor(), python=platform.python_version(), numpy=np.__version__, threads=1)
    open(sys.argv[2], 'w').write(json.dumps(res, indent=1)); print('COST_DONE', flush=True)


if __name__ == '__main__':
    main()
