"""v2/scripts/v2_cpu.py - CPU cost of box_refine.apply() on this PC (PREREG_V2.json cpu): synthetic rows of 10 / 20 / 30 walking people with matched
detections, keypoints and a camera warp; 350 frames per run; median ms per frame; best of 3 runs (the smallest median). Modes off, size, euro, gru.
Writes v2/results/cpu.json. Usage: python v2_cpu.py
"""
import json
import platform
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import v2_lib as L
from src.common.state import APP
from src.detect import model_scan, detect_frame

BR = L.BRM


def frames(n_people, n_frames, rng):
    x0 = rng.uniform(40, 600, n_people); y0 = rng.uniform(120, 400, n_people); h0 = rng.uniform(30, 220, n_people); vx = rng.normal(0, 3, n_people)
    for k in range(n_frames):
        rows = []; dets = []; boxes = []; kpts = []
        for i in range(n_people):
            cx = x0[i] + vx[i] * k; h = h0[i] * (1 + 0.002 * k); w = 0.4 * h; cy = y0[i]
            t = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]; d = [v + rng.normal(0, 1.5) for v in t]
            rows.append(t + [i + 1, 0.9, 0, i]); dets.append(d + [0.9, 0.]); boxes.append(d)
            kpts.append(np.c_[np.full(17, cx), np.linspace(cy - h / 2, cy + h / 2, 17), np.full(17, 0.9)])
        yield rows, np.array(dets), dict(boxes=np.array(boxes), kpts=np.array(kpts))


def bench(mode, n_people, rng_seed):
    BR.BR.update(mode=mode, blend_s=0.0, errors=0, reset_key=None); BR.BR.pop('obj', None); rng = np.random.default_rng(rng_seed); ms = []
    H = np.array([[1.0, 0.0, 1.5], [0.0, 1.0, -0.5]])
    for k, (rows, dets, kp) in enumerate(frames(n_people, 350, rng)):
        detect_frame.KP['cur'] = kp; APP.CTL['lead_motion'] = (0, 0, H)
        t0 = time.perf_counter(); out = BR.apply(rows, dets, 0.04 * k + 1.0); ms.append((time.perf_counter() - t0) * 1000.0)
        assert mode == 'off' or out is not rows
    assert BR.BR['errors'] == 0, BR.BR['err']
    return float(np.median(ms[10:]))


def main():
    t0 = time.time(); APP.args = SimpleNamespace(det='near_kpt'); APP.CTL = dict(lead_motion=None); APP.tracker = object()
    BR.BR['model'] = L.gru_model(); res = dict(machine=platform.processor(), python=platform.python_version(), numpy=np.__version__, size=dict(BR.SIZE), modes={})
    with patch.object(model_scan, 'KPT_DETS', {'near_kpt'}):
        for mode in ('off', 'size', 'euro', 'gru'):
            res['modes'][mode] = {}
            for n in (10, 20, 30):
                runs = [bench(mode, n, s) for s in range(3)]; best = min(runs)
                res['modes'][mode][n] = dict(ms_frame=round(best, 4), ms_person=round(best / n, 4), runs=[round(r, 4) for r in runs])
                print('CPU %-5s people %2d  %.3f ms per frame  %.4f ms per person  (runs %s)' % (mode, n, best, best / n, ', '.join('%.3f' % r for r in runs)), flush=True)
    res['seconds'] = round(time.time() - t0, 1)
    (L.V2 / 'results' / 'cpu.json').write_text(json.dumps(res, indent=1)); print('CPU_DONE', flush=True)


if __name__ == '__main__':
    main()
