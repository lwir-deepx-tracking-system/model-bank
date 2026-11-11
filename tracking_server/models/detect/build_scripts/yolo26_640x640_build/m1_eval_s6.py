"""M1: six-head 640x640 DXNN (box_s8/16/32, cls_s8/16/32) accuracy + latency on the FLIR video_thermal_test frames.
Frame path as m1_eval_e2e.py (pad 64 rows with 114 -> 640x640 -> GRAY2RGB -> one NPU core). Host: decode (anchor -/+ dist) * stride, sigmoid, class-aware NMS IoU .7, conf >= .001, max 300.
Checks float32 output shapes against tensor_requirements.md. Usage: python3 -B m1_eval_s6.py <model.dxnn> <out_dir> [max_frames]. Writes PRED_E2E.jsonl and RESULT.json."""
import os, sys, json, time
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import numpy as np, cv2, dx_engine as dx
model, out = Path(sys.argv[1]), Path(sys.argv[2]); maxf = int(sys.argv[3]) if len(sys.argv) > 3 else 0
out.mkdir(parents=True, exist_ok=False)
DATA = Path('<YOUR_PATH>')
ims = sorted(json.loads((DATA / 'coco.json').read_text(encoding='utf-8'))['images'], key=lambda i: i['file_name'])
if maxf: ims = ims[:maxf]
o = dx.InferenceOption(); o.devices = [0]; o.bound_option = dx.InferenceOption.BOUND_OPTION.NPU_0; o.use_ort = True; o.buffer_count = 1
e = dx.InferenceEngine(str(model), o)
info = e.get_input_tensors_info()[0]; assert tuple(int(v) for v in info['shape']) == (1, 640, 640, 3), info
outs = e.get_output_tensors_info(); print('outputs', outs, flush=True)
want = {'box_s8': (1, 4, 80, 80), 'box_s16': (1, 4, 40, 40), 'box_s32': (1, 4, 20, 20), 'cls_s8': (1, 16, 80, 80), 'cls_s16': (1, 16, 40, 40), 'cls_s32': (1, 16, 20, 20)}
names = [v['name'] for v in outs]; assert sorted(names) == sorted(want), names
for v in outs: assert tuple(int(d) for d in v['shape']) == want[v['name']] and np.dtype(v['dtype']) == np.float32, v
A, S = [], []
for s in (8, 16, 32):
    n = 640 // s; y, x = np.meshgrid(np.arange(n, dtype=np.float32) + .5, np.arange(n, dtype=np.float32) + .5, indexing='ij'); A.append(np.stack([x.ravel(), y.ravel()], 1)); S.append(np.full(n * n, s, np.float32))
A = np.concatenate(A); S = np.concatenate(S)[:, None]
def nms(b, sc, c):
    off = b + (c[:, None].astype(np.float32) * 4096); x1, y1, x2, y2 = off.T; ar = (x2 - x1).clip(0) * (y2 - y1).clip(0); ord_ = np.argsort(-sc, kind='stable'); keep = []
    while ord_.size and len(keep) < 300:
        i = ord_[0]; keep.append(i); r = ord_[1:]
        w = (np.minimum(x2[i], x2[r]) - np.maximum(x1[i], x1[r])).clip(0); h = (np.minimum(y2[i], y2[r]) - np.maximum(y1[i], y1[r])).clip(0); it = w * h
        ord_ = r[it / (ar[i] + ar[r] - it + 1e-9) <= .7]
    return np.array(keep, int)
lat = []; fh = (out / 'PRED_E2E.jsonl').open('w')
for im in ims:
    g = cv2.imread(str(DATA / im['file_name']), cv2.IMREAD_GRAYSCALE)
    if g is None or g.shape != (512, 640): continue
    x = cv2.cvtColor(cv2.copyMakeBorder(g, 64, 64, 0, 0, cv2.BORDER_CONSTANT, value=114), cv2.COLOR_GRAY2RGB)[None]
    t0 = time.perf_counter_ns(); r = e.wait(e.run_async(x, user_arg=None)); lat.append((time.perf_counter_ns() - t0) / 1e6)
    d = dict(zip(names, r))
    bx = np.concatenate([d[k].reshape(4, -1) for k in ('box_s8', 'box_s16', 'box_s32')], 1).T; lg = np.concatenate([d[k].reshape(16, -1) for k in ('cls_s8', 'cls_s16', 'cls_s32')], 1).T
    xy = np.concatenate([(A - bx[:, :2]) * S, (A + bx[:, 2:]) * S], 1); sc = 1 / (1 + np.exp(-np.clip(lg, -50, 50)))
    ai, ci = np.nonzero(sc >= .001); v = sc[ai, ci]
    if v.size > 5000: t = np.argpartition(-v, 5000)[:5000]; ai, ci, v = ai[t], ci[t], v[t]
    k = nms(xy[ai], v, ci)
    fh.write(json.dumps(dict(image_id=Path(im['file_name']).name, xyxy=np.round(xy[ai][k], 2).tolist(), conf=np.round(v[k], 5).tolist(), cls=ci[k].tolist()), separators=(',', ':')) + '\n')
fh.close(); a = np.array(lat)
(out / 'RESULT.json').write_text(json.dumps(dict(model=str(model), frames=len(lat), call_ms_p50=float(np.percentile(a, 50)), call_ms_p95=float(np.percentile(a, 95)), output=[(v['name'], [int(d) for d in v['shape']], np.dtype(v['dtype']).name) for v in outs]), indent=1))
print('DONE frames', len(lat), 'call ms p50 %.2f p95 %.2f' % (np.percentile(a, 50), np.percentile(a, 95)), flush=True)
