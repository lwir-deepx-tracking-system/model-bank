"""M1: YOLO26s NMS-free DXNN whose output0 = [1,300,6] (x1,y1,x2,y2,score,cls in the 640x640 canvas) -- accuracy on the FLIR video_thermal_test frames.
Same frame path as m1_flir_eval_s.py D640 (pad 64 rows top/bottom with 114 -> 640x640 -> GRAY2RGB -> one NPU core), no NMS (the model already outputs the final rows).
Usage (DX-M1): python3 m1_eval_e2e.py <model.dxnn> <out_dir> [max_frames]
Writes <out_dir>/PRED_E2E.jsonl (all 300 rows, conf >= .001) and RESULT.json with latency. Score locally with score_m1_s.py-style code."""
import os, sys, json, time
from pathlib import Path
import numpy as np, cv2, dx_engine as dx
model, out = Path(sys.argv[1]), Path(sys.argv[2]); maxf = int(sys.argv[3]) if len(sys.argv) > 3 else 0
out.mkdir(parents=True, exist_ok=False)
DATA = Path('/home/orangepi/workspace/runtime-latency-chatgpt-20261022/.codex_latency_checks/g3_g4_m1_latency_01/claude_flir_video_thermal_test')
coco = json.loads((DATA / 'coco.json').read_text(encoding='utf-8'))
ims = sorted(coco['images'], key=lambda i: i['file_name'])
if maxf: ims = ims[:maxf]
o = dx.InferenceOption(); o.devices = [0]; o.bound_option = dx.InferenceOption.BOUND_OPTION.NPU_0; o.use_ort = True; o.buffer_count = 1
e = dx.InferenceEngine(str(model), o)
info = e.get_input_tensors_info()[0]; assert tuple(int(v) for v in info['shape']) == (1, 640, 640, 3), info
outs = e.get_output_tensors_info(); sizes = e.get_output_tensor_sizes(); print('outputs', [(v['name'], v['shape']) for v in outs], sizes, flush=True)
buf = np.empty(sum(sizes) // 4, np.float32)
lat = []; fh = (out / 'PRED_E2E.jsonl').open('w')
for k, im in enumerate(ims):
    g = cv2.imread(str(DATA / im['file_name']), cv2.IMREAD_GRAYSCALE)
    if g is None or g.shape != (512, 640): continue
    x = cv2.cvtColor(cv2.copyMakeBorder(g, 64, 64, 0, 0, cv2.BORDER_CONSTANT, value=114), cv2.COLOR_GRAY2RGB)[None]
    t0 = time.perf_counter_ns(); r = e.wait(e.run_async(x, user_arg=None, output_buffer=buf)); lat.append((time.perf_counter_ns() - t0) / 1e6)
    y = [np.asarray(v, np.float32) for v in r if np.asarray(v).size == 1800][0].reshape(-1, 6); y = y[y[:, 4] >= 0.001]
    fh.write(json.dumps(dict(image_id=Path(im['file_name']).name, xyxy=np.round(y[:, :4], 2).tolist(), conf=np.round(y[:, 4], 5).tolist(), cls=y[:, 5].astype(int).tolist()), separators=(',', ':')) + '\n')
fh.close(); a = np.array(lat)
(out / 'RESULT.json').write_text(json.dumps(dict(model=str(model), frames=len(lat), call_ms_p50=float(np.percentile(a, 50)), call_ms_p95=float(np.percentile(a, 95)), output=[(v['name'], v['shape']) for v in outs]), indent=1))
print('DONE frames', len(lat), 'call ms p50 %.2f p95 %.2f' % (np.percentile(a, 50), np.percentile(a, 95)), flush=True)
