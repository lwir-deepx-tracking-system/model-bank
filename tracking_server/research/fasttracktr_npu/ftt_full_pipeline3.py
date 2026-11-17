"""Device: FastTrackTr NPU variant, fastest layout: AE (backbone + encoder + encoder head, NHWC outputs, pure NPU) on core(s)
A, the decoder chunks on core B; the host only reshapes. Random weights / frames: speed only.
Usage: python ftt_full_pipeline3.py <ae.dxnn> <b0.dxnn> ... [--a-cores 1] [--b-core 2] [--a-inflight 2] [--frames 120]
--a-cores 1,0 = one AE engine per listed core, frames dealt round-robin."""
import json
import sys
import time

import dx_engine as dx
import numpy as np

args = sys.argv[1:]
opt = {'--a-cores': '1', '--b-core': '2', '--a-inflight': '2', '--frames': '120'}
files, i = [], 0
while i < len(args):
    if args[i] in opt:
        opt[args[i]] = args[i + 1]; i += 2
    else:
        files.append(args[i]); i += 1
ACS, BC, K, N = [int(c) for c in opt['--a-cores'].split(',')], int(opt['--b-core']), int(opt['--a-inflight']), int(opt['--frames'])
BO = dx.InferenceOption.BOUND_OPTION


def engine(p, core, buf):
    o = dx.InferenceOption(); o.devices = [0]; o.bound_option = getattr(BO, 'NPU_%d' % core); o.use_ort = True; o.buffer_count = buf
    return dx.InferenceEngine(p, o)


eas = [engine(files[0], c, K + 1) for c in ACS]
ebs = [engine(p, BC, 2) for p in files[1:]]
names_a = [t['name'] for t in eas[0].get_output_tensors_info()]
rng = np.random.default_rng(0)
ia = eas[0].get_input_tensors_info()[0]
imgs = [np.ascontiguousarray(rng.integers(0, 256, size=ia['shape']).astype(ia['dtype'])) for _ in range(4)]


def run_b(ao, traj):
    d = dict(zip(names_a, [np.asarray(v, np.float32) for v in ao]))
    st = dict(kv=d['kv4'].reshape(1, -1, 256), tgt=d['tgt4'].reshape(1, -1, 256), ref_unact=d['ref4'].reshape(1, -1, 4), traj=traj)
    for e in ebs:
        out = e.run([np.ascontiguousarray(st[t['name']].reshape(t['shape']).astype(t['dtype'])) for t in e.get_input_tensors_info()])
        res = dict(zip([t['name'] for t in e.get_output_tensors_info()], [np.asarray(v, np.float32) for v in out]))
        if 'tgt_out' in res:
            st['tgt'], st['ref_unact'] = res['tgt_out'], res['ref_out']
    return res['hs'].reshape(1, 300, 256)


def stats(name, lat, wall, extra=None):
    w = sorted(lat[3:])
    d = dict(mode=name, fps=round((len(lat) - 3) / wall, 1), latency_mean_ms=round(sum(w) / len(w), 1), latency_p95_ms=round(w[int(0.95 * (len(w) - 1))], 1))
    d.update(extra or {})
    return d


for e in eas:
    for _ in range(3):
        run_b(e.run([imgs[0]]), np.zeros((1, 300, 256), np.float32))
traj, lat, la, lb = np.zeros((1, 300, 256), np.float32), [], [], []
for f in range(N):
    t0 = time.perf_counter(); ao = eas[0].run([imgs[f % 4]]); t1 = time.perf_counter(); traj = run_b(ao, traj); t2 = time.perf_counter()
    lat.append((t2 - t0) * 1e3); la.append((t1 - t0) * 1e3); lb.append((t2 - t1) * 1e3)
    if f == 2:
        tr0 = time.perf_counter()
seq = stats('sequential (one frame at a time)', lat, time.perf_counter() - tr0, dict(AE_ms=round(float(np.mean(la[3:])), 1), decoder_ms=round(float(np.mean(lb[3:])), 1)))
traj, lat, q, ts = np.zeros((1, 300, 256), np.float32), [], [], []
nin = K * len(eas)
for f in range(min(nin, N)):
    e = eas[f % len(eas)]; ts.append(time.perf_counter()); q.append((e, e.run_async([imgs[f % 4]])))
for f in range(N):
    e_, r_ = q.pop(0); ao = e_.wait(r_)
    if f + nin < N:
        e = eas[(f + nin) % len(eas)]; ts.append(time.perf_counter()); q.append((e, e.run_async([imgs[(f + nin) % 4]])))
    traj = run_b(ao, traj)
    lat.append((time.perf_counter() - ts[f]) * 1e3)
    if f == 2:
        tr0 = time.perf_counter()
pip = stats('pipelined: AE on cores %s (%d in flight each), decoder core %d' % (ACS, K, BC), lat, time.perf_counter() - tr0)
# paced at 30 fps (the camera), one frame at a time: latency from 'frame arrives' to 'result', and whether it keeps up
traj, lat = np.zeros((1, 300, 256), np.float32), []
period, t_start = 1 / 30.0, time.perf_counter()
for f in range(N):
    arrive = t_start + f * period
    while time.perf_counter() < arrive:
        time.sleep(0.0005)
    traj = run_b(eas[f % len(eas)].run([imgs[f % 4]]), traj)
    lat.append((time.perf_counter() - arrive) * 1e3)
w = sorted(lat[3:])
paced = dict(mode='camera pace 30 fps, one frame at a time', kept_up=bool(lat[-1] < 2 * period * 1e3 + 80),
             latency_mean_ms=round(sum(w) / len(w), 1), latency_p95_ms=round(w[int(0.95 * (len(w) - 1))], 1), last_ms=round(lat[-1], 1))
print(json.dumps(dict(ae=files[0].split('/')[-1], chunks=len(ebs), results=[seq, pip, paced]), indent=1))
