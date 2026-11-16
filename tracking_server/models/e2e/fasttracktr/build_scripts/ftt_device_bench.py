"""Device timing of the live FastTrackTr graphs, stage by stage (2026-11-15, user: "속도 더 빠르게 못 하나?").
Runs AE (backbone + encoder), decoder layer 0 and layer 1 on recorded frames exactly as track/e2e_live.py does (NPU core given), N frames,
and prints mean / p90 ms of: picture prep, AE, b0, b1, total. Two file sets can be compared (e.g. normal vs aggressive partitioning).
Usage (device, server stopped or idle core): python3 ftt_device_bench.py <frames dir with png/jpg> <n> <core> name=<ae.dxnn>,<b0.dxnn>,<b1.dxnn> ..."""
import glob, json, sys, time
import numpy as np
from PIL import Image
import dx_engine as dx

fd, n, core = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sets = dict(a.split('=', 1) for a in sys.argv[4:])
files = sorted(glob.glob(fd + '/*.png') + glob.glob(fd + '/*.jpg'))[:n]
frames = []
for f in files:
    g = np.asarray(Image.open(f).convert('L'))
    if g.shape == (512, 640): g = g[16:496]
    frames.append(g)


def eng(path):
    o = dx.InferenceOption(); o.devices = [0]; o.bound_option = getattr(dx.InferenceOption.BOUND_OPTION, 'NPU_%d' % core); o.use_ort = True
    return dx.InferenceEngine(path, o)


out = {}
for name, trip in sets.items():
    pa, p0, p1 = trip.split(','); ea, ebs = eng(pa), [eng(p0), eng(p1)]
    ae_out = [t['name'] for t in ea.get_output_tensors_info()]
    b_in = [[(t['name'], tuple(t['shape']), t['dtype']) for t in e.get_input_tensors_info()] for e in ebs]
    b_out = [[t['name'] for t in e.get_output_tensors_info()] for e in ebs]
    traj = np.zeros((1, 300, 256), np.float32); T = {k: [] for k in ('prep', 'ae', 'b0', 'b1', 'total')}
    for k, g in enumerate(frames):
        t0 = time.perf_counter(); x = np.ascontiguousarray(np.repeat(g[None, :, :, None], 3, axis=3)); t1 = time.perf_counter()
        ao = dict(zip(ae_out, [np.asarray(v, np.float32) for v in ea.run([x])])); t2 = time.perf_counter()
        st = dict(kv=ao['kv4'].reshape(1, -1, 256), tgt=ao['tgt4'].reshape(1, -1, 256), ref_unact=ao['ref4'].reshape(1, -1, 4), traj=traj); tb = []
        for e, ins, outs in zip(ebs, b_in, b_out):
            ts = time.perf_counter(); r = dict(zip(outs, [np.asarray(v, np.float32) for v in e.run([np.ascontiguousarray(st[nm].reshape(s).astype(dt)) for nm, s, dt in ins])]))
            tb.append(time.perf_counter() - ts)
            if 'tgt_out' in r: st['tgt'], st['ref_unact'] = r['tgt_out'], r['ref_out']
        traj = r['hs'].reshape(1, 300, 256); t3 = time.perf_counter()
        if k >= 3:   # first frames warm the engines
            T['prep'].append(t1 - t0); T['ae'].append(t2 - t1); T['b0'].append(tb[0]); T['b1'].append(tb[1]); T['total'].append(t3 - t0)
    out[name] = {k: dict(mean=round(1e3 * float(np.mean(v)), 2), p90=round(1e3 * float(np.percentile(v, 90)), 2)) for k, v in T.items()}
    out[name]['frames'] = len(T['total']); print(name, json.dumps(out[name]), flush=True)
    del ea, ebs
print(json.dumps(out))
