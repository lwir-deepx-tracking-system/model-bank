"""WSL (dx-com env): compile the trained FastTrackTr device graphs written by ftt_trained_export.py (AE, decoder layer 0, decoder layer 1)
with their calibration files, the same settings as the NPU port (ema, opt 1, no aggressive partitioning; several inputs = dict by ONNX input name).
Usage: python ftt_compile.py <export dir> [ae b0 b1]"""
import os, sys, time, json
import numpy as np, torch, onnx, dx_com

d = os.path.abspath(sys.argv[1]); parts = sys.argv[2:] or ['ae', 'b0', 'b1']; res = {}
for part in parts:
    onx = os.path.join(d, 'ftt_npu_trained_%s.onnx' % part); cal = np.load(os.path.join(d, 'calib_%s.npz' % part))
    names = [i.name for i in onnx.load(onx).graph.input]
    if part == 'ae': items = [(cal['image'][i],) for i in range(len(cal['image']))]
    else: items = [(cal['tgt'][i], cal['ref'][i], cal['kv'][i], cal['traj'][i]) for i in range(len(cal['tgt']))]

    class Cal(torch.utils.data.Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, i):
            t_ = [torch.from_numpy(np.ascontiguousarray(v)) for v in items[i]]
            return t_[0] if len(t_) == 1 else {n: v for n, v in zip(names, t_)}
    od = os.path.join(d, 'part_' + part); os.makedirs(od, exist_ok=True); t = time.monotonic()
    dx_com.compile(model=onx, output_dir=od, dataloader=torch.utils.data.DataLoader(Cal(), batch_size=1, num_workers=0, shuffle=False),
                   calibration_num=len(items), calibration_method='ema', quantization_device='cpu', opt_level=1, aggressive_partitioning=False)
    f = [x for x in os.listdir(od) if x.endswith('.dxnn')]; res[part] = dict(files=f, seconds=round(time.monotonic() - t, 1))
    print('COMPILED', part, json.dumps(res[part]), flush=True)
json.dump(res, open(os.path.join(d, 'COMPILE.json'), 'w'), indent=1); print('ALL COMPILED')
