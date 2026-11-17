"""WSL (dx-com env /root/lig1ch/env). Trained body-box pose model (experiment B) -> a DXNN the CURRENT tracking server takes as it is.

The server (model-bank tracking_server/src/detect/model_scan.py) only lists detectors with a 640x480 or 640x640 uint8 input and exactly six
outputs box_s8 box_s16 box_s32 (4 ch) cls_s8 cls_s16 cls_s32 (16 ch), class 0 = person, and decodes them on the host (post.cpp).
So this export keeps the box and class branches of the pose model and leaves the keypoint branches out:
  box_s*  : the model's box branch = the BODY BOX it learned (head top to hips, may reach outside the picture)
  cls_s*  : channel 0 = the model's person logit, channels 1..15 = constant -20 (sigmoid ~ 2e-9, never above any threshold)
The server then tracks the body box. Keypoints are not used by the current server (needs a server change, see PLAN.md).
Aim point between face and chest = 0.372 of the body box height from its top (template, tools/skeleton_box.py) -> LWIR_GIMBAL_AIM_Y.
Conventions as the bank's build480.py / build_26x.py: raw 0..255 input, Div(255) inside, 97 own-camera calibration frames, ema, opt_level 1.

Usage: python -B build_bodybox6.py export|compile <workdir> <trained .pt> <name>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

STEP, WORK, PT, NAME = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
ONNX = WORK / (NAME + '.onnx')
CALIB = Path('/root/dxacc/calib')
NAMES6 = ['box_s8', 'box_s16', 'box_s32', 'cls_s8', 'cls_s16', 'cls_s32']
WANT = [(1, 4, 60, 80), (1, 4, 30, 40), (1, 4, 15, 20), (1, 16, 60, 80), (1, 16, 30, 40), (1, 16, 15, 20)]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def calibration():
    import cv2
    out = []
    for f in sorted(CALIB.glob('*.png')):
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        assert im is not None and im.shape == (640, 640, 3), (f, None if im is None else im.shape)
        out.append(np.ascontiguousarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)[80:560].transpose(2, 0, 1)[None].astype(np.float32)))
    return out


class SixHeads(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net; self.head = net.model[-1]

    def feats(self, images):
        x = images / 255.0; saved = []
        for m in self.net.model[:-1]:
            if m.f != -1: x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x); saved.append(x if m.i in self.net.save else None)
        return [saved[j] for j in self.head.f]

    def forward(self, images):
        f = self.feats(images); h = self.head
        boxes = [h.cv2[i](f[i]) for i in range(3)]
        cls = []
        for i in range(3):
            c = h.cv3[i](f[i])                                            # (1, 1, H, W) person logit
            cls.append(torch.cat([c, torch.full_like(c, -20.0).repeat(1, 15, 1, 1)], 1))
        return (*boxes, *cls)


if STEP == 'export':
    import onnx
    import onnxruntime as ort
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(4); WORK.mkdir(parents=True, exist_ok=True)
    net = YOLO(str(PT)).model.float().eval(); head = net.model[-1]
    info = dict(ultralytics=ultralytics.__version__, pt=str(PT), pt_sha=sha(PT), head=type(head).__name__, nc=int(head.nc), reg_max=int(head.reg_max),
                kpt_shape=list(head.kpt_shape), stride=[float(s) for s in head.stride])
    print('MODEL', json.dumps(info), flush=True)
    assert info['nc'] == 1 and info['reg_max'] == 1 and info['stride'] == [8.0, 16.0, 32.0], info
    wrap = SixHeads(net).eval(); raw = calibration()
    with torch.inference_mode():
        ref = [[v.numpy().copy() for v in wrap(torch.from_numpy(x))] for x in raw[:8]]
        own = head.forward_head(wrap.feats(torch.from_numpy(raw[0])), **head.one2many)
    assert [v.shape for v in ref[0]] == WANT, [v.shape for v in ref[0]]
    cat = lambda vs: np.concatenate([v.reshape(1, v.shape[1], -1) for v in vs], 2)
    err = dict(boxes=float(np.abs(cat(ref[0][0:3]) - own['boxes'].numpy()).max()), person=float(np.abs(cat([v[:, :1] for v in ref[0][3:6]]) - own['scores'].numpy()).max()))
    print('WRAPPER vs head.forward_head max abs', err, flush=True); assert max(err.values()) < 1e-5, err
    top = [float(1 / (1 + np.exp(-max(r[i][:, 0].max() for i in (3, 4, 5))))) for r in ref]
    print('max person score per calibration frame (first 8)', [round(v, 3) for v in top], flush=True)
    torch.onnx.export(wrap, (torch.from_numpy(raw[0]),), str(ONNX), input_names=['images'], output_names=NAMES6, opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(ONNX)), full_check=True)
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(str(ONNX), sess_options=so, providers=['CPUExecutionProvider']); chk = []
    for x, r in zip(raw[:8], ref):
        o = s.run(NAMES6, {'images': x})
        assert [v.shape for v in o] == WANT and all(v.dtype == np.float32 for v in o)
        chk.append(max(float(np.abs(p - q).max()) for p, q in zip(o, r)))
        assert all(np.allclose(p, q, atol=2e-2, rtol=1e-3) for p, q in zip(o, r)), chk
    print('PARITY onnx vs torch', chk, flush=True)
    (WORK / 'EXPORT.json').write_text(json.dumps(dict(name=NAME, onnx_sha=sha(ONNX), onnx_bytes=ONNX.stat().st_size, model=info, wrapper_err=err, parity_max_abs=chk,
                                                      max_person_score_first8=top, calibration=dict(dir=str(CALIB), files=len(raw))), indent=1))
    print('EXPORT DONE', ONNX.stat().st_size, flush=True)

elif STEP == 'compile':
    import dx_com
    raw = calibration()

    class Frames(torch.utils.data.Dataset):
        def __len__(self): return len(raw)
        def __getitem__(self, i): return torch.from_numpy(raw[i][0].copy())

    out = WORK / 'compiler'; t = time.monotonic()
    dx_com.compile(model=str(ONNX), output_dir=str(out), dataloader=torch.utils.data.DataLoader(Frames(), batch_size=1, num_workers=0, shuffle=False),
                   calibration_num=len(raw), calibration_method='ema', quantization_device='cpu', opt_level=1, aggressive_partitioning=False)
    files = list(out.glob('*.dxnn')); assert len(files) == 1, files
    res = dict(name=NAME, dxnn=str(files[0]), sha=sha(files[0]), bytes=files[0].stat().st_size, compile_s=round(time.monotonic() - t, 1), dx_com=dx_com.__version__,
               calibration_num=len(raw), calibration_method='ema', opt_level=1, aggressive_partitioning=False, onnx_sha=sha(ONNX))
    (WORK / 'RESULT.json').write_text(json.dumps(res, indent=1))
    print('COMPILE DONE', json.dumps(res), flush=True)
