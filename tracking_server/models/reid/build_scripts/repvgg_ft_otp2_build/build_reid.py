"""WSL (dx-com env /root/lig1ch/env). Trained appearance model (reid_partial.py final_ema.pt) -> DXNN in the board's re-ID format.

Format (models/reid/README.md, repvgg_tmot_camel_reid_compile_metadata.json): ONNX input `images` [1,3,128,64] float raw 0..255 (the ImageNet
normalisation is inside the model), output `embedding` [1,128] float32, not normalised (the server L2-normalises). DX-COM: 256 calibration crops,
ema, opt_level 1 -> the DXNN input becomes uint8 NHWC [1,128,64,3].
Before export the QARepVGG body is fused into plain 3x3 conv + ReLU (reid_v2_common.deploy_model) and the fused copy is checked against the trained one.

Workdir layout: <work>/r7_common.py, <work>/dep/reid_v2_common.py, <work>/final_ema.pt, <work>/calib_256.npy (uint8 [256,128,64]).
Usage: REID7_ROOT=<work> python -B build_reid.py export|compile <work> <name>
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

STEP, WORK, NAME = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
os.environ.setdefault('REID7_ROOT', str(WORK)); sys.path.insert(0, str(WORK)); sys.path.insert(0, str(WORK / 'dep'))
import r7_common as C  # noqa: E402
from reid_v2_common import deploy_model  # noqa: E402

ONNX = WORK / (NAME + '.onnx')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


calib = np.load(str(WORK / 'calib_256.npy')); assert calib.shape == (256, 128, 64) and calib.dtype == np.uint8, (calib.shape, calib.dtype)
X = torch.from_numpy(calib).float().unsqueeze(1).expand(-1, 3, -1, -1).contiguous()

if STEP == 'export':
    import onnx
    import onnxruntime as ort
    torch.set_num_threads(4)
    ck = torch.load(str(WORK / 'final_ema.pt'), map_location='cpu', weights_only=False)
    m = C.build_student(torch); m.load_state_dict(ck['state_dict'], strict=True); m.eval()
    d = deploy_model(m).eval()
    with torch.no_grad():
        a, b = m(X[:64]), d(X[:64])
    err = float((a - b).abs().max()); cos = float(torch.nn.functional.cosine_similarity(a, b).min())
    print('FUSE max abs %.2e, min cosine %.6f' % (err, cos), flush=True); assert cos > 0.9999, cos
    torch.onnx.export(d, (X[:1],), str(ONNX), input_names=['images'], output_names=['embedding'], opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(ONNX)), full_check=True)
    s = ort.InferenceSession(str(ONNX), providers=['CPUExecutionProvider'])
    o = np.concatenate([s.run(['embedding'], {'images': X[i:i + 1].numpy()})[0] for i in range(16)])
    with torch.no_grad(): r = d(X[:16]).numpy()
    perr = float(np.abs(o - r).max()); print('PARITY onnx vs torch max abs %.2e' % perr, flush=True); assert perr < 1e-3, perr
    (WORK / 'EXPORT.json').write_text(json.dumps(dict(name=NAME, weights_sha=sha(WORK / 'final_ema.pt'), onnx_sha=sha(ONNX), fuse_max_abs=err, fuse_min_cos=cos,
                                                      parity_max_abs=perr, calib_sha=sha(WORK / 'calib_256.npy')), indent=1))
    print('EXPORT DONE', ONNX.stat().st_size, flush=True)

elif STEP == 'compile':
    import dx_com

    class Frames(torch.utils.data.Dataset):
        def __len__(self): return len(X)
        def __getitem__(self, i): return X[i].clone()

    out = WORK / 'compiler'; t = time.monotonic()
    dx_com.compile(model=str(ONNX), output_dir=str(out), dataloader=torch.utils.data.DataLoader(Frames(), batch_size=1, num_workers=0, shuffle=False),
                   calibration_num=256, calibration_method='ema', quantization_device='cpu', opt_level=1, aggressive_partitioning=False)
    files = list(out.glob('*.dxnn')); assert len(files) == 1, files
    res = dict(name=NAME, dxnn=str(files[0]), sha=sha(files[0]), bytes=files[0].stat().st_size, compile_s=round(time.monotonic() - t, 1), dx_com=dx_com.__version__,
               calibration_num=256, calibration_method='ema', opt_level=1, onnx_sha=sha(ONNX))
    (WORK / 'RESULT.json').write_text(json.dumps(res, indent=1))
    print('COMPILE DONE', json.dumps(res), flush=True)
