"""PC 에서 장치 없이 돌려 보기 위한 가짜 NPU 엔진 (옵션 --mock)."""
import os
import sys
from pathlib import Path
import numpy as np
from ..common.shared_values import HERE, SHAPE, SHAPE_TT, SHAPE_RT480, anchors


# ------------------------------------------------------------------------------------------------------------ mock (PC)

def make_mock_dx(args):
    """PC dry run: fake dx_engine - detector tiles via the TorchScript B0 trace (float, 0..1), ViT via the folded student."""
    for o_ in ('mock_data', 'mock_trace', 'mock_vit_src') + (('mock_tt_onnx',) if args.det == 'tt75' else ()) + (('mock_rt480_onnx',) if args.det == 'rt480' else ()):   # public repository: these PC paths have no default any more
        if not getattr(args, o_): raise SystemExit('--mock needs --%s <PC path>: this option has no default in the public repository' % o_.replace('_', '-'))
    import types, torch
    sys.path.insert(0, str(HERE)); sys.path.insert(0, args.mock_vit_src)
    from vit_student import build_from_checkpoint
    mdev = torch.device(args.mock_device)
    trace = torch.jit.load(args.mock_trace, map_location='cpu').eval().to(mdev)
    ck = torch.load(args.mock_student, map_location='cpu', weights_only=False); student = build_from_checkpoint(ck).to(mdev)
    dx = types.ModuleType('dx_engine'); NS = types.SimpleNamespace
    dx.InferenceOption = type('InferenceOption', (), dict(BOUND_OPTION=NS(NPU_0='NPU_0', NPU_1='NPU_1', NPU_2='NPU_2'), __init__=lambda self: None))
    class _St:
        def get_memory_used(self): return 0
        def get_core_utilization(self, i): return 0
        def get_temperature(self, i): return -1
        def get_npu_clock(self, i): return -1
    dx.DeviceStatus = NS(get_current_status=lambda i: _St())
    class Configuration:
        ITEM = NS(CUSTOM_INTRA_OP_THREADS='a', CUSTOM_INTER_OP_THREADS='b'); ATTRIBUTE = NS(CUSTOM_INTRA_OP_THREADS_NUM='c', CUSTOM_INTER_OP_THREADS_NUM='d')
        def set_attribute(self, *a): pass
        def set_enable(self, *a): pass
        def get_version(self): return 'MOCK(torch %s)' % torch.__version__
    dx.Configuration = Configuration
    tt_sess = {}
    if args.det == 'tt75':      # two-tower mock through onnxruntime (raw255 ONNX: float32 NCHW 0..255 -> boxes (1,4,6300), scores (1,16,6300))
        import onnxruntime as ort
        for n in ('TTA75', 'TTB75'):
            so = ort.SessionOptions(); so.intra_op_num_threads = 2
            tt_sess[n] = ort.InferenceSession(str(Path(args.mock_tt_onnx) / (n + '.onnx')), sess_options=so, providers=['CPUExecutionProvider'])
    if args.det == 'rt480':     # new m RT480 tiles through onnxruntime (kit raw255 ONNX: float32 NCHW 0..255 -> boxes (1,4,3780), scores (1,16,3780))
        import onnxruntime as ort
        for n in ('NEWMRT480A', 'NEWMRT480B'):
            so = ort.SessionOptions(); so.intra_op_num_threads = 2
            tt_sess[n] = ort.InferenceSession(str(Path(args.mock_rt480_onnx) / n / (n + '.onnx')), sess_options=so, providers=['CPUExecutionProvider'])
    class InferenceEngine:
        def __init__(self, path, opt):
            self.name = Path(path).stem; self.tt = self.name in SHAPE_TT or self.name in SHAPE_RT480; self.vit = self.name not in SHAPE and not self.tt; self.jobs = {}; self.k = 0
            self.H, self.W = (ck['cfg']['img_hw'] if self.vit else ({**SHAPE_TT, **SHAPE_RT480}[self.name] if self.tt else SHAPE[self.name]))
            self.n = 0 if self.vit else anchors(self.H, self.W)
        def get_input_tensors_info(self): return [dict(name='images', shape=[1, self.H, self.W, 3], dtype=np.uint8)]
        def get_output_tensors_info(self): return [dict(name='embedding', shape=[1, 128])] if self.vit else [dict(name='boxes'), dict(name='scores')]
        def get_output_tensor_sizes(self): return [128 * 4] if self.vit else [4 * self.n * 4, 16 * self.n * 4]
        def run_async(self, x, user_arg=None, output_buffer=None):
            assert x.dtype == np.uint8 and x.shape == (1, self.H, self.W, 3), (x.shape, self.H, self.W)
            if self.tt:
                b, s = tt_sess[self.name].run(['boxes', 'scores'], {'images': np.ascontiguousarray(x.transpose(0, 3, 1, 2)).astype(np.float32)})
                nb = b.size; output_buffer[:nb] = b.ravel(); output_buffer[nb:nb + s.size] = s.ravel()
                self.k += 1; self.jobs[self.k] = [output_buffer[:nb].reshape(b.shape), output_buffer[nb:nb + s.size].reshape(s.shape)]; return self.k
            xt = (torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float() / 255.0).to(mdev)
            with torch.inference_mode():
                if self.vit:
                    e = student(xt).cpu().numpy().ravel(); output_buffer[:128] = e; outs = [output_buffer[:128].reshape(1, 128)]
                else:
                    b, s = [v.cpu().numpy() for v in trace(xt)]; nb = b.size; output_buffer[:nb] = b.ravel(); output_buffer[nb:nb + s.size] = s.ravel()
                    outs = [output_buffer[:nb].reshape(b.shape), output_buffer[nb:nb + s.size].reshape(s.shape)]
            self.k += 1; self.jobs[self.k] = outs; return self.k
        def wait(self, k): return self.jobs.pop(k)
        def get_npu_inference_time(self): return 0
        def dispose(self): pass
    dx.InferenceEngine = InferenceEngine
    sys.modules['dx_engine'] = dx
    # native_post / runtime replacements (numpy decode + PC fast_nms + PC join_tiles), as claude_g4rt_deploy_01/dryrun_m1_mock.py
    import importlib.util
    def load(name, path):
        s = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
    nms = load('native_nms_pc', os.environ.get('LWIR_MOCK_NMS_PY') or sys.exit('--mock needs the environment variable LWIR_MOCK_NMS_PY = path of the PC file native_nms.py (no default in the public repository)'))
    def decode_raw_np(box, scores, out, h, w):
        gx, gy, st = [], [], []
        for s in (8, 16, 32):
            hh, ww = h // s, w // s; j = np.arange(hh * ww); gx.append(j % ww + .5); gy.append(j // ww + .5); st.append(np.full(hh * ww, s))
        gx, gy, st = (np.concatenate(v).astype(np.float32) for v in (gx, gy, st)); b = box.reshape(4, -1)
        l, t, r, bb = gx - b[0], gy - b[1], gx + b[2], gy + b[3]
        out[0, 0] = (l + r) * .5 * st; out[0, 1] = (t + bb) * .5 * st; out[0, 2] = (r - l) * st; out[0, 3] = (bb - t) * st
    sys.modules['native_post'] = types.SimpleNamespace(lib=types.SimpleNamespace(decode_raw=decode_raw_np), fast_nms=lambda y: nms.fast_nms(y))
    sys.modules['runtime'] = load('runtime', os.environ.get('LWIR_MOCK_RUNTIME_PY') or sys.exit('--mock needs the environment variable LWIR_MOCK_RUNTIME_PY = path of the PC file runtime.py (no default in the public repository)'))
