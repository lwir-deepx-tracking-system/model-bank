"""E10 pose detector split into two overlapping, simultaneous NPU tiles.

This module is an opt-in scheduling path. It retains all 17 pose points and
uses the ordinary tracker and KeepID downstream. It never changes the default
detector or substitutes a different model when the verified asset is absent.
"""
import ctypes as c
import os
from pathlib import Path
import time

import numpy as np

from ..common.shared_values import HERE, sha
from ..common.state import APP
from .split_native import NativeTiles
from .tile_job_profile import TileJobProfile


KEY = 'yolo26m_pose_ft16v8_aug_e10_bodybox_provisional_kpt_640x480'
SOURCE_SHA = '220331cc79e4da0d3ba8909965d84474386fa93f4c03c5e7471cf260d8541df1'
MODEL_SHA = '47fd961044bf18fdc3e5b8a90d74e499699a05ae2a114629cd9ee9751820e4a9'
MODEL = HERE / 'models' / 'optimized' / 'E10_KPT_480x384.dxnn'
TONE = HERE / 'src' / 'detect' / 'tone_map.so'
NATIVE_SPLIT = HERE / 'src' / 'detect' / 'liblwir_split_sdk.so'
NATIVE_CHECK_FRAMES = 4
H, W, TW, RIGHT = 480, 640, 384, 256
STRIDES = (8, 16, 32)
N = sum((H // s) * (TW // s) for s in STRIDES)
BOXES = tuple(f'box_s{s}' for s in STRIDES)
CLASSES = tuple(f'/cv3.{i}/cv3.{i}.2/Conv_output_0' for i in range(3))
POSES = tuple(f'kpt_s{s}_nhwc' for s in STRIDES)


def assets_reason():
    if not MODEL.is_file() or not TONE.is_file():
        return '분할 검출 모델 또는 빠른 전처리가 없습니다. 장치 준비 스크립트를 실행하세요'
    try:
        if sha(MODEL) != MODEL_SHA:
            return '분할 검출 모델의 확인값이 다릅니다'
    except OSError:
        return '분할 검출 모델을 읽을 수 없습니다'
    return None


def selected():
    return getattr(APP.args, 'det', None) == 'bank:' + KEY and APP.man.get(KEY) == SOURCE_SHA


class Tone:
    """Exact 1st/99th percentile mapping; temporal lo/hi remain on APP.LIVE."""
    def __init__(self):
        self.lib = c.CDLL(str(TONE))
        i = np.ctypeslib.ndpointer(dtype=np.uint16, ndim=2, flags='C_CONTIGUOUS')
        o = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS')
        self.lib.percentile16.argtypes = [i, c.c_size_t, c.POINTER(c.c_uint32), c.POINTER(c.c_uint32)]
        self.lib.percentile16.restype = None
        self.lib.apply_tone.argtypes = [i, o, c.c_size_t, c.c_float, c.c_float]
        self.lib.apply_tone.restype = None

    def __call__(self, raw):
        if APP.args.live_map == 'fixed':
            return APP.lv_to8(raw)
        if raw.dtype != np.uint16 or raw.shape != (H, W) or not raw.flags.c_contiguous:
            raise ValueError('dual tile requires the 16-bit 640x480 camera buffer')
        a, b = c.c_uint32(), c.c_uint32()
        self.lib.percentile16(raw, raw.size, c.byref(a), c.byref(b))
        lo, hi = float(a.value) * 16, float(b.value) * 16 + 16
        if hi - lo < 200:
            hi = lo + 200
        if APP.LIVE.lo is None:
            APP.LIVE.lo, APP.LIVE.hi = lo, hi
        else:
            APP.LIVE.lo += .1 * (lo - APP.LIVE.lo)
            APP.LIVE.hi += .1 * (hi - APP.LIVE.hi)
        out = np.empty((H, W), np.uint8)
        self.lib.apply_tone(raw, out, raw.size, APP.LIVE.lo, 255.0 / (APP.LIVE.hi - APP.LIVE.lo))
        return out


class PairBackend:
    def __init__(self, model):
        self.engines, self.buffers, self.names = [], [], None
        self.model = model
        self.native = None
        self.native_checks = 0
        self.native_error = None
        self.native_allowed = (vars(APP).get('LIVE') is not None and NATIVE_SPLIT.is_file()
                               and os.environ.get('LWIR_CPP_SPLIT', '1') == '1')
        self.profile = TileJobProfile(APP.out, APP.dx) if vars(APP).get('LIVE') is not None else None
        self.profile_pending = None
        self.inputs = [np.empty((1, H, TW, 3), np.uint8) for _ in range(2)]
        self.shapes = {}
        for stride, box, cls, pose in zip(STRIDES, BOXES, CLASSES, POSES):
            h, w = H // stride, TW // stride
            self.shapes.update({box: (1, 4, h, w), cls: (1, 1, h, w), pose: (1, h, w, 51)})
        cfg = APP.dx.Configuration()
        if cfg.get_enable(cfg.ITEM.NFH_ACCELERATION):
            raise RuntimeError('dual tile requires NFH acceleration disabled')
        self.open_python()

    @property
    def using_native(self):
        return self.native is not None and self.native_checks >= NATIVE_CHECK_FRAMES

    def open_python(self):
        try:
            for core in (0, 1):
                opt = APP.dx.InferenceOption()
                opt.devices = [0]
                opt.bound_option = getattr(opt.BOUND_OPTION, f'NPU_{core}')
                opt.use_ort = False
                opt.buffer_count = 1
                engine = APP.dx.InferenceEngine(str(self.model), opt)
                self.engines.append(engine)
                inputs = engine.get_input_tensors_info()
                if len(inputs) != 1 or tuple(inputs[0]['shape']) != (1, H, TW, 3) or np.dtype(inputs[0]['dtype']) != np.uint8:
                    raise RuntimeError(f'unexpected tile input: {inputs}')
                names = list(engine.get_output_tensor_names())
                sizes = list(engine.get_output_tensor_sizes())
                if len(names) != 9 or set(names) != set(self.shapes) or sizes != [int(np.prod(self.shapes[n])) * 4 for n in names]:
                    raise RuntimeError(f'unexpected tile outputs: {names}/{sizes}')
                if self.names is not None and self.names != names:
                    raise RuntimeError('tile engines disagree on output order')
                self.names = names
                self.buffers.append(np.empty(sum(sizes)//4, np.float32))
        except BaseException:
            self.close_python()
            raise

    def close_python(self):
        for engine in self.engines:
            try:
                engine.dispose()
            except Exception:
                pass
        self.engines.clear()
        self.buffers.clear()

    def disable_native(self, error):
        if self.native is not None:
            self.native.close()
            self.native = None
        self.native_allowed = False
        self.native_error = str(error)[:180]
        print(f"NATIVE_SPLIT fallback to Python SDK: {self.native_error}", flush=True)
        if not self.engines:
            self.open_python()

    def check_native(self, original):
        # Compare the two SDK paths sequentially on the same live frame.
        # The existing result is published during this bounded verification.
        try:
            if self.native is None:
                self.native = NativeTiles(NATIVE_SPLIT, self.model, (1, H, TW, 3), self.shapes)
            jobs = self.native.submit(self.inputs)
            candidate = self.native.wait(jobs, self.names)
            for core, (before, after) in enumerate(zip(original, candidate, strict=True)):
                for name, a, b in zip(self.names, before, after, strict=True):
                    if not np.array_equal(np.asarray(a).view(np.uint32), b.view(np.uint32)):
                        raise ValueError(f"raw tensor differs at core {core}, {name}")
            self.native_checks += 1
        except Exception as error:
            self.disable_native(error)
            return
        if self.using_native:
            self.close_python()
            print(f"NATIVE_SPLIT active: {self.native_checks} live frames matched all 18 tensors, "
                  f"{self.model.name}", flush=True)

    def run_async(self, gray, user_arg=None, output_buffer=None):
        if gray.dtype != np.uint8 or gray.shape != (H, W):
            raise ValueError(f'unexpected dual-tile input: {gray.shape}/{gray.dtype}')
        trace = self.profile.begin((user_arg or {}).get('f')) if self.profile is not None else None
        APP.cv2.cvtColor(gray[:, :TW], APP.cv2.COLOR_GRAY2RGB, dst=self.inputs[0][0])
        APP.cv2.cvtColor(gray[:, RIGHT:], APP.cv2.COLOR_GRAY2RGB, dst=self.inputs[1][0])
        jobs = None
        if self.using_native:
            try:
                jobs = self.native.submit(self.inputs)
            except Exception as error:
                self.disable_native(error)
            else:
                if trace is not None:
                    timing = self.native.timing
                    trace["backend"] = "cpp"
                    trace["tiles"] = [{"core": core, "job": job,
                                       "submit_start_ns": timing.submit_start[core],
                                       "submit_done_ns": timing.submit_done[core]}
                                      for core, job in enumerate(jobs)]
        if jobs is None:
            jobs = self.submit_python(user_arg, trace)
        if trace is not None:
            self.profile_pending = (jobs, trace)
        return jobs

    def submit_python(self, user_arg, trace):
        jobs = []
        try:
            for core, (eng, image, out) in enumerate(zip(self.engines, self.inputs, self.buffers)):
                submit_start = time.perf_counter_ns() if trace is not None else None
                job = eng.run_async(image, user_arg=user_arg, output_buffer=out)
                jobs.append(job)
                if trace is not None:
                    trace["tiles"].append({"core": core, "job": int(job),
                                           "submit_start_ns": submit_start,
                                           "submit_done_ns": time.perf_counter_ns()})
        except BaseException:
            for eng, job in zip(self.engines, jobs):
                try:
                    eng.wait(job)
                except BaseException:
                    pass
            raise
        return jobs

    def wait(self, jobs):
        pending = self.profile_pending
        trace = pending[1] if pending is not None and pending[0] is jobs else None
        if trace is not None:
            self.profile_pending = None
        native = self.using_native
        if native:
            try:
                waited = self.native.wait(jobs, self.names)
                results = self.pack(waited)
            except Exception as error:
                self.disable_native(error)
                # Same input, newly recreated Python engines. The failed
                # native jobs were drained before their buffers are reused.
                jobs = self.submit_python(None, None)
                trace = None
                waited = self.wait_python(jobs, trace)
                results = self.pack(waited)
            else:
                if trace is not None:
                    timing = self.native.timing
                    for core, tile in enumerate(trace["tiles"]):
                        tile.update(wait_start_ns=timing.wait_start[core],
                                    wait_done_ns=timing.wait_done[core],
                                    wait_cpu_ns=timing.wait_cpu[core])
        else:
            waited = self.wait_python(jobs, trace)
            results = self.pack(waited)
        if trace is not None:
            trace["packed_ns"] = time.perf_counter_ns()
            self.profile.record_result(trace, jobs, self.metrics())
        if not native and self.native_allowed:
            self.check_native(waited)
        return results

    def wait_python(self, jobs, trace):
        # Drain both submissions before their buffers may be reused.
        waited, failure = [], None
        for eng, job in zip(self.engines, jobs):
            tile = trace["tiles"][len(waited)] if trace is not None else None
            if tile is not None:
                tile["wait_start_ns"] = time.perf_counter_ns()
            try:
                waited.append(eng.wait(job))
            except BaseException as exc:
                waited.append(None)
                if failure is None:
                    failure = exc
            finally:
                if tile is not None:
                    tile["wait_done_ns"] = time.perf_counter_ns()
        if failure is not None:
            raise failure
        return waited

    def pack(self, waited):
        results = []
        for raw in waited:
            if len(raw) != len(self.names):
                raise RuntimeError('tile output count changed')
            heads = {}
            for name, value in zip(self.names, raw):
                array = np.asarray(value)
                if array.shape != self.shapes[name] or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise RuntimeError(f'invalid tile output {name}: {array.shape}/{array.dtype}')
                heads[name] = array
            boxes = np.concatenate([heads[n].reshape(1, 4, -1) for n in BOXES], axis=2)
            # This converted model has one person class. Do not allocate the
            # 15 absent class planes merely to sigmoid and discard them.
            scores = np.concatenate([heads[n].reshape(-1) for n in CLASSES])
            poses = np.concatenate([heads[n].reshape(-1, 51) for n in POSES], axis=0)
            results.extend((boxes, scores, poses))
        return results

    def metrics(self):
        return self.native.engines if self.using_native else self.engines

    def get_npu_inference_time(self):
        return max(e.get_npu_inference_time() for e in self.metrics())

    def get_latency(self):
        return max(e.get_latency() for e in self.metrics())

    def dispose(self):
        if self.profile is not None:
            self.profile.close()
        if self.native is not None:
            self.native.close()
            self.native = None
        self.close_python()


class Engine:
    def __init__(self):
        reason = assets_reason()
        if reason:
            raise RuntimeError(reason)
        self.name, self.core, self.cores, self.depth = 'E10#two-tiles', 0, (0, 1), 1
        self.in_shape, self.split6, self.kpt, self.n = (1, H, W, 3), True, True, N * 2
        self.b = PairBackend(MODEL)
        self.packed = np.empty(0, np.float32)
        self.names = list(self.b.names)

    def submit(self, gray, frame):
        APP.pending[self.name] = self.b.run_async(gray, user_arg={'f': frame}, output_buffer=self.packed)

    def wait(self):
        return self.b.wait(APP.pending.pop(self.name))


def is_result(raw):
    return isinstance(raw, (list, tuple)) and len(raw) == 6 and raw[0].shape == (1, 4, N)


_decode_cache = None


def detect(raw, detector):
    """Return NMS boxes and set the standard KP cache, preserving tile ownership."""
    global _decode_cache
    if _decode_cache is None:
        # The standard NMS consumes 16 classes. All but person are absent
        # from this model, so their immutable output planes are zero.
        _decode_cache = (np.zeros((1, 20, N), np.float32),
                         np.zeros((1, 20, N), np.float32),
                         np.zeros((1, 20, N * 2), np.float32))
    left, right, decoded = _decode_cache
    for out, boxes_, scores_ in ((left, raw[0], raw[1]), (right, raw[3], raw[4])):
        APP.native_post.lib.decode_raw(boxes_, scores_, out, H, TW)
        person_score = out[0, 4:5, :]
        np.negative(scores_.reshape(1, N), out=person_score)
        APP.cv2.exp(person_score, person_score)
        np.add(person_score, np.float32(1), out=person_score)
        np.reciprocal(person_score, out=person_score)
    right[:, 0, :] += RIGHT
    left[0, 4:, left[0, 0] >= W // 2] = 0
    right[0, 4:, right[0, 0] < W // 2] = 0
    decoded[:, :, :N] = left
    decoded[:, :, N:] = right
    person_nms = getattr(APP.native_post, 'fast_nms_person', APP.fast_nms)
    rows = person_nms(decoded).copy()
    from . import detect_frame
    person = rows[rows[:, 5] == 0]
    anchors = detect_frame._kp_anchors(decoded[0], person)
    local, side = anchors % N, anchors // N
    if np.any(side > 1):
        raise RuntimeError('invalid dual-tile pose anchor')
    gx, gy, gs = detect_frame._kp_grid(H, TW)
    pose_raw = np.empty((len(anchors), 17, 3), np.float32)
    for which in (0, 1):
        mask = side == which
        if np.any(mask):
            pose_raw[mask] = raw[2 if which == 0 else 5][local[mask]].reshape(-1, 17, 3)
    points = np.empty_like(pose_raw)
    points[:, :, 0] = (pose_raw[:, :, 0] + gx[local, None]) * gs[local, None] + (side * RIGHT)[:, None]
    points[:, :, 1] = (pose_raw[:, :, 1] + gy[local, None]) * gs[local, None] + APP.det_yoff
    points[:, :, 2] = 1 / (1 + np.exp(-pose_raw[:, :, 2]))
    boxes = person[:, :4].copy()
    boxes[:, [1, 3]] += APP.det_yoff
    detect_frame.KP['prev'] = detect_frame.KP['cur']
    detect_frame.KP['cur'] = {'boxes': boxes, 'kpts': points}
    rows[:, [1, 3]] += APP.det_yoff
    return rows
