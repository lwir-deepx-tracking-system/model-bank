"""NPU 엔진 다루기: 모델 올리기, 그림 넣기·결과 받기(Engine, AsyncEmbedder), 검출기 출력 풀기(Decoder), 두 타일 결과 합치기(join_tiles)."""
from collections import deque
import time
import numpy as np
from ..common.state import APP
from ..common.shared_values import HERE, anchors
from ..detect.model_scan import BANK, E2E_KEYS, RAW_KEYS, KPT_KEYS, raw_shape


def join_tiles(left, right, out=None):   # = tile_compact_17/runtime.py join_tiles (copied so that the module's torch import is not needed)
    n = left.shape[-1]; assert right.shape == left.shape
    if out is None: out = np.empty((1, 20, n * 2), np.float32)
    assert out.shape == (1, 20, n * 2); out[:, :, :n] = left; out[:, :, n:] = right; out[:, 0, n:] += 256
    a, b = out[:, :, :n], out[:, :, n:]; a[0, 4:, a[0, 0] >= 320] = 0; b[0, 4:, b[0, 0] < 320] = 0
    return out


class Decoder:
    def __init__(self, h, w):
        self.h, self.w, self.n = h, w, anchors(h, w); self.out = np.empty((1, 20, self.n), np.float32); self.so = self.out[:, 4:].reshape(16, self.n)
    def __call__(self, raw):
        APP.native_post.lib.decode_raw(raw[0], raw[1], self.out, self.h, self.w)
        np.negative(raw[1].reshape(16, self.n), out=self.so); APP.cv2.exp(self.so, self.so); np.add(self.so, np.float32(1), out=self.so); np.reciprocal(self.so, out=self.so)
        return self.out


class Engine:
    def __init__(self, name, core, depth=1, model=None):
        self.name = name; self.depth = depth; self.q = []; self.nsub = 0; self.split6 = False; self.kpt = False; model = model or name   # v9: model = the DXNN when it differs from the pending key ('ZOOM')
        o = APP.dx.InferenceOption(); o.devices = [0]; o.bound_option = getattr(APP.dx.InferenceOption.BOUND_OPTION, 'NPU_%d' % core); o.use_ort = True; o.buffer_count = depth
        self.b = APP.dx.InferenceEngine(str(BANK[model] if model in BANK else HERE / 'models' / 'links' / (model + '.dxnn')) if not APP.args.mock else model, o)
        i = self.b.get_input_tensors_info()[0]; assert np.dtype(i['dtype']) == np.dtype('uint8'), i
        self.in_shape = tuple(int(v) for v in i['shape']); outs = self.b.get_output_tensors_info(); self.names = [v['name'] for v in outs]
        if model in E2E_KEYS:   # v30: one finished [1,300,6] output, used as it is
            H, W = APP.det_shape[model]; assert self.in_shape == (1, H, W, 3), i
            assert len(outs) == 1 and tuple(int(d) for d in outs[0]['shape']) == (1, 300, 6), outs
            self.slots = [np.empty((1, 300, 6), np.float32)]
        elif model in RAW_KEYS:   # 11-13: one decoded output over all anchors, [1,20,N] (YOLOv8 form) or [1,N,21] (YOLOX form); the host only does NMS (detect_frame.raw_decoded)
            H, W = APP.det_shape[model]; shp_ = raw_shape(RAW_KEYS[model], anchors(H, W)); assert self.in_shape == (1, H, W, 3), i
            assert len(outs) == 1 and tuple(int(d) for d in outs[0]['shape']) == shp_, outs
            self.slots = [np.empty(shp_, np.float32)]
        elif model in APP.det_shape:
            H, W = APP.det_shape[model]; self.n = anchors(H, W); assert self.in_shape == (1, H, W, 3), i
            if APP.args.split6:
                self.split6 = True; self.kpt = model in KPT_KEYS   # 11-14: + kpt_s8/16/32
                assert self.names == ['box_s8', 'box_s16', 'box_s32', 'cls_s8', 'cls_s16', 'cls_s32'] + (['kpt_s8', 'kpt_s16', 'kpt_s32'] if self.kpt else []), self.names
                self.slots = [np.empty(tuple(int(d) for d in v['shape']), np.float32) for v in outs]
                assert [s_.shape for s_ in self.slots] == [(1, c_, H // st_, W // st_) for c_ in ((4, 16, 51) if self.kpt else (4, 16)) for st_ in (8, 16, 32)], [s_.shape for s_ in self.slots]
            else:
                self.slots = [np.empty((1, 4 if v['name'] == 'boxes' else 16, self.n), np.float32) for v in outs]
        else:
            assert self.in_shape[0] == 1 and self.in_shape[3] == 3, i; self.H, self.W = self.in_shape[1], self.in_shape[2]
            sizes = self.b.get_output_tensor_sizes(); assert len(sizes) == 1, sizes
            self.slots = [np.empty((1, sizes[0] // 4), np.float32)]
        assert self.b.get_output_tensor_sizes() == [x.nbytes for x in self.slots]
        self.packed = np.empty(sum(v.size for v in self.slots), np.float32); k = 0
        for j, v in enumerate(self.slots): self.slots[j] = self.packed[k:k + v.size].reshape(v.shape); k += v.size
        self.packs = [self.packed] + [np.empty_like(self.packed) for _ in range(depth - 1)]   # --npu-depth: one output buffer per job in flight
    def submit(self, x, f):
        if self.depth == 1: APP.pending[self.name] = self.b.run_async(x, user_arg={'f': f}, output_buffer=self.packed); return
        s_ = self.nsub % self.depth; self.nsub += 1
        self.q.append((self.b.run_async(x, user_arg={'f': f}, output_buffer=self.packs[s_]), x))   # the input stays referenced until wait
    def wait(self):
        vals = self.b.wait(APP.pending.pop(self.name) if self.depth == 1 else self.q.pop(0)[0]); d = {}
        for n, v, s in zip(self.names, vals, self.slots):
            v = np.asarray(v); assert v.size == s.size and np.isfinite(v).all(), (n, v.shape, s.shape)
            if v.shape != s.shape: v = v.reshape(s.shape)
            # pipeline mode: the next frame is submitted into the same output buffer before this frame is post-processed -> copy out
            d[n] = np.array(v, copy=True) if (APP.args.pipeline or not (v.ctypes.data == s.ctypes.data and v.strides == s.strides)) else v
        o_ = [d[n] for n in self.names]
        if self.split6:
            r_ = [np.concatenate([x_.reshape(1, 4, -1) for x_ in o_[:3]], axis=2), np.concatenate([x_.reshape(1, 16, -1) for x_ in o_[3:6]], axis=2)]
            if self.kpt: r_.append(np.concatenate([x_.reshape(1, 51, -1) for x_ in o_[6:9]], axis=2))   # 11-14: raw keypoints [1,51,N], same anchor order as boxes
            return r_
        return o_


class AsyncEmbedder:
    """--rl-async: up to N batch-1 KeepID refresh crops in flight on NPU_2 (own engine, buffer_count N, one input/output buffer per slot);
                    collect() at the next frame waits them; crops beyond N are embedded synchronously (rl_embed)"""
    def __init__(self, name, n=16):
        o = APP.dx.InferenceOption(); o.devices = [0]; o.bound_option = APP.dx.InferenceOption.BOUND_OPTION.NPU_2; o.use_ort = True; o.buffer_count = n
        self.b = APP.dx.InferenceEngine(str(HERE / 'models' / 'links' / (name + '.dxnn')) if not APP.args.mock else name, o); self.n = n
        i_ = self.b.get_input_tensors_info()[0]; self.H, self.W = int(i_['shape'][1]), int(i_['shape'][2])
        self.outs = [np.empty(self.b.get_output_tensor_sizes()[0] // 4, np.float32) for _ in range(n)]
        self.ins = [np.zeros((1, self.H, self.W, 3), np.uint8) for _ in range(n)]; self.sub_ms = []; self.col_ms = []; self.inflight = 0
        self.quarantined = False

    def _require_usable(self):
        if self.quarantined:
            raise RuntimeError('ReID async engine quarantined after an SDK failure')

    def _submit_slot(self, slot):
        self._require_usable()
        try:
            job = self.b.run_async(self.ins[slot], user_arg=None, output_buffer=self.outs[slot])
        except BaseException:
            self.quarantined = True
            raise
        self.inflight += 1
        return job

    def _wait_slot(self, job):
        self._require_usable()
        try:
            outputs = self.b.wait(job)
        except BaseException:
            self.quarantined = True
            raise
        self.inflight -= 1
        return outputs

    def _drain_known(self, jobs):
        # Cleanup may finish other known jobs after quarantine. The failed job is never retried.
        for job in jobs:
            try:
                self.b.wait(job)
            except BaseException:
                self.quarantined = True
            else:
                self.inflight -= 1

    def submit(self, ks, g_, boxes):
        self._require_usable()
        if self.inflight:
            raise RuntimeError('ReID async slots still have pending work')
        t_ = time.perf_counter_ns(); jobs = deque(); extra = APP.rl_embed(list(ks[self.n:]), g_, boxes) if len(ks) > self.n else {}
        try:
            for s_, k_ in enumerate(ks[:self.n]):
                x1, y1, x2, y2 = boxes[k_]; a_, b_, c_, d_ = int(max(0, x1)), int(max(0, y1)), int(min(640, x2)), int(min(512, y2))
                if c_ - a_ < 2 or d_ - b_ < 2: continue
                self.ins[s_][0] = APP.cv2.cvtColor(APP.cv2.resize(g_[b_:d_, a_:c_], (self.W, self.H), interpolation=APP.cv2.INTER_LINEAR), APP.cv2.COLOR_GRAY2RGB)
                jobs.append((k_, self._submit_slot(s_)))
        except BaseException:
            self._drain_known(job for _, job in jobs)
            raise
        self.sub_ms.append((time.perf_counter_ns() - t_) / 1e6)

        def collect():
            self._require_usable()
            t2 = time.perf_counter_ns(); got = dict(extra)
            try:
                while jobs:
                    k_, job = jobs.popleft()
                    e = np.asarray(self._wait_slot(job)[0], np.float32).ravel(); got[k_] = e / max(np.linalg.norm(e), 1e-12)
            except BaseException:
                self._drain_known(job for _, job in jobs)
                jobs.clear()
                raise
            self.col_ms.append((time.perf_counter_ns() - t2) / 1e6); return got
        return collect
