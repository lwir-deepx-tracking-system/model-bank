"""사람 조각을 외형 모델에 넣어 특징을 뽑는 함수들과, 외형 모델을 새로 올리거나 비우는 함수."""
from collections import deque
import os
import time
import numpy as np
from ..common.state import APP
from ..common.shared_values import HERE, PERSON_CLS, sha


def _crop_into(gray, box, output, width, height):
    x1, y1, x2, y2 = box
    left, top, right, bottom = int(max(0, x1)), int(max(0, y1)), int(min(640, x2)), int(min(512, y2))
    if right - left < 2 or bottom - top < 2:
        return False
    resized = APP.cv2.resize(gray[top:bottom, left:right], (width, height), interpolation=APP.cv2.INTER_LINEAR)
    output[0] = APP.cv2.cvtColor(resized, APP.cv2.COLOR_GRAY2RGB)
    return True


def _embed_slot_batch(engine, keys, gray, boxes, got):
    start = time.perf_counter_ns()
    jobs = deque()
    try:
        for slot, key in enumerate(keys):
            if not _crop_into(gray, boxes[key], engine.ins[slot], engine.W, engine.H):
                continue
            jobs.append((key, engine._submit_slot(slot), engine.outs[slot].size))
        count = len(jobs)
        while jobs:
            key, job, size = jobs.popleft()
            outputs = engine._wait_slot(job)
            e = np.asarray(outputs[0]).ravel()
            if e.size != size or not np.isfinite(e).all():
                raise AssertionError(('re-ID output', e.shape, size))
            # Division allocates a feature independent of the reusable NPU output slot.
            got[key] = e / max(np.linalg.norm(e), 1e-12)
    except BaseException:
        engine._drain_known(job for _, job, _ in jobs)
        raise
    if count:
        if not engine.birth_pipeline_batches:
            print('REID same-frame pipeline active: max 2 jobs, shared KeepID/BoT-SORT/refresh-overflow path', flush=True)
        engine.birth_pipeline_batches.append((count, (time.perf_counter_ns() - start) / 1e6))


def rl_embed(ks, g_, boxes):
    """Return features within this call for KeepID, BoT-SORT, and refresh overflow.

    Optional slot batches use at most two batch-1 jobs. Their wall time is recorded
    separately from serial per-crop and deferred-refresh timings.
    """
    if os.environ.get('LWIR_REID_BIRTH_PIPELINE') == '1':
        engine = vars(APP).get('rl_async')
        if engine is not None and not engine.quarantined and engine.inflight == 0 and engine.n > 1:
            ks = tuple(ks)
            if len(ks) > 1:
                if not hasattr(engine, 'birth_pipeline_batches'):
                    engine.birth_pipeline_batches = deque(maxlen=512)
                got = {}
                for start in range(0, len(ks), 2):
                    _embed_slot_batch(engine, ks[start:start + 2], g_, boxes, got)
                return got

    got = {}
    for k_ in ks:
        t0_ = time.perf_counter_ns()
        if not _crop_into(g_, boxes[k_], APP.rl_x, APP.rl_eng.W, APP.rl_eng.H): continue
        APP.rl_eng.submit(APP.rl_x, -1); e = APP.rl_eng.wait()[0].ravel(); got[k_] = e / max(np.linalg.norm(e), 1e-12)
        APP.rl_crop_ms.append((time.perf_counter_ns() - t0_) / 1e6)
        if not APP.args.mock: APP.rl_npu.append(APP.rl_eng.b.get_npu_inference_time() / 1000)
    return got


def reid_make_(key_):
    """New KeepID engine pair on NPU core 2 for models/<key>.dxnn, made while the old pair still works. Raises on any problem (the caller keeps the old pair).
                Input size and output length come from the file itself (Engine reads them); the crop rule is the one every re-ID path here uses:
                gray crop -> cv2.resize(W, H, linear) -> 3 channels uint8 NHWC -> output flattened -> L2 normalised."""
    import gc
    if not APP.args.mock and sha(HERE / 'models' / 'links' / (key_ + '.dxnn')) != APP.man.get(key_): raise RuntimeError('파일이 MANIFEST.json 기록과 다름')
    e_ = APP.Engine(key_, 2); a_ = None
    try:
        x_ = np.zeros((1, e_.H, e_.W, 3), np.uint8); e_.submit(x_, -1); v_ = e_.wait()[0]   # one test run: Engine.wait refuses a non-finite output
        if v_.size < 16: raise RuntimeError('출력 길이 %d: 외형 특징이 아님' % v_.size)
        if APP.rl_async is not None: a_ = APP.AsyncEmbedder(key_)
    except BaseException:
        APP.pending.pop(key_, None); e_ = None; a_ = None; gc.collect(); raise
    return e_, a_, x_


def bench_crop(g_, box):    # == crop_into: same clip to 640x512, INTER_LINEAR resize to the ViT input, GRAY2RGB
    x1, y1, x2, y2 = box; a_, b2, c_, d_ = int(max(0, x1)), int(max(0, y1)), int(min(640, x2)), int(min(512, y2))
    if c_ - a_ < 2 or d_ - b2 < 2: return False
    APP.bench_x[0] = APP.cv2.cvtColor(APP.cv2.resize(g_[b2:d_, a_:c_], (APP.bench_eng.W, APP.bench_eng.H), interpolation=APP.cv2.INTER_LINEAR), APP.cv2.COLOR_GRAY2RGB); return True


def vit_bench(g_, arr_, frame_no):
    """--vit-bench-n N: exactly N synchronous batch-1 ViT inferences on NPU_2 (submit + wait each, as sync_embed); embeddings discarded."""
    t0_ = time.perf_counter_ns()
    sel = arr_[(arr_[:, 5].astype(int) == PERSON_CLS) & (arr_[:, 4] >= 0.10) & ((arr_[:, 3] - arr_[:, 1]) >= 32.0)]
    boxes = [r[:4] for r in sel[np.argsort(-sel[:, 4], kind='stable')]]
    if not boxes: APP.bench_st['frames_no_person'] += 1
    for k_ in range(APP.args.vit_bench_n):
        ti = time.perf_counter_ns()
        if boxes and APP.bench_crop(g_, boxes[k_ % len(boxes)]): APP.bench_st['real'] += 1
        else: APP.bench_crop(g_, APP.BENCH_CENTRE); APP.bench_st['centre'] += 1
        APP.bench_eng.submit(APP.bench_x, frame_no); e = APP.bench_eng.wait()[0].ravel(); e = e / max(np.linalg.norm(e), 1e-12)
        APP.bench_inf.append((time.perf_counter_ns() - ti) / 1e6)
        if not APP.args.mock: APP.bench_npu.append(APP.bench_eng.b.get_npu_inference_time() / 1000)
    APP.bench_st['crops'] += APP.args.vit_bench_n
    return (time.perf_counter_ns() - t0_) / 1e6


def reid_drain_():
    if APP.relabeler is not None: APP.relabeler.reset()   # waits the old engine's crops in flight, drops KeepID's stored features (person numbers start again)
    if APP.tracker is not None:
        for tk_ in list(getattr(APP.tracker, 'tracked', [])) + list(getattr(APP.tracker, 'lost', [])): tk_.feat = None   # BoT-SORT track features of the old model
