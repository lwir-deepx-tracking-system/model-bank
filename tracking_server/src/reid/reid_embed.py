"""사람 조각을 외형 모델에 넣어 특징을 뽑는 함수들과, 외형 모델을 새로 올리거나 비우는 함수."""
import time
import numpy as np
from ..common.state import APP
from ..common.shared_values import HERE, PERSON_CLS, sha


def rl_embed(ks, g_, boxes):
    """--relabel: crop -> submit -> wait, batch 1 on NPU_2, one after another (crop rule = crop_from_pred.py on the saved-PRED box form)"""
    got = {}
    for k_ in ks:
        t0_ = time.perf_counter_ns(); x1, y1, x2, y2 = boxes[k_]
        a_, b_, c_, d_ = int(max(0, x1)), int(max(0, y1)), int(min(640, x2)), int(min(512, y2))
        if c_ - a_ < 2 or d_ - b_ < 2: continue
        APP.rl_x[0] = APP.cv2.cvtColor(APP.cv2.resize(g_[b_:d_, a_:c_], (APP.rl_eng.W, APP.rl_eng.H), interpolation=APP.cv2.INTER_LINEAR), APP.cv2.COLOR_GRAY2RGB)
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
