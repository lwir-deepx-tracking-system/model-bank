"""프레임 한 장의 길: 읽기(read_frame) -> 검출기 입력으로 만들기(to_canvas, submit_tiles) -> 검출 상자 만들기(detect)."""
import time
import numpy as np
from ..common.state import APP
from ..common.shared_values import PERSON_CLS
from ..detect.model_scan import S6_DETS, W480_DETS, E2E_DETS


def avg_merge(ra, rb):   # claude_tt75_deploy_01 AvgMerge: element-wise average of the two towers' raw boxes / score logits
    for a_, b_, o in zip(ra, rb, APP.mrg_out): np.add(a_, b_, out=o); np.multiply(o, APP.half, out=o)
    return APP.mrg_out


def _detect(ra, rb):     # raw NPU outputs -> NMS detections in FRAME coordinates (512x640), columns x1,y1,x2,y2,conf,cls
    if APP.args.det in E2E_DETS:   # v30: end-to-end head; rows are x1,y1,x2,y2,score,class in model pixels, already one box per object (no host NMS)
        p_ = ra[0].reshape(-1, 6); p_ = p_[(p_[:, 4] > .001) & (p_[:, 2] > p_[:, 0]) & (p_[:, 3] > p_[:, 1])].astype(np.float32); p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
        return p_
    if APP.args.det == 's640':   # v16: outputs by name (boxes, scores), model y -> frame y
        p_ = APP.fast_nms(APP.dec_s(ra if APP.a.names[0] == 'boxes' else ra[::-1])).copy(); p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
        return p_
    if APP.args.det in W480_DETS:   # v16 (v22h: s480)
        p_ = APP.fast_nms(APP.dec_tt(ra)).copy(); p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
        return p_
    if APP.args.det in S6_DETS:   # v22 (v22g: s6): six heads merged by the Engine into boxes/scores over the 8,400 anchors of 640x640, model y -> frame y
        p_ = APP.fast_nms(APP.dec_s(ra)).copy(); p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
        return p_
    if APP.args.det == 'tt75':
        p_ = APP.fast_nms(APP.dec_tt(APP.avg_merge(ra, rb)))
        if APP.det_yoff: p_ = p_.copy(); p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
        return p_
    if APP.args.det == 'rt480':   # = m1_flir_eval_pipe.py post('NEWMRT480'), then band y -> frame y
        p_ = APP.fast_nms(APP.join_tiles(APP.dec480[0](ra), APP.dec480[1](rb), APP.jo480)).copy(); p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
        return p_
    return APP.fast_nms(APP.join_tiles(APP.dec[0](ra), APP.dec[1](rb), APP.jo))


def detect(ra, rb):      # v27 (user 11-10): person boxes under --min-person-h px tall are not people for this system; dropped before the tracker (0 = off = unchanged)
    p_ = APP._detect(ra, rb)
    if APP.args.min_person_h > 0 and len(p_): p_ = p_[~((p_[:, 5] == PERSON_CLS) & ((p_[:, 3] - p_[:, 1]) < APP.args.min_person_h))]
    return p_


def to_canvas(g_, name_):
    """Standard input 640x480 8-bit gray -> rows 16:496 of a 640x512 canvas (edge rows replicated); 640x512 as is (validated FLIR form); anything else stops with a message."""
    if g_ is None: raise SystemExit('cannot read frame %s' % name_)
    if g_.dtype != np.uint8 or g_.ndim != 2 or g_.shape[1] != 640 or g_.shape[0] not in (480, 512):
        raise SystemExit('frame %s: need an 8-bit gray image of 640x480 (camera) or 640x512, got %s %s' % (name_, g_.dtype, tuple(g_.shape)))
    if APP.FRAME_ROWS[0] is None: APP.FRAME_ROWS[0] = g_.shape[0]; APP.FRAME_DY[0] = 16.0 if g_.shape[0] == 480 else 0.0
    elif APP.FRAME_ROWS[0] != g_.shape[0]: raise SystemExit('frame %s: all frames of a run must have the same size (first frame had %d rows, this one has %d)' % (name_, APP.FRAME_ROWS[0], g_.shape[0]))
    if g_.shape[0] == 512: return g_
    c_ = np.empty((512, 640), np.uint8); c_[16:496] = g_; c_[:16] = g_[0]; c_[496:] = g_[-1]
    return c_


def read_frame(k):
    if APP.LIVE is not None: return APP.live_next(), 0.0
    if APP.mem is not None: return APP.mem[k], 0.0
    t = time.perf_counter_ns(); g_ = APP.to_canvas(APP.cv2.imread(str(APP.DATA / APP.images[k]['file_name']), APP.cv2.IMREAD_GRAYSCALE | APP.cv2.IMREAD_ANYDEPTH), APP.images[k]['file_name'])
    return g_, (time.perf_counter_ns() - t) / 1e6


def submit_tiles(g_, k):
    t = time.perf_counter_ns()
    if APP.args.det == 'rt480':  # = m1_flir_eval_pipe.py prep('NEWMRT480'): camera band rows 16:496, column tiles 0:384 / 256:640
        src_ = g_[16:496]
        if APP.args.live_port:   # v14: reuse a ring of preallocated input buffers (see TILE_BUFS)
            tb_ = APP.TILE_BUFS[k % len(APP.TILE_BUFS)]; APP.cv2.cvtColor(src_[:, 0:384], APP.cv2.COLOR_GRAY2RGB, dst=tb_[0][0]); APP.cv2.cvtColor(src_[:, 256:640], APP.cv2.COLOR_GRAY2RGB, dst=tb_[1][0]); APP.a.submit(tb_[0], k); APP.b.submit(tb_[1], k)
        else: APP.a.submit(APP.cv2.cvtColor(src_[:, 0:384], APP.cv2.COLOR_GRAY2RGB)[None], k); APP.b.submit(APP.cv2.cvtColor(src_[:, 256:640], APP.cv2.COLOR_GRAY2RGB)[None], k)
    elif (APP.args.det == 's640' or APP.args.det in S6_DETS):   # v16: YOLO26s letterbox; v22: same canvas for YOLO26m_(FLIR) 640x640
        if APP.args.live_port:
            sb_ = APP.S640_BUFS[k % len(APP.S640_BUFS)]; APP.cv2.cvtColor(g_[16:496], APP.cv2.COLOR_GRAY2RGB, dst=sb_[0, 80:560]); APP.a.submit(sb_, k)
        elif APP.FRAME_DY[0] == 16.0: APP.a.submit(APP.cv2.cvtColor(APP.cv2.copyMakeBorder(g_[16:496], 80, 80, 0, 0, APP.cv2.BORDER_CONSTANT, value=114), APP.cv2.COLOR_GRAY2RGB)[None], k)
        else: APP.a.submit(APP.cv2.cvtColor(APP.cv2.copyMakeBorder(g_, 64, 64, 0, 0, APP.cv2.BORDER_CONSTANT, value=114), APP.cv2.COLOR_GRAY2RGB)[None], k)
    elif APP.args.det in W480_DETS:   # v16: one model on the whole camera band (rows 16:496); v22h: s480
        if APP.args.live_port:
            wb_ = APP.WHOLE_BUFS[k % len(APP.WHOLE_BUFS)]; APP.cv2.cvtColor(g_[16:496], APP.cv2.COLOR_GRAY2RGB, dst=wb_[0]); APP.a.submit(wb_, k)
        else: APP.a.submit(APP.cv2.cvtColor(g_[16:496], APP.cv2.COLOR_GRAY2RGB)[None], k)
    elif APP.args.det == 'tt75':   # the SAME camera band (rows 16:496) to both towers
        x = APP.cv2.cvtColor(g_[16:496], APP.cv2.COLOR_GRAY2RGB)[None]; APP.a.submit(x, k); APP.b.submit(x.copy(), k)
    else:
        APP.a.submit(APP.cv2.cvtColor(g_[:, 0:384], APP.cv2.COLOR_GRAY2RGB)[None], k); APP.b.submit(APP.cv2.cvtColor(g_[:, 256:640], APP.cv2.COLOR_GRAY2RGB)[None], k)
    return t
