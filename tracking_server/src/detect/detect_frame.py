"""프레임 한 장의 길: 읽기(read_frame) -> 검출기 입력으로 만들기(to_canvas, submit_tiles) -> 검출 상자 만들기(detect)."""
import time
import numpy as np
from ..common.state import APP
from ..common.shared_values import PERSON_CLS
from ..detect.model_scan import S6_DETS, W480_DETS, E2E_DETS, RAW_DETS, KPT_DETS


def avg_merge(ra, rb):   # claude_tt75_deploy_01 AvgMerge: element-wise average of the two towers' raw boxes / score logits
    for a_, b_, o in zip(ra, rb, APP.mrg_out): np.add(a_, b_, out=o); np.multiply(o, APP.half, out=o)
    return APP.mrg_out


RAW_BUF = {}   # anchors N -> [1,20,N] work buffer of raw_decoded (used by the frame-loop thread only, like the Decoder buffers)
def raw_decoded(kind, out):
    """11-13 (user): the one output of a 'v8' / 'yolox' model (model_scan.RAW_DETS) -> the float32 C-contiguous [1,20,N] array that fast_nms reads
    (rows 0..3 = cx,cy,w,h in model pixels, rows 4..19 = 16 class scores in 0..1) = the same form Decoder gives for a six-head model, so the score threshold and NMS are shared.
    'v8' [1,20,N]: already that form (the file's own CPU part did the box decode and the sigmoid). 'yolox' [1,N,21]: per anchor cx,cy,w,h, objectness, 16 class scores
    (pixels and 0..1 already); score = objectness x class score (the YOLOX rule)."""
    if kind == 'v8': return np.ascontiguousarray(out, np.float32).reshape(1, 20, -1)
    o = np.asarray(out, np.float32).reshape(-1, 21); n = o.shape[0]; y = RAW_BUF.get(n)
    if y is None: y = RAW_BUF[n] = np.empty((1, 20, n), np.float32)
    y[0, :4] = o[:, :4].T; np.multiply(o[:, 5:].T, o[:, 4], out=y[0, 4:])
    return y


KP = {'cur': None, 'prev': None}   # 11-14: keypoints of the person boxes of the last two frames: dict(boxes [n,4] x1,y1,x2,y2 frame coords, kpts [n,17,3] x,y frame coords + visibility 0..1)
_KP_GRID = {}
def _kp_grid(h, w):
    g = _KP_GRID.get((h, w))
    if g is None:
        cx, cy, st = [], [], []
        for s in (8, 16, 32):
            gh, gw = h // s, w // s; j = np.arange(gh * gw); cx.append(j % gw + .5); cy.append(j // gw + .5); st.append(np.full(gh * gw, s, np.float32))
        g = _KP_GRID[(h, w)] = (np.concatenate(cx).astype(np.float32), np.concatenate(cy).astype(np.float32), np.concatenate(st))
    return g


def attach_kpts(dec, kraw, p_, h, w, yoff):
    """11-14: keypoints of the person boxes NMS kept. dec = decoded candidates [1,20,N] (model px), kraw = raw keypoints [1,51,N] in the same anchor order,
    p_ = NMS rows (model px, before the y shift). A row's anchor is the one whose person score and box equal the row (NMS copies them).
    Decode = ultralytics Pose26.kpts_decode: x = (raw_x + col + .5) * stride, y = (raw_y + row + .5) * stride, visible = sigmoid(raw_v). Kept in KP (frame coords)."""
    KP['prev'] = KP['cur']; KP['cur'] = None
    try:
        m = p_[:, 5] == 0
        if not m.any(): KP['cur'] = dict(boxes=np.zeros((0, 4), np.float32), kpts=np.zeros((0, 17, 3), np.float32)); return
        d = dec[0]; x1 = d[0] - d[2] * .5; y1 = d[1] - d[3] * .5; gx, gy, gs = _kp_grid(h, w); rows = p_[m]; idx = []
        for r in rows:
            c = np.flatnonzero(np.abs(d[4] - r[4]) < 1e-6)
            if len(c) == 0: c = np.arange(d.shape[1])
            idx.append(int(c[np.argmin(np.abs(x1[c] - r[0]) + np.abs(y1[c] - r[1]))]))
        idx = np.array(idx); k = kraw[0][:, idx].reshape(17, 3, -1).transpose(2, 0, 1).astype(np.float32)
        out = np.empty_like(k); out[:, :, 0] = (k[:, :, 0] + gx[idx, None]) * gs[idx, None]; out[:, :, 1] = (k[:, :, 1] + gy[idx, None]) * gs[idx, None] + yoff
        out[:, :, 2] = 1. / (1. + np.exp(-k[:, :, 2]))
        b = rows[:, :4].astype(np.float32).copy(); b[:, 1] += yoff; b[:, 3] += yoff
        KP['cur'] = dict(boxes=b, kpts=out)
    except Exception as e:
        print('KPT attach failed', repr(e)[:120], flush=True)


def _detect(ra, rb):     # raw NPU outputs -> NMS detections in FRAME coordinates (512x640), columns x1,y1,x2,y2,conf,cls
    if APP.args.det in E2E_DETS:   # v30: end-to-end head; rows are x1,y1,x2,y2,score,class in model pixels, already one box per object (no host NMS)
        p_ = ra[0].reshape(-1, 6); p_ = p_[(p_[:, 4] > .001) & (p_[:, 2] > p_[:, 0]) & (p_[:, 3] > p_[:, 1])].astype(np.float32); p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
        return p_
    if APP.args.det in RAW_DETS:   # 11-13: one decoded output (YOLOv8 / YOLOX form) -> the same host NMS as the six-head models, model y -> frame y
        p_ = APP.fast_nms(raw_decoded(RAW_DETS[APP.args.det], ra[0])).copy(); p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
        return p_
    if APP.args.det == 's640':   # v16: outputs by name (boxes, scores), model y -> frame y
        p_ = APP.fast_nms(APP.dec_s(ra if APP.a.names[0] == 'boxes' else ra[::-1])).copy(); p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
        return p_
    if APP.args.det in W480_DETS:   # v16 (v22h: s480)
        d_ = APP.dec_tt(ra); p_ = APP.fast_nms(d_).copy()
        if APP.args.det in KPT_DETS and len(ra) > 2: attach_kpts(d_, ra[2], p_, 480, 640, APP.det_yoff)   # 11-14
        p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
        return p_
    if APP.args.det in S6_DETS:   # v22 (v22g: s6): six heads merged by the Engine into boxes/scores over the 8,400 anchors of 640x640, model y -> frame y
        d_ = APP.dec_s(ra); p_ = APP.fast_nms(d_).copy()
        if APP.args.det in KPT_DETS and len(ra) > 2: attach_kpts(d_, ra[2], p_, 640, 640, APP.det_yoff)   # 11-14
        p_[:, 1] += APP.det_yoff; p_[:, 3] += APP.det_yoff
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
