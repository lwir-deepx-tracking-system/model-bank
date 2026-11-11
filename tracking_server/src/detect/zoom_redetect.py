"""작은 사람 다시 검출하기: 애매한 작은 상자 주변을 확대해 한 번 더 검출한다 (옵션 --zoom)."""
import numpy as np
from ..common.shared_values import PERSON_CLS


# ------------------------------------------------------------------------------------------------------------- zoom (v9)
ZOOM_HW = (480, 384)          # the zoom canvas = one NEWMRT480A tile (H, W)
FRAME_HW = (512, 640)


def zoom_iou_1n(b, B):
    inter = np.maximum(np.minimum(b[2:4], B[:, 2:4]) - np.maximum(b[:2], B[:, :2]), 0).prod(1)
    return inter / np.maximum((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2:4] - B[:, :2]).prod(1) - inter, 1e-9)


def zoom_nms_xyxy(b, s, thr):
    """greedy IoU NMS on xyxy rows; returns the kept row indices in score order"""
    order = np.argsort(-s, kind='stable'); keep = []
    while len(order):
        i = int(order[0]); keep.append(i)
        if len(order) == 1: break
        order = order[1:][zoom_iou_1n(b[i], b[order[1:]]) <= thr]
    return keep


def zoom_candidates(jo, yoff, lo, hi, maxh, maxn, extra=None):
    """jo [1,20,n] = the joined, DECODED, pre-NMS output of the normal tiles (band coords cx,cy,w,h + 16 sigmoid class scores; join_tiles has zeroed
    the seam). Candidates = anchors whose top class is person with lo <= person score < hi and box height < maxh, as frame xyxy (y + yoff);
    extra = optional [m,5] frame x1,y1,x2,y2,score rows (--zoom-lost) joined before the light NMS (.5). Returns [k,5] rows, score desc, k <= maxn."""
    sc = jo[0, 4:]; sp = sc[PERSON_CLS]
    idx = np.flatnonzero((sp >= lo) & (sp < hi) & (jo[0, 3] < maxh) & (jo[0, 3] > 0) & (jo[0, 2] > 0))
    if len(idx): idx = idx[sc[:, idx].argmax(0) == PERSON_CLS]
    rows = np.empty((len(idx), 5), np.float32)
    if len(idx):
        cx, cy, w, h = jo[0, 0, idx], jo[0, 1, idx] + yoff, jo[0, 2, idx], jo[0, 3, idx]
        rows[:, 0] = cx - w / 2; rows[:, 1] = cy - h / 2; rows[:, 2] = cx + w / 2; rows[:, 3] = cy + h / 2; rows[:, 4] = sp[idx]
    if extra is not None and len(extra): rows = np.concatenate([rows, np.asarray(extra, np.float32).reshape(-1, 5)])
    if not len(rows): return rows
    return rows[zoom_nms_xyxy(rows[:, :4], rows[:, 4], 0.5)[:maxn]]


def zoom_windows(cands, ctx, min_side, target, max_scale, floor=0.0, canvas_hw=ZOOM_HW, frame_hw=FRAME_HW):
    """One square context window per candidate row (x1,y1,x2,y2,score): side = max(ctx x h, min_side) (int), centred on the box, shifted (not shrunk)
    into the frame; scale = clamp(target / h, 1, max_scale) (floor > 0: target = floor when more than 2 candidates), S = round(side x scale);
    shelf-packed in candidate order into the canvas (windows never overlap); a window with no room is skipped. Returns the placed windows as dicts
    x0, y0, side (frame), S, scale = S / side (exact), px, py (canvas), cand (the row)."""
    H, W = canvas_hw; FH, FW = frame_hw; wins = []; x = y = shelf = 0
    tgt = floor if (floor > 0 and len(cands) > 2) else target
    for r in cands:
        h = float(r[3] - r[1]); side = min(int(round(max(ctx * h, min_side))), FH, FW)
        cx, cy = float(r[0] + r[2]) / 2, float(r[1] + r[3]) / 2
        x0 = min(max(int(round(cx - side / 2)), 0), FW - side); y0 = min(max(int(round(cy - side / 2)), 0), FH - side)
        S = min(int(round(side * min(max(tgt / max(h, 1e-3), 1.0), max_scale))), H, W)
        if x + S > W: y += shelf; x = 0; shelf = 0
        if y + S > H: continue
        wins.append(dict(x0=x0, y0=y0, side=side, S=S, scale=S / side, px=x, py=y, cand=np.asarray(r, np.float32))); x += S; shelf = max(shelf, S)
    return wins


def zoom_pack(gray, wins, canvas):
    """cut every window from gray (uint8 frame), resize INTER_LINEAR to S x S and paste it at (px, py) of canvas (uint8 H x W, pre-filled with the pad)"""
    import cv2
    for w in wins:
        canvas[w['py']:w['py'] + w['S'], w['px']:w['px'] + w['S']] = cv2.resize(gray[w['y0']:w['y0'] + w['side'], w['x0']:w['x0'] + w['side']], (w['S'], w['S']), interpolation=cv2.INTER_LINEAR)


def zoom_to_canvas(box, w):
    """frame xyxy (or xy point) -> canvas coords through window w"""
    b = np.asarray(box, np.float64).copy(); b[0::2] = (b[0::2] - w['x0']) * w['scale'] + w['px']; b[1::2] = (b[1::2] - w['y0']) * w['scale'] + w['py']; return b


def zoom_to_frame(box, w):
    """canvas xyxy (or xy point) -> frame coords through window w (exact inverse of zoom_to_canvas)"""
    b = np.asarray(box, np.float64).copy(); b[0::2] = (b[0::2] - w['px']) / w['scale'] + w['x0']; b[1::2] = (b[1::2] - w['py']) / w['scale'] + w['y0']; return b


def zoom_merge(pz, wins, normal_xyxy, acc_conf, acc_h, fates=None, frame_hw=FRAME_HW):
    """pz [n,6] canvas x1,y1,x2,y2,conf,cls = fast_nms of the canvas. A box belongs to the window that holds its centre (none -> dropped). Accept:
    person, conf >= acc_conf, mapped height < acc_h, IoU <= .5 against every normal detection (frame xyxy rows); then NMS .5 among the accepted.
    Returns [m,7] float32 frame x1,y1,x2,y2 (clipped to the frame), conf, cls (= person), window index. fates: optional counter dict."""
    def f(k, n=1):
        if fates is not None and n: fates[k] = fates.get(k, 0) + int(n)
    out = []
    if len(pz) and not wins: f('no_window', len(pz))
    elif len(pz):
        cx = (pz[:, 0] + pz[:, 2]) / 2; cy = (pz[:, 1] + pz[:, 3]) / 2
        for j in range(len(pz)):
            if int(pz[j, 5]) != PERSON_CLS: f('not_person'); continue
            if pz[j, 4] < acc_conf: f('low_conf'); continue
            wi = next((k for k, w in enumerate(wins) if w['px'] <= cx[j] < w['px'] + w['S'] and w['py'] <= cy[j] < w['py'] + w['S']), None)
            if wi is None: f('no_window'); continue
            b = zoom_to_frame(pz[j, :4], wins[wi])
            if b[3] - b[1] >= acc_h: f('tall'); continue
            b[0::2] = np.clip(b[0::2], 0, frame_hw[1]); b[1::2] = np.clip(b[1::2], 0, frame_hw[0])
            if b[2] - b[0] <= 0 or b[3] - b[1] <= 0: f('empty'); continue
            if len(normal_xyxy) and float(zoom_iou_1n(b, np.asarray(normal_xyxy, np.float64)).max()) > 0.5: f('overlaps_normal'); continue
            out.append([b[0], b[1], b[2], b[3], float(pz[j, 4]), float(PERSON_CLS), float(wi)])
    if not out: return np.zeros((0, 7), np.float32)
    out = np.asarray(out, np.float32); keep = zoom_nms_xyxy(out[:, :4], out[:, 4], 0.5); f('duplicate', len(out) - len(keep)); f('accepted', len(keep))
    return out[sorted(keep)]
