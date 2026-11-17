"""Skin temperature in person boxes, from the camera's own temperature map (2026-11-14, research feature, display / log only).

Question (user): a single-frame, real-time cue that uses only the thermal radiation: is there skin (about 28-38 C on the surface) in the
box? The picture the tracker gets is AGC-equalised (not temperatures), so cam_pipe, started with LWIR_CAM_TEMP (app.py --cam-temp N),
puts the SDK's CalcTemp map of some frames into a shared-memory ring (src/camera/temp_ring.py). Per person box (real camera
coordinates) this module gives:
  share   part of the box's pixels at 28-38 C
  p95     95th percentile temperature of the box (nearest rank)
  top1    mean of the hottest 1 % of the box's pixels (at least 1 pixel)
  spot    a compact region of >= 20 pixels at 30-38 C (a face, a hand): the 30-38 C mask is opened with a 3x3 square (thin edges of
          hotter things such as a 45 C laptop, which pass through 30-38 C on their blurred border, are removed; outside the box counts
          as 'in range' for the erosion, as in OpenCV, so a face cut by the box edge is kept) and the largest 8-connected piece is taken;
          spot_px = its size (0 = none), spot_c = its 90th percentile temperature
  score   0.5 * min(1, share / 0.15) + 0.5 * (spot present): 0 .. 1
  skin    spot present (the cue)
Shown on the picture (preview_worker) as a small label next to each drawn person box: the spot temperature in green when there is a spot,
otherwise the box's top-1 % temperature in grey. Only while the switch is on AND the map is younger than FRESH_S; never on a recording
(the replay program replaces cam_pipe and the camera is not used). Nothing goes to the tracker or the gimbal.
Not yet known (device test): whether CalcTemp gives real temperatures right after cam_pipe's AGC read (slot flag bit 0) and whether the
factory calibration is right. A map whose median is outside MEDIAN_OK (C) is therefore not used at all (counted as 'implausible', its
median is in the state and the 60 s line).

Cost: the frame loop only hands the person boxes to one background thread (one slot, the newest frame replaces a waiting one) and
looks up the cue of each track id; the thread reads the ring and computes the cues once per new map (and for ids it has not seen on that
map). The label of a box therefore shows the cue of the newest map, at the box's current place.
Switch: POST /rules/set?temp_show=1|0 (not kept over a restart). A one-line summary goes to camera_events.log every 60 s while it is on.
The map can be up to FRESH_S older than the frame it is drawn on (a person who moves fast gets the temperature of where they were)."""
import collections
import threading
import time

import numpy as np

from ..camera import temp_ring

SKIN_LO, SKIN_HI = 28.0, 38.0   # share
SPOT_LO, SPOT_HI = 30.0, 38.0   # spot
SPOT_MIN_PX = 20
SHARE_FULL = 0.15
FRESH_S = 1.0
MEDIAN_OK = (-40.0, 80.0)   # a map whose median is outside this (C) is not used (CalcTemp after the AGC read not checked yet)
SUMMARY_S = 60.0
SKIN_COLOR, OTHER_COLOR = (80, 255, 80), (175, 175, 175)   # BGR

TEMP = dict(show=False, calls=0, ms_sum=0.0, ms_n=0, passes=0, work_ms_sum=0.0, work_n=0, maps=0, no_map=0, stale=0, skipped=0, boxes=0, skin_boxes=0,
            errors=0, last_error='', age=None, frame=None, map_seq=None, ring_state=None, every=None, agc_read=None, median_c=None, implausible=0,
            map_t_ns=None, emissivity=None, calc_us=None, by_id={}, cache_key=None,
            labels=(), labels_frame_t=None, t_sum=0.0, acc=None, recent=collections.deque(maxlen=8))
INLINE = [False]   # tests: run the background pass in the caller's thread
_READER = [None]
_WORKER = [None]
try:
    import cv2 as _cv2
    _K3 = np.ones((3, 3), np.uint8)
except ImportError:   # numpy fallback below gives the same masks
    _cv2 = None


def _open3(m):
    """3x3 opening of a bool mask; erosion treats outside as set, dilation as unset (OpenCV's default borders)"""
    if _cv2 is not None:
        return _cv2.morphologyEx(m.view(np.uint8), _cv2.MORPH_OPEN, _K3).view(bool)
    h, w = m.shape
    p = np.pad(m, 1, constant_values=True); e = np.ones_like(m)
    for dy in range(3):
        for dx in range(3):
            e &= p[dy:dy + h, dx:dx + w]
    p = np.pad(e, 1, constant_values=False); d = np.zeros_like(m)
    for dy in range(3):
        for dx in range(3):
            d |= p[dy:dy + h, dx:dx + w]
    return d


def _largest_piece(m):
    """bool mask -> bool mask of its largest 8-connected piece (OpenCV when present, else the whole mask)"""
    if _cv2 is None or not m.any():
        return m
    n, lab, st, _ = _cv2.connectedComponentsWithStats(m.view(np.uint8), connectivity=8)
    if n <= 2:
        return lab == 1
    return lab == 1 + int(np.argmax(st[1:, _cv2.CC_STAT_AREA]))


def box_cue(tc, box):
    """tc: float32 [H, W] temperatures in C; box (x0, y0, x1, y1) in the map's pixels -> dict, or None when the box is not on the map"""
    H, W = tc.shape
    x0 = max(0, int(np.floor(box[0]))); y0 = max(0, int(np.floor(box[1])))
    x1 = min(W, int(np.ceil(box[2]))); y1 = min(H, int(np.ceil(box[3])))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    c = np.ascontiguousarray(tc[y0:y1, x0:x1])
    v = c.ravel(); n = v.size
    share = float(np.count_nonzero((v >= SKIN_LO) & (v <= SKIN_HI))) / n
    k = max(1, n // 100); i95 = min(n - 1, int(np.ceil(0.95 * n)) - 1)
    part = np.partition(v, sorted({i95, n - k}))
    p95 = float(part[i95]); top1 = float(part[n - k:].mean())
    m = (c >= SPOT_LO) & (c <= SPOT_HI)
    spot_px, spot_c = 0, None
    if np.count_nonzero(m) >= SPOT_MIN_PX:
        piece = _largest_piece(_open3(m))
        px = int(np.count_nonzero(piece))
        if px >= SPOT_MIN_PX:
            s = c[piece]; j = int(np.ceil(0.9 * px)) - 1
            spot_px, spot_c = px, float(np.partition(s, j)[j])
    skin = spot_px > 0
    score = 0.5 * min(1.0, share / SHARE_FULL) + 0.5 * (1.0 if skin else 0.0)
    return dict(share=round(share, 4), p95=round(p95, 2), top1=round(top1, 2), spot_px=spot_px, spot_c=None if spot_c is None else round(spot_c, 2),
                score=round(score, 3), skin=skin)


def cues(tmap, boxes, frame_wh=None, t_offset=temp_ring.TEMP_OFFSET, t_div=temp_ring.TEMP_DIV):
    """tmap: uint16 CalcTemp map [H, W] (T_C = (v - t_offset) / t_div; or a float map in C); boxes [(x0, y0, x1, y1)] in real camera
    coordinates of a frame_wh (w, h) picture (default: the map's size) -> [dict or None] in the same order. Only the boxes' pixels are converted."""
    a = np.asarray(tmap)
    H, W = a.shape
    sx, sy = (1.0, 1.0) if frame_wh is None else (W / float(frame_wh[0]), H / float(frame_wh[1]))
    out = []
    for b in boxes:
        x0 = max(0, int(np.floor(b[0] * sx))); y0 = max(0, int(np.floor(b[1] * sy)))
        x1 = min(W, int(np.ceil(b[2] * sx))); y1 = min(H, int(np.ceil(b[3] * sy)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            out.append(None); continue
        crop = a[y0:y1, x0:x1]
        tc = temp_ring.to_celsius(crop, t_offset, t_div) if crop.dtype == np.uint16 else crop.astype(np.float32)
        out.append(box_cue(tc, (0, 0, x1 - x0, y1 - y0)))
    return out


def label(cue):
    """the text and colour drawn next to a box"""
    if cue['skin']:
        return '%.1fC' % cue['spot_c'], SKIN_COLOR
    return '%.1fC' % cue['top1'], OTHER_COLOR


def set_show(on):
    TEMP['show'] = bool(on)
    TEMP.update(labels=(), labels_frame_t=None, by_id={}, cache_key=None, map_t_ns=None)
    if on:
        TEMP['t_sum'] = time.monotonic(); TEMP['acc'] = None


def _reader():
    if _READER[0] is None:
        _READER[0] = temp_ring.TempRing()
    return _READER[0]


def _pass(people, ok, reader, frame_wh):
    """background pass: newest map from the ring; cues of the given people on it (all of them on a new map, only unseen ids otherwise)"""
    t0 = time.perf_counter()
    TEMP['passes'] += 1
    try:
        m = (reader or _reader()).newest()
        if m is None:
            TEMP['no_map'] += 1; TEMP.update(age=None, map_t_ns=None, by_id={}, cache_key=None)
            return
        age = (time.monotonic_ns() - m.t_ns) / 1e9
        TEMP.update(age=round(age, 3), frame=m.frame, map_seq=m.map_seq, ring_state=m.state, every=m.every, agc_read=bool(m.flags & temp_ring.FLAG_AGC_READ),
                    emissivity=round(m.emissivity, 3), calc_us=m.calc_us)
        if not ok:
            TEMP['skipped'] += 1
            return
        if not -0.2 <= age <= FRESH_S:   # old map (cam_pipe stopped the maps, CalcTemp seldom) or another clock
            TEMP['stale'] += 1; TEMP.update(map_t_ns=m.t_ns, by_id={}, cache_key=None)
            return
        key = (m.ring_id, m.map_seq, m.t_ns); new = TEMP['cache_key'] != key
        if new:   # 11-14: whether CalcTemp gives real temperatures after the AGC read is not known yet: a map whose middle value is no room temperature is not used
            TEMP['median_c'] = round(float(np.median(temp_ring.to_celsius(m.values[::8, ::8], m.t_offset, m.t_div))), 2)
        if TEMP['median_c'] is None or not MEDIAN_OK[0] <= TEMP['median_c'] <= MEDIAN_OK[1]:
            if new: TEMP['implausible'] += 1
            TEMP.update(map_t_ns=None, by_id={}, cache_key=key)
            return
        by_id = {} if new else dict(TEMP['by_id'])
        todo = [p for p in people if int(p[0]) not in by_id]
        got = []
        if todo:
            for p, cue in zip(todo, cues(m.values, [p[1:5] for p in todo], frame_wh, m.t_offset, m.t_div)):
                by_id[int(p[0])] = cue   # None = box not on the map (not tried again on this map)
                if cue is not None: got.append((int(p[0]), cue))
        TEMP.update(by_id=by_id, cache_key=key, map_t_ns=m.t_ns)
        if new: TEMP['maps'] += 1
        if got:
            TEMP['boxes'] += len(got); TEMP['skin_boxes'] += sum(1 for g in got if g[1]['skin'])
            TEMP['recent'].append(dict(at=time.strftime('%H:%M:%S'), frame=m.frame, map_seq=m.map_seq, age=round(age, 3), boxes=[dict(id=g[0], **g[1]) for g in got[:6]]))
            _acc(got, age)
        if todo:
            TEMP['work_ms_sum'] += (time.perf_counter() - t0) * 1e3; TEMP['work_n'] += 1
    except Exception as e:   # never let the display feature hurt the server
        TEMP['errors'] += 1; TEMP['last_error'] = repr(e)[:200]
    finally:
        _summary()


class _Worker:
    def __init__(self):
        self._slot, self._ev, self._lock = None, threading.Event(), threading.Lock()
        threading.Thread(target=self._run, name='skin-temp', daemon=True).start()

    def submit(self, job):
        with self._lock:
            self._slot = job
        self._ev.set()

    def _run(self):
        while True:
            self._ev.wait(); self._ev.clear()
            with self._lock:
                job, self._slot = self._slot, None
            if job is not None and TEMP['show']:
                try:   # 11-14 (review): _pass catches its own errors, but _summary runs in its finally; nothing may end this thread
                    _pass(*job)
                except Exception as e:
                    TEMP['errors'] += 1; TEMP['last_error'] = repr(e)[:200]


def on_frame(frame_t, people, ok=True, reader=None, now_ns=None, frame_wh=(640, 480)):
    """called by the server for every published frame while TEMP['show'] is on.
    people: [(track id, x0, y0, x1, y1)] person boxes in real camera coordinates; ok = False for a frame that is not the live camera
    (recording, invalid raw): no labels then. Returns [(track id, box, cue)] (cues of the newest map, boxes of this frame) and keeps it for
    preview_labels(frame_t, ...). With a reader given (tests) or INLINE on, the ring is read in this thread."""
    if not TEMP['show']:
        return []
    t0 = time.perf_counter()
    TEMP['calls'] += 1
    out = []
    try:
        people = [tuple(p) for p in people]
        job = (people, bool(ok), reader, frame_wh)
        if reader is not None or INLINE[0]:
            _pass(*job)
        else:
            if _WORKER[0] is None:
                _WORKER[0] = _Worker()
            _WORKER[0].submit(job)
        mt = TEMP['map_t_ns']
        if ok and mt is not None and -0.2 <= ((time.monotonic_ns() if now_ns is None else now_ns) - mt) / 1e9 <= FRESH_S:
            by_id = TEMP['by_id']
            for p in people:
                cue = by_id.get(int(p[0]))
                if cue is not None:
                    out.append((int(p[0]), tuple(float(v) for v in p[1:5]), cue))
    except Exception as e:
        TEMP['errors'] += 1; TEMP['last_error'] = repr(e)[:200]
        out = []
    TEMP['labels'] = tuple(out); TEMP['labels_frame_t'] = frame_t
    TEMP['ms_sum'] += (time.perf_counter() - t0) * 1e3; TEMP['ms_n'] += 1
    return out


def _acc(got, age):
    a = TEMP['acc'] or dict(maps=0, boxes=0, skin=0, age=0.0, temps=[])
    a['maps'] += 1; a['boxes'] += len(got); a['skin'] += sum(1 for g in got if g[1]['skin']); a['age'] += age
    if len(a['temps']) < 2000:
        a['temps'].extend(g[1]['spot_c'] if g[1]['skin'] else g[1]['top1'] for g in got)
    TEMP['acc'] = a


def _summary():
    now = time.monotonic()
    if now - TEMP['t_sum'] < SUMMARY_S:
        return
    TEMP['t_sum'] = now
    a, TEMP['acc'] = TEMP['acc'], None
    if a and a['maps']:
        t = 'TEMP 60 s: maps used %d, person boxes on them %d, with a skin spot %d, map age avg %.2f s, label C median %.1f, map median %s C, map %s (frame %s), emissivity %s (0 = camera default), CalcTemp %s us, pass %.1f ms' % (
            a['maps'], a['boxes'], a['skin'], a['age'] / a['maps'], float(np.median(a['temps'])) if a['temps'] else float('nan'), TEMP['median_c'], TEMP['map_seq'], TEMP['frame'],
            TEMP['emissivity'], TEMP['calc_us'], TEMP['work_ms_sum'] / TEMP['work_n'] if TEMP['work_n'] else 0.0)
    else:
        t = 'TEMP 60 s: no fresh temperature map with people (passes %d, no map %d, old map %d, implausible %d with median %s C, ring state %s, last age %s s): is the server started with --cam-temp N (LWIR_CAM_TEMP)?' % (
            TEMP['passes'], TEMP['no_map'], TEMP['stale'], TEMP['implausible'], TEMP['median_c'], TEMP['ring_state'], TEMP['age'])
    try:
        from ..camera import camera_extra
        camera_extra.note('server', t)
    except Exception:
        pass


def preview_labels(frame_t, sc, c=None, tids=None):
    """labels for the picture of frame frame_t: ((x1, y1), (x2, y2), text, colour) in picture pixels. c = the stabilizer's correction of the
    shown frame (picture = real + c), sc = picture scale, tids = the ids whose boxes are drawn (None = all)"""
    if not TEMP['show'] or TEMP['labels_frame_t'] != frame_t or not TEMP['labels']:
        return ()
    try:   # 11-14 (review): called from the frame loop (page_handler._preview_snapshot, outside its try): a bad value means no label, never an error there
        cx, cy = (c or (0.0, 0.0))[:2]
        out = []
        for tid, b, cue in TEMP['labels']:
            if tids is not None and tid not in tids:
                continue
            text, col = label(cue)
            out.append(((int((b[0] + cx) * sc), int((b[1] + cy) * sc)), (int((b[2] + cx) * sc), int((b[3] + cy) * sc)), text, col))
        return tuple(out)
    except Exception as e:
        TEMP['errors'] += 1; TEMP['last_error'] = repr(e)[:200]
        return ()


def state():
    r = _READER[0]
    return dict(show=TEMP['show'], age=TEMP['age'], fresh=TEMP['age'] is not None and -0.2 <= TEMP['age'] <= FRESH_S, calls=TEMP['calls'],
                ms=round(TEMP['ms_sum'] / TEMP['ms_n'], 3) if TEMP['ms_n'] else None, pass_ms=round(TEMP['work_ms_sum'] / TEMP['work_n'], 2) if TEMP['work_n'] else None,
                passes=TEMP['passes'], maps=TEMP['maps'], no_map=TEMP['no_map'], stale=TEMP['stale'], skipped=TEMP['skipped'], boxes=TEMP['boxes'],
                skin_boxes=TEMP['skin_boxes'], labels=len(TEMP['labels']), frame=TEMP['frame'], map_seq=TEMP['map_seq'], ring_state=TEMP['ring_state'], every=TEMP['every'],
                agc_read=TEMP['agc_read'], median_c=TEMP['median_c'], implausible=TEMP['implausible'], emissivity=TEMP['emissivity'], calc_us=TEMP['calc_us'],
                errors=TEMP['errors'], last_error=TEMP['last_error'], recent=list(TEMP['recent'])[-3:],
                ring=dict(r.stats) if r is not None else None, ring_path=r.path if r is not None else temp_ring.ring_path())
