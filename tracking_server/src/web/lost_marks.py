"""Prediction marks of a lost person, kept for as long as the person can still get the number back (11-16, user: the
dashed box and the search circle "vanish too quickly, both").

Before: the marks (pred_show: dashed predicted box, last-seen box, path, growing search circle) were drawn straight from
the tracker's lost pool (position_memory.POSP['boxes']). The tracker forgets a lost track after its buffer (30 frames,
about 1 s), so both marks vanished 1 s after the loss, although the number can still come back after that:
  - KeepID gives a returning person the old number for LOST_MAX frames (90, about 3 s) inside the growing search range;
  - the gimbal keeps waiting for the followed number until the bridge's --reacquire-ms (10 s), and the wrong-person guard
    can hand following back at the predicted place for that long.

Now this module remembers each lost person's marks after the tracker dropped the track:
  - the predicted box stays where the tracker last predicted it; it, the last-seen box and the path are carried through
    the camera motion of every frame (the GMC warps the tracker uses), so they stay on the same place of the scene while
    the gimbal turns;
  - the search circle keeps growing by the KeepID rule (radius = last height x min(0.5 + 0.05 x lost frames, 4));
  - the marks fade (alpha 1 -> FADE_MIN over their life) and end when the number can no longer come back: the followed
    person when the gimbal stops waiting for it, anybody else after the KeepID window (without KeepID: with the tracker);
  - they disappear at once when the number is seen again (also when KeepID moved it to a new tracker id).
Numbers are OUTPUT ids (after KeepID), as drawn on the boxes; the tracker's lost pool has tracker ids.

Display only: nothing here reaches the tracker, KeepID or the gimbal. update() is called once per published frame."""
import math

import numpy as np

from ..track.position_memory import POSP

FADE_MIN = 0.4          # alpha at the end of the life
FOLLOW_S = 10.0         # followed person: at most this long when the bridge's --reacquire-ms is not known
KEEPID_FRAMES = 90      # = reid.relabel_online.LOST_MAX = relink_online_judge.WINDOW
SR_BASE, SR_GROW, SR_MAX = 0.5, 0.05, 4.0   # = reid.relabel_online (the drawn circle is the KeepID search range)
PATH_MAX = 90

LM = dict(items={}, t=None, tracker=None)


def reset():
    LM['items'] = {}
    LM['t'] = None


def _warp(w, x, y):
    return float(w[0][0] * x + w[0][1] * y + w[0][2]), float(w[1][0] * x + w[1][1] * y + w[1][2])


def _carry(item, w):
    """Move the stored places (canvas coordinates) with the camera motion of the frames since the last update."""
    s = math.sqrt(abs(float(w[0][0] * w[1][1] - w[0][1] * w[1][0]))) or 1.0
    for key in ('box', 'last'):
        b = item[key]
        if b is not None:
            x, y = _warp(w, b[0], b[1])
            item[key] = (x, y, b[2] * s, b[3] * s)
    item['pts'] = [_warp(w, x, y) for x, y in item['pts']]


def update(frame_t, shown_ids, follow_id=0, *, tracker=None, id_map=None, keepid=False, fresh=True, motion=None,
           follow_s=FOLLOW_S, fps=30.0):
    """Once per published frame. shown_ids: output ids that have a box in this frame; follow_id: the output id the gimbal
    follows or waits for (0 = none); tracker: the tracker object (a new one forgets everything); id_map: KeepID's
    tracker id -> output id (None = the same numbers); keepid: KeepID is on; fresh: POSP['boxes'] is of this frame;
    motion(t0, t1) -> 2x3 canvas warp or None. Returns the entries to draw: [(output id, (cx, cy, w, h), info)], info =
    dict(src, n, last, pts, r, alpha, mem (True = after the tracker's own memory), age (s))."""
    if not POSP['show']:
        if LM['items']: reset()
        return []
    if tracker is not LM['tracker']:
        reset(); LM['tracker'] = tracker
    items, prev_t = LM['items'], LM['t']
    LM['t'] = frame_t
    step = frame_t - prev_t if prev_t is not None else 0.0
    if items and step > 0:
        w = None
        if motion is not None:
            try:
                w = motion(prev_t, frame_t)
                w = np.asarray(w, dtype=float) if w is not None else None
                if w is not None and (w.shape != (2, 3) or not np.isfinite(w).all()): w = None
            except Exception:   # noqa: BLE001  (display only: without the camera motion the marks stay where they are)
                w = None
        if w is not None:
            for item in items.values(): _carry(item, w)
    out_of = (lambda tid: id_map.get(tid)) if id_map is not None else (lambda tid: tid)
    seen_now = set()
    for e in (POSP['boxes'] if fresh else ()):
        tid, box = int(e[0]), tuple(float(v) for v in e[1][:4])
        info = e[2] if len(e) > 2 and e[2] else {}
        if not all(map(math.isfinite, box)): continue
        seen_now.add(tid)
        n = int(info.get('n', 0))
        item = items.get(tid)
        if item is None:
            last = info.get('last')
            item = items[tid] = dict(out=out_of(tid), last=tuple(float(v) for v in last) if last else None, pts=[],
                                     t0=frame_t - n / max(1.0, fps), lh=float(last[3]) if last else box[3])
        item.update(box=box, n=n, src=info.get('src', ''), mem=False, r=info.get('r'))
        item['pts'].append((box[0], box[1]))
        if len(item['pts']) > PATH_MAX: del item['pts'][:-PATH_MAX]
    frames = max(1, int(round(step * fps))) if step > 0 else 1
    out = []
    for tid in list(items):
        item = items[tid]
        if tid not in seen_now:   # the tracker forgot it (or gave no list this frame): our memory goes on
            item['n'] += frames; item['mem'] = True; item['r'] = None
        oid = out_of(tid)
        if oid is not None: item['out'] = oid
        age = max(0.0, frame_t - item['t0'])
        followed = bool(follow_id) and item['out'] == follow_id
        life = follow_s if followed else (KEEPID_FRAMES / max(1.0, fps) if keepid else None)
        if (oid is None or item['out'] in shown_ids                      # KeepID moved the number on / it is seen again
                or (item['mem'] and (life is None or age > life))):      # ... or it can no longer come back
            del items[tid]
            continue
        alpha = 1.0 if not life else max(FADE_MIN, 1.0 - (1.0 - FADE_MIN) * min(1.0, age / life))
        out.append((item['out'], item['box'], dict(src=item['src'], n=item['n'], last=item['last'], pts=list(item['pts']),
                                                   r=item['r'] if item['r'] is not None else item['lh'] * min(SR_BASE + SR_GROW * item['n'], SR_MAX),
                                                   alpha=alpha, mem=item['mem'], age=age)))
    return out
