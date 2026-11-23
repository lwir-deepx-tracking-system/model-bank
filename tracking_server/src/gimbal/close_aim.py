"""Close-person aim (2026-11-21, user "둘 다 해"): aim the gimbal higher on the FOLLOWED box when the person is very close.

Why: the detector boxes run head to hip. With the aim at the box centre (11-15 decision, --aim-y 0.5) a person who walks up to the
camera fills the picture and the head leaves the top edge; the detector then loses the person (seen live: head cut, no box, follow lost
after 10 s). So, for the followed box only:
  - box height below NEAR_LO (45 %) of the picture: the bridge's own aim (today 0.5 = box centre), nothing changes;
  - from NEAR_LO to NEAR_HI (80 %) the aim moves smoothly (smoothstep) up to CLOSE_AIM (0.25 of the box height from its top);
  - a box whose top touches the picture top edge (<= TOUCH_PX) counts as fully close, and stays so for TOUCH_HOLD_S after the last touch
    (otherwise the camera tilts up, the top leaves the edge, the aim drops back and the head is cut again);
  - the aim fraction is smoothed over time: up with RISE_S (a person walks in fast), down with FALL_S (box size flicker must not shake it);
  - never lower than the bridge's aim (a bridge already aiming at the head, 0.08, is left alone); the horizontal aim never changes.
How: the bridge computes its aim point as top + aim_y x h from the box it receives. The server moves the followed row up/down by
(aim - bridge aim) x h before it is written (same width / height), so the bridge's aim point becomes top + aim x h of the real box.
The predictive lead and the re-entry tilt limit see the same moved row, so all three stay on one reference. Lost / predicted rows
(PRED, SEEK) are not moved.
Switch: start value LWIR_CLOSE_AIM (default 1 = on, 0 = off); while running POST /rules/set?close_aim=1|0 (saved over a restart);
GET /rules shows close_aim, close_aim_now (aim fraction in use, None when not following), close_aim_rows (moved rows).
"""
import math
import os

NEAR_LO = 0.45       # box height / picture height where the aim starts to rise (top margin still 0.275 x picture at aim 0.5)
NEAR_HI = 0.80       # ... and where it is fully CLOSE_AIM (at aim 0.5 the top would be only 0.10 x picture from the edge)
CLOSE_AIM = 0.25     # aim point of a close person, fraction of the box height from its top (chest / shoulders of a head-to-hip box)
TOUCH_PX = 2.0       # a box top at or above this row (real picture pixels) touches the top edge
TOUCH_HOLD_S = 2.0   # a touch keeps the close aim this long
RISE_S = 0.3         # time constant of the aim moving up
FALL_S = 1.0         # time constant of the aim moving back down
GAP_S = 1.0          # longer than this without the followed row: start again from the target (no stale smoothing)


def _start():
    return os.environ.get('LWIR_CLOSE_AIM', '1').strip() != '0'


STATE = {'on': _start(), 'sel': None, 't': None, 'f': None, 'touch_t': None, 'aim': None, 'rows': 0}


def set_on(on):
    STATE['on'] = bool(on); reset()


def reset():
    STATE.update(sel=None, t=None, f=None, touch_t=None, aim=None)


def state():
    return dict(close_aim=bool(STATE['on']), close_aim_now=STATE['aim'], close_aim_rows=int(STATE['rows']))


def closeness(h_frac):
    """0 below NEAR_LO, 1 above NEAR_HI, smoothstep between."""
    if not math.isfinite(h_frac):
        return 0.0
    u = min(1.0, max(0.0, (h_frac - NEAR_LO) / (NEAR_HI - NEAR_LO)))
    return u * u * (3.0 - 2.0 * u)


def aim_for(base, f):
    """aim fraction for closeness f in [0, 1]: base at 0, min(base, CLOSE_AIM) at 1."""
    return base + (min(base, CLOSE_AIM) - base) * f


def step(selected, y1, h, picture_h, frame_t):
    """one frame of the followed box (real picture rows) -> smoothed closeness f in [0, 1]."""
    if STATE['sel'] != selected:
        reset(); STATE['sel'] = selected
    if y1 <= TOUCH_PX:
        STATE['touch_t'] = frame_t
    touch = STATE['touch_t'] is not None and 0.0 <= frame_t - STATE['touch_t'] <= TOUCH_HOLD_S
    target = 1.0 if touch else closeness(h / picture_h)
    t0, f = STATE['t'], STATE['f']
    dt = None if t0 is None else frame_t - t0
    if f is None or dt is None or dt < 0.0 or dt > GAP_S:
        f = target   # first frame of this follow, after a gap, or a clock jump back
    elif dt > 0.0:   # dt == 0: the same frame again, keep f
        tau = RISE_S if target > f else FALL_S
        f += (target - f) * (1.0 - math.exp(-(frame_t - t0) / tau))
    STATE['t'], STATE['f'] = frame_t, f
    return f


def apply(real, selected, base_aim, frame_t, picture_h, following=True):
    """real: rows (id, x1, y1, x2, y2, score, ...) in real picture pixels as written to the bridge. -> rows with the followed one
    moved vertically so that the bridge's aim point (top + base_aim x h) lands on top + aim x h of the real box. Others unchanged."""
    base = 0.5 if base_aim is None else float(base_aim)
    if not STATE['on'] or not selected or not following:
        reset()
        return real
    if not picture_h > 0 or not (isinstance(frame_t, (int, float)) and math.isfinite(frame_t)):
        return real
    idx = [i for i, p in enumerate(real) if int(p[0]) == selected]
    if len(idx) != 1:
        STATE['aim'] = None
        return real   # lost (or ambiguous): keep the smoothing, the gap rule decides on return
    p = real[idx[0]]
    y1, y2 = float(p[2]), float(p[4]); h = y2 - y1
    if not (math.isfinite(y1) and math.isfinite(h) and h > 0):
        return real
    aim = aim_for(base, step(selected, y1, h, float(picture_h), float(frame_t)))
    STATE['aim'] = round(aim, 3)
    d = (aim - base) * h
    if abs(d) < 0.05:   # below the 0.1 px the row is written with
        return real
    STATE['rows'] += 1
    out = list(real)
    out[idx[0]] = (p[0], p[1], y1 + d, p[3], y2 + d, *p[5:])
    return out
