"""One live-track snapshot for the web click and existing PTZ transport.

The tracker computes boxes/IDs. This adapter only shares those results; it
does not implement a second tracker or change the camera control law.
"""
import math
import os
import time

import numpy as np

from ..camera import replay_link
from ..common import frame_math
from ..common.state import APP
from ..track.stabilizer import stab_people_real
from . import bounded_seek, follow_guard, native_control, predictive_lead, search_follow
from .gimbal_link import reset_lead, without_lead


def current_source():
    native = getattr(APP.LIVE, 'camera_bytetrack', None)
    source = native.control_source() if native is not None else ('python',)
    if source == ('python',):
        # Keep the actual object alive until the next publication, so an id()
        # reused by Python cannot make a new tracker look like the old one.
        rl = APP.__dict__.get('relabeler')
        if getattr(rl, 'compact_ids', False) and APP.__dict__.get('RULES', {}).get('relabel'):
            # A toggle/reset changes which person the same number identifies.
            # A stable token avoids comparing the mutable raw/public mapping.
            return ('python', APP.tracker, rl.public_identity_token)
        return ('python', APP.tracker)
    return source


def _boundary_locked(gb, source):
    """Called with the same lock as click and the bridge writer queue."""
    if hasattr(gb, 'people_source') and gb.people_source == source:
        return
    # Drop only queued tracking/motion work, preserving unrelated replies and
    # heartbeats. H must not be lost when the ordinary 64-command queue is full.
    gb.wq = type(gb.wq)(c for c in gb.wq
                       if c.split(b' ', 1)[0].strip() not in (b'A', b'H', b'V', b'F', b'G'))
    gb.wframe = None
    gb.observation_barrier = None  # H below already guarantees STOP for this generation.
    gb.observation_ns = 0
    gb.tracks_ticket = getattr(gb, 'tracks_ticket', 0) + 1
    gb.wq.appendleft(('H %d\n' % gb.tracks_ticket).encode())
    gb.people = []
    gb.click_owners = {}
    gb.people_source = source
    gb.people_t = 0.0
    gb.selection_reset = True
    gb.lead_pending = 0
    gb.reentry_tilt = None
    reset_lead()
    gb.wcond.notify()


def stop_with_ack_locked(gb):
    """Cancel pending selections and ask the bridge to acknowledge STOP."""
    gb.people_source = object()
    _boundary_locked(gb, current_source())


def halted(ticket):
    """Bridge has applied this boundary's STOP, not merely queued it."""
    gb = APP.GB
    with gb.wcond:
        if ticket == getattr(gb, 'tracks_ticket', None):
            gb.state.update(auto=0, lost=0, moving=0, id=0)
            gb.selection_reset = False
            gb.selection_ack_wait_state = False


def _bridge_option(gb, name):
    """The value of a launch option of the running bridge (the last one wins, as in the bridge), else None."""
    argv = getattr(getattr(gb, 'proc', None), 'args', None)
    if not isinstance(argv, (list, tuple)) or len(argv) % 2 != 1:
        return None
    value = None
    for flag, arg in zip(argv[1::2], argv[2::2]):
        if flag == name:
            value = arg
    return value


def _bridge_aim_y(gb):
    # v27 advertises LEAD only after parsing its launch options. Use that
    # process's actual argv, not the web display or mutable server settings.
    value = _bridge_option(gb, '--aim-y')
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if 0.0 <= value <= 1.0 else None


def search_enabled(gb):
    """11-15: the running bridge said SEARCH_CONTROL 1 (started with --search 1: it sweeps when the followed id is lost).
    Its own line, not the launch options: a wrapper may add options the server never sees."""
    return bool(getattr(gb, 'search_control_supported', False))


# 11-13 (user): FOLLOW HANDOFF. A person close to the camera who moves fast is often split by the detector (a whole-body box and a
# part box, then only the part) and the tracker gives the remaining box a new id. The gimbal follows an id, so it waited for
# the old one and following was dropped (recording run3 of 11-13: id 471 gone at 0.40 s, id 472 in the same place, following
# dropped at 2.0 s). With LWIR_FOLLOW_HANDOFF=1 (off by default): when the followed id has been missing for HANDOFF_MISS_S and
# one other box has stood in its last place (IoU >= HANDOFF_IOU with the last box, centre within HANDOFF_GATE of its width)
# for HANDOFF_HOLD_S without a change of id, following moves to that id (the same 'A id' a click sends). A person who only
# walks in front and leaves again does not stay that long in the same place; on the 11-13 recordings this rule changed id once,
# in the case above, and never while the old id came back later (two cases of a passer-by and a short gap were not taken).
HANDOFF_IOU, HANDOFF_GATE, HANDOFF_MISS_S, HANDOFF_HOLD_S, HANDOFF_MAX_S = .3, .5, .8, .5, 5.


def _iou(a, b):
    w_ = min(a[2], b[2]) - max(a[0], b[0]); h_ = min(a[3], b[3]) - max(a[1], b[1])
    if w_ <= 0 or h_ <= 0: return 0.
    i_ = w_ * h_; u_ = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i_
    return i_ / u_ if u_ > 0 else 0.


def _handoff_locked(gb, selected, frame_t):
    """Called with gb.wcond held, after gb.people is set for this frame. Returns the id following moved to, else 0."""
    if os.environ.get('LWIR_FOLLOW_HANDOFF') != '1' or search_enabled(gb) or guard_enabled():   # 11-15: a searching gimbal moves the picture; _recatch_locked / _guard_locked replace this rule
        return 0
    if (not selected or not gb.state.get('auto') or getattr(gb, 'lead_pending', 0) or getattr(gb, 'mover', None)
            or getattr(gb, 'selection_reset', False) or not gb.people):
        if not selected or not gb.state.get('auto'): gb.handoff = None
        return 0
    h = getattr(gb, 'handoff', None)
    if h is None or h['sel'] != selected:
        h = gb.handoff = dict(sel=selected, box=None, t=0., cand=0, cand_t=0.)
    mine = next((p for p in gb.people if int(p[0]) == selected), None)
    if mine is not None:
        h.update(box=tuple(float(v) for v in mine[1:5]), t=frame_t, cand=0)
        return 0
    lb = h['box']
    if lb is None or not 0 < frame_t - h['t'] <= HANDOFF_MAX_S:
        return 0
    w, lcx = lb[2] - lb[0], (lb[0] + lb[2]) / 2.
    best = None
    for p in gb.people:
        if int(p[0]) == selected: continue
        b = tuple(float(v) for v in p[1:5]); o = _iou(b, lb)
        if o >= HANDOFF_IOU and abs((b[0] + b[2]) / 2. - lcx) <= HANDOFF_GATE * w and (best is None or o > best[1]):
            best = (int(p[0]), o)
    if best is None:
        h['cand'] = 0
        return 0
    if best[0] != h['cand']:
        h['cand'], h['cand_t'] = best[0], frame_t
    if frame_t - h['t'] < HANDOFF_MISS_S or frame_t - h['cand_t'] < HANDOFF_HOLD_S or len(gb.wq) >= 62:
        return 0
    reset_lead()
    gb.lead_pending = getattr(gb, 'lead_pending', 0) + 1
    if gb.wframe is not None:   # the bridge sees the new id in a frame before it is selected (as for a click)
        gb.wq.append(without_lead(gb.wframe)); gb.wframe = None
    gb.wq.append(('A %d\n' % best[0]).encode())
    gb.handoff = dict(sel=best[0], box=None, t=0., cand=0, cand_t=0.)
    gb.handoff_n = getattr(gb, 'handoff_n', 0) + 1
    print('FOLLOW handoff %d -> %d (IoU %.2f with its last box, %.2f s after it was last seen)' % (selected, best[0], best[1], frame_t - h['t']), flush=True)
    return best[0]


# 11-15 (user): SEARCH RE-CATCH UNDER A NEW ID, only with the bridge option --search 1 AND LWIR_SEARCH_NEW_ID=1 (both off by
# default). The user's rule is that a re-catch must give back the SAME id (tracker, KeepID appearance, Kalman position memory):
# the bridge follows only its selected id, so by default a person who comes back under a new id is not taken. This opt-in
# rule is kept for experiments with that case only. The last seen box is carried through the camera motion (GMC
# warp of every frame pair, the same warps the tracker uses) and moved on with the person's last speed for RECATCH_WALK_S.
# A candidate must (1) not have been on screen when ours was last seen (so it is not one of the others around it), (2) lie
# within the gate (RECATCH_R0 x box width + RECATCH_GROW px/s since last seen, at most RECATCH_RMAX x width) with a height
# within RECATCH_SIZE x, (3) be the ONLY box near the prediction (none else within RECATCH_CROWD x the gate: in a crowd the
# rule does nothing), (4) stay so for RECATCH_N frames in a row. A frame pair without camera motion breaks the chain: no
# re-catch for that loss. The selection is the same 'A id' a click sends.
RECATCH_N, RECATCH_R0, RECATCH_GROW, RECATCH_RMIN, RECATCH_RMAX = 3, .5, 120., 24., 1.5
RECATCH_SIZE, RECATCH_CROWD, RECATCH_WALK_S, RECATCH_MAX_S, RECATCH_MAX_V = 1.5, 1.5, 1.0, 10., 600.
CANVAS_Y = 16.0   # GMC warps work on the padded canvas: people rows are 16 px higher


def _motion(t0, t1):
    """The 2x3 camera-motion warp from frame time t0 to frame time t1 composed from APP.CTL['motion_hist'], else None."""
    if t0 == t1:
        return np.array([[1., 0., 0.], [0., 1., 0.]])
    hist = APP.CTL.get('motion_hist')
    if isinstance(hist, frame_math.MotionHistory):
        out = hist.compose(t0, t1)
        if out is not None:
            return out
    hist = list(hist or ())
    out, t = None, t0
    for _ in range(len(hist)):
        step = next((m for m in hist if m[0] == t), None)
        if step is None:
            return None
        w = np.asarray(step[2], dtype=float)
        if w.shape != (2, 3) or not np.isfinite(w).all():
            return None
        out = w if out is None else np.vstack([w, [0., 0., 1.]])[:2] @ np.vstack([out, [0., 0., 1.]])
        t = step[1]
        if t == t1:
            return out
        if t > t1:
            return None
    return None


def _warp_point(w, x, y):
    px, py = w @ (x, y + CANVAS_Y, 1.0)
    return float(px), float(py - CANVAS_Y)


def _recatch_locked(gb, selected, frame_t):
    """Called with gb.wcond held, after gb.people is set for this frame. Returns the id following moved to, else 0."""
    if not search_enabled(gb) or os.environ.get('LWIR_SEARCH_NEW_ID') != '1':
        return 0
    if (not selected or not gb.state.get('auto') or getattr(gb, 'lead_pending', 0) or getattr(gb, 'mover', None)
            or getattr(gb, 'selection_reset', False) or not getattr(gb, 'people_t', 0.0)):
        if not selected or not gb.state.get('auto'): gb.recatch = None
        return 0
    h = getattr(gb, 'recatch', None)
    if h is None or h['sel'] != selected:
        h = gb.recatch = dict(sel=selected, t_seen=None, ok=False, cand=0, cand_n=0, vx=0., vy=0.)
    mine = [p for p in gb.people if int(p[0]) == selected]
    if mine:
        p = mine[0]
        cx, cy, w, hh = (p[1] + p[3]) / 2., (p[2] + p[4]) / 2., p[3] - p[1], p[4] - p[2]
        if h['t_seen'] is not None and 0 < frame_t - h['t_seen'] <= .25:
            m = _motion(h['t_seen'], frame_t)
            if m is not None:
                px, py = _warp_point(m, h['cx'], h['cy']); dt = frame_t - h['t_seen']
                h['vx'] = max(-RECATCH_MAX_V, min(RECATCH_MAX_V, .7 * h['vx'] + .3 * (cx - px) / dt))
                h['vy'] = max(-RECATCH_MAX_V, min(RECATCH_MAX_V, .7 * h['vy'] + .3 * (cy - py) / dt))
        else:
            h['vx'] = h['vy'] = 0.
        h.update(cx=cx, cy=cy, w=w, h=hh, t_seen=frame_t, pred=(cx, cy, w, hh), pred_t=frame_t, ok=True,
                 ids={int(q[0]) for q in gb.people}, cand=0, cand_n=0)
        return 0
    if h['t_seen'] is None or not h['ok']:
        return 0
    age = frame_t - h['t_seen']
    if not 0 < age <= RECATCH_MAX_S or frame_t <= h['pred_t']:
        return 0
    m = _motion(h['pred_t'], frame_t)
    if m is None:
        h['ok'] = False
        gb.recatch_broken = getattr(gb, 'recatch_broken', 0) + 1
        print('SEARCH re-catch off for this loss of %d: no camera motion for a frame pair' % selected, flush=True)
        return 0
    px, py, pw, ph = h['pred']
    s = math.sqrt(abs(float(np.linalg.det(m[:, :2])))) or 1.
    px, py = _warp_point(m, px, py)
    walk = max(0., min(age, RECATCH_WALK_S) - min(h['pred_t'] - h['t_seen'], RECATCH_WALK_S))
    px += h['vx'] * walk; py += h['vy'] * walk; pw *= s; ph *= s
    h['pred'], h['pred_t'] = (px, py, pw, ph), frame_t
    gate = min(max(RECATCH_RMAX * pw, RECATCH_RMIN), max(RECATCH_RMIN, RECATCH_R0 * pw) + RECATCH_GROW * age)
    cands, near = [], 0
    for q in gb.people:
        qid = int(q[0])
        if qid == selected: continue
        d = math.hypot((q[1] + q[3]) / 2. - px, (q[2] + q[4]) / 2. - py)
        if d <= RECATCH_CROWD * gate: near += 1
        size = (q[4] - q[2]) / ph if ph > 0 else 0.
        if qid not in h['ids'] and d <= gate and 1. / RECATCH_SIZE <= size <= RECATCH_SIZE:
            cands.append((qid, d, size))
    if len(cands) != 1 or near != 1:
        h['cand'], h['cand_n'] = 0, 0
        return 0
    qid, d, size = cands[0]
    if qid != h['cand']:
        h['cand'], h['cand_n'] = qid, 0
    h['cand_n'] += 1
    if h['cand_n'] < RECATCH_N or len(gb.wq) >= 62:
        return 0
    reset_lead()
    gb.lead_pending = getattr(gb, 'lead_pending', 0) + 1
    if gb.wframe is not None:   # the bridge sees the new id in a frame before it is selected (as for a click)
        gb.wq.append(without_lead(gb.wframe)); gb.wframe = None
    gb.wq.append(('A %d\n' % qid).encode())
    gb.recatch = dict(sel=qid, t_seen=None, ok=False, cand=0, cand_n=0, vx=0., vy=0.)
    gb.recatch_n = getattr(gb, 'recatch_n', 0) + 1
    print('SEARCH re-catch %d -> %d (%.0f px from the prediction, gate %.0f px, height x%.2f, %.2f s after it was last seen)'
          % (selected, qid, d, gate, size, age), flush=True)
    return qid


def coast_enabled(gb):
    """11-15: the running bridge said COAST_CONTROL 1 (started with --coast 1: it follows the predicted place for a short time
    when the followed id is missing). Only such a bridge is sent the PRED suffix (an older bridge refuses unknown suffixes)."""
    return bool(getattr(gb, 'coast_control_supported', False))


# 11-15 (live A/B test on the real camera): a runtime switch for the coast that keeps the bridge and the follow selection.
# LWIR_COAST_SWITCH=<file>: the coast is on only while that file exists. Its first word 'vel' / 'fused' (optional) overrides
# LWIR_COAST_MODE. The file is read at most every COAST_SWITCH_S. Off = no PRED is sent, so a --coast 1 bridge stops on a
# loss exactly as a bridge without --coast. Without the variable: always on (the bridge's --coast 1 decides, as before).
COAST_SWITCH_S = 0.1
COAST_SWITCH = dict(path=None, checked=float('-inf'), on=True, mode=None, said=None)


def coast_switch():
    """-> (on, mode or None). See COAST_SWITCH."""
    path = os.environ.get('LWIR_COAST_SWITCH')
    if not path:
        return True, None
    s = COAST_SWITCH
    now = time.monotonic()
    if s['path'] != path or now - s['checked'] >= COAST_SWITCH_S:
        try:
            with open(path, encoding='ascii', errors='replace') as f:
                word = (f.read(64).split() or [''])[0]
            on, mode = True, (word if word in ('vel', 'fused') else None)
        except OSError:
            on, mode = False, None
        s.update(path=path, checked=now, on=on, mode=mode)
        if s['said'] != (on, mode):
            s['said'] = (on, mode)
            print('COAST switch %s' % (('on (%s)' % (mode or coast_mode_env())) if on else 'off: no PRED is sent'), flush=True)
    return s['on'], s['mode']


def coast_switch_state():
    """'off' / 'vel' / 'fused' for /gimbal/people state 'coast_sw' (only with LWIR_COAST_SWITCH), else None."""
    if not os.environ.get('LWIR_COAST_SWITCH'):
        return None
    on, mode = coast_switch()
    return (mode or coast_mode_env()) if on else 'off'


# 11-16 (user: while the prediction is shown the gimbal should search there too): SEEK, see search_follow.py. On when the
# running bridge said SEEK_CONTROL 1 (started with --seek 1); that one option is the switch. For tests, LWIR_SEEK_SWITCH=<file>
# makes it a runtime switch like LWIR_COAST_SWITCH: on only while the file exists (read at most every COAST_SWITCH_S).
SEEK_SWITCH = dict(path=None, checked=float('-inf'), on=True, said=None)


def seek_enabled(gb):
    if not (getattr(gb, 'seek_control_supported', False)
            or getattr(gb, 'bounded_seek_supported', False)):
        return False
    path = os.environ.get('LWIR_SEEK_SWITCH')
    if not path:
        return True
    s = SEEK_SWITCH
    now = time.monotonic()
    if s['path'] != path or now - s['checked'] >= COAST_SWITCH_S:
        s.update(path=path, checked=now, on=os.path.exists(path))
        if s['said'] != s['on']:
            s['said'] = s['on']
            print('SEEK switch %s' % ('on' if s['on'] else 'off: no SEEK is sent'), flush=True)
    return s['on']


def _seek_locked(gb, g, d, selected, frame_t, kal, on_guard, v2):
    """Called by _guard_locked after the guard's step of this frame. -> the box the gimbal should look at (gb.people
    coordinates) or None. Also sets /gimbal/people state 'seek' = [phase, cx, cy, radius, amplitude, pan, target or None]
    (search_follow.PHASES; the circle and the sweep the page draws) and the guard's wide gate for the next frame."""
    sk = getattr(gb, 'fseek', None)
    if sk is None or sk.sel != selected:
        sk = gb.fseek = search_follow.Seek(selected)
    lost = bool(d['mask']) or not any(int(q[0]) == selected for q in gb.people)
    if not lost or g.trust is None:
        if g.trust is not None and g.trust_t == frame_t:
            sk.seen(g.trust, gb.people, getattr(g, 'partners', ()))
        elif sk.phase != 'idle':
            sk.seen_untrusted()
        if v2: g.wide = None
        gb.state.pop('seek', None)
        gb.seek_phase = None
        return None
    sy = _bridge_option(gb, '--aim-screen-y')
    try: sy = float(sy) if sy is not None else .5
    except ValueError: sy = .5
    ay = _bridge_aim_y(gb)
    can = (not on_guard) or (v2 and bool(g.tmpl))
    box, st = sk.step(frame_t, g.age(frame_t), g.trust, (g.vx, g.vy), kal, g.pred, _motion, (float(APP.LIVE_W), float(APP.LIVE_H)),
                      (APP.LIVE_W / 2., sy * APP.LIVE_H), .5 if ay is None else ay, hidden=bool(d['mask']), can_reaccept=can, people=gb.people)
    if v2:   # the same number may come back anywhere in the displayed circle, by appearance (not in a crowd)
        g.wide = (st['c'][0], st['c'][1], st['r']) if st['phase'] not in ('crowd', 'blind') else None
    if st['phase'] != getattr(gb, 'seek_phase', None):
        gb.seek_phase = st['phase']
        if st['phase'] in ('sweep', 'return', 'crowd', 'blind'):
            print('SEEK %d: %s (circle %.0f px, sweep +-%.0f px; when last trusted %d near, %d others; now %d near, %d others)'
                  % (selected, st['phase'], st['r'], st['a'], st['near'], st['others'], st['now_near'], st['now_others']), flush=True)
    gb.state['seek'] = [st['ph'], st['c'][0], st['c'][1], st['r'], st['a'], st['pan'], st['to']]
    return box


def guard_enabled():
    """11-15: LWIR_FOLLOW_GUARD=1 (v1, position and motion) or 2 (v2, + appearance; off by default): the wrong-person guard of
    follow_guard.py decides what the gimbal follows."""
    return os.environ.get('LWIR_FOLLOW_GUARD') in ('1', '2')


def _appearance(oid):
    """11-15 guard v2: the newest RepVGG feature the KeepID / relink judge computed for a box of output id oid, as
    (feature, age in frames), or None. Read only; the judge keeps per tracker id observations ('m1' = feature)."""
    rl = getattr(APP, 'relabeler', None)
    if rl is None:
        return None
    try:
        now = int(getattr(rl, 't', 0))
        idm = getattr(rl, 'id_map', {}) or {}
        hist = getattr(rl, 'hist', None)
        if isinstance(hist, dict) and hist and isinstance(next(iter(hist.values()), None), list):   # relink_online_judge
            best = None
            for b, o in idm.items():
                if o != oid: continue
                for ob in reversed(hist.get(b, [])[-12:]):
                    if ob.get('m1') is not None:
                        if best is None or ob['t'] > best[1]: best = (ob['m1'], ob['t'])
                        break
            return None if best is None else (best[0], now - best[1])
        g = getattr(rl, 'gallery', {}).get(oid)   # relabel_online
        if g:
            return (g[-1], now - getattr(rl, 'last_emb', {}).get(oid, -10 ** 6))
    except Exception:   # noqa: BLE001
        return None
    return None


def predict_log_enabled():
    """11-15: LWIR_FOLLOW_PREDICT_LOG=1 (tests, off by default): run the prediction for /gimbal/people state 'guard' even when
    neither the coast nor the guard is on. It never changes what is sent to the bridge."""
    return os.environ.get('LWIR_FOLLOW_PREDICT_LOG') == '1'


def _kalman_box(selected):
    """11-15: the tracker's own prediction (Kalman + GMC, OC-SORT/ByteTrack lost pool) of the selected OUTPUT id, as (cx, cy, w, h)
    in gb.people coordinates, or None. Output ids come from the tracker ids through KeepID (relabeler.id_map)."""
    try:
        rl = getattr(APP, 'relabeler', None)
        idm = getattr(rl, 'id_map', None) if rl is not None else None
        bids = {b for b, o in idm.items() if o == selected} if idm else {selected}
        for t in getattr(APP.tracker, 'lost', ()) or ():
            if int(t.track_id) in bids and t.mean is not None:
                m = np.asarray(t.mean, dtype=float)[:4]
                if np.isfinite(m).all() and m[2] > 0 and m[3] > 0:
                    return (float(m[0]), float(m[1]) - follow_guard.CANVAS_Y, float(m[2]), float(m[3]))
    except Exception:   # noqa: BLE001  (a tracker without a lost pool)
        return None
    return None


def coast_mode_env():
    return 'fused' if os.environ.get('LWIR_COAST_MODE') == 'fused' else 'vel'


def coast_mode():
    """11-15: LWIR_COAST_MODE = vel (default: follow the velocity prediction) | fused (keep the velocity prediction and the
    tracker's position memory in the picture with the smallest motion, follow_guard.keep_in_view). The first word of the
    LWIR_COAST_SWITCH file overrides it."""
    return coast_switch()[1] or coast_mode_env()


GUARD_PHASES = {'track': 0, 'risk': 1, 'lost': 2, 'masked': 3}


def _bridge_aim_screen_y(gb):
    value = _bridge_option(gb, '--aim-screen-y')
    try:
        return float(value) if value is not None else .5
    except ValueError:
        return .5


def _guard_locked(gb, selected, frame_t, correction=None):
    """Called with gb.wcond held, after gb.people is set for this frame. -> (id to hide from the bridge this frame or 0,
    predicted (cx, cy, w, h) in gb.people coordinates for the PRED suffix or None). follow_guard.py; off unless the bridge
    has --coast 1, LWIR_FOLLOW_GUARD=1 or LWIR_FOLLOW_PREDICT_LOG=1 (log only). The coast part also needs the runtime
    switch LWIR_COAST_SWITCH (when that is set)."""
    sw = coast_switch_state()
    if sw is not None:
        gb.state['coast_sw'] = sw   # read by /gimbal/people (live A/B log)
    on_guard, on_coast = guard_enabled(), coast_enabled(gb) and coast_switch()[0]
    seek_supported = seek_enabled(gb)
    bounded = getattr(gb, 'bounded_seek_supported', False)
    want_bounded = os.environ.get('LWIR_COAST_BOUNDED_SEEK') == '1'
    on_seek = seek_supported and not bounded and not want_bounded
    gb.seek_out = gb.seek_raw = None
    if (not (on_guard or on_coast or on_seek or predict_log_enabled()) or not selected or not gb.state.get('auto') or getattr(gb, 'lead_pending', 0)
            or getattr(gb, 'mover', None) or getattr(gb, 'selection_reset', False) or not getattr(gb, 'people_t', 0.0)):
        previous = getattr(gb, 'fguard', None)
        if previous is not None and previous.bounded_seek is not None:
            previous.bounded_seek.cancel()
        if not selected or not gb.state.get('auto'):
            gb.fguard = None
            gb.fseek = None
            gb.state.pop('guard', None)
            gb.state.pop('seek', None)
        return 0, None
    v2 = os.environ.get('LWIR_FOLLOW_GUARD') == '2'
    g = getattr(gb, 'fguard', None)
    if g is None or g.sel != selected or g.guard != on_guard or isinstance(g, follow_guard.FollowGuardV2) != v2:
        g = gb.fguard = follow_guard.FollowGuardV2(selected) if v2 else follow_guard.FollowGuard(selected, on_guard)
    if v2:
        g.app, g.kal = _appearance, _kalman_box(selected)
    d = g.step(frame_t, gb.people, _motion)
    p = d['pred'] or g.pred
    kal = _kalman_box(selected) if d['mask'] or not any(int(q[0]) == selected for q in gb.people) else None
    fused = on_coast and coast_mode() == 'fused'
    if fused and on_guard and os.environ.get('LWIR_COAST_UPPER_GUARD') == '1':
        if g.upper_coast_view is None:
            g.upper_coast_view = follow_guard.UpperCoastView()
        if g.trust_t == frame_t and d['phase'] == 'track' and not d['mask'] and not d['reselect']:
            cx, cy, width, height = g.trust
            dx, dy = correction if correction is not None else (0., 0.)
            g.upper_coast_view.observe(frame_t, (cx - dx, cy - dy, width, height), _motion,
                                       (float(APP.LIVE_W), float(APP.LIVE_H)))
    else:
        g.upper_coast_view = None
    if d['pred'] is not None and fused:
        sy = _bridge_aim_screen_y(gb)
        ay = _bridge_aim_y(gb)
        ay = .5 if ay is None else ay
        k = kal if kal is not None and math.hypot(kal[0] - d['pred'][0], kal[1] - d['pred'][1]) <= follow_guard.COAST_MAX_H * d['pred'][3] else None
        d['pred'] = follow_guard.keep_in_view([d['pred'], k], g.age(frame_t), (APP.LIVE_W / 2., sy * APP.LIVE_H), ay,
                                              (float(APP.LIVE_W), float(APP.LIVE_H)))
    if seek_supported and bounded and want_bounded and g.upper_coast_view is not None:
        if g.bounded_seek is None:
            g.bounded_seek = bounded_seek.BoundedSeek()
        if gb.people or d['reselect']:
            if g.trust_t == frame_t and d['phase'] == 'track' and not d['mask'] and not d['reselect']:
                g.bounded_seek = bounded_seek.BoundedSeek()
            else:
                g.bounded_seek.cancel()
        elif not d['mask']:
            aim = _bridge_aim_y(gb)
            dx, dy = correction if correction is not None else (0., 0.)
            prediction = (p[0] - dx, p[1] - dy, p[2], p[3]) if p is not None else None
            gb.seek_raw = g.bounded_seek.step(
                frame_t, g.upper_coast_view, _motion, (float(APP.LIVE_W), float(APP.LIVE_H)),
                (APP.LIVE_W / 2., _bridge_aim_screen_y(gb) * APP.LIVE_H), .5 if aim is None else aim,
                prediction=prediction)
        else:
            g.bounded_seek.cancel()
        if g.bounded_seek.first is not None:
            d['pred'] = None
    if on_seek and not d['reselect']:   # 11-16: where to look once the prediction is no longer trusted (search_follow.py)
        gb.seek_out = _seek_locked(gb, g, d, selected, frame_t, kal, on_guard, v2)
        if gb.seek_out is not None:
            d['pred'] = None   # one target per frame: the search has taken over from the coast
    elif not on_seek:
        gb.state.pop('seek', None)
        if v2: g.wide = None
    # read by /gimbal/people (tests): phase, mask, velocity prediction (cx, cy, h), PRED sent, tracker position memory (cx, cy, h),
    # the PRED box centre sent
    gb.state['guard'] = [GUARD_PHASES.get(d['phase'], -1), int(d['mask']),
                         None if p is None else [round(p[0], 1), round(p[1], 1), round(p[3], 1)], int(d['pred'] is not None and on_coast),
                         None if kal is None else [round(kal[0], 1), round(kal[1], 1), round(kal[3], 1)],
                         None if d['pred'] is None else [round(d['pred'][0], 1), round(d['pred'][1], 1)]]
    if d['reselect'] and len(gb.wq) < 62:
        reset_lead()
        gb.lead_pending = getattr(gb, 'lead_pending', 0) + 1
        if gb.wframe is not None:   # the bridge sees the new id in a frame before it is selected (as for a click)
            gb.wq.append(without_lead(gb.wframe)); gb.wframe = None
        gb.wq.append(('A %d\n' % d['reselect']).encode())
        gb.guard_n = getattr(gb, 'guard_n', 0) + 1
        print('FOLLOW guard %d -> %d (%s)' % (selected, d['reselect'], d['why']), flush=True)
        return 0, None
    if v2 and g.wide_reaccepts > getattr(g, 'wide_said', 0):   # 11-16: taken back away from the predicted place (search)
        g.wide_said = g.wide_reaccepts
        gb.wide_n = getattr(gb, 'wide_n', 0) + 1
        print('FOLLOW guard: %d followed again inside the search circle (%s)' % (selected, g.why), flush=True)
    if d['mask'] and not getattr(gb, 'guard_masking', False):
        print('FOLLOW guard: %d hidden from the gimbal (%s)' % (selected, d['why']), flush=True)
    gb.guard_masking = bool(d['mask'])
    return (selected if d['mask'] else 0), (d['pred'] if on_coast else None)


def _search_velocity_locked(gb, selected, observed, frame_t, correction, tilt_hold=False):
    """11-15: ' VEL <id> <vx> <vy>' for a bridge that said SEARCH_CONTROL 1: the followed person's velocity against the
    background (picture px/s) from the predictive-control estimator (predictive_lead.estimate, the same one LEAD uses; when
    LEAD is on its own sample of this frame is reused). The bridge uses it only to keep turning where the person should be
    after a loss, so the tracker / KeepID can give the same id back. '' when there is nothing to send."""
    if not getattr(gb, 'search_control_supported', False) or not selected or not gb.state.get('auto'):
        gb.search_lv = None
        return ''
    lv = APP.CTL.get('lv')
    if not (APP.CTL.get('lead') and lv is not None and getattr(lv, 'track_id', 0) == selected and getattr(lv, 'time', None) == frame_t):
        why = predictive_lead.LAST.get('why')
        lv, _ = predictive_lead.estimate(getattr(gb, 'search_lv', None), selected, observed, frame_t,
                                         APP.CTL.get('lead_motion'), 0, aim_y=_bridge_aim_y(gb) if _bridge_aim_y(gb) is not None else .5,
                                         correction=correction)
        predictive_lead.LAST['why'] = why   # the lead counters stay those of the lead path
        gb.search_lv = lv
    return '' if lv is None or lv.time != frame_t else native_control.velocity(selected, lv.vx, lv.vy, tilt_hold)


def _record_click_owners(gb, raw_owners, namespace):
    previous = getattr(gb, 'click_owners', {}) if getattr(gb, 'click_namespace', None) is namespace else {}
    gb.click_owners = {}
    gb.click_namespace = namespace
    visible = {p[0] for p in gb.people}
    for public, raw in raw_owners.items():
        if public in visible:
            owner = previous.get(public)
            gb.click_owners[public] = owner if owner is not None and owner[0] == raw else (raw, object())


def publish(people, correction, frame_t, raw16, source, blocked=False, *, healthy_live=False,
            click_owners=None, click_namespace=None):
    """Publish the same IDs as the drawn boxes, with existing coordinate rules."""
    gb = APP.GB
    if gb is None:
        return 0
    with gb.wcond:
        if not gb.wopen:
            gb.people = []
            reset_lead()
            return 0
        # Validate before converting to signed int64 ns or advancing any watermark.
        now = time.perf_counter()
        stamp_ok = (isinstance(frame_t, (int, float)) and math.isfinite(frame_t)
                    and 0 < frame_t <= now and 0 < frame_t * 1e9 < 2**63)
        captured_ns = int(frame_t * 1e9) if stamp_ok else 0
        gap = frame_t - getattr(gb, 'people_t', 0.0) if stamp_ok else -1
        if getattr(gb, 'people_t', 0.0) and not 0 < gap <= 1.0:
            gb.people_source = None
        _boundary_locked(gb, source)
        marked = getattr(gb, 'observation_status_supported', False)
        healthy = (healthy_live and stamp_ok and now - frame_t < .25 and not blocked
                   and source is not None and source[0] in ('python', 'cpp')
                   and source == current_source() and not gb.selection_reset
                   and captured_ns > getattr(gb, 'observation_ns', 0))
        # Never offer frozen, replayed, or switch-pending rows for a new click.
        gb.people = [] if blocked or source is None or not stamp_ok or (marked and not healthy) else list(people[:60])
        gb.people_t = frame_t if stamp_ok and (not marked or healthy) else 0.0
        _record_click_owners(gb, (click_owners or {}) if healthy else {}, click_namespace)
        if healthy:
            gb.observation_ns = captured_ns
        gb.frame += 1
        if marked:
            gb.observation_ok = bool(healthy)
            field = 'observation_good' if healthy else 'observation_bad'
            setattr(gb, field, getattr(gb, field, 0) + 1)
            if not healthy:
                # A later good frame may replace wframe. This separate one-slot STOP
                # barrier must reach the bridge first, without clearing the selected ID.
                gb.observation_barrier = f'F {gb.frame} {captured_ns} 0 OBS 0\n'.encode()
        selected = 0 if gb.selection_reset else int(gb.state.get('id', 0))
        _handoff_locked(gb, selected, frame_t)   # 11-13: LWIR_FOLLOW_HANDOFF=1 only
        _recatch_locked(gb, selected, frame_t)   # 11-15: bridge --search 1 and LWIR_SEARCH_NEW_ID=1 only
        hidden, pred = _guard_locked(gb, selected, frame_t, correction)   # 11-15: bridge --coast 1 or LWIR_FOLLOW_GUARD=1 only
        width, height = APP.LIVE_W, APP.LIVE_H
        real, seen = native_control.prepare(gb.people, hidden, correction, selected, float(width), float(height))
        aim_y = _bridge_aim_y(gb)
        aim_screen = _bridge_aim_screen_y(gb) * APP.LIVE_H
        tilt_hold = False
        if APP.CTL['lead'] and selected and gb.state.get('auto') and aim_y is not None:
            tilt = getattr(gb, 'reentry_tilt', None)
            if tilt is None or tilt.selected != selected:
                tilt = gb.reentry_tilt = native_control.reentry(selected)
            raw_correction = (0., 0.) if correction is None else correction
            tilt_hold = tilt.step(frame_t, seen, raw_correction, _motion)
        else:
            gb.reentry_tilt = None
        gb.state['reentry_tilt_hold'] = tilt_hold
        gb.state['control_backend'] = 'cpp' if native_control.native is not None else 'python'
        text, observed = native_control.observations(real, selected, tilt_hold, aim_y, aim_screen)
        lead = ''
        epoch = getattr(gb, 'control_epoch', None)
        if APP.CTL['lead'] and gb.state.get('auto'):   # 11-13: counters for /rules lead_stats (why prediction did or did not reach the bridge)
            why_ = ('bridge has no LEAD support' if not getattr(gb, 'lead_control_supported', False) else 'no control epoch yet' if epoch is None
                    else 'selection pending' if getattr(gb, 'lead_pending', 0) else 'lost or error' if gb.state.get('lost') or gb.state.get('error') else '')
            if why_: st_ = APP.CTL.setdefault('lead_stats', {}); st_[why_] = st_.get(why_, 0) + 1
        if (APP.CTL['lead'] and getattr(gb, 'lead_control_supported', False) and epoch is not None
                and not getattr(gb, 'lead_pending', 0) and gb.state.get('auto')
                and not gb.state.get('lost') and not gb.state.get('error')):
            # Use the rounded actual observation before the tilt limit.
            previous = APP.CTL['lv']
            sample, (dx, dy) = predictive_lead.estimate(
                previous, selected, observed, frame_t, APP.CTL.get('lead_motion'), APP.CTL['lead_ms'],
                aim_y=aim_y, correction=correction)
            if tilt_hold:
                dy = 0.
            # A settings request can reset the sample while NumPy is running.
            st_ = APP.CTL.setdefault('lead_stats', {}); w_ = predictive_lead.LAST.get('why') or '?'
            if w_ == 'sent' and not math.hypot(dx, dy): w_ = 'zero'
            st_[w_] = st_.get(w_, 0) + 1
            if APP.CTL['lead'] and APP.CTL['lv'] is previous:
                APP.CTL['lv'] = sample
                APP.CTL['lead_px'] = math.hypot(dx, dy)
                if APP.CTL['lead_px']:
                    lead = f' LEAD {selected} {dx!r} {dy!r} {epoch}'
        else:
            reset_lead()
        prediction = ''
        if pred is not None:   # Where to coast while the followed person is missing.
            cx, cy, w, h = pred
            q = stab_people_real([(selected, cx - w / 2., cy - h / 2., cx + w / 2., cy + h / 2., 1.0)], correction,
                                 float(APP.LIVE_W), float(APP.LIVE_H))[0]
            if all(map(math.isfinite, q[1:5])):
                view = gb.fguard.upper_coast_view
                if view is not None:
                    aim = _bridge_aim_y(gb)
                    cx, cy, width, height = view.limit(frame_t, follow_guard.cwh(q), _motion,
                                                       float(APP.LIVE_H), .5 if aim is None else aim,
                                                       _bridge_aim_screen_y(gb))
                    q = (selected, cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
                prediction = native_control.target_suffix('PRED', selected, q[1:5], False, tilt_hold, aim_y, aim_screen)
        seek = getattr(gb, 'seek_out', None)
        bounded = '' if gb.seek_raw is None else native_control.target_suffix(
            'SEEK', selected, gb.seek_raw, True, tilt_hold, aim_y, aim_screen)
        sweep = ''
        if seek is not None:   # 11-16: where to look while the person stays lost, for the bridge's --seek 1 (like PRED)
            cx, cy, w, h = seek
            q = stab_people_real([(selected, cx - w / 2., cy - h / 2., cx + w / 2., cy + h / 2., 1.0)], correction,
                                 float(APP.LIVE_W), float(APP.LIVE_H))[0]
            sweep = native_control.target_suffix('SEEK', selected, q[1:5], False, tilt_hold, aim_y, aim_screen)
        vel = _search_velocity_locked(gb, selected, observed, frame_t, correction, tilt_hold)
        picture = (gb.gui and getattr(gb, 'gui_picture', None) is not False
                   and raw16 is not None and raw16.shape == (height, width))
        # LEAD stays last because STOP and selection can strip it from a queued frame.
        header = native_control.header(gb.frame, captured_ns, len(real), bool(picture), width, height,
                                       text, marked, healthy, (prediction, bounded, sweep, vel, lead))
        APP.gb_send_frame(header, np.ascontiguousarray(raw16, dtype=np.uint16).tobytes() if picture else b'')
        return selected


def hit_id(people, x, y):
    inside = (p for p in people if p[1] <= x <= p[3] and p[2] <= y <= p[4])
    hit = min(inside, key=lambda p: (p[3] - p[1]) * (p[4] - p[2]), default=None)
    return hit[0] if hit is not None else None


AMBIGUOUS_CLICK = object()


def unique_hit_id(people, x, y):
    """Reject intersecting people and any second box with the target ID."""
    ids = [p[0] for p in people if p[1] <= x <= p[3] and p[2] <= y <= p[4]]
    if not ids:
        return None
    if len(ids) != 1 or sum(p[0] == ids[0] for p in people) != 1:
        return AMBIGUOUS_CLICK
    return ids[0]


def preview_people(frame_t, visible_ids):
    """Snapshot click boxes and the acknowledged tracker boundary for one JPEG."""
    gb = APP.GB
    with gb.wcond:
        if (not gb.wopen or getattr(gb, 'selection_reset', True)
                or getattr(gb, 'people_t', None) != frame_t):
            return None, None
        people = tuple(tuple(p[:5]) for p in gb.people if p[0] in visible_ids)
        return people, getattr(gb, 'tracks_ticket', 0)


def preview_click_owners(frame_t):
    """Selection ownership tokens from the frame used by the browser."""
    gb = APP.GB
    with gb.wcond:
        if getattr(gb, 'people_t', None) != frame_t or getattr(gb, 'selection_reset', True):
            return {}
        return {public: token for public, (_raw, token) in getattr(gb, 'click_owners', {}).items()}


_UNGUARDED_CLICK = object()


def click(x, y, *, expected_id=_UNGUARDED_CLICK, expected_ticket=None,
          shown_frame_t=None, expected_owner=None, viewer_ip=None):
    """Select from the published live snapshot, then send its frame BEFORE A."""
    if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= APP.LIVE_W or not 0 <= y <= APP.LIVE_H:
        return False, '영상 밖의 좌표입니다'
    gb = APP.GB
    with gb.wcond:
        if not getattr(gb, 'track_control_supported', False):
            return False, '짐벌 연결 프로그램을 새로 빌드하고 서버를 다시 시작해 주세요'
        if getattr(APP.LIVE, 'restart_pending', False):
            return False, '서버가 다시 시작되는 중입니다'
        source = current_source()
        if (not gb.wopen or source is None or getattr(gb, 'selection_reset', True)
                or source != getattr(gb, 'people_source', None)
                or not 0 <= time.perf_counter() - getattr(gb, 'people_t', 0.0) < 0.5):
            return False, '현재 카메라 추적 결과를 기다려 주세요'
        if shown_frame_t is not None:
            live = APP.LIVE
            if (not 0 <= time.perf_counter() - shown_frame_t < 0.5
                    or replay_link.frame_blocked(shown_frame_t)
                    or getattr(gb, 'usb_hold', False)
                    or any(getattr(live, flag, False) for flag in
                           ('stopped', 'cam_hold', 'cam_lost', 'restart_pending'))):
                return False, '선택할 수 있는 최신 카메라 화면을 기다려 주세요'
            target = unique_hit_id(gb.people, x, y)
        else:
            target = hit_id(gb.people, x, y)
        if target is AMBIGUOUS_CLICK:
            return False, '상자가 겹치거나 같은 추적 ID가 여러 곳에 있습니다. 최신 화면에서 다시 누르세요'
        if expected_id is not _UNGUARDED_CLICK and (
                target != expected_id or expected_ticket != getattr(gb, 'tracks_ticket', 0)):
            return False, '화면의 사람과 최신 추적 결과가 다릅니다. 새 화면에서 다시 누르세요'
        if shown_frame_t is not None and target is not None:
            owner = getattr(gb, 'click_owners', {}).get(target)
            if owner is None or owner[1] is not expected_owner:
                return False, '같은 추적 ID가 다른 사람에 연결되었습니다. 최신 화면에서 다시 누르세요'
        if target is not None and len(gb.wq) >= (61 if viewer_ip is not None else 62):
            return False, '명령 전송이 밀렸습니다. 잠시 뒤 다시 눌러 주세요'
        if viewer_ip is not None:
            # Accept the viewer and enqueue its heartbeat only with a validated H.264 click.
            gb.mover_t[viewer_ip] = time.monotonic()
            gb.mover = None
            if len(gb.wq) < 64:
                gb.wq.append(b'P\n')
        if target is None:
            # Empty-picture clicks retain Python's stop behavior.
            # Reuse the same acknowledged boundary mechanism; repeated empty
            # clicks cannot pile up STOP commands or retain queued selections.
            stop_with_ack_locked(gb)
        else:
            reset_lead()
            gb.lead_pending = getattr(gb, 'lead_pending', 0) + 1
            # The writer normally flushes commands before the latest G/F slot.
            # Send its observed rows before selecting, without carrying an old lead into the selection.
            if gb.wframe is not None:
                gb.wq.append(without_lead(gb.wframe))
                gb.wframe = None
            gb.wq.append(('A %d\n' % target).encode())
        gb.wcond.notify()
        return True, ''
