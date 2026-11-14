"""One live-track snapshot for the web click and existing PTZ transport.

The tracker computes boxes/IDs. This adapter only shares those results; it
does not implement a second tracker or change the camera control law.
"""
import math
import os
import time

import numpy as np

from ..common.state import APP
from ..track.stabilizer import stab_people_real
from . import predictive_lead
from .gimbal_link import reset_lead, without_lead


def current_source():
    native = getattr(APP.LIVE, 'camera_bytetrack', None)
    source = native.control_source() if native is not None else ('python',)
    if source == ('python',):
        # Keep the actual object alive until the next publication, so an id()
        # reused by Python cannot make a new tracker look like the old one.
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
    gb.people_source = source
    gb.people_t = 0.0
    gb.selection_reset = True
    gb.lead_pending = 0
    reset_lead()
    gb.wcond.notify()


def halted(ticket):
    """Bridge has applied this boundary's STOP, not merely queued it."""
    gb = APP.GB
    with gb.wcond:
        if ticket == getattr(gb, 'tracks_ticket', None):
            gb.state.update(auto=0, lost=0, moving=0, id=0)
            gb.selection_reset = False


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
    if os.environ.get('LWIR_FOLLOW_HANDOFF') != '1' or search_enabled(gb):   # 11-15: a searching gimbal moves the picture; _recatch_locked replaces this rule
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
    hist = list(APP.CTL.get('motion_hist') or ())
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


def _search_velocity_locked(gb, selected, real, rows, frame_t, correction):
    """11-15: ' VEL <id> <vx> <vy>' for a bridge that said SEARCH_CONTROL 1: the followed person's velocity against the
    background (picture px/s) from the predictive-control estimator (predictive_lead.estimate, the same one LEAD uses; when
    LEAD is on its own sample of this frame is reused). The bridge uses it only to keep turning where the person should be
    after a loss, so the tracker / KeepID can give the same id back. '' when there is nothing to send."""
    if not getattr(gb, 'search_control_supported', False) or not selected or not gb.state.get('auto'):
        gb.search_lv = None
        return ''
    lv = APP.CTL.get('lv')
    if not (APP.CTL.get('lead') and lv is not None and getattr(lv, 'track_id', 0) == selected and getattr(lv, 'time', None) == frame_t):
        matches = [row for person, row in zip(real, rows) if person[0] == selected]
        observed = None
        if len(matches) == 1:
            x, y, w, h = map(float, matches[0].split()[1:5])
            observed = (selected, x, y, x + w, y + h)
        why = predictive_lead.LAST.get('why')
        lv, _ = predictive_lead.estimate(getattr(gb, 'search_lv', None), selected, observed, frame_t,
                                         APP.CTL.get('lead_motion'), 0, aim_y=_bridge_aim_y(gb) if _bridge_aim_y(gb) is not None else .5,
                                         correction=correction)
        predictive_lead.LAST['why'] = why   # the lead counters stay those of the lead path
        gb.search_lv = lv
    if lv is None or lv.time != frame_t or not (math.isfinite(lv.vx) and math.isfinite(lv.vy)) or (lv.vx == 0 and lv.vy == 0):
        return ''
    return ' VEL %d %.1f %.1f' % (selected, lv.vx, lv.vy)


def publish(people, correction, frame_t, raw16, source, blocked=False, *, healthy_live=False):
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
        real = stab_people_real(gb.people, correction, float(APP.LIVE_W), float(APP.LIVE_H))
        rows = [' %d %.1f %.1f %.1f %.1f %.3f' %
                (p[0], p[1], p[2], p[3]-p[1], p[4]-p[2], p[5]) for p in real]
        text = ''.join(rows)
        if marked:
            text += ' OBS %d' % bool(healthy)  # Before LEAD: without_lead must preserve this.
        epoch = getattr(gb, 'control_epoch', None)
        if APP.CTL['lead'] and gb.state.get('auto'):   # 11-13: counters for /rules lead_stats (why prediction did or did not reach the bridge)
            why_ = ('bridge has no LEAD support' if not getattr(gb, 'lead_control_supported', False) else 'no control epoch yet' if epoch is None
                    else 'selection pending' if getattr(gb, 'lead_pending', 0) else 'lost or error' if gb.state.get('lost') or gb.state.get('error') else '')
            if why_: st_ = APP.CTL.setdefault('lead_stats', {}); st_[why_] = st_.get(why_, 0) + 1
        if (APP.CTL['lead'] and getattr(gb, 'lead_control_supported', False) and epoch is not None
                and not getattr(gb, 'lead_pending', 0) and gb.state.get('auto')
                and not gb.state.get('lost') and not gb.state.get('error')):
            matches = [row for person, row in zip(real, rows) if person[0] == selected]
            observed = None
            if len(matches) == 1:
                # Use the clipped, rounded observation that the controller reads.
                x, y, w, h = map(float, matches[0].split()[1:5])
                observed = (selected, x, y, x + w, y + h)
            previous = APP.CTL['lv']
            sample, (dx, dy) = predictive_lead.estimate(
                previous, selected, observed, frame_t, APP.CTL.get('lead_motion'), APP.CTL['lead_ms'],
                aim_y=_bridge_aim_y(gb), correction=correction)
            # A settings request can reset the sample while NumPy is running.
            st_ = APP.CTL.setdefault('lead_stats', {}); w_ = predictive_lead.LAST.get('why') or '?'
            if w_ == 'sent' and not math.hypot(dx, dy): w_ = 'zero'
            st_[w_] = st_.get(w_, 0) + 1
            if APP.CTL['lead'] and APP.CTL['lv'] is previous:
                APP.CTL['lv'] = sample
                APP.CTL['lead_px'] = math.hypot(dx, dy)
                if APP.CTL['lead_px']:
                    text += f' LEAD {selected} {dx!r} {dy!r} {epoch}'
        else:
            reset_lead()
        vel = _search_velocity_locked(gb, selected, real, rows, frame_t, correction)
        if vel:   # before LEAD: without_lead() cuts from LEAD to the end of the line
            text = text.replace(' LEAD ', vel + ' LEAD ', 1) if ' LEAD ' in text else text + vel
        if gb.gui and raw16 is not None and raw16.shape == (APP.LIVE_H, APP.LIVE_W):
            APP.gb_send_frame('G %d %d %d %d %d' %
                              (gb.frame, captured_ns, APP.LIVE_W, APP.LIVE_H, len(real)) + text,
                              np.ascontiguousarray(raw16, dtype=np.uint16).tobytes())
        else:
            # Use the latest-frame slot for F too. A frame backlog must not
            # delay a new STOP or selection behind dozens of old positions.
            APP.gb_send_frame('F %d %d %d' % (gb.frame, captured_ns, len(real)) + text, b'')
        return selected


def hit_id(people, x, y):
    inside = (p for p in people if p[1] <= x <= p[3] and p[2] <= y <= p[4])
    hit = min(inside, key=lambda p: (p[3] - p[1]) * (p[4] - p[2]), default=None)
    return hit[0] if hit is not None else None


def preview_people(frame_t, visible_ids):
    """Snapshot click boxes and the acknowledged tracker boundary for one JPEG."""
    gb = APP.GB
    with gb.wcond:
        if (not gb.wopen or getattr(gb, 'selection_reset', True)
                or getattr(gb, 'people_t', None) != frame_t):
            return None, None
        people = tuple(tuple(p[:5]) for p in gb.people if p[0] in visible_ids)
        return people, getattr(gb, 'tracks_ticket', 0)


_UNGUARDED_CLICK = object()


def click(x, y, *, expected_id=_UNGUARDED_CLICK, expected_ticket=None):
    """Select from the published live snapshot, then send its frame BEFORE A."""
    if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= APP.LIVE_W or not 0 <= y <= APP.LIVE_H:
        return False, '영상 밖의 좌표입니다'
    gb = APP.GB
    with gb.wcond:
        if not getattr(gb, 'track_control_supported', False):
            return False, '짐벌 연결 프로그램을 새로 빌드하고 서버를 다시 시작해 주세요'
        source = current_source()
        if (not gb.wopen or source is None or getattr(gb, 'selection_reset', True)
                or source != getattr(gb, 'people_source', None)
                or not 0 <= time.perf_counter() - getattr(gb, 'people_t', 0.0) < 0.5):
            return False, '현재 카메라 추적 결과를 기다려 주세요'
        target = hit_id(gb.people, x, y)
        if expected_id is not _UNGUARDED_CLICK and (
                target != expected_id or expected_ticket != getattr(gb, 'tracks_ticket', 0)):
            return False, '화면의 사람과 최신 추적 결과가 다릅니다. 새 화면에서 다시 누르세요'
        if target is None:
            # Empty-picture clicks retain Python's stop behavior.
            # Reuse the same acknowledged boundary mechanism; repeated empty
            # clicks cannot pile up STOP commands or retain queued selections.
            gb.people_source = object()
            _boundary_locked(gb, source)
        else:
            if len(gb.wq) >= 62:
                return False, '명령 전송이 밀렸습니다. 잠시 뒤 다시 눌러 주세요'
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
