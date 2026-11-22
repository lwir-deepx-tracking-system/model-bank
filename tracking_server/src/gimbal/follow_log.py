"""11-22: follow-loss recorder. Why does a follow end? (default on, LWIR_FOLLOW_LOG=0 = off)

Cheap: one small tuple per frame in a ring (about 5 s) while a target is followed; nothing is written until the bridge's state changes
(from following to idle / lost, or the followed id changes). Then ONE JSON line goes to follows.log (next to camera_events.log; 2 MB, then .1)
and one short line `FOLLOWLOSS ...` to camera_events.log. The last 20 events are kept in memory: GET /follows (full), /healthz follow_events (short).

Event = dict(t (wall time), up (server seconds), kind lost | ended | switched | server_start, id, new_id, reason, why (detail), n_missing (frames the id was
absent from the people rows before the event), cause (tag + age of the last server-side action), last3s (<= 8 samples: dt, present, box, score,
max overlap with another person, that person's id, distance of the box to the picture edge in px), near (other ids close to the last box, overlap),
guard (follow_guard phase, mask, retarget counters), judge (relink judge / feat_guard counters now), state (bridge auto, lost, error, search)).

Reason codes: user_released, user_clicked_other, user_manual_move, camera_stall, usb_reset, tracker_or_model_switch (selection boundary),
bridge_restart, bridge_error, id_switched / id_switched_relink (new id stands where the old box was), target_left_picture, covered_by_other_person,
detection_missing_<n>_frames (id absent < 15 frames), track_ended_lost_age (id absent >= 15 frames), bridge_dropped_target_visible (the id was in the
last frame: the bridge's own timeout / guard ended it), server_start.
"""
import collections
import json
import os
import time

RING = 150           # frames kept (about 5 s at 30 fps)
KEEP = 20            # events kept for /follows
MISS_LOST = 15       # id absent this many frames or more: the track ended (lost age)
EDGE_FRAC = 0.03
CAUSE_S = 2.0        # a server-side action this recent explains the end
NEAR_IOU = 0.05
COVER_IOU = 0.3
LOG_MAX = 2 * 1024 * 1024

_ring = collections.deque(maxlen=RING)
_events = collections.deque(maxlen=KEEP)
_cause = [None, 0.0]
_t0 = time.monotonic()
_path = [None]


def enabled():
    return os.environ.get('LWIR_FOLLOW_LOG', '1') != '0'


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    i = w * h; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def cause(tag):
    """a server-side action that ends or may end a follow (page stop / home / click / manual move, camera gap, USB reset, boundary, bridge restart)"""
    _cause[0], _cause[1] = tag, time.monotonic()


def frame(gb, t, selected, people, width=640, height=480):
    """publish(), with gb.wcond held, once per frame. Only while a target is followed."""
    try:
        if not enabled(): return
        if not selected or not gb.state.get('auto'):
            if _ring: _ring.clear()
            return
        mine = None; others = []
        for p in people:
            if int(p[0]) == selected: mine = p
            else: others.append(p)
        if mine is None:
            _ring.append((t, 0, None, None, 0.0, 0, None, selected))
            return
        box = tuple(float(v) for v in mine[1:5])
        score = float(mine[5]) if len(mine) > 5 and isinstance(mine[5], (int, float)) else None
        best, bid = 0.0, 0
        for p in others:
            o = _iou(box, tuple(float(v) for v in p[1:5]))
            if o > best: best, bid = o, int(p[0])
        edge = min(box[0], box[1], width - box[2], height - box[3])
        _ring.append((t, 1, box, score, best, bid, edge, selected))
    except Exception:
        pass


def _summary(now_t, last_people):
    rows = [r for r in _ring if now_t - r[0] <= 3.0]
    step = max(1, len(rows) // 8)
    out = []
    for r in rows[::step][-8:]:
        out.append(dict(dt=round(r[0] - now_t, 2), present=r[1], box=None if r[2] is None else [round(v) for v in r[2]],
                        score=None if r[3] is None else round(r[3], 2), overlap=round(r[4], 2), other=r[5], edge=None if r[6] is None else round(r[6])))
    return out


def _guard(gb):
    g = getattr(gb, 'fguard', None); out = dict(phase=gb.state.get('guard'))
    if g is not None:
        out['why'] = getattr(g, 'why', '')   # 11-22: why the guard masked / held (e.g. 'looks like a person near it', 'jump ...', 'occlusion hold ...')
        for k in ('reaccepts', 'swaps', 'app_swaps', 'app_masks', 'masked_n', 'holds'):
            v = getattr(g, k, None)
            if v is not None: out[k] = v
    return out


def _judge():
    try:
        from ..reid import relink_online_judge as rj
        out = dict(feat_guard=bool(rj.GUARD.get('feat_guard')))
        from ..common.state import APP
        tr = getattr(APP, 'tracker', None)
        st = getattr(tr, 'guard_stats', None) or getattr(getattr(tr, 'judge', None), 'guard_stats', None)
        if isinstance(st, dict): out.update({k: v for k, v in st.items() if isinstance(v, (int, float))})
        return out
    except Exception:
        return None


def _write(ev):
    try:
        if _path[0] is None:
            from ..common.state import APP
            _path[0] = os.path.join(os.path.dirname(os.path.abspath(str(APP.out))), 'follows.log')
        p = _path[0]
        try:
            if os.path.getsize(p) > LOG_MAX: os.replace(p, p + '.1')
        except OSError:
            pass
        with open(p, 'a') as f: f.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + '\n')
    except Exception:
        pass
    try:
        from ..camera import camera_extra
        camera_extra.note('server', 'FOLLOWLOSS %s id=%s%s reason=%s%s' % (ev['kind'], ev['id'], '' if not ev.get('new_id') else '->%s' % ev['new_id'], ev['reason'], ' (%s)' % ev['why'] if ev.get('why') else ''))
    except Exception:
        pass


def classify(kind, old, new, missing, last, people, cause_tag, error, width=640):
    """-> (reason, why)"""
    if cause_tag:
        return {'click': ('user_clicked_other', 'a click on a person or empty picture'), 'stop': ('user_released', 'stop'), 'home': ('user_released', 'home'),
                'manual_move': ('user_manual_move', 'a page moved the gimbal by hand')}.get(cause_tag, (cause_tag, ''))
    if error: return 'bridge_error', 'the serial link reported an error'
    if new and new != old:
        if last is not None and last[2] is not None:
            for p in people:
                if int(p[0]) == new and _iou(last[2], tuple(float(v) for v in p[1:5])) >= 0.3: return 'id_switched_relink', 'new id %s stands where %s was' % (new, old)
        return 'id_switched', 'to id %s' % new
    if missing >= 1 and last is not None and last[2] is not None:
        if last[6] is not None and last[6] <= EDGE_FRAC * width: return 'target_left_picture', 'last box %.0f px from the edge' % last[6]
        if last[4] >= COVER_IOU: return 'covered_by_other_person', 'overlap %.2f with id %s' % (last[4], last[5])
        if missing >= MISS_LOST: return 'track_ended_lost_age', 'absent %d frames' % missing
        return 'detection_missing_%d_frames' % missing, 'id absent from the tracker rows'
    if missing >= 1: return 'track_ended_lost_age', 'absent %d frames' % missing
    return 'bridge_dropped_target_visible', 'the id was in the last frame: the bridge timeout or the guard ended it'


def state(gb, old, new):
    """gb_reader on a bridge 'S' line, gb.wcond held. old / new = (auto, lost, error, id)."""
    try:
        if not enabled(): return
        oa, ol, oe, oid = old; na, nl, ne, nid = new
        if not oa: return
        if na and not nl and nid == oid: return
        if na and nl and not ol: kind = 'lost'
        elif not na: kind = 'ended'
        elif nid != oid: kind = 'switched'
        else: return
        now = time.monotonic()
        tag = _cause[0] if now - _cause[1] <= CAUSE_S else None
        last = _ring[-1] if _ring else None
        missing = 0
        for r in reversed(_ring):
            if r[1]: break
            missing += 1
        people = getattr(gb, 'people', None) or []
        reason, why = classify(kind, oid, nid if na else 0, missing, next((r for r in reversed(_ring) if r[1]), None), people, tag, ne)
        if tag in ('camera_gap', 'usb_reset', 'boundary', 'bridge_restart'): reason, why = {'camera_gap': ('camera_stall', 'input gap > 1 s: selection dropped'), 'usb_reset': ('usb_reset', 'camera reset'),
                                                                                           'boundary': ('tracker_or_model_switch', 'selection boundary'), 'bridge_restart': ('bridge_restart', 'gimbal bridge restarted')}[tag]
        ref = next((r for r in reversed(_ring) if r[1]), None)
        near = []
        if ref is not None:
            cx, cy = (ref[2][0] + ref[2][2]) / 2, (ref[2][1] + ref[2][3]) / 2; w = max(1.0, ref[2][2] - ref[2][0])
            for p in people:
                if int(p[0]) == oid: continue
                b = tuple(float(v) for v in p[1:5])
                if abs((b[0] + b[2]) / 2 - cx) <= 1.5 * w and abs((b[1] + b[3]) / 2 - cy) <= 1.5 * w * 1.5:
                    near.append([int(p[0]), round(_iou(ref[2], b), 2)])
        ev = dict(t=time.strftime('%F %T'), up=round(now - _t0, 1), kind=kind, id=oid, new_id=nid if na else 0, reason=reason, why=why, n_missing=missing,
                  cause=None if tag is None else dict(tag=tag, age=round(now - _cause[1], 2)), last3s=_summary(ref[0] if ref else 0.0, people), near=near[:6],
                  guard=_guard(gb), judge=_judge(), state=dict(auto=na, lost=nl, error=ne, search=gb.state.get('search', 0)))
        _events.append(ev)
        _write(ev)
        if not na: _ring.clear()
    except Exception:
        pass


def start():
    ev = dict(t=time.strftime('%F %T'), up=0.0, kind='server_start', id=0, new_id=0, reason='server_start', why='a follow before this line ended with the restart')
    _events.append(ev); _write(ev)


def events(n=KEEP):
    return list(_events)[-n:]


def health():
    ev = list(_events)[-5:]
    return [dict(t=e['t'], kind=e['kind'], id=e['id'], new_id=e.get('new_id'), reason=e['reason']) for e in ev]
