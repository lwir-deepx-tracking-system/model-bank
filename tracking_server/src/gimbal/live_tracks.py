"""One live-track snapshot for the web click and existing PTZ transport.

The tracker computes boxes/IDs. This adapter only shares those results; it
does not implement a second tracker or change the camera control law.
"""
import math
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


def _bridge_aim_y(gb):
    # v27 advertises LEAD only after parsing its launch options. Use that
    # process's actual argv, not the web display or mutable server settings.
    argv = getattr(getattr(gb, 'proc', None), 'args', None)
    if not isinstance(argv, (list, tuple)) or len(argv) % 2 != 1:
        return None
    value = None
    for flag, arg in zip(argv[1::2], argv[2::2]):
        if flag == '--aim-y':
            try:
                value = float(arg)
            except (TypeError, ValueError):
                return None
    return value if value is not None and 0.0 <= value <= 1.0 else None


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
        real = stab_people_real(gb.people, correction, float(APP.LIVE_W), float(APP.LIVE_H))
        rows = [' %d %.1f %.1f %.1f %.1f %.3f' %
                (p[0], p[1], p[2], p[3]-p[1], p[4]-p[2], p[5]) for p in real]
        text = ''.join(rows)
        if marked:
            text += ' OBS %d' % bool(healthy)  # Before LEAD: without_lead must preserve this.
        epoch = getattr(gb, 'control_epoch', None)
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
            if APP.CTL['lead'] and APP.CTL['lv'] is previous:
                APP.CTL['lv'] = sample
                APP.CTL['lead_px'] = math.hypot(dx, dy)
                if APP.CTL['lead_px']:
                    text += f' LEAD {selected} {dx!r} {dy!r} {epoch}'
        else:
            reset_lead()
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
