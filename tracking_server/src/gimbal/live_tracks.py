"""One live-track snapshot for the web click and existing PTZ transport.

The tracker computes boxes/IDs. This adapter only shares those results; it
does not implement a second tracker or change the camera control law.
"""
import math
import time

import numpy as np

from ..common.state import APP
from ..track.stabilizer import stab_people_real


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
    gb.tracks_ticket = getattr(gb, 'tracks_ticket', 0) + 1
    gb.wq.appendleft(('H %d\n' % gb.tracks_ticket).encode())
    gb.people = []
    gb.people_source = source
    gb.people_t = 0.0
    gb.selection_reset = True
    APP.CTL['lv'] = None
    APP.CTL['lead_px'] = 0.0
    gb.wcond.notify()


def halted(ticket):
    """Bridge has applied this boundary's STOP, not merely queued it."""
    gb = APP.GB
    with gb.wcond:
        if ticket == getattr(gb, 'tracks_ticket', None):
            gb.state.update(auto=0, lost=0, moving=0, id=0)
            gb.selection_reset = False


def publish(people, correction, frame_t, raw16, source, blocked=False):
    """Publish the same IDs as the drawn boxes, with existing coordinate rules."""
    gb = APP.GB
    if gb is None:
        return 0
    with gb.wcond:
        if not gb.wopen:
            gb.people = []
            return 0
        gap = frame_t - getattr(gb, 'people_t', 0.0)
        if getattr(gb, 'people_t', 0.0) and not 0 < gap <= 1.0:
            gb.people_source = None
        _boundary_locked(gb, source)
        # Never offer frozen, replayed, or switch-pending rows for a new click.
        gb.people = [] if blocked or source is None else list(people[:60])
        gb.people_t = frame_t
        gb.frame += 1
        selected = 0 if gb.selection_reset else int(gb.state.get('id', 0))
        real = stab_people_real(gb.people, correction, float(APP.LIVE_W), float(APP.LIVE_H))
        if APP.CTL['lead']:
            real = APP.lead_shift(real, gb.people, selected, frame_t,
                                  float(APP.LIVE_W), float(APP.LIVE_H))
        elif APP.CTL['lv'] is not None or APP.CTL['lead_px']:
            APP.CTL['lv'] = None
            APP.CTL['lead_px'] = 0.0
        text = ''.join(' %d %.1f %.1f %.1f %.1f %.3f' %
                       (p[0], p[1], p[2], p[3]-p[1], p[4]-p[2], p[5]) for p in real)
        if gb.gui and raw16 is not None and raw16.shape == (APP.LIVE_H, APP.LIVE_W):
            APP.gb_send_frame('G %d %d %d %d %d' %
                              (gb.frame, int(frame_t*1e9), APP.LIVE_W, APP.LIVE_H, len(real)) + text,
                              np.ascontiguousarray(raw16, dtype=np.uint16).tobytes())
        else:
            # Use the latest-frame slot for F too. A frame backlog must not
            # delay a new STOP or selection behind dozens of old positions.
            APP.gb_send_frame('F %d %d %d' % (gb.frame, int(frame_t*1e9), len(real)) + text, b'')
        return selected


def click(x, y):
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
        inside = [p for p in gb.people if p[1] <= x <= p[3] and p[2] <= y <= p[4]]
        if not inside:
            # Empty-picture clicks retain Python's stop behavior.
            # Reuse the same acknowledged boundary mechanism; repeated empty
            # clicks cannot pile up STOP commands or retain queued selections.
            gb.people_source = object()
            _boundary_locked(gb, source)
        else:
            if len(gb.wq) >= 62:
                return False, '명령 전송이 밀렸습니다. 잠시 뒤 다시 눌러 주세요'
            target = min(inside, key=lambda p: (p[3]-p[1])*(p[4]-p[2]))[0]
            # The writer normally flushes commands before the latest G/F slot.
            # Move that already-published frame into the FIFO before selection.
            if gb.wframe is not None:
                gb.wq.append(gb.wframe)
                gb.wframe = None
            gb.wq.append(('A %d\n' % target).encode())
        gb.wcond.notify()
        return True, ''
