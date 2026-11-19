"""Optional native calculations for the gimbal observation protocol."""
import math
import os

from ..track.stabilizer import stab_people_real

try:
    from . import _native_control as native
except ImportError:
    native = None

if os.environ.get('LWIR_NATIVE_CONTROL', '1') == '0':
    native = None


def prepare(people, hidden, correction, selected, width, height):
    if native is not None:
        return native.prepare(people, hidden, correction, selected, width, height)
    real = stab_people_real([person for person in people if not hidden or int(person[0]) != hidden],
                            correction, width, height)
    targets = [p for p in real if p[0] == selected]
    if len(targets) != 1:
        return real, None
    _, x1, y1, x2, y2, *_ = targets[0]
    return real, ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def observations(real, selected, tilt_hold, aim, screen):
    if native is not None:
        return native.observations(real, selected, tilt_hold, .5 if aim is None else aim, screen)
    from .reentry_tilt import limit_downward

    rows = []
    matches = []
    for person in real:
        oid, x1, y1, x2, y2, score = person[:6]
        if oid == selected:
            x, y, width, height = map(float, ('%.1f %.1f %.1f %.1f' % (x1, y1, x2 - x1, y2 - y1)).split())
            matches.append((selected, x, y, x + width, y + height))
            if tilt_hold:
                x1, y1, x2, y2 = limit_downward((x1, y1, x2, y2), aim, screen)
        rows.append(' %d %.1f %.1f %.1f %.1f %.3f' % (oid, x1, y1, x2 - x1, y2 - y1, score))
    return ''.join(rows), matches[0] if len(matches) == 1 else None


def target_suffix(kind, selected, coords, xywh, tilt_hold, aim, screen):
    if native is not None:
        return native.target_suffix(kind, selected, coords, xywh, tilt_hold, .5 if aim is None else aim, screen)
    from .reentry_tilt import limit_downward

    if not all(map(math.isfinite, coords)):
        return ''
    x, y, a, b = coords
    if xywh:
        if tilt_hold and y + aim * b > screen + .05:
            return ''
    else:
        if tilt_hold:
            x, y, a, b = limit_downward(coords, aim, screen)
        a, b = a - x, b - y
    return ' %s %d %.1f %.1f %.1f %.1f' % (kind, selected, x, y, a, b)


def velocity(selected, vx, vy, tilt_hold):
    if native is not None:
        return native.velocity(selected, vx, vy, tilt_hold)
    if not (math.isfinite(vx) and math.isfinite(vy)):
        return ''
    vy = min(0., vy) if tilt_hold else vy
    return ' VEL %d %.1f %.1f' % (selected, vx, vy) if vx or vy else ''


def header(frame, captured_ns, count, picture, width, height, text, marked, healthy, suffixes):
    if native is not None:
        return native.header(frame, captured_ns, count, picture, int(width), int(height), text,
                             int(healthy) if marked else -1, suffixes)
    prefix = ('G %d %d %d %d %d' % (frame, captured_ns, width, height, count)
              if picture else 'F %d %d %d' % (frame, captured_ns, count))
    observation = ' OBS %d' % healthy if marked else ''
    return prefix + text + observation + ''.join(suffixes)


class NativeReentryTilt:
    def __init__(self, selected):
        self.selected = selected
        self.handle = native.new_reentry()

    def step(self, time, observed, correction, motion):
        return native.reentry_step(self.handle, time, observed, correction, motion)


def reentry(selected):
    if native is not None:
        return NativeReentryTilt(selected)
    from .reentry_tilt import ReentryTilt

    return ReentryTilt(selected)
