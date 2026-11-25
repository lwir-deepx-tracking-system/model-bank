"""Bounded image-space feedforward using only consecutive measured frames."""

from dataclasses import dataclass
import math

import numpy as np

from . import native_control


MIN_INTERVAL = 0.010
MAX_INTERVAL = 0.250
TIME_CONSTANT = 0.120
NEAR_PAN_TIME_CONSTANT = 0.055
NEAR_HEIGHT_FRACTION = 0.55
NEAR_PAN_MIN_SPEED = 300.0
NEAR_PAN_START_MAX_INTERVAL = 0.075
NEAR_PAN_START_MAX_WIDTH_FRACTION = 0.15
NEAR_PAN_LEAD_GAIN = 1.5
MIN_SPEED = 15.0
MAX_LEAD = 60.0
CANVAS_Y_OFFSET = 16.0
NO_LEAD = (0.0, 0.0)
LAST = {'why': ''}   # 11-13: why the last estimate() gave (or did not give) a lead; counted in /rules lead_stats


@dataclass(frozen=True)
class Sample:
    track_id: int
    time: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    aim_y: float = 0.5
    correction: tuple = (0.0, 0.0)
    height: float = 0.0
    step_vx: float = 0.0


def estimate(previous, track_id, observed, frame_time, motion_pair, lead_ms, *,
             space="scene", aim_y=0.5, correction=None, picture_height=None):
    """Return the next sample and a control offset, without changing any box.

    ``observed`` uses raw camera coordinates; ``correction`` shifts raw to the
    stabilized picture. GMC maps padded, stabilized canvas coordinates between
    the two frames whose timestamps are in ``motion_pair``.
    """
    if space not in ("scene", "image"):
        raise ValueError("lead space must be scene or image")
    if (track_id <= 0 or observed is None or not math.isfinite(frame_time)
            or aim_y is None or not 0.0 <= aim_y <= 1.0):
        LAST['why'] = 'no box'
        return None, NO_LEAD
    x1, y1, x2, y2 = observed[1:5]
    if not x1 < x2 or not y1 < y2:
        LAST['why'] = 'bad box'
        return None, NO_LEAD
    x = x1 + (x2 - x1) / 2.0
    y = y1 + aim_y * (y2 - y1)
    cx, cy = (0.0, 0.0) if correction is None else correction
    if not all(map(math.isfinite, (x, y, cx, cy))):
        LAST['why'] = 'bad numbers'
        return None, NO_LEAD
    height = y2 - y1
    seed = Sample(track_id, frame_time, x, y, aim_y=aim_y, correction=(cx, cy), height=height)
    if previous is None or previous.track_id != track_id or previous.aim_y != aim_y:
        LAST['why'] = 'first frame of this id'
        return seed, NO_LEAD

    if (not previous.time + MIN_INTERVAL <= frame_time < previous.time + MAX_INTERVAL
            or motion_pair is None):
        LAST['why'] = 'frame spacing or no camera motion'
        return seed, NO_LEAD
    source_time, destination_time, warp = motion_pair
    if source_time != previous.time or destination_time != frame_time:
        LAST['why'] = 'camera motion of other frames'
        return seed, NO_LEAD
    warp = np.asarray(warp, dtype=float)
    if warp.shape != (2, 3) or not np.isfinite(warp).all():
        LAST['why'] = 'bad camera motion'
        return seed, NO_LEAD

    dt = frame_time - previous.time
    if space == "scene":
        # Express the previous raw anchor on its stabilized canvas, then
        # map the result back to the current raw frame before differencing.
        prev_cx, prev_cy = previous.correction
        px, py = map(float, warp @ (previous.x + prev_cx,
                                    previous.y + prev_cy + CANVAS_Y_OFFSET, 1.0))
        px -= cx
        py -= cy + CANVAS_Y_OFFSET
    else:
        # Experimental delay compensation: predict where the observation moves
        # in the raw image, including camera motion. Still require valid GMC.
        px, py = previous.x, previous.y
    dx, dy = x - px, y - py
    step_vx = dx / dt
    stable_close = (picture_height is not None and picture_height > 0
                    and min(previous.height, height) >= NEAR_HEIGHT_FRACTION * picture_height
                    and abs(height - previous.height) <= 0.20 * max(height, previous.height))
    fast_pan = False
    if stable_close and abs(step_vx) >= NEAR_PAN_MIN_SPEED:
        continuing = abs(previous.step_vx) >= NEAR_PAN_MIN_SPEED and previous.step_vx * step_vx > 0
        # Use the first plausible lateral step after slow movement. Leave a
        # gap, a large isolated jump and a reversal on the normal filter.
        starting = (space == "scene" and abs(previous.step_vx) < NEAR_PAN_MIN_SPEED
                    and previous.vx * step_vx >= 0
                    and dt <= NEAR_PAN_START_MAX_INTERVAL
                    and abs(dx) <= NEAR_PAN_START_MAX_WIDTH_FRACTION * (x2 - x1)
                    and abs(dx) >= abs(dy))
        fast_pan = continuing or starting
    if native_control.native is not None and not fast_pan:
        state, vx, vy, dx, dy = native_control.native.filter_velocity(
            dt, previous.vx, previous.vy, dx, dy, lead_ms)
        LAST['why'] = ('bad speed', 'slower than 15 px/s', 'sent')[state]
        sample = Sample(track_id, frame_time, x, y, vx, vy, aim_y, (cx, cy), height, step_vx) if state else seed
        return sample, (dx, dy)
    alpha = -math.expm1(-dt / TIME_CONSTANT)
    alpha_x = -math.expm1(-dt / NEAR_PAN_TIME_CONSTANT) if fast_pan else alpha
    vx = (1.0 - alpha_x) * previous.vx + alpha_x * step_vx
    vy = (1.0 - alpha) * previous.vy + alpha * dy / dt
    speed = math.hypot(vx, vy)
    if not math.isfinite(speed):
        LAST['why'] = 'bad speed'
        return seed, NO_LEAD
    sample = Sample(track_id, frame_time, x, y, vx, vy, aim_y, (cx, cy), height, step_vx)
    if speed < MIN_SPEED:
        LAST['why'] = 'slower than 15 px/s'
        return sample, NO_LEAD
    seconds = max(0.0, min(400.0, lead_ms)) / 1000.0
    if fast_pan:
        # Spend more of the existing lead budget in the direction of the
        # fast pan. Keep the ordinary vertical offset and protocol limit.
        scale = min(seconds, MAX_LEAD / speed)
        dy = vy * scale
        x_limit = math.sqrt(max(0.0, MAX_LEAD * MAX_LEAD - dy * dy))
        dx = vx * min(0.4, NEAR_PAN_LEAD_GAIN * seconds)
        LAST['why'] = 'sent'
        return sample, (max(-x_limit, min(x_limit, dx)), dy)
    scale = min(seconds, MAX_LEAD / speed)
    LAST['why'] = 'sent'
    return sample, (vx * scale, vy * scale)
