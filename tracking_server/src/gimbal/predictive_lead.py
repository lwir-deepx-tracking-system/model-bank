"""Bounded image-space feedforward using only consecutive measured frames."""

from dataclasses import dataclass
import math

import numpy as np


MIN_INTERVAL = 0.010
MAX_INTERVAL = 0.250
TIME_CONSTANT = 0.120
MIN_SPEED = 15.0
MAX_LEAD = 60.0
CANVAS_Y_OFFSET = 16.0
NO_LEAD = (0.0, 0.0)


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


def estimate(previous, track_id, observed, frame_time, motion_pair, lead_ms, *,
             space="scene", aim_y=0.5, correction=None):
    """Return the next sample and a control offset, without changing any box.

    ``observed`` uses raw camera coordinates; ``correction`` shifts raw to the
    stabilized picture. GMC maps padded, stabilized canvas coordinates between
    the two frames whose timestamps are in ``motion_pair``.
    """
    if space not in ("scene", "image"):
        raise ValueError("lead space must be scene or image")
    if (track_id <= 0 or observed is None or not math.isfinite(frame_time)
            or aim_y is None or not 0.0 <= aim_y <= 1.0):
        return None, NO_LEAD
    x1, y1, x2, y2 = observed[1:5]
    if not x1 < x2 or not y1 < y2:
        return None, NO_LEAD
    x = x1 + (x2 - x1) / 2.0
    y = y1 + aim_y * (y2 - y1)
    cx, cy = (0.0, 0.0) if correction is None else correction
    if not all(map(math.isfinite, (x, y, cx, cy))):
        return None, NO_LEAD
    seed = Sample(track_id, frame_time, x, y, aim_y=aim_y, correction=(cx, cy))
    if previous is None or previous.track_id != track_id or previous.aim_y != aim_y:
        return seed, NO_LEAD

    if (not previous.time + MIN_INTERVAL <= frame_time < previous.time + MAX_INTERVAL
            or motion_pair is None):
        return seed, NO_LEAD
    source_time, destination_time, warp = motion_pair
    if source_time != previous.time or destination_time != frame_time:
        return seed, NO_LEAD
    warp = np.asarray(warp, dtype=float)
    if warp.shape != (2, 3) or not np.isfinite(warp).all():
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
    alpha = -math.expm1(-dt / TIME_CONSTANT)
    vx = (1.0 - alpha) * previous.vx + alpha * (x - px) / dt
    vy = (1.0 - alpha) * previous.vy + alpha * (y - py) / dt
    speed = math.hypot(vx, vy)
    if not math.isfinite(speed):
        return seed, NO_LEAD
    sample = Sample(track_id, frame_time, x, y, vx, vy, aim_y, (cx, cy))
    if speed < MIN_SPEED:
        return sample, NO_LEAD
    seconds = max(0.0, min(400.0, lead_ms)) / 1000.0
    scale = min(seconds, MAX_LEAD / speed)
    return sample, (vx * scale, vy * scale)
