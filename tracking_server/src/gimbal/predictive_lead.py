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


def estimate(previous, track_id, observed, frame_time, motion_pair, lead_ms):
    """Return the next sample and a control offset, without changing any box.

    ``observed`` uses picture coordinates. The GMC affine uses the padded
    detector canvas. ``motion_pair`` carries its exact previous/current times.
    """
    if track_id <= 0 or observed is None or not math.isfinite(frame_time):
        return None, NO_LEAD
    x = (observed[1] + observed[3]) / 2.0
    y = (observed[2] + observed[4]) / 2.0
    if not math.isfinite(x) or not math.isfinite(y):
        return None, NO_LEAD
    seed = Sample(track_id, frame_time, x, y)
    if previous is None or previous.track_id != track_id:
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
    px, py = map(float, warp @ (previous.x, previous.y + CANVAS_Y_OFFSET, 1.0))
    alpha = -math.expm1(-dt / TIME_CONSTANT)
    vx = (1.0 - alpha) * previous.vx + alpha * (x - px) / dt
    vy = (1.0 - alpha) * previous.vy + alpha * (y + CANVAS_Y_OFFSET - py) / dt
    speed = math.hypot(vx, vy)
    if not math.isfinite(speed):
        return seed, NO_LEAD
    sample = Sample(track_id, frame_time, x, y, vx, vy)
    if speed < MIN_SPEED:
        return sample, NO_LEAD
    seconds = max(0.0, min(400.0, lead_ms)) / 1000.0
    scale = min(seconds, MAX_LEAD / speed)
    return sample, (vx * scale, vy * scale)
