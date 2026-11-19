"""Inspect both sides of the last trusted prediction during one loss."""

import math

from .follow_guard import KIV_MARGIN, MAX_DT, warp_point

START_S = .82
FIRST_END_S = 1.48
SECOND_BEGIN_S = 1.62
SECOND_END_S = 2.50
RETURN_BEGIN_S = 2.64
END_S = 3.46
MAX_IMAGE_SHIFT = .025
RETURN_TOLERANCE = .01
AIM_OFFSET = .06  # the bounded bridge accepts .04 to .08, on one axis only


class BoundedSeek:
    def __init__(self):
        self.lost_at = self.last_t = None
        self.motion = self.projected = self.origin = self.first = None
        self.stopped = False

    def cancel(self):
        self.stopped = True

    def step(self, t, view, camera_motion, size, aim, aim_fraction, *, prediction=None):
        if self.stopped:
            return None
        width, height = size
        if self.lost_at is None:
            self.lost_at = t
            if view.anchor is None or not 0 < t - view.anchor[0] <= MAX_DT:
                self.cancel()
                return None
            anchor_t, (x, y, _, h) = view.anchor
            self.projected = ((x, y), (x, y - h / 2))
            self.last_t = anchor_t
            if view.direction_at is not None and 0 <= t - view.direction_at <= MAX_DT:
                self.motion = view.direction
        elapsed = t - self.lost_at
        if elapsed < 0 or elapsed >= END_S or t - self.last_t > MAX_DT:
            self.cancel()
            return None
        if t <= self.last_t:
            return None
        transform = camera_motion(self.last_t, t)
        if transform is None:
            self.cancel()
            return None
        centre, top_point = (warp_point(transform, *point) for point in self.projected)
        if not all(map(math.isfinite, (*centre, *top_point))):
            self.cancel()
            return None
        self.last_t = t
        if elapsed < START_S:
            self.projected = centre, top_point
            return None

        top = top_point[1]
        if self.first is None:
            if elapsed >= FIRST_END_S:
                self.cancel()
                return None
            # The guard's prediction is already in this camera frame. Advect it
            # with camera motion afterward, without using any untrusted boxes.
            if prediction is not None and all(map(math.isfinite, prediction)):
                px, py, _, ph = prediction
                if ph > 0 and math.hypot(px - centre[0], py - centre[1]) <= ph:
                    centre = px, py
            self.origin = centre
            self.first = self.motion
            if self.first is None:
                self.first = (1, -1) if top < KIV_MARGIN * height else (0, 1 if centre[0] >= aim[0] else -1)
        self.projected = centre, top_point
        axis, direction = self.first
        span = size[axis]
        shift = centre[axis] - self.origin[axis]
        if elapsed >= RETURN_BEGIN_S:
            if abs(shift) <= RETURN_TOLERANCE * span:
                self.cancel()
                return None
            direction = 1 if shift > 0 else -1
        elif SECOND_BEGIN_S <= elapsed < SECOND_END_S:
            direction = -direction
        elif elapsed >= FIRST_END_S:
            return None

        if axis == 1 and direction > 0 and self.motion != (1, 1):
            if top < (KIV_MARGIN + MAX_IMAGE_SHIFT) * height:
                return None
        if -direction * shift >= MAX_IMAGE_SHIFT * span:
            return None

        target = list(aim)
        target[axis] += direction * AIM_OFFSET * span
        box_size = 20.
        x, y = target[0] - box_size / 2, target[1] - aim_fraction * box_size
        if not (0 <= x <= width - box_size and 0 <= y <= height - box_size):
            return None
        return x, y, box_size, box_size
