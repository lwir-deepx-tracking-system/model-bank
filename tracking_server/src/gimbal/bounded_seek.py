"""Inspect the area around the last trusted prediction during one loss."""

import math

from . import native_control
from .follow_guard import KIV_MARGIN, MAX_DT, warp_point, warp_scale

START_S = .82
FIRST_END_S = 1.48
SECOND_BEGIN_S = 1.62
SECOND_END_S = 2.50
RETURN_BEGIN_S = 2.64
END_S = 3.46
MAX_IMAGE_SHIFT = .025
RETURN_TOLERANCE = .01
AIM_OFFSET = .06  # the bounded bridge accepts .04 to .08, on one axis only
WIDE_PAN_BRIDGE_MS = 6500
TWO_AXIS_END_S = 6.40
BRAKE_MARGIN = .0125  # Reserve room for stopping at low speed on either axis.
LOCAL_RADII = (.052, .055)  # pan/tilt in fractions of the current image span
LOCAL_LEG_BUDGETS = (.72, 1.12, .63)
LOCAL_SETTLE_S = .11


class BoundedSeek:
    def __init__(self, wide_pan=False):
        self.wide_pan = wide_pan
        self.lost_at = self.last_t = None
        self.motion = self.projected = self.origin = self.first = None
        self.stopped = False
        self.half_width = self.half_height = 0.
        self.status = 'parked'
        self.local_legs = ()
        self.local_index = 0
        self.local_start = None
        self.local_ready = 0.
        self.local_direction = 0

    def cancel(self, reason='cancelled'):
        if not self.stopped:
            self.status = reason
        self.stopped = True

    def maintain(self, t, view, camera_motion):
        """Carry the clean anchor through ambiguous frames without moving the camera."""
        if self.stopped:
            return False
        if self.last_t is None:
            anchor = view.seek_anchor
            if anchor is None:
                self.status = 'no_trusted_view'
                return False
            anchor_t, (x, y, w, h), toward_edge = anchor
            if not 0 <= t - anchor_t <= MAX_DT:
                self.status = 'stale_trusted_view'
                return False
            self.half_width = w / 2
            self.half_height = h / 2
            self.projected = ((x, y), (x, y - h / 2))
            self.last_t = anchor_t
            if toward_edge is not None:
                self.motion = toward_edge
            elif view.direction_at is not None and 0 <= t - view.direction_at <= MAX_DT:
                self.motion = view.direction
        if not 0 <= t - self.last_t <= MAX_DT:
            self.cancel('frame_gap')
            return False
        if t == self.last_t:
            return True
        transform = camera_motion(self.last_t, t)
        if transform is None:
            if self.first is not None:
                self.cancel('camera_motion_unavailable')
            else:
                self.status = 'camera_motion_unavailable'
            return False
        centre, top_point = (warp_point(transform, *point) for point in self.projected)
        scale = warp_scale(transform)
        half_width, half_height = self.half_width * scale, self.half_height * scale
        if not all(map(math.isfinite, (*centre, *top_point, half_width, half_height))):
            self.cancel('invalid_projection')
            return False
        self.half_width, self.half_height = half_width, half_height
        self.last_t = t
        self.projected = centre, top_point
        self.status = 'parked'
        return True

    def step(self, t, view, camera_motion, size, aim, aim_fraction, *, prediction=None):
        if self.stopped:
            return None
        width, height = size
        if self.lost_at is None:
            anchor = view.seek_anchor
            if anchor is not None and anchor[2] is not None and 0 <= t - anchor[0] <= MAX_DT:
                self.motion = anchor[2]
            elif view.direction_at is not None and t - view.direction_at > MAX_DT:
                self.motion = None
            self.lost_at = t
        elapsed = t - self.lost_at
        end = TWO_AXIS_END_S if self.wide_pan else END_S
        if elapsed < 0 or elapsed >= end:
            self.cancel('search_deadline')
            return None
        if not self.maintain(t, view, camera_motion):
            return None
        if elapsed < START_S:
            self.status = 'waiting_for_coast'
            return None

        centre, top_point = self.projected
        top = top_point[1]
        if self.first is None:
            if elapsed >= FIRST_END_S:
                self.cancel('missed_first_leg')
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
                near_top = top < .20 * height
                near_bottom = top + 2 * self.half_height > .85 * height
                if self.wide_pan and (near_top or near_bottom):
                    self.first = 1, -1 if near_top else 1
                else:
                    self.first = (1, -1) if top < KIV_MARGIN * height else (0, 1 if centre[0] >= aim[0] else -1)
            if self.wide_pan:
                axis, direction = self.first
                other = 1 - axis
                other_direction = 1 if centre[other] >= aim[other] else -1
                if (not self._has_room(axis, direction, size, centre, top_point)
                        and self._has_room(axis, -direction, size, centre, top_point)):
                    direction = -direction
                # Check the other axis before finishing the first sweep.
                # A brief loss can end before both axes have returned to the anchor.
                self.local_legs = tuple(
                    (a, d * sign * LOCAL_RADII[a] * size[a], limit)
                    for sign, limit in zip((1, -1, 0), LOCAL_LEG_BUDGETS)
                    for a, d in ((axis, direction), (other, other_direction)))
        self.projected = centre, top_point
        axis, direction = self.first
        self.status = 'scan_paused'
        if self.local_legs:
            action = self._local_action(t, size, centre, top_point)
            if action is None:
                return None
            axis, direction = action
        elif native_control.native is not None:
            cancel, target = native_control.native.bounded_action(
                elapsed, centre, self.origin, size, aim, axis, direction, top, self.motion == (1, 1), aim_fraction)
            if cancel:
                self.cancel('returned_to_anchor')
            elif target is not None:
                self.status = 'scan'
            return target
        else:
            span = size[axis]
            shift = centre[axis] - self.origin[axis]
            if elapsed >= RETURN_BEGIN_S:
                if abs(shift) <= RETURN_TOLERANCE * span:
                    self.cancel('returned_to_anchor')
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
        target[axis] += direction * AIM_OFFSET * size[axis]
        box_size = 20.
        x, y = target[0] - box_size / 2, target[1] - aim_fraction * box_size
        if not (0 <= x <= width - box_size and 0 <= y <= height - box_size):
            return None
        self.status = 'scan'
        return x, y, box_size, box_size

    def _local_action(self, t, size, centre, top_point):
        if self.local_index == len(self.local_legs):
            self.cancel('scan_complete')
            return None
        if t < self.local_ready:
            return None
        axis, waypoint, limit = self.local_legs[self.local_index]
        span = size[axis]
        remaining = waypoint - (self.origin[axis] - centre[axis])
        if self.local_start is None:
            self.local_start = t
            self.local_direction = 1 if remaining > 0 else -1
        direction = self.local_direction
        if (direction * remaining <= RETURN_TOLERANCE * span
                or not self._has_room(axis, direction, size, centre, top_point)
                or t - self.local_start >= limit):
            self.local_index += 1
            self.local_ready = t + (LOCAL_SETTLE_S if t > self.local_start else 0.)
            self.local_start = None
            return None
        return axis, direction

    def _has_room(self, axis, direction, size, centre, top_point):
        span = size[axis]
        half = (self.half_width, self.half_height)[axis]
        last_centre = (top_point[0], top_point[1] + self.half_height)[axis]
        room = (min(centre[axis], last_centre) - half if direction > 0
                else span - max(centre[axis], last_centre) - half)
        return room > (KIV_MARGIN + BRAKE_MARGIN) * span
