"""Keep the reappearance area in view after a sudden drop of the followed box."""

from collections import deque
import math

from .follow_guard import MAX_DT, warp_point, warp_scale


DROP_WINDOW = .18
RECOVERY_TIME = .35


class ReentryTilt:
    def __init__(self, selected):
        self.selected = selected
        self.recent = deque(maxlen=12)
        self.anchor = None
        self.visible_since = None
        self.seen = False

    @staticmethod
    def _project(sample, time, correction, motion):
        stamp, (x, y, _, height), (old_dx, old_dy) = sample
        warp = motion(stamp, time)
        if warp is None:
            return None
        _, top = warp_point(warp, x + old_dx, y - height / 2 + old_dy)
        height *= warp_scale(warp)
        top -= correction[1]
        return (top, height) if math.isfinite(top) and math.isfinite(height) and height > 0 else None

    def step(self, time, observed, correction, motion):
        """Return whether downward control should be inhibited on this frame."""
        if observed is None:
            # No observation can justify moving further down. Retain the last
            # visible region for the same track when it starts to reappear.
            self.visible_since = None
            if self.anchor is None and self.recent:
                self.anchor = self.recent[-1]
            return self.seen

        self.seen = True
        sample = time, observed, correction
        top = observed[1] - observed[3] / 2
        height = observed[3]
        previous = self.recent[-1] if self.recent else None
        if self.anchor is None:
            while self.recent and time - self.recent[0][0] > DROP_WINDOW:
                self.recent.popleft()
            oldest = self.recent[0] if self.recent else None
            candidates = (oldest,) if oldest is previous else (oldest, previous)
            for earlier in candidates:
                if earlier is None or not 0 < time - earlier[0] <= DROP_WINDOW:
                    continue
                region = self._project(earlier, time, correction, motion)
                if region is None:
                    continue
                old_top, old_height = region
                old_centre = old_top + old_height / 2
                if (top - old_top >= max(20., .18 * old_height)
                        and observed[1] - old_centre >= max(10., .09 * old_height)):
                    self.anchor = earlier
                    self.visible_since = None
                    break
        elif previous is not None:
            region = self._project(self.anchor, time, correction, motion)
            last = self._project(previous, time, correction, motion)
            last_top = last[0] if last is not None else previous[1][1] - previous[1][3] / 2
            reference_height = region[1] if region is not None else self.anchor[1][3]
            if height >= .8 * reference_height:
                if region is not None and top <= region[0] + .1 * reference_height:
                    self.anchor = None
                elif 0 < time - previous[0] <= MAX_DT and abs(top - last_top) <= max(3., .05 * height):
                    if self.visible_since is None:
                        self.visible_since = time
                    if time - self.visible_since >= RECOVERY_TIME:
                        self.anchor = None
                else:
                    self.visible_since = None
            else:
                self.visible_since = None
            if self.anchor is None:
                self.recent.clear()
                self.visible_since = None
        self.recent.append(sample)
        return self.anchor is not None


def limit_downward(box, aim_fraction, screen_y):
    """Limit a bridge-only xyxy box's vertical target while preserving pan."""
    x1, y1, x2, y2 = box
    offset = max(0., y1 + aim_fraction * (y2 - y1) - screen_y)
    return x1, y1 - offset, x2, y2 - offset
