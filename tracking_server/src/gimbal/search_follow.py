"""Gimbal search that follows the displayed prediction (11-16, user: "예측 표시가 그러면 짐벌도 훑어야 하는 거 아닌가").

While the followed number is lost the page shows where the person is expected: a dashed predicted box and a search circle
that grows with time (pred_show; radius = last height x min(0.5 + 0.05 x lost frames, 4), the range in which KeepID can still
give the number back). Until now the gimbal only kept that place in view for 0.8 s (bridge --coast 1) and then stood still.
With the bridge option --seek 1 and LWIR_SEEK=1 (both off by default) the gimbal looks at the part of that circle the picture
does not show:

  1. PREDICT (0 .. PREDICT_S after the last trusted box): unchanged, the coast (PRED) keeps the predicted place in view.
  2. SWEEP (PREDICT_S .. SWEEP_END_S = the KeepID window): the circle's centre C is the displayed predicted box (the tracker's
     memory of the lost track while it has one, then the same place carried through the camera motion of every frame). The
     picture already covers C +- (half the picture width - margin). When the circle is wider than that, the camera turns so
     that the picture reaches the circle's edge: first to the side the person was moving to, then to the other side, each
     turn as far as the circle has grown by then:  A = min(R - (W/2 - margin), W/2 - margin - box width / 2).
     The second limit keeps the predicted place itself inside the picture at every moment, so a person who was only missed by
     the detector (the usual loss) is still seen where they are, and the guard can still hand following back there.
     A circle that fits in the picture needs no sweep: the gimbal stays as it is.
  3. RETURN (after SWEEP_END_S, at most RETURN_S): the number can now come back only at the predicted place (the guard's
     hand-back), so the camera turns back to C and waits there until the bridge gives up (--reacquire-ms).

What is sent to the bridge is only a point to look at (SEEK <id> <box>, the box whose control point the normal controller
centres): C shifted by the wanted offset. The camera's real turning is read from the picture (C moves the other way), so the
loop is closed on the picture, not on a speed table. The bridge alone decides how it moves there: SPD at most --seek-speed
(8 = about 250 px/s of picture motion, 8 px per frame: with the 31 px/s per SPD measured on 11-15), every change of direction
through a full STOP, nothing later than --seek-ms after the loss, and every safety stop of following.

NO SWEEP (the gimbal behaves as before) when
  - another person stood within NEAR_H heights of ours when it was last trusted (without overlapping it), or more than
    MAX_OTHERS other people were in the picture then (in the 11-15 virtual tests the tracker gave the number to other people
    during a sweep in crowds);
  - another person is in the picture within NEAR_H heights of the predicted place NOW (for NEAR_FRAMES frames in a row), or
    more than MAX_OTHERS others are: the sweep pauses (the gimbal stops) and goes on when they are gone. The turn back to
    the predicted place at the end is still made.
    A box that only OVERLAPPED ours right before the loss does not count by itself (11-16 device smoke runs: at an exit the
    detector often puts a second box on the same person, also in a scene with one person, and that blocked every sweep);
    if it is another person, they are still in the picture during the loss and the rule above sees them;
  - the guard hides the followed number (it is on another person: someone is there);
  - the number could not be re-accepted away from the predicted place anyway (guard v1, or guard v2 without an appearance
    memory of the person: see FollowGuardV2.wide);
  - the camera motion of a frame pair is unknown (flat picture, GMC off): the camera's turning cannot be measured.
"""
import math

import numpy as np

from . import native_control

FPS = 30.0
SR_BASE, SR_GROW, SR_MAX = 0.5, 0.05, 4.0   # the displayed circle = reid.relabel_online.search_range
PREDICT_S = 0.8        # s after the last trusted box: the prediction itself is followed (bridge --coast-ms)
SWEEP_END_S = 3.0      # s: the KeepID window (90 frames); later the number cannot come back away from the predicted place
RETURN_S = 1.2         # s: at most this long for the turn back to the predicted place
MARGIN = 0.10          # of the picture width kept free at each side (as follow_guard.keep_in_view)
MIN_A = 24.0           # px: a smaller sweep is not worth a move
LEG_TOL = 40.0         # px: back at the predicted place counts as reached this close (above the controller's deadband exit, 32 px)
LEG_LEAD = 120.0       # px: during a turn the point sent lies this far beyond the turn's end, so the speed stays at the bridge's cap
LEG_STOP = 30.0        # px: ... and the turn ends this far before its end (the gimbal runs on about 25 px after a STOP at SPD 8)
LEG_MAX_S = 2.5        # s: a turn that does not arrive (end stop, bad camera-motion estimate) is given up
NEAR_H = 2.0           # heights: another person this close when ours was last trusted = no sweep
NEAR_FRAMES = 3        # frames in a row another person must be near the predicted place before the sweep pauses
MAX_OTHERS = 2         # more other people than this in the picture then = no sweep
V_MIN_H = 0.15         # heights/s: the person's own speed picks the first side when it is above this
EDGE_PX = 8.0          # the point sent to the bridge stays this far inside the picture
BOX_W, BOX_H = 16.0, 32.0   # the box sent is only a carrier of that point (small, so no clipping at the picture edge can move it)

PHASES = {'idle': 0, 'predict': 1, 'sweep': 2, 'return': 3, 'wait': 4, 'crowd': 5, 'blind': 6, 'fits': 7, 'hold': 8}
PHASE_NAMES = {index: name for name, index in PHASES.items()}


def radius(h0, age):
    """the displayed search circle after `age` seconds without the person (px)"""
    return h0 * min(SR_BASE + SR_GROW * max(0.0, age) * FPS, SR_MAX)


def amplitude(r, w0, width):
    """how far (px) the camera may look to either side of the circle's centre: to the circle's edge, the centre staying in view"""
    m = MARGIN * width
    return max(0.0, min(r - (width / 2.0 - m), width / 2.0 - m - w0 / 2.0))


def count_near(people, selected, cx, cy, distance, ignored=()):
    if native_control.native is not None:
        return native_control.native.search_crowd(people, selected, cx, cy, distance, ignored or None)
    others = [q for q in people if int(q[0]) != selected and int(q[0]) not in ignored]
    near = sum(1 for q in others
               if math.hypot((q[1] + q[3]) / 2.0 - cx, (q[2] + q[4]) / 2.0 - cy) <= distance)
    return near, len(others)


class Seek:
    """One followed id. seen() on every frame its box is trusted, step() on every frame it is missing."""

    def __init__(self, sel):
        self.sel = int(sel)
        self.near = self.others = 0
        self.sweeps = self.skipped = 0     # counters: losses with a sweep / losses without one because of other people
        self._clear()

    def _clear(self):
        self.phase = 'idle'
        self.c = None          # (x, y): the circle's centre in picture px of frame c_t
        self.c_t = None
        self.h0 = self.w0 = 0.0
        self.dir0 = 0
        self.leg = 0
        self.leg_t = None
        self.swept = self.returned = self.blind = False
        self.near_run = self.now_near = self.now_others = 0
        self.state = None

    def seen(self, box, people, partners=()):
        """The followed box (cx, cy, w, h) was trusted in this frame; people = every box of the frame (id, x1, y1, x2, y2, ..);
        partners = ids whose box overlaps ours (often a second box on the same person: not counted here)."""
        cx, cy, w, h = box
        self.near, self.others = count_near(people, self.sel, cx, cy, NEAR_H * h, partners)
        if self.phase != 'idle':
            self._clear()

    def seen_untrusted(self):
        """The followed number has a box again that is followed but not trusted yet (another box on it): the loss is over."""
        if self.phase != 'idle':
            self._clear()

    def crowd(self):
        return self.near > 0 or self.others > MAX_OTHERS

    def step(self, t, age, trust, vel, kal, pred, motion, size, aim, aim_frac, *, hidden=False, can_reaccept=True, people=()):
        """t: frame time; age: s since the last trusted box; trust: that box (cx, cy, w, h); vel: the person's (vx, vy) px/s;
        kal / pred: the tracker's memory of the lost track / the guard's prediction in this frame, (cx, cy, w, h) or None;
        motion(t0, t1): 2x3 canvas warp or None; size: (W, H); aim: the picture point the controller centres; aim_frac: the
        control point on a box from its top; hidden: the guard hides the number (it has a box); can_reaccept: see above;
        people: the boxes of this frame (id, x1, y1, x2, y2, ..), for the other people in the picture now.
        -> (the box to look at (cx, cy, w, h) or None, state for the page)"""
        W, H = size
        if self.c is None:   # the first frame of this loss
            self.h0, self.w0 = float(trust[3]), float(trust[2])
            start = kal or pred or trust
            self.c, self.c_t = (float(start[0]), float(start[1])), t
        elif t > self.c_t:
            m = motion(self.c_t, t) if motion is not None else None
            if kal is not None:   # the displayed predicted box: the tracker carries it through the camera motion itself
                self.c = (float(kal[0]), float(kal[1]))
            elif m is not None:
                m = np.asarray(m, dtype=float)
                x, y = m @ (self.c[0], self.c[1] + 16.0, 1.0)   # GMC warps work on the padded canvas (rows 16 px lower)
                self.c = (float(x), float(y) - 16.0)
            if m is None:
                self.blind = True   # the camera's turning is unknown from here on: no search for this loss
            self.c_t = t
        self.now_near, self.now_others = count_near(people, self.sel, *self.c, NEAR_H * self.h0)
        self.near_run = self.near_run + 1 if (self.now_near or self.now_others > MAX_OTHERS) else 0
        if native_control.native is not None:
            vx = float(vel[0]) if vel is not None else 0.
            phase, self.dir0, self.leg, self.sweeps, self.skipped, leg_time, self.swept, self.returned, r, a, target, lead, pan = (
                native_control.native.sweep_step(
                    t, age, W, self.h0, self.w0, self.c[0], aim[0], vx, SWEEP_END_S,
                    self.crowd() or (self.near_run >= NEAR_FRAMES and age < SWEEP_END_S),
                    hidden or not can_reaccept, self.blind, PHASES[self.phase], self.dir0, self.leg,
                    self.sweeps, self.skipped, self.leg_t or 0., self.swept, self.returned))
            self.phase = PHASE_NAMES[phase]
            self.leg_t = leg_time if self.swept else None
        else:
            r = radius(self.h0, age)
            a = amplitude(r, self.w0, W)
            pan = aim[0] - self.c[0]   # how far right of the centre the camera looks now
            target, lead = None, 0.0
            if age < PREDICT_S:
                self.phase = 'predict'
            elif self.crowd() or (self.near_run >= NEAR_FRAMES and age < SWEEP_END_S):
                if self.phase != 'crowd': self.skipped += 1
                self.phase = 'crowd'
            elif hidden or not can_reaccept:
                self.phase = 'hold'   # the number has a box the guard does not trust, or could not be taken back far away
            elif self.blind:
                self.phase = 'blind'
            elif age < SWEEP_END_S:
                if a < MIN_A and not self.swept:
                    self.phase = 'fits'
                else:
                    if not self.swept:
                        self.swept, self.leg, self.leg_t = True, 0, t
                        self.sweeps += 1
                        vx = float(vel[0]) if vel is not None else 0.0
                        self.dir0 = ((1 if vx > 0 else -1) if abs(vx) >= V_MIN_H * self.h0
                                     else (1 if self.c[0] >= aim[0] else -1))
                    self.phase = 'sweep'
                    for _ in range(2):
                        d = self.dir0 if self.leg % 2 == 0 else -self.dir0
                        target = d * a
                        if d * (target - pan) > LEG_STOP and t - self.leg_t <= LEG_MAX_S:
                            break
                        self.leg += 1; self.leg_t = t   # this turn's end is reached (or given up): the other side
                    lead = d * LEG_LEAD
            elif self.swept and not self.returned and age < SWEEP_END_S + RETURN_S:
                self.phase = 'return'
                target = 0.0
                if abs(pan) <= LEG_TOL:
                    self.returned = True
                    self.phase, target = 'wait', None
            else:
                self.phase = 'wait'
        box = None
        if target is not None:
            x = min(max(self.c[0] + target + lead, EDGE_PX), W - EDGE_PX)
            box = (x, aim[1] - aim_frac * BOX_H + BOX_H / 2.0, BOX_W, BOX_H)   # no tilt: the control point is at the aim height
        self.state = dict(ph=PHASES[self.phase], phase=self.phase, c=(round(self.c[0], 1), round(self.c[1], 1)), r=round(r, 1),
                          a=round(a, 1), pan=round(pan, 1), to=None if target is None else round(target, 1), leg=self.leg,
                          near=self.near, others=self.others, now_near=self.now_near, now_others=self.now_others)
        return box, self.state
