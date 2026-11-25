"""Followed-person prediction and wrong-person guard for click-to-follow (11-15, user: "re-catch the same person after a loss").

Two opt-in parts, both off by default (live_tracks.publish uses them):

PREDICTION (for the bridge's --coast 1, sent as ' PRED <id> <x> <y> <w> <h>' only to a bridge that printed COAST_CONTROL 1).
While the followed id is seen without another box on it, its box and velocity are kept: velocity against the background
(camera motion removed with the GMC warp of every frame pair, the same warps the tracker uses), smoothed with TAU. When the id
is missing (or masked by the guard), the last box is carried through the camera motion of each frame and moved on with that
velocity for at most WALK_S. The gimbal's own turning is therefore not taken for person motion. The prediction is offered for
at most COAST_S after the last trusted box and while it is within COAST_MAX_H body heights of it; the bridge bounds the time
again (--coast-ms). No GMC for a frame pair = no prediction until the person is seen again.

GUARD (LWIR_FOLLOW_GUARD=1). In the virtual tests (BU-TIV Lab, PTB meetion1) following ended on another person mostly because
OC-SORT moved the followed id onto the crossing person with no gap (36 of 67), or the old id came back on another person after
a short gap (29 of 67); the image-place handoff did it twice (recatch/why_switch.py). The guard never changes the tracker; it
decides only what the gimbal follows:
  - RISK: another box overlaps the followed box (IoU > RISK_IOU). The trusted box/velocity are frozen; the prediction carries on.
  - When the boxes separate again (or the id comes back after a gap), the followed box must lie within the gate of the
    prediction (G0 + G1 x age heights, at most GMAX; height ratio within HR_MAX). If another separated box is clearly nearer
    the prediction (BETTER x the distance, the followed box at least MIN_SWAP_H heights off), for N frames in a row, following
    moves to that box's id (the ids were swapped). If the followed box is outside the gate, it is MASKED: the bridge does not
    see it (it handles a loss: stop, or --coast), until a box consistent with the prediction appears: the same id again, or
    for N frames in a row one other box alone near the prediction that was not on screen when ours was last trusted.
All distances are in body heights of the predicted box, so they do not depend on the zoom or the distance to the person."""
import math
import os

import numpy as np

from . import native_control
from ..reid.cos_map import cmap as _cmap

CANVAS_Y = 16.0     # GMC warps work on the padded canvas: people rows are 16 px higher
TAU = 0.25          # s, velocity smoothing
VMAX_H = 1.5        # body heights per second at most
WALK_S = 1.0        # s, the velocity moves the prediction for at most this long after the last trusted box
MAX_DT = 0.25       # s, a longer frame gap restarts the velocity
COAST_S = 1.0       # s, prediction offered for the bridge's coast at most this long after the last trusted box
COAST_MAX_H = 1.5   # body heights, ... and while the prediction is at most this far from the last trusted box
RISK_IOU = 0.05     # another box touches ours: risk
MERGE_IOU = 0.3     # ... covers it so much that the box may be both people: frozen
PART_D, VCONS_H = 0.15, 0.3   # partly covered: trusted only within PART_D heights of the prediction and if its motion
                              # differs from the person's by at most VCONS_H heights/s ...
CONS_MIN_S, CONS_MAX_S = 0.1, 0.6   # ... measured over this time since the last trusted box
RISK_WALK_S = 3.0   # s, while covered the velocity moves the prediction for at most this long
G0, G1, GMAX = 0.35, 0.5, 0.8   # gate: heights, heights per second of age, heights at most
HR_MAX = 1.5
WR_MAX = 1.5        # width ratio (a box of two people is wider)
FOREGROUND_W, FOREGROUND_H = 1.32, 1.28
FOREGROUND_JUMP_H, FOREGROUND_LARGE = 0.16, 1.60
FOREGROUND_CLEAR_N = 2
BETTER, MIN_SWAP_H = 0.6, 0.25
N = 3
LOST_MAX_S = 10.0   # the bridge drops the selection after its reacquire timeout (10 s)


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.
    i = w * h; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.


def overlaps(reference, others):
    if native_control.native is not None:
        return native_control.native.overlaps(reference, others)
    return [iou(reference[1:5], person[1:5]) for person in others]


def cwh(p):
    """(id, x1, y1, x2, y2, ...) -> (cx, cy, w, h)"""
    return ((p[1] + p[3]) / 2., (p[2] + p[4]) / 2., p[3] - p[1], p[4] - p[2])


def box(c):
    cx, cy, w, h = c
    return (cx - w / 2., cy - h / 2., cx + w / 2., cy + h / 2.)


def warp_point(w, x, y):
    px, py = w @ (x, y + CANVAS_Y, 1.0)
    return float(px), float(py - CANVAS_Y)


def warp_scale(w):
    return math.sqrt(abs(float(np.linalg.det(np.asarray(w, dtype=float)[:, :2])))) or 1.


class FollowGuard:
    """One followed id. step() once per published frame, with gb.people (stabilized picture coordinates) and motion(t0, t1) ->
    the 2x3 canvas warp from frame time t0 to t1 or None. guard=False: prediction only (never masks, never re-selects)."""

    def __init__(self, sel, guard):
        self.sel, self.guard = int(sel), bool(guard)
        self.trust = None          # (cx, cy, w, h) of the last trusted box, in the coordinates of trust_t
        self.trust_t = None
        self.vx = self.vy = 0.
        self.pred = None           # (cx, cy, w, h) at pred_t
        self.pred_t = None
        self.phase = 'track'       # track | risk | lost | masked
        self.ids_at_trust = set()
        self.cand, self.cand_n = 0, 0
        self.why = ''
        self.masked_n = self.swaps = self.reaccepts = 0
        self.partners = set()      # ids whose box overlapped ours since it was last seen alone
        self.v_known = False       # the velocity was measured on boxes without overlap
        self.mask_from_risk = False
        self.upper_coast_view = None
        self.bounded_seek = None

    def age(self, t):
        return t - self.trust_t if self.trust_t is not None else float('inf')

    def gate(self, t):
        return min(GMAX, G0 + G1 * max(0., self.age(t)))

    def _accept(self, t, c, people, motion, velocity):
        if velocity and self.trust is not None and 0 < t - self.trust_t <= MAX_DT:
            m = motion(self.trust_t, t)
            if m is not None:
                px, py = warp_point(m, self.trust[0], self.trust[1])
                dt = t - self.trust_t
                a = -math.expm1(-dt / TAU)
                self.vx = (1 - a) * self.vx + a * (c[0] - px) / dt
                self.vy = (1 - a) * self.vy + a * (c[1] - py) / dt
                s = math.hypot(self.vx, self.vy); vm = VMAX_H * c[3]
                if s > vm: self.vx, self.vy = self.vx * vm / s, self.vy * vm / s
                self.v_known = True
        elif velocity:
            self.vx = self.vy = 0.
        self.trust, self.trust_t = c, t
        self.pred, self.pred_t = c, t
        self.ids_at_trust = {int(p[0]) for p in people}

    def _propagate(self, t, motion):
        if self.pred is None or self.pred_t is None or t <= self.pred_t:
            return
        m = motion(self.pred_t, t)
        if m is None:
            self.pred = None; self.why = 'no camera motion'
            return
        cx, cy, w, h = self.pred
        s = warp_scale(m)
        cx, cy = warp_point(m, cx, cy)
        cap = RISK_WALK_S if self.phase == 'risk' else WALK_S
        walk = max(0., min(t - self.trust_t, cap) - min(self.pred_t - self.trust_t, cap))
        self.pred = (cx + self.vx * walk, cy + self.vy * walk, w * s, h * s)
        self.pred_t = t

    def dist(self, c):
        return math.hypot(c[0] - self.pred[0], c[1] - self.pred[1]) / max(1., self.pred[3])

    def fits(self, c, t):
        if native_control.native is not None:
            return native_control.native.guard_geometry(c, self.pred, self.gate(t))[1]
        hr = max(c[3], self.pred[3]) / max(1e-6, min(c[3], self.pred[3]))
        wr = max(c[2], self.pred[2]) / max(1e-6, min(c[2], self.pred[2]))
        return self.dist(c) <= self.gate(t) and hr <= HR_MAX and wr <= WR_MAX

    def _count(self, qid):
        if qid != self.cand: self.cand, self.cand_n = qid, 0
        self.cand_n += 1
        return self.cand_n >= N

    def step(self, t, people, motion):
        """-> dict(mask=bool (hide the followed id from the bridge this frame), reselect=id or 0, pred=(cx, cy, w, h) for the
        bridge's coast or None, phase, why)"""
        self._propagate(t, motion)
        mine = [p for p in people if int(p[0]) == self.sel]
        mine = mine[0] if len(mine) == 1 else None
        return self.step_after_propagate(t, people, motion, mine, dict(mask=False, reselect=0, pred=None))

    def step_after_propagate(self, t, people, motion, mine, out):
        others = [p for p in people if int(p[0]) != self.sel]
        if mine is not None:
            c = cwh(mine)
            near = max(overlaps(mine, others), default=0.) > RISK_IOU
            if self.pred is None or self.trust is None:   # first box, or the prediction was broken: start again from this box
                if not near or not self.guard or self.trust is None:
                    self._accept(t, c, people, motion, velocity=not near)
                    self.phase = 'risk' if near else 'track'
                    self.cand_n = 0
                else:
                    self.phase = 'masked'; out['mask'] = True; self.why = 'no prediction during an overlap'
            elif not self.guard:
                self._accept(t, c, people, motion, velocity=not near)   # prediction only: every box of the id is trusted
                self.phase = 'risk' if near else 'track'
            else:
                self._guard_seen(t, mine, c, people, others, motion, out)
                if out['reselect']:
                    return self._out(out, t)
        else:
            if self.phase not in ('lost', 'masked'): self.phase = 'lost'
            if self.guard and self.pred is not None and self.age(t) <= LOST_MAX_S:
                g = self.gate(t)
                fit = [q for q in others if self.fits(cwh(q), t) and self._allowed(int(q[0]))]
                crowd = [q for q in others if self.dist(cwh(q)) <= 1.5 * g]
                if len(fit) == 1 and len(crowd) == 1:
                    if self._count(int(fit[0][0])):
                        out['reselect'] = int(fit[0][0])
                        self.why = 're-catch: new id %d at the prediction' % out['reselect']
                        self.propose_retarget(out, cwh(fit[0]), t, people, "reaccepts")
                        return self._out(out, t)
                else:
                    self.cand_n = 0
        if (out['mask'] or mine is None) and self.pred is not None and self.trust is not None:
            moved = math.hypot(self.pred[0] - self.trust[0], self.pred[1] - self.trust[1]) / max(1., self.trust[3])
            if self.age(t) <= COAST_S and moved <= COAST_MAX_H:
                out['pred'] = self.pred
        return self._out(out, t)

    def _out(self, out, t):
        out['phase'], out['why'] = self.phase, self.why
        return out

    def _allowed(self, qid):
        """may following move to qid? A new id (not on screen when ours was last trusted), or, right after an overlap in which
        our id ended up on someone else, a box that overlapped ours (the tracker swapped the ids)."""
        return qid not in self.ids_at_trust or (self.mask_from_risk and qid in self.partners)

    def _allow_risk_without_identity(self):
        return True

    def _mask(self, why, from_risk):
        if self.phase != 'masked':
            self.masked_n += 1; self.why = why
            self.mask_from_risk = from_risk
        self.phase = 'masked'; self.cand_n = 0

    def _guard_seen(self, t, mine, c, people, others, motion, out):
        """guard decision for a frame in which the followed id has one box"""
        ovs = list(zip(overlaps(mine, others), (int(q[0]) for q in others)))
        ov = max((o for o, _ in ovs), default=0.)
        self.partners |= {i for o, i in ovs if o > RISK_IOU}
        ok, d = self.fits(c, t), self.dist(c)
        if ov > MERGE_IOU:   # the box may be both people: frozen, hidden only when it is clearly elsewhere
            if not ok:
                self._mask('jump during an overlap: %.2f heights from the prediction' % d, True); out['mask'] = True
            elif self.phase != 'masked':
                self.phase = 'risk'
            else:
                out['mask'] = True
            return
        better = None
        if self.phase in ('risk', 'masked', 'lost') or not ok:
            for q in others:
                if max(overlaps(q, [r for r in others if r is not q]), default=0.) > MERGE_IOU: continue
                qc = cwh(q); dq = self.dist(qc)
                if (self.fits(qc, t) and (not ok or (d >= MIN_SWAP_H and dq < BETTER * d)) and
                        (self._allowed(int(q[0])) or (self.phase == 'risk' and self._allow_risk_without_identity())) and (better is None or dq < better[1])):
                    better = (int(q[0]), dq, qc)
        if better is not None:
            if self._count(better[0]):
                out['reselect'] = better[0]
                self.why = 'swap: %d is nearer the prediction (%.2f vs %.2f heights)' % (better[0], better[1], d)
                self.propose_retarget(out, better[2], t, people, "swaps")
                return
            out['mask'] = True
            if not ok: self._mask('jump: %.2f heights from the prediction' % d, self.phase == 'risk')
            return
        if not ok:
            self._mask('jump: %.2f heights from the prediction' % d, self.phase == 'risk'); out['mask'] = True
            return
        if ov > RISK_IOU:   # partly covered: trusted only if it moved, since it was last trusted, the way the person was moving
            v = self._motion_since_trust(t, c, motion)
            # strict once the person's velocity was measured while alone; before that (walking with others since the click)
            # any box near the prediction teaches the velocity
            tight, cons = (PART_D, VCONS_H) if self.v_known else (G0, VMAX_H)
            stale = v is None and t - self.trust_t > CONS_MAX_S   # held for long: the prediction alone decides
            if d > tight or (not stale and (v is None or math.hypot(v[0] - self.vx, v[1] - self.vy) > cons * c[3])):
                if self.phase == 'masked': out['mask'] = True
                else: self.phase = 'risk'
                return
            if v is not None:
                a = -math.expm1(-(t - self.trust_t) / TAU)
                self.vx, self.vy = (1 - a) * self.vx + a * v[0], (1 - a) * self.vy + a * v[1]
            if self.phase in ('masked', 'lost'): self.reaccepts += 1
            self._accept(t, c, people, motion, velocity=False)
            self.phase = 'risk'; self.cand_n = 0
            return
        if self.phase in ('masked', 'lost'): self.reaccepts += 1
        self._accept(t, c, people, motion, velocity=self.phase == 'track' and ov <= RISK_IOU)
        self.phase = 'track' if ov <= RISK_IOU else 'risk'
        self.cand_n = 0
        if ov <= RISK_IOU:
            self.partners, self.mask_from_risk = set(), False

    def _motion_since_trust(self, t, c, motion):
        """the box's velocity against the background since the last trusted box (picture px/s), or None (too soon, too old,
        no camera motion)"""
        dt = t - self.trust_t
        if not CONS_MIN_S <= dt <= CONS_MAX_S:
            return None
        m = motion(self.trust_t, t)
        if m is None:
            return None
        px, py = warp_point(m, self.trust[0], self.trust[1])
        return ((c[0] - px) / dt, (c[1] - py) / dt)

    def propose_retarget(self, out, c, t, people, counter, *, forget=None):
        proposal = (out['reselect'], c, t, people, counter, forget)
        if getattr(self, 'defer_retarget', False):
            out['retarget'] = proposal
        else:
            self.accept_retarget(proposal)

    def accept_retarget(self, proposal):
        new_id, c, t, people, counter, forget = proposal
        self.retarget(new_id, c, t, people)
        setattr(self, counter, getattr(self, counter) + 1)
        if forget is not None:
            self.distract.pop(forget, None)

    def retarget(self, new_id, c, t, people):
        """Following moved to new_id (guard re-select, or a click/handoff for a new id): keep the velocity, trust its box."""
        self.sel = int(new_id)
        self.upper_coast_view = self.bounded_seek = None
        self.trust, self.trust_t, self.pred, self.pred_t = c, t, c, t
        self.ids_at_trust = {int(p[0]) for p in people}
        self.phase, self.cand, self.cand_n = 'track', 0, 0
        self.partners, self.mask_from_risk = set(), False


# 11-15 (user): KEEP-IN-VIEW coast (LWIR_COAST_MODE=fused). During a loss the first goal is to keep the person in the picture
# so the detector sees them again; exact centring does not matter. Two estimates that cover each other's mistakes: the
# constant-velocity prediction above and the tracker's own position memory (the Kalman prediction of the lost track, GMC
# warped). The region they span, widened by a growing uncertainty, is kept inside the picture with a margin, with the
# smallest camera motion: no motion while it is inside; if it is wider than the picture, the middle of it.
KIV_U0, KIV_U1, KIV_UMAX = 0.15, 0.5, 1.0   # uncertainty: heights + heights per second of age, at most
KIV_MARGIN = 0.10                            # of the picture width / height on each side


def keep_in_view(estimates, age, aim, aim_frac, size=(640., 480.)):
    """estimates: [(cx, cy, w, h), ...] in picture px; aim = (x, y) picture point the controller centres; aim_frac = the
    control point on a box, from its top (bridge --aim-y). -> a box (cx, cy, w, h) whose control point is where the aim must
    go (the aim itself when no motion is needed), or None."""
    est = [e for e in estimates if e is not None and all(map(math.isfinite, e)) and e[3] > 0]
    if not est:
        return None
    h = est[0][3]
    r = min(KIV_UMAX, KIV_U0 + KIV_U1 * max(0., age)) * h
    W, H = size
    shift = []
    for lo, hi, lim, a in ((min(e[0] - e[2] / 2 for e in est) - r, max(e[0] + e[2] / 2 for e in est) + r, W, aim[0]),
                           (min(e[1] - e[3] / 2 + aim_frac * e[3] for e in est) - r, max(e[1] - e[3] / 2 + aim_frac * e[3] for e in est) + r, H, aim[1])):
        m = KIV_MARGIN * lim
        if hi - lo > lim - 2 * m: s = (lo + hi) / 2 - lim / 2   # wider than the picture: its middle at the centre
        elif hi > lim - m: s = hi - (lim - m)
        elif lo < m: s = lo - m
        else: s = 0.
        shift.append(a + s)
    w = est[0][2]
    return (shift[0], shift[1] - aim_frac * h + h / 2, w, h)



class UpperCoastView:
    """Keep a last trusted box top and recent descent evidence in raw camera pixels."""

    def __init__(self):
        self.anchor = None
        self.down_steps = 0
        self.descent_t = None
        self.direction = self.direction_at = None
        self.motion_step = None

    def observe(self, t, estimate, motion, size):
        cx, cy, width, height = estimate
        frame_w, frame_h = size
        if not all(map(math.isfinite, estimate)) or min(width, height) <= 0:
            self.down_steps = 0
            return
        if not (0 < cx - width / 2 and cx + width / 2 < frame_w
                and 0 < cy - height / 2 and cy + height / 2 < frame_h):
            self.down_steps = 0
            return
        previous = self.anchor
        if previous is None or t - previous[0] > COAST_S:
            self.anchor = t, estimate
            self.down_steps = 0
            self.descent_t = None
            self.direction = self.direction_at = self.motion_step = None
            return

        previous_t, (old_x, old_y, _, old_h) = previous
        dt = t - previous_t
        if dt <= 0:
            return
        warp = motion(previous_t, t)
        if warp is None:
            self.down_steps = 0
            return
        expected_h = old_h * warp_scale(warp)
        if not 0.75 * expected_h <= height <= 1.25 * expected_h:
            self.down_steps = 0
            return
        _, old_top = warp_point(warp, old_x, old_y - old_h / 2)
        moved_x, old_centre = warp_point(warp, old_x, old_y)
        top_change = cy - height / 2 - old_top
        dy = cy - old_centre
        if dy * top_change > 0:
            dy = math.copysign(min(abs(dy), abs(top_change)), dy)
        else:
            dy = 0.
        dx = cx - moved_x
        axis, distance = max(enumerate((dx, dy)), key=lambda step: abs(step[1]))
        inferred = (axis, 1 if distance > 0 else -1) if abs(distance) >= max(1., .015 * height) and dt <= MAX_DT else None
        self.direction = inferred if inferred == self.motion_step else None
        self.direction_at = t if self.direction is not None else None
        self.motion_step = inferred
        least_descent = max(1., .015 * height)
        if (dt <= MAX_DT and cy - height / 2 - old_top >= least_descent
                and cy - old_centre >= least_descent):
            self.down_steps += 1
            if self.down_steps >= 2:
                self.descent_t = t
        else:
            self.down_steps = 0
            self.descent_t = None
        self.anchor = t, estimate

    def limit(self, t, prediction, motion, image_height, aim_fraction, aim_screen):
        """Limit only ambiguous downward aim; preserve the pan and box size."""
        if self.anchor is None:
            return prediction
        anchor_t, (anchor_x, anchor_y, _, anchor_h) = self.anchor
        if not 0 < t - anchor_t <= COAST_S:
            return prediction
        if self.descent_t is not None and t - self.descent_t <= MAX_DT:
            return prediction
        cx, cy, width, height = prediction
        aim_y = cy + (aim_fraction - .5) * height
        requested = aim_y - aim_screen * image_height
        if requested <= 0:
            return prediction
        warp = motion(anchor_t, t)
        if warp is None:
            return prediction
        _, top = warp_point(warp, anchor_x, anchor_y - anchor_h / 2)
        if not math.isfinite(top) or top >= image_height:
            return prediction
        room = max(0., top - KIV_MARGIN * image_height)
        if requested <= room:
            return prediction
        return (cx, cy - (requested - room), width, height)



# 11-15 GUARD v2 (LWIR_FOLLOW_GUARD=2): distractor-aware hand-back (KeepTrack / LTMU / DAM4SAM / Siam R-CNN style, see the
# re-catch report). v1 judged by position and motion only, and in the close-range BU-TIV scenes that does not separate the
# person from the one crossing (offline on the recorded runs it hid the clicked person about as often as the other one).
# v2 adds the person's appearance (the KeepID / relink RepVGG features the server already computes for some boxes every frame):
#   - the TARGET memory = the last APP_KEEP features of the followed box, taken only while it is seen alone (no other box on
#     it); frozen during a loss and whenever another box overlaps it;
#   - DISTRACTORS = the people near it (within APP_NEAR box widths): their last feature taken while they did not overlap ours;
#   - after an overlap or a loss the followed id is followed again only if its feature is nearer the target memory than every
#     distractor by APP_MARGIN (and at least APP_MIN); if it is nearer a distractor it is hidden (an id swap);
#   - a candidate for the hand-back (another id) must (a) not be a still-tracked distractor id unless its feature beats the
#     distractors by twice the margin, (b) look like the target as above, (c) be inside the motion gate of the velocity
#     prediction or of the tracker's Kalman memory (fused), (d) for N frames in a row.
# Without a fresh feature for the box in question it falls back to the v1 rules for that frame.
APP_KEEP, APP_NEAR, APP_MIN, APP_MARGIN, APP_FRESH = 5, 2.0, 0.55, 0.05, 6   # APP_FRESH: feature at most this many frames old
# 11-16 (gimbal search, search_follow.py): while the gimbal looks for the person inside the displayed search circle, the SAME
# number may come back anywhere in that circle (KeepID gives it back by appearance and its own search range), not only inside
# the motion gate. The guard then asks more of the appearance: at least APP_WIDE_MIN, APP_WIDE_MARGIN above every
# distractor, a height within WIDE_HR x of the last trusted box, and no other box on it. .wide = (cx, cy, radius) of the
# circle in the coordinates of this frame, or None (= the motion gate only, as before).
APP_WIDE_MIN, APP_WIDE_MARGIN, WIDE_HR = 0.65, 0.10, 2.0
SEARCH_VERIFY_S, SEARCH_VERIFY_SAMPLES = 0.9, 2
SEARCH_STRICT_START_H, SEARCH_STRICT_FULL_H = .35, 1.50
SEARCH_FAR_SCORE, SEARCH_FAR_MARGIN, SEARCH_FAR_SAMPLES_H = .82, .16, .85
# 11-22 OCCLUSION HOLD (LWIR_FOLLOW_OCC_HOLD=1, default off, proposal after the 12:41 loss of id 44 under id 23, IoU 0.6-0.88 for 5+ s): while the
# followed box is heavily covered (IoU > MERGE_IOU) the feature of that blended box says nothing about who it is, so v2 no longer masks it by
# appearance ('looks like a person near it'); it keeps following the id as long as the box stays inside the motion gate, for HOLD_MAX_S at most.
# The occluder's id is never taken as the new target during the hold, and the coast prediction is offered for the whole hold (not COAST_S).
# The hold ends when the overlap ends (IoU <= RISK_IOU): the normal appearance rules decide then (accept the id again, or mask / hand back).
HOLD_MAX_S = 6.0
# 11-22 POSITION TRUST (LWIR_FOLLOW_POS_TRUST=1, default off). Real event (15:31 board clock, id 18, a small person, prediction height 118-170 px, the other
# person on it only IoU 0.0-0.42): the tracker kept id 18 on the person, its box agreed with the guard's own prediction to 0.05 heights, and v2 still
# masked it three times with 'looks like a person near it' (the feature of a small, partly covered crop is near the neighbour's) and once with 'outside the
# gate'; a mask is sticky (it ends only when the overlap is <= RISK_IOU and the feature says target), so the bridge let go. An appearance-only verdict
# must not hide a person whose position says it is the same one. Rule, in step() before the appearance judgement:
#   eligible = the id has been present for POS_RUN_N frames, absences of at most POS_GAP_S (a flicker) not counting; a longer absence (a lost and returned id) restarts the count;
#   agrees   = its box is within POS_D heights of the guard's prediction and POS_SIZE of its height (width POS_WR), no other box is clearly nearer the
#              prediction (the v1 swap test: BETTER x the distance, the box at least MIN_SWAP_H off), and, once the person's velocity was measured
#              alone, the box's own motion over the last 0.15-0.6 s matches it within VCONS_H (two people crossing while the tracker swaps the ids);
#   an eligible, agreeing id is followed (never masked by appearance, whatever the overlap), its box becomes the trusted box again (the prediction
#   cannot go stale during a long walk-by), for POS_MAX_S of one overlap at most; a masked (by appearance or gate) id that agrees for POS_UNMASK_N
#   frames is unmasked at once, whatever the overlap. Otherwise the appearance rules decide as before, but a 'looks like a person near it' verdict
#   needs POS_DIS_N frames in a row before it masks. A jump, a new id, an id that returned after a gap (until the ordinary rules have accepted it once), or a nearer other box, or a box that moves unlike the measured walk keep the old behaviour.
POS_D, POS_SIZE, POS_WR = 0.3, 1.25, 1.5
POS_RUN_N, POS_GAP_S, POS_MAX_S, POS_DIS_N, POS_UNMASK_N = 3, 0.2, 10.0, 2, 2
POS_TRAIL_S, POS_V_MIN_S = 0.6, 0.15   # once the person's velocity was measured alone, the box's own motion over this window must match it (VCONS_H): two people
                                        # crossing while the tracker swaps the ids keep their own velocities


def hold_enabled():
    return os.environ.get('LWIR_FOLLOW_OCC_HOLD') == '1'


def pos_trust_enabled():
    return os.environ.get('LWIR_FOLLOW_POS_TRUST') == '1'


def unit(v):
    v = np.asarray(v, dtype=float).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 and np.isfinite(n) else None


class FollowGuardV2(FollowGuard):
    """FollowGuard with appearance. Before each step() the caller sets .app = function(output id) -> (feature, age in frames, observation frame)
    or None, and .kal = the tracker's Kalman box (cx, cy, w, h) of the followed id or None."""

    def __init__(self, sel, guard=True):
        super().__init__(sel, guard)
        self.app = lambda oid: None
        self.kal = None
        self.tmpl = []
        self.distract = {}
        self.app_masks = self.app_swaps = 0
        self.wide = None
        self.wide_reaccepts = 0
        self.allow_masked_seek = False
        self.masked_seek_lock = False
        self.trusted_control_owner = None
        self.continuous_control_owner = False
        self.hold_t0 = None
        self.holds = 0
        self.pos_trust = self.pos_keeps = self.pos_unmasks = self.pos_waits = 0   # frames trusted by position / of those, frames the appearance said distractor / unmasks / first-frame waits
        self.pt_t0 = None          # start of the current overlap segment (cap POS_MAX_S)
        self.pt_capped = False
        self.pt_last_t = None      # time of the last frame it had one
        self.pt_trail = []         # (t, cx, cy) of the last trusted boxes
        self.pt_reborn = False
        self.pt_run = 0            # frames it has had a box, flickers of at most POS_GAP_S not breaking the run
        self.pt_ok_n = 0           # frames in a row masked and agreeing (unmask counter)
        self.dis_n, self._dis_seen = 0, False
        self.mask_kind = ''
        self.foreground_blocked = False
        self.foreground_clear_n = 0
        self.search_verify_id = None
        self.search_verify_stamp = None
        self.search_verify_n = 0

    def _allowed(self, qid):
        """Never let the geometry-only fallback transfer identity unverified.

        Appearance-based hand-back is handled by _candidate; requiring it here
        prevents the base guard's fallback from choosing a different person
        when the embedding is stale, absent, ambiguous or mismatched.
        """
        # _candidate already accumulates appearance evidence across frames.
        # Do not allow the base class to count the same frame a second time.
        return False

    def _allow_risk_without_identity(self):
        return False

    def _wide_ok(self, c, f):
        """the followed number's box c with feature f is inside the search circle and clearly looks like the person"""
        if self.wide is None or f is None or self.trust is None:
            return False
        if math.hypot(c[0] - self.wide[0], c[1] - self.wide[1]) > self.wide[2]:
            return False
        if max(c[3], self.trust[3]) / max(1e-6, min(c[3], self.trust[3])) > WIDE_HR:
            return False
        st_, sd = self._sims(f)
        return st_ is not None and st_ >= APP_WIDE_MIN and st_ >= sd + APP_WIDE_MARGIN

    def recovery_radius(self, t):
        """Recovery radius in predicted body heights, including the wide match."""
        age = self.age(t)
        if self.pred is None or age > LOST_MAX_S:
            return 0.0
        if self.tmpl and age > 0.9:
            return min(1.8, GMAX + 0.6 * (age - 0.9))
        return self.gate(t)

    def _outside_gate_recovery(self, c, f, t, row, people):
        """Expand the match region after a loss, including during bounded seek."""
        age = self.age(t)
        if self.pred is None or f is None or not 0.9 < age <= LOST_MAX_S:
            return False
        if self.dist(c) > self.recovery_radius(t):
            return False
        h = self.pred[3]
        if max(c[3], h) > WIDE_HR * min(c[3], h):
            return False
        if any(int(other[0]) != int(row[0]) and iou(row[1:5], other[1:5]) > RISK_IOU
               for other in people):
            return False
        st, sd = self._sims(f)
        if st is None or st < APP_WIDE_MIN or st < sd + APP_WIDE_MARGIN:
            return False
        template = unit(np.mean(self.tmpl, 0))
        for other in people:
            if int(other[0]) == int(row[0]):
                continue
            feature = self._feat(other[0])
            if feature is not None and st < _cmap(float(feature @ template)) + APP_WIDE_MARGIN:
                return False
        return True

    def _feat(self, oid):
        a = self.app(int(oid))
        if not a or a[0] is None or a[1] is None or a[1] > APP_FRESH:
            return None
        return unit(a[0])

    def _sims(self, f, exclude=()):
        t = unit(np.mean(self.tmpl, 0)) if self.tmpl else None
        st_ = _cmap(float(f @ t)) if t is not None else None   # 11-20: per-model cosine map (reid/cos_map.py)
        sd = max((_cmap(float(f @ g)) for i, g in self.distract.items() if i not in exclude), default=-1.)
        return st_, sd

    def _looks(self, f, qid=None):
        """'target' / 'distractor' / None (cannot tell)"""
        st_, sd = self._sims(f, exclude=(qid,) if qid is not None else ())
        if st_ is None:
            return None
        m = APP_MARGIN * (2 if qid is not None and qid in self.distract else 1)
        if st_ >= APP_MIN and st_ >= sd + m:
            return 'target'
        if sd > st_ + APP_MARGIN:
            return 'distractor'
        return None

    def _in_gate(self, c, t):
        if self.pred is not None and self.fits(c, t):
            return True
        k = self.kal
        return k is not None and math.hypot(c[0] - k[0], c[1] - k[1]) / max(1., k[3]) <= self.gate(t)

    def _remember(self, mine, people):
        """target memory (alone) and distractor memory (near, not overlapping ours)"""
        if mine is not None:
            alone = max(overlaps(mine, [q for q in people if int(q[0]) != self.sel]), default=0.) <= RISK_IOU
            f = self._feat(self.sel)
            if alone and f is not None and self.phase == 'track':
                self.tmpl = (self.tmpl + [f])[-APP_KEEP:]
            ref = cwh(mine)
        elif self.pred is not None:
            ref = self.pred
        else:
            return
        for q in people:
            qid = int(q[0])
            if qid == self.sel: continue
            qc = cwh(q)
            if math.hypot(qc[0] - ref[0], qc[1] - ref[1]) > APP_NEAR * max(ref[2], qc[2]): continue
            if mine is not None and iou(mine[1:5], q[1:5]) > RISK_IOU: continue
            f = self._feat(qid)
            if f is not None:
                self.distract[qid] = f

    def _candidate(self, t, people, out):
        for q in people:
            qid = int(q[0])
            if qid == self.sel: continue
            if self.hold_t0 is not None and qid in self.partners: continue   # 11-22: never the occluder during an occlusion hold
            f = self._feat(qid)
            c = cwh(q)
            if f is None:
                continue
            if not (self._in_gate(c, t) or
                    (self.phase in ('lost', 'masked') and self._outside_gate_recovery(c, f, t, q, people))):
                continue
            if self._looks(f, qid) != 'target':
                continue
            if self._count(qid):
                out['reselect'] = qid
                self.why = 'hand-back by appearance: %d looks like the person and is in the gate' % qid
                self.propose_retarget(out, cwh(q), t, people, "app_swaps", forget=qid)
            else:
                out['mask'] = out['mask'] or self.phase == 'masked'
            return True
        return False

    def _mask(self, why, from_risk):
        fresh = self.phase != 'masked'
        super()._mask(why, from_risk)
        if fresh:
            self.mask_kind = 'appearance' if why.startswith(('looks like', 'outside the gate', 'awaiting appearance')) else 'other'

    def _accept(self, t, c, people, motion, velocity):
        super()._accept(t, c, people, motion, velocity)
        self.pt_trail = [(a, x, y) for a, x, y in self.pt_trail if t - a <= POS_TRAIL_S] + [(t, c[0], c[1])]

    def _trail_velocity(self, t, c, motion):
        """picture px/s of the box against the background over the last 0.15-0.6 s of trusted boxes, or None"""
        old = [(a, x, y) for a, x, y in self.pt_trail if POS_V_MIN_S <= t - a <= POS_TRAIL_S]
        if not old:
            return None
        a, x, y = old[0]
        m = motion(a, t)
        if m is None:
            return None
        px, py = warp_point(m, x, y)
        return ((c[0] - px) / (t - a), (c[1] - py) / (t - a))

    def retarget(self, new_id, c, t, people):
        super().retarget(new_id, c, t, people)
        self.pt_trail = []
        self.pt_last_t, self.pt_run, self.pt_ok_n, self.dis_n, self.pt_reborn = None, 0, 0, 0, False
        self.pt_t0, self.pt_capped, self.mask_kind = None, False, ''

    def _runs(self, t, mine):
        """frames the followed id has had a box, a flicker (absent at most POS_GAP_S) not breaking the run; a longer absence restarts it"""
        if mine is not None:
            if self.pt_last_t is not None and t - self.pt_last_t <= POS_GAP_S:
                self.pt_run += 1
            else:
                self.pt_run = 1
                self.pt_reborn = self.pt_last_t is not None   # came back after a real gap: the ordinary rules must accept it once
            self.pt_last_t = t

    def _pos_trust(self, t, people, motion, mine, out):
        """LWIR_FOLLOW_POS_TRUST: True = this frame is decided here (followed, or kept masked while the unmask evidence builds)"""
        c = cwh(mine)
        others = [q for q in people if int(q[0]) != self.sel]
        ov = max(overlaps(mine, others), default=0.)
        if ov <= RISK_IOU:
            self.pt_t0, self.pt_capped = None, False
            if self.phase == 'track':
                self.pt_ok_n = 0
                return False          # alone and trusted: the ordinary path
        elif self.pt_t0 is None:
            self.pt_t0 = t
        if self.pt_t0 is not None and t - self.pt_t0 > POS_MAX_S:
            self.pt_capped = True
        d = self.dist(c)
        hr = max(c[3], self.pred[3]) / max(1e-6, min(c[3], self.pred[3]))
        wr = max(c[2], self.pred[2]) / max(1e-6, min(c[2], self.pred[2]))
        nearer = d >= MIN_SWAP_H and any(self.dist(cwh(q)) < BETTER * d for q in others)
        v = self._trail_velocity(t, c, motion) if self.v_known else (0., 0.)
        moves = v is not None and math.hypot(v[0] - self.vx, v[1] - self.vy) <= VCONS_H * c[3]   # same walk as the person
        if (self.pt_capped or self.pt_run < POS_RUN_N or self.pt_reborn or d >= POS_D or hr > POS_SIZE or wr > POS_WR
                or nearer or not moves or (self.phase == 'masked' and self.mask_kind != 'appearance')):
            self.pt_ok_n = 0
            return False
        if self.phase == 'masked':
            self.pt_ok_n += 1
            if self.pt_ok_n < POS_UNMASK_N:
                out['mask'] = True
                return True
            self.pos_unmasks += 1
        f = self._feat(self.sel)
        if f is not None and self.tmpl and ov > RISK_IOU and self._looks(f) == 'distractor':
            self.pos_keeps += 1       # the appearance would have hidden it
        if self.phase in ('masked', 'lost'): self.reaccepts += 1
        self.pos_trust += 1; self.pt_ok_n = 0; self.dis_n = 0
        self.why = 'position trust: same id, %.2f heights from the prediction, covered %.2f' % (d, ov)
        self._accept(t, c, people, motion, velocity=False)
        self.cand_n = 0
        if ov > RISK_IOU:
            self.phase = 'risk'
        else:
            self.phase = 'track'; self.partners, self.mask_from_risk = set(), False; self.hold_t0 = None
        return True

    def step(self, t, people, motion):
        self._dis_seen = False
        mine = [p for p in people if int(p[0]) == self.sel]
        self._runs(t, mine[0] if len(mine) == 1 else None)
        out = self._step(t, people, motion)
        if not self._dis_seen: self.dis_n = 0
        if self.phase == 'track' and out['phase'] == 'track' and any(int(q[0]) == self.sel for q in people): self.pt_reborn = False   # accepted by the ordinary rules
        return out

    def _foreground_occlusion(self, t, mine):
        # A foreground object can inherit the selected ID, leaving no second row
        # for the usual overlap check. Use the pre-occlusion motion and scale.
        if not self.guard or self.trust is None:
            return False
        reference = self.pred if self.pred is not None else self.trust
        if mine is not None:
            c = cwh(mine)
            width = c[2] / max(1., reference[2])
            height = c[3] / max(1., reference[3])
            jump = math.hypot(c[0] - reference[0], c[1] - reference[1]) / max(1., reference[3])
            grew = width >= FOREGROUND_W and height >= FOREGROUND_H
            abrupt = grew and (jump >= FOREGROUND_JUMP_H or min(width, height) >= FOREGROUND_LARGE)
            if self.pred is not None and self.age(t) <= MAX_DT and abrupt:
                self.foreground_blocked = True
            consistent = (1 / POS_WR <= width <= POS_SIZE and 1 / POS_SIZE <= height <= POS_SIZE
                          and jump < POS_D)
            if self.foreground_blocked and consistent:
                self.foreground_clear_n += 1
                if self.foreground_clear_n >= FOREGROUND_CLEAR_N:
                    self.foreground_blocked = False
            else:
                self.foreground_clear_n = 0
        else:
            self.foreground_clear_n = 0
        return self.foreground_blocked

    def _verify_search(self, t, people, motion, mine, out):
        # Widening a positional hypothesis must not widen identity acceptance.
        eligible = []
        if self.pred is not None and self.tmpl:
            for row in people:
                ident = int(row[0])
                try:
                    observation = self.app(ident)
                    raw, age, stamp = observation[:3] if observation and len(observation) >= 3 else (None, None, None)
                except (TypeError, ValueError):
                    continue
                if raw is None or age is None or stamp is None or not 0 <= age <= APP_FRESH:
                    continue
                feature = unit(raw)
                if feature is None:
                    continue
                c = cwh(row)
                tight = self.fits(c, t)
                if not (tight or self._wide_ok(c, feature)
                        or self._outside_gate_recovery(c, feature, t, row, people)):
                    continue
                if any(other is not row and iou(row[1:5], other[1:5]) > RISK_IOU for other in people):
                    continue
                similarity, distractor = self._sims(feature)
                if similarity is not None and similarity >= APP_WIDE_MIN and similarity >= distractor + APP_WIDE_MARGIN:
                    eligible.append((similarity, ident, c, stamp, age, tight, distractor))
        eligible.sort(key=lambda entry: entry[0], reverse=True)
        if eligible:
            similarity, ident, c, stamp, age, tight, distractor = eligible[0]
            distance = self.dist(c)
            strength = min(1., max(0., (distance - SEARCH_STRICT_START_H)
                                    / (SEARCH_STRICT_FULL_H - SEARCH_STRICT_START_H)))
            min_score = APP_WIDE_MIN + strength * (SEARCH_FAR_SCORE - APP_WIDE_MIN)
            margin = APP_WIDE_MARGIN + strength * (SEARCH_FAR_MARGIN - APP_WIDE_MARGIN)
            other_score = eligible[1][0] if len(eligible) > 1 else -1.
            clear = similarity >= min_score and similarity >= max(other_score, distractor) + margin
        else:
            clear = False
        if clear:
            required = SEARCH_VERIFY_SAMPLES + int(distance >= SEARCH_FAR_SAMPLES_H)
            if ident != self.search_verify_id:
                self.search_verify_id, self.search_verify_stamp, self.search_verify_n = ident, None, 0
            if age <= 2 and (self.search_verify_stamp is None or stamp > self.search_verify_stamp):
                self.search_verify_stamp = stamp
                self.search_verify_n += 1
            if self.search_verify_n >= required:
                self.search_verify_id, self.search_verify_stamp, self.search_verify_n = None, None, 0
                if ident != self.sel:
                    out['reselect'] = ident
                    self.why = 'search recovered: %d verified by separate appearance observations' % ident
                    self.propose_retarget(out, c, t, people, 'app_swaps', forget=ident)
                    return self._out(out, t)
                self.masked_seek_lock = False
                self.reaccepts += 1
                self.wide_reaccepts += int(not tight)
                self._accept(t, c, people, motion, velocity=False)
                self.vx = self.vy = 0.
                self.phase, self.cand_n = 'track', 0
                self.partners, self.mask_from_risk = set(), False
                self.why = 'search recovered: selected id verified by separate appearance observations'
                return self._finish(out, t, mine)
        else:
            self.search_verify_id, self.search_verify_stamp, self.search_verify_n = None, None, 0
        out.update(mask=True, search_verification=True, seek_masked=self.pred is not None)
        self._mask('search waiting for appearance confirmation', self.phase == 'risk' or self.mask_from_risk)
        return self._finish(out, t, mine)

    def _step(self, t, people, motion):
        self._propagate(t, motion)
        mine = [p for p in people if int(p[0]) == self.sel]
        mine = mine[0] if len(mine) == 1 else None
        out = dict(mask=False, reselect=0, pred=None)
        if self._foreground_occlusion(t, mine):
            if self.why != 'foreground box jump on selected id':
                self._mask('foreground box jump on selected id', self.phase == 'risk')
            out.update(mask=True, foreground_blocked=True)
            return self._finish(out, t, mine)
        if self.guard and self.trust is not None and self.age(t) > SEARCH_VERIFY_S:
            return self._verify_search(t, people, motion, mine, out)
        f = self._feat(self.sel) if mine is not None else None
        if self.masked_seek_lock:
            # During a guarded search only the selected ID, verified against
            # the last trusted prediction and appearance, can resume control.
            comparable = mine is not None and self.pred is not None and f is not None and bool(self.tmpl)
            c = cwh(mine) if comparable else None
            look = self._looks(f) if comparable else None
            fits = comparable and self.fits(c, t)
            wide = comparable and not fits and self._outside_gate_recovery(c, f, t, mine, people)
            if (fits or wide) and look == 'target' and not any(
                    int(q[0]) != self.sel and iou(mine[1:5], q[1:5]) > RISK_IOU for q in people):
                self.masked_seek_lock = False
                self.reaccepts += 1
                self._accept(t, c, people, motion, velocity=False)
                self.phase = 'track'
                self.cand_n = 0
                self.partners, self.mask_from_risk = set(), False
                return self._finish(out, t, mine)
            # The public number alone does not establish continuity. Only
            # the uninterrupted tracker owner from a trusted control frame may
            # recover through the existing position-trust checks.
            if (self.continuous_control_owner and pos_trust_enabled() and self.mask_kind == 'appearance'
                    and mine is not None and self.pred is not None and self.trust is not None
                    and self.age(t) <= MAX_DT
                    and not any(int(q[0]) != self.sel and (
                        self.dist(cwh(q)) <= .55 or iou(mine[1:5], q[1:5]) > RISK_IOU)
                                for q in people)
                    and self._pos_trust(t, people, motion, mine, out)):
                if not out['mask']:
                    self.masked_seek_lock = False
                return self._finish(out, t, mine)
            self.pt_ok_n = 0
            out['mask'] = True
            out['seek_masked'] = self.pred is not None and (
                not people or (len(people) == 1 and comparable and (look == 'distractor' or not fits)))
            return self._finish(out, t, mine)
        seek_candidate = (self.allow_masked_seek and len(people) == 1
                          and mine is not None and self.pred is not None and self.trust is not None)
        position_disputed = seek_candidate and not self.fits(cwh(mine), t)
        if not position_disputed:
            self._remember(mine, people)
        if (pos_trust_enabled() and self.guard and mine is not None and self.pred is not None and self.trust is not None
                and not seek_candidate and self._pos_trust(t, people, motion, mine, out)):
            return self._finish(out, t, mine)
        if mine is not None and self.pred is not None and self.trust is not None and f is not None and self.tmpl:
            c = cwh(mine)
            others = [q for q in people if int(q[0]) != self.sel]
            overlap_values = overlaps(mine, others)
            ov = max(overlap_values, default=0.)
            if ov > RISK_IOU:
                self.partners |= {int(q[0]) for q, value in zip(others, overlap_values) if value > RISK_IOU}
            if ov <= RISK_IOU: self.hold_t0 = None   # the overlap ended: the normal rules decide
            if (hold_enabled() and ov > MERGE_IOU and not position_disputed and self.phase in ('track', 'risk', 'lost')
                    and not self.pt_capped and (self.hold_t0 is None or t - self.hold_t0 <= HOLD_MAX_S) and self.age(t) <= HOLD_MAX_S and self._in_gate(c, t)):
                if self.hold_t0 is None: self.hold_t0 = t; self.holds += 1
                self.phase = 'risk'; self.why = 'occlusion hold: covered %.2f by another box, inside the gate' % ov
                return self._finish(out, t, mine)
            if self.phase in ('risk', 'masked', 'lost') or ov > RISK_IOU or position_disputed:
                look = self._looks(f)
                gate = not position_disputed if seek_candidate else self._in_gate(c, t)
                if not gate and ov <= RISK_IOU and self.phase in ('lost', 'masked') and (
                        self._wide_ok(c, f) or
                        (look == 'target' and self._outside_gate_recovery(c, f, t, mine, people))):
                    self.reaccepts += 1; self.wide_reaccepts += 1   # 11-16: found again inside the search circle
                    self.why = 're-catch in the search circle by appearance'
                    self._accept(t, c, people, motion, velocity=False)
                    self.vx = self.vy = 0.
                    self.phase = 'track'; self.cand_n = 0
                    self.partners, self.mask_from_risk = set(), False
                    return self._finish(out, t, mine)
                if look == 'distractor' and gate and self.phase != 'masked' and pos_trust_enabled():
                    self._dis_seen = True; self.dis_n += 1
                    if self.dis_n < POS_DIS_N:   # one frame of 'looks like a person near it' does not hide anyone
                        self.pos_waits += 1
                        self.why = 'looks like a person near it, waiting for a second frame'
                        if ov > RISK_IOU: self.phase = 'risk'
                        return self._finish(out, t, mine)
                if look == 'distractor' or not gate:
                    if self.phase != 'masked':
                        self.app_masks += 1
                        self._mask('looks like a person near it' if look == 'distractor' else 'outside the gate', self.phase == 'risk')
                    out['mask'] = True
                    if seek_candidate:
                        self.masked_seek_lock = True
                        out['seek_masked'] = True
                    else:
                        self._candidate(t, people, out)
                    return self._finish(out, t, mine)
                if look == 'target' and ov <= RISK_IOU:
                    if self.phase in ('masked', 'lost'): self.reaccepts += 1
                    self._accept(t, c, people, motion, velocity=False)
                    self.phase = 'track'; self.cand_n = 0
                    self.partners, self.mask_from_risk = set(), False
                    return self._finish(out, t, mine)
                if look is None and ov <= RISK_IOU:
                    # A geometrically plausible selected ID may now belong to
                    # a different person. Try a separately identified target;
                    # never switch from geometry alone while it is ambiguous.
                    if self._candidate(t, people, out):
                        if out['reselect']:
                            return self._out(out, t)
                        out['mask'] = True
                        return self._finish(out, t, mine)
                if self.phase == 'masked':
                    out['mask'] = True
                    self._candidate(t, people, out)
                    return self._finish(out, t, mine)
                self.phase = 'risk'   # overlapped or not yet clear: follow it, trust nothing
                return self._finish(out, t, mine)
        if (mine is not None and self.guard and self.pred is not None
                and self.trust is not None and self.phase in ('risk', 'masked', 'lost')
                and self.tmpl and f is None):
            overlap = max((iou(mine[1:5], q[1:5]) for q in people
                           if int(q[0]) != self.sel), default=0.)
            velocity = self._motion_since_trust(t, cwh(mine), motion)
            consistent_motion = (self.v_known and velocity is not None and
                                 math.hypot(velocity[0] - self.vx, velocity[1] - self.vy)
                                 <= VCONS_H * cwh(mine)[3])
            if overlap <= RISK_IOU and not consistent_motion:
                # Do not trust a possibly swapped ID just because the boxes
                # separated before a clean identity feature became available.
                self._mask('awaiting appearance after ambiguity', self.phase == 'risk' or self.mask_from_risk)
                out['mask'] = True
                return self._finish(out, t, mine)
        if mine is None and self.guard and self.pred is not None and self.tmpl:
            if self.phase not in ('lost', 'masked'): self.phase = 'lost'
            matched = self._candidate(t, people, out)
            if matched and out['reselect']:
                return self._out(out, t)
            if not matched:
                self.cand, self.cand_n = 0, 0
            # The base path resets candidate evidence when no geometry-only
            # candidate is allowed; preserve the appearance counter instead.
            return self._finish(out, t, mine)
        return super().step_after_propagate(t, people, motion, mine, out)

    def _finish(self, out, t, mine):
        if (out['mask'] or mine is None) and self.pred is not None and self.trust is not None:
            moved = math.hypot(self.pred[0] - self.trust[0], self.pred[1] - self.trust[1]) / max(1., self.trust[3])
            if self.age(t) <= (HOLD_MAX_S if self.hold_t0 is not None else COAST_S) and moved <= COAST_MAX_H:
                out['pred'] = self.pred
        return self._out(out, t)
