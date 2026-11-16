"""PC simulation of the gimbal search that follows the displayed prediction (bridge --seek 1, ../search_follow.py):
does the camera have the person in the picture again while the number can still come back?

Run from tracking_server:   python -m src.gimbal.sim.seek_sim [--runs 400] [--seed 1] [--cap 8] [--json out.json]

What is real: follow_guard.FollowGuard (prediction), follow_guard.keep_in_view (the fused coast) and search_follow.Seek.
What is a model (stated so the numbers are read as geometry and timing, not as proof on the device):
  - one axis (pan); the picture shows scene x - pan; 640 px wide, 30 frames a second;
  - the bridge: P control (Kp 30, far gain 1.6 from 240 to 320 px, deadband 19 px in / 32 px out), SPD ramp 150 per second
    from SPD 1, a direction change through STOP, coast 800 ms, seek SPD cap; no D term, no lead;
  - the gimbal: 31 px/s per SPD, dead time 0.108 s, first-order lag 0.03 s, STOP after 0.064 s (measured 11-15);
  - the server answers 35 ms after the frame; the camera motion between frames is known exactly (GMC);
  - the tracker's memory of the lost track = the last velocity for 30 frames;
  - the person is found by the detector when at least half of the box is inside the picture, except during the "blind"
    time that starts the loss (occlusion / detector miss); no other people, no appearance, no wrong person.
Scenario of one run: the person (height 120 / 200 / 300 px) is followed at the centre, walking at 0..350 px/s. At t = 0
the detector goes blind for 0.3..2.5 s. The person keeps walking (half of the runs), stops after 0.2..1.5 s (30 %) or
turns back after 0.3..1.2 s (20 %). Counted: is the person in the picture and found again within 3 s (the KeepID window)
and within 10 s (the bridge's wait), when, and for how long they were outside the picture.
"""
import argparse
import json
import math
import random

import numpy as np

from .. import follow_guard as F
from .. import search_follow as S

W, H, FPS = 640.0, 480.0, 30.0
SUB = 10                       # plant steps per frame
DT = 1.0 / FPS
PX_PER_SPD, DEAD_S, LAG_S, STOP_S = 31.0, 0.108, 0.03, 0.064
SERVER_S = 0.035
KP, FAR_GAIN, FAR0, FAR1 = 30.0, 1.6, 240.0, 320.0
DB_IN, DB_OUT = 0.06 * 320.0, 0.10 * 320.0
RATE = 150.0
COAST_S = 0.8


class Gimbal:
    """Commands (time, signed SPD) -> pan (px)."""

    def __init__(self):
        self.cmds = [(-1e9, 0.0)]
        self.pan = 0.0
        self.v = 0.0

    def command(self, t, spd):
        if spd != self.cmds[-1][1]:
            self.cmds.append((t, spd))

    def advance(self, t0, t1):
        n = SUB
        h = (t1 - t0) / n
        for i in range(n):
            t = t0 + (i + 0.5) * h
            want = 0.0
            for tc, spd in reversed(self.cmds):
                if t >= tc + (STOP_S if spd == 0 else DEAD_S):
                    want = spd * PX_PER_SPD
                    break
            self.v += (want - self.v) * (1.0 - math.exp(-h / LAG_S)) if want else -self.v   # a STOP holds at once (measured: within 5 ms)
            self.pan += self.v * h
        if len(self.cmds) > 40: del self.cmds[:-20]


class Bridge:
    """The executor's output for one target per frame (None = stop)."""

    def __init__(self, cap):
        self.cap = cap
        self.dir = 0
        self.spd = 0.0
        self.hold = False

    def step(self, target_x, dt, capped):
        if target_x is None:
            self.dir, self.spd, self.hold = 0, 0.0, False
            return 0.0
        e = target_x - W / 2
        self.hold = abs(e) <= (DB_OUT if self.hold else DB_IN)
        if self.hold:
            self.dir, self.spd = 0, 0.0
            return 0.0
        want = KP * abs(e) / (W / 2)
        if abs(e) > FAR0: want *= 1.0 + (FAR_GAIN - 1.0) * min(1.0, (abs(e) - FAR0) / (FAR1 - FAR0))
        want = min(max(want, 1.0), 100.0)
        if capped: want = min(want, self.cap)
        d = 1 if e > 0 else -1
        if self.dir and d != self.dir:   # reversal through STOP
            self.dir, self.spd = 0, 0.0
            return 0.0
        if not self.dir:
            self.dir, self.spd = d, 1.0
        else:
            self.spd = min(max(want, self.spd - RATE * dt), self.spd + RATE * dt)
        return self.dir * round(self.spd)


def run(rng, seek_on, cap=8, h=200.0, v=150.0, blind=1.0, behaviour='walk', change_after=0.5, seconds=10.0, sweep_end=None):
    w = 0.4 * h
    gim, bridge = Gimbal(), Bridge(cap)
    guard, seek = F.FollowGuard(7, False), S.Seek(7)
    hist = {}

    def motion(t0, t1):
        out, t = np.eye(3), t0
        while t < t1:
            if t not in hist: return None
            t, m = hist[t]
            out = np.vstack([m, [0, 0, 1]]) @ out
        return out[:2] if t == t1 else None

    def person_x(t):
        if t <= 0 or behaviour == 'walk': return W / 2 + v * t
        if behaviour == 'stop': return W / 2 + v * min(t, change_after)
        return W / 2 + v * change_after - v * max(0.0, t - change_after) if t > change_after else W / 2 + v * t

    t = -2.0
    pending = []   # (time the command is written, signed SPD)
    lost_at = None
    back_at = None
    out_s = 0.0
    kal0 = None
    in_view_at_blind_end = None
    seen_far = False
    while t < seconds:
        t0, pan0 = t, gim.pan
        t = round(t + DT, 6)
        while pending and pending[0][0] <= t:
            gim.command(*pending.pop(0))
        gim.advance(t0, t)
        hist[t0] = (t, np.array([[1.0, 0.0, -(gim.pan - pan0)], [0.0, 1.0, 0.0]]))
        if len(hist) > 64: hist.pop(min(hist))
        px = person_x(t) - gim.pan
        visible = -w / 2 + 0.5 * w <= px <= W + w / 2 - 0.5 * w   # at least half of the box inside the picture
        detected = visible and not (0 < t <= blind)
        if t > 0 and not visible: out_s += DT
        if in_view_at_blind_end is None and t > blind: in_view_at_blind_end = visible
        people = [(7, px - w / 2, 240 - h / 2, px + w / 2, 240 + h / 2, 0.9)] if detected else []
        if detected and lost_at is not None and back_at is None:   # where the guard expected the person, before it sees the box
            guard._propagate(t, motion)
            seen_far = guard.pred is not None and abs(px - guard.pred[0]) > F.GMAX * h
        d = guard.step(t, people, motion)
        target, capped = None, False
        if detected:
            if lost_at is not None and back_at is None:
                back_at = t
            target = px
            seek.seen(guard.trust, people)
            kal0 = None
        else:
            if lost_at is None and t > 0: lost_at = t
            age = guard.age(t)
            if kal0 is None and guard.trust is not None:
                kal0 = [guard.trust[0], guard.vx, 0]
            kal = None
            if kal0 is not None and kal0[2] < 30:   # the tracker's memory of the lost track: carried by the camera motion, moved by its velocity
                m = motion(t0, t)
                kal0[0] = float((m @ (kal0[0], 256.0, 1.0))[0]) + kal0[1] * DT
                kal0[2] += 1
                kal = (kal0[0], 240.0, w, h)
            if d['pred'] is not None and age <= COAST_S:
                k = kal if kal is not None and abs(kal[0] - d['pred'][0]) <= F.COAST_MAX_H * h else None
                box = F.keep_in_view([d['pred'], k], age, (W / 2, H / 2), 0.5, (W, H))
                target = box[0] if box is not None else None
            if seek_on and guard.trust is not None:
                if sweep_end is not None: S.SWEEP_END_S = sweep_end
                box, st = seek.step(t, age, guard.trust, (guard.vx, guard.vy), kal, guard.pred, motion, (W, H), (W / 2, H / 2), 0.5)
                if box is not None:
                    target, capped = box[0], True
        if back_at is not None and t - back_at > 1.0:
            break
        pending.append((t + SERVER_S, bridge.step(target, DT, capped)))
    return dict(back=None if back_at is None else back_at - (lost_at or 0.0), out_s=out_s, in_view_at_blind_end=bool(in_view_at_blind_end),
                far=seen_far, sweeps=seek.sweeps)


def scenarios(n, seed):
    rng = random.Random(seed)
    for i in range(n):
        yield dict(h=rng.choice((120.0, 200.0, 300.0)), v=rng.choice((-1, 1)) * rng.uniform(0.0, 350.0), blind=rng.uniform(0.3, 2.5),
                   behaviour=rng.choices(('walk', 'stop', 'turn'), (0.5, 0.3, 0.2))[0],
                   change_after=rng.uniform(0.2, 1.5))


def summarize(rows):
    n = len(rows)
    back = [r['back'] for r in rows if r['back'] is not None]
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    return dict(n=n, back_3s=sum(1 for b in back if b <= 3.0) / n, back_10s=len(back) / n,
                in_view_when_detector_recovers=sum(r['in_view_at_blind_end'] for r in rows) / n,
                back_median_s=q(back, 0.5), out_of_view_mean_s=sum(r['out_s'] for r in rows) / n,
                far_from_prediction=sum(r['far'] for r in rows) / n, runs_with_sweep=sum(1 for r in rows if r['sweeps']) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=400)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--cap', type=int, default=8)
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    out = {}
    sc = list(scenarios(a.runs, a.seed))
    for name, on in (('off', False), ('seek', True)):
        rows = [dict(run(None, on, a.cap, **s), **s) for s in sc]
        groups = {'all': rows}
        for b in ('walk', 'stop', 'turn'): groups[b] = [r for r in rows if r['behaviour'] == b]
        groups['slow (< 100 px/s)'] = [r for r in rows if abs(r['v']) < 100]
        groups['fast (>= 200 px/s)'] = [r for r in rows if abs(r['v']) >= 200]
        for hh in (120.0, 200.0, 300.0): groups['height %d' % hh] = [r for r in rows if r['h'] == hh]
        out[name] = {g: summarize(r) for g, r in groups.items() if r}
    print('%-22s %5s | %-21s | %-21s | %-21s | %-19s | %s' % ('group', 'n', 'found again <= 3 s', 'found again <= 10 s', 'in view at detector', 'out of view, mean s', 'sweeps'))
    print('%-22s %5s | %-21s | %-21s | %-21s | %-19s |' % ('', '', 'off -> seek', 'off -> seek', 'recovery off -> seek', 'off -> seek'))
    for g in out['off']:
        o, s = out['off'][g], out['seek'][g]
        print('%-22s %5d | %8.2f -> %-9.2f | %8.2f -> %-9.2f | %8.2f -> %-9.2f | %7.2f -> %-8.2f | %.2f'
              % (g, o['n'], o['back_3s'], s['back_3s'], o['back_10s'], s['back_10s'], o['in_view_when_detector_recovers'],
                 s['in_view_when_detector_recovers'], o['out_of_view_mean_s'], s['out_of_view_mean_s'], s['runs_with_sweep']))
    print('found again by the search farther than 0.8 heights from the prediction (needs the guard\'s wide gate): %.2f of the runs'
          % out['seek']['all']['far_from_prediction'])
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(dict(args=vars(a), result=out), f, indent=1)


if __name__ == '__main__':
    main()
