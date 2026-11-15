#!/usr/bin/env python3
"""The search that follows the displayed prediction, closed loop in real time with the REAL bridge (nothing moves: the gimbal
is a pseudo terminal): server side = follow_guard.FollowGuard + keep_in_view + search_follow.Seek exactly as live_tracks
wires them; the bridge runs with the options of the live system plus --seek 1; a small plant model turns the serial commands
into camera pan (31 px/s per SPD, dead time 0.108 s, STOP after 0.064 s: measured 11-15) and the picture is the scene
shifted by that pan.
Scene: the followed person walks right at 200 px/s and is not detected from t = 0 on (walks behind something, then out of the
picture) and stops 380 px right of the place of the loss, 1.9 s later: outside the picture for a gimbal that stands still.
Usage: python3 test_seek_loop.py <path of gimbal_bridge_v27>   (test_gimbal_bridge.py, follow_guard.py, search_follow.py next to it)"""
import math, sys
sys.argv = sys.argv[:2]
import numpy as np
import test_gimbal_bridge as t
import follow_guard as F
import search_follow as S

W, H, DT = 640.0, 480.0, 1 / 30
LIVE = ('--controller', 'pid', '--kp', '30', '--ki', '0', '--kd', '6', '--integral-limit', '3', '--deadband', '0.06', '--speed-max', '100',
        '--reacquire-ms', '10000', '--deadband-exit', '0.10', '--axis-reversal', '1', '--aim-screen-y', '0.5', '--aim-y', '0.5',
        '--coast', '1', '--coast-ms', '800', '--speed-rate', '150', '--far-gain', '1.6', '--far-start-px', '240', '--far-full-px', '320')
PX_PER_SPD, DEAD_S, STOP_S = 31.0, 0.108, 0.064


class Plant:
    def __init__(self):
        self.cmds = [(-1e9, 0.0)]   # (time written, signed px/s)
        self.pan = 0.0
        self.t = t.time.monotonic()

    def write(self, c, now):
        self.cmds.append((now, 0.0 if c == t.STOP else c['X'] * c['SPD'] * PX_PER_SPD))

    def advance(self, now):
        n = 20
        h = (now - self.t) / n
        for i in range(n):
            x = self.t + (i + .5) * h
            v = 0.0
            for tc, s in reversed(self.cmds):
                if x >= tc + (STOP_S if s == 0 else DEAD_S):
                    v = s
                    break
            self.pan += v * h
        self.t = now


def run(seek, seconds=6.0, person_stop=380.0, h=200.0, v=200.0):
    w = 0.4 * h
    r = t.Rig(*LIVE, *(('--seek', '1', '--seek-speed', '8', '--seek-ms', '4500') if seek else ()))
    log = dict(cmds=[], lines=[], back=None, seek_spd=0, seek_cmds=[], pan=[])
    try:
        assert r.serial(1.0) == t.STOP
        plant, guard, sk = Plant(), F.FollowGuard(7, False), S.Seek(7)
        hist, frame_t = {}, None
        t0 = t.time.monotonic() + 1.5   # the loss
        next_ping, selected = 0.0, False
        kal = None

        def motion(a, b):
            out, x = np.eye(3), a
            while x < b:
                if x not in hist: return None
                x, m = hist[x]
                out = np.vstack([m, [0, 0, 1]]) @ out
            return out[:2] if x == b else None

        while True:
            now = t.time.monotonic()
            if now - t0 > seconds: break
            if now >= next_ping: r.send('P'); next_ping = now + .2
            pan0 = plant.pan
            plant.advance(now)
            if frame_t is not None: hist[frame_t] = (now, np.array([[1., 0., -(plant.pan - pan0)], [0., 1., 0.]]))
            tt = now - t0
            scene_x = W / 2 + v * tt if tt < 0 else min(W / 2 + v * tt, W / 2 + person_stop)
            px = scene_x - plant.pan
            visible = w / 2 - 0.5 * w <= px <= W - w / 2 + 0.5 * w
            detected = tt < 0 or (visible and tt > 1.2 and scene_x >= W / 2 + person_stop)   # hidden while walking away, found once it stands in the picture
            people = [(7, px - w / 2, 240 - h / 2, px + w / 2, 240 + h / 2, .9)] if detected else []
            d = guard.step(now, people, motion)
            suffix = ''
            if detected:
                if tt > 0 and log['back'] is None: log['back'] = tt
                sk.seen(guard.trust, people)
                kal = None
            else:
                age = guard.age(now)
                if kal is None: kal = [guard.trust[0], guard.vx, 0]
                k = None
                if kal[2] < 30:
                    m = motion(frame_t, now)
                    kal[0] = float((m @ (kal[0], 256., 1.))[0]) + kal[1] * (now - frame_t)
                    kal[2] += 1
                    k = (kal[0], 240., w, h)
                box = None
                if seek:
                    box, st = sk.step(now, age, guard.trust, (guard.vx, guard.vy), k, guard.pred, motion, (W, H), (W / 2, H / 2), .5)
                if box is not None:
                    suffix = ' SEEK 7 %.1f %.1f %.1f %.1f' % (box[0] - box[2] / 2, box[1] - box[3] / 2, box[2], box[3])
                elif d['pred'] is not None:
                    kk = k if k is not None and abs(k[0] - d['pred'][0]) <= F.COAST_MAX_H * h else None
                    b = F.keep_in_view([d['pred'], kk], age, (W / 2, H / 2), .5, (W, H))
                    suffix = ' PRED 7 %.1f %.1f %.1f %.1f' % (b[0] - b[2] / 2, b[1] - b[3] / 2, b[2], b[3])
            r.frame_id += 1
            rows = ''.join(' %d %.1f %.1f %.1f %.1f 0.900' % (p[0], p[1], p[2], p[3] - p[1], p[4] - p[2]) for p in people)
            r.send('F %d %d %d%s OBS 1%s' % (r.frame_id, t.time.monotonic_ns(), len(people), rows, suffix))
            frame_t = now
            if not selected and tt > -1.0:
                r.send('A 7'); selected = True
            slot = now + DT
            while True:
                c = r.serial(max(slot - t.time.monotonic(), 0))
                if c is None: break
                at = t.time.monotonic()
                plant.write(c, at)
                log['cmds'].append((at - t0, c))
                if ' SEEK ' in suffix or (log['seek_cmds'] and c == t.STOP):
                    log['seek_cmds'].append((at - t0, c))
            while True:
                a = r.line(.0005)
                if a is None: break
                log['lines'].append((tt, a))
            log['pan'].append((tt, plant.pan))
    finally:
        r.close()
    return log


def main():
    off = run(False)
    assert off['back'] is None, off['back']
    moves_late = [c for x, c in off['cmds'] if x > 1.2 and c != t.STOP]
    assert not moves_late, moves_late
    print('ok   without --seek: the coast only, the gimbal stands still from 0.8 s on and the person is not seen again (pan at the end %.0f px)' % off['pan'][-1][1])

    on = run(True)
    sk = [(x, c) for x, c in on['seek_cmds'] if c != t.STOP]
    assert sk, 'no command while SEEK was sent'
    top = max(c['SPD'] for _, c in sk)
    assert top == 8, top
    assert all(c['Y'] == 0 for _, c in sk)
    assert any(a == 'SEARCH 5' for _, a in on['lines']), [a for _, a in on['lines'] if a.startswith('SEARCH')]
    seq = [c for _, c in on['cmds'] if _ > 0.7]
    for a, b in zip(seq, seq[1:]):   # a change of direction never without a STOP between
        assert not (a != t.STOP and b != t.STOP and a['X'] and b['X'] and a['X'] != b['X']), (a, b)
    assert on['back'] is not None and on['back'] < 2.5, on['back']
    st = [a for _, a in on['lines'] if a.startswith('S ')]
    assert st and st[-1].split()[1:3] == ['1', '0'] and st[-1].split()[5] == '7', st[-3:]   # following number 7 again, not lost
    first = min(x for x, _ in on['seek_cmds'])
    print('ok   --seek 1: first search command %.2f s after the loss, SPD at most %d (= %d px/s), no tilt, turns through STOP' % (first, top, top * PX_PER_SPD))
    print('ok   --seek 1: the person, who stopped 380 px to the right 1.9 s after the loss, is found %.2f s after the loss and number 7 is followed again (pan then %.0f px)'
          % (on['back'], next(p for x, p in on['pan'] if x >= on['back'])))

    far = run(True, person_stop=2500.0)   # the person is gone for good
    assert far['back'] is None
    late = [(x, c) for x, c in far['cmds'] if x > 4.6 and c != t.STOP]
    assert not late, late
    pans = [p for x, p in far['pan'] if x > 0.8]
    limit = 200.0 + (W / 2 - 0.1 * W - 0.4 * 200.0 / 2) + 60.0   # the predicted place (200 px ahead) + the amplitude + stopping distance
    assert max(pans) <= limit and min(pans) >= -limit, (min(pans), max(pans), limit)
    assert abs(far['pan'][-1][1] - 200.0) <= S.LEG_TOL + 45.0, far['pan'][-1][1]
    print('ok   --seek 1, nobody comes back: swept between %.0f and %.0f px, back at the predicted place (%.0f px) and still from 4.6 s on'
          % (min(pans), max(pans), far['pan'][-1][1]))


if __name__ == '__main__':
    main()
