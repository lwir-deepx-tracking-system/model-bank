#!/usr/bin/env python3
"""STAND-IN for gimbal_bridge_v25 / gimbal_bridge_v27, for the dry run of pid_hil.py / lead_hil.py on a PC that cannot build the real programs
(2026-11-11). It is NOT the real control code: it is a short re-write of the rules read in team/src/control/*.cpp and v27/src/control/*.cpp
(PD on the normalized error, deadband, speed ramp 60 SPD/s, 25 ms period, STOP on stale input / lost heartbeat / closed stdin, reversal by STOP;
v27: --deadband-exit hysteresis, --axis-reversal). Its only job is to give the test tool something that answers like a bridge, so that the
tool's own stops and limits can be exercised. Nothing measured with it says anything about v25 against v27.
Never used on the device: there --dry starts the real programs on a pseudo terminal.
stdin / stdout protocol and options as in gimbal_bridge_v25.cpp. Serial commands are appended, one JSON line each, to the file given by --device."""
import json, queue, sys, threading, time

V27_ONLY = ('--deadband-exit', '--stop-rate', '--axis-reversal', '--reversal-join-speed', '--reversal-hold-ms')
KNOWN = ('--device', '--manual-speed', '--speed-max', '--speed-min', '--speed-rate', '--x-sign', '--y-sign', '--kp', '--ki', '--kd', '--controller', '--integral-limit',
         '--deadband', '--aim-y', '--heartbeat-ms', '--input-timeout-ms', '--reacquire-ms', '--model-count', '--min-confidence', '--derivative-filter-ms')


def main():
    a = sys.argv[1:]
    if len(a) < 2 or a[0] != '--standin' or a[1] not in ('v25', 'v27'): sys.exit('usage: dry_bridge_standin.py --standin v25|v27 --device <file> [bridge options]')
    variant = a[1]; o = {}
    for i in range(2, len(a), 2):
        if i + 1 >= len(a): print('gimbal_bridge: missing value of ' + a[i], file=sys.stderr); sys.exit(2)
        if a[i] not in KNOWN and not (variant == 'v27' and a[i] in V27_ONLY): print('gimbal_bridge: unknown option ' + a[i], file=sys.stderr); sys.exit(2)
        o[a[i]] = a[i + 1]
    g = lambda k, d: float(o.get(k, d))
    kp, kd, db, aim = g('--kp', 30), (g('--kd', 0) if o.get('--controller', 'p') in ('pd', 'pid') else 0.0), g('--deadband', 0.04), g('--aim-y', 0.5)
    smax, smin, rate, man = int(g('--speed-max', 30)), int(g('--speed-min', 1)), g('--speed-rate', 60), int(g('--manual-speed', 30))
    xs, ys, hb, itmo, reacq = int(g('--x-sign', 1)), int(g('--y-sign', 1)), g('--heartbeat-ms', 1500) / 1e3, g('--input-timeout-ms', 250) / 1e3, g('--reacquire-ms', 1000) / 1e3
    db_exit, axis_rev, join, hold_s = g('--deadband-exit', 0), o.get('--axis-reversal', '0') == '1', g('--reversal-join-speed', 5), g('--reversal-hold-ms', 500) / 1e3
    dev = open(o['--device'], 'ab', buffering=0)
    out = lambda t: (sys.stdout.write(t + '\n'), sys.stdout.flush())
    q = queue.Queue()

    def reader():
        for line in sys.stdin: q.put(line.strip())
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()

    st = dict(auto=0, lost=0, moving=0, err=0, id=0); cur = [0, 0, 0]; lim = 0.0

    def serial(c): dev.write((json.dumps(c, separators=(',', ':')) + '\n').encode())

    def stop():
        nonlocal lim
        serial({'T': 135}); st['moving'] = 0; cur[:] = [0, 0, 0]; lim = 0.0

    def send(x, y, spd):
        if [x, y, spd] == cur: return
        serial({'T': 141, 'X': x * xs, 'Y': y * ys, 'SPD': spd}); cur[:] = [x, y, spd]; st['moving'] = int(bool(x or y))

    ctl = dict(init=False, pe=[0.0, 0.0], de=[0.0, 0.0], fid=0, t=0.0, hold=[False, False], cache=None)

    def reset(): ctl.update(init=False, pe=[0.0, 0.0], de=[0.0, 0.0], hold=[False, False], cache=None)

    def invalidate():
        reset(); st.update(auto=0, id=0, lost=0)
        if st['moving']: stop()

    stop()                                              # the real executor sends one STOP at start
    out('READY'); reported = None; connected = False; ping = 0.0; latest = None; lost_since = 0.0; next_auto = 0.0; last_out = None
    rhold = [False, False]; rsince = [0.0, 0.0]; nxt = time.monotonic()

    def find(i):
        if latest is None or not (0 <= time.monotonic_ns() / 1e9 - latest[1] < itmo): return None
        return next((t for t in latest[2] if t[0] == i), None)

    while True:
        try: line = q.get(timeout=0.005)
        except queue.Empty: line = ''
        if line is None: break
        f = line.split()
        if f:
            alive = connected and time.monotonic() - ping < hb
            if f[0] == 'F' and len(f) >= 4:
                n = int(f[3]); tr = [(int(f[4 + 6 * k]), *(float(v) for v in f[5 + 6 * k:10 + 6 * k])) for k in range(n)] if len(f) >= 4 + 6 * n else None
                if tr is not None and (latest is None or int(f[1]) > latest[0]): latest = (int(f[1]), int(f[2]) / 1e9, tr)
            elif f[0] == 'P':
                if not connected: invalidate(); connected = True
                ping = time.monotonic()
            elif f[0] == 'A':
                i = int(f[1]) if len(f) > 1 else -1
                if not alive: out('R A NOT_READY')
                elif i <= 0 or find(i) is None: out('R A INVALID')
                else: invalidate(); st.update(auto=1, id=i); out('R A OK')
            elif f[0] == 'M': invalidate() if st['auto'] else None; out('R M OK')
            elif f[0] == 'V' and len(f) == 3:
                x, y = int(f[1]), int(f[2])
                if st['auto']: invalidate()
                if not x and not y: stop() if st['moving'] else None
                elif alive: send(x, y, man)
            elif f[0] == 'H': invalidate()
        now = time.monotonic()
        if now >= nxt:
            nxt = now + 0.025
            alive = connected and now - ping < hb
            if not alive:
                if st['auto'] or st['moving']: invalidate()
            elif st['auto']:
                t = find(st['id'])
                if st['lost'] and now - lost_since >= reacq: invalidate()
                elif t is None:
                    if not st['lost']: reset(); st['lost'] = 1; lost_since = now
                    if st['moving']: stop()
                else:
                    st['lost'] = 0
                    if ctl['init'] and latest[0] == ctl['fid'] and ctl['cache'] is not None: p = ctl['cache']
                    else:
                        e = [(t[1] + t[3] / 2 - 320) / 320.0, (t[2] + aim * t[4] - 240) / 240.0]; dt = latest[1] - ctl['t'] if ctl['init'] else 0.0; p = [0.0, 0.0]
                        for k in (0, 1):
                            leave = max(db, db_exit); ctl['hold'][k] = abs(e[k]) <= (leave if ctl['hold'][k] else db)
                            if ctl['hold'][k]: ctl['pe'][k] = 0.0; ctl['de'][k] = 0.0; continue
                            if ctl['init'] and dt > 0 and kd > 0: ctl['de'][k] += dt / (0.1 + dt) * ((e[k] - ctl['pe'][k]) / dt - ctl['de'][k])
                            p[k] = kp * e[k] + kd * ctl['de'][k]; ctl['pe'][k] = e[k]
                        ctl.update(init=True, fid=latest[0], t=latest[1], cache=p)
                    sg = lambda v: 1 if v > 0 else -1 if v < 0 else 0
                    dx, dy = sg(p[0]), -sg(p[1])
                    if not dx and not dy:
                        if st['moving']: stop()
                    elif now >= next_auto:
                        next_auto = now + 0.025; rx, ry = bool(cur[0] and dx and cur[0] != dx), bool(cur[1] and dy and cur[1] != dy); wx, wy = dx, dy
                        want = min(max(max(abs(p[0]), abs(p[1])), smin), smax); go = True
                        if not axis_rev:
                            if rx or ry: stop(); go = False
                        else:
                            for k, r, d in ((0, rx, dx), (1, ry, dy)):
                                if r: rhold[k] = True; rsince[k] = now
                                if rhold[k] and not r and (not d or lim <= max(smin, join) or now - rsince[k] >= hold_s): rhold[k] = False
                            if rhold[0]: wx = 0
                            if rhold[1]: wy = 0
                            if not wx and not wy: stop(); go = False
                            else: want = min(max(max(abs(p[0]) if wx else 0.0, abs(p[1]) if wy else 0.0), smin), smax)
                        if go:
                            dt = 0.025 if last_out is None else min(now - last_out, 0.025); step = rate * dt
                            lim = min(max(want, max(smin, lim - step)), lim + step) if st['moving'] else float(smin)
                            last_out = now; send(wx, wy, min(max(int(round(lim)), smin), smax))
        s = 'S %d %d %d %d %d' % (st['auto'], st['lost'], st['moving'], st['err'], st['id'])
        if s != reported: out(s); reported = s
    invalidate()                                        # stdin closed: cancel following, STOP when it was moving
    try: out('S 0 0 0 0 0')
    except Exception: pass


if __name__ == '__main__': main()
