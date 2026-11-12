#!/usr/bin/env python3
"""예측 제어 on/off on the REAL gimbal with a VIRTUAL person (user 2026-11-11: "hils 처럼 가상의 추적 있다 생각하고 실제 제어하면서 보라고").
Reuses pid_hil.py of 2026-11-10 unchanged (real camera, real bridge gimbal_bridge_v25, real gimbal, virtual person glued to the background).
Addition 1: in steps with "lead": 1 the box sent to the bridge first goes through lead_shift, taken verbatim from the live server file,
with the camera-motion warp = measured scene shift between two frames. The logged error is always that of the REAL (unshifted) virtual person.
Addition 2 (second run, rule written in PLAN_LEAD_ONOFF_20261111.md before the run): when a block dies with "image shift lost" the gimbal is
stopped, the current view is registered again against the START view (15 small patches, at least 3 must agree), the gimbal goes home and the
same block is repeated (name + _tryN). At most 3 attempts per block, 8 repeats per run. blocks.csv says which attempt of each block counts.
Addition 3 (after the 7 s attempt of 09:36, rule written in the plan before the next run): the plan starts with 10 s of waiting without measurement
(the first frames after the camera opens come before its shutter correction); a "static" step that dies before the gimbal has ever moved
waits 3 s and takes a NEW start view (2 more tries); every loss leaves pictures lost_NN_*.png (moment of the loss, 0.5 s, 1.0 s, found).
Run in ~/pid_hil_20261110/ :  python3 lead_hil.py --plan plan_lead2.json --out lead2   (add --dry for a run without camera and gimbal;
LEAD_DRY_FORCE_LOSS=step:seconds[,step:seconds] forces one loss in each of those steps, dry mode only)

v27 test copy (2026-11-11, every changed line carries "v27"; list and reasons in CHANGES_FROM_M1_TOOLS_KO.md). Compares gimbal_bridge_v25 with
gimbal_bridge_v27: a track step names its program with "bridge": "v25" | "v27" and its extra options with "extra": [...] (handled in pid_hil.py).
lead_shift is read from the NEW server folder (src/gimbal/gimbal_link.py). A frozen or old picture stops the gimbal at once (pid_hil.py) and is then
handled like a lost match (start view found again on a LIVE picture, home, same block again). After the last STOP of a run the end view is matched
against the start view and written to end_state.json. Run in the folder of these files through run_v27_hil.sh (stage1 | stage2 | stage3).
HIL_DRY_FAULT=kind:step:seconds[:duration][,...] forces faults, dry mode only. kind = freeze | homefreeze | stall | old | raise | sigterm | sigint | hang
| bridgekill | homebridgekill | stopfail | nomatch | zfreeze   (the last five: second round, "v27b").

Second round, same day ("v27b"): a block that ends as "person lost" (error limit, pid_hil.py) is recorded and the run goes on, at most 3 per run;
"pulse" steps = the stage 1 identification: short timed moves both ways for one speed command on one axis. The length of every move comes from what was
measured so far (rule in PLAN_V27_HILS_20261111_KO.md, section "1단계"), a command that could reach the travel budget even with the shortest move is
left out together with all larger ones, and the stage ends when a move turns out far faster than expected. Results: ident.csv.

Third round, 2026-11-12 ("v27c": the second safety review, ../review2; every rule written in the plan first): in a walk of FAST_WALK px/s or faster a block
that loses the picture matching is repeated once only, and the second such loss in the same walk skips the rest of that walk (blocks.csv: "skipped");
a block ended by the up-down guard (pid_hil.py) is recorded as "person lost"; the first move of a command never measured on this gimbal (ID_NEW) is the
shortest one; after the last STOP two views more than 4 px apart mean the gimbal still moves: loud warning, STOP again, exit code 7."""
import glob, json, os, signal, sys, textwrap, threading, time, types   # v27: + signal, threading, types
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pid_hil as H

d = glob.glob('/dev/serial/by-id/usb-Silicon_Labs_CP2102N_*')
if d: H.DEV = d[0]
L = open(H.ROOT + '/src/gimbal/gimbal_link.py', encoding='utf-8').read().split('\n')   # v27: lead_shift lives in this file of the new server folder
a = next(i for i, s in enumerate(L) if s.strip().startswith('def lead_shift('))
b = next(i for i in range(a + 1, len(L)) if L[i].strip() and len(L[i]) - len(L[i].lstrip()) <= len(L[a]) - len(L[a].lstrip()))
ERR = []
NS = dict(np=np, APP=types.SimpleNamespace(CTL=dict(lead=True, lead_ms=150, lv=None, lead_px=0.0, W=None)), print=lambda *x, **k: ERR.append(x))   # v27: the function now reads its values from APP.CTL
exec(textwrap.dedent('\n'.join(L[a:b])), NS)
lead_shift, CTL = NS['lead_shift'], NS['APP'].CTL   # v27
FORCE = {int(v.split(':')[0]): float(v.split(':')[1]) for v in os.environ.get('LEAD_DRY_FORCE_LOSS', '').split(',') if v}
MAX_TRY, MAX_REPEAT, PATCH_MIN, AGREE = 3, 8, 0.6, 3
FAST_WALK, FAST_WALK_LOST = 160.0, 2   # v27c (second review; plan section 5, rule 4): a walk in which the virtual person moves at this many px/s or faster; picture-matching losses inside its following blocks after which the rest of that walk is skipped (so a block is repeated once at most)
FAULT = [(f[0], int(f[1]), float(f[2]), float(f[3]) if len(f) > 3 else 1.0) for f in (v.split(':') for v in os.environ.get('HIL_DRY_FAULT', '').split(',') if v)]   # v27: forced faults, dry mode only
# v27b, stage 1 (identification). All written in the plan before any run:
ID_PRIOR = 160.0     # px/s expected for SPD 1 before anything is measured (the 11-10 record; the note in the unused v26 program says the same)
ID_PROBE = 2.5       # the first move of every command is sized for a gimbal 2.5 x faster than expected (expected = the speed of the command before it x the ratio of the two commands)
ID_UP = 1.3          # later moves of the same command are sized for 1.3 x what this command's own moves showed
ID_FAR = 1.5         # "far above the expectation": the stage ends
ID_JIT = 0.02        # s added to every move when its travel is predicted (the pulse may end this much late)
ID_TMIN = 0.02       # s, the shortest move that is tried; below this the command is left out
ID_TMAX = 0.6        # s, the longest move
ID_FLOOR = 100.0     # px/s, a measured speed below this is not used to expect the next command
ID_GROW = 2.5        # a move is at most this many times longer than the longest one before it (same command)
ID_DELTA = 0.10      # s, when a move is sized from this command's own earlier move, the gimbal may have moved for this much less time than commanded
ID_GATE = 110.0      # px between two frames accepted during the pulses (elsewhere 36)
ID_MOVE = 5.0        # px, less travel than this = "no clear motion"
ID_PAIRS = 3         # pairs of moves (one each way) per command
ID_NEW = {'x': 50, 'y': 20}   # v27c (second review; plan section 1, rule 3): commands at or above this value were never measured on this gimbal (left-right 50 and 100, up-down 20 and above): their first move is the shortest one (ID_TMIN) whatever is expected, and without clear motion at most twice that
ID_FIELDS = 'step axis spd pair dir status T_plan_ms T_act_ms v_expected v_plan room_px travel_px v_pulse v_peak delay_lo_ms delay_hi_ms s_before s_after frames note'.split()


class S2(H.Session):
    def __init__(s, a):
        s.lead_now = False; s.prevS = None; s.llog = open(os.path.join(a.out, 'lead.csv'), 'w') if os.makedirs(a.out, exist_ok=True) is None else None
        s.llog.write('fid,lead,shift_x,shift_y\n'); s.blog = open(os.path.join(a.out, 'blocks.csv'), 'w'); s.blog.write('step,name,lead,status,bridge\n')   # v27: + bridge column
        s.idf = open(os.path.join(a.out, 'ident.csv'), 'w'); s.idf.write(','.join(ID_FIELDS) + '\n'); s.idm = {}; s.idsgn = {}; s.idlost = 0   # v27b: stage 1 results and state
        s.end_view_checked = False                    # v27d: stays False unless where() matched the end view (an end in which where() did not run or failed counts as not checked)
        s.key0 = None; s.in_run = False; s.t_step = time.monotonic(); s.forced = set(); s.proc = 0.0; s.nlost = 0; s.moved = False; s.t_home = time.monotonic(); s.fdone = set(); super().__init__(a)   # v27: + t_home, fdone
        if a.dry and FAULT: s.hook = s.dry_fault      # v27

    def dry_fault(s):
        """v27, dry mode only: forced faults. freeze / homefreeze = the same picture keeps coming (run step / home step), stall = no picture at all,
        raise = a program error, sigterm / sigint = a signal, hang = the main loop sleeps while the gimbal has a velocity command.
        v27b: bridgekill / homebridgekill = the bridge program is killed (it sends no STOP), stopfail = from now on the emergency stop cannot be written
        (plus a frozen picture, so that one is needed), nomatch = pictures that match nothing, zfreeze = frozen picture while the command is ZERO."""
        for i, (kind, step, sec, dur) in enumerate(FAULT):
            if i in s.fdone or step != s.step_i: continue
            if kind in ('homefreeze', 'homebridgekill'): due = s.name == 'home' and time.monotonic() - s.t_home > sec   # v27b: + homebridgekill
            elif kind == 'zfreeze': due = (s.in_run or s.name == 'static') and time.monotonic() - s.t_step > sec and s.tap.last[2] == 0   # v27b
            else: due = (s.in_run or s.name.startswith('pulse_')) and time.monotonic() - s.t_step > sec   # v27b: + the stage 1 moves
            if kind != 'zfreeze': due = due and s.tap.last[2] != 0   # only while a velocity command is active: the dangerous case
            if not due: continue
            s.fdone.add(i); print('  DRY FAULT %s in step %d (pretend gimbal at %.1f, %.1f, command %s)' % (kind, step, s.cam.pos[0], s.cam.pos[1], s.tap.last), flush=True)
            if kind in ('freeze', 'homefreeze', 'zfreeze'): s.cam.freeze_until = time.monotonic() + dur
            elif kind == 'stall': s.cam.stall_until = time.monotonic() + dur
            elif kind == 'old': s.cam.old_until = time.monotonic() + dur
            elif kind == 'raise': raise RuntimeError('forced program error (dry mode only)')
            elif kind == 'sigterm': threading.Thread(target=signal.raise_signal, args=(signal.SIGTERM,), daemon=True).start()
            elif kind == 'sigint': threading.Thread(target=signal.raise_signal, args=(signal.SIGINT,), daemon=True).start()
            elif kind == 'hang': time.sleep(dur)
            elif kind in ('bridgekill', 'homebridgekill'): s.br.p.kill()   # v27b
            elif kind == 'stopfail': s.tap.dry_fail = 1; s.cam.freeze_until = time.monotonic() + dur   # v27b
            elif kind == 'nomatch': s.cam.noise_until = time.monotonic() + dur   # v27b

    def snap(s, tag):
        """picture of the newest camera frame, to see afterwards WHY the match was lost (shutter correction? a person? something else?)"""
        try:
            c = s.cam.cur
            if c is not None: cv2.imwrite(os.path.join(s.a.out, 'lost_%02d_%s_f%d.png' % (s.nlost, tag, c[0])), cv2.normalize(c[2], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        except Exception: pass

    def frame(s):
        try:
            if s.a.dry and s.step_i in FORCE and s.step_i not in s.forced and (s.in_run or s.name == 'static') and time.monotonic() - s.t_step > FORCE[s.step_i]:
                s.forced.add(s.step_i); raise H.Abort('image shift lost (forced on purpose, dry mode only)')
            r = super().frame()
        except H.Abort as e:
            if str(e).startswith('image shift lost'): s.nlost += 1; s.snap('a_lost')
            raise
        if s.key0 is None and s.est.key is not None: s.key0 = s.est.key.copy()      # the START view, kept for the whole run
        return r

    def track(s, st, home=False):
        s.lead_now = bool(st.get('lead')) and not home; CTL.update(lv=None, lead_px=0.0, W=None, lead_ms=int(st.get('lead_ms', 150))); s.prevS = None
        s.in_run = not home; s.moved = True
        if not home: s.t_step = time.monotonic()
        else: s.t_home = time.monotonic()             # v27
        try: return super().track(st, home)
        finally: s.lead_now = False; s.in_run = False

    def static_step(s, st):
        """a static step that loses the match before the gimbal has ever moved: the start view itself was bad (taken before the camera's
        shutter correction) -> wait 3 s, take a NEW start view, repeat the step. After any motion this is not allowed (the start pose is the reference)."""
        for k in range(3):
            try: s.t_step = time.monotonic(); return s.static(st)
            except TypeError: raise H.Abort('camera silent for 2 s (while waiting for the camera to settle)')
            except H.Abort as e:
                if not str(e).startswith('image shift lost') or s.moved or k == 2: raise
                print('  start view not steady (%s): waiting 3 s, then a new start view (retry %d of 2)' % (e, k + 1), flush=True); t0 = time.monotonic()
                while time.monotonic() - t0 < 3.0:
                    fr = s.cam.get(s.last, 2.0)
                    if fr is None: raise H.Abort('camera silent for 2 s (while waiting for a steady start view)')
                    s.last = fr[0]
                s.snap('d_newstart'); s.est = H.Shift(); s.key0 = None; s.S = np.zeros(2)
                s.same = 0; s.prev_m = None            # v27: a frozen picture before any motion cut the command line (pid_hil.py); open it again for the new start view
                if not s.tap.open_line(): raise H.Abort('the command line stays cut (watchdog stop or a STOP that could not be written)')   # v27b: was s.tap.block = False

    def feed(s, fid, ns, pw, vis):
        xi, yi = H.W / 2 + pw[0] + s.S[0], H.H / 2 + pw[1] + s.S[1]
        bx = s.box(xi, yi) if vis and 0 <= xi < H.W and 0 <= yi < H.H else None; inside = bx is not None   # v27b: the box of this step, cut at the picture's edge (pid_hil.py)
        lat = ns + int(s.a.latency_ms * 1e6) - time.monotonic_ns()
        if lat > 0: time.sleep(lat / 1e9)
        dS = (s.S - s.prevS) if s.prevS is not None else None; s.prevS = s.S.copy(); sx = sy = 0.0
        if inside:
            box = (H.TID, bx[0], bx[1], bx[2], bx[3], 0.9); out = box   # v27b
            if s.lead_now:
                CTL['W'] = None if dS is None else [[1.0, 0.0, float(dS[0])], [0.0, 1.0, float(dS[1])]]
                out = lead_shift([box], [box], H.TID, ns / 1e9, float(H.W), float(H.H))[0]; sx, sy = out[1] - box[1], out[2] - box[2]
            s.br.send('F %d %d 1 %d %.2f %.2f %.2f %.2f 0.900' % (fid, ns, H.TID, out[1], out[2], out[3] - out[1], out[4] - out[2]))   # v27b: width and height of the (cut) box
        else:
            if s.lead_now: lead_shift([], [], H.TID, ns / 1e9, float(H.W), float(H.H))
            s.br.send('F %d %d 0' % (fid, ns))
        s.llog.write('%d,%d,%.2f,%.2f\n' % (fid, int(s.lead_now), sx, sy))
        return (xi, yi), ((xi - H.W / 2) / (H.W / 2), (yi - H.H / 2) / (H.H / 2)), int(bool(inside))

    def rereg(s, raw):
        """scene displacement of this frame against the START view: 15 patches of 60x50 half-resolution pixels searched in the whole frame"""
        f = s.est.prep(raw); res = []
        for y0 in (20, 95, 170):
            for x0 in (10, 70, 130, 190, 250):
                r = cv2.matchTemplate(f, s.key0[y0:y0 + 50, x0:x0 + 60], cv2.TM_CCOEFF_NORMED); _, mx, _, (px, py) = cv2.minMaxLoc(r)
                if mx >= PATCH_MIN: res.append((px - x0, py - y0))
        if len(res) < AGREE: return None, len(res), f
        p = np.array(res, float); m = max(((np.abs(p - c).max(1) <= 3) for c in p), key=lambda v: int(v.sum()))
        if int(m.sum()) < AGREE: return None, int(m.sum()), f
        return p[m].mean(0) * 2.0, int(m.sum()), f

    def recover(s, why):
        """gimbal stopped and the command line cut; nothing moves again before the start view has been found twice in a row"""
        s.tap.estop()                                     # stop command straight to the device and the command line cut, before anything else
        try: s.close_bridge()
        except Exception: pass
        s.br = None; time.sleep(0.15); s.tap.estop(); s.name = 'recover'; s.proc = 0.0; t0 = time.monotonic(); prev = None; was = s.S.copy()
        print('  LOST: %s -> gimbal stopped, looking for the start view' % why, flush=True); marks = [(0.5, 'b_0.5s'), (1.0, 'c_1.0s')]
        if s.tap.stop_failed: raise H.Abort('EMERGENCY STOP NOT WRITTEN to the gimbal device: the run ends, the gimbal may still be moving')   # v27b
        if s.key0 is None: raise H.Abort('image shift lost before a start view existed')
        while time.monotonic() - t0 < 20.0:
            fr = s.cam.get(s.last, 2.0)
            if fr is None: raise H.Abort('camera silent for 2 s (while looking for the start view)')
            s.last = fr[0]; t = time.monotonic() - t0
            while marks and t >= marks[0][0]: s.snap(marks.pop(0)[1])
            if t < 1.5 or (prev is not None and t - prev[0] < 0.3): continue
            D, n, f = s.rereg(fr[2]); s.row('recover', t, fr[0], fr[1], D if D is not None else was, n / 15.0)
            if D is None: prev = None; continue
            if prev is not None and np.array_equal(fr[2], prev[2]): continue   # v27: the picture is still frozen: two equal pictures prove nothing, keep waiting
            if prev is not None and np.abs(D - prev[1]).max() <= 4.0:
                if abs(D[0]) > s.a.limx or abs(D[1]) > s.a.limy: raise H.Abort('excursion limit after the loss: scene displacement %.0f, %.0f px' % (D[0], D[1]))   # v27b: the limits of this run (were the built-in numbers)
                s.est.key = f; s.est.off = D.copy(); s.est.S = D.copy(); s.est.bad = 0; s.S = D.copy()
                if not s.tap.open_line(): raise H.Abort('the command line stays cut (%s)' % (s.tap.trip or 'a STOP could not be written'))   # v27b: was s.tap.block = False
                true = getattr(s.cam, 'pos', None)
                print('  FOUND after %.1f s: scene displacement %.1f, %.1f px (%d of 15 patches agree; before the loss %.1f, %.1f%s)' % (
                    t, D[0], D[1], n, was[0], was[1], '' if true is None else '; dry truth %.1f, %.1f' % (true[0], true[1])), flush=True)
                s.snap('d_found'); s.same = 0; s.prev_m = None; return   # v27: + frozen-picture counter cleared
            prev = (t, D, fr[2])                       # v27: + the picture itself
        s.snap('e_notfound'); raise H.Abort('image shift lost and the start view was not found again in 20 s')

    def after_stop(s):
        """v27b: after a STOP that was not caused by a lost picture (a block ended as "person lost"): close the bridge, watch 0.6 s of pictures with the line cut, open the line."""
        try: s.close_bridge()
        except Exception: pass
        s.br = None; s.name = 'rest'; t0 = time.monotonic()
        while time.monotonic() - t0 < 0.6:
            fid, ns, S, resp = s.frame(); s.row('rest', time.monotonic() - t0, fid, ns, S, resp)
        if not s.tap.open_line(): raise H.Abort('the command line stays cut (%s)' % (s.tap.trip or 'a STOP could not be written'))

    # ---------------- v27b: stage 1, identification by short timed moves ----------------
    def idrow(s, **k):
        s.idf.write(','.join(str(k.get(f, '')).replace(',', ';') for f in ID_FIELDS) + '\n'); s.idf.flush()   # no commas inside a field

    def at_rest(s, k, t0, phase, least, most):
        """pictures until the view stands still (4 pictures in a row less than 0.4 px apart) and at least <least> s have passed; at most <most> s. Returns the mean of the last 3 good values."""
        a = time.monotonic(); v = []; still = 0; prev = None; tr = []
        while True:
            fid, ns, S, resp = s.frame(); s.row(phase, time.monotonic() - t0, fid, ns, S, resp); el = time.monotonic() - a; tr.append((ns, float(S[k]), s.est.bad))
            if s.est.bad == 0:
                still = still + 1 if prev is not None and abs(S[k] - prev) < 0.4 else 0; prev = float(S[k]); v.append(prev)
            if (still >= 4 and el >= least and len(v) >= 3) or el > most: break
        return (float(np.mean(v[-3:])) if v else float(s.S[k])), tr

    def one_pulse(s, k, dr, T, t0):
        """one timed move: the tap thread starts the clock when the command reaches the gimbal line and writes the STOP itself T seconds later."""
        s._p = None; S0, _ = s.at_rest(k, t0, 'rest', 0.25, 1.5); tap = s.tap; tap.t_pvel = tap.t_pstop = None; tap.arm = T; s._p = dict(S0=S0, tr=[])
        s.br.send(('V %d 0' if k == 0 else 'V 0 %d') % dr); a = time.monotonic(); tr = s._p['tr']
        while tap.t_pstop is None and time.monotonic() - a < T + 1.0:
            fid, ns, S, resp = s.frame(); s.row('pulse', time.monotonic() - t0, fid, ns, S, resp); tr.append((ns, float(S[k]), s.est.bad))
        if tap.t_pvel is None: tap.arm = None; s.br.send('V 0 0'); tap.estop(); raise H.Abort('stage 1: the move command did not reach the gimbal line within 1 s (STOP sent)')
        if tap.t_pstop is None: tap.estop(); raise H.Abort('stage 1: the timed move did not end in time (STOP sent)')
        s.br.send('V 0 0'); S1, tr2 = s.at_rest(k, t0, 'coast', 0.3, 1.5); tr += tr2
        if not tap.open_line(): raise H.Abort('the command line stays cut (%s)' % (tap.trip or 'a STOP could not be written'))
        return S0, S1

    def pulse_stats(s, S0):
        """first motion and highest speed between two pictures, from the pictures of the last move"""
        tr = [p for p in s._p['tr'] if p[2] == 0]; t0 = s.tap.t_pvel; lo = hi = None; peak = 0.0
        for i, (ns, v, _) in enumerate(tr):
            if ns > t0 and abs(v - S0) > 2.0: hi = (ns - t0) / 1e6; lo = max(0.0, (tr[i - 1][0] - t0) / 1e6) if i else None; break
        for (n0, v0, _), (n1, v1, _) in zip(tr, tr[1:]):
            if n1 - n0 > 5e6 and n1 > t0: peak = max(peak, abs(v1 - v0) / ((n1 - n0) / 1e9))
        return lo, hi, peak, len(tr)

    def pulse(s, st):
        ax = st['axis']; sp = int(st['speed']); k = 0 if ax == 'x' else 1; B = float(st['budget']); pairs = int(st.get('pairs', ID_PAIRS)); s.name = 'pulse_%s_%d' % (ax, sp)
        m = s.idm.setdefault(ax, dict(v=ID_PRIOR, spd=1, stop='')); base = dict(step=s.step_i, axis=ax, spd=sp)
        if m['stop']: s.idrow(status='skip', note=m['stop'], **base); print('  SKIP SPD %d on %s: %s' % (sp, ax, m['stop']), flush=True); return
        v_exp = max(m['v'], ID_FLOOR) * sp / m['spd']; v_up = ID_PROBE * v_exp; t0 = time.monotonic()   # the first move of a command is sized for a gimbal ID_PROBE x faster than expected
        if B / v_up - ID_JIT < ID_TMIN:
            m['stop'] = 'SPD %d and above left out: expected up to %.0f px/s (from SPD %d), so even a %.0f ms move could pass the %.0f px budget' % (sp, v_up, m['spd'], 1e3 * (ID_TMIN + ID_JIT), B)
            s.idrow(status='skip', v_expected='%.0f' % v_exp, v_plan='%.0f' % v_up, note=m['stop'], **base); print('  SKIP %s' % m['stop'], flush=True); return
        s.tap.guard_y = None                           # v27c: no up-down guard in a stage 1 move (its guard is the move's own command value)
        new = sp >= ID_NEW.get(ax, 10 ** 9)            # v27c: never measured on this gimbal
        s.tap.guard = min(sp, s.a.spd_abort); s.est.gate = ID_GATE; s.open_bridge(['--kp', '30.0', '--manual-speed', str(sp)]); v_own = None; tmax = 0.0; best = 0.0; moved = False; tried = 0
        print('  SPD %d on %s: expected %.0f px/s (sized for %.0f), budget %.0f px' % (sp, ax, v_exp, v_up, B), flush=True)
        try:
            for j in range(pairs):
                first = None
                for h in (0, 1):
                    sg = s.idsgn.get(ax); cur = float(s.S[k])
                    if h == 1: dr = -first
                    elif sg is not None and abs(cur) > 10: dr = int(-np.sign(cur) * sg)       # the first move of a pair goes towards the start view
                    else: dr = 1 if j % 2 == 0 else -1
                    if h == 0: first = dr
                    outward = sg is None or (sg * dr) * cur > 0; room = B - abs(cur) if outward else B
                    Tp = room / v_up - ID_JIT                                                    # the probe length: safe for a gimbal up to ID_PROBE x the expectation
                    Tp0 = min(Tp, ID_TMIN) if new else Tp                                         # v27c (second review): a command never measured on this gimbal starts with the shortest move, whatever is expected
                    if v_own is None: T = Tp0 if tried == 0 else min(2 * Tp0, room / (ID_UP * v_exp) - ID_JIT)   # no clear motion yet: at most twice the probe length   # v27c: Tp0 (was Tp)
                    else: T = min(room / (ID_UP * best) - ID_JIT, max(Tp, min(room / (ID_UP * v_own) - ID_JIT, ID_GROW * tmax)))   # longer only as far as this command's own measured moves allow
                    T = min(T, ID_TMAX); v_plan = room / (T + ID_JIT); tried += 1; row = dict(base, pair=j, dir=dr, v_expected='%.0f' % v_exp, v_plan='%.0f' % v_plan, room_px='%.0f' % room, T_plan_ms='%.0f' % (1e3 * T))
                    if T < ID_TMIN: s.idrow(status='skip-short', note='no room for a move of %.0f ms or more' % (1e3 * ID_TMIN), **row); continue
                    try: S0, S1 = s.one_pulse(k, dr, T, t0); lost = False
                    except H.Abort as e:
                        if not str(e).startswith('image shift lost') or s._p is None: raise   # a loss before the move began is handled by the caller (found again, home)
                        s.idlost += 1; S0 = s._p['S0']; ta = ((s.tap.t_pstop or time.monotonic_ns()) - s.tap.t_pvel) / 1e9 if s.tap.t_pvel else T
                        s.recover(str(e)); S1 = float(s.S[k]); lost = True          # gimbal stopped, bridge closed, the view found again against the START view on a live picture
                    D = S1 - S0; ta = ta if lost else (s.tap.t_pstop - s.tap.t_pvel) / 1e9; vp = abs(D) / max(ta, 1e-3); lo, hi, peak, nf = (None, None, 0.0, 0) if lost else s.pulse_stats(S0)
                    row.update(status='lost' if lost else 'ok', T_act_ms='%.1f' % (1e3 * ta), travel_px='%.1f' % D, v_pulse='%.0f' % vp, v_peak='%.0f' % peak, delay_lo_ms='' if lo is None else '%.0f' % lo, delay_hi_ms='' if hi is None else '%.0f' % hi,
                               s_before='%.1f' % S0, s_after='%.1f' % S1, frames=nf, note='picture matching lost during the move: travel from the start-view match' if lost else '')
                    s.idrow(**row); print('    %s %+d  %.0f ms -> %+.1f px = %.0f px/s (first motion seen %s ms, fastest between two pictures %.0f px/s)%s' % (ax, dr, 1e3 * ta, D, vp, '-' if hi is None else '%.0f' % hi, peak, ' LOST' if lost else ''), flush=True)
                    if abs(D) >= ID_MOVE:
                        moved = True; best = max(best, vp); own = abs(D) / max(ta - ID_DELTA, 0.25 * ta)
                        if v_own is None or ta >= tmax - 0.005: v_own = own if (v_own is None or ta > tmax + 0.005) else max(v_own, own)   # the longest move so far gives this command's own speed (the larger value of equally long moves)
                        tmax = max(tmax, ta)
                        if ax not in s.idsgn: s.idsgn[ax] = 1 if D * dr > 0 else -1
                    if abs(D) / (ta + 0.05) > ID_FAR * v_exp or abs(D) > room + 0.25 * B:
                        s.tap.estop(); raise H.Abort('stage 1 stop rule: SPD %d on %s moved %.0f px in %.0f ms (%.0f px/s); expected %.0f px/s, room %.0f px' % (sp, ax, abs(D), 1e3 * ta, vp, v_exp, room))
                    if lost:
                        m['stop'] = 'SPD %d and above left out: the picture matching could not follow SPD %d' % (sp, sp)
                        if s.idlost > 3: raise H.Abort('stage 1: the picture matching was lost in %d moves' % s.idlost)
                        break
                if m['stop']: break
        finally: s.est.gate = H.GATE_PX
        if s.br is not None: s.close_bridge()
        if moved: m.update(v=best, spd=sp)
        else: s.idrow(status='note', note='no clear motion in the allowed move time: the next command is expected from SPD %d as before' % m['spd'], **base)

    @staticmethod
    def walk_speed(st):
        """v27c: the fastest leg of the virtual person's path in this block, px/s (from the way points of the plan)"""
        wp = st.get('wp') or []
        return max([max(abs(b[1] - a[1]), abs(b[2] - a[2])) / (b[0] - a[0]) for a, b in zip(wp, wp[1:]) if b[0] > a[0]] or [0.0])

    def run(s, plan):
        total = 0; wlost = {}; wskip = {}              # v27c: picture-matching losses inside the following blocks of each fast walk; walks whose remaining blocks are skipped
        for i, st in enumerate(plan):
            s.step_i = i; s.t_step = time.monotonic(); print('STEP %d %s' % (i, json.dumps(st)[:160]), flush=True)
            if st['type'] == 'static': s.static_step(st)
            elif st['type'] == 'ident': s.moved = True; s.ident(st); s.home()
            elif st['type'] == 'peek': s.moved = True; s.peek(st)
            elif st['type'] == 'pulse':                # v27b: stage 1
                s.moved = True
                for att in range(3):
                    try:
                        if att == 0: s.pulse(st)
                        s.home(); break
                    except H.Abort as e:               # a lost picture inside a move is handled in pulse(); this is a loss between the moves or on the way home
                        if not str(e).startswith('image shift lost') or total >= MAX_REPEAT or att == 2: raise
                        total += 1; print('  REPEAT %d of %d (stage 1, on the way home)' % (total, MAX_REPEAT), flush=True); s.est.gate = H.GATE_PX; s.recover(str(e))
            elif st['type'] == 'track' and (st.get('walk') or st['name'].split('_')[0]) in wskip:   # v27c (second review): this walk lost the picture matching twice: its remaining blocks are not run
                s.blog.write('%d,%s,%d,skipped,%s\n' % (i, st['name'], int(bool(st.get('lead'))), st.get('bridge', 'v25'))); s.blog.flush()
                print('  SKIPPED block %s: %s' % (st['name'], wskip[st.get('walk') or st['name'].split('_')[0]]), flush=True)
            elif st['type'] == 'track':
                att, done, need_home = 1, False, False
                wk = st.get('walk') or st['name'].split('_')[0]; fast = s.walk_speed(st) >= FAST_WALK - 0.5   # v27c: the walk this block belongs to (stage 3: its "walk" entry; stage 2: the first word of its name); is it a fast walk?
                while True:
                    cur = dict(st, name=st['name'] + ('' if att == 1 else '_try%d' % att)); started = False
                    try:
                        if not done:
                            if need_home: s.home(); need_home = False
                            started = True; s.track(cur); done = True
                            s.blog.write('%d,%s,%d,ok,%s\n' % (i, cur['name'], int(bool(st.get('lead'))), st.get('bridge', 'v25'))); s.blog.flush()
                        if not st.get('nohome'): s.home()
                        break
                    except H.ErrLimit as e:            # v27b: the block ended as "person lost" (gimbal stopped first). Recorded, not repeated; the run goes on unless it happened too often
                        s.nerrlim += 1; done = True; s.blog.write('%d,%s,%d,personlost,%s\n' % (i, cur['name'], int(bool(st.get('lead'))), st.get('bridge', 'v25'))); s.blog.flush()
                        print('  PERSON LOST (%d of at most %d in a run): %s' % (s.nerrlim, H.MAX_ERRLIM, e), flush=True)
                        if s.nerrlim > H.MAX_ERRLIM: raise H.Abort('%d blocks ended as person lost (more than %d): the run ends. Last: %s' % (s.nerrlim, H.MAX_ERRLIM, e))
                        s.after_stop()
                    except H.Abort as e:
                        lost = str(e).startswith('image shift lost'); mid = started and not done
                        if lost and mid: s.blog.write('%d,%s,%d,lost,%s\n' % (i, cur['name'], int(bool(st.get('lead'))), st.get('bridge', 'v25'))); s.blog.flush()
                        if not lost or total >= MAX_REPEAT or (mid and att >= MAX_TRY): raise
                        total += 1; att += int(mid); skip_now = False
                        if mid and fast:               # v27c (second review): a fast walk gets one repeat; its second loss ends the walk (the gimbal is stopped, the start view is found again, home, then the next walk)
                            wlost[wk] = wlost.get(wk, 0) + 1
                            if wlost[wk] >= FAST_WALK_LOST:
                                wskip[wk] = 'the picture matching was lost %d times in the following blocks of the walk %s (%.0f px/s): the rest of this walk is left out' % (wlost[wk], wk, s.walk_speed(st)); done = True; skip_now = True
                                print('  WALK SKIPPED (recovery %d of %d): %s' % (total, MAX_REPEAT, wskip[wk]), flush=True)
                        if not skip_now: print('  REPEAT %d of %d (block %s, next attempt %d)' % (total, MAX_REPEAT, st['name'], att), flush=True)   # v27c: + the condition
                        need_home = True; s.recover(str(e))
            s.f.flush(); s.slog.flush(); print('  end: scene displacement %.1f, %.1f px, serial commands %d' % (s.S[0], s.S[1], s.tap.n), flush=True)

    def where(s):
        """v27: called after the last STOP. Where does the camera look now, against the START view? Written to end_state.json and printed.
        sx > 0: the picture content sits further right than at the start; sy < 0: the camera looks LOWER than at the start (2026-11-11 morning: -340)."""
        out = dict(why=s.why, step=s.step_i, name=s.name, last_measured_px=[round(float(s.S[0]), 1), round(float(s.S[1]), 1)],
                   last_good_measurement_age_s=None if s.meas_t is None else round(time.monotonic() - s.meas_t, 2), highest_spd=s.tap.maxspd,
                   stops_on_device_line=s.tap.nstop, lost=s.nlost, end_view_px=None, end_view_patches=0, limits_px=[s.a.limx, s.a.limy],
                   last_stop_written=not s.tap.stop_failed, stops_not_written=s.tap.nfail, device_opened_again=s.tap.nreopen, same_while_zero=s.nsame0, person_lost_blocks=s.nerrlim, spd_abort=s.a.spd_abort)   # v27b
        t0 = time.monotonic(); prev = None; p0 = None; moving = None; lastpic = None   # v27c: + moving, lastpic
        while s.key0 is not None and time.monotonic() - t0 < 5.0:
            fr = s.cam.get(s.last, 1.0)
            if fr is None: break
            s.last = fr[0]; t = time.monotonic() - t0
            live = lastpic is not None and not np.array_equal(fr[2], lastpic); lastpic = fr[2]   # v27c: a picture identical to the one right before it comes from a frozen camera. It shows where the gimbal looked earlier, not now: an old picture followed by a live one must not count as "the view moved" (PC check: 3 false warnings of 3 without this line, dry3/n_w2)
            if t < 0.8 or (prev is not None and t - prev[0] < 0.3): continue      # let the gimbal come to rest first
            if not live: continue                      # v27c: only live pictures are used below (also for the end view)
            if p0 is None and hasattr(s.cam, 'pos'): p0 = s.cam.pos.copy()
            D, n, f = s.rereg(fr[2])
            if D is None: prev = None; continue
            if prev is not None and not np.array_equal(fr[2], prev[2]) and np.abs(D - prev[1]).max() <= 4.0:
                out.update(end_view_px=[round(float(D[0]), 1), round(float(D[1]), 1)], end_view_patches=n); break
            if prev is not None and not np.array_equal(fr[2], prev[2]):          # v27c (second review): two live views, 0.3 s or more apart and more than 0.8 s after the last STOP, differ by more than 4 px: the gimbal did not obey the STOP
                dd = float(np.abs(D - prev[1]).max())
                if moving is None:
                    print('WARNING: THE VIEW IS STILL MOVING AFTER THE LAST STOP: %.0f px between two views %.1f s apart (%.1f s after the STOP). STOP is written once more. GO TO THE GIMBAL.' % (dd, t - prev[0], t), flush=True)
                    s.tap.estop()
                moving = max(moving or 0.0, dd)
            prev = (t, D, fr[2])
        s.moving_after_stop = moving; out.update(view_moving_after_stop=moving is not None, view_moving_after_stop_px=None if moving is None else round(moving, 1))   # v27c: pid_hil.py main() ends with exit code 7 when this is set
        s.end_view_checked = out['end_view_px'] is not None; out.update(end_view_checked=s.end_view_checked)   # v27d (third review, minor 1): the end view could not be matched to the start view = nobody knows whether the gimbal stands (a gimbal that ignores the STOP at 800 px/s is out of the start view after 0.8 s). pid_hil.py main(): loud line, exit code 7; make_plan_v27.py: no next plan
        if not s.end_view_checked: print('WARNING: END VIEW NOT CHECKED - LOOK AT THE GIMBAL. The view after the last STOP could not be matched to the start view, so the tool cannot tell whether the gimbal stands still.', flush=True)   # v27d
        if hasattr(s.cam, 'pos'): out.update(dry_truth_px=[round(float(s.cam.pos[0]), 1), round(float(s.cam.pos[1]), 1)], dry_moved_after_stop_px=None if p0 is None else round(float(np.abs(s.cam.pos - p0).max()), 2), dry_last_command=list(s.tap.last))
        json.dump(out, open(os.path.join(s.a.out, 'end_state.json'), 'w'), indent=1)
        print('END VIEW: %s | last measured %s px | matched against the start view %s px (%d of 15 patches)%s%s' % (
            s.why, out['last_measured_px'], out['end_view_px'], out['end_view_patches'], '' if moving is None else ' | STILL MOVING AFTER THE LAST STOP (%.0f px between two views)' % moving, '' if 'dry_truth_px' not in out else ' | dry truth %s, moved after the STOP %s px, last command %s' % (out['dry_truth_px'], out['dry_moved_after_stop_px'], out['dry_last_command'])), flush=True)

    def close(s):
        s.tap.estop()                                  # v27: STOP first
        try:                                           # v27: then the bridge is closed and the end view recorded (nothing can move: the command line stays cut)
            try: s.close_bridge()
            except Exception: pass
            s.br = None
            if not s.llog.closed: s.where()
        except BaseException as e: s.tap.estop(); print('end view not recorded: %r' % (e,), flush=True)
        r = super().close(); s.llog.close(); s.blog.close(); s.idf.close()   # v27b: + idf
        if ERR: print('LEAD errors:', ERR[:3], flush=True)
        return r


H.Session = S2
if __name__ == '__main__': H.main()
