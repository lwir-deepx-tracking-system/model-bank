#!/usr/bin/env python3
"""Reads serial.jsonl of one run folder and says whether the gimbal line ended with a STOP (standard library only, reads only).
Usage:  python3 check_stop.py <run folder> [<run folder> ...]
A command marked "blocked" never reached the gimbal line (the tool had cut the line); one marked "werr" was not written (the device refused it).
"estop" = the tool's own STOP written straight to the device; "pulse_stop" = the STOP that ends a timed move of stage 1.
Second round (2026-11-11, review): an estop whose record says ok 0 was NOT written and does not count as a stop.
PASS = after the last velocity command that reached the line there is a STOP that was really written (the bridge's or the tool's)."""
import json, os, sys


def check(D):
    moving = False; nvel = nstop = nestop = nblocked = nfail = nwerr = npulse = nold = 0; last_vel = None; stop_after = None; whys = []; reopened = []; last_estop_ok = None
    for line in open(os.path.join(D, 'serial.jsonl')):
        try: d = json.loads(line)
        except ValueError: continue
        if 'estop' in d or 'pulse_stop' in d:
            if 'ok' not in d: nold += 1                      # a record of the first version: no word on the write
            written = bool(d.get('ok', 1))
            if 'estop' in d:
                last_estop_ok = written
                if d.get('why') and d['why'] not in whys: whys.append(d['why'])
                if d.get('reopened'): reopened.append(d['reopened'])
            if not written: nfail += 1; continue             # a STOP that was not written is not a stop
            if 'estop' in d: nestop += 1
            else: npulse += 1
            if moving and stop_after is None: stop_after = ('tool STOP', (d['ns'] - last_vel) / 1e6)
            moving = False
        elif 'cmd' in d:
            c = d['cmd']
            if d.get('blocked'): nblocked += 1; continue
            if d.get('werr'): nwerr += 1; continue
            if c.get('T') == 141 and c.get('SPD', 0) > 0 and (c.get('X', 0) or c.get('Y', 0)): moving = True; nvel += 1; last_vel = d['ns']; stop_after = None
            elif c.get('T') == 135 or c.get('T') == 141:
                nstop += 1
                if moving and stop_after is None: stop_after = ('bridge STOP', (d['ns'] - last_vel) / 1e6)
                moving = False
    es = None
    try: es = json.load(open(os.path.join(D, 'end_state.json')))
    except Exception: pass
    ok = not moving
    print('%s  %s' % ('PASS' if ok else 'FAIL', D))
    print('   velocity commands on the line %d, bridge stops %d, tool stops %d (+ %d that end a timed move), commands held back after the line was cut %d' % (nvel, nstop, nestop, npulse, nblocked))
    if nfail or nwerr: print('   NOT WRITTEN: %d tool stops, %d bridge commands (the device refused the write)%s' % (nfail, nwerr, '' if ok else ' -> the last velocity command was NOT followed by a written STOP'))
    if reopened: print('   the gimbal device was opened again for a STOP: %s' % ', '.join(reopened))
    if last_estop_ok is False: print('   WARNING: the tool\'s LAST emergency stop was not written')
    if nold: print('   note: %d stop records without a word on the write (old record format): counted as written' % nold)
    if stop_after: print('   the last velocity command was followed by a %s after %.0f ms' % stop_after)
    if whys: print('   watchdog reasons: %s' % '; '.join(whys))
    if es: print('   end: %s | end view %s px | last measured %s px%s' % (es.get('why'), es.get('end_view_px'), es.get('last_measured_px'),
                 '' if 'dry_truth_px' not in es else ' | dry truth %s px, moved after the STOP %s px, last command %s' % (es['dry_truth_px'], es.get('dry_moved_after_stop_px'), es.get('dry_last_command'))))
    return ok


if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit(__doc__)
    r = [check(d) for d in sys.argv[1:]]
    print('ALL PASS' if all(r) else 'SOME FAILED'); sys.exit(0 if all(r) else 1)
