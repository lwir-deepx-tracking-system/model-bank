#!/usr/bin/env python3
"""Measure STOP submission using a PTY: no motor/camera is opened.

Times are host protocol-to-serial times, NOT physical servo stopping times.
Usage: python3 test_v27_disappearance.py ./gimbal_bridge_v27
"""
import json
import time
import test_gimbal_bridge as t
import test_v27_lead_protocol as p


def start():
    r = t.Rig('--deadband', '.06', '--controller', 'p', '--kp', '30',
              '--aim-y', '.5', '--input-timeout-ms', '250', '--reacquire-ms', '1000')
    assert r.serial(1) == t.STOP
    assert r.wait_line('LEAD_CONTROL ') == 'LEAD_CONTROL 1'
    p.feed(r, 'F', seconds=.1)
    epoch = p.select(r)
    commands = p.feed(r, 'F', ' LEAD 7 60 0 %d' % epoch, seconds=.4)
    assert commands and max(c['SPD'] for c in commands) >= 11
    # Clear bytes already emitted before the disappearance event.
    while r.serial(.003) is not None: pass
    return r, epoch


def frame(r, mode, rows, epoch, stamp=None):
    r.frame_id += 1
    ns = time.monotonic_ns() if stamp is None else stamp
    prefix = '%s %d %d' % (mode, r.frame_id, ns)
    if mode == 'G': prefix += ' 640 480'
    body = ''.join(' %d %.2f %.2f 40 80 .9' % (i, x-20, y-40) for i,x,y in rows)
    r.proc.stdin.write(('%s %d%s LEAD 7 60 0 %d\n' % (prefix,len(rows),body,epoch)).encode())
    if mode == 'G': r.proc.stdin.write(bytes(640*480*2))


def wait_stop(r, before, maximum_ms):
    while (time.monotonic()-before)*1000 < maximum_ms:
        c = r.serial(.01)
        if c == t.STOP: return (time.monotonic()-before)*1000
    raise AssertionError('STOP missing within %d ms' % maximum_ms)


def vanished(mode, other_person):
    r, epoch = start()
    try:
        before = time.monotonic()
        frame(r, mode, [(8,384,240)] if other_person else [], epoch)
        ms = wait_stop(r,before,100)
        # State and generation notifications can arrive in either order.
        # Consume together: waiting for one prefix must not discard the other.
        lost = False
        loss_epoch = epoch
        deadline = time.monotonic()+.25
        while time.monotonic()<deadline and (not lost or loss_epoch==epoch):
            r.send('P')
            line = r.line(.03)
            if line and line.startswith('CONTROL_EPOCH '):
                loss_epoch = max(loss_epoch,int(line.split()[1]))
            elif line and line.startswith('S '):
                values=list(map(int,line.split()[1:6]))
                lost = values==[1,1,0,0,7] or lost
        assert lost and loss_epoch!=epoch, (lost,loss_epoch,epoch)
        # Nearby other IDs must not inherit the selected target or its forecast.
        got = r.feed(.15,(8,384,240))
        assert not [c for c in got if c['T']==141], got
        centered = p.feed(r,mode,' LEAD 7 60 0 %d' % epoch,seconds=.1,x=320,y=240)
        assert not centered, 'old prediction moved a reappeared centered target'
        # Same-ID return can resume, but the old prediction must be rejected.
        moves = p.feed(r,mode,' LEAD 7 60 0 %d' % epoch,seconds=.2)
        assert moves and max(c['SPD'] for c in moves) <= 6, moves
        return dict(case=mode+('_other_id' if other_person else '_empty'),stop_ms=ms,
                    stale_forecast_rejected=True,other_id_not_followed=True)
    finally: r.close()


def frozen_or_future(future):
    r,epoch=start()
    try:
        before=time.monotonic()
        # Ping continues. Only the image stream is stale, so viewer timeout
        # cannot be the reason this test passes.
        r.send('P')
        if future: frame(r,'F',[(7,384,240)],epoch,time.monotonic_ns()+1000000000)
        ms=wait_stop(r,before,100 if future else 350)
        got=[]
        end=time.monotonic()+.2
        while time.monotonic()<end:
            r.send('P')
            c=r.serial(.02)
            if c is not None: got.append(c)
        assert not [c for c in got if c['T']==141],got
        return dict(case='future_timestamp' if future else 'camera_stalled_ping_alive',stop_ms=ms)
    finally: r.close()


def repeated_capture_time():
    r,epoch=start()
    try:
        stamp=time.monotonic_ns()
        before=time.monotonic()
        stopped=None
        while (time.monotonic()-before)<.35:
            r.send('P')
            frame(r,'F',[(7,384,240)],epoch,stamp)
            command=r.serial(.01)
            if command==t.STOP:
                stopped=(time.monotonic()-before)*1000
                break
        assert stopped is not None,'new frame numbers refreshed a stale image timestamp'
        return dict(case='new_frame_id_same_capture_time',stop_ms=stopped)
    finally: r.close()


if __name__ == '__main__':
    results=[vanished(mode,other) for mode in ('F','G') for other in (False,True)]
    results.extend(frozen_or_future(future) for future in (False,True))
    results.append(repeated_capture_time())
    print(json.dumps(results,indent=2))
    print('ALL OK: protocol STOP only; no physical stopping-time claim')
