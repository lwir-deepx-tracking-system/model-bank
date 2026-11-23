"""Approaching-person tilt simulation for close_aim (PC only, no gimbal). Run from tracking_server: python src/gimbal/sim/close_aim_sim.py
A rough model, not the bridge code: it shows the trend only.

Person walks toward the camera: head-to-hip box height grows from 25 % to 130 % of the picture in 6 s, then stays 3 s.
Camera at chest height: the point 0.35 of the box from its top stays at the world horizon (row 240 when tilt = 0).
Detector: box = visible part (clipped to the picture); no box when the head (top 12 % of the real box) is above the picture top.
Tilt: 100 ms delay, rate = 900 px/s x error (error in half-frame units), deadband 0.06 / exit 0.10, clamp 300 px/s (speed 100-ish).
Server: aim rows through close_aim.apply exactly as live_tracks.publish does (bridge --aim-y 0.5, --aim-screen-y 0.5).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))   # tracking_server
from src.gimbal import close_aim as CA

H = 480.; FPS = 30.; DT = 1 / FPS


def run(on, anchor=0.35):
    CA.STATE['on'] = on; CA.reset()
    c = 0.; hist = []; lost = 0; first_lost = None; hold = False; min_head = 1e9
    for k in range(int(9 * FPS)):
        t = k * DT
        hb = H * (0.25 + (1.30 - 0.25) * min(1., t / 6.))
        top = 240. - anchor * hb - c; bot = top + hb
        head_in = top >= -0.12 * hb
        min_head = min(min_head, top + 0.06 * hb)
        cmd = 0.
        if head_in:
            vt, vb = max(top, 0.), min(bot, H)
            rows = CA.apply([(1, 200., vt, 320., vb, .9)], 1, .5, 100. + t, H)
            y1, y2 = rows[0][2], rows[0][4]
            e = (y1 + .5 * (y2 - y1) - 240.) / 240.
            if hold: hold = abs(e) > .06
            else: hold = abs(e) > .10
            if hold: cmd = max(-300., min(300., 900. * e))
        else:
            lost += 1; first_lost = t if first_lost is None else first_lost
        hist.append(cmd)
        c += (hist[-4] if len(hist) >= 4 else 0.) * DT   # ~100 ms delay
    return lost, first_lost, min_head


for anchor in (0.35, 0.45, 0.55):
    for on in (False, True):
        lost, first, mh = run(on, anchor)
        print('person point %.2f at camera height, close_aim %-3s: frames with head out %3d / 270, first at %s, lowest head-centre row %.0f px'
              % (anchor, 'on' if on else 'off', lost, '-' if first is None else '%.2f s' % first, mh))
