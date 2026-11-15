"""11-15: the camera sends frames but the picture has no scene (cam_pipe v2.13 lines SCENE_LOST / SCENE_BACK).

What happened on the board (11-15 16:35-17:2x): after many camera opens in a short time (each moved the shutter) the picture became a flat field: the middle of the
picture one value, a noisy rim in the shape of the shutter blade, frames still at 30 fps. Moving the gimbal changed nothing; the calibrations of the next opens did
not help; a hand replug on site did. cam_pipe now checks the picture once a second (top of cam_pipe.cpp, v2.13 (c)): after LWIR_CAM_BLANK_S seconds (default 20)
without a scene it prints SCENE_LOST and, once per camera power-on, resets the camera's engine (about 30 s without a picture); after that only the line.

The server side: camera_extra._tail (the camera log thread, every 3 s) passes every SCENE_ line of cam_pipe.log to on_line. The state is shown on the page
(/rules key cam_scene, like cam_frozen) until a SCENE_BACK line or until a new cam_pipe starts (it checks again; the notice comes back after its LWIR_CAM_BLANK_S
if the picture still has no scene). Nothing here touches the camera or USB. The page's camera reset button (camera_link.cam_reset_work) treats the state like a
dead camera: behind the board's switchable hub it cuts the camera's 5 V; on a port without a power switch it answers with the replug notice and opens nothing.
"""
import re
import time

TITLE = '카메라 영상에 장면이 없습니다 (셔터가 닫힌 채 멈춘 것으로 보임)'
REPLUG = '카메라 USB 선을 10초 뽑았다 다시 꽂아 주세요.'
RESETTING = '카메라가 스스로 한 번 다시 시작하는 중입니다 (약 30초). 그 뒤에도 장면이 없으면 ' + REPLUG
NO_AUTO = '자동 복구는 이 카메라 전원에서 이미 한 번 했습니다. ' + REPLUG
_AT = re.compile(r' at (\d\d:\d\d:\d\d)\s*$')


def on_line(live_, line_):   # one cam_pipe line (camera log thread). Returns True when the state changed
    if live_ is None: return False
    if line_.startswith('SCENE_LOST '):
        m_ = _AT.search(line_)
        old_ = getattr(live_, 'cam_scene', None)
        live_.cam_scene = dict(at=(old_ or {}).get('at') or (m_.group(1) if m_ else time.strftime('%H:%M:%S')), reset='one engine reset now' in line_, line=line_[:300])
        return True
    if line_.startswith('SCENE_BACK '):
        changed_ = getattr(live_, 'cam_scene', None) is not None
        live_.cam_scene = None
        return changed_
    return False


def on_spawn(live_):   # lv_spawn, before a camera cam_pipe starts: the new cam_pipe checks the picture again
    live_.cam_scene = None


def notice(hub_):   # the reset button's answer for a picture without a scene. hub_ = the camera is behind the switchable hub (the button cuts its power)
    return TITLE + '. ' + REPLUG + (' 또는 카메라 초기화 버튼을 누르세요.' if hub_ else ' 이 USB 포트는 소프트웨어로 전원을 끌 수 없어 초기화 버튼으로는 고칠 수 없음')


def state(live_):
    """For the page (/rules) and /healthz: None = the picture has a scene (or nothing is known), else dict(at, reset, title, text)."""
    sc_ = getattr(live_, 'cam_scene', None) if live_ is not None else None
    if sc_ is None: return None
    return dict(at=sc_['at'], reset=sc_['reset'], title=TITLE, text=(RESETTING if sc_['reset'] else NO_AUTO) + ' (%s 부터)' % sc_['at'])
