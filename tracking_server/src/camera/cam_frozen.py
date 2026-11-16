"""11-15: the camera's own USB chip froze (cam_pipe v2.12 exit code 8, line "CAMERA_FROZEN <reason>: ...").

What happens on the board (close_range_20261113/camera_usb_drops/ in LIG-Project has the evidence): the i3system camera's USB chip (0547:0080) stops answering
while it stays on the bus. Only cutting its power clears it: a hand replug, or the 5 V step (/usr/local/sbin/lwir_usb5v_cycle) when the camera sits behind the
board's Genesys hub (bus 1 / 2 behind a hub port, e.g. 1-1.2), the only USB power the board can switch. On a board root port (3-1 / 4-1 today) nothing in
software and no reboot cuts it. usbreset only made the kernel drop the frozen device.

The server side (camera_link.py calls these):
  - lv_cam_wait: after exit 8 the state below is set and cam_pipe is not started again until the camera's USB entry changes (a replug: new device number,
    or gone and back) or RETRY_S has passed; later failed tries keep the state. While the state is set the frame loop looks every POLL_S s instead of 10 s,
    so a replug is picked up within a few seconds. A try started by a replug that fails (the camera may still be starting) is followed after AFTER_REPLUG_S.
  - the readers: the first camera frame clears the state (replay frames do not).
  - the reset button: never usbreset; the 5 V step only when the camera is (or, when it is off the bus, was last seen) behind the hub, else the replug notice.
  - the page (/rules key cam_frozen) shows the notice, /healthz has cam_frozen (true / false) for scripts/keepalive.sh (no reboot while frozen).
While cam_pipe never ends with code 8 the state stays None and the only thing done here is noting, at each camera cam_pipe start, where the camera is on USB
(LIVE.cam_usb_place: 'hub' / 'root'; memory only, so after a server restart a camera that is off the bus has no known place and gets no 5 V step).
"""
import os
import time

EXIT_FROZEN = 8          # cam_pipe v2.12
RETRY_S = 60.0           # while frozen: a new cam_pipe at most this often unless the camera's USB entry changes
AFTER_REPLUG_S = 15.0    # 11-15 review: a try started by a replug failed (a camera just powered on may not answer the open yet): the next one this soon (the old exit-2 pause), not 60 s
POLL_S = 2.0             # while frozen: the frame loop's wait for a frame (then lv_cam_wait looks at the USB bus)
SYSFS = '/sys/bus/usb/devices'
CAM_ID = ('0547', '0080')
HUB_BUSES = (1, 2)       # xhci bus 1 / 2: the board's Genesys hub (05e3:0610 / 0620) sits on their port 1; lwir_usb5v_cycle cuts its 5 V

TITLE = '카메라가 멈췄습니다 (USB 칩 응답 없음)'
REPLUG = '카메라 USB 선을 10초 뽑았다 다시 꽂아 주세요.'
BUTTON = '또는 카메라 초기화 버튼을 누르세요.'
NO_SWITCH = '카메라가 소프트웨어로 전원을 끌 수 없는 USB 포트(보드에 직접 연결)에 있어 USB 는 건드리지 않음. 영상이 안 나오면 ' + REPLUG + ' '
UNKNOWN_PORT = '카메라가 어느 USB 포트에 있는지 알 수 없어 USB 전원은 건드리지 않음. 영상이 안 나오면 ' + REPLUG + ' '


def usb_cams(root_=None):
    """The camera's entries on the USB bus, sorted: [(sysfs name, busnum, devnum)]. [] = not on the bus, None = the bus cannot be read."""
    root_ = root_ or SYSFS   # read at call time (tests point SYSFS at a folder)
    try: names_ = os.listdir(root_)
    except OSError: return None
    readable_ = 0; out_ = []
    for d_ in names_:
        p_ = os.path.join(root_, d_)
        try:
            with open(os.path.join(p_, 'idVendor')) as f_: vid_ = f_.read().strip()
        except OSError: continue   # interfaces (1-1.3:1.0) have no idVendor
        readable_ += 1
        try:
            with open(os.path.join(p_, 'idProduct')) as f_: pid_ = f_.read().strip()
            if (vid_, pid_) != CAM_ID: continue
            with open(os.path.join(p_, 'busnum')) as f_: bus_ = int(f_.read().strip())
            with open(os.path.join(p_, 'devnum')) as f_: dev_ = int(f_.read().strip())
        except (OSError, ValueError): continue
        out_.append((d_, bus_, dev_))
    return sorted(out_) if readable_ else None


def place(name_, bus_):
    """'hub' = behind a hub port on bus 1 / 2 (e.g. 1-1.2, 2-1.4): the 5 V step cuts its power. 'root' = anything else (3-1, 4-1, ...): no software switch."""
    return 'hub' if bus_ in HUB_BUSES and '.' in name_ else 'root'


def where(live_=None, root_=None):
    """Where the camera is now: 'hub' / 'root', remembered in live_.cam_usb_place. None = not on the bus or the bus cannot be read."""
    cams_ = usb_cams(root_)
    if not cams_: return None
    p_ = place(cams_[0][0], cams_[0][1])
    if live_ is not None: live_.cam_usb_place = p_
    return p_


def last_place(live_):   # where the camera was last seen in this server run ('hub' / 'root' / None = never seen)
    return getattr(live_, 'cam_usb_place', None)


def reason_from_log(path_):   # the reason of the last CAMERA_FROZEN line in cam_pipe.log ('' = none found)
    if not path_: return ''
    try:
        with open(path_, 'rb') as f_:
            f_.seek(0, 2); n_ = f_.tell(); f_.seek(max(0, n_ - 65536)); tail_ = f_.read().decode('utf-8', 'replace')
    except OSError: return ''
    for ln_ in reversed(tail_.splitlines()):
        if ln_.startswith('CAMERA_FROZEN '): return ln_[len('CAMERA_FROZEN '):].split(': ', 1)[0].strip()[:200]
    return ''


def state(live_):
    """For the page (/rules) and /healthz: None = not frozen, else dict(at, reason, hub, title, text). hub = the reset button can cut the camera's power."""
    fz_ = getattr(live_, 'cam_frozen', None)
    if fz_ is None: return None
    return dict(at=fz_['at'], reason=fz_['reason'], hub=fz_['hub'], title=TITLE, text=REPLUG + (' ' + BUTTON if fz_['hub'] else '') + ' (%s 부터)' % fz_['at'])


def notice(hub_):   # the reset button's answer while frozen. hub_ = the camera is behind the switchable hub (the button cuts its power)
    return TITLE + '. ' + REPLUG + (' ' + BUTTON if hub_ else ' 이 USB 포트는 소프트웨어로 전원을 끌 수 없어 초기화 버튼으로는 고칠 수 없음. 다시 꽂으면 서버가 카메라를 다시 엶')


def on_exit(live_, rc_, now_, log_path_=None, root_=None):
    """lv_cam_wait, when a camera cam_pipe has ended (once per exit). Returns the seconds to wait before the next start, or None = not frozen (the usual rule).
    Exit 8 sets (or renews) the state; while it is set, every other failed try (not 0, not 7 = the old cam_pipe still closing) keeps it and waits as well."""
    fz_ = getattr(live_, 'cam_frozen', None)
    if rc_ == EXIT_FROZEN:
        cams_ = usb_cams(root_)
        if cams_: live_.cam_usb_place = place(cams_[0][0], cams_[0][1])
        live_.cam_frozen = dict(t=now_, at=time.strftime('%H:%M:%S'), reason=reason_from_log(log_path_) or 'cam_pipe exit 8',
                                ref=cams_, hub=last_place(live_) == 'hub', tries=0)
        return RETRY_S
    if fz_ is not None and rc_ not in (0, 7):
        fz_['tries'] += 1
        return AFTER_REPLUG_S if fz_.get('after_replug') else RETRY_S
    return None


def say_tail(live_):   # the end of lv_cam_wait's "cam_pipe ended" line while frozen
    fz_ = getattr(live_, 'cam_frozen', None)
    if fz_ is None: return ''
    return ('; CAMERA FROZEN since %s (%s), camera %s: no new cam_pipe until the camera USB entry changes (replug) or %.0f s pass'
            % (fz_['at'], fz_['reason'], 'behind the switchable hub' if fz_['hub'] else 'on a port without a software power switch' if last_place(live_) else 'not seen on USB',
               AFTER_REPLUG_S if fz_.get('after_replug') else RETRY_S))


def replugged(live_, root_=None):
    """lv_cam_wait while it waits: True when the camera is on the bus and its entry differs from the one at the last try (replug, or a power cut of its port)."""
    fz_ = getattr(live_, 'cam_frozen', None)
    if fz_ is None: return False
    cams_ = usb_cams(root_)
    if cams_:
        live_.cam_usb_place = place(cams_[0][0], cams_[0][1]); fz_['hub'] = live_.cam_usb_place == 'hub'
    hit_ = bool(cams_) and cams_ != fz_['ref']
    if hit_: fz_['replug_hit'] = True   # 11-15 review: the try lv_cam_wait now starts is a replug try (on_spawn moves this to after_replug)
    return hit_


def on_spawn(live_, root_=None):   # lv_spawn, before a camera cam_pipe starts: where the camera is (last place); while frozen, the USB entry it starts with (a later change = replug)
    cams_ = usb_cams(root_)
    if cams_: live_.cam_usb_place = place(cams_[0][0], cams_[0][1])
    fz_ = getattr(live_, 'cam_frozen', None)
    if fz_ is not None: fz_['ref'] = cams_; fz_['after_replug'] = fz_.pop('replug_hit', False)   # 11-15 review: if this try fails, the next comes after AFTER_REPLUG_S


def on_frame(live_):   # a reader, under LIVE.cond, only while the state is set: a camera frame came, the camera answers again. Returns the log line
    fz_ = live_.cam_frozen; live_.cam_frozen = None
    return 'LIVE camera: frames again: the frozen-camera state (since %s, %s, %d failed tries since) is cleared' % (fz_['at'], fz_['reason'], fz_['tries'])
