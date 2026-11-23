"""11-22 (user approved): the server clears a frozen camera by itself.

Why: the i3system camera's USB chip sometimes stops answering while it stays on the bus (cam_frozen.py). 11-22 09:44 it froze, the server kept
retrying and every open failed with `libusb_timeout` for 49 minutes until a person pressed the page's camera reset. Only a power cut clears it:
behind the board's switchable hub that is the 5 V step of the page button (camera_link.cam_reset_work / cam_usb_power).

What this does: after N (default 5) opens in a row that failed with the same cause, with no frame in between, a watcher thread starts exactly the
page button's reset (APP.cam_reset_request, so every guard of the button applies as well). Cause = the log of that try shows `libusb_timeout`
(first-open failure `DIAG open: op=4(send) result=-7(libusb_timeout)` + `OPEN_FAILED`, exit 2 / 3 / 6) or the try ended with exit 8 (`CAMERA_FROZEN`).
Five failures take about 3 to 5 minutes at the server's retry pace (15 s after exit 2, 60 s after exit 8, plus the 20 s open).

Safety:
  - at most one reset per 30 minutes, kept over a server restart within the same boot (~/.lwir_cam_autoreset.json), so a camera that the 5 V step
    does not bring back never makes the board power-cycle its USB hub in a loop;
  - not while a person drives the gimbal by hand (a page's move command in the last 10 s, a page holding the move, the bridge reporting a move);
    it asks again every 2 s and goes when the gimbal has been idle;
  - not while a replay runs (a replay does not use the camera: the recording is the picture source and the camera program is not started), not during
    a measurement, not while a reset is already running, not for a camera that is not behind the switchable hub (software cannot cut its power);
  - the same reset as the button: the gimbal is stopped first and moves are held, the hub's 5 V goes off 3 s and on.

The hub also carries the gimbal's serial adapter (it drops and comes back as a new device). gimbal_link.gb_watch restarts the bridge by itself; after
the reset this watcher checks gimbal_alive / gimbal_ready and, when the bridge is dead or not ready after 30 / 45 s, restarts it the same way (retried).

Switch: LWIR_CAM_AUTORESET = 1 (default, on) | 0 (off) | dry (decide and log, never reset); on a running server /rules/set?cam_autoreset=1|0|dry
(kept over a restart like the other rules). N: LWIR_CAM_AUTORESET_N (5). State: /healthz cam_autoreset. Log lines: CAMAUTORESET ... in camera_events.log.
"""
import json
import os
import threading
import time

from . import cam_frozen, camera_extra

ENV_MODE = 'LWIR_CAM_AUTORESET'
EXIT_FROZEN = 8
USB_EXITS = (2, 3, 6)    # first open failed / reopen failed after a reset / four engine resets without a frame: counted only when the last open of the try timed out (libusb_timeout)
GAP_S = 1800.0           # at most one reset per 30 minutes
MANUAL_IDLE_S = 10.0     # a manual move command this recently = a person is driving
POLL_S = 2.0
GIMBAL_SPAWN_S = 30.0    # after the reset: bridge still dead this long -> start it
GIMBAL_END_S = 45.0      # after the reset: bridge alive but not ready this long -> end it (gb_watch starts the next one)
CHECK_S = 150.0          # how long after the reset ends the result is waited for (camera frames, gimbal ready)
RESET_S = 240.0          # how long the reset thread may take
CUT_TEXTS = ('껐다 켬', '껐다 켰지만')   # cam_usb_power's answers when the 5 V really went off and on (the second: the camera did not show up again, the power was still cut)
REFUSED_RETRY_S = 300.0  # the reset ran but no power step happened (guards inside cam_usb_power): ask again after this, at most REFUSED_MAX times in a row
REFUSED_MAX = 3
STATE_FILE = os.path.expanduser('~/.lwir_cam_autoreset.json')

_mode = [None]           # set by /rules/set


def _env_int(name_, default_):
    try: return max(1, int(os.environ.get(name_, default_)))
    except ValueError: return default_


def start_mode():
    v_ = os.environ.get(ENV_MODE, '1').strip().lower()
    return '0' if v_ in ('0', 'off', 'false', 'no') else 'dry' if v_ == 'dry' else '1'


def mode():
    return _mode[0] or start_mode()


def set_mode(value_):   # '1' | '0' | 'dry'
    _mode[0] = value_ if value_ in ('1', '0', 'dry') else None


def need_n():
    return _env_int('LWIR_CAM_AUTORESET_N', 5)


def switch_state():   # for GET /rules
    return dict(cam_autoreset=mode(), cam_autoreset_n=need_n(), cam_autoreset_gap_s=int(GAP_S))


# ---------------------------------------------------------------- cause of a failed open (camera_link hooks)

def last_open_diag(text_):
    """The last `DIAG open: op=<n>(<name>) result=<code>(<reason>)` line of a try ('' = none). Real lines (cam_pipe v2.13): first open of 11-22 10:16:
    `DIAG open: op=4(send) result=-7(libusb_timeout)` then `OPEN_FAILED`; a cable pulled during a reopen: `DIAG open: op=2(open) result=7(device_not_connected)`."""
    for ln_ in reversed((text_ or '').splitlines()):
        if ln_.startswith('DIAG open:'): return ln_
    return ''


def usb_cause(rc_, text_):
    """'CAMERA_FROZEN' / 'libusb_timeout' when this cam_pipe try failed because the camera's USB chip does not answer, else ''.
    text_ = what that try wrote to cam_pipe.log. Exit 8 is the frozen exit of cam_pipe v2.12 on its own."""
    if rc_ == EXIT_FROZEN or 'CAMERA_FROZEN' in (text_ or ''): return 'CAMERA_FROZEN'
    if rc_ in USB_EXITS and 'libusb_timeout' in last_open_diag(text_): return 'libusb_timeout'   # the open that ended the try, not an earlier stall of the same run
    return ''


def state(live_):
    st_ = getattr(live_, 'cam_ar', None)
    if st_ is None:
        st_ = live_.cam_ar = dict(usb_n=0, cause='', off=0, phase='idle', phase_t=0.0, last_wall=None, last_text='', result='', skip='', skip_say_t=-1e9,
                                  count=0, dry=0, day='', up_s=None, boot='', refused_n=0, retry_t=0.0, left=0, tries=dict(spawn_t=-1e9, end_n=0), t_check=0.0)
    return st_


def on_spawn(live_):   # lv_spawn, before a camera cam_pipe starts: where its lines in cam_pipe.log begin
    try:
        name_ = getattr(getattr(live_, 'log', None), 'name', None)
        state(live_)['off'] = os.path.getsize(name_) if name_ else 0
    except Exception: pass


def _new_text(live_, size_=65536):
    try:
        name_ = getattr(getattr(live_, 'log', None), 'name', None)
        if not name_: return ''
        with open(name_, 'rb') as f_:
            f_.seek(0, 2); n_ = f_.tell(); f_.seek(max(state(live_)['off'], n_ - size_)); return f_.read().decode('utf-8', 'replace')
    except Exception: return ''


def on_exit(live_, rc_):
    """lv_cam_wait, once per ended camera cam_pipe, after live_.cam_fail_n was counted. Counts the failures in a row that have the USB cause."""
    try:
        st_ = state(live_)
        cause_ = usb_cause(rc_, _new_text(live_))
        if cause_:
            st_['usb_n'] = st_['usb_n'] + 1 if getattr(live_, 'cam_fail_n', 1) > 1 else 1   # a frame in between (cam_fail_n restarted at 1): a new chain
            st_['cause'] = cause_
        elif rc_ != 7:   # 7 = our previous cam_pipe was still closing: neither a cause nor a break
            st_['usb_n'] = 0
        return cause_
    except Exception: return ''


# ---------------------------------------------------------------- the persistent 30 minute limit

def read_uptime():
    try:
        with open('/proc/uptime') as f_: return float(f_.read().split()[0])
    except Exception: return time.monotonic()


def read_boot_id():
    try:
        with open('/proc/sys/kernel/random/boot_id') as f_: return f_.read().strip()
    except Exception: return ''


class Store:
    """last reset (uptime seconds of this boot), count and date: kept over a server restart. A new boot starts free (the board has no RTC: uptime, not the clock)."""
    def __init__(self, path_=None, uptime_=read_uptime, boot_=read_boot_id):
        self.path = path_ if path_ is not None else STATE_FILE; self.uptime, self.boot = uptime_, boot_

    def load(self, day_):   # dict(up_s, count, dry, day)
        try:
            with open(self.path) as f_: d_ = json.load(f_)
        except (OSError, ValueError): d_ = {}
        if not isinstance(d_, dict): d_ = {}
        if d_.get('boot') != self.boot(): d_['up_s'] = None   # another boot: the limit is over
        if d_.get('day') != day_: d_['count'] = 0; d_['dry'] = 0
        return dict(up_s=d_.get('up_s'), count=int(d_.get('count') or 0), dry=int(d_.get('dry') or 0), day=day_)

    def save(self, up_s_, count_, dry_, day_):
        try:
            tmp_ = '%s.%d.tmp' % (self.path, os.getpid())
            with open(tmp_, 'w') as f_: json.dump(dict(boot=self.boot(), up_s=up_s_, count=count_, dry=dry_, day=day_), f_)
            os.replace(tmp_, self.path)
        except OSError: pass


# ---------------------------------------------------------------- the watcher

class Watcher:
    """hooks: reset_request() -> (ok, text) | replay_active() -> bool | measuring() -> bool | manual_active(now) -> reason or '' | camera_place() -> 'hub' / 'root' / None |
    gimbal() -> dict(present, alive, ready, error, dev_ok) | gimbal_restart(kind) with kind 'spawn' / 'end' | reset_text() -> last text of the reset | say(text)"""
    def __init__(self, live_, hooks_, store_=None, clock_=time.monotonic, day_=lambda: time.strftime('%Y-%m-%d')):
        self.live, self.h, self.clock, self.day = live_, hooks_, clock_, day_
        self.store = store_ if store_ is not None else Store()
        st_ = state(live_); loaded_ = self.store.load(day_())
        st_.update(up_s=loaded_['up_s'], count=loaded_['count'], dry=loaded_['dry'], day=loaded_['day'], boot=self.store.boot())

    # ---- helpers
    def _skip(self, reason_, now_, say_=True):
        st_ = state(self.live)
        st_['skip'] = reason_
        if say_ and now_ - st_['skip_say_t'] >= 60.0:
            st_['skip_say_t'] = now_
            self.h.say('CAMAUTORESET waiting: %d failed opens in a row (%s), not resetting now: %s' % (st_['usb_n'], st_['cause'], reason_))
        return 'skip'

    def _remember(self, up_now_, counted_):
        st_ = state(self.live); day_ = self.day()
        if st_['day'] != day_: st_['count'] = 0; st_['dry'] = 0; st_['day'] = day_
        st_['up_s'] = up_now_
        st_['count' if counted_ == 'real' else 'dry'] += 1
        st_['last_wall'] = time.strftime('%H:%M:%S')
        self.store.save(st_['up_s'], st_['count'], st_['dry'], st_['day'])

    def limit_left(self):   # seconds until the next reset may start (0 = now)
        st_ = state(self.live)
        if st_['up_s'] is None: return 0.0
        return max(0.0, GAP_S - (self.store.uptime() - st_['up_s']))

    # ---- one poll
    def step(self):
        live_, st_, now_ = self.live, state(self.live), self.clock()
        if getattr(live_, 'stopped', False): return 'stopped'
        st_['left'] = int(self.limit_left())   # for /healthz next_in_s
        if st_['phase'] != 'idle': return self._after_reset(now_)
        if mode() == '0': st_['skip'] = ''; return 'off'
        if getattr(live_, 'cam_fail_n', 0) == 0: st_['usb_n'] = 0   # a frame came, or a reset ran: the chain is over
        if st_['usb_n'] < need_n():
            st_['skip'] = ''; return 'idle'
        if self.h.replay_active(): return self._skip('a replay is running (it does not use the camera)', now_)
        if getattr(live_, 'cam_hold', False): return self._skip('a camera reset is already running', now_, False)
        if self.h.measuring(): return self._skip('a measurement is running', now_)
        why_ = self.h.manual_active(now_)
        if why_: return self._skip('a person drives the gimbal by hand (%s); asking again when it is idle' % why_, now_)
        if now_ < st_['retry_t']: return self._skip('the last try did not cut the power, retry in %d s' % (st_['retry_t'] - now_), now_, False)
        left_ = self.limit_left()
        if left_ > 0: return self._skip('at most one reset per %d min: next in %d s' % (GAP_S / 60, left_), now_, False)
        place_ = self.h.camera_place()
        if place_ != 'hub': return self._skip('the camera is not behind the switchable hub (%s): software cannot cut its power' % (place_ or 'not on USB'), now_)
        what_ = '%d failed opens in a row, cause %s, camera behind the switchable hub' % (st_['usb_n'], st_['cause'])
        up_now_ = self.store.uptime()
        if mode() == 'dry':
            self._remember(up_now_, 'dry')
            st_['result'] = 'dry run at %s' % st_['last_wall']
            self.h.say('CAMAUTORESET dry run (LWIR_CAM_AUTORESET=dry): would run the camera reset now: %s (dry runs today: %d); the next decision in %d min' % (what_, st_['dry'], GAP_S / 60))
            return 'dry'
        ok_, text_ = self.h.reset_request()
        if not ok_: return self._skip('the camera reset was refused: %s' % text_, now_)
        self._remember(up_now_, 'real')
        st_['skip'] = ''; st_['phase'] = 'reset'; st_['phase_t'] = now_; st_['tries'] = dict(spawn_t=-1e9, end_n=0)
        self.h.say('CAMAUTORESET starting the camera reset (the page button\'s 5 V step on the camera hub): %s; reset %d today, at most one per %d min' % (what_, st_['count'], GAP_S / 60))
        return 'started'

    def _after_reset(self, now_):
        st_, live_ = state(self.live), self.live
        if st_['phase'] == 'reset':
            if getattr(live_, 'cam_hold', False) and now_ - st_['phase_t'] < RESET_S: return 'resetting'
            text_ = self.h.reset_text()
            if getattr(live_, 'cam_hold', False): self.h.say('CAMAUTORESET the reset thread did not end in %d s, checking the result anyway' % RESET_S)
            if not any(k_ in (text_ or '') for k_ in CUT_TEXTS):   # the reset ran but its power step did not (a guard in cam_usb_power, no software switch, helper error)
                st_['refused_n'] += 1; st_['count'] = max(0, st_['count'] - 1)   # count_today = resets that cut the power
                if st_['refused_n'] < REFUSED_MAX:
                    st_['up_s'] = None; st_['retry_t'] = now_ + REFUSED_RETRY_S   # does not count against the 30 minutes; one more try soon
                self.store.save(st_['up_s'], st_['count'], st_['dry'], st_['day'])
                st_['result'] = 'no power step: %s' % (text_ or '?')[:160]
                self.h.say('CAMAUTORESET the reset did not cut the power: %s%s' % ((text_ or '?')[:200], ' (retry in %d s)' % REFUSED_RETRY_S if st_['refused_n'] < REFUSED_MAX else ' (giving up for 30 min)'))
                if st_['refused_n'] >= REFUSED_MAX: st_['refused_n'] = 0   # gave up: the 30 minute limit stands, a new round after it
                st_['phase'] = 'idle'; st_['usb_n'] = 0; return 'no-power-step'
            st_['refused_n'] = 0
            st_['phase'] = 'check'; st_['phase_t'] = now_; st_['t_check'] = now_
            return 'checking'
        # phase 'check': camera frames and the gimbal link
        since_ = now_ - st_['t_check']
        cam_ok_ = getattr(live_, 'cam_fail_n', 1) == 0 and now_ - getattr(live_, 'last_frame_t', -1e9) < 10.0
        g_ = self.h.gimbal(); g_ok_ = (not g_.get('present')) or (g_.get('alive') and g_.get('ready') and not g_.get('error'))
        if g_.get('present') and not g_ok_:
            t_ = st_['tries']; total_ = now_ - st_['phase_t']
            if not g_.get('alive') and g_.get('dev_ok') and total_ >= GIMBAL_SPAWN_S and now_ - t_['spawn_t'] >= 20.0:
                t_['spawn_t'] = now_; self.h.say('CAMAUTORESET gimbal bridge is not running %d s after the reset: starting it' % total_); self.h.gimbal_restart('spawn')
            elif g_.get('alive') and not g_.get('ready') and total_ >= GIMBAL_END_S + 20.0 * t_['end_n'] and t_['end_n'] < 2:
                t_['end_n'] += 1; self.h.say('CAMAUTORESET gimbal bridge is not ready %d s after the reset: restarting it (try %d)' % (total_, t_['end_n'])); self.h.gimbal_restart('end')
        if (cam_ok_ and g_ok_) or since_ >= CHECK_S:
            st_['phase'] = 'idle'; st_['usb_n'] = 0
            st_['result'] = 'camera %s, gimbal %s' % ('frames again' if cam_ok_ else 'still no frame', 'ok' if g_ok_ else 'not ready (alive=%s ready=%s error=%s)' % (g_.get('alive'), g_.get('ready'), g_.get('error')))
            self.h.say('CAMAUTORESET result %.0f s after the reset: %s%s' % (since_, st_['result'], '' if cam_ok_ else '; the next reset is possible in %d min' % (self.limit_left() / 60)))
            return 'done'
        return 'checking'

    def run(self):   # the thread
        while not getattr(self.live, 'stopped', False):
            time.sleep(POLL_S)
            try: self.step()
            except Exception as e_: self.h.say('CAMAUTORESET watcher error: %r' % (e_,))


# ---------------------------------------------------------------- the real hooks and the page's view

class Hooks:
    """The real thing: APP (the server's namespace) and the camera / gimbal links."""
    def __init__(self, app_):
        self.app = app_

    def say(self, text_): camera_extra.say(text_)

    def reset_request(self): return self.app.cam_reset_request()

    def replay_active(self):
        from . import replay_link
        live_ = self.app.LIVE
        return bool(replay_link.blocking() or getattr(live_, 'replay', None) is not None)

    def measuring(self): return getattr(self.app.LIVE, 'meas', None) is not None

    def manual_active(self, now_):
        gb_ = getattr(self.app, 'GB', None)
        if gb_ is None: return ''
        if getattr(gb_, 'mover', None) is not None: return 'a page holds a move'
        ago_ = now_ - getattr(gb_, 'last_manual_t', -1e9)
        if ago_ < MANUAL_IDLE_S: return 'a move command %.0f s ago' % ago_
        if gb_.state.get('moving'): return 'the bridge reports a manual move'
        return ''

    def camera_place(self):
        live_ = self.app.LIVE
        return cam_frozen.where(live_) or cam_frozen.last_place(live_)

    def reset_text(self):
        r_ = getattr(self.app.LIVE, 'cam_reset', None)
        return (r_ or {}).get('text', '')

    def gimbal(self):
        gb_ = getattr(self.app, 'GB', None)
        if gb_ is None: return dict(present=False)
        p_ = gb_.proc
        return dict(present=True, alive=p_ is not None and p_.poll() is None, ready=bool(gb_.state['ready']), error=bool(gb_.state['error']),
                    dev_ok=self.app.gb_dev_sig(self.app.gb_device()) is not None)

    def gimbal_restart(self, kind_):
        gb_ = self.app.GB; p_ = gb_.proc
        if kind_ == 'spawn':
            if p_ is None or p_.poll() is not None: self.app.gb_spawn()   # gb_watch starts a bridge only when none is alive: same rule here
        elif p_ is not None and p_.poll() is None:
            self.app.gb_end(p_, 'camera auto-reset: the bridge is not ready after the hub lost its power')   # gb_watch starts the next one


_watcher = [None]


def install(app_):
    """app.py, once the live server runs (--live-wait 0: the same condition as the page's reset button). Starts the watcher thread."""
    if _watcher[0] is not None: return _watcher[0]
    w_ = Watcher(app_.LIVE, Hooks(app_))
    threading.Thread(target=w_.run, daemon=True, name='cam-autoreset').start()
    _watcher[0] = w_
    camera_extra.say('CAMAUTORESET %s: after %d failed opens in a row with libusb_timeout / CAMERA_FROZEN the camera is reset like the page button does, at most once per %d min'
                     % ({'1': 'on', '0': 'off', 'dry': 'dry run'}[mode()], need_n(), GAP_S / 60))
    return w_


def health(live_):   # /healthz cam_autoreset
    st_ = getattr(live_, 'cam_ar', None)
    if st_ is None: return dict(mode=mode(), usb_fail_n=0, need_n=need_n(), count_today=0, last=None, phase='idle', result='', skip='', next_in_s=0)
    w_ = _watcher[0]
    left_ = w_.limit_left() if w_ is not None else 0.0
    return dict(mode=mode(), usb_fail_n=st_['usb_n'], need_n=need_n(), count_today=st_['count'], dry_today=st_['dry'], last=st_['last_wall'], phase=st_['phase'],
                result=st_['result'], skip=st_['skip'], next_in_s=int(st_.get('left', 0) if w_ is None else left_))
