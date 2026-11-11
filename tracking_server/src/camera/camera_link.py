"""실시간 카메라: 카메라 프로그램(cam_pipe) 띄우기와 읽기, 16비트 -> 8비트 그림 바꾸기, 다음 프레임 꺼내기, 카메라 초기화와 USB 전원 껐다 켜기."""
import os
import time
import numpy as np
from ..common.state import APP
from ..common.shared_values import HERE
from ..track.stabilizer import STAB, stab_live, stab_off
from . import replay_link   # 11-12: recording replay from the page (does nothing until it is used)
from . import camera_extra   # 2026-11-11: 지워지지 않는 카메라 기록(기본 켜짐), 프레임 시간표와 공유 메모리(환경 변수로만 켜짐)


class LiveImages:   # endless list of camera frames; it ends (IndexError) when the live time is over
    def __len__(self): return 10 ** 9
    def __getitem__(self, i):
        if APP.LIVE.stopped: raise IndexError(i)
        return dict(id=int(i), file_name='live_%09d.png' % i, extra_info=dict(video_id='live'))


class _Live: pass


def lv_reader(p_):   # v21: reads one camera process; a later reopen starts a new reader
    if camera_extra.stamped(p_): return camera_extra.reader(p_)   # only for a cam_pipe started with LWIR_CAM_STAMP=1 (off by default)
    n_ = APP.LIVE_W * APP.LIVE_H * 2; buf_ = bytearray(n_); cnt_ = 0; tl_ = time.monotonic()
    while True:   # v24c: never stop reading before EOF; after LIVE.stopped the frames are thrown away (cam_pipe must not block on a full pipe: it then cannot act on SIGTERM and was SIGKILLed with the camera open)
        mv_ = memoryview(buf_); got_ = 0
        while got_ < n_:
            k_ = p_.stdout.readinto(mv_[got_:])
            if not k_:
                if p_ is APP.LIVE.proc: APP.LIVE.cam_lost = True
                with APP.LIVE.cond: APP.LIVE.cond.notify_all()
                return
            got_ += k_
        if APP.LIVE.stopped: continue
        raw_ = np.frombuffer(buf_, np.uint16).reshape(APP.LIVE_H, APP.LIVE_W).copy()
        ta_ = time.perf_counter()
        with APP.LIVE.cond:
            if len(APP.LIVE.q) == APP.LIVE.q.maxlen: APP.LIVE.q_drop += 1   # v26b: the oldest waiting frame is thrown away (the tracker is 3 frames behind)
            APP.LIVE.q.append((ta_, raw_)); APP.LIVE.cond.notify()
        rc_ = APP.LIVE.rec   # v22d: measurement video
        if rc_ is not None:
            with rc_['cond']:
                if len(rc_['q']) < 240: rc_['q'].append((ta_, raw_)); rc_['cond'].notify()
                else: rc_['dropped'] += 1
        cnt_ += 1; now_ = time.monotonic(); APP.LIVE.last_frame_t = now_; APP.LIVE.cam_fail_n = 0; APP.LIVE.cam_unrec_n = 0; APP.LIVE.frames_in += 1
        if now_ - tl_ >= 1.0: APP.LIVE.cam_fps = cnt_ / (now_ - tl_); cnt_ = 0; tl_ = now_


def lv_spawn():   # v21: open (or reopen) the camera process and its reader
    if camera_extra.wanted(): return camera_extra.spawn()   # LWIR_CAM_STAMP=1 (off by default): the same start with the setting passed on to cam_pipe
    APP.LIVE.proc = APP.lv_sp.Popen(replay_link.cam_argv(), stdout=APP.lv_sp.PIPE, stderr=APP.LIVE.log, bufsize=0); APP.LIVE.cam_lost = False; APP.LIVE.proc_t0 = time.monotonic()   # v26: cam_lost is cleared after the new process is in place (the old reader's end-of-file could set it for the new one and start a second cam_pipe)
    APP.lv_th.Thread(target=APP.lv_reader, args=(APP.LIVE.proc,), daemon=True).start()
    camera_extra.log_start()   # camera_events.log: cam_pipe's lines and the server's camera lines, kept over restarts (LWIR_CAM_LOG=0 turns it off)


def lv_cam_wait():   # v21 (--live-wait 0): no frame: keep the page and the gimbal link up. v24: cam_pipe repairs a stalled camera itself (ResetMainBoard + reopen, about 30 s), so the server only restarts it when it ended on its own or stayed silent for 240 s. (v21-v23 restarted a healthy cam_pipe after any 10 s gap once it was older than 180 s; the next open then often stayed dead.)
    now_ = time.monotonic(); since_ = now_ - APP.LIVE.t0; idle_ = now_ - max(APP.LIVE.last_frame_t, APP.LIVE.proc_t0)
    if since_ > APP.args.live_seconds: APP.LIVE.stopped = True; raise RuntimeError('no camera frame within --live-seconds (%.0f s)' % since_)
    if APP.LIVE.cam_hold: return   # v33: the reset / reboot button is handling the camera process
    if APP.LIVE.cam_lost and replay_link.on_cam_lost(now_): return   # 11-12: the replay program ended (end of a single pass, or an error): back to the camera
    if APP.LIVE.cam_lost and now_ - APP.LIVE.proc_t0 > 20:   # v24d: cam_pipe ended on its own. Exit 0 = normal; 6 = four engine resets without a frame; 7 = another cam_pipe holds the camera; 2/3 = scan/open failed. Failures are retried with a growing pause (60 s, 5 min, 15 min) so a dead camera does not get an engine reset every minute, and the page shows the FAILED state.
        rc_ = APP.LIVE.proc.poll()
        if rc_ is None:   # v26b: its output is closed but the process is still there (it is closing the camera). A new cam_pipe now would find the camera busy; wait for the old one, ask it to end after a minute.
            try: rc_ = APP.LIVE.proc.wait(2)
            except Exception: rc_ = None
            if rc_ is None and now_ - getattr(APP.LIVE, 'cam_eof_t', now_) > 60: APP.LIVE.proc.terminate()
            if rc_ is None and not hasattr(APP.LIVE, 'cam_eof_t'): APP.LIVE.cam_eof_t = now_
        if rc_ is not None:
            if hasattr(APP.LIVE, 'cam_eof_t'): del APP.LIVE.cam_eof_t
            if APP.LIVE.cam_next_spawn is None:
                APP.LIVE.cam_fail_n = APP.LIVE.cam_fail_n + 1 if rc_ != 0 else 0
                APP.LIVE.cam_unrec_n = APP.LIVE.cam_unrec_n + 1 if rc_ in (3, 6) else APP.LIVE.cam_unrec_n
                # v26: the growing pause is only for the codes after which an engine reset was sent and did not help: 6 (four resets without a frame) and 3 (reopen failed after a reset). v26b: the pause is 60 s, then 5 min (was up to 15 min:
                # the picture then stayed off for 15 min after the camera was back). Code 2 (camera not found / open failed: USB cable out, no reset was sent), a signal death and code 7 (another cam_pipe still closing) are retried quickly.
                wait_ = 0 if APP.LIVE.cam_fail_n == 0 else 5 if rc_ == 7 else (60, 300, 300)[min(APP.LIVE.cam_unrec_n - 1, 2)] if rc_ in (3, 6) else 15
                APP.LIVE.cam_next_spawn = now_ + wait_
                camera_extra.say('LIVE camera: cam_pipe ended with code %s after %.0f s without a frame (failures in a row: %d), next start in %d s' % (rc_, idle_, APP.LIVE.cam_fail_n, wait_))
            if now_ >= APP.LIVE.cam_next_spawn:
                APP.LIVE.cam_next_spawn = None; camera_extra.say('LIVE camera: starting cam_pipe'); APP.lv_spawn()
    elif idle_ > 330 and not APP.LIVE.cam_lost:   # cam_pipe alive but silent for 5.5 min: SIGTERM, wait up to 45 s (never kill while the camera may be open), start a new one. v26: was 240 s, which cut into cam_pipe's own last repair round (it gives up with code 6 after about 230 s)
        APP.LIVE.proc.terminate()
        try: APP.LIVE.proc.wait(45)
        except Exception:
            camera_extra.say('LIVE camera: cam_pipe did not end on SIGTERM in 45 s, SIGKILL'); APP.LIVE.proc.kill()
            try: APP.LIVE.proc.wait(5)   # v26: the killed process must be gone (camera lock released) before the next one starts
            except Exception: pass
        camera_extra.say('LIVE camera: no frame for %.0f s from a live cam_pipe, restarting it' % idle_); APP.LIVE.cam_next_spawn = None; APP.lv_spawn()
    img_ = np.zeros((480, 640, 3), np.uint8)
    APP.cv2.putText(img_, 'NO CAMERA FRAME  %.0f s' % idle_, (90, 230), APP.cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    if APP.LIVE.cam_fail_n: APP.cv2.putText(img_, 'CAMERA FAILED: %d restarts without a frame - check the USB cable / power' % APP.LIVE.cam_fail_n, (20, 275), APP.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    else: APP.cv2.putText(img_, 'the camera restarts itself (about 30 s); page and gimbal keys stay live', (20, 275), APP.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    rp_ = getattr(APP.LIVE, 'replay', None)
    if rp_ is not None or getattr(APP.LIVE, 'replay_busy', False): APP.cv2.putText(img_, (replay_link.top_text(rp_) + ': loading the frames') if rp_ is not None else 'REPLAY: switching the picture source', (20, 320), APP.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)   # 11-12
    if APP.GB is not None:
        st_ = APP.GB.state; gtxt_ = 'gimbal: not running' if not st_['running'] else 'gimbal: ERROR' if st_['error'] else 'gimbal: manual, moving' if st_['moving'] else 'gimbal: idle'
        if APP.GB.gui: gu_ = st_['gui']; gtxt_ += ' | GUI %s' % ('connected' if gu_['client'] else 'no client')
        APP.cv2.putText(img_, gtxt_, (8, 44), APP.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    ok_, jb_ = APP.cv2.imencode('.jpg', img_, [APP.cv2.IMWRITE_JPEG_QUALITY, 80])
    with APP.LIVE.jcond: APP.LIVE.jpeg = jb_.tobytes(); APP.LIVE.seq += 1; APP.LIVE.jcond.notify_all()


def lv_to8(raw_):
    if APP.args.live_map == 'fixed': return APP.cv2.convertScaleAbs(raw_, alpha=255.0 / 65535.0)
    hist_ = np.bincount((raw_ >> 4).ravel(), minlength=4096).cumsum(); tot_ = hist_[-1]
    lo_ = float(np.searchsorted(hist_, tot_ * 0.01)) * 16; hi_ = float(np.searchsorted(hist_, tot_ * 0.99)) * 16 + 16
    if hi_ - lo_ < 200: hi_ = lo_ + 200
    if APP.LIVE.lo is None: APP.LIVE.lo, APP.LIVE.hi = lo_, hi_
    else: APP.LIVE.lo += 0.1 * (lo_ - APP.LIVE.lo); APP.LIVE.hi += 0.1 * (hi_ - APP.LIVE.hi)
    return np.clip((raw_.astype(np.float32) - APP.LIVE.lo) * (255.0 / (APP.LIVE.hi - APP.LIVE.lo)), 0, 255).astype(np.uint8)


def live_next():
    while True:
        with APP.LIVE.cond:
            while not APP.LIVE.q:
                if APP.LIVE.stopped or (APP.LIVE.cam_lost and APP.args.live_wait): break
                wait_s_ = (APP.args.live_wait or 10) if APP.LIVE.last is None else 10   # v20: the real camera takes >10 s to start (scan, open, shutter calibration)
                if APP.LIVE.cond.wait(timeout=wait_s_): continue
                if APP.args.live_wait: raise RuntimeError('no camera frame for %d s' % wait_s_)
                break   # v21: --live-wait 0: no frame yet
            if APP.LIVE.q:
                if not APP.args.pipeline:
                    while len(APP.LIVE.q) > 1: APP.LIVE.q.popleft()
                t_, raw_ = APP.LIVE.q.popleft()
                if APP.LIVE.prev_t is not None and t_ - APP.LIVE.prev_t > 1.0:   # v26b: the camera gave no frame for over 1 s (repair, restart). The person at the old place may be someone else now:
                    APP.LIVE.gap_n += 1                                         # the gimbal selection is dropped here, before this frame goes to the bridge, and the tracker forgets its tracks at this frame (see the main loop). The person is clicked again.
                    if APP.GB is not None: APP.gb_send('H')
                    camera_extra.say('LIVE input gap of %.1f s: selection dropped, tracks restart' % (t_ - APP.LIVE.prev_t))
                APP.LIVE.prev_t = t_; break
            if APP.LIVE.stopped or APP.args.live_wait:
                if APP.LIVE.last is None: raise RuntimeError('camera delivered no frame')
                return APP.LIVE.last
        with APP.LIVE.cam_lock: APP.lv_cam_wait()   # v33: cam_lock. v21: outside the lock: may reopen the camera process
    APP.FRAME_DY[0] = 16.0; g8_ = APP.lv_to8(raw_); gg_ = np.empty((512, 640), np.uint8); gg_[16:496] = g8_; gg_[:16] = g8_[0]; gg_[496:] = g8_[-1]
    APP.LIVE.stabq.append(stab_live(g8_, t_, gg_) if STAB['on'] else stab_off())   # stab: LIVE.raws / LIVE.rawq below keep the unmoved frames
    APP.LIVE.arr.append(t_); APP.LIVE.last = gg_
    if APP.GB is not None and APP.GB.gui: APP.LIVE.raws.append(raw_ if APP.args.gui_map == 'team' else (g8_.astype(np.uint16) * 257))   # v20: the picture for the GUI, paired with LIVE.arr
    APP.LIVE.rawq.append(raw_)   # v24f/g: raw frame for the warmth veto, paired with LIVE.arr (always, so the veto can be switched on live)
    return gg_


def cam_usb_seen():   # v33: is the camera (i3system 0547:0080) on the USB bus at all? None = cannot tell
    try: rc_ = APP.lv_sp.run(['lsusb', '-d', '0547:0080'], stdout=APP.lv_sp.DEVNULL, stderr=APP.lv_sp.DEVNULL, timeout=10).returncode
    except Exception: return None
    if rc_ == 0: return True
    try:   # v35: "not there" needs a second look (sysfs): lsusb also fails for its own reasons
        n_ = 0
        for d_ in os.listdir('/sys/bus/usb/devices'):
            f_ = '/sys/bus/usb/devices/%s/' % d_
            if not os.path.exists(f_ + 'idVendor'): continue
            n_ += 1
            if open(f_ + 'idVendor').read().strip() == '0547' and open(f_ + 'idProduct').read().strip() == '0080': return True
        return False if n_ else None   # no USB device readable at all = cannot tell
    except Exception: return None


def cam_usb_gone(wait_=12):   # v35: True only when the camera stays off the USB bus for the whole wait (one miss is not enough)
    for _ in range(int(wait_ / 2)):
        if APP.cam_usb_seen() is not False: return False
        time.sleep(2)
    return APP.cam_usb_seen() is False


def cam_usb_power():   # v35: USB host 5 V off 3 s and on (root helper, same as pulling the plug). Everything on the board USB ports loses power, the gimbal serial adapter too.
    # returns (state, text). state: 'cut' = power went off and on, 'no' = power not touched, 'off' = power may still be off or the result is unknown
    try:
        for ln_ in open('/proc/mounts'):
            d_ = ln_.split()[0]
            if d_.startswith('/dev/') and '/usb' in os.path.realpath('/sys/class/block/' + os.path.basename(d_)):
                return 'no', 'USB 저장장치(%s)가 꽂혀 있어 USB 전원을 끄지 않음. 저장장치를 꺼낸 뒤 다시 누르세요. ' % d_
    except Exception: pass
    try:
        import glob as cu_glob
        open_ = False
        if APP.LIVE.stopped: return 'no', '서버 종료 중이라 USB 전원을 끄지 않음. '
        if APP.GB is not None:
            APP.GB.usb_hold = True; APP.GB.mover = None   # manual moves and clicks are turned into stop while the hold is on
            if APP.GB.closing: return 'no', '서버 종료 중이라 USB 전원을 끄지 않음. '
            with APP.GB.wcond: open_ = APP.GB.wopen
            if not open_ and APP.GB.proc is not None and APP.GB.proc.poll() is None: return 'no', '짐벌 연결을 다시 시작하는 중이라 USB 전원을 끄지 않음. 잠시 뒤 다시 누르세요. '
        if not open_:   # our bridge is closed: that is not "idle", another program may hold the gimbal device and be moving it
            dv_ = set(cu_glob.glob('/dev/serial/by-id/usb-Silicon_Labs_CP2102N_*'))
            if getattr(APP.args, 'gimbal', '') and os.path.exists(APP.args.gimbal): dv_.add(APP.args.gimbal)
            for d_ in sorted(dv_):
                try: who_ = APP.lv_sp.run(['fuser', d_], stdin=APP.lv_sp.DEVNULL, stdout=APP.lv_sp.PIPE, stderr=APP.lv_sp.DEVNULL, timeout=5).stdout.decode('ascii', 'replace').strip()
                except Exception: return 'no', '짐벌 장치를 누가 쓰는지 확인하지 못해 USB 전원을 끄지 않음. 잠시 뒤 다시 누르세요. '
                if who_: return 'no', '다른 프로그램(프로세스 %s)이 짐벌 장치를 쓰고 있어 USB 전원을 끄지 않음. 그 프로그램을 멈춘 뒤 다시 누르세요. ' % ' '.join(who_.split())
        if APP.GB is not None:
            if open_:   # bridge closed and nobody else on the device = nothing is driving the gimbal
                if not (APP.gb_send('V 0 0') and APP.gb_send('H')): return 'no', '짐벌 정지 명령을 보내지 못해 USB 전원을 끄지 않음. 잠시 뒤 다시 누르세요. '
                for _ in range(15):
                    time.sleep(0.2)
                    if not APP.GB.state['moving'] and not APP.GB.state['auto']: break
                else: return 'no', '짐벌 정지가 확인되지 않아 USB 전원을 끄지 않음. 짐벌을 멈춘 뒤 다시 누르세요. '
                time.sleep(0.3)
                if APP.GB.state['moving'] or APP.GB.state['auto']: return 'no', '짐벌이 다시 움직여 USB 전원을 끄지 않음. 짐벌을 멈춘 뒤 다시 누르세요. '   # a command from the PC GUI came in between
        def run_(need_=True):   # (code, output); code None = no answer in 30 s. Not subprocess.run: its kill() of a root process raises and then waits without limit.
            try:   # the helper cuts the power only before this time: a start that hangs and comes late does nothing. The folder is ours alone (not /dev/shm, where anyone could plant the file)
                fd_ = os.open(os.path.expanduser('~/lwir_usb5v/deadline'), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
                try: os.write(fd_, ('%.3f' % (time.monotonic() + 15)).encode())
                finally: os.close(fd_)
            except Exception:
                if need_: raise   # no deadline = the helper would refuse anyway
            p_ = APP.lv_sp.Popen(['sudo', '-n', '/usr/local/sbin/lwir_usb5v_cycle'], stdin=APP.lv_sp.DEVNULL, stdout=APP.lv_sp.PIPE, stderr=APP.lv_sp.STDOUT)
            try: o_ = p_.communicate(timeout=30)[0]
            except APP.lv_sp.TimeoutExpired:
                try: p_.kill()
                except Exception: pass
                return None, b''
            return p_.returncode, o_
        try: c_, o_ = run_()
        except Exception as e_: return 'no', 'USB 전원 껐다 켜기 실패(%s). 케이블을 뽑았다 꽂으세요. ' % type(e_).__name__
        if c_ == 7:   # power went off and did not come back: the helper only turns it on when it finds it off
            camera_extra.say('CAMRESET usb power: code 7, turning on once more'); time.sleep(1)
            try: c_, o_ = run_(False)   # turning on needs no deadline
            except Exception: c_ = 7
            if c_ != 0: return 'off', 'USB 전원이 다시 켜지지 않음. 버튼을 다시 누르고, 그래도 안 되면 보드를 재부팅하세요. '
        if c_ is None: return 'off', 'USB 전원 단계가 30초 안에 끝나지 않음(전원이 꺼졌다 켜졌는지 알 수 없음. 늦게 시작돼도 전원은 끄지 않음). 다시 눌러 보세요. '
        if c_ != 0:
            camera_extra.say('CAMRESET usb power (code %d): %s' % (c_, o_.decode('utf-8', 'replace').strip()[:200]))
            if c_ == 1: return 'no', 'USB 전원 껐다 켜기 실패(보드에 sudo 허용 설정 필요). 케이블을 뽑았다 꽂으세요. '
            if c_ in (2, 3, 4, 5, 6, 8): return 'no', 'USB 전원 껐다 켜기 실패(코드 %d, 전원은 건드리지 않음). 케이블을 뽑았다 꽂으세요. ' % c_
            return 'off', 'USB 전원 단계가 비정상 종료(코드 %d, 전원이 꺼져 있을 수 있음). 버튼을 다시 누르고(꺼져 있으면 켜기만 함), 그래도 안 되면 보드를 재부팅하세요. ' % c_
        if APP.GB is not None: APP.GB.usb_hold = False   # power is back: manual gimbal moves are allowed again while the camera is looked for
        for _ in range(10):   # the camera needs a few seconds to show up again
            time.sleep(2)
            if APP.cam_usb_seen(): return 'cut', 'USB 전원 껐다 켬, 카메라 다시 보임. '
        return 'cut', 'USB 전원을 껐다 켰지만 카메라가 보이지 않음 (케이블을 뽑았다 꽂으세요). '
    except Exception as e_: return 'off', 'USB 전원 단계 오류(%s). ' % type(e_).__name__
    finally:
        if APP.GB is not None: APP.GB.usb_hold = False


def cam_reset_work():   # v33: runs in its own thread. LIVE.cam_hold keeps lv_cam_wait from starting a second cam_pipe meanwhile.
    say_ = lambda t_, busy_=True: (setattr(APP.LIVE, 'cam_reset', dict(busy=busy_, text=t_, at=time.strftime('%H:%M:%S'))), camera_extra.say('CAMRESET ' + t_))
    try:
        dead_ = time.monotonic() - APP.LIVE.last_frame_t > 10
        t0_ = getattr(APP.LIVE, 'cam_reset_t', None)   # v35: when the last reset started cam_pipe again
        el_ = None if t0_ is None or APP.LIVE.last_frame_t >= t0_ else time.monotonic() - t0_   # seconds since that reset, only while no frame came after it
        if dead_ and el_ is not None and el_ < 90 and getattr(APP.LIVE, 'cam_reset_lock', True): say_('방금 초기화함(%d초 전). 영상을 기다리는 중. 90초가 지나도 안 나오면 다시 누르세요' % el_, False); return
        again_ = dead_ and el_ is not None   # an earlier reset brought no frame -> 5 V power cycle
        def usbreset_():
            say_('USB 초기화 중')
            try: r_ = APP.lv_sp.run(['usbreset', '0547:0080'], stdout=APP.lv_sp.PIPE, stderr=APP.lv_sp.STDOUT, timeout=20); t_ = 'USB 초기화 %s. ' % ('완료' if r_.returncode == 0 else '실패(코드 %d)' % r_.returncode)
            except Exception as e_: t_ = 'USB 초기화 실패(%s). ' % type(e_).__name__
            time.sleep(3); return t_
        say_('cam_pipe 종료 중'); stuck_ = False
        if APP.LIVE.proc.poll() is None:
            APP.LIVE.proc.terminate()
            try: APP.LIVE.proc.wait(45)
            except Exception:
                if not dead_: say_('cam_pipe 가 45초 안에 끝나지 않음. 강제 종료는 하지 않음. 잠시 뒤 다시 누르세요', False); return
                stuck_ = True; say_('cam_pipe 가 45초 안에 끝나지 않음. USB 상태 확인 중')   # hung inside a USB call: only the power cut can end it
        time.sleep(1.5)   # the old reader thread sees end-of-file first
        seen_ = APP.cam_usb_seen(); usb_ = ''; st_ = None
        if dead_ and seen_ and not again_ and not stuck_: usb_ = usbreset_(); seen_ = APP.cam_usb_seen()
        if dead_ and seen_ is False: say_('카메라가 USB 에 보이지 않음. 다시 보이는지 12초 확인 중')
        if dead_ and ((seen_ is False and APP.cam_usb_gone()) or (again_ and seen_)):
            say_('USB 전원(5V) 껐다 켜는 중')
            st_, t_ = APP.cam_usb_power(); usb_ += t_
            if st_ == 'no' and again_ and seen_ and not stuck_: usb_ += usbreset_()   # the power step was refused: at least the old USB reset
        if stuck_:
            killed_ = False
            if st_ == 'cut':   # the camera lost its power, so the blocked call has ended or never will
                try: APP.LIVE.proc.wait(20)
                except Exception:
                    camera_extra.say('CAMRESET cam_pipe still there after the power cycle, SIGKILL'); APP.LIVE.proc.kill(); killed_ = True
                    try: APP.LIVE.proc.wait(5)
                    except Exception: pass
            if APP.LIVE.proc.poll() is None:
                APP.LIVE.cam_reset_t = time.monotonic(); APP.LIVE.cam_reset_lock = False   # counts as a try: the next press goes to the power cycle, without the 90 s wait
                tail_ = ('다시 누르면 USB 전원 껐다 켜기를 시도합니다' if st_ is None and seen_ else '위 문제를 해결한 뒤 다시 누르세요' if st_ == 'no' else 'USB 상태를 확인하지 못해 전원을 끄지 않음. 다시 눌러 보세요' if st_ is None else '')
                say_(usb_ + ('cam_pipe 가 강제 종료에도 끝나지 않음. 다시 눌러도 안 되면 보드를 재부팅하세요' if killed_ else 'cam_pipe 가 끝나지 않음. 지금은 강제 종료하지 않음(영상이 5분 넘게 없으면 서버가 자동으로 강제 종료 후 다시 시작함). ' + tail_), False); return
        if hasattr(APP.LIVE, 'cam_eof_t'): del APP.LIVE.cam_eof_t
        APP.LIVE.cam_fail_n = 0; APP.LIVE.cam_unrec_n = 0; APP.LIVE.cam_next_spawn = None
        with APP.LIVE.cam_lock:   # live_close takes the same lock: no new cam_pipe after the server began to stop
            if APP.LIVE.stopped: say_('서버 종료 중. cam_pipe 를 다시 시작하지 않음', False); return
            APP.lv_spawn(); APP.LIVE.cam_reset_t = time.monotonic(); APP.LIVE.cam_reset_lock = st_ in (None, 'cut')   # v35: a refused or failed power step does not block the next press
        say_(usb_ + ('cam_pipe 다시 시작함. 영상은 10~60초 뒤' if st_ != 'off' and APP.cam_usb_seen() is not False else 'cam_pipe 는 다시 시작했지만 카메라가 USB 에 보이지 않아 영상이 나오지 않을 수 있음'), False)
    except Exception as e_: say_('초기화 오류: %s' % type(e_).__name__, False)
    finally: APP.LIVE.cam_hold = False


def cam_reset_request():
    with APP.LIVE.sys_lock, APP.LIVE.cam_lock:   # cam_lock: the main loop is not inside lv_cam_wait once the hold flag is set
        if APP.LIVE.stopped or APP.args.live_wait: return False, '이 실행 방식에서는 쓸 수 없음 (--live-wait 0 필요)'
        if APP.LIVE.cam_hold: return False, '이미 초기화 중'
        if APP.LIVE.meas is not None: return False, '측정 중에는 초기화하지 않음'
        if replay_link.blocking(): return False, '녹화 재생 중에는 카메라 초기화를 하지 않음. 먼저 재생을 멈추세요'   # 11-12
        APP.LIVE.cam_hold = True
    APP.lv_th.Thread(target=APP.cam_reset_work, daemon=True).start(); return True, '카메라 초기화 시작'
