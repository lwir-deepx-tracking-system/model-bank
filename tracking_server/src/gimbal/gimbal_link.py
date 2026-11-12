"""짐벌 연결: 짐벌 프로그램(gimbal_bridge) 띄우기와 지켜보기, 명령·프레임 보내기, 따라갈 사람의 예측 제어."""
import math
import time
import numpy as np
from ..common.state import APP
from ..common.shared_values import HERE


def lead_shift(real_, seen_, id_, t_, w_, h_):
    """예측 제어 (target-velocity feedforward, Corke & Good, IEEE T-RA 1996). seen_ = people in picture coordinates (GB.people), real_ = the same people in
        camera coordinates (what the gimbal bridge gets). The followed person's speed against the background (the previous centre is first moved by the camera-motion
        warp prev -> current) is smoothed and the box in real_ is moved ahead by speed x lead_ms. Returns a new list; any error = real_ unchanged."""
    try:
        p_ = next((x_ for x_ in seen_ if x_[0] == id_), None) if id_ else None
        if p_ is None: APP.CTL['lv'] = None; APP.CTL['lead_px'] = 0.0; return real_
        cx_, cy_ = (p_[1] + p_[3]) / 2.0, (p_[2] + p_[4]) / 2.0; lv_ = APP.CTL['lv']; vx_ = vy_ = 0.0
        if lv_ is not None and lv_[0] == id_ and 0.0 < t_ - lv_[1] < 0.25:
            px_, py_ = float(lv_[2]), float(lv_[3]) + 16.0; W_ = APP.CTL['W']
            if W_ is not None:
                W_ = np.asarray(W_, dtype=np.float64)
                if W_.shape[0] >= 2 and W_.shape[1] >= 3 and np.isfinite(W_[:2, :3]).all():
                    px_, py_ = W_[0, 0] * px_ + W_[0, 1] * py_ + W_[0, 2], W_[1, 0] * px_ + W_[1, 1] * py_ + W_[1, 2]
            py_ -= 16.0; dt_ = t_ - lv_[1]
            vx_ = 0.7 * lv_[4] + 0.3 * (cx_ - px_) / dt_; vy_ = 0.7 * lv_[5] + 0.3 * (cy_ - py_) / dt_
        APP.CTL['lv'] = (id_, t_, cx_, cy_, vx_, vy_)
        if (vx_ * vx_ + vy_ * vy_) ** 0.5 < 15.0: APP.CTL['lead_px'] = 0.0; return real_   # slower than 15 px/s: no lead (standing person, box jitter)
        k_ = max(0, min(400, int(APP.CTL['lead_ms']))) / 1000.0
        sx_ = max(-60.0, min(60.0, vx_ * k_)); sy_ = max(-60.0, min(60.0, vy_ * k_)); out_ = []
        for q_ in real_:
            if q_[0] == id_:
                sx_ = max(min(0.0, -q_[1]), min(max(0.0, w_ - q_[3]), sx_)); sy_ = max(min(0.0, -q_[2]), min(max(0.0, h_ - q_[4]), sy_))   # the moved box stays inside the picture
                q_ = (q_[0], q_[1] + sx_, q_[2] + sy_, q_[3] + sx_, q_[4] + sy_) + tuple(q_[5:])
            out_.append(q_)
        APP.CTL['lead_px'] = float((sx_ * sx_ + sy_ * sy_) ** 0.5)
        return out_
    except Exception as e_:
        APP.CTL['lv'] = None; APP.CTL['lead_px'] = 0.0
        if not APP.CTL.get('lead_err'): APP.CTL['lead_err'] = repr(e_)[:120]; print('LEAD failed (box sent unchanged):', APP.CTL['lead_err'], flush=True)
        return real_


def gb_device():   # v26: the device given on the command line; when it is gone (USB dropped and came back under another ttyUSB name), the gimbal's fixed name under /dev/serial/by-id
    if APP.lv_os.path.exists(APP.args.gimbal): return APP.args.gimbal
    c_ = sorted(APP.lv_glob.glob('/dev/serial/by-id/usb-Silicon_Labs_CP2102N_*'))
    return c_[0] if c_ else APP.args.gimbal


def gb_dev_sig(dev_):   # identity of the device node: a new node after a USB drop has another number or inode even under the same name
    try: s_ = APP.lv_os.stat(dev_); return (s_.st_rdev, s_.st_ino)
    except OSError: return None


def gb_in_use(dev_):   # v26b: another program holds the gimbal device. Not fatal any more (the picture must never be off for a gimbal reason): no bridge is started while it is held, and it is tried again every 5 s.
    try: return APP.lv_sp.run(['fuser', dev_], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception: return ''


def gb_cfg():
    """읽기 전용 (GET /rules 의 gimbal_cfg, 화면의 '짐벌 제어기' 구역): 서버가 짐벌 프로그램에 넘기는 옵션을 gb_spawn 과 같은 순서로 다시 만들어 읽는다. 아무것도 바꾸지 않는다.
        짐벌 프로그램처럼 "이름 값" 쌍으로 읽고, 같은 이름이 두 번 나오면 뒤의 것이 이긴다. 값은 넘긴 글자 그대로. 주지 않은 값은 None (짐벌 프로그램의 기본값).
        program 은 파일 이름만. 장치 이름(--device)과 PC GUI 옵션(--gui-*)은 넣지 않는다. 그 밖의 옵션은 --gimbal-opts 에 준 그대로 extra 에 모은다. 짐벌 없이 실행하면 None."""
    try:
        if not APP.args.gimbal: return None
        import shlex
        names_ = {'--controller': 'controller', '--kp': 'kp', '--ki': 'ki', '--kd': 'kd', '--deadband': 'deadband', '--speed-max': 'speed_max', '--manual-speed': 'manual_speed', '--reacquire-ms': 'reacquire_ms', '--aim-y': 'aim_y'}
        c_ = dict(program=str(APP.args.gimbal_bridge or 'gimbal_bridge_v25').replace('\\', '/').rsplit('/', 1)[-1], controller=None, kp=None, ki=None, kd=None, deadband=None, speed_max=None, manual_speed=None, reacquire_ms=None, aim_y=None, extra='')
        a_ = ['--manual-speed', str(APP.args.gimbal_speed), '--speed-max', str(APP.args.gimbal_speed), '--kp', str(APP.args.gimbal_kp), '--aim-y', str(APP.args.gimbal_aim_y), *shlex.split(APP.args.gimbal_opts)]   # gb_spawn 과 같은 순서 (장치 · 축 방향 · 모델 수 · PC GUI 옵션은 뺌)
        x_ = []
        for i_ in range(0, len(a_), 2):
            n_ = a_[i_]; v_ = a_[i_ + 1] if i_ + 1 < len(a_) else None
            if n_ in names_ and v_ is not None: c_[names_[n_]] = v_
            elif n_ != '--device' and not n_.startswith('--gui-'): x_ += [n_] if v_ is None else [n_, v_]
        c_['extra'] = ' '.join(x_)
        return c_
    except Exception as e_: return dict(error=repr(e_)[:80])


class _Gimbal: pass


def reset_lead():
    APP.CTL['lv'] = None
    APP.CTL['lead_px'] = 0.0


def without_lead(frame):
    if frame is None:
        return None
    end = frame.find(b'\n')
    start = frame.find(b' LEAD ', 0, end)
    return frame if start < 0 else frame[:start] + frame[end:]


def _reset_bridge_lead(gb):
    gb.lead_control_supported = False
    gb.lead_pending = 0
    gb.control_epoch = None
    reset_lead()


def gb_send(line_):
    with APP.GB.wcond:
        if not APP.GB.wopen: return False
        if len(APP.GB.wq) >= 64: APP.GB.skipped += 1; return False
        if line_.split(' ', 1)[0] in ('H', 'M', 'V'):
            reset_lead()
            APP.GB.wframe = without_lead(APP.GB.wframe)
        APP.GB.wq.append((line_ + '\n').encode()); APP.GB.wcond.notify()
    return True


def gb_send_frame(header_, picture_):
    with APP.GB.wcond:
        if not APP.GB.wopen: return
        if APP.GB.wframe is not None: APP.GB.skipped += 1
        APP.GB.wframe = (header_ + '\n').encode() + picture_; APP.GB.wcond.notify()


def gb_writer(p_):
    try:
        while True:
            with APP.GB.wcond:
                APP.GB.wcond.wait_for(lambda: APP.GB.wq or APP.GB.wframe is not None or not APP.GB.wopen or APP.GB.proc is not p_)
                if APP.GB.proc is not p_: return   # a newer bridge owns the queue now
                if not APP.GB.wopen and not APP.GB.wq and APP.GB.wframe is None: break
                chunks_ = list(APP.GB.wq); APP.GB.wq.clear()
                if APP.GB.wframe is not None: chunks_.append(APP.GB.wframe); APP.GB.wframe = None
            for c_ in chunks_: p_.stdin.write(c_)
            p_.stdin.flush()
    except (OSError, ValueError): pass
    with APP.GB.wcond:
        if APP.GB.proc is p_:
            APP.GB.wopen = False; APP.GB.wq.clear(); APP.GB.wframe = None
            _reset_bridge_lead(APP.GB)


def parse_pose(fields):
    """POSE pan tilt [pan_min pan_max tilt_min tilt_max], in measured degrees."""
    if len(fields) not in (2, 6):
        return None
    try:
        values = tuple(float(field) for field in fields)
    except ValueError:
        return None
    if not all(math.isfinite(value) and abs(value) <= 360 for value in values):
        return None
    pose = {'pan': {'deg': values[0]}, 'tilt': {'deg': values[1]}}
    if len(values) == 6:
        for axis, low, high in (('pan', values[2], values[3]), ('tilt', values[4], values[5])):
            if low >= high:
                return None
            pose[axis].update(min_deg=low, max_deg=high)
    return pose


def gb_sample(name):
    sample = getattr(APP.GB, name, None)
    if not sample:
        return None
    bridge, sampled_at, pose = sample
    if bridge is APP.GB.proc and APP.GB.state['running'] and not APP.GB.state['error'] and time.monotonic() - sampled_at < 1.5:
        return pose
    return None


def gb_reader(p_):
    while True:
        line_ = p_.stdout.readline()
        if not line_: break
        line_ = line_.decode(errors='replace'); f_ = line_.split()
        if not f_: continue
        if p_ is not APP.GB.proc: continue   # v26: lines of a bridge that was replaced
        if f_[0] == 'READY': APP.GB.state['ready'] = True; print('GIMBAL ready', APP.GB.dev, flush=True)
        elif f_[0] in ('TRACK_CONTROL', 'LEAD_CONTROL') and f_[1:] == ['1']:
            with APP.GB.wcond:
                if p_ is APP.GB.proc and APP.GB.wopen:
                    if f_[0] == 'TRACK_CONTROL': APP.GB.track_control_supported = True
                    else: APP.GB.lead_control_supported = True
        elif (f_[0] == 'CONTROL_EPOCH' and len(f_) == 2 and len(f_[1]) <= 20
              and f_[1].isascii() and f_[1].isdigit()):
            epoch = int(f_[1])
            with APP.GB.wcond:
                if (p_ is APP.GB.proc and epoch <= 0xffffffffffffffff
                        and getattr(APP.GB, 'control_epoch', None) != epoch):
                    APP.GB.control_epoch = epoch
                    reset_lead()
                    APP.GB.wframe = without_lead(APP.GB.wframe)
        elif f_[0] == 'HALTED' and len(f_) == 2:
            from .live_tracks import halted
            try: halted(int(f_[1]))
            except ValueError: pass
        elif f_[0] == 'R' and len(f_) >= 3:
            with APP.GB.wcond:
                if p_ is not APP.GB.proc: continue
                if f_[1] == 'A':
                    APP.GB.lead_pending = max(0, getattr(APP.GB, 'lead_pending', 0) - 1)
                    reset_lead()
                APP.GB.state['answer'] = {'OK': '', 'INVALID': 'that person is not available', 'NOT_READY': 'the gimbal is not ready'}.get(f_[2], f_[2])
            print('GIMBAL', line_.strip(), flush=True)
        elif f_[0] == 'S' and len(f_) >= 6:
            auto, lost, moving, error, track_id = map(int, f_[1:6])
            with APP.GB.wcond:
                if p_ is not APP.GB.proc: continue
                state = APP.GB.state
                if (auto, lost, error, track_id) != tuple(state.get(key, 0) for key in ('auto', 'lost', 'error', 'id')):
                    reset_lead()
                    APP.GB.wframe = without_lead(APP.GB.wframe)
                state.update(auto=auto, lost=lost, moving=moving, error=error, id=track_id)
            print('GIMBAL', line_.strip(), flush=True)
        elif f_[0] == 'POSE':
            APP.GB.pose_sample = (p_, time.monotonic(), parse_pose(f_[1:]))
        elif f_[0] == 'POSITION_EST':
            estimate = parse_pose(f_[1:]) if len(f_) == 3 else None
            if estimate and all(0 <= axis['deg'] <= 100 for axis in estimate.values()):
                percent = {key: axis['deg'] for key, axis in estimate.items()}
                APP.GB.position_sample = (p_, time.monotonic(), percent)
        elif f_[0] == 'M' and len(f_) >= 3:   # v23: the PC GUI asks for a model change: M <req> <id>
            r_ = APP.model_request(int(f_[2]), int(f_[1])); print('MODEL request from the PC GUI: id', f_[2], '->', 'queued' if r_ < 0 else r_, flush=True)
            if r_ >= 0: APP.gb_send('N %d %d' % (int(f_[1]), r_))
        elif f_[0] == 'T' and len(f_) >= 7:   # v20: GUI link status, once a second
            g_ = APP.GB.state['gui']; client_ = int(f_[1])
            if client_ != g_['client']:
                with APP.GB.wcond:
                    if p_ is not APP.GB.proc: continue
                    reset_lead()
                    APP.GB.wframe = without_lead(APP.GB.wframe)
                print('GUI client', 'connected' if client_ else 'disconnected', flush=True)
            g_.update(client=client_, given=int(f_[2]), sent=int(f_[3]), encoded=int(f_[4]), rtp_bytes=int(f_[5]), failed=int(f_[6]))
    # the bridge ended. After a normal end it has sent STOP itself; after a crash the last move would go on, so this stop is sent in any case
    with APP.GB.wcond:
        if p_ is APP.GB.proc:
            _reset_bridge_lead(APP.GB)
            APP.GB.wframe = without_lead(APP.GB.wframe)
    try: rc_ = p_.wait(3)
    except Exception: rc_ = None
    print('GIMBAL bridge ended, code', rc_, flush=True)
    with APP.GB.lock:   # the same lock as gb_spawn: never write to the device while a newer bridge is being started on it
        if p_ is not APP.GB.proc: return
        APP.GB.state.update(running=False, ready=False, auto=0, lost=0, moving=0, id=0)
        with APP.GB.wcond:
            APP.GB.wopen = False; APP.GB.wq.clear(); APP.GB.wframe = None; APP.GB.wcond.notify_all()
        try:
            fd_ = APP.lv_os.open(APP.GB.dev, APP.lv_os.O_WRONLY | APP.lv_os.O_NOCTTY | APP.lv_os.O_NONBLOCK); APP.lv_os.write(fd_, b'{"T":135}\n'); APP.lv_os.close(fd_)
        except OSError: pass


def gb_spawn():   # v26: start (or start again) the bridge. A new bridge sends STOP when it opens the device and follows nobody until a person is clicked again.
    with APP.GB.lock:
        if APP.GB.closing or APP.LIVE.stopped: return False
        dev_ = APP.gb_device(); sig_ = APP.gb_dev_sig(dev_)
        if sig_ is None: return False
        APP.GB.busy = APP.gb_in_use(dev_)
        if APP.GB.busy: return False
        p_ = APP.lv_sp.Popen([APP.args.gimbal_bridge or str(HERE / 'src' / 'gimbal' / 'gimbal_bridge_v25'), '--device', dev_, '--manual-speed', str(APP.args.gimbal_speed), '--speed-max', str(APP.args.gimbal_speed), '--kp', str(APP.args.gimbal_kp), '--aim-y', str(APP.args.gimbal_aim_y),
                          '--x-sign', str(APP.args.gimbal_x_sign), '--y-sign', str(APP.args.gimbal_y_sign), '--model-count', str(len(APP.MODEL['catalog'])), *APP.shlex.split(APP.args.gimbal_opts), *APP.gui_opts_], stdin=APP.lv_sp.PIPE, stdout=APP.lv_sp.PIPE, stderr=APP.GB.log)
        APP.GB.dev = dev_; APP.GB.dev_sig = sig_; APP.GB.spawn_t = time.monotonic(); APP.GB.restarts += 1
        APP.GB.state.update(running=True, ready=False, auto=0, lost=0, moving=0, error=0, id=0); APP.GB.state['gui'].update(client=0)
        with APP.GB.wcond:
            APP.GB.track_control_supported = False
            APP.GB.people_source = object()  # a replacement bridge needs a fresh STOP acknowledgement
            _reset_bridge_lead(APP.GB)
            APP.GB.wq.clear(); APP.GB.wframe = None; APP.GB.proc = p_; APP.GB.wopen = True; APP.GB.wcond.notify_all()
        APP.lv_th.Thread(target=APP.gb_writer, args=(p_,), daemon=True).start(); APP.lv_th.Thread(target=APP.gb_reader, args=(p_,), daemon=True).start()
        print('GIMBAL bridge started on %s (start number %d)' % (dev_, APP.GB.restarts + 1), flush=True)
    return True


def gb_end(p_, why_):   # ask a bridge to end (on SIGTERM it sends STOP and leaves; its input pipe is not closed from here because the writer thread may be inside a write), kill it if it does not
    print('GIMBAL bridge is being restarted:', why_, flush=True)
    with APP.GB.wcond:
        if APP.GB.proc is p_:
            APP.GB.wopen = False; APP.GB.wq.clear(); APP.GB.wframe = None; APP.GB.wcond.notify_all()
            _reset_bridge_lead(APP.GB)
    p_.terminate()
    try: p_.wait(5)
    except Exception:
        p_.kill()
        try: p_.wait(5)
        except Exception: pass


def gb_watch():   # v26: keeps the gimbal link alive without restarting the server (the picture never stops for a gimbal problem)
    wait_i_ = 0; next_ = 0.0; bad_ = 0; told_ = False; told_busy_ = False
    while not APP.LIVE.stopped and not APP.GB.closing:
        time.sleep(1.0)
        try:
            now_ = time.monotonic(); p_ = APP.GB.proc
            if p_ is not None and p_.poll() is None:
                if APP.GB.state['ready'] and not APP.GB.state['error'] and now_ - APP.GB.spawn_t > 60: wait_i_ = 0   # a bridge that works for a minute clears the back-off
                why_ = 'the serial link reports an error' if APP.GB.state['error'] else 'the gimbal device was unplugged or came back as a new device' if APP.gb_dev_sig(APP.GB.dev) != APP.GB.dev_sig else ''
                bad_ = bad_ + 1 if why_ else 0
                if bad_ >= 3: bad_ = 0; APP.gb_end(p_, why_)
                continue
            if now_ < next_: continue
            if APP.gb_dev_sig(APP.gb_device()) is None:
                if not told_: told_ = True; print('GIMBAL device is not there (%s): waiting for it, the tracker keeps running' % APP.args.gimbal, flush=True)
                continue
            told_ = False
            if APP.gb_spawn(): next_ = time.monotonic() + (2, 5, 10, 30)[min(wait_i_, 3)]; wait_i_ += 1; bad_ = 0; told_busy_ = False
            else:
                next_ = time.monotonic() + 5
                if APP.GB.busy and not told_busy_: told_busy_ = True; print('GIMBAL device %s is in use by process %s: waiting for it to be free, the tracker keeps running' % (APP.GB.dev if APP.GB.proc is not None else APP.gb_device(), APP.GB.busy), flush=True)
        except Exception as e_: print('GIMBAL watch error', repr(e_), flush=True)


def gb_mover_watch():   # v26b: a manual move belongs to the page that asked for it. When that page stops sending (closed, connection lost) the move is stopped within 1.5 s, even if another open page keeps the shared heartbeat alive.
    while not APP.LIVE.stopped and not APP.GB.closing:
        time.sleep(0.2)
        m_ = APP.GB.mover
        if m_ is not None and time.monotonic() - APP.GB.mover_t.get(m_, 0.0) > 1.5:
            APP.GB.mover = None; APP.gb_send('V 0 0'); print('GIMBAL manual move stopped: the page that started it (%s) went silent' % m_, flush=True)
