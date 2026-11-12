"""화면의 시스템 버튼."""
import os
import time
from ..common.state import APP
from ..detect import frame_feed
from .preview_worker import invalidate_preview


RESTART_REQUESTED = object()


def restart_conflict(live):
    if not getattr(live, 'restart_supported', False) or live.stopped:
        return '표준 실행 스크립트로 서버를 시작한 뒤 사용할 수 있습니다'
    if live.cam_hold or live.restart_pending:
        return '시스템 작업이 진행 중입니다'
    if live.meas is not None or getattr(live, 'replay', None) is not None:
        return '측정 또는 녹화 재생을 종료한 뒤 다시 시도하세요'
    if APP.GB is not None and (APP.GB.state.get('auto') or APP.GB.state.get('moving')):
        return '짐벌이 멈춘 뒤 다시 시도하세요'
    return None


def npu_feed_state():
    state = frame_feed.status()
    reason = frame_feed.switch_conflict()
    if reason is None and state['pending']:
        reason = 'NPU 검출 모드를 바꾸고 있습니다'
    if reason is None and state['requested'] == 'normal' and len(APP.FEED.eng) != 2:
        reason = 'NPU 1에 검출 모델이 올라가 있지 않습니다'
    state.update(can_toggle=reason is None, toggle_reason=reason)
    return state


def request_npu_feed(mode):
    with APP.LIVE.sys_lock, APP.MODEL['lock']:
        reason = frame_feed.switch_conflict()
        if reason is not None:
            return False, reason
        return APP.FEED.request_mode(mode)


def reserve_restart():
    live = APP.LIVE
    with live.sys_lock, live.cam_lock, live.meas_lock:
        reason = restart_conflict(live)
        if reason is not None:
            return False, reason
        feed = vars(APP).get('FEED')
        if feed is not None and feed.pending is not None:
            return False, 'NPU 검출 모드를 바꾸고 있습니다'
        live.restart_pending = True
        live.cam_hold = True
        invalidate_preview(live, before=time.perf_counter())
    return True, '서버 재시작 요청을 받았습니다'


def stop_for_restart():
    live = APP.LIVE
    print('Tracking server restart requested', flush=True)
    with live.cond:
        # Only the frame loop can mark the stream stopped. An in-flight frame
        # still uses the live image index until its camera-motion work is queued.
        live.restart_stop = True
        live.cond.notify_all()


def sys_reboot():   # v33: reboot button. Same command as the keepalive's last rung (no sudo; polkit allows it for this user).
    try: up_ = float(open('/proc/uptime').read().split()[0])
    except Exception: up_ = 1e9
    if up_ < 300: return False, '부팅한 지 %d초. 5분 안에는 다시 재부팅하지 않음' % up_
    with APP.LIVE.sys_lock, APP.LIVE.cam_lock:   # cam_lock: the main loop is not inside lv_cam_wait once the hold flag is set
        if APP.LIVE.stopped or APP.args.live_wait: return False, '이 실행 방식에서는 쓸 수 없음 (--live-wait 0 필요)'
        if APP.LIVE.cam_hold: return False, '카메라 초기화 중. 끝난 뒤 다시 누르세요'
        APP.LIVE.cam_hold = True
        invalidate_preview(APP.LIVE, before=time.perf_counter())
    print('SYSREBOOT requested from the page', flush=True)
    try:
        if APP.GB is not None: APP.gb_send('H')
    except Exception: pass
    try:
        if APP.LIVE.proc.poll() is None:
            APP.LIVE.proc.terminate()
            try: APP.LIVE.proc.wait(15)
            except Exception: pass
        try:
            with open(os.path.expanduser('~/lwir_page_reboots.log'), 'a') as f_: f_.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
            os.sync()
        except Exception: pass   # a read-only disk must not block the reboot
        r_ = APP.lv_sp.run(['systemctl', 'reboot'], stdout=APP.lv_sp.PIPE, stderr=APP.lv_sp.STDOUT, timeout=30)
        if r_.returncode == 0: return True, '재부팅 시작. 1~2분 뒤 화면이 다시 뜹니다'
        msg_ = '재부팅 명령 실패(코드 %d): %s' % (r_.returncode, r_.stdout.decode('utf-8', 'replace').strip()[:200])
    except Exception as e_: msg_ = '재부팅 명령 오류: %s' % type(e_).__name__
    print('SYSREBOOT failed: ' + msg_, flush=True)
    APP.LIVE.cam_next_spawn = None; APP.LIVE.cam_hold = False   # the camera process was stopped: lv_cam_wait starts it again
    return False, msg_
