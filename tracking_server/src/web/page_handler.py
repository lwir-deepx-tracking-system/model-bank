"""웹 화면: 보여 줄 그림 만들기(live_publish), 페이지 요청 처리(lv_Handler, lv_Server), 서버 끝내기(live_close)."""
from colorsys import hsv_to_rgb
from functools import lru_cache
import json
import math
import os
import time
from urllib.parse import parse_qs, urlsplit
import numpy as np
from ..common.state import APP
from ..common.shared_values import HERE
from ..gimbal.gimbal_link import gb_sample
from ..gimbal import live_tracks
from ..track.position_memory import POSP
from ..track.stabilizer import STAB, stab_rows_real
from ..camera import replay_link
from . import system_buttons
from . import preview_worker
from ..camera import replay_eval   # 11-12: ground-truth comparison while a labelled recording is replayed


def init_preview(live, *, transition_jpeg=None):
    async_live = os.environ.get('LWIR_ASYNC_PREVIEW') == '1'
    async_replay = os.environ.get('LWIR_ASYNC_REPLAY_PREVIEW') == '1'
    if async_live or async_replay:
        if transition_jpeg is None:
            transition_jpeg = preview_worker.transition_jpeg(APP.cv2)
        live.preview_worker = preview_worker.PreviewWorker(live, APP.cv2, transition_jpeg=transition_jpeg,
                                                           async_live=async_live, async_replay=async_replay)


def _preview_snapshot(g_, t_in, sc_, draw_, gtd_, sC_, overlay_rows, camera_rows,
                      rows_, row_, npers_, now_, fps_, gb_id_, *, owned, click_people=None,
                      click_ticket=None, replay_position=None):
    gt_boxes = ()
    if gtd_ is not None and replay_eval.DRAW[0]:
        gt_boxes = tuple(tuple(b) for b in replay_eval.draw_boxes(t_in, gtd_, sC_))
    predictions = []
    if overlay_rows is None and camera_rows is None and POSP['show'] and now_ - POSP['boxes_t'] < 0.5:
        try:
            seen_ = {int(r_[4]) for r_ in rows_}
            for tid_, b_ in POSP['boxes']:
                if tid_ in seen_ or (APP.GB is not None and tid_ != gb_id_): continue
                pc_ = (0, 220, 255) if gb_id_ and tid_ == gb_id_ else track_color(tid_)
                p1_ = (int((b_[0] - b_[2] / 2) * sc_), int((b_[1] - b_[3] / 2 - 16) * sc_)); p2_ = (int((b_[0] + b_[2] / 2) * sc_), int((b_[1] + b_[3] / 2 - 16) * sc_))
                predictions.append((p1_, p2_, pc_, int(tid_)))
        except Exception: pass
    txt_ = 'TRACK %.0f fps | CAM %.0f fps | delay %.0f ms | compute %.0f ms | persons %d | newID %d relink %d' % (fps_, APP.LIVE.cam_fps, (now_ - t_in) * 1000, row_['frame_total_ms'], npers_, APP.relabeler.stats['new_ids'] if APP.relabeler is not None else 0, APP.relabeler.stats['relabelled'] if APP.relabeler is not None else 0)
    rp_ = getattr(APP.LIVE, 'replay', None)
    if camera_rows is not None:
        txt_ = 'ByteTrack C++ | TRACK %.0f fps | CAM %.0f fps | delay %.0f ms | persons %d' % (fps_, APP.LIVE.cam_fps, (now_ - t_in) * 1000, npers_)
    if rp_ is not None: txt_ = '%s | TRACK %.0f fps | CAM %.0f fps | delay %.0f ms | persons %d' % (replay_link.top_text(rp_), fps_, APP.LIVE.cam_fps, (now_ - t_in) * 1000, npers_)
    gtxt_, gc_ = None, (255, 255, 255)
    if APP.GB is not None and rp_ is None:
        st_ = APP.GB.state
        gtxt_ = 'gimbal: not running' if not st_['running'] else 'gimbal: ERROR' if st_['error'] else ('gimbal: lost %d, waiting' % st_['id'] if st_['lost'] else 'gimbal: following %d' % st_['id'] if st_['auto'] else 'gimbal: manual, moving' if st_['moving'] else 'gimbal: idle, click a person')
        if APP.GB.gui: gu_ = st_['gui']; gtxt_ += ' | GUI %s: sent %d, failed %d' % ('connected' if gu_['client'] else 'no client', gu_['sent'], gu_['failed'])
        if st_['auto']: gc_ = (0, 220, 255)
    gray_ = g_[16:496]
    if owned:
        # bytes owns an immutable copy; the ndarray view cannot re-enable writes.
        gray_ = np.frombuffer(gray_.tobytes(), dtype=gray_.dtype).reshape(gray_.shape)
    return preview_worker.PreviewFrame(t_in, gray_, sc_, tuple(draw_), gt_boxes,
                                       tuple(predictions), txt_, rp_ is not None, gtxt_, gc_,
                                       click_people, click_ticket, replay_position)


def _click_live(x, y):
    live = APP.LIVE
    worker = getattr(live, 'preview_worker', None)
    if worker is None or not worker.async_live:
        return live_tracks.click(x, y)
    with live.jcond:
        frame_t = getattr(live, 'jpeg_frame_t', None)
        people = getattr(live, 'jpeg_people', None)
        ticket = getattr(live, 'jpeg_tracks_ticket', None)
        if (frame_t is None or people is None or ticket is None
                or not 0 <= time.perf_counter() - frame_t < 0.5
                or live.stopped or getattr(live, 'cam_hold', False)
                or getattr(live, 'cam_lost', False) or replay_link.frame_blocked(frame_t)):
            return False, '최신 카메라 화면을 기다려 주세요'
        return live_tracks.click(x, y, expected_id=live_tracks.hit_id(people, x, y),
                                 expected_ticket=ticket)


def _record_perf(live, now, frame_t, fps, compute_ms):
    acc = live.perf_acc
    acc[1] += 1
    acc[4] += compute_ms
    if frame_t is not None:
        delay = (now - frame_t) * 1000.0
        acc[2] += delay
        acc[3] = max(acc[3], delay)
        live.perf_paired_n = getattr(live, 'perf_paired_n', 0) + 1
    if now - acc[0] >= 1.0:
        paired = getattr(live, 'perf_paired_n', 0)
        avg_delay = round(acc[2] / paired, 1) if paired else None
        max_delay = round(acc[3], 1) if paired else None
        with live.dlock:
            live.perf.append((round(time.time(), 1), round(fps, 1), round(float(live.cam_fps), 1),
                              avg_delay, max_delay, round(acc[4] / acc[1], 1), paired))
        live.perf_acc = [now, 0, 0.0, 0.0, 0.0]
        live.perf_paired_n = 0


def frozen_check(raw_):
    """11-12: True when this 16-bit frame is exactly the previous one (every 4th pixel in both directions). A working thermal camera never gives
    two identical frames (sensor noise), so this only fires when the picture is frozen while frames still arrive. Logs the start and end of each frozen spell."""
    L_ = APP.LIVE
    if raw_ is None: return False
    s_ = raw_[::4, ::4]; p_ = getattr(L_, 'frozen_prev', None)
    same_ = p_ is not None and p_.shape == s_.shape and np.array_equal(p_, s_)
    L_.frozen_prev = s_.copy(); n_ = getattr(L_, 'frozen_n', 0)
    if same_:
        if n_ == 0: print('FROZEN picture: this frame is the same as the previous one; the gimbal gets no people until the picture moves', flush=True)
        L_.frozen_n = n_ + 1; L_.frozen_total = getattr(L_, 'frozen_total', 0) + 1
    elif n_:
        print('FROZEN picture ended after %d frames' % n_, flush=True); L_.frozen_n = 0
    return same_


def live_gimbal_position():
    """Return position only when the published JPEG is from the current live camera."""
    frame_t = getattr(APP.LIVE, 'jpeg_frame_t', None)
    if frame_t is None or replay_link.frame_blocked(frame_t):
        return dict(pose=None, position_estimate=None)
    return dict(pose=gb_sample('pose_sample'), position_estimate=gb_sample('position_sample'))


@lru_cache(maxsize=512)
def track_color(track_id):
    # Golden-angle spacing gives consecutive IDs distinguishable hues.
    r, g, b = hsv_to_rgb((track_id * 0.61803398875) % 1.0, 0.72, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def live_publish(g_, rows_, row_, detections=None, tracker_detections=None):
    paired_stamp_ = bool(APP.LIVE.arr)
    now_ = time.perf_counter(); t_in = APP.LIVE.arr.popleft() if paired_stamp_ else now_
    if not isinstance(t_in, (int, float)) or not math.isfinite(t_in) or not 0 < t_in <= now_:
        # Keep UI timing arithmetic valid, but never turn fabricated time into OBS1.
        paired_stamp_ = False
        t_in = now_
    worker_ = getattr(APP.LIVE, 'preview_worker', None)
    replay_ = getattr(APP.LIVE, 'replay', None)
    async_live_ = worker_ is not None and worker_.async_live and replay_ is None
    async_replay_ = replay_ is not None and replay_link.async_preview(replay_, worker_)
    token_ = (worker_.token(t_in, replay=replay_ is not None, source=replay_)
              if worker_ is not None and paired_stamp_ else None)
    replay_source_ = None
    if worker_ is None and replay_ is not None and paired_stamp_:
        with APP.LIVE.jcond:
            replay_source_ = preview_worker.replay_preview_source(APP.LIVE, t_in)
    cam_tracker = getattr(APP.LIVE, 'camera_bytetrack', None)
    camera_rows = (cam_tracker.step(tracker_detections, t_in,
                   not replay_link.frame_blocked(t_in))
                   if cam_tracker is not None else None)
    display_rows = rows_ if camera_rows is None else camera_rows
    now_ = time.perf_counter()
    raw16_ = APP.LIVE.raws.popleft() if APP.LIVE.raws else None   # v20
    rq_ = APP.LIVE.rawq.popleft() if APP.LIVE.rawq else None   # v24f
    raw_valid_ = (isinstance(rq_, np.ndarray) and rq_.dtype == np.uint16
                  and rq_.shape == (APP.LIVE_H, APP.LIVE_W))
    rep_ = replay_eval.is_repeat(t_in)   # 11-12: a frame the replay program repeats (while paused, and the few at the end) is not a camera fault
    frozen_ = False if rep_ or not raw_valid_ else frozen_check(rq_)   # Invalid/missing RAW is separately denied below.
    sC_ = APP.LIVE.stabq.popleft() if APP.LIVE.stabq else None; STAB['px'] = sC_ or (0.0, 0.0)   # stab: correction of the frame being shown
    APP.LIVE.done_ts.append(now_)
    if APP.LIVE.meas is not None:   # v22
        with APP.LIVE.meas_lock:
            if APP.LIVE.meas is not None: APP.LIVE.meas['boxes'].append((t_in, [(float(b_[0]), float(b_[1]) - 16.0, float(b_[2]), float(b_[3]) - 16.0, int(b_[4]), float(b_[5]), int(b_[6])) for b_ in stab_rows_real(display_rows, sC_)]))   # v22d
            if APP.LIVE.meas is not None: APP.LIVE.meas['rows'].append((now_, (now_ - t_in) * 1000.0, float(row_.get('frame_total_ms', 0.0)), float(row_.get('pre_model_ms', 0.0)), float(row_.get('tracker_ms', 0.0)), float(APP.LIVE.cam_fps)))
    fps_ = (len(APP.LIVE.done_ts) - 1) / (APP.LIVE.done_ts[-1] - APP.LIVE.done_ts[0]) if len(APP.LIVE.done_ts) > 2 and APP.LIVE.done_ts[-1] > APP.LIVE.done_ts[0] else 0.0
    # v26c: track output timing, independent of whether a JPEG is made.
    _record_perf(APP.LIVE, now_, t_in if paired_stamp_ else None, fps_, float(row_.get('frame_total_ms', 0.0)))
    sc_ = APP.args.live_scale
    # 11-12: boxes are collected first and drawn after the gimbal line below has gone out (the page picture takes about 6 ms on the board)
    npers_ = 0; gb_people_ = []; gb_id_ = APP.GB.state['id'] if APP.GB is not None else 0; draw_ = []
    for r_ in display_rows:
        x1_, y1_, x2_, y2_ = [float(v_) for v_ in r_[:4]]; tid_, cls_ = int(r_[4]), int(r_[6]); y1_ -= 16; y2_ -= 16
        if cls_ != 0: continue   # v16: people only
        col_ = (0, 220, 255) if gb_id_ and tid_ == gb_id_ else track_color(tid_); npers_ += 1
        if APP.GB is not None and len(gb_people_) < 60: gb_people_.append((tid_, x1_, y1_, x2_, y2_, float(r_[5])))
        if gb_id_ and tid_ != gb_id_: continue   # v39 (user 11-10): while one person is followed only that person's box is drawn
        draw_.append(((int(x1_ * sc_), int(y1_ * sc_)), (int(x2_ * sc_), int(y2_ * sc_)), col_, tid_))
    healthy_ = False
    if APP.GB is not None:
        source_ = ('replay',) if replay_link.frame_blocked(t_in) else live_tracks.current_source()
        healthy_ = (paired_stamp_ and tracker_detections is not None and raw_valid_
                    and not rep_ and not frozen_ and not replay_link.frame_blocked(t_in)
                    and not any(getattr(APP.LIVE, flag, False) for flag in ('cam_lost', 'cam_hold', 'stopped', 'restart_stop', 'restart_pending')))
        gb_id_ = live_tracks.publish(gb_people_, sC_, t_in, raw16_, source_,
                                    frozen_ or replay_link.frame_blocked(t_in), healthy_live=healthy_)
        # The same live rows now drive the picture, click list and PTZ input.
        draw_ = [((int(r[0] * sc_), int((r[1] - 16) * sc_)),
                  (int(r[2] * sc_), int((r[3] - 16) * sc_)),
                  (0, 220, 255) if gb_id_ and int(r[4]) == gb_id_ else track_color(int(r[4])), int(r[4]))
                 for r in display_rows if int(r[6]) == 0 and (not gb_id_ or int(r[4]) == gb_id_)]
    preview_receipt = []
    receipt_args = {'preview_receipt': preview_receipt} if async_replay_ else {}
    gtd_, replay_rows = replay_eval.on_frame(t_in, rows_, sC_, row_.get('image_id'), detections,
                                            tracker_detections, **receipt_args)
    position = preview_receipt[0] if preview_receipt else None
    overlay_rows = replay_rows
    if overlay_rows is not None:
        rows_ = overlay_rows
        draw_ = [((int(r[0] * sc_), int((r[1] - 16) * sc_)),
                  (int(r[2] * sc_), int((r[3] - 16) * sc_)), track_color(int(r[4])), int(r[4]))
                 for r in rows_ if int(r[6]) == 0]
        npers_ = len(draw_)
    # 11-12: the page picture only when somebody can see it: every frame while /stream has a viewer, otherwise once a second (/snap.jpg at most 1 s old)
    # and every frame for 3 s after a /snap.jpg request. Nothing else reads this picture (tracking, gimbal, measurement and the PC GUI use the frame itself).
    preview_allowed_ = (token_ is not None if worker_ is not None else
                        replay_ is None or replay_source_ is not None)
    required = async_replay_ and worker_.replay_required(position, replay_.get('frames') or 0)
    if preview_allowed_ and (required or APP.LIVE.viewers > 0
            or now_ - APP.LIVE.pic_t >= 1.0 or now_ - APP.LIVE.snap_t < 3.0):
        click_people, click_ticket = None, None
        if async_live_ and healthy_ and overlay_rows is None:
            click_people, click_ticket = live_tracks.preview_people(t_in, {d[3] for d in draw_})
        frame_ = _preview_snapshot(g_, t_in, sc_, draw_, gtd_, sC_, overlay_rows, camera_rows,
                                   rows_, row_, npers_, now_, fps_, gb_id_, owned=async_live_ or async_replay_,
                                   click_people=click_people, click_ticket=click_ticket, replay_position=position)
        if async_live_ or async_replay_:
            worker_.submit(token_, frame_, requested_at=now_)
        else:
            try:
                jpeg_ = preview_worker.render_preview(APP.cv2, frame_)
            except Exception:
                jpeg_ = None
            if worker_ is not None:
                worker_.commit_inline(token_, t_in, jpeg_, requested_at=now_, replay=replay_ is not None)
            else:
                with APP.LIVE.jcond:
                    source_ok_ = (replay_ is None or (
                        preview_worker.replay_preview_source(APP.LIVE, t_in) is replay_source_))
                    if source_ok_:
                        APP.LIVE.pic_t = now_
                        if jpeg_:
                            preview_worker.publish_jpeg_locked(APP.LIVE, jpeg_, t_in, paired=paired_stamp_)
                        else:
                            APP.LIVE.preview_errors = getattr(APP.LIVE, 'preview_errors', 0) + 1
    if time.monotonic() - APP.LIVE.t0 > APP.args.live_seconds or (APP.LIVE.cam_lost and APP.args.live_wait):
        if worker_ is not None: worker_.close()
        APP.LIVE.stopped = True
    if APP.args.mamba and len(APP.relabeler.events) and APP.relabeler.events[-1] is not getattr(APP.LIVE, 'rl_last', None):
        new_ = []
        for ev_ in reversed(APP.relabeler.events):
            if ev_ is getattr(APP.LIVE, 'rl_last', None): break
            new_.append(ev_)
        for ev_ in reversed(new_): print('RLEVENT', json.dumps(ev_), flush=True)
        APP.LIVE.rl_last = APP.relabeler.events[-1]
    if APP.relabeler is not None and time.monotonic() - getattr(APP.LIVE, 'rl_t', 0.0) > 20:
        APP.LIVE.rl_t = time.monotonic(); print('RLSTAT', json.dumps(dict(APP.relabeler.stats, last_event=(APP.relabeler.events[-1] if APP.relabeler.events else None), **({'ms': APP.relabeler.timing()} if APP.args.mamba else {}))), flush=True)
    if len(APP.rows) > 6000:   # long live runs: keep only the recent records in memory
        del APP.rows[:3000]; del APP.preds[:3000]; del APP.tracks_out[:3000]
        if APP.relabeler is not None: del APP.tracks_base[:3000]; del APP.relabeler.per_frame[:3000]


def make_lv_Handler():
    class lv_Handler(APP.lv_http.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            # a page on another site cannot send this header without permission, so it cannot move the gimbal through the viewer's browser
            if self.path.startswith('/model/set') and self.headers.get('X-Gimbal') == '1':   # v23: model change from the page (works without the gimbal too)
                from urllib.parse import urlsplit as us_, parse_qs as pq_
                try: mid_ = int(pq_(us_(self.path).query)['id'][0])
                except (KeyError, ValueError, IndexError): self.send_error(400); return
                r_ = APP.model_request(mid_)
                if r_ >= 0:
                    with APP.MODEL['lock']: APP.MODEL['last'] = dict(id=mid_, result=r_, text=APP.MODEL_TEXT.get(r_, '?'), at=time.strftime('%H:%M:%S'))
                b_ = json.dumps(APP.model_state()).encode()
                try: self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                except (BrokenPipeError, ConnectionResetError, OSError): pass
                return
            system_path, _, query = self.path.partition('?')
            if self.path == '/system/restart' or system_path == '/system/npu-feed':
                host = self.headers.get('Host')
                origin = self.headers.get('Origin')
                if not host or self.headers.get('X-Gimbal') != '1' or origin not in (f'http://{host}', f'https://{host}'):
                    self.send_error(403)
                    return
                mode = None
                bad_mode = False
                if system_path == '/system/npu-feed':
                    modes = parse_qs(query, keep_blank_values=True).get('mode', [])
                    bad_mode = len(modes) != 1 or modes[0] not in ('arrival', 'normal')
                    if not bad_mode:
                        mode = modes[0]
                if bad_mode:
                    ok, message = False, 'NPU 검출 모드는 arrival 또는 normal이어야 합니다'
                elif mode is not None:
                    ok, message = system_buttons.request_npu_feed(mode)
                else:
                    ok, message = system_buttons.reserve_restart()
                body = json.dumps(dict(ok=ok, text=message, instance=APP.LIVE.instance), ensure_ascii=False).encode()
                try:
                    self.send_response(400 if bad_mode else (202 if ok else 409))
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    self.wfile.flush()
                except OSError:
                    pass
                finally:
                    if ok and system_path == '/system/restart':
                        system_buttons.stop_for_restart()
                return
            if self.path.startswith('/system/reboot') and self.headers.get('X-Gimbal') == '1':   # v33: reboot button. The page sends confirm=REBOOT after a confirm dialog.
                from urllib.parse import urlsplit as us3_, parse_qs as pq3_
                ok_, text_ = (APP.sys_reboot() if pq3_(us3_(self.path).query).get('confirm', [''])[0] == 'REBOOT' else (False, '확인 값이 없음'))
                b_ = json.dumps(dict(ok=ok_, text=text_), ensure_ascii=False).encode()
                try: self.send_response(200 if ok_ else 409); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                except (BrokenPipeError, ConnectionResetError, OSError): pass
                return
            if self.path.startswith('/camera/reset') and self.headers.get('X-Gimbal') == '1':   # v33: camera reset button
                ok_, text_ = APP.cam_reset_request()
                b_ = json.dumps(dict(ok=ok_, text=text_), ensure_ascii=False).encode()
                try: self.send_response(200 if ok_ else 409); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                except (BrokenPipeError, ConnectionResetError, OSError): pass
                return
            if self.path.startswith('/tracker/camera?') and self.headers.get('X-Gimbal') == '1':
                backend = parse_qs(urlsplit(self.path).query).get('backend', [''])[0]
                cam_tracker = getattr(APP.LIVE, 'camera_bytetrack', None)
                if backend != 'cpp_bytetrack' or cam_tracker is None:
                    self.send_error(400); return
                if replay_link.blocking():
                    ok_, text_ = False, '카메라로 돌아온 뒤 선택하세요'
                elif APP.GB is not None and APP.GB.state.get('auto'):
                    ok_, text_ = False, '현재 따라가기를 멈춘 뒤 선택하세요'
                else:
                    ok_ = cam_tracker.select(True)
                    text_ = '다음 카메라 프레임에 적용' if ok_ else 'C++ ByteTrack 라이브러리가 없습니다'
                b_ = json.dumps(dict(APP.rules_state(), ok=ok_, text=text_), ensure_ascii=False).encode()
                try: self.send_response(200 if ok_ else 409); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                except (BrokenPipeError, ConnectionResetError, OSError): pass
                return
            if self.path.startswith('/rules/set') and self.headers.get('X-Gimbal') == '1':   # v24g: live rule / tracker switches
                from urllib.parse import urlsplit as us2_, parse_qs as pq2_
                ok_, text_ = APP.rules_request(pq2_(us2_(self.path).query))
                b_ = json.dumps(dict(APP.rules_state(), ok=ok_, text=text_), ensure_ascii=False).encode()
                try: self.send_response(200 if ok_ else 400); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                except (BrokenPipeError, ConnectionResetError, OSError): pass
                return
            if self.path.startswith('/measure/') and self.headers.get('X-Gimbal') == '1':   # v22: measurement button (works without the gimbal too)
                try:
                    ln_ = min(int(self.headers.get('Content-Length') or 0), 8192); bd_ = json.loads(self.rfile.read(ln_).decode('utf-8')) if ln_ else {}
                    if not isinstance(bd_, dict): bd_ = {}
                except (ValueError, UnicodeDecodeError): self.send_error(400); return
                if self.path.startswith('/measure/start'): APP.meas_start(bd_.get('name', ''), bd_.get('memo', ''), bd_.get('video', True) is not False)
                elif self.path.startswith('/measure/stop'): APP.meas_stop(bd_.get('memo'))
                elif self.path.startswith('/measure/note'):
                    try: n_ = int(bd_.get('n'))
                    except (TypeError, ValueError): self.send_error(400); return
                    if not APP.meas_note(n_, bd_.get('name'), bd_.get('memo')): self.send_error(404); return
                else: self.send_error(404); return
                b_ = json.dumps(APP.meas_state()).encode()
                try: self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                except (BrokenPipeError, ConnectionResetError, OSError): pass
                return
            if self.path.startswith('/replay/') and self.headers.get('X-Gimbal') == '1':   # 11-12: recording replay (works without the gimbal too)
                from urllib.parse import urlsplit as us4_, parse_qs as pq4_
                u4_ = us4_(self.path)
                if u4_.path == '/replay/start': code_, text_ = replay_link.start_request(pq4_(u4_.query))
                elif u4_.path == '/replay/stop': code_, text_ = replay_link.stop_request()
                elif u4_.path == '/replay/seek':
                    with APP.LIVE.cond:
                        code_, text_ = replay_link.seek_request(pq4_(u4_.query))   # 11-12: jump to a frame (0-based index)
                elif u4_.path == '/replay/pause': code_, text_ = replay_link.pause_request()
                elif u4_.path == '/replay/step': code_, text_ = replay_link.step_request()
                elif u4_.path == '/replay/play': code_, text_ = replay_link.play_request()
                elif u4_.path == '/replay/speed': code_, text_ = replay_link.speed_request(pq4_(u4_.query))   # 11-12: frames per second while playing
                elif u4_.path == '/replay/eval/reset': code_, text_ = replay_link.restart_request()
                elif u4_.path == '/replay/eval/draw': code_, text_ = replay_eval.draw_request(pq4_(u4_.query))   # 11-12: ground-truth boxes on the page picture
                else: self.send_error(404); return
                b_ = json.dumps(dict(ok=code_ == 200, text=text_, state=replay_link.state(), eval=replay_eval.state(), eval_last=replay_eval.last()), ensure_ascii=False).encode()
                try: self.send_response(code_); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                except (BrokenPipeError, ConnectionResetError, OSError): pass
                return
            if APP.GB is None or self.headers.get('X-Gimbal') != '1': self.send_error(403); return
            u_ = urlsplit(self.path); q_ = parse_qs(u_.query)
            try:
                cam_tracker = getattr(APP.LIVE, 'camera_bytetrack', None)
                if u_.path == '/gimbal/click' and cam_tracker is not None and cam_tracker.blocks_selection():
                    self.send_error(409, 'Tracker switch is pending'); return
                if getattr(APP.LIVE, 'restart_pending', False) and (
                    u_.path == '/gimbal/click' or
                    (u_.path == '/gimbal/move' and (int(q_['x'][0]) or int(q_['y'][0])))
                ):
                    self.send_error(409)
                    return
                ip_ = self.client_address[0]; APP.GB.mover_t[ip_] = time.monotonic()
                APP.gb_send('P')
                if u_.path in ('/gimbal/click', '/gimbal/move') and replay_link.blocking() and (u_.path == '/gimbal/click' or int(q_['x'][0]) or int(q_['y'][0])):   # 11-12: no gimbal move while a recording is played (a move of 0 0 = stop is let through)
                    APP.GB.mover = None; APP.GB.state['answer'] = '녹화 재생 중에는 짐벌을 움직이지 않습니다. 재생을 멈춘 뒤 사람을 다시 누르세요'
                    b_ = json.dumps(dict(APP.GB.state, cam_idle=round(time.monotonic() - APP.LIVE.last_frame_t, 1), cam_fail=APP.LIVE.cam_fail_n, refused=True), ensure_ascii=False).encode()
                    self.send_response(409); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_); return
                if getattr(APP.GB, 'usb_hold', False) and u_.path in ('/gimbal/click', '/gimbal/move'): u_ = u_._replace(path='/gimbal/stop')   # v35: USB power is being cycled, no new move
                if u_.path in ('/gimbal/click', '/gimbal/stop'): APP.GB.mover = None
                elif u_.path == '/gimbal/move': APP.GB.mover = ip_ if (int(q_['x'][0]) or int(q_['y'][0])) else None
                if u_.path == '/gimbal/click':
                    x_, y_ = float(q_['x'][0]) * APP.LIVE_W, float(q_['y'][0]) * APP.LIVE_H
                    ok_, answer_ = _click_live(x_, y_)
                    APP.GB.state['answer'] = answer_
                    if not ok_:
                        b_ = json.dumps(dict(APP.GB.state, refused=True), ensure_ascii=False).encode()
                        self.send_response(409); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_); return
                elif u_.path == '/gimbal/stop': APP.GB.state['answer'] = ''; APP.gb_send('H')
                elif u_.path == '/gimbal/move': APP.GB.state['answer'] = ''; APP.gb_send('V %d %d' % (int(q_['x'][0]), int(q_['y'][0])))
                elif u_.path != '/gimbal/ping': self.send_error(404); return
            except (KeyError, ValueError, IndexError): self.send_error(400); return
            try:
                b_ = json.dumps(dict(APP.GB.state, **live_gimbal_position(), cam_idle=round(time.monotonic() - APP.LIVE.last_frame_t, 1), cam_fail=APP.LIVE.cam_fail_n)).encode()   # v24: cam_idle = seconds since the last camera frame; v24d: cam_fail = cam_pipe failures in a row (0 = fine or repairing)
                self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
            except (BrokenPipeError, ConnectionResetError, OSError): pass
        def do_GET(self):
            try:
                if self.path.startswith('/stream'):
                    self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame'); self.send_header('Cache-Control', 'no-cache'); self.end_headers()
                    try:   # v26b: a slow viewer must get the newest picture, not a queue of old ones: small send buffer (about 2 pictures), no Nagle delay; a viewer that takes nothing for 30 s is closed (the page opens the stream again)
                        self.connection.setsockopt(APP.lv_sock.IPPROTO_TCP, APP.lv_sock.TCP_NODELAY, 1); self.connection.setsockopt(APP.lv_sock.SOL_SOCKET, APP.lv_sock.SO_SNDBUF, 65536); self.connection.settimeout(30)
                    except OSError: pass
                    last_ = 0
                    with APP.LIVE.jcond: APP.LIVE.viewers += 1   # 11-12: pictures are made for every frame while this is above 0
                    try:
                     while not APP.LIVE.stopped:
                         with APP.LIVE.jcond:
                             APP.LIVE.jcond.wait_for(lambda: APP.LIVE.seq != last_ or APP.LIVE.stopped, timeout=2)
                             if APP.LIVE.seq == last_: continue
                             last_ = APP.LIVE.seq; jb_ = APP.LIVE.jpeg
                         self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jb_) + jb_ + b'\r\n')
                    finally:
                        with APP.LIVE.jcond: APP.LIVE.viewers -= 1
                elif self.path.startswith('/gimbal/people') and APP.GB is not None:   # v19: read only
                    b_ = json.dumps(dict(state=APP.GB.state, **live_gimbal_position(), people=[dict(id=p_[0], x1=p_[1], y1=p_[2], x2=p_[3], y2=p_[4], conf=p_[5]) for p_ in APP.GB.people], width=APP.LIVE_W, height=APP.LIVE_H)).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/rules'):   # v24g
                    b_ = json.dumps(APP.rules_state(), ensure_ascii=False).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/perf'):   # v26c: read only. Per-second history for the graph: t (unix s), track fps, cam fps, delay mean ms, delay max ms, compute mean ms
                    from urllib.parse import urlsplit as us3_, parse_qs as pq3_
                    try: since_ = float(pq3_(us3_(self.path).query).get('since', ['0'])[0])   # the page asks only for the samples after the last one it has
                    except ValueError: since_ = 0.0
                    with APP.LIVE.dlock: pr_ = [r_ for r_ in APP.LIVE.perf if r_[0] > since_]
                    b_ = json.dumps(dict(now=round(time.time(), 1), delay_kind='tracker_output', cols=['t', 'track_fps', 'cam_fps', 'delay_ms', 'delay_max_ms', 'compute_ms', 'paired_n'], rows=pr_)).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/healthz'):   # v24d: read-only liveness for the keep-alive supervisor (never POST /gimbal/ping from a script: that is the page's safety heartbeat)
                    now_ = time.monotonic(); last_out_ = APP.LIVE.done_ts[-1] if APP.LIVE.done_ts else None
                    try: rss_ = int(open('/proc/self/statm').read().split()[1]) * 4096 // 1048576
                    except Exception: rss_ = None
                    h_ = dict(instance=getattr(APP.LIVE, 'instance', None), up=round(now_ - APP.LIVE.t0, 1), cam_idle=round(now_ - APP.LIVE.last_frame_t, 1), out_idle=round(time.perf_counter() - last_out_ if last_out_ is not None else now_ - APP.LIVE.t0, 1), frames_in=APP.LIVE.frames_in,   # v26: before the first result out_idle is the time since the start (a tracker that hangs before its first result is seen too)
                              cam_fail=APP.LIVE.cam_fail_n, warm_dropped=APP.LIVE.warm_dropped, frozen_frames=getattr(APP.LIVE, 'frozen_total', 0), pictures=APP.LIVE.pic_n, viewers=APP.LIVE.viewers, cam_pipe_alive=APP.LIVE.proc.poll() is None, gimbal_alive=(APP.GB.proc is not None and APP.GB.proc.poll() is None) if APP.GB is not None else None,
                              gimbal_ready=bool(APP.GB.state['ready']) if APP.GB is not None else None, gimbal_error=int(APP.GB.state['error']) if APP.GB is not None else None, gimbal_starts=(APP.GB.restarts + 1) if APP.GB is not None else None,
                              observation_status=bool(getattr(APP.GB, 'observation_status_supported', False)), observation_healthy=bool(getattr(APP.GB, 'observation_ok', False)), observation_good=getattr(APP.GB, 'observation_good', 0), observation_bad=getattr(APP.GB, 'observation_bad', 0),
                              gimbal_device=(APP.gb_dev_sig(APP.gb_device()) is not None) if APP.GB is not None else None, gimbal_busy=bool(APP.GB.busy) if APP.GB is not None else None, rss_mb=rss_, stopped=APP.LIVE.stopped,
                              q_len=len(APP.LIVE.q), q_drop=APP.LIVE.q_drop, gaps=APP.LIVE.gap_n, replay=(getattr(APP.LIVE, 'replay', None) or {}).get('name'), track_fps=round((len(APP.LIVE.done_ts) - 1) / (APP.LIVE.done_ts[-1] - APP.LIVE.done_ts[0]), 1) if len(APP.LIVE.done_ts) > 2 and APP.LIVE.done_ts[-1] > APP.LIVE.done_ts[0] else 0.0, cam_fps=round(APP.LIVE.cam_fps, 1))
                    worker_ = getattr(APP.LIVE, 'preview_worker', None)
                    h_.update(delay_kind='jpeg_ready', async_preview=bool(worker_ and worker_.async_live),
                              async_replay_preview=bool(worker_ and worker_.async_replay),
                              npu_feed=system_buttons.npu_feed_state())
                    h_.update(worker_.stats() if worker_ is not None else dict(preview_dropped=0, preview_errors=getattr(APP.LIVE, 'preview_errors', 0)))
                    cut_ = time.perf_counter() - 30.0
                    with APP.LIVE.dlock: dl_ = [d_ for t_, d_ in APP.LIVE.delays if t_ >= cut_]   # arrival -> successfully published JPEG, last 30 s
                    if dl_: dl_.sort(); h_.update(delay_n=len(dl_), delay_p50=round(dl_[len(dl_) // 2], 1), delay_p95=round(dl_[min(len(dl_) - 1, int(len(dl_) * 0.95))], 1), delay_max=round(dl_[-1], 1))
                    else: h_.update(delay_n=0, delay_p50=None, delay_p95=None, delay_max=None)
                    b_ = json.dumps(h_).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/model'):   # v23
                    b_ = json.dumps(APP.model_state()).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/measure'):   # v22
                    q_n_ = self.path[len('/measure/'):] if self.path.startswith('/measure/') else ''
                    b_ = json.dumps(next((h_ for h_ in APP.LIVE.meas_hist if str(h_['n']) == q_n_), None) if q_n_ else APP.meas_state()).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/replay'):   # 11-12: recordings that can be played, and the replay state
                    from urllib.parse import urlsplit as us5_, parse_qs as pq5_
                    u5_ = us5_(self.path)
                    b_ = json.dumps(replay_eval.results(pq5_(u5_.query)) if u5_.path == '/replay/eval' else replay_link.get_info(pq5_(u5_.query)), ensure_ascii=False).encode()   # 11-12: /replay/eval = the comparison alone (+ eval_last = the numbers of the last finished replay)
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/snap.jpg'):
                    with APP.LIVE.jcond:
                        jb_ = None
                        if not APP.LIVE.stopped:
                            APP.LIVE.snap_t = time.perf_counter()   # 11-12: pictures for every frame during the next 3 s
                            jb_ = APP.LIVE.jpeg
                    if jb_ is None:
                        self.send_error(503)
                        return
                    self.send_response(200); self.send_header('Content-Type', 'image/jpeg'); self.send_header('Content-Length', str(len(jb_))); self.end_headers(); self.wfile.write(jb_)
                else:
                    lp_ = HERE / 'page' / 'live_page.html'   # v22: read on every request, so the page can be changed without restarting the server
                    body_ = lp_.read_bytes() if APP.GB is not None and lp_.exists() else APP.GB_PAGE if APP.GB is not None else ("<!doctype html><html><head><meta charset='utf-8'><title>LWIR live</title><style>body{background:#111;color:#ddd;font-family:sans-serif;margin:0;padding:10px}"
                             "img{max-width:100%;display:block;margin:auto}</style></head><body><div>DX-M1 final tracker on the live camera: boxes + track ids (green = person)</div><img src='/stream'></body></html>").encode()
                    self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8'); self.send_header('Content-Length', str(len(body_))); self.end_headers(); self.wfile.write(body_)
            except (BrokenPipeError, ConnectionResetError, OSError): pass
    return lv_Handler


def make_lv_Server():
    class lv_Server(APP.lv_http.ThreadingHTTPServer):   # v26: IP_FREEBIND (Linux option 15): the page can be bound to the Tailscale address before that address exists (right after a boot, or while Tailscale restarts), so the tracker and the gimbal link start without waiting for the network
        def server_bind(self):
            try: self.socket.setsockopt(0, 15, 1)
            except OSError: pass
            super().server_bind()
    return lv_Server


def live_close():
    worker_ = getattr(APP.LIVE, 'preview_worker', None)
    if worker_ is not None: worker_.close()
    APP.LIVE.stopped = True
    APP.LIVE.camera_bytetrack.close()
    try: APP.meas_stop()   # v22: keep a measurement that is still running
    except Exception as e_: print('MEASURE save failed', e_, flush=True)
    try: replay_eval.finish('server stopped')   # 11-12: a replay still running: its comparison is written
    except Exception as e_: print('REPLAY EVAL save failed', e_, flush=True)
    with APP.LIVE.jcond: APP.LIVE.jcond.notify_all()
    APP.lv_srv.shutdown()
    if APP.lv_srv_local is not None: APP.lv_srv_local.shutdown()
    if APP.GB is not None:
        with APP.GB.lock: APP.GB.closing = True; gp_ = APP.GB.proc   # v26: no new bridge after this
        with APP.GB.wcond: APP.GB.wopen = False; APP.GB.wcond.notify_all()   # v20: the writer thread lets go of the pipe
        if gp_ is not None:
            try: gp_.stdin.close()   # the bridge sends STOP and ends
            except (OSError, ValueError): pass
            try: gp_.wait(5)
            except Exception: gp_.kill()
    with APP.LIVE.cam_lock: APP.LIVE.cam_hold = True; cp_ = APP.LIVE.proc   # v33: a camera reset still running cannot start a new cam_pipe after this
    cp_.terminate()
    try: cp_.wait(20)   # v24c: was 5 s
    except Exception: cp_.kill()
    if worker_ is not None: worker_.join(0.5)
