"""웹 화면: 보여 줄 그림 만들기(live_publish), 페이지 요청 처리(lv_Handler, lv_Server), 서버 끝내기(live_close)."""
import json
import time
import numpy as np
from ..common.state import APP
from ..common.shared_values import HERE
from ..track.position_memory import POSP
from ..track.stabilizer import STAB, stab_rows_real, stab_people_real
from ..camera import replay_link
from ..camera import replay_eval   # 11-12: ground-truth comparison while a labelled recording is replayed


def dash_rect(img, p1, p2, col, th=2, dash=7):
    """v36: dashed rectangle (an estimate, not a detection)"""
    import cv2
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), col, th); cv2.line(img, (x, y2), (min(x + dash, x2), y2), col, th)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), col, th); cv2.line(img, (x2, y), (x2, min(y + dash, y2)), col, th)


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


def live_publish(g_, rows_, row_):
    now_ = time.perf_counter(); t_in = APP.LIVE.arr.popleft() if APP.LIVE.arr else now_
    raw16_ = APP.LIVE.raws.popleft() if APP.LIVE.raws else None   # v20
    rq_ = APP.LIVE.rawq.popleft() if APP.LIVE.rawq else None   # v24f
    frozen_ = frozen_check(rq_)   # 11-12: the same picture as the previous frame -> the gimbal gets no people for this frame
    sC_ = APP.LIVE.stabq.popleft() if APP.LIVE.stabq else None; STAB['px'] = sC_ or (0.0, 0.0)   # stab: correction of the frame being shown
    APP.LIVE.done_ts.append(now_)
    if APP.LIVE.meas is not None:   # v22
        with APP.LIVE.meas_lock:
            if APP.LIVE.meas is not None: APP.LIVE.meas['boxes'].append((t_in, [(float(b_[0]), float(b_[1]) - 16.0, float(b_[2]), float(b_[3]) - 16.0, int(b_[4]), float(b_[5]), int(b_[6])) for b_ in stab_rows_real(rows_, sC_)]))   # v22d
            if APP.LIVE.meas is not None: APP.LIVE.meas['rows'].append((now_, (now_ - t_in) * 1000.0, float(row_.get('frame_total_ms', 0.0)), float(row_.get('pre_model_ms', 0.0)), float(row_.get('tracker_ms', 0.0)), float(APP.LIVE.cam_fps)))
    fps_ = (len(APP.LIVE.done_ts) - 1) / (APP.LIVE.done_ts[-1] - APP.LIVE.done_ts[0]) if len(APP.LIVE.done_ts) > 2 and APP.LIVE.done_ts[-1] > APP.LIVE.done_ts[0] else 0.0
    # v26c: one sample per second of the numbers in the top line of the picture (TRACK fps, CAM fps, delay, compute), kept 15 min for the graph on the page (GET /perf)
    pa_ = APP.LIVE.perf_acc; d_ms_ = (now_ - t_in) * 1000.0; pa_[1] += 1; pa_[2] += d_ms_; pa_[3] = max(pa_[3], d_ms_); pa_[4] += float(row_.get('frame_total_ms', 0.0))
    if now_ - pa_[0] >= 1.0:
        with APP.LIVE.dlock: APP.LIVE.perf.append((round(time.time(), 1), round(fps_, 1), round(float(APP.LIVE.cam_fps), 1), round(pa_[2] / pa_[1], 1), round(pa_[3], 1), round(pa_[4] / pa_[1], 1)))
        APP.LIVE.perf_acc = [now_, 0, 0.0, 0.0, 0.0]
    sc_ = APP.args.live_scale
    # 11-12: boxes are collected first and drawn after the gimbal line below has gone out (the page picture takes about 6 ms on the board)
    npers_ = 0; gb_people_ = []; gb_id_ = APP.GB.state['id'] if APP.GB is not None else 0; draw_ = []
    for r_ in rows_:
        x1_, y1_, x2_, y2_ = [float(v_) for v_ in r_[:4]]; tid_, cls_ = int(r_[4]), int(r_[6]); y1_ -= 16; y2_ -= 16
        if cls_ != 0: continue   # v16: people only
        col_ = (0, 220, 255) if gb_id_ and tid_ == gb_id_ else (60, 220, 60); npers_ += 1
        if APP.GB is not None and len(gb_people_) < 60: gb_people_.append((tid_, x1_, y1_, x2_, y2_, float(r_[5])))
        if gb_id_ and tid_ != gb_id_: continue   # v39 (user 11-10): while one person is followed only that person's box is drawn
        draw_.append(((int(x1_ * sc_), int(y1_ * sc_)), (int(x2_ * sc_), int(y2_ * sc_)), col_, tid_))
    if APP.GB is not None:
        APP.GB.people = gb_people_; APP.GB.frame += 1
        if sC_ is not None: gb_people_ = stab_people_real(gb_people_, sC_, float(APP.LIVE_W), float(APP.LIVE_H))   # stab: the F / G lines below carry real camera coordinates
        if frozen_: gb_people_ = []   # 11-12 frozen picture: the gimbal program sees its person as lost and stops (selection kept for the reacquire time)
        if replay_link.frame_blocked(t_in): gb_people_ = []; APP.GB.people = []   # 11-12 recording replay: the gimbal never gets the replayed people (and a click cannot pick one)
        if APP.CTL['lead']: gb_people_ = APP.lead_shift(gb_people_, APP.GB.people, gb_id_, t_in, float(APP.LIVE_W), float(APP.LIVE_H))   # v40 예측 제어: only the followed person's box in the F / G line moves ahead
        elif APP.CTL['lv'] is not None or APP.CTL['lead_px']: APP.CTL['lv'] = None; APP.CTL['lead_px'] = 0.0
        people_txt_ = ''.join(' %d %.1f %.1f %.1f %.1f %.3f' % (p_[0], p_[1], p_[2], p_[3] - p_[1], p_[4] - p_[2], p_[5]) for p_ in gb_people_)
        if APP.GB.gui and raw16_ is not None and raw16_.shape == (APP.LIVE_H, APP.LIVE_W):   # v20: frame + picture to the gimbal executor and the PC GUI
            APP.gb_send_frame('G %d %d %d %d %d' % (APP.GB.frame, int(t_in * 1e9), APP.LIVE_W, APP.LIVE_H, len(gb_people_)) + people_txt_, np.ascontiguousarray(raw16_, dtype=np.uint16).tobytes())
        else: APP.gb_send('F %d %d %d' % (APP.GB.frame, int(t_in * 1e9), len(gb_people_)) + people_txt_)
    gtd_ = replay_eval.on_frame(t_in, rows_, sC_, row_.get('image_id')) if replay_eval.EV is not None else None   # 11-12: score this frame against the labels (replay only; after the gimbal line)
    # 11-12: the page picture only when somebody can see it: every frame while /stream has a viewer, otherwise once a second (/snap.jpg at most 1 s old)
    # and every frame for 3 s after a /snap.jpg request. Nothing else reads this picture (tracking, gimbal, measurement and the PC GUI use the frame itself).
    if APP.LIVE.viewers > 0 or now_ - APP.LIVE.pic_t >= 1.0 or now_ - APP.LIVE.snap_t < 3.0:
        APP.LIVE.pic_t = now_; APP.LIVE.pic_n += 1
        img_ = APP.cv2.cvtColor(g_[16:496], APP.cv2.COLOR_GRAY2BGR)
        if sc_ != 1: img_ = APP.cv2.resize(img_, None, fx=sc_, fy=sc_, interpolation=APP.cv2.INTER_LINEAR)
        for p1_, p2_, col_, tid_ in draw_:
            APP.cv2.rectangle(img_, p1_, p2_, col_, 2); APP.cv2.putText(img_, str(tid_), (p1_[0], max(p1_[1] - 4, 12)), APP.cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_, 2)
        if gtd_ is not None and replay_eval.DRAW[0]:   # 11-12: ground-truth boxes, thin and dashed, over the track boxes, on the page picture only (white = person, pink = ignore region)
            hh_, ww_ = img_.shape[:2]
            for x1_, y1_, x2_, y2_, k_ in replay_eval.draw_boxes(t_in, gtd_, sC_):
                a_, b_ = (min(max(int(x1_ * sc_), 0), ww_ - 1), min(max(int(y1_ * sc_), 0), hh_ - 1)), (min(max(int(x2_ * sc_), 0), ww_ - 1), min(max(int(y2_ * sc_), 0), hh_ - 1))
                if a_[0] < b_[0] and a_[1] < b_[1]: dash_rect(img_, a_, b_, (255, 255, 255) if k_ == 0 else (200, 90, 255), 1, 4)
        if POSP['show'] and now_ - POSP['boxes_t'] < 0.5:   # v32: predicted position of lost people (picture only, never sent to the gimbal)
            try:
                seen_ = {int(r_[4]) for r_ in rows_}
                for tid_, b_ in POSP['boxes']:
                    if tid_ in seen_ or (APP.GB is not None and tid_ != gb_id_): continue   # v36 (user 11-10): only the person being followed (no gimbal link: every lost person)
                    pc_ = (0, 220, 255) if gb_id_ and tid_ == gb_id_ else (60, 220, 60)
                    p1_ = (int((b_[0] - b_[2] / 2) * sc_), int((b_[1] - b_[3] / 2 - 16) * sc_)); p2_ = (int((b_[0] + b_[2] / 2) * sc_), int((b_[1] + b_[3] / 2 - 16) * sc_))
                    dash_rect(img_, p1_, p2_, pc_, 2); APP.cv2.putText(img_, '%d predict' % tid_, (p1_[0], max(p1_[1] - 4, 12)), APP.cv2.FONT_HERSHEY_SIMPLEX, 0.55, pc_, 2)   # v36: same colour as the track, dashed = estimate
            except Exception: pass
        txt_ = 'TRACK %.0f fps | CAM %.0f fps | delay %.0f ms | compute %.0f ms | persons %d | newID %d relink %d' % (fps_, APP.LIVE.cam_fps, (now_ - t_in) * 1000, row_['frame_total_ms'], npers_, APP.relabeler.stats['new_ids'] if APP.relabeler is not None else 0, APP.relabeler.stats['relabelled'] if APP.relabeler is not None else 0)
        rp_ = getattr(APP.LIVE, 'replay', None)
        if rp_ is not None: txt_ = '%s | TRACK %.0f fps | CAM %.0f fps | delay %.0f ms | persons %d' % (replay_link.top_text(rp_), fps_, APP.LIVE.cam_fps, (now_ - t_in) * 1000, npers_)   # 11-12: shorter line so the name fits
        APP.cv2.rectangle(img_, (0, 0), (img_.shape[1], 26), (0, 0, 0) if rp_ is None else (0, 0, 150), -1); APP.cv2.putText(img_, txt_, (8, 19), APP.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if APP.GB is not None:
            st_ = APP.GB.state
            gtxt_ = 'gimbal: not running' if not st_['running'] else 'gimbal: ERROR' if st_['error'] else ('gimbal: lost %d, waiting' % st_['id'] if st_['lost'] else 'gimbal: following %d' % st_['id'] if st_['auto'] else 'gimbal: manual, moving' if st_['moving'] else 'gimbal: idle, click a person')
            if APP.GB.gui: gu_ = st_['gui']; gtxt_ += ' | GUI %s: sent %d, failed %d' % ('connected' if gu_['client'] else 'no client', gu_['sent'], gu_['failed'])   # v20
            APP.cv2.rectangle(img_, (0, 26), (img_.shape[1], 50), (0, 0, 0), -1); APP.cv2.putText(img_, gtxt_, (8, 44), APP.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255) if st_['auto'] else (255, 255, 255), 1)
        ok_, jb_ = APP.cv2.imencode('.jpg', img_, [APP.cv2.IMWRITE_JPEG_QUALITY, 80])
        with APP.LIVE.jcond: APP.LIVE.jpeg = jb_.tobytes(); APP.LIVE.seq += 1; APP.LIVE.jcond.notify_all()
    td_ = time.perf_counter()
    with APP.LIVE.dlock: APP.LIVE.delays.append((td_, (td_ - t_in) * 1000.0))   # v26b: camera frame arrival -> picture ready for the page, kept for /healthz
    if time.monotonic() - APP.LIVE.t0 > APP.args.live_seconds or (APP.LIVE.cam_lost and APP.args.live_wait): APP.LIVE.stopped = True
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
                elif u4_.path == '/replay/eval/reset': code_, text_ = replay_eval.reset_request()   # 11-12: count again from now
                elif u4_.path == '/replay/eval/draw': code_, text_ = replay_eval.draw_request(pq4_(u4_.query))   # 11-12: ground-truth boxes on the page picture
                else: self.send_error(404); return
                b_ = json.dumps(dict(ok=code_ == 200, text=text_, state=replay_link.state(), eval=replay_eval.state()), ensure_ascii=False).encode()
                try: self.send_response(code_); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                except (BrokenPipeError, ConnectionResetError, OSError): pass
                return
            if APP.GB is None or self.headers.get('X-Gimbal') != '1': self.send_error(403); return
            from urllib.parse import urlsplit, parse_qs
            u_ = urlsplit(self.path); q_ = parse_qs(u_.query)
            try:
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
                    inside_ = [p_ for p_ in APP.GB.people if p_[1] <= x_ <= p_[3] and p_[2] <= y_ <= p_[4]]
                    if inside_:
                        APP.GB.state['answer'] = ''; APP.gb_send('A %d' % min(inside_, key=lambda p_: (p_[3] - p_[1]) * (p_[4] - p_[2]))[0])
                    else: APP.GB.state['answer'] = ''; APP.gb_send('H')   # v40: a tap / click on an empty place = stop (the page has no stop key any more)
                elif u_.path == '/gimbal/stop': APP.GB.state['answer'] = ''; APP.gb_send('H')
                elif u_.path == '/gimbal/move': APP.GB.state['answer'] = ''; APP.gb_send('V %d %d' % (int(q_['x'][0]), int(q_['y'][0])))
                elif u_.path != '/gimbal/ping': self.send_error(404); return
            except (KeyError, ValueError, IndexError): self.send_error(400); return
            try:
                b_ = json.dumps(dict(APP.GB.state, cam_idle=round(time.monotonic() - APP.LIVE.last_frame_t, 1), cam_fail=APP.LIVE.cam_fail_n)).encode()   # v24: cam_idle = seconds since the last camera frame; v24d: cam_fail = cam_pipe failures in a row (0 = fine or repairing)
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
                    b_ = json.dumps(dict(state=APP.GB.state, people=[dict(id=p_[0], x1=p_[1], y1=p_[2], x2=p_[3], y2=p_[4], conf=p_[5]) for p_ in APP.GB.people], width=APP.LIVE_W, height=APP.LIVE_H)).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/rules'):   # v24g
                    b_ = json.dumps(APP.rules_state(), ensure_ascii=False).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/perf'):   # v26c: read only. Per-second history for the graph: t (unix s), track fps, cam fps, delay mean ms, delay max ms, compute mean ms
                    from urllib.parse import urlsplit as us3_, parse_qs as pq3_
                    try: since_ = float(pq3_(us3_(self.path).query).get('since', ['0'])[0])   # the page asks only for the samples after the last one it has
                    except ValueError: since_ = 0.0
                    with APP.LIVE.dlock: pr_ = [r_ for r_ in APP.LIVE.perf if r_[0] > since_]
                    b_ = json.dumps(dict(now=round(time.time(), 1), cols=['t', 'track_fps', 'cam_fps', 'delay_ms', 'delay_max_ms', 'compute_ms'], rows=pr_)).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/healthz'):   # v24d: read-only liveness for the keep-alive supervisor (never POST /gimbal/ping from a script: that is the page's safety heartbeat)
                    now_ = time.monotonic(); last_out_ = APP.LIVE.done_ts[-1] if APP.LIVE.done_ts else None
                    try: rss_ = int(open('/proc/self/statm').read().split()[1]) * 4096 // 1048576
                    except Exception: rss_ = None
                    h_ = dict(up=round(now_ - APP.LIVE.t0, 1), cam_idle=round(now_ - APP.LIVE.last_frame_t, 1), out_idle=round(time.perf_counter() - last_out_ if last_out_ is not None else now_ - APP.LIVE.t0, 1), frames_in=APP.LIVE.frames_in,   # v26: before the first result out_idle is the time since the start (a tracker that hangs before its first result is seen too)
                              cam_fail=APP.LIVE.cam_fail_n, warm_dropped=APP.LIVE.warm_dropped, frozen_frames=getattr(APP.LIVE, 'frozen_total', 0), pictures=APP.LIVE.pic_n, viewers=APP.LIVE.viewers, cam_pipe_alive=APP.LIVE.proc.poll() is None, gimbal_alive=(APP.GB.proc is not None and APP.GB.proc.poll() is None) if APP.GB is not None else None,
                              gimbal_ready=bool(APP.GB.state['ready']) if APP.GB is not None else None, gimbal_error=int(APP.GB.state['error']) if APP.GB is not None else None, gimbal_starts=(APP.GB.restarts + 1) if APP.GB is not None else None,
                              gimbal_device=(APP.gb_dev_sig(APP.gb_device()) is not None) if APP.GB is not None else None, gimbal_busy=bool(APP.GB.busy) if APP.GB is not None else None, rss_mb=rss_, stopped=APP.LIVE.stopped,
                              q_len=len(APP.LIVE.q), q_drop=APP.LIVE.q_drop, gaps=APP.LIVE.gap_n, replay=(getattr(APP.LIVE, 'replay', None) or {}).get('name'), track_fps=round((len(APP.LIVE.done_ts) - 1) / (APP.LIVE.done_ts[-1] - APP.LIVE.done_ts[0]), 1) if len(APP.LIVE.done_ts) > 2 and APP.LIVE.done_ts[-1] > APP.LIVE.done_ts[0] else 0.0, cam_fps=round(APP.LIVE.cam_fps, 1))
                    with APP.LIVE.dlock: dl_ = [d_ for t_, d_ in APP.LIVE.delays if time.perf_counter() - t_ <= 30.0]   # v26b: delay of the last 30 s (camera frame arrival -> picture ready), ms
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
                    b_ = json.dumps(replay_eval.state() if u5_.path == '/replay/eval' else replay_link.get_info(pq5_(u5_.query)), ensure_ascii=False).encode()   # 11-12: /replay/eval = the comparison alone
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/snap.jpg'):
                    APP.LIVE.snap_t = time.perf_counter()   # 11-12: pictures for every frame during the next 3 s
                    with APP.LIVE.jcond: jb_ = APP.LIVE.jpeg
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
    APP.LIVE.stopped = True
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
