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
from ..detect import spec_proposals   # 11-14: people proposals from the spatial spectrum (display only)
from ..detect import skin_temp   # 11-14: skin temperature per person box from the camera's temperature map (display only, off by default)
from ..detect import auto_det   # 11-15: automatic NEAR / FAR detector choice (off by default)
from ..detect import frame_feed   # 11-16: sent() = the period monitor's 'result is out' time of each frame
from ..track import box_refine   # 11-15: box smoothing of the drawn / gimbal rows only (off by default)
from ..track.stabilizer import STAB, stab_rows_real, stab_people_real
from ..camera import replay_link
from . import system_buttons
from . import preview_worker
from . import stream_sender   # 11-15 (user: 화면 전송 지연): GET /stream, one viewer
from . import picture_first   # 11-15 (user: total delay about one camera frame): picture at frame arrival, boxes drawn by the page
from . import lost_marks   # 11-16 (user: the prediction marks vanish too quickly): marks kept while the number can still come back
from ..camera import replay_eval   # 11-12: ground-truth comparison while a labelled recording is replayed
from ..camera import stall_diag   # 11-13: stall log (LWIR_STALL_DIAG=1 only)


def init_preview(live, *, transition_jpeg=None):
    async_live = os.environ.get('LWIR_ASYNC_PREVIEW') == '1'
    async_replay = os.environ.get('LWIR_ASYNC_REPLAY_PREVIEW') == '1'
    if async_live or async_replay:
        if transition_jpeg is None:
            transition_jpeg = preview_worker.transition_jpeg(APP.cv2)
        live.preview_worker = preview_worker.PreviewWorker(live, APP.cv2, transition_jpeg=transition_jpeg,
                                                           async_live=async_live, async_replay=async_replay)
    try:
        fixed_map = APP.args.live_map == 'fixed'
    except (NameError, AttributeError):
        fixed_map = False
    live.picture_worker = picture_first.PictureWorker(live, APP.cv2, fixed_map=fixed_map)   # 11-15: acts only while picture first is on


def _lost_marks(t_in, rows_, gb_id_, fps_, own_pool=True):
    """11-16: the lost people to draw this frame (lost_marks.update: the tracker's lost pool, then our own memory of it for as
    long as the number can come back). Once per published frame, before the picture / overlay of that frame, after box_refine.apply
    of the same frame. own_pool: the drawn rows are the server tracker's (False = the camera C++ ByteTrack made them, other numbers).
    11-16 lost_same (box_refine switch): the lost pool is read from APP.tracker after this frame's update (lost_marks.pool_now; the
    Python trackers and the C++ BoT-SORT / OC-SORT / SORT give the same pool) and drawn in the box_refine definition (box_refine.lost_boxes).
    Not readable (camera C++ ByteTrack, end-to-end model, a tracker without a pool): today's POSP['boxes'] path, the reason on the page."""
    if not POSP['show']:
        return lost_marks.update(t_in, (), 0)
    try:
        def app_(name_):   # APP raises NameError for a value that was never set (PC tests)
            try: return getattr(APP, name_)
            except (NameError, AttributeError): return None
        rl_ = app_('relabeler')
        if rl_ is not None and not (app_('RULES') or {}).get('relabel', True): rl_ = None   # KeepID switched off on the page
        idm_ = getattr(rl_, 'id_map', None) if rl_ is not None else None
        follow_s_ = lost_marks.FOLLOW_S
        if APP.GB is not None and gb_id_:
            ms_ = live_tracks._bridge_option(APP.GB, '--reacquire-ms')
            try: follow_s_ = max(1.0, float(ms_) / 1000.0) if ms_ is not None else 1.0   # the bridge's own default is 1 s
            except ValueError: pass
        same_ = {}
        if box_refine.BR['lost_same']:   # 11-16 lost_same (default off: the call below is the very same as before)
            why_ = ''; now_ = None
            if not own_pool: why_ = '카메라 C++ ByteTrack 사용 중(번호가 다름): 예전 점선 상자'
            elif box_refine._e2e_on(): why_ = '엔드투엔드 사용 중: 예전 점선 상자'
            else:
                now_ = lost_marks.pool_now(app_('tracker'))
                if now_ is None: why_ = '이 추적기의 놓친 목록을 읽지 못함: 예전 점선 상자'
            box_refine.BR['lost_why'] = why_
            if now_ is not None: same_ = dict(lost_now=now_, convert=box_refine.lost_boxes)
        return lost_marks.update(t_in, {int(r_[4]) for r_ in rows_}, int(gb_id_ or 0), tracker=app_('tracker'),
                                 id_map=idm_ if isinstance(idm_, dict) else None, keepid=idm_ is not None,
                                 fresh=time.perf_counter() - POSP['boxes_t'] < 0.5, motion=_canvas_motion,
                                 follow_s=follow_s_, fps=min(60.0, max(5.0, fps_)) if fps_ else 30.0, **same_)
    except Exception:   # noqa: BLE001  (display only)
        return []


def _canvas_motion(t0, t1):
    return live_tracks._motion(t0, t1)   # the GMC warps of the frame pairs, canvas coordinates (as the tracker's boxes)


def _predictions(sc_, marks_, gb_id_):
    """Predicted boxes of lost people (위치 기억 · 예측 이동), in picture pixels times sc_. 11-15: also sent to the page for
    'picture first' (picture_first.overlay_message). 11-16: from lost_marks (output ids; kept after the tracker's own
    memory, fading: the 9th value is the alpha)."""
    predictions = []
    try:
        src_name_ = {'kalman': 'Kalman', 'gru': 'GRU', 'mamba': 'Mamba', 'gru_kpt': 'GRU+kpt', 'mamba_kpt': 'Mamba+kpt', 'gru_near_kpt': 'GRU+kpt near'}
        for tid_, b_, inf_ in marks_:
            if APP.GB is not None and tid_ != gb_id_: continue
            pc_ = (0, 220, 255) if gb_id_ and tid_ == gb_id_ else track_color(tid_)
            p1_ = (int((b_[0] - b_[2] / 2) * sc_), int((b_[1] - b_[3] / 2 - 16) * sc_)); p2_ = (int((b_[0] + b_[2] / 2) * sc_), int((b_[1] + b_[3] / 2 - 16) * sc_))
            # 11-14 (user): predicted path (예측 이동) from the last seen place (위치 기억), predictor name and lost frames in the label
            # 11-16: after the tracker's own memory the label says how long the person has been lost
            lab_ = ('%d lost %.1fs' % (int(tid_), inf_['age']) if inf_.get('mem')
                    else '%d %s %df' % (int(tid_), src_name_.get(inf_.get('src'), str(inf_.get('src', ''))), int(inf_.get('n', 0))))
            l_ = inf_.get('last')
            last_ = (int(l_[0] * sc_), int((l_[1] - 16) * sc_), int(l_[2] * sc_), int(l_[3] * sc_)) if l_ else None
            pts_ = tuple((int(x_ * sc_), int((y_ - 16) * sc_)) for x_, y_ in (inf_.get('pts') or ()))
            rad_ = int(inf_.get('r', 0) * sc_)   # 11-14: growing search range
            predictions.append((p1_, p2_, pc_, int(tid_), lab_, last_, pts_, rad_, round(float(inf_.get('alpha', 1.0)), 2)))
    except Exception: pass
    return predictions


def _preview_snapshot(g_, t_in, sc_, draw_, gtd_, sC_, overlay_rows, camera_rows,
                      rows_, row_, npers_, now_, fps_, gb_id_, *, owned, click_people=None,
                      click_ticket=None, click_owners=None, replay_position=None, marks=(), seek=None):
    gt_boxes = ()
    if gtd_ is not None and replay_eval.DRAW[0]:
        gt_boxes = tuple(tuple(b) for b in replay_eval.draw_boxes(t_in, gtd_, sC_))
    predictions = _predictions(sc_, marks, gb_id_) if overlay_rows is None and camera_rows is None else []
    txt_ = 'TRACK %.0f fps | CAM %.0f fps | delay %.0f ms | compute %.0f ms | persons %d | newID %d relink %d' % (fps_, APP.LIVE.cam_fps, (now_ - t_in) * 1000, row_['frame_total_ms'], npers_, APP.relabeler.stats['new_ids'] if APP.relabeler is not None else 0, APP.relabeler.stats['relabelled'] if APP.relabeler is not None else 0)
    rp_ = getattr(APP.LIVE, 'replay', None)
    if camera_rows is not None:
        txt_ = 'ByteTrack C++ | TRACK %.0f fps | CAM %.0f fps | delay %.0f ms | persons %d' % (fps_, APP.LIVE.cam_fps, (now_ - t_in) * 1000, npers_)
    if rp_ is not None: txt_ = '%s | TRACK %.0f fps | CAM %.0f fps | delay %.0f ms | persons %d' % (replay_link.top_text(rp_), fps_, APP.LIVE.cam_fps, (now_ - t_in) * 1000, npers_)
    gtxt_, gc_ = None, (255, 255, 255)
    if APP.GB is not None and rp_ is None:
        st_ = APP.GB.state
        gtxt_ = 'gimbal: not running' if not st_['running'] else 'gimbal: ERROR' if st_['error'] else (('gimbal: lost %d, searching' if seek and seek[0] in (2, 3) else 'gimbal: lost %d, waiting') % st_['id'] if st_['lost'] else 'gimbal: following %d' % st_['id'] if st_['auto'] else 'gimbal: manual, moving' if st_['moving'] else 'gimbal: idle, click a person')
        if APP.GB.gui: gu_ = st_['gui']; gtxt_ += ' | GUI %s: sent %d, failed %d' % ('connected' if gu_['client'] else 'no client', gu_['sent'], gu_['failed'])
        if st_['auto']: gc_ = (0, 220, 255)
    gray_ = g_[16:496]
    if owned:
        # bytes owns an immutable copy; the ndarray view cannot re-enable writes.
        gray_ = np.frombuffer(gray_.tobytes(), dtype=gray_.dtype).reshape(gray_.shape)
    proposals_ = ()
    SP_ = spec_proposals.PROP
    if SP_['show'] and SP_['boxes'] and time.monotonic() - SP_['boxes_t'] < 1.0:   # 11-14: raw = real coordinates; stabilized picture = real + c
        cx_, cy_ = (sC_ or (0.0, 0.0))[:2]
        proposals_ = tuple(((int((x0 + cx_) * sc_), int((y0 + cy_) * sc_)), (int((x1 + cx_) * sc_), int((y1 + cy_) * sc_)), float(s_), bool(cov_))
                           for x0, y0, x1, y1, s_, cov_ in SP_['boxes'])
    temps_ = skin_temp.preview_labels(t_in, sc_, sC_, {d_[3] for d_ in draw_}) if skin_temp.TEMP['show'] and overlay_rows is None else ()   # 11-14
    return preview_worker.PreviewFrame(t_in, gray_, sc_, tuple(draw_), gt_boxes,
                                       tuple(predictions), txt_, rp_ is not None, gtxt_, gc_,
                                       click_people, click_ticket, replay_position, proposals_, temps=temps_,
                                       click_owners=click_owners,
                                       seek=((int(seek[0]), seek[1] * sc_, seek[2] * sc_, seek[4] * sc_, None if seek[6] is None else seek[6] * sc_)
                                             if seek and overlay_rows is None and camera_rows is None else None))


def _click_live(x, y, *, shown_key=None, shown_id=None, viewer_ip=None):
    live = APP.LIVE
    worker = getattr(live, 'preview_worker', None)
    pf_ = picture_first.active(live)   # 11-15: the people of the newest boxes sent to the page (the boxes the viewer sees)
    if shown_key is None and not pf_ and (worker is None or not worker.async_live):
        return live_tracks.click(x, y)
    with live.jcond:
        if shown_key is not None:
            snapshot = next((record for record in reversed(getattr(live, 'overlay_clicks', ()))
                             if record[0] == shown_key), None)
            if not pf_ or snapshot is None:
                return False, '표시된 화면에 해당하는 추적 결과를 기다려 주세요'
            _, frame_t, people, ticket, owners = snapshot
            if (not 0 <= time.perf_counter() - frame_t < 0.5
                    or live.stopped or getattr(live, 'cam_hold', False)
                    or getattr(live, 'cam_lost', False) or replay_link.frame_blocked(frame_t)):
                return False, '표시된 화면이 지연되었습니다. 최신 화면에서 다시 누르세요'
            target = live_tracks.unique_hit_id(people, x, y)
            if target is live_tracks.AMBIGUOUS_CLICK:
                return False, '상자가 겹치거나 같은 추적 ID가 여러 곳에 있습니다. 최신 화면에서 다시 누르세요'
            if target != shown_id:
                return False, '화면의 사람과 같은 프레임의 추적 결과가 다릅니다. 다시 누르세요'
            owner = owners.get(target)
            if target is not None and owner is None:
                return False, '해당 추적 ID의 사람이 하나로 확인되는 최신 화면을 기다려 주세요'
            return live_tracks.click(x, y, expected_id=target, expected_ticket=ticket,
                                     shown_frame_t=frame_t, expected_owner=owner, viewer_ip=viewer_ip)
        frame_t = getattr(live, 'click_frame_t' if pf_ else 'jpeg_frame_t', None)
        people = getattr(live, 'click_people' if pf_ else 'jpeg_people', None)
        ticket = getattr(live, 'click_ticket' if pf_ else 'jpeg_tracks_ticket', None)
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
            # 11-13: frame arrival -> page picture ready, mean of the pictures published in this window (None = no picture was made)
            pictures = [ms for ready, ms in getattr(live, 'delays', ()) if ready >= acc[0]]
            picture_delay = round(sum(pictures) / len(pictures), 1) if pictures else None
            live.perf.append((round(time.time(), 1), round(fps, 1), round(float(live.cam_fps), 1),
                              avg_delay, max_delay, round(acc[4] / acc[1], 1), paired, picture_delay))
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


def _raw_click_owners(raw_ids, raw_to_public=None):
    """Find an unambiguous raw owner for each public ID in this frame."""
    owners, duplicate = {}, set()
    for raw in raw_ids:
        public = raw if raw_to_public is None else raw_to_public.get(raw)
        if public is None:
            continue
        if public in owners:
            duplicate.add(public)
        owners[public] = raw
    for public in duplicate:
        del owners[public]
    return owners


def live_publish(g_, rows_, row_, detections=None, tracker_detections=None, *,
                 click_raw_ids=(), click_raw_to_public=None, click_namespace=None):
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
            if APP.LIVE.meas is not None and not APP.LIVE.meas['video']: APP.LIVE.meas.setdefault('image_ids', []).append(row_.get('image_id'))   # 11-15: without the video, tracks_by_result.csv names each result by its frame name
            if APP.LIVE.meas is not None: APP.LIVE.meas['rows'].append((now_, (now_ - t_in) * 1000.0, float(row_.get('frame_total_ms', 0.0)), float(row_.get('pre_model_ms', 0.0)), float(row_.get('tracker_ms', 0.0)), float(APP.LIVE.cam_fps)))
    fps_ = (len(APP.LIVE.done_ts) - 1) / (APP.LIVE.done_ts[-1] - APP.LIVE.done_ts[0]) if len(APP.LIVE.done_ts) > 2 and APP.LIVE.done_ts[-1] > APP.LIVE.done_ts[0] else 0.0
    # v26c: track output timing, independent of whether a JPEG is made.
    _record_perf(APP.LIVE, now_, t_in if paired_stamp_ else None, fps_, float(row_.get('frame_total_ms', 0.0)))
    sc_ = APP.args.live_scale
    # 11-15 (user "2번부터", "변할 때 자연스럽게"): box_refine (default off = the very same list) smooths only the boxes drawn and sent to the gimbal.
    # The tracker, the measurement record, scoring, the automatic detector choice and the skin temperature keep display_rows.
    shown_rows_ = box_refine.apply(display_rows, tracker_detections, t_in)
    # Send the gimbal frame before preparing any preview annotations.
    has_gimbal_ = APP.GB is not None
    npers_, gb_id_, gb_people_ = 0, 0, []
    for r_ in shown_rows_:
        if int(r_[6]) != 0: continue   # v16: people only
        npers_ += 1
        if has_gimbal_ and len(gb_people_) < 60:
            x1_, y1_, x2_, y2_ = map(float, r_[:4])
            gb_people_.append((int(r_[4]), x1_, y1_ - 16, x2_, y2_ - 16, float(r_[5])))
    if (spec_proposals.LOG['on'] or spec_proposals.PROP['show']) and raw_valid_ and not rep_:   # 11-14: real camera coordinates, like the raw
        dl_ = ([d_[1:] for d_ in stab_people_real([(0, float(d_[0]), float(d_[1]) - 16.0, float(d_[2]), float(d_[3]) - 16.0, float(d_[4]))
                                                   for d_ in detections if int(d_[5]) == 0], sC_, float(APP.LIVE_W), float(APP.LIVE_H))]
               if detections is not None else [])   # every person detection (all scores, before the tracker)
        tr_ = [(float(b_[0]), float(b_[1]) - 16.0, float(b_[2]), float(b_[3]) - 16.0) for b_ in stab_rows_real(display_rows, sC_) if int(b_[6]) == 0]
        if spec_proposals.LOG['on'] and detections is not None:   # research log
            spec_proposals.log_frame(rq_, dl_, tr_, spec_proposals.PROP['boxes'] if spec_proposals.PROP['show'] and time.monotonic() - spec_proposals.PROP['boxes_t'] < 1.0 else ())
        if spec_proposals.PROP['show']:   # (user) an aid for what the tracker missed: its weak boxes looked at again by the spectrum
            spec_proposals.submit(t_in, rq_, tr_, spec_proposals.weak_boxes(dl_, tr_))
    healthy_ = False
    if has_gimbal_:
        source_ = ('replay',) if replay_link.frame_blocked(t_in) else live_tracks.current_source()
        healthy_ = (paired_stamp_ and tracker_detections is not None and raw_valid_
                    and not rep_ and not frozen_ and not replay_link.frame_blocked(t_in)
                    and not any(getattr(APP.LIVE, flag, False) for flag in ('cam_lost', 'cam_hold', 'stopped', 'restart_stop', 'restart_pending')))
        click_owners_ = {}
        if healthy_ and source_ is not None:
            if source_[0] == 'cpp' and camera_rows is not None:
                click_owners_ = _raw_click_owners(int(r[4]) for r in camera_rows if int(r[6]) == 0)
                click_namespace = None
            elif source_[0] == 'python' and camera_rows is None:
                click_owners_ = _raw_click_owners(click_raw_ids, click_raw_to_public)
        gb_id_ = live_tracks.publish(gb_people_, sC_, t_in, raw16_, source_,
                                    frozen_ or replay_link.frame_blocked(t_in), healthy_live=healthy_,
                                    click_owners=click_owners_, click_namespace=click_namespace)
    pipeline_ = getattr(APP.LIVE, 'pipeline_perf', None)
    if pipeline_ is not None and paired_stamp_:
        pipeline_.tracked(t_in, time.perf_counter())
    frame_feed.sent(t_in)   # 11-16 period monitor (feed_auto): this frame's result is out (gimbal); its deadline is arrival + one camera period
    marks_ = _lost_marks(t_in, display_rows, gb_id_, fps_, own_pool=camera_rows is None)   # 11-16: every frame, so the kept marks follow the camera motion
    seek_ = APP.GB.state.get('seek') if has_gimbal_ and gb_id_ else None   # 11-16: the gimbal's search while the followed person is lost (bridge --seek 1)
    if skin_temp.TEMP['show']:   # 11-14: person boxes in real camera coordinates against the newest temperature map (no labels on a recording); after the gimbal frame (display only)
        skin_temp.on_frame(t_in, [(int(b_[4]), float(b_[0]), float(b_[1]) - 16.0, float(b_[2]), float(b_[3]) - 16.0) for b_ in stab_rows_real(display_rows, sC_) if int(b_[6]) == 0],
                           ok=raw_valid_ and not rep_ and not replay_link.frame_blocked(t_in))
    preview_receipt = []
    receipt_args = {'preview_receipt': preview_receipt} if async_replay_ else {}
    gtd_, overlay_rows = replay_eval.on_frame(t_in, rows_, sC_, row_.get('image_id'), detections,
                                             tracker_detections, **receipt_args)
    position = preview_receipt[0] if preview_receipt else None
    auto_det.on_frame(t_in, display_rows, repeat=rep_, frame=row_.get('image_id'), stab=sC_)   # 11-15 (user): automatic NEAR / FAR detector choice; off by default, only queues a request for the frame loop, never raises
    # 11-12: the page picture only when somebody can see it: every frame while /stream has a viewer, otherwise once a second (/snap.jpg at most 1 s old)
    # and every frame for 3 s after a /snap.jpg request. Nothing else reads this picture (tracking, gimbal, measurement and the PC GUI use the frame itself).
    preview_allowed_ = (token_ is not None if worker_ is not None else
                        replay_ is None or replay_source_ is not None)
    required = async_replay_ and worker_.replay_required(position, replay_.get('frames') or 0)
    pf_ = picture_first.active(APP.LIVE) and overlay_rows is None   # 11-15: the picture went out at frame arrival; here only its boxes
    if pf_ and (APP.LIVE.viewers > 0 or now_ - APP.LIVE.snap_t < 3.0):
        boxes_ = []
        for r_ in shown_rows_:   # the boxes drawn on the picture = the boxes sent to the gimbal (box_refine, 11-15)
            if int(r_[6]) != 0: continue
            tid_ = int(r_[4])
            if gb_id_ and tid_ != gb_id_: continue   # while following, only the followed person's box (as in the drawn picture)
            boxes_.append((float(r_[0]), float(r_[1]) - 16.0, float(r_[2]), float(r_[3]) - 16.0, tid_, 1 if gb_id_ else 0))
        click_people, click_ticket = (live_tracks.preview_people(t_in, {b_[4] for b_ in boxes_})
                                      if has_gimbal_ and healthy_ else (None, None))
        msg_ = picture_first.overlay_message(t_in, boxes_, gb_id_, time.perf_counter(), time.time(),
                                             predictions=_predictions(1.0, marks_, gb_id_), seek=seek_)
        click_owners_ = live_tracks.preview_click_owners(t_in) if click_ticket is not None else None
        with APP.LIVE.jcond:
            picture_first.publish_overlay_locked(APP.LIVE, msg_, people=click_people, ticket=click_ticket,
                                                 frame_t=t_in, owners=click_owners_)
    if not pf_ and preview_allowed_ and (required or APP.LIVE.viewers > 0
            or now_ - APP.LIVE.pic_t >= 1.0 or now_ - APP.LIVE.snap_t < 3.0):
        # Replay overlays use their own rows and colors. Live boxes use the ID returned by publish.
        preview_rows_, selected_ = shown_rows_, gb_id_
        if overlay_rows is not None:
            rows_ = preview_rows_ = overlay_rows
            selected_ = 0
        float_coords_ = not has_gimbal_ and overlay_rows is None
        draw_ = []
        for r_ in preview_rows_:
            if int(r_[6]) != 0: continue
            tid_ = int(r_[4])
            if selected_ and tid_ != selected_: continue
            coords_ = r_[:4]
            if float_coords_:
                # Keep the original gimbal-free live rounding at integer pixel boundaries.
                coords_ = map(float, coords_)
            x1_, y1_, x2_, y2_ = coords_
            col_ = (0, 220, 255) if selected_ else track_color(tid_)
            draw_.append(((int(x1_ * sc_), int((y1_ - 16) * sc_)),
                          (int(x2_ * sc_), int((y2_ - 16) * sc_)), col_, tid_))
        if overlay_rows is not None:
            npers_ = len(draw_)
        click_people, click_ticket, click_owners = None, None, None
        if (async_live_ or vars(APP).get('direct_live', False)) and healthy_ and overlay_rows is None:
            click_people, click_ticket = live_tracks.preview_people(t_in, {d[3] for d in draw_})
            if vars(APP).get('direct_live', False):
                click_owners = live_tracks.preview_click_owners(t_in)
        frame_ = _preview_snapshot(g_, t_in, sc_, draw_, gtd_, sC_, overlay_rows, camera_rows,
                                   rows_, row_, npers_, now_, fps_, gb_id_, owned=async_live_ or async_replay_,
                                   click_people=click_people, click_ticket=click_ticket,
                                   click_owners=click_owners, replay_position=position,
                                   marks=marks_, seek=seek_)
        if async_live_ or async_replay_:
            worker_.submit(token_, frame_, requested_at=now_)
        else:
            drawn_ = []
            try:
                jpeg_ = preview_worker.render_preview(APP.cv2, frame_, drawn_)
            except Exception:
                jpeg_ = None
            if worker_ is not None:
                worker_.commit_inline(token_, t_in, jpeg_, requested_at=now_, replay=replay_ is not None,
                                      drawn=drawn_[0] if drawn_ else None, click_people=click_people,
                                      click_ticket=click_ticket, click_owners=click_owners)
            else:
                with APP.LIVE.jcond:
                    source_ok_ = (replay_ is None or (
                        preview_worker.replay_preview_source(APP.LIVE, t_in) is replay_source_))
                    if source_ok_:
                        APP.LIVE.pic_t = now_
                        if jpeg_:
                            preview_worker.publish_jpeg_locked(APP.LIVE, jpeg_, t_in, paired=paired_stamp_,
                                                               high_drawn=drawn_[0] if drawn_ else None,
                                                               click_people=click_people, click_ticket=click_ticket,
                                                               click_owners=click_owners)
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
        td_ = stall_diag.mark()   # None unless LWIR_STALL_DIAG=1
        del APP.rows[:3000]; del APP.preds[:3000]; del APP.tracks_out[:3000]
        if APP.relabeler is not None: del APP.tracks_base[:3000]; del APP.relabeler.per_frame[:3000]
        stall_diag.span(td_, 'trim of the kept frame records (3000 frames freed)')


def make_lv_Handler():
    class lv_Handler(APP.lv_http.BaseHTTPRequestHandler):
        # 11-15 (user: 화면 전송 지연): kept-alive connections, so the page's small requests (heartbeat every 0.4 s, /time, /rules ...) do not
        # pay a TCP handshake (one round trip) each; every answer below has a Content-Length, the two endless ones (/stream, /overlay)
        # say 'Connection: close'. No Nagle delay: the headers and the body are two writes. An idle connection is closed after 65 s.
        protocol_version = 'HTTP/1.1'
        disable_nagle_algorithm = True
        timeout = 65
        def log_message(self, *a): pass
        def do_POST(self):
            if (self.headers.get('Content-Length') or '0').strip() != '0' and not self.path.startswith('/measure/'):
                self.close_connection = True   # a body nobody reads must not be taken for the next request of a kept-alive connection
            # a page on another site cannot send this header without permission, so it cannot move the gimbal through the viewer's browser
            if self.path.startswith('/model/set') and self.headers.get('X-Gimbal') == '1':   # v23: model change from the page (works without the gimbal too)
                from urllib.parse import urlsplit as us_, parse_qs as pq_
                try: mid_ = int(pq_(us_(self.path).query)['id'][0])
                except (KeyError, ValueError, IndexError): self.send_error(400); return
                r_ = APP.model_request(mid_)
                if r_ in (-1, 0): auto_det.manual_choice('page')   # 11-15: a model chosen by hand turns the automatic choice off
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
                    bad_mode = len(modes) != 1 or modes[0] not in ('arrival', 'normal', 'auto', 'tile')
                    if not bad_mode:
                        mode = modes[0]
                if bad_mode:
                    ok, message = False, 'NPU 검출 모드는 arrival, normal, auto, tile 가운데 하나여야 합니다'
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
            u_ = urlsplit(self.path); q_ = parse_qs(u_.query, keep_blank_values=True)
            strict_ = u_.path == '/gimbal/click-frame'
            try:
                if strict_:
                    u_ = u_._replace(path='/gimbal/click')
                    frame_, = q_['frame']
                    shown_, = q_['shown']
                    shown_key_ = int(frame_)
                    shown_id_ = None if shown_ == 'none' else int(shown_)
                cam_tracker = getattr(APP.LIVE, 'camera_bytetrack', None)
                if u_.path == '/gimbal/home':
                    reason_ = ('녹화 재생 중에는 원점으로 이동하지 않습니다' if replay_link.blocking() else
                               '서버 재시작 중에는 원점으로 이동하지 않습니다' if getattr(APP.LIVE, 'restart_pending', False) else
                               'USB 복구 중에는 원점으로 이동하지 않습니다' if getattr(APP.GB, 'usb_hold', False) else
                               '원점 복귀를 지원하는 새 짐벌 브리지가 필요합니다' if not getattr(APP.GB, 'home_control_supported', False) else '')
                    if reason_:
                        b_ = json.dumps(dict(APP.GB.state, answer=reason_, refused=True), ensure_ascii=False).encode()
                        self.send_response(409); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_); return
                if u_.path == '/gimbal/click' and cam_tracker is not None and cam_tracker.blocks_selection():
                    self.send_error(409, 'Tracker switch is pending'); return
                if getattr(APP.LIVE, 'restart_pending', False) and (
                    u_.path == '/gimbal/click' or
                    (u_.path == '/gimbal/move' and (int(q_['x'][0]) or int(q_['y'][0])))
                ):
                    self.send_error(409)
                    return
                ip_ = self.client_address[0]
                if not strict_:
                    APP.GB.mover_t[ip_] = time.monotonic()
                    APP.gb_send('P')
                if u_.path in ('/gimbal/click', '/gimbal/move') and replay_link.blocking() and (u_.path == '/gimbal/click' or int(q_['x'][0]) or int(q_['y'][0])):   # 11-12: no gimbal move while a recording is played (a move of 0 0 = stop is let through)
                    if not strict_:
                        APP.GB.mover = None
                    APP.GB.state['answer'] = '녹화 재생 중에는 짐벌을 움직이지 않습니다. 재생을 멈춘 뒤 사람을 다시 누르세요'
                    b_ = json.dumps(dict(APP.GB.state, cam_idle=round(time.monotonic() - APP.LIVE.last_frame_t, 1), cam_fail=APP.LIVE.cam_fail_n, refused=True), ensure_ascii=False).encode()
                    self.send_response(409); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_); return
                if not strict_ and getattr(APP.GB, 'usb_hold', False) and u_.path in ('/gimbal/click', '/gimbal/move'): u_ = u_._replace(path='/gimbal/stop')   # v35: USB power is being cycled, no new move
                if not strict_ and u_.path in ('/gimbal/click', '/gimbal/stop', '/gimbal/home'): APP.GB.mover = None
                elif u_.path == '/gimbal/move': APP.GB.mover = ip_ if (int(q_['x'][0]) or int(q_['y'][0])) else None
                if u_.path == '/gimbal/click':
                    x_, y_ = float(q_['x'][0]) * APP.LIVE_W, float(q_['y'][0]) * APP.LIVE_H
                    if strict_:
                        ok_, answer_ = _click_live(x_, y_, shown_key=shown_key_, shown_id=shown_id_,
                                                  viewer_ip=ip_)
                    else:
                        ok_, answer_ = _click_live(x_, y_)
                    APP.GB.state['answer'] = answer_
                    if not ok_:
                        b_ = json.dumps(dict(APP.GB.state, refused=True), ensure_ascii=False).encode()
                        self.send_response(409); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_); return
                elif u_.path == '/gimbal/stop': APP.GB.state['answer'] = ''; APP.gb_send('H')
                elif u_.path == '/gimbal/home':
                    APP.GB.state['answer'] = '원점 이동 요청: pan 0°, tilt 0°'
                    if not APP.gb_send('C'): self.send_error(503, 'Gimbal command unavailable'); return
                elif u_.path == '/gimbal/move': APP.GB.state['answer'] = ''; APP.gb_send('V %d %d' % (int(q_['x'][0]), int(q_['y'][0])))
                elif u_.path != '/gimbal/ping': self.send_error(404); return
            except (KeyError, ValueError, IndexError): self.send_error(400); return
            try:
                b_ = json.dumps(dict(APP.GB.state, **live_gimbal_position(), cam_idle=round(time.monotonic() - APP.LIVE.last_frame_t, 1), cam_fail=APP.LIVE.cam_fail_n, handoffs=getattr(APP.GB, 'handoff_n', 0))).encode()   # v24: cam_idle = seconds since the last camera frame; v24d: cam_fail = cam_pipe failures in a row (0 = fine or repairing)
                self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
            except (BrokenPipeError, ConnectionResetError, OSError): pass
        def do_GET(self):
            try:
                if self.path.startswith('/stream'):
                    # 11-15: f=1 = the page reads the stream with fetch, which WebKit (iPhone Safari) refuses for x-mixed-replace
                    ctype_ = 'multipart/mixed' if 'f=1' in urlsplit(self.path).query.split('&') else 'multipart/x-mixed-replace'
                    self.send_response(200); self.send_header('Content-Type', ctype_ + '; boundary=frame'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Connection', 'close'); self.end_headers()
                    try:   # v26b: a slow viewer must get the newest picture, not a queue of old ones: small send buffer (about 2 pictures), no Nagle delay; a viewer that takes nothing for 30 s is closed (the page opens the stream again)
                        self.connection.setsockopt(APP.lv_sock.IPPROTO_TCP, APP.lv_sock.TCP_NODELAY, 1); self.connection.setsockopt(APP.lv_sock.SOL_SOCKET, APP.lv_sock.SO_SNDBUF, 32768); self.connection.settimeout(30)   # 11-15: 32 KB (was 64 KB); the queue wait below keeps at most one picture in flight
                    except OSError: pass
                    # 11-15 (user: 화면 전송 지연): newest picture once the previous one has left the socket, next part's headers right
                    # after each picture (browsers draw a part only then), capture time in a JPEG comment, a newer stream of the same
                    # page (/stream?p=<page id>&n=<count>) replaces the older one, a viewer whose picture takes more than 50 ms
                    # to get through gets a lighter picture (320x240, LWIR_STREAM_LIGHT). src/web/stream_sender.py
                    stream_sender.serve(APP.LIVE, self.connection, self.wfile.write, urlsplit(self.path).query, self.client_address[0],
                                        light=stream_sender.light_maker(APP.cv2),
                                        high=stream_sender.high_maker(APP.cv2))
                elif self.path.startswith('/overlay'):   # 11-15: boxes for 'picture first' as server-sent events (src/web/picture_first.py)
                    self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Connection', 'close'); self.end_headers()
                    try: self.connection.setsockopt(APP.lv_sock.IPPROTO_TCP, APP.lv_sock.TCP_NODELAY, 1); self.connection.settimeout(30)
                    except OSError: pass
                    picture_first.serve_overlay(APP.LIVE, self.wfile.write)
                elif self.path.startswith('/time'):   # 11-15: the server's clock for the page's picture-delay readout (page/js/picture_stream.js); with p=<page id> also what that page's stream measured (stream_sender.page_stats)
                    b_ = json.dumps(dict(t=round(time.time(), 4), v=stream_sender.page_stats(APP.LIVE, (parse_qs(urlsplit(self.path).query).get('p') or [''])[0]))).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/gimbal/people') and APP.GB is not None:   # v19: read only
                    b_ = json.dumps(dict(state=APP.GB.state, **live_gimbal_position(), people=[dict(id=p_[0], x1=p_[1], y1=p_[2], x2=p_[3], y2=p_[4], conf=p_[5]) for p_ in APP.GB.people], width=APP.LIVE_W, height=APP.LIVE_H)).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/rules'):   # v24g
                    b_ = json.dumps(auto_det.dry_log() if self.path.startswith('/rules/auto_det_log') else APP.rules_state(), ensure_ascii=False).encode()   # 11-15: every kept "would-have switch" of a replay (read only)
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/perf'):   # v26c: read only. Per-second history for the graph: t (unix s), track fps, cam fps, delay mean ms, delay max ms, compute mean ms
                    from urllib.parse import urlsplit as us3_, parse_qs as pq3_
                    try: since_ = float(pq3_(us3_(self.path).query).get('since', ['0'])[0])   # the page asks only for the samples after the last one it has
                    except ValueError: since_ = 0.0
                    with APP.LIVE.dlock: pr_ = [r_ for r_ in APP.LIVE.perf if r_[0] > since_]
                    b_ = json.dumps(dict(now=round(time.time(), 1), delay_kind='tracker_output', cols=['t', 'track_fps', 'cam_fps', 'delay_ms', 'delay_max_ms', 'compute_ms', 'paired_n', 'picture_delay_ms'], rows=pr_, pipeline=APP.LIVE.pipeline_perf.snapshot())).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-cache'); self.send_header('Content-Length', str(len(b_))); self.end_headers(); self.wfile.write(b_)
                elif self.path.startswith('/healthz'):   # v24d: read-only liveness for the keep-alive supervisor (never POST /gimbal/ping from a script: that is the page's safety heartbeat). 11-15: cam_frozen = the camera's USB chip froze (cam_pipe exit 8, no camera frame since): keepalive.sh does not reboot then
                    now_ = time.monotonic(); last_out_ = APP.LIVE.done_ts[-1] if APP.LIVE.done_ts else None
                    try: rss_ = int(open('/proc/self/statm').read().split()[1]) * 4096 // 1048576
                    except Exception: rss_ = None
                    h_ = dict(instance=getattr(APP.LIVE, 'instance', None), up=round(now_ - APP.LIVE.t0, 1), cam_idle=round(now_ - APP.LIVE.last_frame_t, 1), out_idle=round(time.perf_counter() - last_out_ if last_out_ is not None else now_ - APP.LIVE.t0, 1), frames_in=APP.LIVE.frames_in,   # v26: before the first result out_idle is the time since the start (a tracker that hangs before its first result is seen too)
                              cam_fail=APP.LIVE.cam_fail_n, cam_frozen=getattr(APP.LIVE, 'cam_frozen', None) is not None, cam_scene=getattr(APP.LIVE, 'cam_scene', None) is not None, warm_dropped=APP.LIVE.warm_dropped, frozen_frames=getattr(APP.LIVE, 'frozen_total', 0), pictures=APP.LIVE.pic_n, viewers=APP.LIVE.viewers, cam_pipe_alive=APP.LIVE.proc.poll() is None, gimbal_alive=(APP.GB.proc is not None and APP.GB.proc.poll() is None) if APP.GB is not None else None,
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
    if vars(APP).get('direct_link') is not None: APP.direct_link.close()
    if APP.lv_srv is not None: APP.lv_srv.shutdown()
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
