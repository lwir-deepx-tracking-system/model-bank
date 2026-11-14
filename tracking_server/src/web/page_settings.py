"""화면에서 바꾸는 설정: 거름 기준과 추적기 종류(rules_*), 검출 모델 바꾸기(model_*, do_model_swap), 마지막 선택 저장(sel_save)."""
import os
import json
import time
from ..common import settings
from ..common.state import APP
from ..common.shared_values import HERE, CORE, DET, sha
from ..detect.model_scan import BANK, KPT_DETS
from ..track import native_backend
from ..track.position_memory import POSP, POS_NAMES, pos_list, pos_select
from ..track.stabilizer import STAB
from ..track.bytetrack import ByteTrackTracker
from ..track.botsort import BotSortTracker
from ..track.ocsort import OCSortTracker
from ..track.sort import SortTracker
from ..track.confirm_time import ConfirmByTime, MS_MAX
from ..reid.reid_models import reid_catalog
from ..gimbal.gimbal_link import gb_cfg
from .system_buttons import npu_feed_state


def sel_save(**over_):
    if not APP.SEL_ON: return
    try:
        with APP.SEL_LOCK:
            APP.SEL.update(over_); t_ = APP.SEL_FILE.with_suffix('.json.tmp')
            t_.write_text(json.dumps(dict(APP.SEL, saved=time.strftime('%Y-%m-%d %H:%M:%S')), ensure_ascii=False, indent=1), encoding='utf-8'); os.replace(t_, APP.SEL_FILE)
    except Exception as e_: print('LAST selection not saved:', repr(e_)[:120], flush=True)


def sel_det_late(dv_, tries_=30, gap_=2.0):
    """fix 11-11: the saved detector is a file the scan lists by itself ('bank:<file name>'). At start it is not in the list yet, so this thread looks every 2 s
    for up to 60 s (the scan itself still runs at most every 5 s) and asks for the model ONCE when it is listed. Any model choice made in the meantime wins
    (page or PC GUI); a file that never gets listed = nothing happens, one line is printed."""
    why_ = 'file not listed within %d s' % int(tries_ * gap_)
    try:
        for _ in range(tries_):
            time.sleep(gap_)
            if APP.MODEL['req'] is not None or APP.MODEL['busy'] or APP.MODEL['last'] is not None or 'det' in APP.SEL: why_ = 'another model choice came first'; break
            APP.bank_refresh()
            with APP.MODEL['lock']: c_ = next((x_ for x_ in APP.MODEL['catalog'] if x_['det'] == dv_), None)
            if c_ is None: continue
            g_ = getattr(APP, 'GB', None)   # review 11-11: a model change pauses the frames for a moment and that drops a running follow, so never do it while the gimbal follows or moves
            if g_ is not None and (g_.state.get('auto') or g_.state.get('moving')): why_ = 'the gimbal is following or moving'; break
            r_ = APP.model_request(c_['id'])
            print('LAST selection model %s -> %s' % (dv_, 'queued (listed late as id %d)' % c_['id'] if r_ == -1 else 'not taken (%s)' % APP.MODEL_TEXT.get(r_, '?')), flush=True); return
    except Exception as e_: why_ = repr(e_)[:120]
    print('LAST selection model %s -> not taken (%s)' % (dv_, why_), flush=True)


ENGINE_KINDS = ('ocsort', 'botsort', 'sort')   # trackers whose engine (C++ first, or Python) is chosen on the page


def tracker_engine():
    """'auto' = C++ when its extension is built (the default click), 'python' = the operator chose Python."""
    return APP.RULES.get('engine', 'auto')


def mk_tracker_(kind_):
    use_native = not (kind_ in ENGINE_KINDS and tracker_engine() == 'python')
    alternative = {'ocsort': OCSortTracker, 'sort': SortTracker}.get(kind_)
    if alternative is not None:
        kwargs = dict(n_init=max(3, APP.args.n_init), class_gate=APP.args.class_gate, radius=APP.args.radius)
        return native_backend.create(kind_, kind_, kwargs, lambda: alternative(**kwargs), native=use_native)
    cls = BotSortTracker if kind_ == 'botsort' else ByteTrackTracker
    py_mode = 'rebind' if kind_ == 'rebind' else 'bytetrack'
    native_mode = 'botsort' if kind_ == 'botsort' else py_mode
    kwargs = dict(sim=APP.args.sim, app=False, late_window=APP.args.late_window, app_first=False, gate=APP.args.gate,
                  motion_fallback=not APP.args.no_motion_fallback, vit_conf=APP.args.vit_conf, vit_min_h=APP.args.vit_min_h,
                  hybrid=True, sync_k=APP.args.sync_k, hybrid_fallback=False, radius=APP.args.radius,
                  n_init=APP.args.n_init, class_gate=APP.args.class_gate)
    return native_backend.create(kind_, native_mode, kwargs, lambda: cls(py_mode, **kwargs), native=use_native)


def ctl_state():
    try: rl_ok_ = APP.rl_eng is not None
    except NameError: rl_ok_ = False
    try: gbs_ = dict(APP.GB.state) if APP.GB is not None else None
    except Exception: gbs_ = None
    rq_ = APP.REID['req']
    try: cf_ = APP.CONFIRM.state(APP.tracker) if APP.CONFIRM is not None else None   # 11-11 --confirm-ms: {ms, frames now, measured fps, changes}; None = off (frames counted as before)
    except Exception: cf_ = None
    return dict(confirm=cf_, reid=APP.REID['cur'], reid_hot=bool(rl_ok_), reid_pending=rq_['key'] if rq_ else None, reid_last=APP.REID['last'],   # v41
                reids=reid_catalog(HERE / 'models' / 'links', APP.man, APP.args.relabel or '') if rl_ok_ else [],
                app=APP.CTL['app'], apps=[dict(key='off', name='끔', available=True, note=''),
                                      dict(key='repvgg', name='RepVGG_(TMOT,CAMEL)', available=rl_ok_, note='' if rl_ok_ else 'KeepID 모델이 이 실행에 없습니다'),
                                      dict(key='vit', name='ViT', available=False, note='ViT 파일을 장치에 올려야 열립니다')],
                lead=bool(APP.CTL['lead']), lead_ms=int(APP.CTL['lead_ms']), lead_px=round(float(APP.CTL['lead_px']), 1), lead_stats=dict(APP.CTL.get('lead_stats') or {}), warm_on=bool(settings.WARM_MIN > 0 or settings.WARM_MAX > 0), warm_prev=list(APP.CTL['warm_prev']),
                gimbal=dict(present=gbs_ is not None, running=bool(gbs_ and gbs_.get('running')), id=int(gbs_.get('id', 0) or 0) if gbs_ else 0, auto=int(gbs_.get('auto', 0) or 0) if gbs_ else 0))


def rules_state():
    cam_tracker = getattr(APP.LIVE, 'camera_bytetrack', None)
    camera_backend = cam_tracker.state() if cam_tracker is not None else None
    if camera_backend is not None:
        camera_backend['control_supported'] = bool(APP.GB is not None and getattr(APP.GB, 'track_control_supported', False))
    with APP.RULES['lock']:
        return dict(camera_bytetrack=camera_backend,
                    server_restart=bool(getattr(APP.LIVE, 'restart_supported', False)),
                    npu_feed=npu_feed_state(),
                    restart_pending=bool(getattr(APP.LIVE, 'restart_pending', False)), warm_max=settings.WARM_MAX, cam_reset=getattr(APP.LIVE, 'cam_reset', None), sys_buttons=APP.LIVE is not None and not APP.args.live_wait, warm_min=settings.WARM_MIN, edge_birth=settings.EDGE_BIRTH, edge_px=settings.EDGE_PX, tracker=APP.RULES['tracker'], relabel=APP.RULES['relabel'], relabel_available=bool(APP.args.relabel),
                    gate_available=False, tracker_backend=native_backend.backend_of(APP.tracker), engine=tracker_engine(),
                    presets=[dict(key=k_, name=n_, **native_backend.backend_status(k_)) for k_, n_ in APP.RULES_PRESETS], pending=APP.RULES['pending'], last=APP.RULES['last'], changes=APP.RULES['changes'],
                    warm_dropped=(APP.LIVE.warm_dropped if APP.LIVE is not None else None), vit='꺼짐 (재시작 필요)' if APP.args.no_vit else '켜짐', mamba=bool(APP.args.mamba),
                    det_kpt=det_has_kpt(),   # 11-14: the current or the waiting detector gives keypoints -> the keypoint position memories may be chosen
                    predictor=POSP['key'], predictors=pos_list(), pred_show=POSP['show'], pred_err=POSP['err'], pred_used=POSP['used'], **APP.ctl_state())   # v32. v40: app / lead / gimbal / warm_on


def rules_state_2():
    return dict(APP.rules_state0_(), stab=bool(STAB['on']), stab_px=[round(float(STAB['px'][0]), 1), round(float(STAB['px'][1]), 1)], stab_ms=round(float(STAB['ms']), 2), stab_resets=int(STAB['resets']), stab_err=STAB['err'], gimbal_cfg=gb_cfg())   # gimbal_cfg: 읽기 전용, 짐벌 프로그램에 넘긴 옵션 (화면의 '짐벌 제어기' 구역)


def prefer_cpp_bytetrack(q_, tracker_key, cam_tracker):
    """11-13: `engine=auto` on the ByteTrack menu row asks for C++ when it can fully replace Python on the camera:
    library installed, camera (not a recording) on screen, nobody being followed, and a gimbal bridge that takes the
    C++ IDs (or no gimbal). Every other request, including the explicit Python option, keeps the Python engine."""
    if q_.get('engine', [''])[0] != 'auto' or tracker_key != 'confirm3': return False
    try:
        from ..camera import replay_link
        if replay_link.blocking(): return False
        try: gb_ = APP.GB
        except NameError: gb_ = None   # server started without a gimbal
        if gb_ is not None and (gb_.state.get('auto') or not getattr(gb_, 'track_control_supported', False)): return False
        return bool(cam_tracker.state().get('available'))
    except Exception as e_:
        print('RULES engine auto -> Python (%s)' % repr(e_)[:120], flush=True)
        return False


def rules_request(q_):
    """q_: dict of lists from parse_qs. Thresholds change at once; tracker / relabel are queued for the frame boundary. Returns (ok, text)."""
    if 'app' in q_:   # v40: everything new is checked before anything changes
        if q_['app'][0] == 'vit': return False, 'ViT 파일을 장치에 올려야 열립니다'
        if q_['app'][0] not in ('off', 'repvgg'): return False, '없는 외형 비교 종류'
    rk_ = None
    if 'reid' in q_:   # v41: appearance model (KeepID + BoT-SORT 외형 비교). Only checked and queued here; the frame loop loads it between two frames
        rk_ = str(q_['reid'][0])
        try: rl_ok_ = APP.rl_eng is not None
        except NameError: rl_ok_ = False
        if not rl_ok_: return False, '재식별 모델이 이 실행에 없음'
        rc_ = next((x_ for x_ in reid_catalog(HERE / 'models' / 'links', APP.man, APP.args.relabel or '') if x_['key'] == rk_), None)
        if rc_ is None: return False, '없는 외형 모델'
        if not rc_['available']: return False, rc_['why']
        if rk_ == APP.REID['cur'] and APP.REID['req'] is None: rk_ = None   # already running: nothing to load
        else:
            try: gb_busy_ = APP.GB is not None and bool(APP.GB.state.get('auto') or APP.GB.state.get('moving'))
            except Exception: gb_busy_ = False
            if gb_busy_: return False, '따라가기 · 짐벌 움직임을 멈춘 뒤 바꾸세요 (바꾸는 동안 화면이 잠깐 멈춤)'
    if 'lead_ms' in q_:
        try: lm_ = max(0, min(400, int(float(q_['lead_ms'][0]))))
        except (ValueError, IndexError): return False, '숫자가 아님'
    if 'lead' in q_ and q_['lead'][0] in ('1', 'true', 'on'):
        try: gb_ok_ = APP.GB is not None
        except NameError: gb_ok_ = False
        if not gb_ok_: return False, '짐벌이 연결되지 않음'
    if 'warm_on' in q_ and 'warm_min' not in q_ and 'warm_max' not in q_:   # v40: 온도 Filter on / off; the numbers are remembered while it is off
        if q_['warm_on'][0] in ('1', 'true', 'on'):
            if settings.WARM_MIN <= 0 and settings.WARM_MAX <= 0:
                wp_ = APP.CTL['warm_prev'] if (APP.CTL['warm_prev'][0] > 0 or APP.CTL['warm_prev'][1] > 0) else (45000, 0)
                q_ = dict(q_, warm_min=[str(int(wp_[0]))], warm_max=[str(int(wp_[1]))])
        else:
            if settings.WARM_MIN > 0 or settings.WARM_MAX > 0: APP.CTL['warm_prev'] = (int(settings.WARM_MIN), int(settings.WARM_MAX))
            q_ = dict(q_, warm_min=['0'], warm_max=['0'])
    try:
        wn_ = max(0, min(65535, int(float(q_['warm_min'][0])))) if 'warm_min' in q_ else settings.WARM_MIN   # v33: the range is checked before anything changes
        wx_ = max(0, min(65535, int(float(q_['warm_max'][0])))) if 'warm_max' in q_ else settings.WARM_MAX
        if wx_ > 0 and wx_ <= wn_: return False, 'Warmth 최대는 최소보다 커야 함 (0 = 위쪽 제한 없음)'
        for k_ in ('edge_birth', 'edge_px'):
            if k_ in q_: float(q_[k_][0])
        cm_ = None   # 11-11 confirm_ms: the confirmation time of a new track (0 = off = the frame count of --n-init). Checked before anything changes
        if 'confirm_ms' in q_:
            cm_ = float(q_['confirm_ms'][0])
            if not 0 <= cm_ <= MS_MAX: return False, '새 사람 확인 시간은 0 ~ %d ms (0 = 끔)' % MS_MAX
            if not APP.args.live_port: return False, '실시간 서버에서만 바꿀 수 있음'   # (APP.LIVE is still None when the saved choice is taken again at start)
        settings.WARM_MIN, settings.WARM_MAX = wn_, wx_
        if 'edge_birth' in q_: settings.EDGE_BIRTH = max(0.0, min(1.0, float(q_['edge_birth'][0])))
        if 'edge_px' in q_: settings.EDGE_PX = max(0.0, min(100.0, float(q_['edge_px'][0])))
    except (ValueError, IndexError): return False, '숫자가 아님'
    pend_ = {}
    if 'tracker' in q_:
        tk_ = q_['tracker'][0]
        if tk_ == 'gate': return False, 'GRU 확인은 없앰 (11-10)'   # v40
        if tk_ not in ('confirm3', 'rebind', 'ocsort', 'botsort', 'sort'): return False, '없는 추적기 종류'
        pend_['tracker'] = tk_
        if tk_ in ENGINE_KINDS:   # 11-13: the plain click means C++ first; the engine row can ask for Python
            pend_['engine'] = 'python' if q_.get('engine', [''])[0] == 'python' else 'auto'
    if 'relabel' in q_:
        if not APP.args.relabel and q_['relabel'][0] in ('1', 'true', 'on'): return False, '재식별 모델이 이 실행에 없음'
        pend_['relabel'] = q_['relabel'][0] in ('1', 'true', 'on')
    if 'predictor' in q_:   # v32
        pk_ = q_['predictor'][0]
        if pk_ not in {k_ for k_, _ in POS_NAMES}: return False, '없는 위치 예측기'   # 11-14: the list in position_memory (keypoint arms)
        if pk_.endswith('_kpt') and not det_has_kpt(): return False, '관절을 내는 검출 모델([관절])을 먼저 고르세요'   # 11-14 (user)
        if pk_ != 'kalman' and not any(p_['key'] == pk_ and p_['available'] for p_ in pos_list()): return False, '이 위치 예측기는 가중치 파일이 없음 (학습 전)'
        if pk_ != POSP['key']: pend_['predictor'] = pk_
    if 'pred_show' in q_: POSP['show'] = q_['pred_show'][0] in ('1', 'true', 'on')
    if cm_ is not None:   # read once per frame by the frame loop (confirm_apply); the tracker itself is only touched there
        if APP.CONFIRM is None: APP.CONFIRM = ConfirmByTime(cm_)
        else: APP.CONFIRM.set_ms(cm_)
        print('RULES set confirm_ms %g' % APP.CONFIRM.ms, flush=True)
    if 'app' in q_: APP.CTL['app'] = q_['app'][0]   # v40: read once per frame by the BoT-SORT feature step
    if rk_ is not None:
        with APP.REID['lock']: APP.REID['req'] = dict(key=rk_, t=time.time())
        print('RULES reid queued', rk_, flush=True)
    if 'lead_ms' in q_: APP.CTL['lead_ms'] = lm_
    if 'lead' in q_: APP.CTL['lead'] = q_['lead'][0] in ('1', 'true', 'on'); APP.CTL['lv'] = None; APP.CTL['lead_px'] = 0.0; APP.CTL['lead_err'] = ''; APP.CTL['lead_stats'] = {}
    if any(k_ in q_ for k_ in ('app', 'lead', 'lead_ms', 'warm_on')): print('RULES set app %s lead %d lead_ms %d' % (APP.CTL['app'], APP.CTL['lead'], APP.CTL['lead_ms']), flush=True)
    with APP.RULES['lock']:
        if pend_: APP.RULES['pending'] = dict(APP.RULES['pending'] or {}, **pend_, t=time.time())
        APP.RULES['changes'] += 1; APP.RULES['last'] = dict(at=time.strftime('%H:%M:%S'), warm_min=settings.WARM_MIN, warm_max=settings.WARM_MAX, edge_birth=settings.EDGE_BIRTH, queued=sorted(pend_.keys()))
    cam_tracker = getattr(APP.LIVE, 'camera_bytetrack', None)
    if 'tracker' in pend_ and cam_tracker is not None:
        cam_tracker.select(prefer_cpp_bytetrack(q_, pend_['tracker'], cam_tracker))
    print('RULES set warm_min %d warm_max %d edge_birth %.2f pending %s' % (settings.WARM_MIN, settings.WARM_MAX, settings.EDGE_BIRTH, pend_), flush=True)
    APP.sel_save(warm_min=settings.WARM_MIN, warm_max=settings.WARM_MAX, edge_birth=settings.EDGE_BIRTH, edge_px=settings.EDGE_PX, pred_show=bool(POSP['show']), app=APP.CTL['app'], lead=bool(APP.CTL['lead']), lead_ms=int(APP.CTL['lead_ms']), warm_prev=list(APP.CTL['warm_prev']), **(dict(confirm_ms=APP.CONFIRM.ms) if cm_ is not None else {}), **pend_)   # v36. 11-13 (user): the prediction switch (lead) is kept too
    return True, '적용됨' if not (pend_ or rk_ is not None) else '다음 프레임에 적용'


def rules_request_2(q_):
    if 'stab' not in q_: return APP.rules_request0_(q_)
    v_ = str(q_['stab'][0]).strip().lower()
    if v_ not in ('1', '0', 'true', 'false', 'on', 'off'): return False, '흔들림 보정 값은 1 또는 0'
    rest_ = {k_: x_ for k_, x_ in q_.items() if k_ != 'stab'}
    ok_, tx_ = APP.rules_request0_(rest_) if rest_ else (True, '적용됨')
    if not ok_: return ok_, tx_
    STAB['on'] = v_ in ('1', 'true', 'on'); STAB['err'] = ''; APP.sel_save(stab=bool(STAB['on']))   # read once per frame in live_next: the change lands between two frames
    print('RULES set stab %d' % STAB['on'], flush=True)
    return ok_, tx_


def model_state():
    APP.bank_refresh()
    with APP.MODEL['lock']:
        cur_ = next((c_ for c_ in APP.MODEL['catalog'] if c_['det'] == APP.args.det), None)
        return dict(catalog=[dict(id=c_['id'], name=c_['name'], det=c_['det']) for c_ in APP.MODEL['catalog']], current=cur_['id'] if cur_ else None,
                    pending=next((c_['id'] for c_ in APP.MODEL['catalog'] if APP.MODEL['req'] and c_['det'] == APP.MODEL['req']['det']), None), busy=APP.MODEL['busy'] or APP.MODEL['req'] is not None,
                    last=APP.MODEL['last'], history=APP.MODEL['history'][-5:])


def model_request(model_id, req_no=None):
    """-> PipelineResult code when decided now (0 Ok, 1 InvalidId, 2 Busy, 6 NotReady), or -1 when queued for the frame loop."""
    with APP.MODEL['lock']:
        if not APP.MODEL['catalog']: return 6
        c_ = next((x_ for x_ in APP.MODEL['catalog'] if x_['id'] == model_id), None)
        if c_ is None: return 1
        if APP.MODEL['req'] is not None or APP.MODEL['busy']: return 2
        feed = vars(APP).get('FEED')
        if feed is not None and feed.pending is not None: return 2
        if c_['det'] == APP.args.det: APP.sel_save(det=APP.args.det); return 0   # already running
        APP.MODEL['req'] = dict(det=c_['det'], id=c_['id'], req=req_no, t=time.time()); return -1


def det_has_kpt():
    """11-14: True when the running detector, or the one waiting to be switched in, has keypoint heads (kind six_kpt)."""
    if APP.args.det in KPT_DETS: return True
    pid_ = APP.MODEL.get('pending') if isinstance(getattr(APP, 'MODEL', None), dict) else None
    if pid_ is None: return False
    c_ = next((x_ for x_ in APP.MODEL['catalog'] if x_['id'] == pid_), None)
    return bool(c_ and c_['det'] in KPT_DETS)


def do_model_swap(new_det):
    """Release the old engine and create the new one on the same core. -> PipelineResult code (0 Ok, 3 CreationFailed, 4 Restored, 5 RestoreFailed)."""
    import gc
    c_ = next(x_ for x_ in APP.MODEL['catalog'] if x_['det'] == new_det); key_ = c_['key']; old_det, old_key = APP.args.det, APP.det_names[0]
    try: bad_ = (key_ in BANK or not APP.args.mock) and sha(BANK.get(key_, HERE / 'models' / 'links' / (key_ + '.dxnn'))) != APP.man[key_]
    except OSError: bad_ = True   # v28: the file was removed after it was listed
    if bad_: print('MODEL sha mismatch or missing file for', key_, flush=True); return 3
    t_ = time.perf_counter(); APP.eng.pop(old_key, None); APP.a = None; gc.collect(); time.sleep(0.3)
    try: APP.eng[key_] = APP.Engine(key_, CORE[key_], APP.args.npu_depth)
    except Exception as e_:
        print('MODEL creation of', key_, 'failed:', repr(e_)[:200], flush=True)
        try: APP.eng[old_key] = APP.Engine(old_key, CORE[old_key], APP.args.npu_depth); APP.a = APP.eng[old_key]; return 4
        except Exception as e2_: print('MODEL restore of', old_key, 'failed:', repr(e2_)[:200], flush=True); return 5
    APP.a = APP.eng[key_]; APP.b = None; APP.det_names = [key_]; APP.det_yoff = DET[new_det]['yoff']; APP.args.det = new_det
    if POSP['key'].endswith('_kpt') and new_det not in KPT_DETS:   # 11-14 (user): a keypoint position memory needs a keypoint detector
        pos_select('kalman'); print('POS predictor back to kalman: %s gives no keypoints' % new_det, flush=True)
    print('MODEL changed %s -> %s in %.1f s' % (old_det, new_det, time.perf_counter() - t_), flush=True); return 0
