"""두 검출 모델 상주와 즉시 전환 (2026-11-15, 사용자: 근거리·먼 거리 모델을 끊김 없이 바꾸기). 기본은 꺼짐. 장치에서 아직 안 돌려 봤다.

왜: 지금 검출 모델 바꾸기(page_settings.do_model_swap)는 옛 엔진을 놓고 0.3초 쉰 뒤 새 엔진을 같은 코어에 만들고, 도착하는 대로 넣기(frame_feed)는
NPU 코어 1 의 둘째 엔진을 다시 만든다(Feed.pause / resume). 그 사이 0.5초 넘게 프레임이 비고, 1초를 넘으면 추적기의 1초 입력 공백 규칙으로
번호가 모두 끝나고 짐벌 따라가기가 끊긴다. 검출 모델 자동 선택(auto_det.py)은 이 바꾸기를 자주 한다.

어떻게
  - 자동 선택의 두 모델(auto_det.NEAR_KEY, FAR_KEY, 또는 /rules/set?auto_det_near=, auto_det_far= 로 바꾼 이름)을 둘 다 NPU 코어 0 과 1 에 올린다 (엔진 4개).
    지금 쓰는 모델은 도착하는 대로 넣기가 이미 코어 0·1 에 올려 두었으므로, 다른 모델 하나만 코어 0·1 에 더 올린다(이 파일의 "다른 짝").
    NPU 코어 2 는 외형 모델(KeepID)과 엔드투엔드 모델 몫이라 쓰지 않는다.
  - 정해진 동안은 지금처럼 한 모델이 코어 0·1 을 번갈아 쓴다 (frame_feed.Feed 의 eng = [코어 0 엔진, 코어 1 엔진], 속도도 지금과 같다).
  - 바꾸기(instant) = 프레임 반복이 두 프레임 사이(app.py 의 모델 바꾸기 자리)에서 넣는 곳을 다른 짝으로 돌리고, do_model_swap 이 성공했을 때처럼
    APP.args.det, APP.det_names, APP.det_yoff, APP.b, APP.eng 를 바꾼다(page_settings.det_changed). APP.a 는 그대로 넣는 객체(Feed)다. 엔진을 놓지도 만들지도 않는다.
    그 뒤의 기록(APP.MODEL last / history, last_selection.json, 녹화 다시 시작, PC GUI 답)은 옛 길이 결과 0 일 때 하던 그대로다.
  - 옛 모델에 이미 넣은 프레임(두 프레임 사이에는 많아야 1장: 넣어 둔 것 또는 넣는 중인 것)은 버리지도 새 모델에 다시 넣지도 않고 옛 엔진에서 받아 그대로 쓴다
    (그 프레임의 옛 모델 검출, Feed 가 프레임마다 넣은 짝을 기억한다). 프레임 순서는 그대로다. 같은 코어에는 한 번에 한 장만 넣는다(Feed.held).
  - 그래서 두 모델은 입력과 출력의 모양이 같아야 한다(같은 입력 크기, 같은 출력 이름과 크기, 같은 y 위치, 같은 풀이 갈래): 바꾼 직후의 옛 모델 프레임 하나를
    새 모델 이름(APP.args.det)으로 풀기 때문이다. 자동 선택의 두 자세 모델은 같은 모양이다. 다르면 올리지 않는다(아래 옛 길).
  - 서버 기록: 바꿀 때마다 한 줄 'MODEL instant <old> -> <new> (resident, cores 0/1)'.
  - /rules 의 resident_det: 켬(on), 상주 중(active), 상주 중인 모델과 코어(resident, cores), 쉬는 까닭(reason 화면용, why 기록용), 즉시 바꾼 횟수(switches),
    마지막 바꾸기(last_switch: ms = 요청부터 새 모델이 검출한 첫 프레임의 결과를 프레임 반복이 받을 때까지(Feed.wait 뒤, 그 추론 시간 포함),
    ms_switch = 두 프레임 사이에서 바꾼 때부터, old_frames = 그 사이 옛 모델 프레임 수).

켜기: 시작 환경 LWIR_RESIDENT_DET=1 또는 /rules/set?resident_det=1. /rules/set 으로 고른 것은 last_selection.json 에 남아 다음 시작에도 쓰이고,
  환경 변수보다 우선한다(npu_feed.json 과 같은 규칙). 다른 짝은 프레임 반복이 두 프레임 사이에서 만든다(frame_step): 시작 직후, 켰을 때, 검출 모델이 두 모델 가운데
  하나가 되었을 때(예: 자동 선택을 켜서 근거리로 바뀐 뒤). 만드는 동안 프레임이 잠깐 멈추므로 짐벌이 따라가거나 움직이는 동안은 만들지 않고 기다린다. 끄면 다른 짝을 놓는다.
  옛 길로 두 모델 가운데 하나로 바꾼 뒤에는 그 뒤에 들어온 프레임을 프레임 반복이 하나 받을 때까지 만들지 않는다(두 멈춤이 합쳐져 1초를 넘지 않게, 각각은 지금 길이).

옛 길(do_model_swap)로 가는 경우. 한 줄을 찍고, 그 밖에는 지금과 같다.
  - 다른 짝을 못 만듦: NPU 메모리, 파일 없음, sha 다름, 두 모델의 입출력 모양이 다름. 다시 켜거나(resident_det=1) 두 모델 이름이 바뀔 때까지 다시 해 보지 않는다
    (그동안 /rules 는 'failed' 와 그 까닭을 보인다).
  - 켤 수 없는 실행: 도착하는 대로 넣기(코어 0·1 교대)가 아님, --mock, 두 개짜리(타일) 검출기, 검출 모델을 바꿀 수 없는 실행.
  - 기다림: 엔드투엔드 모델 사용 중(그 모델이 NPU 메모리를 쓴다. 켜면 다른 짝을 놓고, 끄면 다시 만든다), 두 모델 가운데 하나가 목록에 없음,
    지금 검출 모델이 두 모델이 아님, 짐벌이 따라가거나 움직임.
    엔드투엔드 모델은 다른 짝을 놓은 뒤에만 만든다(e2e_ready: 프레임 반복이 e2e_live.apply 바로 앞에서 부른다. 다른 짝이 아직 NPU 에서 마지막 프레임을 돌리면
    한 프레임 기다린다). 끄기는 기다리지 않는다.
  - 바꾸려는 모델의 파일이 없거나 sha 가 다름: 프레임 반복이 옛 길 전에 page_settings.swap_refused 로 먼저 보고 결과 3 으로 답한다. 아무것도 멈추거나 놓지 않는다.
  - 두 모델이 아닌 셋째 모델을 손으로 고름: 다른 짝을 먼저 놓고(옛 길이 쓰는 NPU 메모리는 지금과 같게) 옛 길로 바꾼다. 나중에 두 모델 가운데 하나로 돌아오면(옛 길)
    다른 짝을 다시 만든다. 셋째 모델과 짝 하나를 같이 두면 다음 바꾸기마다 무엇을 놓고 만들지 따로 정해야 하므로 "상주 = 근거리·먼 거리 두 모델" 하나만 지킨다.

아직 안 한 것
  - TODO 경계 구간(예: 100~140 px)에서 같은 프레임을 두 모델에 넣고 상자를 합치기 (모델뱅크 research/lwir_pose_plan/README.md 8절 (3)). 이번 단계에 없다.
  - 장치 NPU 메모리(엔진 4개 + KeepID)와 상태별(근거리 번갈아 / 먼 거리 번갈아) 초당 프레임·지연은 재지 않았다.
  - NPU 메모리가 모자랄 때 "근거리 두 코어 + 먼 거리 한 코어"로 줄이는 판은 없다(못 만들면 옛 길).

시험: test_resident_det.py (NPU 없이, 엔진은 흉내).
"""
import gc
import os
import threading
import time

import numpy as np

from ..common.state import APP
from ..common.shared_values import HERE, CORE, DET, sha
from .model_scan import BANK, W480_DETS, S6_DETS, KPT_DETS, E2E_DETS, RAW_DETS
from .frame_feed import SECOND
from . import auto_det
from ..track import e2e_live

CORES = (0, 1)              # NPU cores of both pairs; core 2 stays for KeepID / the end-to-end model
ENV = 'LWIR_RESIDENT_DET'   # '1' = on at start (a choice saved through /rules/set wins)
OLD_PATH = 'Model changes take the old path (release, wait, create).'
# why it cannot be resident now: code -> (server log, plain ASCII; page text). REFUSED codes do not go away while the server runs
WHY = dict(
    mock=('not with --mock', '--mock 실행에서는 쓰지 않음'),
    feed=('it needs the arrival feed on NPU cores 0 and 1 (LWIR_NPU_FEED=arrival with its second engine)', '도착하는 대로 넣기(NPU 0·1 교대)가 켜져 있지 않음'),
    tiled=('the detector uses two models (tiles)', '두 개짜리(타일) 검출 모델'),
    catalog=('this run cannot change its detector', '이 실행에서는 검출 모델을 바꿀 수 없음'),
    e2e=('the end-to-end model is on (it needs NPU memory)', '엔드투엔드 모델 사용 중 · 끄면 다시 올림'),
    missing=('the near or the far model is not listed', '근거리 또는 먼 거리 모델이 모델 목록에 없음'),
    other=('the detector in use is not one of the near / far pair', '지금 검출 모델이 근거리·먼 거리 모델이 아님'),
    gimbal=('waiting for the gimbal to stop following / moving (the frames stop while the engines are made)', '짐벌이 멈추기를 기다림 (올리는 동안 화면이 잠깐 멈춤)'),
    swap=('waiting for a frame after the old-path model change', '옛 길로 바꾼 뒤 새 프레임을 기다림'),
)
REFUSED = ('mock', 'feed', 'tiled', 'catalog')
QUIET = ('swap',)   # waits of a frame or two: no server log line
FAILED_KO = '다른 모델을 NPU 0·1에 올리지 못함 · 다시 켜면 다시 해 봄'


class _State:
    def __init__(self):
        self.lock = threading.Lock()   # frame thread (frame_step, instant, the take hook) vs web threads (state, parse_request)
        self.want = os.environ.get(ENV, '') == '1'
        self.source = 'env' if self.want else 'default'   # where on / off came from: default | env | rules (/rules/set, also the saved choice at start)
        self.pairs = None      # while resident: {det: dict(det, key, eng=[core 0, core 1], busy=[core 0, core 1])} = the model in use and the other one
        self.keys = None       # (near key, far key) the pairs were made for
        self.mode = 'off'      # off | waiting | refused | failed | on
        self.code = None       # WHY code while waiting / refused
        self.why = 'off'; self.reason = '꺼짐'
        self.failed = None     # dict(why, keys) of the last build that failed: no new try until turned on again or the pair changes
        self.drop = None       # (why, reason): a release waiting for the other pair's last frame on the NPU (next frame)
        self.said = None       # the last state printed (one server log line per change)
        self.switches = 0; self.fallbacks = 0; self.builds = 0; self.releases = 0
        self.last = None       # the last instant switch
        self.made = None       # the last build of the other pair
        self.watch = None      # the last instant switch, until the first result of the new model is collected
        self.swap_seq = None   # after an old-path change: the other pair is made only once the loop took a frame of this feed number or later
        self.err = ''


RES = _State()


def reset():
    """A fresh state (server start, tests). Reads LWIR_RESIDENT_DET again."""
    global RES
    RES = _State()


def _app(name, default=None):
    return vars(APP).get(name, default)


def _ascii(s):
    return str(s).encode('ascii', 'backslashreplace').decode()


def _set(mode, code, why, reason):
    R = RES
    if (R.mode, R.code, R.why, R.reason) != (mode, code, why, reason):   # called every frame while it waits: the lock only on a change
        with R.lock: R.mode, R.code, R.why, R.reason = mode, code, why, reason
    if mode in ('waiting', 'refused') and R.want and R.said != code and code not in QUIET:
        R.said = code
        print('RESIDENT %s: %s. %s' % (mode, _ascii(why), OLD_PATH), flush=True)


def _why_not(feed):
    """-> None when the other pair may be (or stay) resident now, 'busy' when a model / feed mode change is going on (look again at the next frame),
    else a WHY code."""
    a_ = _app('args')
    if a_ is None or getattr(a_, 'mock', False): return 'mock'
    if (feed is None or feed.direct or feed.closed or feed.err is not None or feed.mode != 'arrival' or len(feed.eng) != 2): return 'feed'
    if len(_app('det_names') or ()) != 1 or _app('b') is not None: return 'tiled'
    md = _app('MODEL')
    if not isinstance(md, dict) or not md.get('catalog'): return 'catalog'
    if md.get('req') is not None or md.get('busy') or feed.pending is not None: return 'busy'
    if e2e_live.E2E.get('want') == 'ftt' or e2e_live.skip_detector(): return 'e2e'
    near, far, _ = auto_det._entries()
    if near is None or far is None: return 'missing'
    if a_.det not in (near['det'], far['det']): return 'other'
    return None


def _gimbal_busy():
    gb = _app('GB')
    try: return gb is not None and bool(gb.state.get('auto') or gb.state.get('moving'))
    except Exception: return False


def _keys():
    return (auto_det.AUTO.near_key, auto_det.AUTO.far_key)


# ------------------------------------------------------------------------------------------------ frame loop (between two frames)
def frame_step(frame=None):
    """Frame loop, between two frames, after the model change block (app.py). Makes the other pair when it is wanted and can be made, lets it go
    when not. Never raises (an error is kept in state()['err']; model changes then take the old path)."""
    try: _frame_step(frame)
    except Exception as e_:
        RES.err = repr(e_)[:160]
        if RES.said != 'err': RES.said = 'err'; print('RESIDENT error: %s' % _ascii(RES.err), flush=True)


def _frame_step(frame):
    R = RES
    if R.drop is not None: _release(*R.drop)
    if not R.want:
        if R.pairs is not None: _release('turned off', '꺼서 놓음')
        if R.pairs is None and R.mode != 'off': _set('off', None, 'off', '꺼짐')   # after a release its reason stays
        return
    feed = _app('FEED')
    code = _why_not(feed)
    if code == 'busy': return
    if code is not None:
        if R.pairs is not None: _release(*WHY[code]); R.said = code   # the release line says why: no second line
        _set('refused' if code in REFUSED else 'waiting', code, *WHY[code]); return
    keys = _keys()
    if R.pairs is not None:
        if R.keys == keys:
            if R.drop is None and R.mode != 'on': _set('on', None, 'on', '상주 중 · 두 모델 사이는 즉시 바뀜')
            return
        _release('the near / far model names changed', '근거리·먼 거리 모델 이름이 바뀜')   # made again for the new pair at the next frame
        return
    if R.failed is not None:
        if R.failed['keys'] == keys:   # one try per turn-on (or per pair); /rules keeps saying why (a wait in between, e.g. a third model, had replaced it)
            if R.mode != 'failed': _set('failed', None, R.failed['why'], FAILED_KO); R.said = 'failed'
            return
        R.failed = None
    if R.swap_seq is not None:   # the old path just changed the model: its pause and this build stay two pauses (each as long as today), not one over 1 s
        if feed.prev_seq < R.swap_seq: _set('waiting', 'swap', *WHY['swap']); return
        R.swap_seq = None
    if _gimbal_busy():
        _set('waiting', 'gimbal', *WHY['gimbal']); return
    _build(feed, keys, frame)


def _check(cur_det, cur_key, det, key):
    """Checks before any engine is made (as do_model_swap): -> None or why not (log text)."""
    if (DET.get(det) or {}).get('names') != [key]: return '%s is not a single-engine detector' % det
    if CORE.get(key, 0) != 0: return '%s is not a core-0 detector' % key
    try: bad_ = sha(BANK.get(key, HERE / 'models' / 'links' / (key + '.dxnn'))) != APP.man.get(key)
    except OSError: bad_ = True   # the file was removed after it was listed
    if bad_: return 'sha mismatch or missing file for %s' % key
    shape_ = _app('det_shape') or {}
    same_ = (all((cur_det in s_) == (det in s_) for s_ in (W480_DETS, S6_DETS, KPT_DETS, E2E_DETS)) and RAW_DETS.get(cur_det) == RAW_DETS.get(det)
             and (DET.get(cur_det) or {}).get('yoff') == DET[det].get('yoff') and shape_.get(cur_key) == shape_.get(key))
    if not same_: return '%s and %s do not have the same input / output form (the frame of the old model taken just after a switch is decoded with the new names)' % (cur_det, det)
    return None


def _form(e_):
    return (tuple(int(v) for v in e_.in_shape), tuple(e_.names), tuple(tuple(int(d) for d in s_.shape) for s_ in e_.slots),
            bool(getattr(e_, 'split6', False)), bool(getattr(e_, 'kpt', False)))


def _warm(e_):   # as frame_feed.make_second: one run on a blank picture
    try:
        APP.pending[e_.name] = e_.b.run_async(np.zeros(e_.in_shape, np.uint8), user_arg={'f': -1}, output_buffer=e_.packed); e_.wait()
    except BaseException:
        APP.pending.pop(e_.name, None); raise


def _build(feed, keys, frame):
    """Make the other model of the pair on NPU cores 0 and 1. -> True when resident."""
    R = RES
    near, far, _ = auto_det._entries()
    cur_det, cur_key = APP.args.det, APP.det_names[0]
    oth = far if cur_det == near['det'] else near
    det, key = oth['det'], oth['key']
    why = _check(cur_det, cur_key, det, key)
    made = []; t0 = time.perf_counter()
    if why is None:
        L_ = APP.LIVE
        with L_.cond:   # nothing is submitted while the engines are made (as Feed.pause, but nothing is drained: the frame on the NPU is taken as usual)
            feed.paused = True
            while feed.working and feed.th.is_alive(): L_.cond.wait(0.2)
        try:
            for core_ in CORES:
                e_ = APP.Engine(key, core_, 1) if core_ == 0 else APP.Engine(key + SECOND, core_, 1, model=key)   # names as do_model_swap / make_second
                made.append(e_); _warm(e_)
            ref_ = _form(feed.eng[0])
            for e_ in made:
                if _form(e_) != ref_: raise ValueError('engine form %s is not that of %s %s' % (_form(e_), cur_key, ref_))
        except Exception as x_:
            why = 'the %s engines on NPU cores 0 and 1 could not be made (%s)' % (key, repr(x_)[:160])
        finally:
            with L_.cond: feed.paused = False; L_.cond.notify_all()
    if why is not None:
        for e_ in made:
            try: e_.b.dispose()
            except Exception: pass
        made = None; gc.collect()
        with R.lock:
            R.failed = dict(why=why, keys=keys); R.mode = 'failed'; R.code = None; R.why = why
            R.reason = FAILED_KO; R.said = 'failed'
        print('RESIDENT refused: %s. %s' % (_ascii(why), OLD_PATH), flush=True)
        return False
    with R.lock:
        R.pairs = {cur_det: dict(det=cur_det, key=cur_key, eng=feed.eng, busy=feed.busy),   # the feed's own lists: they are the pair in use
                   det: dict(det=det, key=key, eng=made, busy=[False] * len(made))}
        R.keys = keys; R.builds += 1; R.said = 'on'
        R.made = dict(det=det, key=key, seconds=round(time.perf_counter() - t0, 2), at=time.strftime('%H:%M:%S'), frame=frame)
        R.mode = 'on'; R.code = None; R.why = 'on'; R.reason = '상주 중 · 두 모델 사이는 즉시 바뀜'
    print('RESIDENT on: %s and %s on NPU cores 0 and 1 (4 engines; %s made in %.1f s at frame %s). A change between them is instant.'
          % (cur_det, det, key, R.made['seconds'], frame), flush=True)
    return True


def _release(why, reason, final=False):
    """Let the other pair go (both engines disposed); the pair in use stays the feed's (or, after Feed.pause, the old path's).
    -> True when done, False when the other pair still has a frame on the NPU (kept in drop and tried again at the next frame; final = nothing is in flight)."""
    R = RES
    if R.pairs is None: R.drop = None; return True
    feed = _app('FEED'); L_ = _app('LIVE')
    use = (next((p for p in R.pairs.values() if feed is not None and p['eng'] is feed.eng), None) or R.pairs.get(getattr(_app('args'), 'det', None))
           or next((p for p in R.pairs.values() if p['key'] in (_app('eng') or {})), None))
    oth = [p for p in R.pairs.values() if p is not use] if use is not None else []   # not found: dispose nothing (app.py owns the engine in use)
    if not final and any(any(p['busy']) for p in oth):
        R.drop = (why, reason); return False
    if feed is not None:
        if L_ is not None:
            with L_.cond:
                if any(feed.held is p['busy'] for p in oth): feed.held = None
                feed.on_take = None
        else: feed.held = None; feed.on_take = None
    with R.lock:
        R.pairs = None; R.keys = None; R.drop = None; R.watch = None; R.releases += 1
        R.mode = 'off'; R.code = None; R.why = why; R.reason = reason
    names = [p['det'] for p in oth]
    for p in oth:
        for e_ in p['eng']:
            try: e_.b.dispose()
            except Exception: pass
        p['eng'] = None
    oth = use = None; gc.collect()
    print('RESIDENT off: %s; %s left NPU cores 0 and 1. %s' % (_ascii(why), ', '.join(names), OLD_PATH), flush=True)
    return True


def _fallback():
    if RES.want: RES.fallbacks += 1
    return None


def instant(rq_):
    """Frame loop, top of a frame (app.py v23 model change block), before the old path. rq_ = APP.MODEL['req'].
    -> 0 when the detector was switched at once to the other resident pair (no engine made or released; the bookkeeping after it is the old path's
    for result 0), None when the old path (Feed.pause, before_swap, do_model_swap, Feed.resume) must do it."""
    R = RES
    try:
        feed = _app('FEED'); L_ = _app('LIVE')
        if not R.want or R.pairs is None or R.drop is not None: return _fallback()
        tgt = R.pairs.get(rq_['det']); cur = R.pairs.get(APP.args.det)
        if (tgt is None or cur is None or tgt is cur or feed is None or L_ is None or feed.direct or feed.closed or feed.err is not None
                or feed.paused or feed.pending is not None or feed.eng is not cur['eng']): return _fallback()
    except Exception as e_:
        R.err = repr(e_)[:160]; return _fallback()
    t0 = time.perf_counter(); old = APP.args.det
    with L_.cond:   # the feeding thread picks its engine under this lock: every frame from here goes to the new pair; the one already submitted is taken from the old pair
        feed.eng, feed.busy, feed.held = tgt['eng'], tgt['busy'], cur['busy']
        feed.on_take = _taken
        L_.cond.notify_all()
    APP.eng.pop(cur['key'], None); APP.eng[tgt['key']] = tgt['eng'][0]   # as do_model_swap: APP.eng holds the core-0 engine of the model in use (app.py closes it at the end)
    from ..web import page_settings
    page_settings.det_changed(tgt['det'], tgt['key'])
    t_req = rq_.get('t')
    with R.lock:
        R.switches += 1
        R.last = dict({'from': old, 'to': tgt['det']}, at=time.strftime('%H:%M:%S'), queued_ms=round((time.time() - t_req) * 1000.0, 1) if t_req else None,
                      ms=None, ms_switch=None, old_frames=None)
        R.watch = dict(eng=tgt['eng'], t_req=t_req, t0=t0, old=0)
    print('MODEL instant %s -> %s (resident, cores 0/1)' % (old, tgt['det']), flush=True)
    return 0


def _taken(r_, g_=None):
    """Feed.on_take (frame thread, Feed.wait, the result collected): after an instant switch, each frame until the first one submitted to the new pair.
    g_ = the engine pair the frame was submitted to (Feed.wait reads it before _collect drops it from the record)."""
    R = RES; w_ = R.watch; feed = _app('FEED')
    try:
        if w_ is None:
            if feed is not None: feed.on_take = None
            return
        g_ = g_ if g_ is not None else getattr(r_, 'grp', None)
        if g_ is None or g_[0] is not w_['eng']:
            w_['old'] += 1; return   # submitted to the old model before the switch: its result is used as it is
        now_ = time.time(); pc_ = time.perf_counter()
        with R.lock:
            if R.last is not None:
                R.last.update(ms=round((now_ - w_['t_req']) * 1000.0, 1) if w_['t_req'] else None, ms_switch=round((pc_ - w_['t0']) * 1000.0, 1), old_frames=w_['old'])
            R.watch = None
        if feed is not None: feed.on_take = None
    except Exception as e_:
        R.err = repr(e_)[:160]


def before_swap(det):
    """Frame loop, the old path, right after Feed.pause (nothing is on the NPU then) and before do_model_swap. The other pair is released here, so
    do_model_swap works with the NPU memory of today (it releases the engine of the model in use and makes the new one; pause / resume do core 1).
    It is made again by frame_step when the detector is one of the pair after the change, once the loop has taken a frame that came after it (swap_seq)."""
    R = RES
    try:
        feed = _app('FEED')
        if feed is not None: R.swap_seq = feed.seq   # the feed is paused: frames numbered from here are the ones after this change
        if R.pairs is None: return
        if not R.want: w_ = ('turned off', '꺼서 놓음')
        elif det not in R.pairs: w_ = ('%s is not one of the pair (old path)' % det, '두 모델이 아닌 모델(%s)로 바꿈 · 옛 길' % det)
        else: w_ = ('the instant switch was not possible now (old path)', '지금은 즉시 바꿀 수 없어 옛 길로 바꿈')
        _release(*w_, final=True)
    except Exception as e_:
        R.err = repr(e_)[:160]; print('RESIDENT release failed: %s' % _ascii(R.err), flush=True)


def e2e_ready():
    """Frame loop, right before e2e_live.apply (the end-to-end model is made there). When it is about to be made, the other pair is let go first:
    the click on the page may come after this frame's frame_step. -> False while the other pair still has a frame on the NPU (the end-to-end model
    is then made a frame later; frame_step finishes the release). Turning it off is never held back. Never raises."""
    try:
        if e2e_live.E2E.get('want') != 'ftt' or e2e_live.E2E.get('on') == 'ftt' or RES.pairs is None: return True
        _release(*WHY['e2e']); RES.said = 'e2e'   # the release line says why: no second line from frame_step
        return RES.pairs is None
    except Exception as e_:
        RES.err = repr(e_)[:160]
        return RES.pairs is None


def close():
    """End of the frame loop, after Feed.close (nothing is on the NPU then): the other pair is released."""
    try:
        if RES.pairs is not None: _release('the server stops', '서버 끝', final=True)
    except Exception as e_:
        print('RESIDENT close failed: %s' % _ascii(repr(e_)[:160]), flush=True)


# ------------------------------------------------------------------------------------------------ web (/rules, /rules/set)
def state():
    """The /rules answer, key resident_det."""
    with RES.lock:
        R = RES
        det = getattr(_app('args'), 'det', None)
        pairs = [dict(det=p['det'], key=p['key'], cores=list(CORES), in_use=p['det'] == det) for p in (R.pairs or {}).values()]
        return dict(on=R.want, active=R.pairs is not None and R.drop is None, mode=R.mode, reason=R.reason, why=R.why, source=R.source, env=os.environ.get(ENV, ''),
                    keys=dict(near=auto_det.AUTO.near_key, far=auto_det.AUTO.far_key), resident=pairs, cores=list(CORES) if R.pairs else [], current_det=det,
                    switches=R.switches, fallbacks=R.fallbacks, builds=R.builds, releases=R.releases,
                    last_switch=dict(R.last) if R.last else None, made=dict(R.made) if R.made else None,
                    failed=R.failed['why'] if R.failed else None, release_waiting=R.drop[0] if R.drop else None, err=R.err)


def parse_request(q_):
    """q_: {'resident_det': [value]} of a /rules/set query (lists, as parse_qs gives them). -> (ok, text, apply); apply() makes the change
    (called by the caller only when the rest of the query was taken too). The frame loop makes or releases the other pair at its next frame."""
    x_ = q_.get('resident_det')
    v_ = str(x_[0] if isinstance(x_, (list, tuple)) and x_ else x_).strip().lower()
    if v_ not in ('1', '0', 'true', 'false', 'on', 'off'): return False, '두 모델 상주 값은 1 또는 0', None
    on_ = v_ in ('1', 'true', 'on')

    def apply():
        with RES.lock:
            R = RES; R.want = on_; R.source = 'rules'; R.said = None
            if on_: R.failed = None   # turned on (again): one more try
        print('RULES set resident_det %d' % on_, flush=True)
        sel_ = _app('sel_save')
        if sel_ is not None: sel_(resident_det=on_)
    return True, '다음 프레임에 적용', apply
