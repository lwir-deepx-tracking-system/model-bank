"""검출 모델 자동 선택 (11-15, 사용자: "상황에 맞는 최적 모델 알아서 선택"). 기본은 꺼짐.

모델뱅크의 두 검출 모델 가운데 하나를 사람 크기와 가림으로 고른다.
- 근거리(NEAR, 기본) = NEAR_KEY: 실내 가까운 거리, 가려진 사람에서 가장 좋음.
- 먼 거리(FAR) = FAR_KEY: 작고 먼 사람에서 훨씬 좋음. 모델 폴더에 아직 없으면 "먼 거리 모델 없음"을 알리고 쉰다.
둘 다 같은 몸 네모(머리~골반)와 관절 17개를 내므로 바꾼 뒤에도 추적이 이어지게 만든 짝이다.

크기 신호 = 짐벌이 따라가는 사람의 상자 높이(px)의 최근 median_s(1 s) 가운뎃값. 화면 끝에 닿은 프레임은 뺀다(잘린 상자는
작게 나와 "멀다"로 잘못 읽힌다). 따라가는 사람이 없으면 이 프레임의 확인된 사람 상자(점수 min_score 이상) 높이의 가운뎃값.
가림 신호 = 그 사람의 보이는 관절 수(관절 내는 검출 모델일 때). 관절이 kpt_seen(3)개 이상 또렷이 보이는데 kpt_min(12)개보다
적으면 가려진 사람(책상 뒤에 앉으면 다리·골반이 가려 11개 이하). 거의 안 보이는 작은 사람(관절 3개 미만)은 가림이 아니라 불확실로 본다.

규칙 (11-15 조사 권고 close_range_20261113/research_model_selection_20261115/RECOMMENDATION.md 2절 판 1. 값은 모두 PARAMS,
/rules/set?auto_det_<이름>=값 으로 바꿈):
- 사람을 잃으면(따라가던 사람이 안 보이거나 화면이 비면) lost_s(0.5 s) 뒤 바로 FAR. 머무름·1분 상한을 무시한다. 근거리 모델은 작은 사람을
  아예 못 보므로 작아지기를 기다리면 그 전에 사람을 잃는다. 마지막에 본 모습이 가려졌거나 잘렸으면(책상 뒤로, 화면 밖으로) lost_occl_s(3 s) 뒤.
- 크기가 far_px(100) 아래에 hold_s(1.0 s) 동안 계속 있으면 FAR. 단 가려졌거나 잘린 사람이면 작아도 FAR 로 가지 않는다(실내에서 상자가
  작은 이유는 대개 멀어서가 아니라 가려져서).
- 크기가 near_px(140) 위에 hold_s 동안 계속 있으면 NEAR. 또는 far_px 이상인데 가려진 상태가 hold_s 동안 이어지면 NEAR.
  그 사이는 그대로(이력 현상). 지금 모델이 둘 다 아니면(손으로 고른 다른 모델) far_px 이상이 NEAR 쪽이다.
- 바꾼 뒤(자동이든 손이든) dwell_s(5 s) 동안은 다시 바꾸지 않는다. 1분에 max_per_min(4)번까지만 바꾼다. 사람을 잃어서 바꾸는 경우만 예외.
- 크기로 바꿀 때 따라가는 사람이 빠르게 움직이면(1초에 상자 높이의 fast_hps(1.0)배 넘게) defer_s(3 s)까지 미룬다
  (바꾸는 동안 검출이 비는 사이 빠른 사람은 놓치기 쉽다).
- empty_hold_s(3 s) 안에 다시 보이면 쌓던 시간이 이어지고(먼 사람은 검출이 자주 끊긴다), 그보다 길게 없으면 처음부터 다시 센다.
- 바꾸기는 화면의 모델 고르기와 같은 길로만 한다: APP.model_request(id) -> APP.MODEL['req'] -> 프레임 루프가 두 프레임 사이에
  do_model_swap. 이 파일은 NPU 를 부르지 않는다. 다른 모델 바꾸기가 기다리거나 진행 중이면 요청하지 않는다.
- 녹화 재생 중에는 바꾸지 않는다 (검출 모델을 바꾸면 재생이 처음부터 다시 시작한다). 대신 시험 운전(dry run): 같은 규칙으로
  판단만 하고, 바꿨을 순간(재생 위치, 시각, 신호 전부: 크기, 관절 수, 가림, 잘림, 속도, 사람 수, 평균 점수)을 남긴다
  (/rules 의 auto_det.dry_events, dry_recent). 화면 밝기는 남기지 않는다(열화상 장기 자료에서 성능과의 관계가 약했음, 권고 2절).
- 엔드투엔드 모델을 쓰는 동안(검출 모델을 안 씀)은 쉰다.
- 손으로 검출 모델을 고르면(화면, PC GUI) 자동 선택이 꺼진다 (manual_choice).

Chooser / Smooth / observe 는 APP 도 NPU 도 없는 판단만이라 PC 에서 시험한다 (test_auto_det.py). on_frame / state / parse_request 가
서버에 붙는 곳이다.
"""
import collections
import re
import threading
import time

from ..common.state import APP

NEAR_KEY = 'yolo26m_pose_otp2cocomot_20261114_bodybox_occl_kpt_640x480'
FAR_KEY = 'yolo26m_pose_otp2cocomotllvip_20261115_bodybox_far_kpt_640x480'
# reasons for every default: research_model_selection_20261115/RECOMMENDATION.md + the plan README section 8 (model bank lwir_pose_plan)
PARAMS = dict(far_px=100.0, near_px=140.0, hold_s=1.0, dwell_s=5.0, empty_hold_s=3.0, min_score=0.25,
              lost_s=0.5, lost_occl_s=3.0, median_s=1.0, edge_px=3.0, kpt_min=12.0, kpt_seen=3.0, kpt_vis=0.5,
              max_per_min=4.0, fast_hps=1.0, defer_s=3.0)
LIMITS = dict(far_px=(4.0, 480.0), near_px=(4.0, 480.0), hold_s=(0.0, 30.0), dwell_s=(0.0, 600.0), empty_hold_s=(0.0, 600.0),
              min_score=(0.0, 1.0), lost_s=(0.0, 60.0), lost_occl_s=(0.0, 600.0), median_s=(0.0, 10.0), edge_px=(0.0, 100.0),
              kpt_min=(0.0, 17.0), kpt_seen=(0.0, 17.0), kpt_vis=(0.0, 1.0), max_per_min=(1.0, 60.0), fast_hps=(0.0, 100.0), defer_s=(0.0, 60.0))
PERSON_CLS = 0
YOFF = 16.0            # tracker rows: 640x512 canvas, the 640x480 picture starts at y 16 (as page_handler)
KP_IOU = 0.5           # a track box and a detection box with keypoints are the same person at IoU >= this (as position_memory)
FAIL_WAIT_S = 60.0     # a switch the frame loop could not make (file broken, engine not created): no new try for this long
FAIL_OFF = 3           # ... and after this many failures in a row the automatic choice turns itself off
REFUSED_WAIT_S = 1.0   # model_request said no (another change queued in the arrival feed): ask again after this long
EVENTS_KEEP = 500      # dry-run events kept in memory (count keeps growing)
DRY_STEP_MAX = 0.25    # dry run: the most one frame adds to the replay clock (s); replays run at 5 ~ 30 frames a second
KEY_RE = re.compile(r'^[A-Za-z0-9_.-]{1,120}$')
KIND_KO = dict(near='근거리', far='먼 거리', other='다른 모델')


# ------------------------------------------------------------------------------------------------ pure logic (no APP)
def _median(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def kp_count(box, kp_sets, vis=PARAMS['kpt_vis']):
    """Visible keypoints (score >= vis) of the detection that is this track box (best IoU >= KP_IOU over the keypoint sets given,
    newest first: detect_frame.KP cur, prev). None = no keypoints for this box (detector without keypoints, or no match)."""
    best, n = KP_IOU, None
    for s in kp_sets:
        if not s: continue
        try:
            bs, ks = s['boxes'], s['kpts']
            for j in range(len(bs)):
                o = _iou(box, [float(x) for x in bs[j][:4]])
                if o >= best: best = o; n = int(sum(1 for k in ks[j] if float(k[2]) >= vis))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return n


def observe(rows, followed_id=0, p=None, stab=None, kp_sets=(), wh=(640.0, 480.0)):
    """One frame. rows: tracker output rows [x1, y1, x2, y2, id, score, cls, ...] in canvas pixels (the rows drawn and sent to the gimbal).
    stab = stabilizer correction of this frame (rows - stab = real camera position, for the picture-edge test). kp_sets = detect_frame.KP sets.
    -> dict(px, src, id, n, edge, occl, kpts, cx, cy, score)
    src: 'followed' = the followed person's box; 'followed_missing' = following, but that person is not in this frame (px None);
    'scene' = median height of the person rows with score >= min_score that do not touch the picture edge (all of them if every one does,
    then edge True); 'none' = nobody (px None). occl: keypoints say partly hidden (followed: that person; scene: more than half the people)."""
    p = PARAMS if p is None else p
    W, H = wh; m = p['edge_px']
    cx_, cy_ = (stab[0], stab[1]) if stab is not None else (0.0, 0.0)
    people = []
    for r in rows if rows is not None else ():
        try:
            if int(r[6]) != PERSON_CLS: continue
            b = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
            h = b[3] - b[1]
            if not h > 0: continue
            x1, x2, y1, y2 = b[0] - cx_, b[2] - cx_, b[1] - cy_ - YOFF, b[3] - cy_ - YOFF
            edge = x1 <= m or y1 <= m or x2 >= W - m or y2 >= H - m
            people.append(dict(id=int(r[4]), h=h, score=float(r[5]), box=b, edge=edge))
        except (TypeError, ValueError, IndexError):
            continue

    def occl_of(q):
        n = kp_count(q['box'], kp_sets, p['kpt_vis']) if kp_sets else None
        return n, n is not None and p['kpt_seen'] <= n < p['kpt_min']

    if followed_id:
        for q in people:
            if q['id'] == followed_id:
                n, o = occl_of(q)
                return dict(px=q['h'], src='followed', id=q['id'], n=1, edge=q['edge'], occl=o, kpts=n,
                            cx=0.5 * (q['box'][0] + q['box'][2]), cy=0.5 * (q['box'][1] + q['box'][3]), score=q['score'])
        return dict(px=None, src='followed_missing', id=int(followed_id), n=0, edge=False, occl=False, kpts=None, cx=None, cy=None, score=None)
    ok = [q for q in people if q['score'] >= p['min_score']]
    if not ok: return dict(px=None, src='none', id=0, n=0, edge=False, occl=False, kpts=None, cx=None, cy=None, score=None)
    clean = [q for q in ok if not q['edge']]
    use = clean or ok
    occ = [occl_of(q) for q in use]
    ks = [n for n, _ in occ if n is not None]
    return dict(px=_median([q['h'] for q in use]), src='scene', id=0, n=len(ok), edge=not clean,
                occl=sum(1 for _, o in occ if o) * 2 > len(occ), kpts=int(_median(ks)) if ks else None, cx=None, cy=None,
                score=round(sum(q['score'] for q in ok) / len(ok), 3))


def size_signal(rows, followed_id=0, min_score=PARAMS['min_score']):
    """(px or None, source, track id, people counted): observe() without the edge, keypoint and speed parts."""
    o = observe(rows, followed_id, dict(PARAMS, min_score=min_score, edge_px=-1e9))
    return o['px'], o['src'], o['id'], o['n']


class Smooth:
    """The size signal over the last median_s seconds of one person (or of the scene): median height of the frames whose box does not
    touch the picture edge, cut = every frame in the window touches it, occl = more than half the frames partly hidden,
    speed = box centre movement per second in box heights (followed person only). A new person (or the scene) starts again."""
    def __init__(self):
        self.buf = collections.deque(); self.key = None

    def add(self, t, o, window):
        key = (o['src'] in ('followed', 'followed_missing'), o['id'])
        if key != self.key: self.buf.clear(); self.key = key
        if o['px'] is None: return dict(px=None, cut=False, occl=False, speed=None)
        self.buf.append((t, o['px'], o['edge'], o['occl'], o['cx'], o['cy']))
        while len(self.buf) > 1 and self.buf[0][0] < t - window: self.buf.popleft()
        clean = [b[1] for b in self.buf if not b[2]]
        px = _median(clean or [b[1] for b in self.buf])
        occl = sum(1 for b in self.buf if b[3]) * 2 > len(self.buf)
        speed = None
        a, z = self.buf[0], self.buf[-1]
        if a[4] is not None and z[4] is not None and z[0] - a[0] >= 0.2 and px > 0:
            speed = ((z[4] - a[4]) ** 2 + (z[5] - a[5]) ** 2) ** 0.5 / (z[0] - a[0]) / px
        return dict(px=px, cut=not clean, occl=occl, speed=speed)


class Chooser:
    """The decision only: call step() once per frame with the frame time (s), the smoothed size (px or None = person lost / nobody),
    the model in use ('near' | 'far' | 'other'), whether the person is partly hidden or cut by the edge (occl) and moving fast (fast).
    It returns 'near' or 'far' when that model should be asked for now, else None.
    The caller tells it about every switch with switched(t) (auto or by hand), so the dwell time and the per-minute count start there."""
    def __init__(self, params=None):
        self.p = dict(PARAMS, **(params or {}))
        self.last_switch_t = None
        self.times = collections.deque()   # switch times (the last minute)
        self.last_occl = False             # how the person looked the last time they were seen
        self.clear()

    def clear(self):   # forget the collected time (not the last switch: the dwell stays)
        self.below_t = self.above_t = None   # since when the signal stays below far_px / on the NEAR side
        self.empty_t = None                  # since when there is no signal
        self.due_t = None                    # since when a size switch waits for the person to slow down
        self.reason = ''
        self.forced = False                  # the last asked switch was a "person lost" one (no dwell, no cap)

    def switched(self, t, count=True):
        """A switch happened at t (count=False: the same switch told twice, at the request and when the frame loop made it).
        The "person lost" clock starts again here: the frames of the change itself (no detector) are not a lost person."""
        self.last_switch_t = t; self.below_t = self.above_t = None; self.due_t = None; self.empty_t = None
        if count: self.times.append(t)

    def dwell_left(self, t):
        if self.last_switch_t is None: return 0.0
        return max(0.0, self.p['dwell_s'] - (t - self.last_switch_t))

    def recent(self, t):
        while self.times and self.times[0] < t - 60.0: self.times.popleft()
        return len(self.times)

    def step(self, t, px, current, occl=False, fast=False):
        self._due = False; self.forced = False
        w = self._step(t, px, current, occl, fast)
        if not self._due: self.due_t = None
        return w

    def _step(self, t, px, current, occl, fast):
        p = self.p
        if px is None:
            if self.empty_t is None: self.empty_t = t
            gap = t - self.empty_t
            if gap > p['empty_hold_s']: self.below_t = self.above_t = None
            wait = p['lost_occl_s'] if self.last_occl else p['lost_s']
            how = '가려진 채로 ' if self.last_occl else ''
            if current == 'far': self.reason = '사람 안 보임 %.1f초 · 먼 거리 모델 유지' % gap; return None
            if gap >= wait: return self._want('far', t, '%s사람 안 보임 %.1f초 → 먼 거리' % (how, gap), force=True)
            self.reason = '%s사람 안 보임 %.1f초 · %g초 넘으면 먼 거리' % (how, gap, wait); return None
        if self.empty_t is not None:
            if t - self.empty_t > p['empty_hold_s']: self.below_t = self.above_t = None
            self.empty_t = None
        self.last_occl = bool(occl)
        hid = '가려짐/잘림 ' if occl else ''
        if px < p['far_px']:
            self.above_t = None
            if occl:   # small because hidden or cut, not because far: the near model is the one for this
                self.below_t = None
                self.reason = '작음 %.0f px 이지만 가려짐/잘림 · 먼 거리로 안 바꿈' % px; return None
            if self.below_t is None: self.below_t = t
            held = t - self.below_t
            if current == 'far': self.reason = '작음 %.0f px · 먼 거리 모델 유지' % px; return None
            if held >= p['hold_s']: return self._want('far', t, '작음 %.0f px < %g px · %.1f초' % (px, p['far_px'], held), fast=fast)
            self.reason = '작음 %.0f px < %g px · %.1f/%g초' % (px, p['far_px'], held, p['hold_s']); return None
        if px > p['near_px'] or current == 'other' or occl:   # big; a model that is neither of the two; partly hidden from far_px up
            self.below_t = None
            if self.above_t is None: self.above_t = t
            held = t - self.above_t
            if current == 'near': self.reason = '%s큼 %.0f px · 근거리 모델 유지' % (hid, px); return None
            lim = p['near_px'] if px > p['near_px'] else p['far_px']
            if held >= p['hold_s']: return self._want('near', t, '%s큼 %.0f px > %g px · %.1f초' % (hid, px, lim, held), fast=fast)
            self.reason = '%s큼 %.0f px · %.1f/%g초' % (hid, px, held, p['hold_s']); return None
        self.below_t = self.above_t = None
        self.reason = '중간 %.0f px (%g~%g px) · 지금 모델 유지' % (px, p['far_px'], p['near_px'])
        return None

    def _want(self, to, t, why, force=False, fast=False):
        if force:
            self.reason = why + ' (사람을 잃어 머무름·횟수 제한 없이)'; self.forced = True
            return to
        left = self.dwell_left(t)
        if left > 0:
            self.reason = why + ' · 바꾼 뒤 %g초 유지 (%.1f초 남음)' % (self.p['dwell_s'], left); return None
        n = self.recent(t)
        if n >= self.p['max_per_min']:
            self.reason = why + ' · 1분에 %d번까지 (%.0f초 뒤)' % (self.p['max_per_min'], self.times[0] + 60.0 - t); return None
        if fast:
            self._due = True
            if self.due_t is None: self.due_t = t
            if t - self.due_t < self.p['defer_s']:
                self.reason = why + ' · 빠르게 움직이는 중이라 미룸 (%.1f/%g초)' % (t - self.due_t, self.p['defer_s']); return None
        self.reason = why
        return to


# ------------------------------------------------------------------------------------------------ server glue (APP)
class _State:
    def __init__(self):
        self.lock = threading.Lock()   # on_frame (frame thread) vs state / parse_request (web threads)
        self.on = False
        self.params = dict(PARAMS)
        self.near_key, self.far_key = NEAR_KEY, FAR_KEY
        self.ch = Chooser(self.params); self.ch.p = self.params   # one dict: a threshold change reaches the chooser at once
        self.sm = Smooth()
        self.mode = 'off'      # off | waiting (a model missing / cannot change) | paused | live | dry_run
        self.reason = '꺼짐'
        self.signal = None
        self.seen_det = None   # detector seen at the last frame (a change = a switch, whoever made it)
        self.pending = None    # this module's request, until the frame loop has taken it
        self.last_switch = None; self.switches = 0; self.failures = 0
        self.retry_t = None    # no new request before this frame time
        self.dry_ch = None; self.dry_sm = None; self.dry_cur = None; self.dry_clock = 0.0; self.dry_last = None; self.dry_events = []; self.dry_n = 0   # replay (dry run): own chooser and clock
        self.err = ''


AUTO = _State()


def reset():
    """A fresh state (tests; a new server start keeps the module, so the start code may call this too)."""
    global AUTO
    AUTO = _State()


def _app(name, default=None):
    return vars(APP).get(name, default)


def _entries():
    """-> (near catalog entry or None, far entry or None, the detector can be changed in this run)."""
    md = _app('MODEL')
    if not isinstance(md, dict): return None, None, False
    with md['lock']:
        cat = list(md.get('catalog') or [])
    pick = lambda k: next((dict(c) for c in cat if c.get('key') == k or c.get('det') == 'bank:' + k), None)
    return pick(AUTO.near_key), pick(AUTO.far_key), bool(cat)


def _kind(det, near, far):
    if near is not None and det == near['det']: return 'near'
    if far is not None and det == far['det']: return 'far'
    return 'other'


def _followed():
    """Track id the gimbal follows now, 0 = nobody (no gimbal, not following, or a selection being reset)."""
    gb = _app('GB')
    if gb is None: return 0
    try:
        st = gb.state
        if not st.get('auto') or getattr(gb, 'selection_reset', False): return 0
        return int(st.get('id', 0) or 0)
    except Exception:
        return 0


def _kp_sets():
    """Keypoints of the last two detector frames (detect_frame.KP), only while the detector in use has keypoint heads."""
    try:
        from .model_scan import KPT_DETS
        if getattr(_app('args'), 'det', None) not in KPT_DETS: return ()
        from .detect_frame import KP
        return (KP.get('cur'), KP.get('prev'))
    except Exception:
        return ()


def _replay():
    """-> 'off' | 'busy' (switching between camera and recording) | 'on'."""
    live = _app('LIVE')
    if live is None: return 'off'
    if getattr(live, 'replay_busy', False): return 'busy'
    return 'on' if getattr(live, 'replay', None) is not None else 'off'


def _replay_pos():
    try:
        from ..camera import replay_eval
        ev = replay_eval.EV
        return ev.position().get('pos') if ev is not None else None
    except Exception:
        return None


def _e2e_on():
    try:
        from ..track import e2e_live
        return bool(e2e_live.skip_detector())
    except Exception:
        return False


def _model_busy():
    md = _app('MODEL')
    if not isinstance(md, dict): return True
    if md.get('req') is not None or md.get('busy'): return True
    feed = _app('FEED')
    return feed is not None and getattr(feed, 'pending', None) is not None


def on_frame(t, rows, repeat=False, frame=None, stab=None, kp_sets=None):
    """Frame thread, once per published frame (page_handler.live_publish), after the tracker. Never raises.
    rows = the rows drawn and sent to the gimbal (their ids are the ones the gimbal follows). t = arrival time of the frame (s).
    stab = stabilizer correction of the frame shown (None = none). kp_sets: tests only (default: detect_frame.KP)."""
    try:
        with AUTO.lock: return _on_frame(t, rows, repeat, frame, stab, kp_sets)
    except Exception as e_:
        AUTO.err = repr(e_)[:160]
        return None


def _signal(o, s):
    r_ = lambda v, k=1: round(v, k) if isinstance(v, float) else v
    return dict(px=r_(s['px']), now=r_(o['px']), source=o['src'], id=o['id'], n=o['n'], kpts=o['kpts'], occl=bool(s['occl'] or s['cut']),
                hidden=bool(s['occl']), cut=bool(s['cut']), speed=r_(s['speed'], 2), score=o['score'])


def _on_frame(t, rows, repeat, frame, stab, kp_sets):
    A = AUTO
    if not A.on:   # every path that turns it off sets mode 'off' and its reason (by hand, 3 failures): keep that reason for /rules and the page
        if A.mode != 'off': A.mode = 'off'; A.reason = '꺼짐'
        return None
    near, far, can = _entries()
    det = _app('args').det
    if A.seen_det is None: A.seen_det = det
    elif det != A.seen_det:   # the frame loop changed the detector (this module's request or a choice by hand): dwell from here
        A.ch.switched(t, count=A.pending is None or A.pending['det'] != det); A.seen_det = det   # this module's own request was counted when asked
        if A.pending is not None and A.pending['det'] == det: A.failures = 0
        A.pending = None
    if A.pending is not None and not _model_busy():   # the frame loop took the request and the detector is still the old one: it failed
        last = (_app('MODEL') or {}).get('last') or {}
        A.failures += 1; A.retry_t = t + FAIL_WAIT_S
        print('AUTO_DET switch to %s (%s) failed: result %s, %s' % (A.pending['to'], A.pending['det'], last.get('result', '?'),
              'automatic choice OFF after %d failures in a row' % A.failures if A.failures >= FAIL_OFF else 'no new try for %d s' % FAIL_WAIT_S), flush=True)
        A.pending = None
        if A.failures >= FAIL_OFF:
            A.on = False; A.mode = 'off'; A.reason = '모델을 %d번 바꾸지 못해 꺼짐' % A.failures
            sel_ = _app('sel_save')
            if sel_ is not None: sel_(auto_det=False)
            return None
    if not can:
        A.mode = 'waiting'; A.reason = '이 실행에서는 검출 모델을 바꿀 수 없음'; return None
    if far is None or near is None:
        A.mode = 'waiting'; A.reason = ('먼 거리 모델 없음' if far is None else '근거리 모델 없음') + ' · 모델 폴더에 들어오면 시작'
        A.ch.clear(); return None
    if _e2e_on():
        A.mode = 'paused'; A.reason = '엔드투엔드 사용 중 · 검출 모델을 쓰지 않음'; A.ch.clear(); return None
    rp = _replay()
    if rp == 'busy':
        A.mode = 'paused'; A.reason = '녹화 / 카메라 전환 중'; return None
    wh = (float(_app('LIVE_W', 640) or 640), float(_app('LIVE_H', 480) or 480))
    o = observe(rows, _followed() if rp == 'off' else 0, A.params, stab, _kp_sets() if kp_sets is None else kp_sets, wh)
    if rp == 'on':
        if A.mode != 'dry_run':   # a replay started: its own chooser and clock, starting from the model in use
            A.dry_ch = Chooser(); A.dry_ch.p = A.params; A.dry_sm = Smooth(); A.dry_cur = _kind(det, near, far); A.dry_clock = 0.0; A.dry_last = None
            A.dry_events = []; A.dry_n = 0; A.mode = 'dry_run'
        if repeat: A.dry_last = None; return None   # the replay is paused: the same frame again, no time passes in the recording
        # replay clock: the time between two new frames, at most DRY_STEP_MAX (a pause, a seek or a restart does not count as time)
        A.dry_clock += min(max(t - A.dry_last, 0.0), DRY_STEP_MAX) if A.dry_last is not None else 0.0; A.dry_last = t
        s = A.dry_sm.add(A.dry_clock, o, A.params['median_s'])
        A.signal = _signal(o, s)
        k = A.dry_ch.step(A.dry_clock, s['px'], A.dry_cur, s['occl'] or s['cut'], (s['speed'] or 0.0) > A.params['fast_hps'])
        A.reason = '녹화 재생: 바꾸지 않고 기록만 · ' + A.dry_ch.reason
        if k is None: return None
        ev = dict(frame=_replay_pos(), t=round(A.dry_clock, 2), at=time.strftime('%H:%M:%S'), signal=A.signal['px'], source=o['src'], id=o['id'],
                  sig=dict(A.signal), lost=A.dry_ch.forced, **{'from': A.dry_cur, 'to': k, 'why': A.dry_ch.reason})
        A.dry_events.append(ev); del A.dry_events[:-EVENTS_KEEP]; A.dry_n += 1
        A.dry_cur = k; A.dry_ch.switched(A.dry_clock)
        print('AUTO_DET dry run (replay): would switch %s -> %s at recording frame %s, %.1f s (%s)' % (ev['from'], k, ev['frame'], ev['t'], _why_en(A.signal, A.params, A.dry_ch.forced)), flush=True)
        return None
    if A.mode == 'dry_run': A.ch.clear(); A.sm = Smooth()   # back on the camera: the replay's collected time does not count
    A.mode = 'live'
    s = A.sm.add(t, o, A.params['median_s'])
    A.signal = _signal(o, s)
    cur = _kind(det, near, far)
    k = A.ch.step(t, s['px'], cur, s['occl'] or s['cut'], (s['speed'] or 0.0) > A.params['fast_hps'])
    A.reason = A.ch.reason
    if o['src'] == 'followed_missing': A.reason = '따라가는 사람이 안 보임 · ' + A.reason
    if k is None: return None
    if A.pending is not None or _model_busy():
        A.reason += ' · 다른 모델 바꾸기가 끝나기를 기다림'; return None
    if A.retry_t is not None and t < A.retry_t:
        A.reason += ' · %.0f초 뒤 다시 시도' % (A.retry_t - t); return None
    tgt = far if k == 'far' else near
    r = APP.model_request(tgt['id'])   # the page's path: queued for the frame loop (between two frames)
    if r != -1:
        A.retry_t = t + REFUSED_WAIT_S
        A.reason += ' · 요청 안 됨 (%s)' % (_app('MODEL_TEXT') or {}).get(r, r); return None
    forced = A.ch.forced
    A.pending = dict(det=tgt['det'], to=k, t=t)
    A.ch.switched(t); A.switches += 1; A.retry_t = None
    A.last_switch = dict(at=time.strftime('%H:%M:%S'), t=t, signal=A.signal['px'], source=o['src'], id=o['id'], sig=dict(A.signal), lost=forced,
                         frame=frame, **{'from': cur, 'to': k, 'why': A.ch.reason})
    print('AUTO_DET %s -> %s (%s) at %s: queued model id %d %s' % (cur, k, _why_en(A.signal, A.params, forced), frame, tgt['id'], tgt['det']), flush=True)
    return k


def _why_en(sig, p, lost=False):   # the reason in the server log (plain ASCII)
    if sig['px'] is None or lost:
        return 'person lost or nobody seen (%s) -> far after %g s (%g s if last seen hidden or cut), no dwell / cap' % (sig['source'], p['lost_s'], p['lost_occl_s'])
    who = 'followed id %d' % sig['id'] if sig['source'] == 'followed' else 'median of the people in the picture'
    return '%s %.0f px over %g s (now %s), keypoints %s, hidden %d, cut %d, speed %s h/s; far below %g px, near above %g px, held %g s, dwell %g s' % (
        who, sig['px'], p['median_s'], sig['now'], sig['kpts'], sig['hidden'], sig['cut'], sig['speed'], p['far_px'], p['near_px'], p['hold_s'], p['dwell_s'])


def manual_choice(where):
    """A detector was chosen by hand (page /model/set, PC GUI): automatic choice off, so it does not undo that choice."""
    with AUTO.lock:
        if not AUTO.on: return False
        AUTO.on = False; AUTO.pending = None; AUTO.ch.clear(); AUTO.mode = 'off'; AUTO.reason = '손으로 모델을 골라 꺼짐 (%s)' % where
    print('AUTO_DET off: detector chosen by hand (%s)' % where, flush=True)
    sel_ = _app('sel_save')
    if sel_ is not None: sel_(auto_det=False)
    return True


def _num(v):
    return round(v, 2) if isinstance(v, float) else v


def state():
    """The /rules answer, key auto_det."""
    with AUTO.lock:
        A = AUTO
        near, far, can = _entries()
        args = _app('args'); det = getattr(args, 'det', None)
        now = time.perf_counter()
        ls = dict(A.last_switch) if A.last_switch else None
        if ls is not None: ls['ago_s'] = round(now - ls.pop('t'), 1)
        return dict(on=A.on, mode=A.mode, active=A.mode in ('live', 'dry_run'), reason=A.reason, can_switch=can,
                    near=dict(key=A.near_key, present=near is not None, id=near['id'] if near else None, det=near['det'] if near else None),
                    far=dict(key=A.far_key, present=far is not None, id=far['id'] if far else None, det=far['det'] if far else None),
                    current=_kind(det, near, far) if det is not None else None, current_det=det,
                    dry_choice=A.dry_cur if A.mode == 'dry_run' else None, signal=A.signal, pending=A.pending['to'] if A.pending else None,
                    last_switch=ls, switches=A.switches, failures=A.failures, dry_events=A.dry_n, dry_recent=[dict(e) for e in A.dry_events[-10:]],
                    params={k: _num(v) for k, v in A.params.items()}, defaults=dict(PARAMS), err=A.err)


def dry_log():
    """Every dry-run event kept (up to EVENTS_KEEP), for /rules/auto_det_log: the 'would-have switches' of a replay, to check each one."""
    with AUTO.lock:
        return dict(count=AUTO.dry_n, kept=len(AUTO.dry_events), events=[dict(e) for e in AUTO.dry_events], params={k: _num(v) for k, v in AUTO.params.items()})


def parse_request(q_):
    """q_: the auto_det* keys of a /rules/set query (lists, as parse_qs gives them). Everything is checked before anything changes.
    -> (ok, text, apply); apply() makes the change (called by the caller only when the rest of the query was taken too).
    auto_det=1|0, auto_det_<PARAMS name>=number, auto_det_near=<model key>, auto_det_far=<model key>."""
    on_ = None; nums = {}; keys = {}
    for k_, x_ in q_.items():
        v_ = str(x_[0] if isinstance(x_, (list, tuple)) and x_ else x_).strip()
        if k_ == 'auto_det':
            if v_.lower() not in ('1', '0', 'true', 'false', 'on', 'off'): return False, '검출 모델 자동 선택 값은 1 또는 0', None
            on_ = v_.lower() in ('1', 'true', 'on')
            continue
        name = k_[len('auto_det_'):] if k_.startswith('auto_det_') else ''
        if name in ('near', 'far'):
            if not KEY_RE.match(v_): return False, '모델 이름에 쓸 수 없는 글자: %s' % k_, None
            keys[name] = v_
        elif name in PARAMS:
            try: f_ = float(v_)
            except ValueError: return False, '숫자가 아님: %s' % k_, None
            lo_, hi_ = LIMITS[name]
            if not lo_ <= f_ <= hi_: return False, '%s 는 %g ~ %g' % (k_, lo_, hi_), None
            nums[name] = f_
        else:
            return False, '없는 자동 선택 값: %s' % k_, None
    merged = dict(AUTO.params, **nums)
    if not merged['far_px'] < merged['near_px']: return False, '먼 거리 기준(far_px)은 근거리 기준(near_px)보다 작아야 함', None
    if merged['kpt_seen'] > merged['kpt_min']: return False, 'kpt_seen 은 kpt_min 보다 클 수 없음', None
    nk_, fk_ = keys.get('near', AUTO.near_key), keys.get('far', AUTO.far_key)
    if nk_ == fk_: return False, '근거리와 먼 거리 모델이 같음', None
    if on_:
        _, _, can = _entries()
        if not can: return False, '이 실행에서는 검출 모델을 바꿀 수 없음', None

    def apply():
        with AUTO.lock:
            A = AUTO
            A.params.update(nums)
            if keys and (nk_, fk_) != (A.near_key, A.far_key): A.near_key, A.far_key = nk_, fk_; A.ch.clear()
            if on_ is not None and on_ != A.on:
                A.on = on_; A.ch.clear(); A.sm = Smooth(); A.pending = None; A.retry_t = None; A.seen_det = None
                A.mode = 'off' if not on_ else 'waiting'; A.reason = '꺼짐' if not on_ else '다음 프레임부터 판단'
            saved = {k: v for k, v in A.params.items() if v != PARAMS[k]}
            if A.near_key != NEAR_KEY: saved['near'] = A.near_key
            if A.far_key != FAR_KEY: saved['far'] = A.far_key
            on_now = A.on
            text_ = 'RULES set auto_det %d %s near %s far %s' % (A.on, ' '.join('%s=%g' % (k, v) for k, v in sorted(A.params.items())), A.near_key, A.far_key)
        print(text_, flush=True)
        sel_ = _app('sel_save')
        if sel_ is not None: sel_(auto_det=on_now, auto_det_p=saved)
    return True, '적용됨', apply


def restore_query(saved_on, saved_p):
    """The /rules/set query that brings a saved choice back at the next start (last_selection.json keys auto_det, auto_det_p).
    Names that are no longer settings (11-15 v1: empty_default_s went, nobody seen now means far) are left out."""
    q_ = {'auto_det': ['1' if saved_on else '0']}
    if isinstance(saved_p, dict):
        for k_, v_ in saved_p.items():
            if k_ in PARAMS or k_ in ('near', 'far'): q_['auto_det_' + k_] = [str(v_)]
    return q_
