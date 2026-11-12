"""녹화 재생 중 정답과 비교 (2026-11-12). 검수한 라벨이 있는 녹화를 재생하면, 서버가 내보낸 사람 상자를 프레임마다 정답과 비교해 센다.

어느 PNG 인지 아는 법
  - 재생 카메라(replay_cam.py --tag)는 프레임마다 48 바이트 머리(카메라 시간표와 같은 모양)를 붙이고, 머리의 flags 2 번 비트(값 2)를 켜고
    ring_id 에 PNG 번호(파일 이름의 숫자, 000123.png = 123, 아래 20 비트)와 바퀴 번호(위 12 비트, 0 부터), slot 에 녹화 안의 프레임 번호(0 부터)를 적는다.
    멈춘 동안 같은 프레임을 되풀이해 보낼 때는 flags 에 값 4 도 켠다: 되풀이는 채점하지 않고 센 수만 남긴다 (화면 그림은 그대로 갱신된다).
  - 서버는 재생 카메라를 늘 머리를 붙이게 띄운다 (replay_link.spawn). 머리를 읽는 곳(camera_extra._reader)이 프레임을 줄에 넣기 직전에
    tag() 를 불러 "도착 시각 -> (PNG 번호, 바퀴, 칸 번호)" 를 적는다. 화면을 만드는 곳(live_publish)이 그 프레임의 도착 시각으로 찾는다.
    그래서 서버가 중간 프레임을 버리거나 재생 카메라가 건너뛰어도 짝이 어긋나지 않는다. 번호를 모르는 프레임은 채점하지 않는다.
  - 재생 카메라가 건너뛴 프레임 수 = 같은 바퀴 안에서 프레임 번호가 1 보다 크게 뛴 만큼. 서버가 건너뛴 프레임 수 = 받은 프레임 - 화면에 낸 프레임.
채점할 프레임: tracks_edited_mot.txt 에 줄이 있고 review_state.json 의 reviewed 에 있으며 removed·purged 에 없는 프레임.
  review_state.json 이 없으면 라벨 파일의 모든 프레임 ("검수 상태 없음"). 읽을 수 없으면 채점하지 않는다. 자동 라벨(tracks_mot.txt)은 쓰지 않는다.
반복 재생은 바퀴마다 계속 채점하고, 정답 번호와 서버 번호를 모두 (바퀴, 번호) 로 센다: 바퀴 하나 = 영상 하나로 보고 더한다.
  (바퀴가 넘어갈 때 영상은 끝에서 처음으로 건너뛰지만 추적기는 그대로 이어 간다. 둘째 바퀴부터는 앞 바퀴 끝의 궤적을 가진 채 시작한다.)
건너뛰기(SEEK, 11-12): 건너뛸 때마다 새 바퀴가 시작된다. 이미 센 프레임은 지우지 않고, 되감아서 같은 프레임을 다시 지나면 새 바퀴의 프레임으로 한 번 더 센다
  (숫자에 그 장면이 두 번 들어간다. 서로 다른 장면 수 scored_unique 를 함께 보여 주고, 화면이 그렇게 적는다). 처음부터 다시 세려면 "숫자 처음부터".
  건너뛴 직후 추적기는 서버의 "입력이 1 초 넘게 끊기면 추적을 새로 시작" 규칙으로 비워진다 (재생 카메라가 건너뛴 뒤 1.2 초 쉰다).
점수 계산과 규칙: src/track/gt_score.py. 재생이 끝나거나 바뀌면(서버가 끝날 때도) 결과를 <결과 폴더>/replay_eval_<이름>_<시각>.json 에 쓴다.
재생을 쓰지 않으면 EV 가 None 이고, 서버의 다른 길에서 부르는 함수는 모두 바로 돌아간다.
"""
import collections
import hashlib
import json
import os
import threading
import time

from ..common.state import APP
from ..track import gt_score
from ..track.stabilizer import stab_people_real

LABEL = 'tracks_edited_mot.txt'
AUTO = 'tracks_mot.txt'
REVIEW = 'review_state.json'
REP_KEEP = 512           # 되풀이 프레임 표의 크기
TAG_KEEP = 4096          # 도착 시각 -> PNG 번호 표의 크기 (화면에 나가지 않은 프레임의 줄은 오래되면 빠진다)
LOG_MAX = 200000         # 결과 파일에 남길 프레임 목록의 최대 길이

_LK = threading.Lock()
_RL = threading.Lock()
_RT = collections.OrderedDict()   # 되풀이 프레임의 도착 시각 (재생이 끝난 뒤에 나오는 것도 알아보려고 재생과 따로 둔다)
EV = None                # 지금 재생의 채점 (None = 재생 안 함)
LAST = None              # 끝난 재생의 마지막 결과 (화면에 보여 줌)
DRAW = [False]           # 화면 그림에 정답 상자 그리기 (모든 보는 사람이 같은 그림)


def load_labels(rec_dir, name):
    """녹화 폴더 -> dict(ok, why, gt={png: [(번호, x1, y1, x2, y2, 클래스)]}, frames=채점할 PNG 번호 집합, review=글, label_file, ...)"""
    out = dict(ok=False, why='', gt={}, frames=set(), review='', label_file=None, label_sha256=None, bad_lines=0, labelled=0, reviewed=None, removed=0)
    lp = os.path.join(rec_dir, LABEL)
    if not os.path.isfile(lp):
        out['why'] = '정답 라벨(%s)이 없어 채점하지 않음%s' % (LABEL, ' (자동 라벨 %s 는 검수 전이라 쓰지 않음)' % AUTO if os.path.isfile(os.path.join(rec_dir, AUTO)) else '')
        return out
    try:
        raw = open(lp, 'rb').read()
    except OSError as e:
        out['why'] = '정답 라벨을 읽지 못함 (%s)' % type(e).__name__; return out
    out['label_file'] = name + '/' + LABEL; out['label_sha256'] = hashlib.sha256(raw).hexdigest()
    gt = collections.defaultdict(list)
    for ln in raw.decode('utf-8', 'replace').splitlines():
        f = ln.strip().split(',')
        if len(f) < 8 or not ln.strip():
            if ln.strip(): out['bad_lines'] += 1
            continue
        try:
            fr, gid, x, y, w, h, cl = int(float(f[0])), int(float(f[1])), float(f[2]), float(f[3]), float(f[4]), float(f[5]), int(float(f[7]))
        except ValueError:
            out['bad_lines'] += 1; continue
        if fr < 1 or w <= 0 or h <= 0: out['bad_lines'] += 1; continue
        gt[fr].append((gid, x, y, x + w, y + h, cl))
    out['gt'] = dict(gt); out['labelled'] = len(gt)
    rp = os.path.join(rec_dir, REVIEW)
    if not os.path.isfile(rp):
        out['frames'] = set(gt); out['review'] = '검수 상태 없음 (review_state.json 이 없어 라벨이 있는 모든 프레임을 채점)'
    else:
        try:
            st = json.loads(open(rp, 'rb').read().decode('utf-8'))
            rv = {int(v) for v in st.get('reviewed', [])}; rm = {int(v) for v in st.get('removed', [])} | {int(v) for v in st.get('purged', [])}
        except (OSError, ValueError, TypeError, AttributeError) as e:
            out['why'] = '검수 상태(review_state.json)를 읽지 못해 채점하지 않음 (%s)' % type(e).__name__; return out
        out['frames'] = (set(gt) & rv) - rm; out['reviewed'] = len(rv); out['removed'] = len(rm)
        out['review'] = '검수한 프레임 %d 장 가운데 라벨이 있는 %d 장을 채점' % (len(rv - rm), len(out['frames']))
    if not out['frames']: out['why'] = '채점할 프레임이 없음 (라벨이 있고 검수한 프레임이 없음)'; return out
    out['ok'] = True
    return out


class Eval:
    def __init__(self, tgt):
        self.name = tgt['name']; self.fps = tgt['fps']; self.loop = tgt['loop']
        self.lab = load_labels(os.path.dirname(tgt['dir']), tgt['name'])
        self.lock = threading.Lock(); self.tags = collections.OrderedDict(); self.rep = collections.OrderedDict()
        self.src_t = None; self.started = time.strftime('%Y-%m-%d %H:%M:%S'); self.ended = None; self.end_reason = None; self.file = None
        self.last_k = None; self.last_pass = None; self.passes = 0; self.draw_t = None; self.draw_gt = None; self.resets = []
        self.pos_in = None; self.pos_shown = None; self.pass_shown = None; self.pos_t = None; self.seek_to = None; self.seek_t = None; self.seeks = 0
        self.zero()

    def zero(self):
        self.new = gt_score.Score(True); self.old = gt_score.Score(False)
        self.new_h = gt_score.Score(True, False); self.old_h = gt_score.Score(False, False)   # plain per-frame Hungarian (no carry-forward), for the file only
        self.n = dict(received=0, cam_skipped=0, published=0, untagged=0, not_labelled=0, not_reviewed=0, scored=0, tag_evicted=0, repeats=0)
        self.uniq = set()
        self.log = []; self.since = time.strftime('%Y-%m-%d %H:%M:%S'); self.ver = 0; self._sum = None

    def tag(self, ta, rid, slot, flags=2):   # 재생 카메라 읽기 스레드
        png, pss = rid & 0xFFFFF, (rid >> 20) + 1
        with self.lock:
            self.pos_in = slot
            if flags & 4:   # 멈춘 동안의 되풀이: 센 수만 남기고 채점 표에는 넣지 않는다
                self.n['repeats'] += 1; self._sum = None; self.rep[ta] = (png, pss, slot)
                with _RL:
                    _RT[ta] = 1
                    if len(_RT) > REP_KEEP: _RT.popitem(last=False)
                if len(self.rep) > REP_KEEP: self.rep.popitem(last=False)
                return
            self.n['received'] += 1
            if self.last_pass == pss and self.last_k is not None and slot > self.last_k + 1: self.n['cam_skipped'] += slot - self.last_k - 1
            self.last_k = slot; self.last_pass = pss; self.passes = max(self.passes, pss)
            self.tags[ta] = (png, pss, slot)
            if len(self.tags) > TAG_KEEP: self.tags.popitem(last=False); self.n['tag_evicted'] += 1

    def _shown(self, t_in, slot, pss):
        self.pos_shown = slot; self.pass_shown = pss; self.pos_t = t_in
        if self.seek_to is not None and slot == self.seek_to and t_in >= self.seek_t: self.seek_to = None

    def want_seek(self, pos, t):   # 서버가 SEEK 를 보냈다: 이 번호의 프레임이 화면에 나올 때까지 "건너뛰는 중"
        with self.lock: self.seek_to = pos; self.seek_t = t; self.seeks += 1

    def position(self):
        with self.lock: return dict(pos=self.pos_shown, pass_=self.pass_shown, seek_to=self.seek_to, seeks=self.seeks)

    def frame(self, t_in, rows, sc, image_id):   # 주 스레드 (live_publish): 화면에 낸 프레임 하나
        with self.lock:
            if self.src_t is not None and t_in < self.src_t: return None   # 이 재생보다 먼저 도착한 프레임 (옛 영상)
            rp = self.rep.pop(t_in, None)
            if rp is not None:   # 되풀이: 위치만 갱신하고, 채점도 '화면에 낸 프레임' 수도 늘리지 않는다 (정답 상자는 그림에 그리게 돌려준다)
                self._shown(t_in, rp[2], rp[1])
                return self.lab['gt'].get(rp[0]) if self.lab['ok'] and rp[0] in self.lab['frames'] else None
            tg = self.tags.pop(t_in, None)
            self.n['published'] += 1
            if tg is None: self.n['untagged'] += 1; return None
            png, pss, k = tg
            self._shown(t_in, k, pss)
            if not self.lab['ok']: return None
            if png not in self.lab['frames']:
                self.n['not_reviewed' if png in self.lab['gt'] else 'not_labelled'] += 1; return None
            ppl = [(int(r[4]), float(r[0]), float(r[1]) - 16.0, float(r[2]), float(r[3]) - 16.0, float(r[5])) for r in rows if int(r[6]) == 0]
            ppl = stab_people_real(ppl, sc, float(APP.LIVE_W), float(APP.LIVE_H)) if sc is not None else ppl   # 짐벌 줄과 같은 실제 카메라 좌표
            out = [(p[0], p[1], p[2], p[3], p[4]) for p in ppl]
            gt = self.lab['gt'][png]
            fn = self.new.frame(gt, out, pss); fo = self.old.frame(gt, out, pss)
            self.new_h.frame(gt, out, pss); self.old_h.frame(gt, out, pss)
            self.n['scored'] += 1; self.uniq.add(png); self.ver += 1; self._sum = None
            if len(self.log) < LOG_MAX: self.log.append([image_id, png, pss, fn['found'], fn['missed'], fn['false'], fo['found'], fo['missed'], fo['false'], k])
            self.draw_t = t_in; self.draw_gt = gt
            return gt

    def reset(self):
        with self.lock:
            self.resets.append(dict(at=time.strftime('%Y-%m-%d %H:%M:%S'), before=self._summary_locked(False)))
            self.zero()

    def _summary_locked(self, with_rules=True):
        if self._sum is not None and with_rules: return self._sum
        lab = self.lab
        s = dict(name=self.name, fps=self.fps, loop=self.loop, started=self.started, since=self.since, ended=self.ended, end_reason=self.end_reason, file=self.file,
                 scoring=lab['ok'], why=lab['why'] or None, review=lab['review'] or None, label_file=lab['label_file'], label_sha256=lab['label_sha256'],
                 label_bad_lines=lab['bad_lines'], labelled_frames=lab['labelled'], scorable_frames=len(lab['frames']), passes=self.passes, scored_unique=len(self.uniq), seeks=self.seeks,
                 counts=dict(self.n, server_skipped=max(0, self.n['received'] - self.n['published'] - len(self.tags))),
                 new=self.new.result() if lab['ok'] else None, old=self.old.result() if lab['ok'] else None, draw=DRAW[0],
                 matching='CLEAR MOT (Bernardin & Stiefelhagen 2008, as py-motmetrics): last frame pairs kept while IoU >= 0.5, Hungarian for the rest',
                 per_frame_hungarian=dict(note='same frames, plain Hungarian every frame without keeping last frame pairs; not shown on the page',
                                          new=self.new_h.result(), old=self.old_h.result()) if lab['ok'] else None)
        if with_rules: self._sum = s
        return s

    def summary(self):
        with self.lock: return self._summary_locked()

    def close(self, reason):   # 재생이 끝났다: 끝난 때와 이유를 적는다 (라벨이 없어 채점하지 않은 녹화도). 파일은 write() 가 쓴다
        with self.lock:
            if self.ended is None: self.ended = time.strftime('%Y-%m-%d %H:%M:%S'); self.end_reason = reason; self._sum = None

    def write(self, reason):
        self.close(reason)
        if not self.lab['ok']: return None
        with self.lock:
            safe = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in self.name)[:60].strip('_') or 'replay'
            p = APP.out / ('replay_eval_%s_%s.json' % (safe, time.strftime('%Y%m%d_%H%M%S')))
            d = dict(self._summary_locked(False), rules=gt_score.RULES_KO,
                     thresholds=dict(iou=gt_score.IOU_MIN, ioa=gt_score.IOA_MIN, edge_px=gt_score.EDGE_PX, edge_width=gt_score.EDGE_W, small_height=gt_score.SMALL_H),
                     references=['K. Bernardin, R. Stiefelhagen. Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. EURASIP JIVP 2008 (MOTA, ID switches)',
                                 'E. Ristani, F. Solera, R. Zou, R. Cucchiara, C. Tomasi. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCV Workshops 2016 (IDF1)'],
                     counts_note='received = frames from the replay camera with a PNG number; cam_skipped = frames the replay camera skipped (receiver slow); '
                                 'published = frames shown since the start or the last reset; untagged = shown without a known PNG number (not scored); '
                                 'not_reviewed / not_labelled = PNG without a reviewed label (not scored); scored = frames in the numbers; server_skipped = received but not shown; '
                                 'repeats = frames repeated while the replay was paused (not counted anywhere else); scored_unique = different PNGs among the scored frames (a seek back scores a scene again as a new pass)',
                     resets=self.resets, frames_cols=['image_id', 'png', 'pass', 'new_found', 'new_missed', 'new_false', 'old_found', 'old_missed', 'old_false', 'pos'], frames=self.log)
            try:
                p.write_text(json.dumps(d, ensure_ascii=False), encoding='utf-8'); self.file = str(p.name); self._sum = None
                print('REPLAY EVAL written %s: frames scored %d, new false %d missed %d, old false %d missed %d' % (p.name, self.n['scored'], d['new']['false'], d['new']['missed'], d['old']['false'], d['old']['missed']), flush=True)
            except OSError as e:
                print('REPLAY EVAL write failed', type(e).__name__, flush=True)
            return self.file


# ------------------------------------------------------------------ 서버의 다른 곳에서 부르는 것
def prepare(tgt):   # 재생 바꾸기 스레드: 라벨을 미리 읽는다 (잠금 밖)
    try: return Eval(tgt)
    except Exception as e:
        print('REPLAY EVAL prepare failed', type(e).__name__, flush=True); return None


def _write(old, reason):
    try: old.write(reason)
    except Exception as e: print('REPLAY EVAL write failed', type(e).__name__, flush=True)


def activate(ev, src_t, reason, sync=False):   # 재생 바꾸기 (cam_lock 안): 새 채점을 걸고, 옛 채점은 끝내 파일로 쓴다 (따로 도는 스레드에서, 서버가 끝날 때는 바로)
    global EV, LAST
    with _LK:
        old, EV = EV, ev
        if ev is not None: ev.src_t = src_t
    if old is not None:
        old.close(reason); LAST = old
        if sync: _write(old, reason)
        else: threading.Thread(target=_write, args=(old, reason)).start()   # daemon 이 아님: 서버가 끝나도 다 쓴 뒤 끝난다


def tag(p, ta, rid, slot, flags=2):   # camera_extra._reader: 재생 카메라의 프레임 하나 (flags 2, 되풀이는 4 도)
    ev = getattr(p, 'lwir_eval', None)
    if ev is not None: ev.tag(ta, rid, slot, flags)


def on_frame(t_in, rows, sc, image_id):   # live_publish: 재생 중이 아니면 바로 돌아간다
    ev = EV
    if ev is None: return None
    try: return ev.frame(t_in, rows, sc, image_id)
    except Exception as e:
        print('REPLAY EVAL frame failed', type(e).__name__, str(e)[:120], flush=True); return None


def draw_boxes(t_in, gt, sc):   # 화면 그림에만 그릴 정답 상자: [(x1, y1, x2, y2, 종류)] 종류 0 = 사람, 1 = 무시 구역(클래스 1 또는 가장자리 규칙)
    if not DRAW[0] or gt is None: return []
    dx, dy = (sc[0], sc[1]) if sc is not None else (0.0, 0.0)   # 실제 좌표 -> 흔들림 보정한 그림 좌표
    return [(g[1] + dx, g[2] + dy, g[3] + dx, g[4] + dy, 1 if (int(g[5]) == 1 or gt_score.edge_sliver(g[1:5])) else 0) for g in gt if int(g[5]) in (0, 1)]


def state():
    ev = EV
    if ev is not None: return dict(ev.summary(), on=True)
    if LAST is not None: return dict(LAST.summary(), on=False)
    return dict(on=False, scoring=False, why=None, draw=DRAW[0])


def is_repeat(t_in):   # live_publish: 이 프레임은 재생 프로그램이 되풀이로 보낸 것인가 (멈춘 동안, 끝날 때). 멈춘 그림 규칙이 로그를 쏟지 않게. 재생을 안 썼으면 바로 False
    if not _RT: return False
    with _RL: return t_in in _RT


def last():   # 끝난 재생의 마지막 숫자 (서버가 다시 켜지면 없어진다). 없으면 None
    e = LAST
    return dict(e.summary(), on=False) if e is not None else None


def reset_request():
    ev = EV
    if ev is None: return 409, '재생 중이 아님'
    ev.reset(); return 200, '숫자를 처음부터 다시 셈'


def draw_request(q):
    v = q.get('on', [''])[0]
    if v not in ('0', '1'): return 400, 'on 은 0 또는 1'
    DRAW[0] = v == '1'
    for e in (EV, LAST):
        if e is not None:
            with e.lock: e._sum = None
    return 200, '정답 상자 그리기 %s' % ('켬' if DRAW[0] else '끔')


def finish(reason):   # 서버가 끝날 때 (재생 중이었으면 결과 파일을 쓴다)
    if EV is not None: activate(None, None, reason, sync=True)
