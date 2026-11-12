"""녹화 재생 중 정답과 비교 (2026-11-12). 검수한 라벨이 있는 녹화를 재생하면, 서버가 내보낸 사람 상자를 프레임마다 정답과 비교해 센다.

어느 PNG 인지 아는 법
  - 재생 카메라(replay_cam.py --tag)는 프레임마다 48 바이트 머리(카메라 시간표와 같은 모양)를 붙이고, 머리의 flags 2 번 비트(값 2)를 켜고
    ring_id 에 PNG 번호(파일 이름의 숫자, 000123.png = 123, 아래 20 비트)와 바퀴 번호(위 12 비트, 0 부터), slot 에 녹화 안의 프레임 번호(0 부터)를 적는다.
    멈춘 동안 같은 프레임을 되풀이해 보낼 때는 flags 에 값 4 도 켠다: 되풀이는 채점하지 않고 센 수만 남긴다 (화면 그림은 그대로 갱신된다).
  - 서버는 재생 카메라를 늘 머리를 붙이게 띄운다 (replay_link.spawn). 머리를 읽는 곳(camera_extra._reader)이 프레임을 줄에 넣기 직전에
    tag() 를 불러 "도착 시각 -> (PNG 번호, 바퀴, 칸 번호)" 를 적는다. 화면을 만드는 곳(live_publish)이 그 프레임의 도착 시각으로 찾는다.
    그래서 서버가 중간 프레임을 버리거나 재생 카메라가 건너뛰어도 짝이 어긋나지 않는다. 번호를 모르는 프레임은 채점하지 않는다.
  - 재생 카메라가 건너뛴 프레임 수 = 같은 바퀴 안에서 프레임 번호가 1 보다 크게 뛴 만큼. 서버가 건너뛴 프레임 수 = 받은 프레임 - 화면에 낸 프레임.
채점은 첫 회차의 처음부터 끝까지 순서대로 처리한 프레임을 기준으로 한다. 반복·되감기·정지는 점수를 다시 쌓지 않는다.
  검수됨으로 표시된 프레임에는 빈 장면도 포함한다. 삭제된 프레임은 제외한다. 검수 기록이 없다면 주석이 있는 프레임만 평가한다.
  건너뛰거나 프레임이 누락되면 해당 실행은 최종 점수로 확정하지 않는다. HOTA와 사람 AP는 한 회차를 완료하면 별도 스레드에서 계산한다.
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
from ..track import gt_score, replay_metrics
from ..track.stabilizer import stab_off, stab_people_real
from ..track.bytetrack import NativeByteTrack

LABEL = 'tracks_edited_mot.txt'
AUTO = 'tracks_mot.txt'
REVIEW = 'review_state.json'
REP_KEEP = 512           # 되풀이 프레임 표의 크기
TAG_KEEP = 4096          # 도착 시각 -> PNG 번호 표의 크기 (화면에 나가지 않은 프레임의 줄은 오래되면 빠진다)
PREFIX_CACHE_MAX = 8
LOG_MAX = 200000         # 결과 파일에 남길 프레임 목록의 최대 길이

_LK = threading.Lock()
_RL = threading.Lock()
_RT = collections.OrderedDict()   # 되풀이 프레임의 도착 시각 (재생이 끝난 뒤에 나오는 것도 알아보려고 재생과 따로 둔다)
EV = None                # 지금 재생의 채점 (None = 재생 안 함)
FINAL = collections.OrderedDict()   # last completed summary per recording and backend
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
        with open(lp, 'rb') as labels:
            raw = labels.read()
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
            with open(rp, encoding='utf-8') as review:
                st = json.load(review)
            rv = {int(v) for v in st.get('reviewed', [])}; rm = {int(v) for v in st.get('removed', [])} | {int(v) for v in st.get('purged', [])}
        except (OSError, ValueError, TypeError, AttributeError) as e:
            out['why'] = '검수 상태(review_state.json)를 읽지 못해 채점하지 않음 (%s)' % type(e).__name__; return out
        out['frames'] = rv - rm; out['reviewed'] = len(rv); out['removed'] = len(rm)
        out['review'] = '검수한 프레임을 채점 (사람이 없는 검수 프레임 포함)'
    if not out['frames']: out['why'] = '채점할 프레임이 없음'; return out
    out['ok'] = True
    return out


class Eval:
    def __init__(self, tgt):
        self.name = tgt['name']; self.fps = tgt['fps']; self.loop = tgt['loop']
        self.lab = load_labels(os.path.dirname(tgt['dir']), tgt['name'])
        if self.lab['ok'] and os.path.isdir(tgt['dir']):
            pngs = {int(stem) for name in os.listdir(tgt['dir'])
                    for stem, ext in [os.path.splitext(name)] if ext.lower() == '.png' and stem.isascii() and stem.isdigit()}
            self.lab['frames'].intersection_update(pngs)
            if not self.lab['frames']:
                self.lab.update(ok=False, why='검수한 프레임의 PNG가 없음')
        self.sequence_frames = int(tgt.get('frames') or 0)
        self.phase = 'scoring'
        self.pass_frames = 0
        self.gap = None
        self.metric_status = 'unavailable'
        self.metrics = dict(hota=None, ap50=None, map=None)
        self.history = []  # one entry per accepted frame: (PNG, tracking boxes, metric input), or None if unreviewed
        self.prefix_cache = collections.OrderedDict()
        self.prefix_request = None
        self.prefix_busy = None
        self.prefix_worker = None
        self.run_id = str(time.time_ns())
        self.metric_worker = None
        self.id_switches = []
        self.lock = threading.Lock(); self.tags = collections.OrderedDict(); self.rep = collections.OrderedDict()
        self.src_t = None; self.started = time.strftime('%Y-%m-%d %H:%M:%S'); self.ended = None; self.end_reason = None; self.file = None
        self.last_k = None; self.last_pass = None; self.passes = 0; self.draw_t = None; self.draw_gt = None
        self.pos_in = None; self.pos_shown = None; self.pass_shown = None; self.pos_t = None; self.seek_to = None; self.seek_t = None; self.seeks = 0
        self.zero()
        self.backend = tgt.get('backend', 'python')
        self.preview = tgt.get('preview', 'sync')
        self.backend_error = None
        self.native_tracker = NativeByteTrack(tgt['tracker_library']) if self.backend == 'cpp_bytetrack' else None
        self.native_tag = None
        self.native_rows = []
        self.native_frames = 0

    def track_frame(self, t_in, detections):
        if self.backend != 'cpp_bytetrack':
            return None
        with self.lock:
            tag = self.tags.get(t_in)
            repeat = self.rep.get(t_in)
            if self.ended is not None or (tag is None and repeat is None):
                return None
            if repeat is not None:
                return self.native_rows if repeat == self.native_tag else []
            if self.backend_error is not None:
                return []
            try:
                if self.native_tag is not None and (tag[1] != self.native_tag[1] or tag[2] <= self.native_tag[2]):
                    self.native_tracker.reset()
                self.native_rows = self.native_tracker.update(detections)
                self.native_tag = tag
                self.native_frames += 1
                self._sum = None
                return self.native_rows
            except (OSError, RuntimeError, ValueError) as error:
                self.native_tracker.close()
                self.backend_error = str(error)
                self.native_rows = []
                if self.phase == 'scoring':
                    self.phase = 'incomplete'
                self._sum = None
                print('REPLAY C++ ByteTrack failed:', self.backend_error, flush=True)
                return []

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
        with self.lock:
            self.seek_to = pos; self.seek_t = t; self.seeks += 1
            if self.phase == 'scoring':
                self.phase = 'interrupted'
            self._sum = None
            return self.last_pass or 1

    def position(self):
        with self.lock: return dict(pos=self.pos_shown, pass_=self.pass_shown, seek_to=self.seek_to, seeks=self.seeks)

    def first_pass_position(self, t_in):
        with self.lock:
            tag = self.tags.get(t_in)
            return tag[2] if self.phase == 'scoring' and tag is not None and tag[1] == 1 else None

    def _compute(self, frames):
        try:
            metrics = replay_metrics.compute(frames)
            status = 'ready'
        except Exception as exc:
            print('REPLAY METRICS failed', type(exc).__name__, str(exc)[:120], flush=True)
            metrics = dict(hota=None, ap50=None, map=None)
            status = 'error'
        with self.lock:
            self.metrics = metrics
            self.metric_status = status
            self._sum = None
            snapshot = dict(self._summary_locked(), on=False)
        with _LK:
            key = (self.name, self.backend)
            previous = FINAL.get(key)
            if previous is None or previous['run_id'] < self.run_id:
                FINAL[key] = snapshot
                FINAL.move_to_end(key)
                if len(FINAL) > PREFIX_CACHE_MAX:
                    FINAL.popitem(last=False)

    def _complete(self):
        self.phase = 'complete'
        frames = [entry[2] for entry in self.history if entry is not None]
        if not frames:
            return
        self.metric_status = 'pending'
        self.metric_worker = threading.Thread(target=self._compute, args=(frames,), name='replay-metrics')
        self.metric_worker.start()

    def _metric_input(self, gt, tracks, kept, detections, sc):
        people, ignore = gt_score.split_gt(gt, True)
        targets = [(int(gid), *box) for gid, box in people]
        kept_tracks = [tracks[j] for j in kept]
        if detections is None:
            return targets, kept_tracks, None
        det = [(0, float(r[0]), float(r[1]) - 16, float(r[2]), float(r[3]) - 16, float(r[4]))
               for r in detections if int(r[5]) == 0]
        if sc is not None:
            det = stab_people_real(det, sc, float(APP.LIVE_W), float(APP.LIVE_H))
        valid_overlap = gt_score.iou_matrix([b for _, b in people], [d[1:5] for d in det])
        det_out = [(d[5], *d[1:5]) for j, d in enumerate(det)
                   if (valid_overlap[:, j] >= gt_score.IOU_MIN).any()
                   or not any(gt_score.ioa(d[1:5], b) >= gt_score.IOA_MIN for b in ignore)]
        return targets, kept_tracks, det_out

    def frame(self, t_in, rows, sc, image_id, detections=None, *, preview_receipt=None):
        with self.lock:
            if self.src_t is not None and t_in < self.src_t: return None   # 이 재생보다 먼저 도착한 프레임 (옛 영상)
            rp = self.rep.pop(t_in, None)
            if rp is not None:   # 되풀이: 위치만 갱신하고, 채점도 '화면에 낸 프레임' 수도 늘리지 않는다 (정답 상자는 그림에 그리게 돌려준다)
                self._shown(t_in, rp[2], rp[1])
                if preview_receipt is not None:
                    preview_receipt.append((self.run_id, rp[2], rp[1], t_in))
                return self.lab['gt'].get(rp[0], ()) if self.lab['ok'] and rp[0] in self.lab['frames'] else None
            tg = self.tags.pop(t_in, None)
            self.n['published'] += 1
            if tg is None: self.n['untagged'] += 1; return None
            png, pss, k = tg
            self._shown(t_in, k, pss)
            if preview_receipt is not None:
                preview_receipt.append((self.run_id, k, pss, t_in))
            gt = self.lab['gt'].get(png, ()) if self.lab['ok'] and png in self.lab['frames'] else None
            if self.phase != 'scoring':
                return gt
            if pss == 1 and k < self.pass_frames:
                return gt
            if pss != 1 or k != self.pass_frames or not self.sequence_frames or (self.lab['ok'] and not png):
                self.phase = 'incomplete'
                self.gap = dict(expected_slot=self.pass_frames, got_slot=k, pass_no=pss, png=png)
                print('REPLAY EVAL incomplete:', self.gap, flush=True)
                self._sum = None
                return gt
            if gt is None:
                self.n['not_reviewed' if png in self.lab['gt'] else 'not_labelled'] += 1
                self.history.append(None)
                self.pass_frames += 1
                if self.pass_frames == self.sequence_frames:
                    self._complete()
                self._sum = None
                return None
            ppl = [(int(r[4]), float(r[0]), float(r[1]) - 16.0, float(r[2]), float(r[3]) - 16.0, float(r[5])) for r in rows if int(r[6]) == 0]
            ppl = stab_people_real(ppl, sc, float(APP.LIVE_W), float(APP.LIVE_H)) if sc is not None else ppl   # 짐벌 줄과 같은 실제 카메라 좌표
            out = [(p[0], p[1], p[2], p[3], p[4]) for p in ppl]
            switches, kept = [], []
            fn = self.new.frame(gt, out, pss, switches=switches, kept=kept)
            fo = self.old.frame(gt, out, pss)
            self.new_h.frame(gt, out, pss); self.old_h.frame(gt, out, pss)
            self.id_switches.extend(dict(frame=k + 1, png=png, gt_id=g, from_id=old, to_id=new) for g, old, new in switches)
            self.history.append((png, out, self._metric_input(gt, out, kept, detections, sc)))
            self.n['scored'] += 1; self.uniq.add(png); self.ver += 1; self._sum = None
            self.pass_frames += 1
            if self.pass_frames == self.sequence_frames:
                self._complete()
            if len(self.log) < LOG_MAX: self.log.append([image_id, png, pss, fn['found'], fn['missed'], fn['false'], fo['found'], fo['missed'], fo['false'], k])
            self.draw_t = t_in; self.draw_gt = gt
            return gt

    def _summary_locked(self, with_rules=True):
        if self._sum is not None and with_rules: return self._sum
        lab = self.lab
        s = dict(run_id=self.run_id, name=self.name, fps=self.fps, loop=self.loop, started=self.started, since=self.since, ended=self.ended, end_reason=self.end_reason, file=self.file,
                 backend=self.backend, backend_error=self.backend_error, native_frames=self.native_frames,
                 preview=self.preview,
                 scoring=lab['ok'], why=lab['why'] or None, review=lab['review'] or None, label_file=lab['label_file'], label_sha256=lab['label_sha256'],
                 label_bad_lines=lab['bad_lines'], labelled_frames=lab['labelled'], scorable_frames=len(lab['frames']), passes=self.passes, scored_unique=len(self.uniq), seeks=self.seeks,
                 counts=dict(self.n, server_skipped=max(0, self.n['received'] - self.n['published'] - len(self.tags))),
                 phase=self.phase, sequence_frames=self.sequence_frames, pass_frames=self.pass_frames, first_gap=self.gap,
                 metric_status=self.metric_status, id_switches=self.id_switches[-30:],
                 new=dict(self.new.result(), **self.metrics) if lab['ok'] else None, old=self.old.result() if lab['ok'] else None, draw=DRAW[0],
                 matching='CLEAR MOT (Bernardin & Stiefelhagen 2008, as py-motmetrics): last frame pairs kept while IoU >= 0.5, Hungarian for the rest',
                 per_frame_hungarian=dict(note='same frames, plain Hungarian every frame without keeping last frame pairs; not shown on the page',
                                          new=self.new_h.result(), old=self.old_h.result()) if lab['ok'] else None)
        if with_rules: self._sum = s
        return s

    def summary(self):
        with self.lock: return self._summary_locked()

    def _compute_prefix(self):
        while True:
            with self.lock:
                pos = self.prefix_request
                self.prefix_request = None
                self.prefix_busy = pos
                if pos is None:
                    self.prefix_worker = None
                    return
                frames = self.history[:pos + 1]
            try:
                new, old = gt_score.Score(True), gt_score.Score(False)
                metric_inputs = []
                for entry in frames:
                    if entry is None:
                        continue
                    png, rows, metric_input = entry
                    gt = self.lab['gt'].get(png, ())
                    new.frame(gt, rows, 1)
                    old.frame(gt, rows, 1)
                    metric_inputs.append(metric_input)
                metrics = replay_metrics.compute(metric_inputs) if metric_inputs else dict(hota=None, ap50=None, map=None)
                result = dict(new=dict(new.result(), **metrics), old=old.result(), metric_status='ready' if metric_inputs else 'unavailable')
            except Exception as exc:
                print('REPLAY PREFIX failed', type(exc).__name__, str(exc)[:120], flush=True)
                result = dict(new=None, old=None, metric_status='error', why='이 위치의 점수를 계산하지 못했습니다.')
            with self.lock:
                self.prefix_cache[pos] = result
                self.prefix_cache.move_to_end(pos)
                if len(self.prefix_cache) > PREFIX_CACHE_MAX:
                    self.prefix_cache.popitem(last=False)

    def prefix(self, pos):
        with self.lock:
            state = dict(self._summary_locked(), prefix_pos=pos, prefix_pending=False, prefix_unavailable=False)
            if not self.lab['ok'] or not 0 <= pos < len(self.history):
                state.update(prefix_unavailable=True, scoring=False, new=None, old=None, metric_status='unavailable',
                             pass_frames=0, counts={}, id_switches=[], per_frame_hungarian=None,
                             why=self.lab['why'] or '이 위치까지 계산한 결과가 없습니다. 처음부터 끝까지 다시 계산해 주세요.')
                return state
            if self.phase == 'complete' and pos == self.sequence_frames - 1:
                state['prefix_pending'] = self.metric_status == 'pending'
                return state
            scored = sum(entry is not None for entry in self.history[:pos + 1])
            state.update(phase='prefix', pass_frames=pos + 1, counts=dict(scored=scored), scored_unique=scored,
                         id_switches=[event for event in self.id_switches if event['frame'] <= pos + 1][-30:],
                         per_frame_hungarian=None)
            if pos in self.prefix_cache:
                self.prefix_cache.move_to_end(pos)
                state.update(self.prefix_cache[pos])
                return state
            if self.prefix_busy != pos:
                self.prefix_request = pos
            if self.prefix_worker is None:
                self.prefix_worker = threading.Thread(target=self._compute_prefix, name='replay-prefix', daemon=True)
                self.prefix_worker.start()
            state.update(prefix_pending=True, new=None, old=None, metric_status='pending')
            return state

    def close(self, reason):   # 재생이 끝났다: 끝난 때와 이유를 적는다 (라벨이 없어 채점하지 않은 녹화도). 파일은 write() 가 쓴다
        with self.lock:
            if self.ended is None:
                if self.native_tracker is not None:
                    self.native_tracker.close()
                self.ended = time.strftime('%Y-%m-%d %H:%M:%S'); self.end_reason = reason; self._sum = None
                if self.phase == 'scoring':
                    self.phase = 'incomplete' if reason == 'ended' else 'interrupted'

    def write(self, reason):
        self.close(reason)
        if not self.lab['ok']: return None
        if self.metric_worker is not None:
            self.metric_worker.join()
        with self.lock:
            safe = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in self.name)[:60].strip('_') or 'replay'
            stamp = time.strftime('%Y%m%d_%H%M%S') + ('_%09d' % (time.time_ns() % 1_000_000_000))
            p = APP.out / ('replay_eval_%s_%s.json' % (safe, stamp))
            d = dict(self._summary_locked(False), rules=gt_score.RULES_KO,
                     thresholds=dict(iou=gt_score.IOU_MIN, ioa=gt_score.IOA_MIN, edge_px=gt_score.EDGE_PX, edge_width=gt_score.EDGE_W, small_height=gt_score.SMALL_H),
                     references=['K. Bernardin, R. Stiefelhagen. Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. EURASIP JIVP 2008 (MOTA, ID switches)',
                                 'E. Ristani, F. Solera, R. Zou, R. Cucchiara, C. Tomasi. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCV Workshops 2016 (IDF1)'],
                     metric_scope='HOTA uses final track IDs and project ignore rules. AP50 and mAP50:95 use person detections after the detector score filter/NMS, before tracking and the temperature filter. Full first pass only.',
                     counts_note='received = frames from the replay camera with a PNG number; cam_skipped = frames the replay camera skipped (receiver slow); '
                                 'published = frames processed since the start; untagged = shown without a known PNG number (not scored); '
                                 'not_reviewed / not_labelled = PNG without a reviewed label (not scored); scored = frames in the numbers; server_skipped = received but not shown; '
                                 'repeats = frames repeated during pause or at the end (not scored); scored_unique = distinct PNGs scored in the first pass',
                     id_switches=self.id_switches, frames_cols=['image_id', 'png', 'pass', 'new_found', 'new_missed', 'new_false', 'old_found', 'old_missed', 'old_false', 'pos'], frames=self.log)
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


def lossless(p=None):
    ev = EV if p is None else getattr(p, 'lwir_eval', None)
    return ev is not None and ev.lab['ok']


def first_pass_position(t_in):
    ev = EV
    return ev.first_pass_position(t_in) if ev is not None and ev.lab['ok'] else None


def initial_fps():
    ev = EV
    return ev.fps if ev is not None and ev.lab['ok'] and ev.phase == 'scoring' else None


def reset_first_input(t_in):
    if first_pass_position(t_in) != 0:
        return False
    APP.LIVE.lo = APP.LIVE.hi = None
    stab_off()
    return True


def on_frame(t_in, rows, sc, image_id, detections=None, tracker_detections=None, *, preview_receipt=None):
    ev = EV
    if ev is None:
        return None, None
    native_rows = ev.track_frame(t_in, tracker_detections)
    try:
        return ev.frame(t_in, rows if native_rows is None else native_rows, sc, image_id, detections,
                        preview_receipt=preview_receipt), native_rows
    except Exception as e:
        print('REPLAY EVAL frame failed', type(e).__name__, str(e)[:120], flush=True)
        return None, native_rows


def draw_boxes(t_in, gt, sc):   # 화면 그림에만 그릴 정답 상자: [(x1, y1, x2, y2, 종류)] 종류 0 = 사람, 1 = 무시 구역(클래스 1 또는 가장자리 규칙)
    if not DRAW[0] or gt is None: return []
    dx, dy = (sc[0], sc[1]) if sc is not None else (0.0, 0.0)   # 실제 좌표 -> 흔들림 보정한 그림 좌표
    return [(g[1] + dx, g[2] + dy, g[3] + dx, g[4] + dy, 1 if (int(g[5]) == 1 or gt_score.edge_sliver(g[1:5])) else 0) for g in gt if int(g[5]) in (0, 1)]


def state():
    ev = EV or LAST
    if ev is None:
        return dict(on=False, scoring=False, why=None, draw=DRAW[0], final_ready=False)
    current = ev.summary()
    with _LK:
        has_saved_final = (ev.name, ev.backend) in FINAL
    return dict(current, on=ev is EV, final_ready=has_saved_final or (current['scoring'] and current['phase'] == 'complete'))


def results(q):
    ev = EV or LAST
    if q.get('view', [''])[0] == 'final':
        result = ev.summary() if ev is not None else None
        if result and result['phase'] == 'complete' and result['scoring']:
            result = dict(result, on=ev is EV, final_previous=False)
        elif ev is not None:
            with _LK:
                previous = FINAL.get((ev.name, ev.backend))
            result = dict(previous, final_previous=True) if previous is not None else None
        return dict(eval=result, eval_last=last())
    if 'pos' in q:
        try:
            pos = int(q['pos'][0])
        except (IndexError, TypeError, ValueError):
            return dict(eval=None, eval_last=last())
        return dict(eval=dict(ev.prefix(pos), on=ev is EV) if ev is not None else None, eval_last=last())
    return dict(state(), eval_last=last())


def is_repeat(t_in):   # live_publish: 이 프레임은 재생 프로그램이 되풀이로 보낸 것인가 (멈춘 동안, 끝날 때). 멈춘 그림 규칙이 로그를 쏟지 않게. 재생을 안 썼으면 바로 False
    if not _RT: return False
    with _RL: return t_in in _RT


def last():   # 끝난 재생의 마지막 숫자 (서버가 다시 켜지면 없어진다). 없으면 None
    e = LAST
    return dict(e.summary(), on=False) if e is not None else None


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
