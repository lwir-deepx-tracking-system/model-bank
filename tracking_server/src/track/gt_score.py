"""정답 라벨과 비교한 점수 (2026-11-12, 녹화 재생에서 씀). 서버 상태(APP)를 모르는 순수 계산이라 따로 시험할 수 있다.

한 프레임 = 정답 줄 목록 + 서버가 낸 사람 상자 목록. 모두 640x480 화소 좌표 (x1, y1, x2, y2).

규칙 (사용자 규칙, 2026-11-12)
  1. 사람만 본다: 정답 클래스 0 과 서버의 사람 상자.
  2. 짝짓기 (CLEAR MOT, Bernardin & Stiefelhagen 2008; py-motmetrics 와 같음): IoU >= 0.5 인 짝만, 하나는 하나와만.
     먼저 앞 프레임에서 짝이었던 (정답 번호, 서버 번호) 가 이번에도 IoU >= 0.5 이면 그대로 짝짓고, 나머지만 헝가리안 방법으로 짝짓는다
     (scipy linear_sum_assignment; 비용 1 - IoU, 0.5 미만 짝은 아주 큰 비용 -> 맞는 짝 수가 가장 많고 그 안에서 IoU 합이 가장 큰 짝짓기).
     Score(carry=False) = 앞 프레임 짝을 보지 않고 프레임마다 헝가리안만 (결과 파일의 per_frame_hungarian, 비교용).
  3. 두 가지 채점을 늘 함께 센다.
     - 새 채점 (ignore=True): 정답 클래스 1 상자 = 무시 구역. 왼쪽·오른쪽 가장자리에서 4 px 안에 닿고 보이는 폭이 24 px 보다 좁은
       사람 상자도 무시 구역 (가장자리 규칙). 짝이 없는 서버 상자가 무시 구역 하나 안에 자기 넓이의 반 이상 들어가면
       (IoA >= 0.5) 잘못 그린 상자로 세지 않는다. 무시 구역은 놓침으로 세지 않는다.
     - 옛 채점 (ignore=False): 클래스 0 상자만 정답, 가장자리 규칙 없음, 무시 구역 없음 (클래스 1 상자는 버림).
  4. 놓침은 작은 사람(보이는 높이 < 32 px)과 그 밖으로 나눠 센다.
  추적 숫자
     - 번호 바뀜(ID switch): 정답 사람 하나가 짝지어진 서버 번호가, 그 사람이 마지막으로 짝지어졌던 서버 번호와 다를 때 1
       (마지막 짝은 짝이 없던 프레임을 건너 기억한다). Bernardin & Stiefelhagen 2008 의 정의, py-motmetrics 의 SWITCH 와 같은 셈.
     - 번호는 frame() 의 key(바퀴 번호)와 묶어 센다: 반복 재생의 바퀴 하나 = 영상 하나 (여러 영상을 더하는 보통 방법).
     - MOTA = 1 - (놓침 + 잘못 그린 상자 + 번호 바뀜) / 정답 사람 수. Bernardin & Stiefelhagen 2008.
     - IDF1 = 2 IDTP / (정답 상자 수 + 서버 상자 수). IDTP = 정답 번호와 서버 번호를 하나씩 짝지은 것 가운데 (채점한 모든 프레임에 걸쳐 한 번에 고름)
       같은 프레임에서 IoU >= 0.5 인 횟수의 합이 가장 큰 것. Ristani et al. 2016. 새 채점에서 무시 구역 때문에 빠진 서버 상자는 셈에 넣지 않는다.

참고 문헌
  - K. Bernardin, R. Stiefelhagen. Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.
    EURASIP Journal on Image and Video Processing, 2008 (Article ID 246309).
  - E. Ristani, F. Solera, R. Zou, R. Cucchiara, C. Tomasi. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking.
    ECCV 2016 Workshops (BMTT), LNCS 9914.
Python 3.10, numpy, scipy 만 쓴다.
"""
import collections

import numpy as np

W, H = 640.0, 480.0
IOU_MIN = 0.5      # 짝이 되는 IoU
IOA_MIN = 0.5      # 무시 구역 안에 든 비율 (서버 상자 넓이 기준)
EDGE_PX = 4.0      # 가장자리에 "닿음" = 왼쪽 x1 <= 4 또는 오른쪽 x2 >= 636
EDGE_W = 24.0      # 가장자리에 닿고 보이는 폭이 이보다 좁으면 무시 구역 (새 채점만)
SMALL_H = 32.0     # 보이는 높이가 이보다 작으면 "작은 사람"
BIG = 1.0e6

RULES_KO = [
    '사람만 채점: 정답 클래스 0 과 서버가 내보낸 사람 상자 (화면에 보이고 짐벌에 가는 상자와 같은 것, 640x480 좌표)',
    '짝짓기 (CLEAR MOT, py-motmetrics 와 같음): IoU 0.5 이상만, 하나는 하나와만. 앞 프레임의 짝이 아직 IoU 0.5 이상이면 먼저 유지하고, 나머지만 헝가리안 방법',
    '새 채점: 정답 클래스 1 = 무시 구역. 왼쪽·오른쪽 가장자리 4 px 안에 닿고 보이는 폭이 24 px 보다 좁은 사람도 무시 구역. '
    '짝이 없는 서버 상자가 무시 구역 하나 안에 반 이상 들어가면(IoA 0.5 이상) 잘못 그린 상자가 아님. 무시 구역은 놓침이 아님',
    '옛 채점: 클래스 0 상자만 정답, 가장자리 규칙 없음, 무시 구역 없음',
    '놓침은 작은 사람(높이 32 px 미만)과 그 밖으로 나눔. 작은 사람을 놓치는 것은 받아들일 수 있고, 잘못 그린 상자는 안 됨',
    '번호 바뀜·MOTA: Bernardin & Stiefelhagen 2008 (CLEAR MOT). IDF1: Ristani et al. 2016 (ECCV Workshops), 채점한 모든 프레임에 걸쳐 번호를 한 번에 짝지음',
    '정답 라벨 = tracks_edited_mot.txt (검수한 라벨). 라벨 파일에 있고 review_state.json 이 검수했다고 적은 프레임만 채점 (removed·purged 프레임은 뺌)',
    '재생 카메라가 어느 PNG 를 보냈는지 프레임마다 머리에 적어 보냄. 번호를 모르는 프레임과 재생 카메라가 건너뛴 프레임은 채점하지 않음',
    '반복 재생: 바퀴마다 계속 채점. 바퀴 하나를 영상 하나처럼 보고 더함 (정답 번호와 서버 번호 모두 바퀴마다 새로 셈)',
    '재생 시작 직후 추적 기록이 없는 프레임도 그대로 채점에 넣음. 숫자를 좋게 하려고 바꾼 것 없음',
    '결과 파일의 per_frame_hungarian = 같은 프레임을 앞 프레임 짝을 보지 않고 프레임마다 헝가리안만으로 센 숫자. 비교용, 화면에는 없음',
]


def iou_matrix(a, b):
    """a: (n, 4), b: (m, 4) x1 y1 x2 y2 -> (n, m) IoU"""
    a = np.asarray(a, np.float64).reshape(-1, 4); b = np.asarray(b, np.float64).reshape(-1, 4)
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    un = aa[:, None] + ab[None, :] - inter
    return np.where(un > 0, inter / np.where(un > 0, un, 1.0), 0.0)


def ioa(box, region):
    """box 넓이 가운데 region 안에 든 비율"""
    iw = min(box[2], region[2]) - max(box[0], region[0]); ih = min(box[3], region[3]) - max(box[1], region[1])
    ar = (box[2] - box[0]) * (box[3] - box[1])
    if iw <= 0 or ih <= 0 or ar <= 0: return 0.0
    return iw * ih / ar


def match(gt_boxes, out_boxes):
    """-> ([(i, j)], IoU 행렬). IoU >= 0.5 짝만, 하나는 하나와만, 헝가리안."""
    iou = iou_matrix(gt_boxes, out_boxes)
    if not iou.size: return [], iou
    ok = iou >= IOU_MIN
    if not ok.any(): return [], iou
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(np.where(ok, 1.0 - iou, BIG))
    return [(int(i), int(j)) for i, j in zip(r, c) if ok[i, j]], iou


def visible(b):
    """640x480 안에 보이는 부분 (x1, y1, x2, y2)"""
    return (max(0.0, b[0]), max(0.0, b[1]), min(W, b[2]), min(H, b[3]))


def edge_sliver(b):
    """가장자리 규칙: 왼쪽·오른쪽 가장자리 4 px 안에 닿고 보이는 폭이 24 px 보다 좁은 상자"""
    v = visible(b)
    return (v[0] <= EDGE_PX or v[2] >= W - EDGE_PX) and (v[2] - v[0]) < EDGE_W


def split_gt(gt_rows, ignore):
    """gt_rows: [(번호, x1, y1, x2, y2, 클래스)] -> (사람 [(번호, 상자)], 무시 구역 [상자])"""
    ppl, ign = [], []
    for g in gt_rows:
        gid, b, cl = g[0], tuple(float(v) for v in g[1:5]), int(g[5])
        if cl == 0:
            if ignore and edge_sliver(b): ign.append(b)
            else: ppl.append((gid, b))
        elif cl == 1 and ignore: ign.append(b)
    return ppl, ign


class Score:
    """채점 하나 (새 채점 ignore=True, 옛 채점 ignore=False). frame() 을 프레임마다, result() 를 볼 때 부른다."""
    KEYS = ('frames', 'gt', 'found', 'missed', 'missed_small', 'false', 'ignored_out', 'idsw', 'out')

    def __init__(self, ignore, carry=True):
        """carry=True (기본, 화면과 결과 파일의 주 숫자): CLEAR MOT 논문과 py-motmetrics 처럼, 앞 프레임에서 짝이었던 (정답, 서버 번호) 가 이번에도
        IoU >= 0.5 이면 먼저 그대로 짝짓고 나머지만 헝가리안으로 짝짓는다. carry=False: 프레임마다 헝가리안만 (비교용, 사람이 겹칠 때 번호 바뀜이 더 세진다)."""
        self.ignore = bool(ignore); self.carry = bool(carry); self.reset()

    def reset(self):
        self.c = dict.fromkeys(self.KEYS, 0)
        self.last = {}                                 # 정답 번호 -> 마지막으로 짝지어진 서버 번호
        self.pairs = collections.Counter()             # (정답 번호, 서버 번호) -> IoU >= 0.5 였던 프레임 수 (IDF1)
        self.n_gt = collections.Counter(); self.n_out = collections.Counter()
        self._res = None

    def frame(self, gt_rows, out_rows, key=None, *, switches=None, kept=None):
        """gt_rows: [(번호, x1, y1, x2, y2, 클래스)], out_rows: [(서버 번호, x1, y1, x2, y2)] (사람만), key: 정답 번호와 서버 번호 앞에 붙는 값
        (바퀴 번호: 바퀴마다 다른 영상처럼 센다). -> 이 프레임의 셈 dict"""
        ppl, ign = split_gt(gt_rows, self.ignore)
        ob = [tuple(float(v) for v in o[1:5]) for o in out_rows]
        oid = [(key, int(o[0])) for o in out_rows]
        pairs, iou = match([b for _, b in ppl], ob) if not self.carry else self._match_carry(ppl, oid, ob, key)
        mg = {i for i, _ in pairs}; mo = {j for _, j in pairs}
        f = dict(gt=len(ppl), found=len(pairs), missed=0, missed_small=0, false=0, ignored_out=0, idsw=0, out=0)
        for i, j in pairs:
            g = (key, ppl[i][0]); o = oid[j]
            if g in self.last and self.last[g] != o:
                f['idsw'] += 1
                if switches is not None:
                    switches.append((int(g[1]), self.last[g][1], o[1]))
            self.last[g] = o
        for i, (_, b) in enumerate(ppl):
            if i in mg: continue
            f['missed'] += 1
            v = visible(b)
            if v[3] - v[1] < SMALL_H: f['missed_small'] += 1
        keep = []
        for j, b in enumerate(ob):
            if j in mo: keep.append(j); continue
            if self.ignore and any(ioa(b, r) >= IOA_MIN for r in ign): f['ignored_out'] += 1; continue
            f['false'] += 1; keep.append(j)
        if kept is not None:
            kept.extend(keep)
        f['out'] = len(keep)
        for i, (gid, _) in enumerate(ppl):
            g = (key, gid); self.n_gt[g] += 1
            for j in keep:
                if iou[i, j] >= IOU_MIN: self.pairs[(g, oid[j])] += 1
        for j in keep: self.n_out[oid[j]] += 1
        self.c['frames'] += 1
        for k, v in f.items(): self.c[k] += v
        self._res = None
        return f

    def _match_carry(self, ppl, oid, ob, key):
        iou = iou_matrix([b for _, b in ppl], ob)
        fixed = []; ug = set(); uo = set()
        for i, (gid, _) in enumerate(ppl):
            h = self.last.get((key, gid))
            if h is None: continue
            j = next((j for j in range(len(oid)) if j not in uo and oid[j] == h), None)
            if j is not None and iou[i, j] >= IOU_MIN: fixed.append((i, j)); ug.add(i); uo.add(j)
        ri = [i for i in range(len(ppl)) if i not in ug]; rj = [j for j in range(len(ob)) if j not in uo]
        rest = []
        if ri and rj:
            sub = iou[np.ix_(ri, rj)]; ok = sub >= IOU_MIN
            if ok.any():
                from scipy.optimize import linear_sum_assignment
                r, c = linear_sum_assignment(np.where(ok, 1.0 - sub, BIG))
                rest = [(ri[a], rj[b]) for a, b in zip(r, c) if ok[a, b]]
        return fixed + rest, iou

    def idtp(self):
        if not self.pairs: return 0
        gs = sorted({g for g, _ in self.pairs}, key=repr); os_ = sorted({o for _, o in self.pairs}, key=repr)
        gi = {g: i for i, g in enumerate(gs)}; oi = {o: i for i, o in enumerate(os_)}
        m = np.zeros((len(gs), len(os_)))
        for (g, o), n in self.pairs.items(): m[gi[g], oi[o]] = n
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-m)
        return int(round(m[r, c].sum()))

    def result(self):
        if self._res is not None: return self._res
        c = dict(self.c); gt, out = c['gt'], c['out']
        tp = self.idtp()
        pct = lambda a, b: round(100.0 * a / b, 2) if b else None
        c.update(missed_other=c['missed'] - c['missed_small'], recall=pct(c['found'], gt), precision=pct(c['found'], c['found'] + c['false']),
                 mota=round(100.0 * (1.0 - (c['missed'] + c['false'] + c['idsw']) / gt), 2) if gt else None,
                 idtp=tp, idf1=pct(2 * tp, gt + out), idp=pct(tp, out), idr=pct(tp, gt), gt_ids=len(self.n_gt), out_ids=len(self.n_out))
        self._res = c
        return c
