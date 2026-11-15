"""사람 상자 다듬기 (2026-11-15, 사용자 "2번부터 하자", "변할 때 자연스럽게 변해야 한다, 확확 변하면 안 된다", "연속적인 네모 안정적으로 변하도록"). 기본은 꺼짐.

무엇을: 화면에 그리는 사람 상자와 짐벌에 보내는 사람 상자(짐벌 조준점 = 위 + 0.37 x 높이)만 다듬는다. page_handler.live_publish 가 그리기와
짐벌 보내기 바로 전에 apply 를 부른다. 추적기 안의 상자(Kalman 상태), 짝짓기, KeepID, 녹화 채점, 검출 모델 자동 선택, 측정 기록, 피부 온도는
그대로 원래 상자를 쓴다.

모드 (/rules/set?box_refine=off|euro|gru|size, last_selection.json 에 남음):
  off   아무것도 안 함 (기본). 이때 apply 는 받은 행을 그대로 돌려주고 아무 상태도 만들지 않는다.
  size  크기만 부드럽게 (11-15 v2). 위치는 추적기 행 그대로(SIZE['anchor']: 'top' = 위 가장자리와 가로 중심, 'centre' = 중심, 'aim' = 짐벌 조준점
        위 + aim_y x 키 를 추적기 행과 똑같이 둠(aim_y = 돌고 있는 짐벌 프로그램의 --aim-y, 없으면 --gimbal-aim-y): 짐벌 조준은 지금과 같고 그리는 크기만 바뀜), 상자 크기
        (log 가로, log 세로)만 One Euro 필터로 거른다. dt = 실제 초, 거르기 전에 카메라 확대/축소(그 번호의 앞 프레임 뒤 카메라 움직임의
        log sqrt|det H|)를 상태에 더해 카메라 배율은 거르지 않는다. 검출 모델이 바뀌어도(근거리 <-> 원거리, 상자 정의가 다름) 상태를 그대로 두어
        바뀐 크기가 시간에 걸쳐 퍼진다. 머리 위는 두 정의에서 같으므로 'top' 이면 정의가 바뀔 때 아래 가장자리만 천천히 움직인다.
        값(SIZE)은 공개 자료 val-A 에서 미리 적은 규칙으로 고른 것 (pos_refine_20261115/v2/PREREG_V2.json).
  euro  One Euro 필터 (Casiez, Roussel, Vogel, CHI 2012). 사람(번호)마다 검출 상자의 중심(x, y)과 크기(log 가로, log 세로)를 따로 거른다.
        속도 항은 초당 상자 높이로 잰다(사람 크기와 무관). 거르기 전에 앞 상태를 카메라 움직임(GMC, --gmc)만큼 옮긴다:
        짐벌이 돌아 그림이 밀려도 상자가 뒤처지지 않는다. 값(EURO)은 공개 자료 val-A 에서 고른 것 (pos_refine_20261115).
  gru   학습한 GRU (models/track/box_refine_gru_*.npz, 이름순 마지막). 이번 프레임까지의 검출 상자, 관절 17개, 이 모듈의 Kalman(추적기와 같은 식,
        카메라 움직임 보정) 값을 읽고 지금 상자를 고친다(중심 +-0.5 키, 크기 x/ 2.5 안). 이번 프레임 뒤는 보지 않는다(인과).
        목표는 몸 네모 하나의 정의(머리 위 ~ 골반)라서 먼 거리 모델이 작은 사람에게 내는 전신 상자도 같은 정의로 바꾸도록 배웠다.
        관절을 내는 검출 모델([관절])에서만 고를 수 있다(관절 없는 모델로 바뀌면 쉰다). 학습: pos_refine_20261115.
섞기 (/rules/set?box_blend=초, 0 = 끔, 기본 끔): 검출 모델이 바뀌면(자동 선택, 두 모델 상주) 사람마다, 그 사람의 앞 출력이 앞 모델에서 나왔고
  3 프레임 안이며 0.15 초(낮은 프레임 속도에서는 그 사람의 앞 프레임 간격 x 2) 안이면, 앞 출력(카메라 움직임만큼 옮기고, 카메라 몫을 뺀 그 사람의
  마지막 한 프레임 움직임을 초당으로 바꿔 지난 시간만큼 앞으로 옮김)과 새 출력의 차이(중심 / 키, log 크기)를 시작할 때의 섞기 시간 동안 1 에서 0 으로
  줄여 가며 더한다. 옛 바꾸기 길(do_model_swap)처럼 0.5 초 넘게 프레임이 멈춘 뒤에는 섞지 않는다(멈추기 전 자리로 끌려가지 않게).
  size 모드에서는 크기만 섞고 위치는 추적기 행(또는 조준점) 그대로 둔다. 두 모델 상주의 즉시 바꾸기에서 옛 모델에 이미 넣은 한 장은 옛 모델 결과이므로 새 모델 결과가 실제로 나온 프레임부터 새 모델로 본다
  (resident_det RES.watch). 섞기 시간만 바꾸면 사람마다 상태는 그대로 둔다.

사람마다 상태: 번호별. 그 번호가 이번 프레임에 없으면 카메라 움직임을 모으고(gru 는 Kalman 한 걸음 예측) 기다린다. 30 프레임(추적기 buffer) 넘게
없으면(빠진 프레임 > 30), 또는 새 검출이 카메라 움직임으로 옮긴 앞 검출에서 키의 1.5배 넘게 떨어지면(다른 사람) 처음부터.
번호가 바뀌어도(KeepID 다시 붙이기) 새 번호에 이어받을 상태가 없거나 앞 프레임에 안 보였으면, 앞 프레임에 보였다가 이번에 없어진 번호 가운데
마지막 검출(카메라 움직임으로 옮김)과 IoU 0.5 이상인 것의 상태를 넘겨받는다. 추적기가 새로 만들어지면(녹화 다시 시작, 엔드투엔드 켜고 끔, 추적기 바꿈)
모두 비운다. 같은 규칙을 공개 자료 채점(pos_refine_20261115/v2)도 이 클래스로 그대로 쓴다.
쓰지 않는 곳: 엔드투엔드 모델 사용 중(검출 상자·관절이 없음: 꺼진 것처럼 그대로), 녹화를 다른 추적기 결과로 재생할 때 그리는 행(재생 쪽 행은 그대로).
"""
import math
import os
import sys
import threading
import time

import numpy as np

from ..common.state import APP
from ..common.shared_values import PERSON_CLS

MODES = ('off', 'euro', 'gru', 'size')
MODE_KO = dict(off='끔', euro='One Euro 필터', gru='학습 GRU (몸 네모 하나로)', size='크기만 부드럽게')
# One Euro values chosen on val-A (pos_refine_20261115/results/euro_valA.json): centre min_cutoff Hz, beta (per box height per s), size min_cutoff Hz, beta
EURO = dict(mc=3.0, beta=3.0, mc_s=0.4, beta_s=0.3, dcut=1.0)   # grid min_cutoff 0.05..8 Hz x beta 0..3; centre beta at the grid's top (flat: 3.0 / 1.0 within 0.1 %)
# size mode (v2, pos_refine_20261115/v2/PREREG_V2.json): One Euro on log w / log h, min_cutoff Hz, beta (per log unit per s), derivative cutoff Hz;
# anchor 'top' (top edge + centre x of the tracker row) or 'centre'; src 'det' (the matched detection's size) or 'trk' (the tracker row's size).
# val-A choice (v2/results/size_valA.json, 120 points): the pre-set limits (aim error, aim jitter, |log h| within 2 % of the tracker row, size lag within
# 0.01, aim lag vs the detection within 0.005 heights) leave only light filters; the 'top' anchor had no allowed point (DanceTrack aim jitter +6..17 %:
# the top edge keeps the tracker's height noise x 0.5 that its own aim point cancels), so 'centre'.
# v3 (review 2, v2/v3/PREREG_V3.json, v2/v3/results/aim_valA.json): anchor 'aim' keeps the gimbal aim point top + aim_y x h of the tracker row exactly
# (the centre anchor moved it by (0.5 - aim_y) x the height change: thermal aim jitter +1 %, and the device aims at 0.08, not 0.37); the same
# grid and rule pick the same filter (trk, 5 Hz, beta 3), the binding limit is still the DanceTrack size lag.
SIZE = dict(mc_s=5.0, beta_s=3.0, dcut=1.0, anchor='aim', src='trk')
BLEND_MAX = 2.0
BLEND_RECENT = 3       # an id's previous output at most this many frames old starts a detector-switch blend for that id
BLEND_RECENT_S = 0.15  # ... and at most this many seconds old (or 2 x the id's own last frame interval, low frame rates): after a stopped swap
                       # (do_model_swap, > 0.5 s without frames) the old output is where the person was before the stop: no blend from it
AIM_Y = 0.5            # gimbal aim point (fraction of the box height from its top) when the running bridge does not say (size anchor 'aim')
LOST_MAX = 30          # frames a missing id keeps its state (ByteTrack buffer): reset when more than LOST_MAX frames are missing
JUMP_H = 1.5           # a detection farther than this many box heights from the id's previous one (moved by the camera) = another person
HANDOVER_IOU = 0.5     # a new id takes the state of an id that vanished this frame when its detection overlaps that id's last one this much
ROW0, ROW1, XW = 16.0, 496.0, 640.0
NF = 70
REF_C, REF_S, CLIP = 0.5, math.log(2.5), 5.0
KP_IOU = 0.9           # the row's detection and a keypoint detection of the same frame are the same box (detect_frame.KP holds the NMS rows)
ERR_MAX = 50           # errors in a row before the module switches itself off

BR = dict(mode='off', blend_s=0.0, lock=threading.Lock(), tr={}, frame=0, ms=0.0, ms_max=0.0, ms_track=0.0, n=0, frames=0, err='', errors=0, errors_all=0,
          model=None, file=None, note='', mt=None, reset_key=None, blends=0, handovers=0, why='', dir=None)


# ------------------------------------------------------------------------------------------------ pieces (no APP)
def _alpha(cut, dt): return 1.0 / (1.0 + 1.0 / (2 * math.pi * cut * dt))


def log_scale(H):
    """log sqrt|det| of a 2x3 partial affine warp (the camera zoom between two frames), 0 for None."""
    if H is None: return 0.0
    return math.log(max(math.sqrt(abs(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])), 1e-6))


class Euro:
    def __init__(self, p):
        self.p = (p['mc'], p['beta'], p['mc_s'], p['beta_s'], p.get('dcut', 1.0)); self.x = None; self.dx = None

    @staticmethod
    def _a(cut, dt): return _alpha(cut, dt)

    def step(self, z, dt, H):
        mc, beta, mcs, bs, dcut = self.p; s = np.array([z[0], z[1], math.log(max(z[2], 1.)), math.log(max(z[3], 1.))])
        if self.x is None: self.x = s; self.dx = np.zeros(4); return self.box()
        if H is not None:
            self.x[:2] = H[:2, :2] @ self.x[:2] + H[:2, 2]; self.dx[:2] = H[:2, :2] @ self.dx[:2]
            self.x[2:] += log_scale(H)
        dt = max(dt, 1e-3); h = math.exp(self.x[3]); ad = _alpha(dcut, dt)
        self.dx = ad * (s - self.x) / dt + (1 - ad) * self.dx
        a = np.array([_alpha(mc + beta * abs(self.dx[0]) / h, dt), _alpha(mc + beta * abs(self.dx[1]) / h, dt),
                      _alpha(mcs + bs * abs(self.dx[2]), dt), _alpha(mcs + bs * abs(self.dx[3]), dt)])
        self.x = a * s + (1 - a) * self.x; return self.box()

    def box(self): return np.array([self.x[0], self.x[1], math.exp(self.x[2]), math.exp(self.x[3])])


class SizeEuro:
    """One Euro on (log w, log h) only (plain floats: a few microseconds per person). step(lw, lh, dt s, ls = camera log scale since the
    previous step) -> (lw, lh) filtered. The camera scale is added to the state first, so a camera zoom is followed at once and not filtered."""
    __slots__ = ('mc', 'beta', 'dcut', 'x0', 'x1', 'd0', 'd1')

    def __init__(self, mc, beta, dcut=1.0):
        self.mc = float(mc); self.beta = float(beta); self.dcut = float(dcut); self.x0 = None

    def step(self, lw, lh, dt, ls=0.0):
        if self.x0 is None:
            self.x0, self.x1, self.d0, self.d1 = lw, lh, 0.0, 0.0; return lw, lh
        x0 = self.x0 + ls; x1 = self.x1 + ls; dt = max(dt, 1e-3); ad = _alpha(self.dcut, dt)
        d0 = ad * (lw - x0) / dt + (1 - ad) * self.d0; d1 = ad * (lh - x1) / dt + (1 - ad) * self.d1
        a0 = _alpha(self.mc + self.beta * abs(d0), dt); a1 = _alpha(self.mc + self.beta * abs(d1), dt)
        self.x0 = a0 * lw + (1 - a0) * x0; self.x1 = a1 * lh + (1 - a1) * x1; self.d0 = d0; self.d1 = d1
        return self.x0, self.x1


class GRUNp:
    """numpy forward of the refiner (torch GRUCell(70, 48) + Linear(48, 4)); keys = the torch state_dict names."""
    def __init__(self, w):
        g = lambda k: np.asarray(w[k], np.float64)
        self.wih, self.whh, self.bih, self.bhh, self.wo, self.bo = g('cell.weight_ih'), g('cell.weight_hh'), g('cell.bias_ih'), g('cell.bias_hh'), g('head.weight'), g('head.bias')
        self.n = self.whh.shape[1]; assert self.wih.shape == (3 * self.n, NF) and self.wo.shape == (4, self.n)

    def step(self, x, h):
        n = self.n; gi = self.wih @ x + self.bih; gh = self.whh @ h + self.bhh
        r = 1. / (1. + np.exp(-(gi[:n] + gh[:n]))); zz = 1. / (1. + np.exp(-(gi[n:2 * n] + gh[n:2 * n]))); c = np.tanh(gi[2 * n:] + r * gh[2 * n:])
        h = (1. - zz) * c + zz * h
        return self.wo @ h + self.bo, h


SP, SV = 1 / 20, 1 / 160
F8 = np.eye(8); F8[np.arange(4), np.arange(4, 8)] = 1.0


def kf_init(z):
    std = [2 * SP * z[2], 2 * SP * z[3], 2 * SP * z[2], 2 * SP * z[3], 10 * SV * z[2], 10 * SV * z[3], 10 * SV * z[2], 10 * SV * z[3]]
    return np.r_[z, np.zeros(4)], np.diag(np.square(std))


_D8 = np.arange(8)


def kf_predict(m, c, lost, H):
    """kalman.KF.multi_predict + gmc_warp for one track (kron(I4, R) built block by block: the same matrix, cheaper)."""
    m = m.copy()
    if lost: m[6:8] = 0
    q = np.array([SP * m[2], SP * m[3], SP * m[2], SP * m[3], SV * m[2], SV * m[3], SV * m[2], SV * m[3]]); q *= q
    m = F8 @ m; c = F8 @ c @ F8.T; c[_D8, _D8] += q
    if H is not None:
        R = H[:2, :2]; R8 = np.zeros((8, 8)); R8[0:2, 0:2] = R; R8[2:4, 2:4] = R; R8[4:6, 4:6] = R; R8[6:8, 6:8] = R
        m = R8 @ m; m[:2] += H[:2, 2]; c = R8 @ c @ R8.T
    return m, c


_D4 = np.arange(4)


def kf_update(m, c, z):
    r = np.array([SP * m[2], SP * m[3], SP * m[2], SP * m[3]]); pc = c[:4, :4].copy(); pc[_D4, _D4] += r * r; K = np.linalg.solve(pc, c[:, :4].T).T
    return m + (z - m[:4]) @ K.T, c - K @ pc @ K.T


def warp_box(b, H):
    s = math.sqrt(abs(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])); o = np.empty(4)
    o[0] = H[0, 0] * b[0] + H[0, 1] * b[1] + H[0, 2]; o[1] = H[1, 0] * b[0] + H[1, 1] * b[1] + H[1, 2]; o[2] = b[2] * s; o[3] = b[3] * s
    return o


def compose(H2, H1):
    """H2 after H1 (2x3 partial affines)."""
    A = H2[:2, :2] @ H1[:2, :2]; return np.c_[A, H2[:2, :2] @ H1[:2, 2] + H2[:2, 2]]


def iou_c(a, b):
    """IoU of two cxcywh boxes."""
    iw = min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2); ih = min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2)
    if iw <= 0 or ih <= 0: return 0.0
    i = iw * ih; return i / max(a[2] * a[3] + b[2] * b[3] - i, 1e-9)


def features(st, z, kp, H):
    """one seen frame of one id: Kalman predict (+ warp) / update and the 70 refiner inputs (= pos_refine prcommon.ref_series). st is changed."""
    cam = np.zeros(2); new = st.get('m') is None or st['lost'] > LOST_MAX
    if not new:
        st['m'], st['c'] = kf_predict(st['m'], st['c'], st['lost'] > 0, H)
        if st.get('zp') is not None and H is not None:
            zw = warp_box(st['zp'], H); cam = zw[:2] - st['zp'][:2]; st['zp'] = zw
    if new:
        st['m'], st['c'] = kf_init(z); inn = np.zeros(4); zref = z; gap = 0; cam = np.zeros(2)
    else:
        p = st['m']; hp = max(p[3], 1.)
        inn = np.array([(z[0] - p[0]) / hp, (z[1] - p[1]) / hp, math.log(max(z[2], 1.) / max(p[2], 1.)), math.log(max(z[3], 1.) / hp)])
        st['m'], st['c'] = kf_update(st['m'], st['c'], z); zref = st['zp'] if st.get('zp') is not None else z; gap = st['lost']
    h = max(z[3], 1.); w = max(z[2], 1.); x = np.empty(NF)
    x[0:2] = st['m'][4:6] / h; x[2:6] = inn; x[6:8] = (z[:2] - zref[:2]) / h; x[8:10] = np.log(np.maximum(z[2:4], 1.) / np.maximum(zref[2:4], 1.))
    x[10] = math.log(h / 100.); x[11] = math.log(w / h)
    x[12] = (z[0] - w / 2) / h; x[13] = (XW - (z[0] + w / 2)) / h; x[14] = (z[1] - h / 2 - ROW0) / h; x[15] = (ROW1 - (z[1] + h / 2)) / h
    if kp is None: x[16:67] = 0.
    else:
        vis = (kp[:, 2] >= 0.5).astype(np.float64); x[16:67] = np.stack([(kp[:, 0] - z[0]) / h * vis, (kp[:, 1] - z[1]) / h * vis, vis], 1).ravel()
    x[67] = min(gap, LOST_MAX) / 15.; x[68:70] = cam / h
    st['zp'] = z.copy(); st['lost'] = 0
    return np.clip(x, -CLIP, CLIP, out=x)


def blend_off(prev, new):
    h = max(new[3], 1.); return np.r_[(prev[:2] - new[:2]) / h, math.log(max(prev[2], 1.) / max(new[2], 1.)), math.log(max(prev[3], 1.) / max(new[3], 1.))]


def blend_apply(b, off, a):
    h = max(b[3], 1.); return np.r_[b[:2] + a * off[:2] * h, b[2] * math.exp(a * off[2]), b[3] * math.exp(a * off[3])]


def size_box(zr, lw, lh, anchor, aim_y=AIM_Y):
    """size mode output: the tracker row's centre x with the filtered size; y keeps the row's top edge ('top'), centre ('centre') or gimbal
    aim point top + aim_y x h ('aim': the gimbal aims exactly where it aims today, only the drawn size changes)."""
    w = math.exp(lw); h = math.exp(lh)
    if anchor == 'top': cy = zr[1] - zr[3] / 2 + h / 2
    elif anchor == 'aim': cy = zr[1] - zr[3] / 2 + aim_y * (zr[3] - h) + h / 2
    else: cy = zr[1]
    return np.array([zr[0], cy, w, h])


class Refiner:
    """the per-id state machine (no APP): step(list of (id, z = detection cxcywh, kp or None[, the row's own box cxcywh]), t s, H = this frame's
    camera warp or None, det_key, new_det_ready) -> {id: box cxcywh}. Mode 'off' (blend only) starts from the row's own box (the tracker's box),
    'size' takes the position from the row's own box, the other modes start from the detection. The state of id k after a step is self.tr[k]
    ('k' = rows since that state started, 'new' = started on this frame, 'hs' = camera warp since the id's previous frame or None)."""
    def __init__(self, mode, model=None, euro=None, blend_s=0.0, size=None, aim_y=AIM_Y):
        self.mode = mode; self.model = model; self.euro = dict(euro or EURO); self.size = dict(SIZE, **(size or {})); self.blend_s = float(blend_s)
        self.aim_y = float(aim_y)   # the gimbal's aim point (size anchor 'aim'); apply() sets the running bridge's value every frame
        self.tr = {}; self.frame = 0; self.eff_key = None; self.blends = 0; self.handovers = 0

    @staticmethod
    def _recent_s(st):
        """how old (s) the previous output may be to blend from it: BLEND_RECENT_S, or 2 x the id's own last frame interval at a low frame rate."""
        if st['t1'] is not None and st['f1'] == st['f'] - 1: return max(BLEND_RECENT_S, 2.0 * (st['t'] - st['t1']))
        return BLEND_RECENT_S

    @staticmethod
    def _carry(st, Hs, t):
        """the id's previous output carried to this frame: moved (and scaled) by the camera warp since that frame, plus the id's own picture
        motion without the camera part (last frame step, per second) times the time since that frame."""
        prev = np.array(st['out'], np.float64) if Hs is None else warp_box(st['out'], Hs)
        if st['out1'] is not None and st['f1'] == st['f'] - 1 and st['t'] > st['t1']:
            o1 = st['out1'] if st.get('hs') is None else warp_box(st['out1'], st['hs'])
            prev[:2] += (np.asarray(st['out'][:2]) - o1[:2]) * ((t - st['t']) / (st['t'] - st['t1']))
        return prev

    def _jump(self, st, z, Hs):
        p = st['z'] if Hs is None else warp_box(st['z'], Hs)
        return math.hypot(z[0] - p[0], z[1] - p[1]) > JUMP_H * max(p[3], 1.)

    def step(self, items, t, H, det_key=None, new_ready=True):
        self.frame += 1; fr = self.frame; out = {}
        if det_key is not None and new_ready: self.eff_key = det_key   # the detector whose boxes this frame carries (resident switch: not before its own frame)
        cur = set(it[0] for it in items)
        gone = [k for k, s in self.tr.items() if s['f'] == fr - 1 and k not in cur] if self.tr else []
        for it in items:
            tid, z, kp = it[:3]; zr = it[3] if len(it) > 3 and it[3] is not None else z; st = self.tr.get(tid)
            if gone and (st is None or st['f'] != fr - 1):   # hand-over: KeepID gave the person another id this frame
                best, bi = HANDOVER_IOU, -1
                for j, g in enumerate(gone):
                    sg = self.tr[g]; v = iou_c(z, sg['z'] if H is None else warp_box(sg['z'], H))
                    if v >= best: best, bi = v, j
                if bi >= 0:
                    st = self.tr[tid] = self.tr.pop(gone.pop(bi)); self.handovers += 1
            Hs = None   # camera motion since this id's previous frame (its missing frames included)
            if st is not None:
                Hs = H if st['Hacc'] is None else (st['Hacc'] if H is None else compose(H, st['Hacc']))
                if fr - st['f'] - 1 > LOST_MAX or self._jump(st, z, Hs): st = None
            if st is None:
                st = self.tr[tid] = dict(f=fr, t=t, z=z, lost=0, m=None, c=None, zp=None, h=None, eu=None, se=None, Hacc=None, out=None, out1=None, f1=None,
                                         t1=None, bl=None, dk=None, k=-1, new=True, hs=None)
                Hs = None
            else:
                st['new'] = False
            if self.mode == 'size':
                p = self.size; src = z if p['src'] == 'det' else zr
                if st['se'] is None: st['se'] = SizeEuro(p['mc_s'], p['beta_s'], p['dcut'])
                lw, lh = st['se'].step(math.log(max(src[2], 1.)), math.log(max(src[3], 1.)), t - st['t'] if st['out'] is not None else 0.04, log_scale(Hs))
                b = size_box(zr, lw, lh, p['anchor'], self.aim_y)
            elif self.mode == 'euro':
                if st['eu'] is None: st['eu'] = Euro(self.euro)
                b = st['eu'].step(z, max(t - st['t'], 1e-3) if st['out'] is not None else 0.04, Hs)
            elif self.mode == 'gru' and self.model is not None:
                if st['h'] is None: st['h'] = np.zeros(self.model.n)
                x = features(st, z, kp, H); o, st['h'] = self.model.step(x, st['h']); hz = max(z[3], 1.); t_ = np.tanh(o); b = np.empty(4)
                b[0:2] = z[:2] + REF_C * hz * t_[:2]; b[2:4] = z[2:4] * np.exp(REF_S * t_[2:4])
            else:   # off (blend only): the box the tracker draws today
                b = np.asarray(zr, np.float64).copy()
            if (self.blend_s > 0 and st['dk'] is not None and self.eff_key is not None and st['dk'] != self.eff_key and st['out'] is not None
                    and fr - st['f'] <= BLEND_RECENT and t - st['t'] <= self._recent_s(st)):   # this person's previous output came from the other detector: blend from it
                prev = self._carry(st, Hs, t); off = blend_off(prev, b)
                if self.mode == 'size': off[:2] = 0.0   # size mode: the position stays the tracker row's; only the size is blended
                st['bl'] = (off, t, self.blend_s); self.blends += 1
            if st['bl'] is not None:
                off, t0, dur = st['bl']; a = 1.0 - (t - t0) / dur   # the duration the blend started with: a new blend time does not move the box back
                if a <= 0 or self.blend_s <= 0: st['bl'] = None
                elif self.mode == 'size': b = size_box(zr, lw + a * off[2], lh + a * off[3], self.size['anchor'], self.aim_y)
                else: b = blend_apply(b, off, a)
            if st['out'] is not None: st['out1'], st['f1'], st['t1'] = st['out'], st['f'], st['t']
            st.update(f=fr, t=t, z=np.asarray(z, np.float64), out=b, Hacc=None, hs=Hs, dk=self.eff_key, k=st['k'] + 1); out[tid] = b
        for tid in list(self.tr):
            if tid in out: continue
            st = self.tr[tid]
            if fr - st['f'] > LOST_MAX: del self.tr[tid]; continue
            if H is not None: st['Hacc'] = H.copy() if st['Hacc'] is None else compose(H, st['Hacc'])   # applied at the id's next frame
            if self.mode == 'gru' and st.get('m') is not None:   # the missing frame: Kalman predict (+ warp), the last box moved with the camera (= prcommon.ref_series)
                st['m'], st['c'] = kf_predict(st['m'], st['c'], st['lost'] > 0, H); st['lost'] += 1
                if st.get('zp') is not None and H is not None: st['zp'] = warp_box(st['zp'], H)
        return out


# ------------------------------------------------------------------------------------------------ server glue
def _app(name, default=None):
    return vars(APP).get(name, default)


def model_dir():
    return BR['dir'] or os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'models', 'track')


def scan():
    """newest models/track/box_refine_gru_*.npz (a file is read again only when it changed and is older than 5 s)."""
    d = model_dir()
    try: c_ = sorted(f for f in os.listdir(d) if f.startswith('box_refine_gru_') and f.endswith('.npz'))
    except OSError: c_ = []
    if not c_: BR.update(model=None, file=None, note='가중치 파일 없음', mt=None); return None
    p = os.path.join(d, c_[-1])
    try: mt = os.path.getmtime(p)
    except OSError: return None
    if BR['file'] == c_[-1] and BR['mt'] == mt: return BR['model']
    if time.time() - mt < 5: return BR['model']
    try:
        z = np.load(p); m = GRUNp({k: z[k] for k in z.files}); o, _ = m.step(np.zeros(NF), np.zeros(m.n)); assert o.shape == (4,) and np.isfinite(o).all()
        BR.update(model=m, file=c_[-1], mt=mt, note='')
    except Exception as e_:
        BR.update(model=None, file=c_[-1], mt=mt, note='가중치 파일을 읽지 못함: %s' % type(e_).__name__)
    return BR['model']


def _e2e_on():
    try:
        from . import e2e_live
        return bool(e2e_live.skip_detector())
    except Exception:
        return False


def _new_ready():
    """False right after an instant (resident) detector switch until the first frame of the new model has been collected."""
    try:
        from ..detect import resident_det
        return resident_det.RES.watch is None
    except Exception:
        return True


def _det_kpt():
    """the running detector gives keypoints (model_scan kind six_kpt): only then can gru read them (detect_frame.KP is not cleared by other models)."""
    try:
        from ..detect.model_scan import KPT_DETS
        a = _app('args')
        return a is not None and getattr(a, 'det', None) in KPT_DETS
    except Exception:
        return False


def _det_kpt_soon():
    """the running detector gives keypoints, or the one already asked for does (a waiting model change, or at startup the saved detector that
    app.py asks for only after the saved rules): gru may be chosen and waits (apply() pauses it) until that detector runs."""
    if _det_kpt(): return True
    try:
        from ..detect.model_scan import KPT_DETS
        md = _app('MODEL'); req = md.get('req') if isinstance(md, dict) else None
        if req and req.get('det') in KPT_DETS: return True
        sel = _app('SEL'); return isinstance(sel, dict) and sel.get('det') in KPT_DETS
    except Exception:
        return False


def _aim_y():
    """the gimbal's aim point (fraction of the box height from its top): the running bridge's --aim-y, else --gimbal-aim-y, else AIM_Y."""
    try:
        from ..gimbal.live_tracks import _bridge_aim_y
        v = _bridge_aim_y(_app('GB'))
        if v is not None: return float(v)
    except Exception:
        pass
    v = getattr(_app('args'), 'gimbal_aim_y', None)
    try: return float(v) if v is not None and 0.0 <= float(v) <= 1.0 else AIM_Y
    except (TypeError, ValueError): return AIM_Y


def _reset_key():
    """(tracker object, camera tracker generation, camera tracker active, end-to-end on). The tracker object itself, not id(): a new tracker
    can take a freed tracker's address, but not while this key still holds the old one."""
    cam = getattr(_app('LIVE'), 'camera_bytetrack', None)
    return (_app('tracker'), getattr(cam, 'generation', None), getattr(cam, 'active', None), _e2e_on())


def _same_key(a, b):
    return a is not None and b is not None and a[0] is b[0] and a[1:] == b[1:]


def _kp_cur():
    if not _det_kpt(): return None
    try:
        from ..detect.detect_frame import KP
        d = KP.get('cur')
        return d if d and len(d['boxes']) else None
    except Exception:
        return None


def _warp():
    """this frame's camera warp (GMC, previous -> current, canvas px) when --gmc gave one for it, else None."""
    lm = (_app('CTL') or {}).get('lead_motion')
    if lm is None: return None
    try:
        H = np.asarray(lm[2], np.float64)
        return H if H.shape == (2, 3) and np.isfinite(H).all() else None
    except Exception:
        return None


def apply(rows, dets=None, t=None):
    """Frame thread (live_publish), once per published frame: rows = the rows drawn and sent to the gimbal (x1, y1, x2, y2, id, conf, class[, det index])
    -> the same rows with refined person boxes (a new list), or the very same list when off / not usable. Never raises."""
    if BR['mode'] == 'off' and BR['blend_s'] <= 0:
        if 'obj' in BR: BR.pop('obj', None)   # nothing runs while off: a later mode starts from scratch, not from the state of before
        return rows
    t0 = time.perf_counter()
    try:
        with BR['lock']:
            if _e2e_on(): BR['why'] = '엔드투엔드 사용 중: 검출 상자와 관절이 없어 그대로 둠'; BR['tr'] = {}; BR.pop('obj', None); return rows
            if BR['mode'] == 'gru' and not _det_kpt():
                BR['why'] = '관절 없는 검출 모델: 학습 GRU 쉼 (그대로 둠)'; BR.pop('obj', None); return rows
            key = _reset_key(); obj = BR.get('obj')
            if obj is None or not _same_key(key, BR['reset_key']) or obj.mode != BR['mode'] or (BR['mode'] == 'gru' and obj.model is not BR['model']):
                obj = BR['obj'] = Refiner(BR['mode'], BR['model'], EURO, BR['blend_s'], SIZE); BR['reset_key'] = key
            obj.blend_s = BR['blend_s']; obj.aim_y = _aim_y()   # a new blend time keeps every person's state
            if BR['mode'] == 'gru' and obj.model is None:
                BR['why'] = 'GRU 가중치 파일이 없어 그대로 둠'; return rows
            kpd = _kp_cur() if BR['mode'] == 'gru' else None; items = []; idx = []
            for i, r in enumerate(rows):
                if int(r[6]) != PERSON_CLS: continue
                bx = (float(r[0]), float(r[1]), float(r[2]), float(r[3])); zr = np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2, max(bx[2] - bx[0], 1.), max(bx[3] - bx[1], 1.)])
                z = zr
                if dets is not None and len(r) > 7:   # the detection the tracker matched this frame (the tracker's own box is the Kalman posterior)
                    j = int(r[7])
                    if 0 <= j < len(dets):
                        d = dets[j]; d0, d1, d2, d3 = float(d[0]), float(d[1]), float(d[2]), float(d[3])
                        iw = max(0., min(bx[2], d2) - max(bx[0], d0)); ih = max(0., min(bx[3], d3) - max(bx[1], d1)); inter = iw * ih
                        if inter / max((bx[2] - bx[0]) * (bx[3] - bx[1]) + (d2 - d0) * (d3 - d1) - inter, 1e-6) >= 0.3:
                            bx = (d0, d1, d2, d3); z = np.array([(d0 + d2) / 2, (d1 + d3) / 2, max(d2 - d0, 1.), max(d3 - d1, 1.)])
                kp = None
                if kpd is not None:
                    b = kpd['boxes']; iw = np.clip(np.minimum(b[:, 2], bx[2]) - np.maximum(b[:, 0], bx[0]), 0, None); ih = np.clip(np.minimum(b[:, 3], bx[3]) - np.maximum(b[:, 1], bx[1]), 0, None)
                    iou = iw * ih / np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - iw * ih, 1e-6); j = int(np.argmax(iou))
                    if iou[j] >= KP_IOU: kp = np.asarray(kpd['kpts'][j], np.float64)
                items.append((int(r[4]), z, kp, zr)); idx.append(i)
            res = obj.step(items, float(t) if t is not None else time.perf_counter(), _warp(), _app('args').det if _app('args') is not None else None, _new_ready())
            out = list(rows)
            for i, it in zip(idx, items):
                b = res.get(it[0])
                if b is None or not np.isfinite(b).all(): continue
                r = list(rows[i]); r[0], r[1], r[2], r[3] = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2; out[i] = r
            BR['why'] = ''; BR['frames'] += 1; BR['n'] = len(items); BR['blends'] = obj.blends; BR['handovers'] = obj.handovers; BR['errors'] = 0
            ms = (time.perf_counter() - t0) * 1000.0; BR['ms'] += 0.05 * (ms - BR['ms']); BR['ms_max'] = max(BR['ms_max'] * 0.999, ms)
            if items: BR['ms_track'] += 0.05 * (ms / len(items) - BR['ms_track'])
            return out
    except Exception as e_:
        off = False
        with BR['lock']:
            BR['err'] = repr(e_)[:160]; BR['errors'] += 1; BR['errors_all'] += 1; BR.pop('obj', None)
            if BR['errors'] >= ERR_MAX: BR.update(mode='off', blend_s=0.0, why='%d번 잇달아 오류: 스스로 끔' % ERR_MAX); off = True
        if off:
            print('BOX_REFINE off after %d errors in a row: %s' % (ERR_MAX, BR['err']), flush=True)
            sel_ = _app('sel_save')   # the failing mode does not come back after a restart
            if sel_ is not None: sel_(box_refine='off', box_blend=0.0)
        return rows


def state():
    m = scan() if BR['mode'] == 'gru' or BR['model'] is None else BR['model']
    obj = BR.get('obj'); kpt = _det_kpt()
    return dict(mode=BR['mode'], mode_name=MODE_KO.get(BR['mode']), blend_s=BR['blend_s'],
                modes=[dict(key=k, name=MODE_KO[k], available=(k != 'gru' or (m is not None and _det_kpt_soon()))) for k in MODES],
                gru_file=BR['file'], gru_note=BR['note'] or ('' if kpt else '관절을 내는 검출 모델에서만'), euro=dict(EURO), size=dict(SIZE),
                ms=round(BR['ms'], 3), ms_max=round(BR['ms_max'], 3), ms_track=round(BR['ms_track'], 3),
                tracks=len(obj.tr) if obj is not None else 0, people=BR['n'], frames=BR['frames'], blends=BR['blends'], handovers=BR['handovers'], why=BR['why'],
                err=BR['err'], errors=BR['errors'], errors_all=BR['errors_all'])


def restore_queries(sel):
    """startup (app.py): the saved choices as /rules queries, one per key, so a refused mode (no weight file, no keypoint detector) keeps the blend time."""
    return [{k_: [str(sel[k_])]} for k_ in ('box_refine', 'box_blend') if k_ in sel]


def parse_request(q_):
    """q_: box_refine / box_blend keys (parse_qs lists). Everything is checked first. -> (ok, text, apply)."""
    mode = None; blend = None
    if 'box_refine' in q_:
        v_ = str(q_['box_refine'][0]).strip().lower()
        if v_ not in MODES: return False, '상자 다듬기는 off, euro, gru, size 가운데 하나', None
        if v_ == 'gru' and scan() is None: return False, 'GRU 상자 다듬기 가중치 파일이 없음 (models/track/box_refine_gru_*.npz)', None
        if v_ == 'gru' and not _det_kpt_soon(): return False, '학습 GRU 는 관절을 내는 검출 모델([관절])을 먼저 고르세요', None
        mode = v_
    if 'box_blend' in q_:
        try: blend = float(q_['box_blend'][0])
        except (ValueError, IndexError): return False, '섞기 시간은 숫자(초)', None
        if not 0.0 <= blend <= BLEND_MAX: return False, '섞기 시간은 0 ~ %g 초 (0 = 끔)' % BLEND_MAX, None

    def apply_():
        with BR['lock']:   # a new mode makes apply() start a new state machine; a new blend time keeps every person's state
            if mode is not None: BR['mode'] = mode
            if blend is not None: BR['blend_s'] = blend
            BR['err'] = ''; BR['errors'] = 0; BR['why'] = ''; BR['ms'] = 0.0; BR['ms_max'] = 0.0   # no status of the earlier choice stays on the page
            if BR['mode'] == 'gru': scan()
        print('RULES set box_refine %s blend %g s' % (BR['mode'], BR['blend_s']), flush=True)
        sel_ = _app('sel_save')
        if sel_ is not None: sel_(box_refine=BR['mode'], box_blend=BR['blend_s'])
    return True, '적용됨', apply_
