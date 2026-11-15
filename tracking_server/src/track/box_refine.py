"""사람 상자 다듬기 (2026-11-15, 사용자 "2번부터 하자", "변할 때 자연스럽게 변해야 한다, 확확 변하면 안 된다", "연속적인 네모 안정적으로 변하도록"). 기본은 꺼짐.
11-16 (사용자 "상자 다듬기에 뭐가 많은데 필요한 것만 남기자"): 끔 / 크기만 부드럽게 / 상체 네모로 통일 세 가지만 남김. 11-15 에 해 본 One Euro(euro),
학습 GRU 1판(gru), 2판(gru2)은 뺐다: euro 는 짐벌 조준점 떨림을 키웠고, GRU 두 판은 우리 카메라에서 단순한 변환표 + 크기 거르기를 이기지 못했다
(파일과 코드는 git 기록에 있음). 저장된 선택이 그 셋 가운데 하나면 시작할 때 끔으로 바꾸고 한 줄 남긴다.

무엇을: 화면에 그리는 사람 상자와 짐벌에 보내는 사람 상자만 다듬는다. page_handler.live_publish 가 그리기와 짐벌 보내기 바로 전에 apply 를 부른다.
추적기 안의 상자(Kalman 상태), 짝짓기, KeepID, 녹화 채점, 검출 모델 자동 선택, 측정 기록, 피부 온도는 그대로 원래 상자를 쓴다.

모드 (/rules/set?box_refine=off|size|body, last_selection.json 에 남음):
  off   아무것도 안 함 (기본). 이때 apply 는 받은 행을 그대로 돌려주고 아무 상태도 만들지 않는다(섞기 시간이 0 일 때).
  size  크기만 부드럽게 (11-15 v2, 장치에서 쓰는 모드). 위치는 추적기 행 그대로(SIZE['anchor']: 'top' = 위 가장자리와 가로 중심, 'centre' = 중심,
        'aim' = 짐벌 조준점 위 + aim_y x 키 를 추적기 행과 똑같이 둠(aim_y = 돌고 있는 짐벌 프로그램의 --aim-y, 없으면 --gimbal-aim-y, 없으면 0.5):
        짐벌 조준은 지금과 같고 그리는 크기만 바뀜), 상자 크기(log 가로, log 세로)만 One Euro 필터로 거른다. dt = 실제 초, 거르기 전에 카메라
        확대/축소(그 번호의 앞 프레임 뒤 카메라 움직임의 log sqrt|det H|)를 상태에 더해 카메라 배율은 거르지 않는다. 검출 모델이 바뀌어도(근거리 <-> 원거리,
        상자 정의가 다름) 상태를 그대로 두어 바뀐 크기가 시간에 걸쳐 퍼진다. 값(SIZE)은 공개 자료 val-A 에서 미리 적은 규칙으로 고른 것
        (pos_refine_20261115/v2/PREREG_V2.json, v2/v3/PREREG_V3.json).
  body  상체 네모로 통일 (11-15 밤, 사용자: 그리는 / 짐벌에 보내는 사람 상자는 가까이든 멀리든 한 모양 = 머리 위 ~ 골반, 검출 모델은 안 바꿈).
        추적기 행을 변환표(models/track/box_body_table_*.npz, 이름순 마지막; 상자 높이 x 가로세로 비 -> 몸 네모, 겹선형 보간; 이미 몸 네모인 상자는
        그대로)로 바꾸고, 크기(log 가로, log 세로)만 One Euro 로 거른다(값은 파일의 euro, 없으면 BODY). 위치: 변환한 상자의 위 가장자리와 가로 중심.
        dt = 실제 초, 카메라 배율은 거르지 않음, 검출 모델이 바뀌어도 상태 유지, 번호 넘겨받기·다시 시작 규칙은 size 와 같다.
섞기 (/rules/set?box_blend=초, 0 = 끔, 기본 끔): 검출 모델이 바뀌면(자동 선택, 두 모델 상주) 사람마다, 그 사람의 앞 출력이 앞 모델에서 나왔고
  3 프레임 안이며 0.15 초(낮은 프레임 속도에서는 그 사람의 앞 프레임 간격 x 2) 안이면, 앞 출력(카메라 움직임만큼 옮기고, 카메라 몫을 뺀 그 사람의
  마지막 한 프레임 움직임을 초당으로 바꿔 지난 시간만큼 앞으로 옮김)과 새 출력의 차이(중심 / 키, log 크기)를 시작할 때의 섞기 시간 동안 1 에서 0 으로
  줄여 가며 더한다. 옛 바꾸기 길(do_model_swap)처럼 0.5 초 넘게 프레임이 멈춘 뒤에는 섞지 않는다(멈추기 전 자리로 끌려가지 않게).
  size 모드에서는 크기만 섞고 위치는 추적기 행(또는 조준점) 그대로 둔다. 두 모델 상주의 즉시 바꾸기에서 옛 모델에 이미 넣은 한 장은 옛 모델 결과이므로
  새 모델 결과가 실제로 나온 프레임부터 새 모델로 본다(resident_det RES.watch). 섞기 시간만 바꾸면 사람마다 상태는 그대로 둔다. 'off' 에서 섞기를
  켜면 추적기 상자에 바뀔 때 섞기만 한다(11-15 그대로).

사람마다 상태: 번호별. 그 번호가 이번 프레임에 없으면 카메라 움직임을 모으고 기다린다. 30 프레임(추적기 buffer) 넘게 없으면(빠진 프레임 > 30),
또는 새 검출이 카메라 움직임으로 옮긴 앞 검출에서 키의 1.5배 넘게 떨어지면(다른 사람) 처음부터.
번호가 바뀌어도(KeepID 다시 붙이기) 새 번호에 이어받을 상태가 없거나 앞 프레임에 안 보였으면, 앞 프레임에 보였다가 이번에 없어진 번호 가운데
마지막 검출(카메라 움직임으로 옮김)과 IoU 0.5 이상인 것의 상태를 넘겨받는다. 추적기가 새로 만들어지면(녹화 다시 시작, 엔드투엔드 켜고 끔, 추적기 바꿈)
모두 비운다. 같은 규칙을 공개 자료 채점도 이 클래스로 그대로 쓴다.
쓰지 않는 곳: 엔드투엔드 모델 사용 중(검출 상자가 없음: 꺼진 것처럼 그대로), 녹화를 다른 추적기 결과로 재생할 때 그리는 행(재생 쪽 행은 그대로).
"""
import math
import os
import sys
import threading
import time

import numpy as np

from ..common.state import APP
from ..common.shared_values import PERSON_CLS

MODES = ('off', 'size', 'body')
MODE_KO = dict(off='끔', size='크기만 부드럽게', body='상체 네모로 통일 (변환표 + 크기 부드럽게)')
REMOVED_MODES = ('euro', 'gru', 'gru2')   # 11-16: tried on 11-15, removed (a saved choice of these becomes 'off' at startup)
TABLE_PREFIX = 'box_body_table_'          # body mode's own table file: models/track/box_body_table_*.npz, name order last wins (a later file needs no code change)
# body mode: size-only One Euro after the lookup conversion of the tracker row: min_cutoff Hz, beta, derivative cutoff Hz (the size part of the tuned
# 'lookup + One Euro' of pos_refine_20261115/gru_v2/results/valA_selection.json). The table file may carry its own values ('euro').
# anchor 'top' (the converted box's top edge and centre x stay, the filtered size hangs from the head top) or 'centre' (the converted box's centre stays).
BODY = dict(mc_s=0.4, beta_s=0.0, dcut=1.0, anchor='top')
# size mode (v2, pos_refine_20261115/v2/PREREG_V2.json): One Euro on log w / log h, min_cutoff Hz, beta (per log unit per s), derivative cutoff Hz;
# anchor 'top' (top edge + centre x of the tracker row), 'centre' or 'aim'; src 'det' (the matched detection's size) or 'trk' (the tracker row's size).
# v3 (v2/v3/PREREG_V3.json, v2/v3/results/aim_valA.json): anchor 'aim' keeps the gimbal aim point top + aim_y x h of the tracker row exactly; the grid and
# the rule pick (trk, 5 Hz, beta 3), the binding limit is the DanceTrack size lag.
SIZE = dict(mc_s=5.0, beta_s=3.0, dcut=1.0, anchor='aim', src='trk')
BLEND_MAX = 2.0
BLEND_RECENT = 3       # an id's previous output at most this many frames old starts a detector-switch blend for that id
BLEND_RECENT_S = 0.15  # ... and at most this many seconds old (or 2 x the id's own last frame interval, low frame rates): after a stopped swap
                       # (do_model_swap, > 0.5 s without frames) the old output is where the person was before the stop: no blend from it
AIM_Y = 0.5            # gimbal aim point (fraction of the box height from its top) when the running bridge does not say (size anchor 'aim')
LOST_MAX = 30          # frames a missing id keeps its state (ByteTrack buffer): reset when more than LOST_MAX frames are missing
JUMP_H = 1.5           # a detection farther than this many box heights from the id's previous one (moved by the camera) = another person
HANDOVER_IOU = 0.5     # a new id takes the state of an id that vanished this frame when its detection overlaps that id's last one this much
ERR_MAX = 50           # errors in a row before the module switches itself off

BR = dict(mode='off', blend_s=0.0, lock=threading.Lock(), frame=0, ms=0.0, ms_max=0.0, ms_track=0.0, n=0, frames=0, err='', errors=0, errors_all=0,
          reset_key=None, blends=0, handovers=0, why='', dir=None, table=None, tfile=None, tnote='', tmt=None)


# ------------------------------------------------------------------------------------------------ pieces (no APP)
def _alpha(cut, dt): return 1.0 / (1.0 + 1.0 / (2 * math.pi * cut * dt))


def log_scale(H):
    """log sqrt|det| of a 2x3 partial affine warp (the camera zoom between two frames), 0 for None."""
    if H is None: return 0.0
    return math.log(max(math.sqrt(abs(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])), 1e-6))


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


class BodyTable:
    """the body mode's conversion of one models/track/box_body_table_*.npz: tab [n_h, n_ar, 4] (dx / h, dy / h, log w ratio, log h ratio) at the bin
    centres lh_cen (log box height px) x ar_cen (w / h), read with bilinear interpolation (clamped to the centre range); euro = (mc_s, beta_s, dcut)."""
    def __init__(self, w):
        g = lambda k: np.asarray(w[k], np.float64)
        self.tab, self.lh_cen, self.ar_cen = g('tab'), g('lh_cen'), g('ar_cen'); assert self.tab.shape == (len(self.lh_cen), len(self.ar_cen), 4)
        assert (np.diff(self.lh_cen) > 0).all() and (np.diff(self.ar_cen) > 0).all()
        e = tuple(float(v) for v in np.asarray(w['euro']).ravel()) if 'euro' in w else None; assert e is None or len(e) == 3
        self.body = dict(mc_s=e[0], beta_s=e[1], dcut=e[2]) if e is not None else None   # body mode's size filter, when the file gives one
        d = self.conv_delta(np.array([[320., 256., 20., 40.], [320., 256., 180., 300.]])); assert d.shape == (2, 4) and np.isfinite(d).all()

    def conv_delta(self, z):
        """the table read with bilinear interpolation between the bin centres -> [N,4] (dx / h, dy / h, log w ratio, log h ratio)."""
        tab, lh_cen, ar_cen = self.tab, self.lh_cen, self.ar_cen; lh = np.log(np.maximum(z[:, 3], 1.)); ar = z[:, 2] / np.maximum(z[:, 3], 1.)
        i = np.clip(np.searchsorted(lh_cen, lh, 'right') - 1, 0, len(lh_cen) - 2); t = np.clip((lh - lh_cen[i]) / (lh_cen[i + 1] - lh_cen[i]), 0., 1.)[:, None]
        j = np.clip(np.searchsorted(ar_cen, ar, 'right') - 1, 0, len(ar_cen) - 2); u = np.clip((ar - ar_cen[j]) / (ar_cen[j + 1] - ar_cen[j]), 0., 1.)[:, None]
        return (1 - t) * (1 - u) * tab[i, j] + (1 - t) * u * tab[i, j + 1] + t * (1 - u) * tab[i + 1, j] + t * u * tab[i + 1, j + 1]


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
    """the per-id state machine (no APP): step(list of (id, z = matched detection cxcywh[, the row's own box cxcywh]), t s, H = this frame's
    camera warp or None, det_key, new_det_ready) -> {id: box cxcywh}. Mode 'off' (blend only) and 'size' take the position from the row's own box,
    'body' converts the row's own box with the table. The state of id k after a step is self.tr[k] ('k' = rows since that state started,
    'new' = started on this frame, 'hs' = camera warp since the id's previous frame or None)."""
    def __init__(self, mode, table=None, blend_s=0.0, size=None, aim_y=AIM_Y, body=None):
        self.mode = mode; self.table = table; self.size = dict(SIZE, **(size or {})); self.blend_s = float(blend_s)
        self.aim_y = float(aim_y)   # the gimbal's aim point (size anchor 'aim'); apply() sets the running bridge's value every frame
        self.tr = {}; self.frame = 0; self.eff_key = None; self.blends = 0; self.handovers = 0
        self.body = dict(BODY, **((table.body if table is not None and getattr(table, 'body', None) else None) or {}))   # the file's values win over BODY
        if body: self.body.update(body)                                                                                 # ... and an explicit choice over both (scoring)

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
        work = []
        fresh = lambda z_: dict(f=fr, t=t, z=z_, se=None, Hacc=None, out=None, out1=None, f1=None, t1=None, bl=None, dk=None, k=-1, new=True, hs=None)
        for it in items:   # 1) every person: whose state (hand-over, reset)
            tid, z = it[0], it[1]; zr = it[2] if len(it) > 2 and it[2] is not None else z; st = self.tr.get(tid)
            if gone and (st is None or st['f'] != fr - 1):   # hand-over: KeepID gave the person another id this frame
                best, bi = HANDOVER_IOU, -1
                for j, g in enumerate(gone):
                    sg = self.tr[g]; v = iou_c(z, sg['z'] if H is None else warp_box(sg['z'], H))
                    if v >= best: best, bi = v, j
                if bi >= 0:
                    g = gone.pop(bi); st = self.tr[tid] = self.tr.pop(g); self.handovers += 1
            Hs = None   # camera motion since this id's previous frame (its missing frames included)
            if st is not None:
                Hs = H if st['Hacc'] is None else (st['Hacc'] if H is None else compose(H, st['Hacc']))
                if fr - st['f'] - 1 > LOST_MAX or self._jump(st, z, Hs): st = None
            if st is None:
                st = self.tr[tid] = fresh(z); Hs = None
            else:
                st['new'] = False
            work.append((tid, z, zr, st, Hs))
        conv = None
        if self.mode == 'body' and self.table is not None and work:   # body: the lookup conversion of every tracker row in one call
            ZR = np.array([w_[2] for w_ in work], np.float64).reshape(-1, 4); ZR[:, 2:] = np.maximum(ZR[:, 2:], 1.); d_ = self.table.conv_delta(ZR)
            conv = np.c_[ZR[:, 0] + d_[:, 0] * ZR[:, 3], ZR[:, 1] + d_[:, 1] * ZR[:, 3] - ZR[:, 3] * np.exp(d_[:, 3]) / 2, np.log(ZR[:, 2]) + d_[:, 2], np.log(ZR[:, 3]) + d_[:, 3]].tolist()   # cx, top, log w, log h
        for i, (tid, z, zr, st, Hs) in enumerate(work):   # 2) every person: the box
            if conv is not None:   # centre x, top edge of the converted box; (log w, log h) of it through the size filter
                p = self.body; cx_, top_, lw, lh = conv[i]; hc_ = math.exp(lh)
                if st['se'] is None: st['se'] = SizeEuro(p['mc_s'], p['beta_s'], p['dcut'])
                lw, lh = st['se'].step(lw, lh, t - st['t'] if st['out'] is not None else 0.04, log_scale(Hs)); h_ = math.exp(lh)
                b = np.array([cx_, top_ + (h_ if p.get('anchor', 'top') == 'top' else hc_) / 2, math.exp(lw), h_])
            elif self.mode == 'size':
                p = self.size; src = z if p['src'] == 'det' else zr
                if st['se'] is None: st['se'] = SizeEuro(p['mc_s'], p['beta_s'], p['dcut'])
                lw, lh = st['se'].step(math.log(max(src[2], 1.)), math.log(max(src[3], 1.)), t - st['t'] if st['out'] is not None else 0.04, log_scale(Hs))
                b = size_box(zr, lw, lh, p['anchor'], self.aim_y)
            else:   # off (blend only), or body without a table: the box the tracker draws today
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
        return out


# ------------------------------------------------------------------------------------------------ server glue
def _app(name, default=None):
    return vars(APP).get(name, default)


def model_dir():
    return BR['dir'] or os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'models', 'track')


def scan_table():
    """newest models/track/box_body_table_*.npz (the body mode's table; read again only when it changed and is older than 5 s). Name order last wins,
    so a later file dropped into the folder is taken without a code change. An unreadable newest file is said so (not silently an older one)."""
    d = model_dir()
    try: c_ = sorted(f for f in os.listdir(d) if f.startswith(TABLE_PREFIX) and f.endswith('.npz'))
    except OSError: c_ = []
    if not c_: BR.update(table=None, tfile=None, tnote='변환표 파일 없음', tmt=None); return None
    p = os.path.join(d, c_[-1])
    try: mt = os.path.getmtime(p)
    except OSError: return None
    if BR['tfile'] == c_[-1] and BR['tmt'] == mt: return BR['table']
    if time.time() - mt < 5: return BR['table']
    try:
        z = np.load(p, allow_pickle=False); m = BodyTable({k: z[k] for k in z.files})
        BR.update(table=m, tfile=c_[-1], tmt=mt, tnote='')
    except Exception as e_:
        BR.update(table=None, tfile=c_[-1], tmt=mt, tnote='변환표 파일을 읽지 못함: %s' % type(e_).__name__)
    return BR['table']


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
            if _e2e_on(): BR['why'] = '엔드투엔드 사용 중: 검출 상자가 없어 그대로 둠'; BR.pop('obj', None); return rows
            mode = BR['mode']; table = BR['table'] if mode == 'body' else None
            key = _reset_key(); obj = BR.get('obj')
            if obj is None or not _same_key(key, BR['reset_key']) or obj.mode != mode or obj.table is not table:
                obj = BR['obj'] = Refiner(mode, table, BR['blend_s'], SIZE); BR['reset_key'] = key
            obj.blend_s = BR['blend_s']; obj.aim_y = _aim_y()   # a new blend time keeps every person's state
            if mode == 'body' and table is None:
                BR['why'] = '변환표 파일이 없어 그대로 둠'; return rows
            items = []; idx = []
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
                            z = np.array([(d0 + d2) / 2, (d1 + d3) / 2, max(d2 - d0, 1.), max(d3 - d1, 1.)])
                items.append((int(r[4]), z, zr)); idx.append(i)
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
    m = scan_table() if BR['mode'] == 'body' or BR['table'] is None else BR['table']
    obj = BR.get('obj')
    return dict(mode=BR['mode'], mode_name=MODE_KO.get(BR['mode']), blend_s=BR['blend_s'],
                modes=[dict(key=k, name=MODE_KO[k], available=(m is not None) if k == 'body' else True) for k in MODES],
                size=dict(SIZE), body=dict(BODY, **((m.body if m is not None else None) or {})), body_file=BR['tfile'], body_note=BR['tnote'],
                ms=round(BR['ms'], 3), ms_max=round(BR['ms_max'], 3), ms_track=round(BR['ms_track'], 3),
                tracks=len(obj.tr) if obj is not None else 0, people=BR['n'], frames=BR['frames'], blends=BR['blends'], handovers=BR['handovers'], why=BR['why'],
                err=BR['err'], errors=BR['errors'], errors_all=BR['errors_all'])


def restore_queries(sel):
    """startup (app.py): the saved choices as /rules queries, one per key, so a refused mode (no table file) keeps the blend time.
    A saved mode that was removed on 11-16 (euro, gru, gru2) or is unknown comes back as 'off' (said in one line; the next save writes 'off')."""
    qs = []
    if 'box_refine' in sel:
        v_ = str(sel['box_refine']).strip().lower()
        if v_ not in MODES:
            print('BOX_REFINE saved mode %r %s: off' % (sel['box_refine'], 'was removed on 11-16' if v_ in REMOVED_MODES else 'is unknown'), flush=True); v_ = 'off'
        qs.append({'box_refine': [v_]})
    if 'box_blend' in sel: qs.append({'box_blend': [str(sel['box_blend'])]})
    return qs


def parse_request(q_):
    """q_: box_refine / box_blend keys (parse_qs lists). Everything is checked first. -> (ok, text, apply)."""
    mode = None; blend = None
    if 'box_refine' in q_:
        v_ = str(q_['box_refine'][0]).strip().lower()
        if v_ not in MODES: return False, '상자 다듬기는 off, size, body 가운데 하나', None
        if v_ == 'body' and scan_table() is None: return False, '상체 네모 변환표 파일이 없음 (models/track/%s*.npz)' % TABLE_PREFIX, None
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
            if BR['mode'] == 'body': scan_table()
        print('RULES set box_refine %s blend %g s' % (BR['mode'], BR['blend_s']), flush=True)
        sel_ = _app('sel_save')
        if sel_ is not None: sel_(box_refine=BR['mode'], box_blend=BR['blend_s'])
    return True, '적용됨', apply_
