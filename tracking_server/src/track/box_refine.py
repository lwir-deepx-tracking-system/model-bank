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
  body  상체 네모로 통일 (11-15 밤 첫 판, 11-16 2판; 사용자: 그리는 / 짐벌에 보내는 사람 상자는 가까이든 멀리든 한 모양 = 머리 위 ~ 골반,
        검출 모델은 안 바꿈, 확확 변하면 안 되고 짐벌 조준이 더 늦거나 떨리면 안 됨). 규칙은 pos_refine_20261115/body_v2/PREREG_BODY_V2.json.
        1) 바꾸기: 추적기 행을 표 파일(models/track/box_body_table_*.npz, 이름순 마지막)로 몸 네모로 바꾼다. 검출 모델 계열은 돌고 있는 검출 모델
           이름으로 정한다: 이름에 'bodybox' 가 있으면 우리 자세 검출(100 px 위는 몸 네모, 아래는 전신 상자) -> 변환표(상자 높이 x 가로세로 비,
           겹선형 보간; 몸 네모 모양과 100 px 위는 그대로, 작은 전신 상자는 약 절반 키). 그 밖의 검출 모델(YOLO / FLIR 등, 늘 전신 상자) -> 파일의
           person 고정 비율(위 + top_off x h 에서 키 k_h x h, 폭 k_w x h; 학습 분할에서 잰 0.535, 0.331).
        2) 모델이 바뀔 때(자동 선택, 두 모델 상주, 화면): 사람마다 앞 모델의 바꾸기 값에서 새 모델의 값으로, 추적기 행의 키가 앞 정의에서 새 정의로
           옮겨 가는 만큼(칼만이 몇 프레임 걸림) 섞어 넘어간다(최대 1초, 새 검출로 두 정의가 같다고 보이면 바로 끝). 그래서 키와 조준점이 이어진다.
           그 사람이 검출과 짝지어지지 않은 프레임(행 = 칼만 예측, 아직 앞 정의)은 아무것도 알려 주지 않으므로 넘어가기를 끝내지도 나아가지도 않는다
           (11-16 검토: 바뀌는 프레임에 짝이 없으면 바로 끝나 키가 0.535 배로 튀었음).
        표 파일에 NaN / inf 가 있거나 읽지 못하면 쓰지 않고 그 까닭을 화면(body_note, 잠금 글)·거절 글·why 에 보인다.
        3) 크기만 One Euro(BODY / 파일의 euro: val-A 에서 미리 적은 규칙으로 고른 1.5 Hz, beta 0.3). 위치: 기본 'aim' = 바꾼(거르기 전) 상자의
           짐벌 조준점(위 + aim_y x 키, aim_y 는 size 와 같이 돌고 있는 짐벌 프로그램에서 읽음)과 가로 중심을 그대로 두고 거른 크기를 그 둘레에 놓음:
           짐벌 조준(떨림, 늦음)은 바꾼 상자의 것과 똑같다. 'top'(첫 판), 'centre' 도 고를 수 있다.
        dt = 실제 초, 카메라 배율은 거르지 않음, 번호 넘겨받기·다시 시작 규칙은 size 와 같다. 섞기(box_blend)는 쓰지 않는다(2)가 대신함).
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
모두 비운다. 같은 규칙을 공개 자료 채점도 이 클래스로 그대로 쓴다. NaN / inf 행(모든 모드)은 그 번호가 그 프레임에 없는 것처럼 두고(그 행은 그대로 그림,
상태는 그대로), 결과나 크기 필터 상태가 NaN / inf 가 되면 그 번호를 처음부터 한다(11-16 검토).
쓰지 않는 곳: 엔드투엔드 모델 사용 중(검출 상자가 없음: 꺼진 것처럼 그대로), 녹화를 다른 추적기 결과로 재생할 때 그리는 행(재생 쪽 행은 그대로).

놓친 상자 = 보일 때 상자 (11-16, /rules/set?lost_same=1|0, last_selection.json 에 남음, 기본 0 = 지금 그대로; 판정
pos_refine_20261115/unified/judge/JUDGE_RESULT_20261116.md 의 A = U0): 화면의 점선 상자(놓친 사람, 예측 상자 표시)를 보일 때 그린 상자와 같은
정의로 그린다. 표시만: 추적기, 위치 기억, 짝짓기, KeepID, 짐벌은 그대로(추적기는 읽기만 한다).
  - 놓친 첫 장부터 그린다. 지금은 점선 상자가 다음 프레임 예측(POSP['boxes'])에서 만들어져 놓친 첫 장이 비었다. 켜면 이 프레임의 추적기 갱신
    뒤 놓친 목록(tracker.lost: 상자 = t.mean[:4], 놓친 장 수 = tracker.frame_id - t.end_frame)을 읽는다(web/lost_marks.pool_now).
    파이썬 ByteTrack / BoT-SORT / OC-SORT / SORT 와 C++ BoT-SORT / OC-SORT / SORT 는 같은 목록을 내므로(test_native_trackers) 장치 설정
    (botsort, C++)에서도 그대로 된다. 카메라 C++ ByteTrack(camera_bytetrack, 번호가 다름)과 엔드투엔드는 읽을 수 없어 지금 방식으로 돌아간다.
  - 크기: 번호마다 그 번호가 마지막으로 보인 프레임의 거른 크기(size / body 필터 상태 [log w, log h])를 따로 적어 두고(LostHold: 이 모듈의
    KeepID 넘겨받기가 놓친 번호의 필터 상태를 다른 번호에 넘겨도 남게), 안 보이는 프레임마다 카메라 배율만 더한다. 90 프레임 뒤, 모드가 바뀌거나
    다시 시작하면 버린다.
  - size: 가로 중심과 짐벌 조준점(위 + aim_y x 키)은 추적기 놓친 상자 그대로, 크기만 적어 둔 크기(보일 때의 size_box 와 같은 규칙).
    body: 보일 때와 같은 변환표(그 프레임의 놓친 사람 모두를 한 번에)와 같은 기준점(aim / top / centre), 크기는 적어 둔 크기.
    변환표 계열은 그 번호가 마지막으로 보인 프레임의 검출 모델 것: 놓친 동안 검출 모델이 바뀌어도(자동 선택은 사람을 잃고 0.5 초, 가려졌으면 3 초 뒤 원거리로 바꿈)
    추적기 놓친 상자는 앞 모델 상자 정의 그대로이므로 점선 상자도 그대로 둔다. 그 프레임에 모델 바뀜 넘어가기가 돌고 있었으면 그 진행 그대로
    섞는다(보일 때 검출과 짝지어지지 않은 행과 같은 규칙; 11-16 검토: 자세 <-> 일반 검출 바뀜에서 점선 상자가 키의 0.25 만큼 튀었음).
    off: 추적기 상자 그대로(지금과 같음; 켜면 놓친 첫 장의 빈 장만 없어진다).
  - 오류가 나면 그 프레임은 추적기 상자를 그리고 센다(lost_errors). 보일 때 상자(apply 결과)는 켜도 끄도 비트까지 같다.
"""
import math
import os
import sys
import threading
import time

import numpy as np

from ..common.state import APP
from ..common import frame_math
from ..common.shared_values import PERSON_CLS

MODES = ('off', 'size', 'body')
MODE_KO = dict(off='끔', size='크기만 부드럽게', body='상체 네모로 통일 (변환표 + 크기 부드럽게)')
REMOVED_MODES = ('euro', 'gru', 'gru2')   # 11-16: tried on 11-15, removed (a saved choice of these becomes 'off' at startup)
TABLE_PREFIX = 'box_body_table_'          # body mode's own table file: models/track/box_body_table_*.npz, name order last wins (a later file needs no code change)
# body mode (v2, 11-16, pos_refine_20261115/body_v2/PREREG_BODY_V2.json): size-only One Euro after the conversion of the tracker row: min_cutoff Hz,
# beta (per log unit per s), derivative cutoff Hz, chosen on val-A by the pre-registered rule; the table file carries the chosen values ('euro', they win).
# anchor 'aim' (default): the CONVERTED box's gimbal aim point top + aim_y x h and centre x stay, the filtered size is placed around it, so the
# gimbal aims exactly where the converted (unfiltered) row aims; 'top' (the 11-16 first version: the converted top edge stays) and 'centre' selectable.
# sw: the detector-switch transition (below) on / off (off = the first version, only for scoring).
BODY = dict(mc_s=1.5, beta_s=0.3, dcut=1.0, anchor='aim', sw=True)   # = the committed file's euro (val-A choice, body_v2/results/valA_grid.json)
# detector families (body): our pose detectors give a body box (head top to hips) from about 100 px up and a whole-person box below, which the table
# maps by box height x aspect; every other detector (plain YOLO / YOLOX / FLIR models) gives a whole-person box at every size, which the file's
# 'person' constant maps (top + top_off x h, height k_h x h, width k_w x h). The family is read from the detector key in effect: 'bodybox' in it = pose.
POSE_MARK = 'bodybox'
POSE_BODYBOX_KEYS = {'bank:yolo26m_pose_ft16v7_reviewed264_e24_provisional_kpt_640x480'}
FAMILY_KO = dict(pose='자세 검출(몸 네모) 변환표', person='사람 전체 상자의 위쪽 0.535 · 가려지거나 앉은 가까운 사람은 작게 나옴(자세 검출 모델 권장)')
# detector switch (body): per person, the conversion moves from the old detector's delta to the new detector's in step with the tracker row's own
# height moving from the old box definition to the new one (the Kalman filter needs a few frames), so the drawn box and the aim point stay continuous
SW_T = 1.0             # s: the transition ends at the latest after this long
SW_SAME = 0.15         # |log h| between the new detection and the row before the switch under this: the definitions agree, no transition
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
LOST_HOLD = 90         # lost_same: frames an id's held size is kept after its last seen frame (= the KeepID window, web/lost_marks.KEEPID_FRAMES)

BR = dict(mode='off', blend_s=0.0, lock=threading.Lock(), frame=0, ms=0.0, ms_max=0.0, ms_track=0.0, n=0, frames=0, err='', errors=0, errors_all=0,
          reset_key=None, blends=0, handovers=0, why='', dir=None, table=None, tfile=None, tnote='', tmt=None,
          lost_same=False, lost_ms=0.0, lost_ms_max=0.0, lost_n=0, lost_frames=0, lost_err='', lost_errors=0, lost_why='')   # 11-16 lost_same (default off)


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


def det_family(key):
    """the box family of a detector key ('bank:<file stem>' or an internal name): 'pose' (our body-box pose detectors) or 'person' (whole-person
    boxes at every size). No key (offline use without one) = 'pose'."""
    if key is None: return 'pose'
    name = str(key).lower()
    return 'pose' if name in POSE_BODYBOX_KEYS or POSE_MARK in name else 'person'


class BodyTable:
    """the body mode's conversion of one models/track/box_body_table_*.npz: tab [n_h, n_ar, 4] (dx / h, dy / h, log w ratio, log h ratio) at the bin
    centres lh_cen (log box height px) x ar_cen (w / h), read with bilinear interpolation (clamped to the centre range); person = (top offset, height,
    width) / person box height for the whole-person family (absent: the table for every detector, the first file); euro = (mc_s, beta_s, dcut)."""
    def __init__(self, w):
        g = lambda k: np.asarray(w[k], np.float64)
        self.tab, self.lh_cen, self.ar_cen = g('tab'), g('lh_cen'), g('ar_cen'); assert self.tab.shape == (len(self.lh_cen), len(self.ar_cen), 4)
        assert (np.diff(self.lh_cen) > 0).all() and (np.diff(self.ar_cen) > 0).all()
        assert np.isfinite(self.tab).all() and np.isfinite(self.lh_cen).all() and np.isfinite(self.ar_cen).all()   # one NaN / inf cell would poison a person's filter
        e = tuple(float(v) for v in np.asarray(w['euro']).ravel()) if 'euro' in w else None; assert e is None or (len(e) == 3 and all(map(math.isfinite, e)))
        self.body = dict(mc_s=e[0], beta_s=e[1], dcut=e[2]) if e is not None else None   # body mode's size filter, when the file gives one
        self.person = tuple(float(v) for v in np.asarray(w['person']).ravel()) if 'person' in w else None
        assert self.person is None or (len(self.person) == 3 and all(map(math.isfinite, self.person)) and 0.2 < self.person[1] < 1.0 and 0.1 < self.person[2] < 1.0)
        for fam in ('pose', 'person'):
            d = self.conv_delta(np.array([[320., 256., 20., 40.], [320., 256., 180., 300.]]), fam); assert d.shape == (2, 4) and np.isfinite(d).all()

    def conv_delta(self, z, family='pose'):
        """-> [N,4] (dx / h, dy / h, log w ratio, log h ratio) of the body box against the boxes z [N,4] cxcywh: the table read with bilinear interpolation
        between the bin centres (family 'pose', or any family when the file has no person constant), or the person constant (family 'person': centre x
        kept, top + top_off x h, height k_h x h, width k_w x h)."""
        if family == 'person' and self.person is not None:
            to, kh, kw = self.person; h = np.maximum(z[:, 3], 1.); n = len(z)
            return np.c_[np.zeros(n), np.full(n, -0.5 + to + kh / 2), np.log(kw * h / np.maximum(z[:, 2], 1.)), np.full(n, math.log(kh))]
        tab, lh_cen, ar_cen = self.tab, self.lh_cen, self.ar_cen; lh = np.log(np.maximum(z[:, 3], 1.)); ar = z[:, 2] / np.maximum(z[:, 3], 1.)
        native = frame_math.body_deltas(tab, lh_cen, ar_cen, lh, ar)
        if native is not None:
            return native
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
    'body' converts the row's own box (table or person constant by the detector family; a matched detection is z itself, an unmatched row passes
    z = the row object). The state of id k after a step is self.tr[k] ('k' = rows since that state started, 'new' = started on this frame,
    'hs' = camera warp since the id's previous frame or None, body: 'dl' the conversion delta used, 'lr' the row's log height, 'tw' a running
    detector-switch transition [delta0, log h0, t0, progress]). family_of(det_key) -> 'pose' | 'person' (det_family; scoring may replace it)."""
    def __init__(self, mode, table=None, blend_s=0.0, size=None, aim_y=AIM_Y, body=None):
        self.mode = mode; self.table = table; self.size = dict(SIZE, **(size or {})); self.blend_s = float(blend_s)
        self.aim_y = float(aim_y)   # the gimbal's aim point (size anchor 'aim'); apply() sets the running bridge's value every frame
        self.tr = {}; self.frame = 0; self.eff_key = None; self.blends = 0; self.handovers = 0
        self.body = dict(BODY, **((table.body if table is not None and getattr(table, 'body', None) else None) or {}))   # the file's values win over BODY
        if body: self.body.update(body)                                                                                 # ... and an explicit choice over both (scoring)
        self.family_of = det_family; self.switches = 0

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
        fresh = lambda z_: dict(f=fr, t=t, z=z_, se=None, Hacc=None, out=None, out1=None, f1=None, t1=None, bl=None, dk=None, k=-1, new=True, hs=None,
                                dl=None, lr=None, tw=None)
        for it in items:   # 1) every person: whose state (hand-over, reset)
            tid, z = it[0], it[1]; zr = it[2] if len(it) > 2 and it[2] is not None else z; st = self.tr.get(tid)
            if not (np.isfinite(z).all() and np.isfinite(zr).all()): continue   # a NaN / inf row = a missing frame for this id: its state waits, apply draws the row as it is
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
        D = None
        if self.mode == 'body' and self.table is not None and work:   # body: the conversion of every tracker row (and matched detection) in one call
            fam = self.family_of(self.eff_key)
            ZR = np.array([w_[2] for w_ in work], np.float64).reshape(-1, 4); ZR[:, 2:] = np.maximum(ZR[:, 2:], 1.); D = self.table.conv_delta(ZR, fam)
            DZ = D
            if any(w_[3]['tw'] is not None or (w_[3]['dk'] is not None and w_[3]['dk'] != self.eff_key) for w_ in work):   # a switch transition: the matched detections too
                Z_ = np.array([w_[1] for w_ in work], np.float64).reshape(-1, 4); Z_[:, 2:] = np.maximum(Z_[:, 2:], 1.); DZ = self.table.conv_delta(Z_, fam)
        for i, (tid, z, zr, st, Hs) in enumerate(work):   # 2) every person: the box
            if D is not None:
                p = self.body; d = D[i]; ls_ = log_scale(Hs); lr = math.log(max(zr[3], 1.))
                if p.get('sw', True):   # detector switch: follow the row's own move from the old box definition to the new one
                    if (st['dk'] is not None and self.eff_key is not None and st['dk'] != self.eff_key and st['out'] is not None and st['dl'] is not None
                            and fr - st['f'] <= BLEND_RECENT and t - st['t'] <= self._recent_s(st)):
                        st['tw'] = [np.array(st['dl']), st['lr'] + ls_, t, 0.0]; self.switches += 1
                    elif st['tw'] is not None: st['tw'][1] += ls_   # the camera zoom since the previous row moves the old height too
                    if st['tw'] is not None:
                        d0, lh0, t0, a = st['tw']; matched = z is not zr; lz = math.log(max(z[3], 1.))
                        if t - t0 >= SW_T or (matched and abs(lz - lh0) < SW_SAME): a = 1.0   # an unmatched row (z = the row) says nothing of the new definition
                        elif matched: a = max(a, min(max((lr - lh0) / (lz - lh0), 0.0), 1.0))
                        if a >= 1.0: st['tw'] = None   # the row is in the new definition: its own conversion from here on
                        else: d = (1.0 - a) * d0 + a * (DZ[i] if matched else d); st['tw'][3] = a
                st['dl'] = np.array(d); st['lr'] = lr
                hc_ = zr[3] * math.exp(d[3]); cx_ = zr[0] + d[0] * zr[3]; cy_ = zr[1] + d[1] * zr[3]; top_ = cy_ - hc_ / 2
                if st['se'] is None: st['se'] = SizeEuro(p['mc_s'], p['beta_s'], p['dcut'])
                lw, lh = st['se'].step(math.log(max(zr[2], 1.)) + d[2], math.log(hc_), t - st['t'] if st['out'] is not None else 0.04, ls_); h_ = math.exp(lh)
                an = p.get('anchor', 'aim')   # 'aim': the converted box's aim point top + aim_y x h stays; 'top': its top edge; 'centre': its centre
                tf = top_ + self.aim_y * (hc_ - h_) if an == 'aim' else top_ if an == 'top' else cy_ - h_ / 2
                b = np.array([cx_, tf + h_ / 2, math.exp(lw), h_])
            elif self.mode == 'size':
                p = self.size; src = z if p['src'] == 'det' else zr
                if st['se'] is None: st['se'] = SizeEuro(p['mc_s'], p['beta_s'], p['dcut'])
                lw, lh = st['se'].step(math.log(max(src[2], 1.)), math.log(max(src[3], 1.)), t - st['t'] if st['out'] is not None else 0.04, log_scale(Hs))
                b = size_box(zr, lw, lh, p['anchor'], self.aim_y)
            else:   # off (blend only), or body without a table: the box the tracker draws today
                b = np.asarray(zr, np.float64).copy()
            if (self.blend_s > 0 and self.mode != 'body' and st['dk'] is not None and self.eff_key is not None and st['dk'] != self.eff_key and st['out'] is not None
                    and fr - st['f'] <= BLEND_RECENT and t - st['t'] <= self._recent_s(st)):   # this person's previous output came from the other detector: blend from it
                prev = self._carry(st, Hs, t); off = blend_off(prev, b)
                if self.mode == 'size': off[:2] = 0.0   # size mode: the position stays the tracker row's; only the size is blended
                st['bl'] = (off, t, self.blend_s); self.blends += 1
            if st['bl'] is not None and self.mode != 'body':   # body has its own switch transition (above); box_blend serves off / size
                off, t0, dur = st['bl']; a = 1.0 - (t - t0) / dur   # the duration the blend started with: a new blend time does not move the box back
                if a <= 0 or self.blend_s <= 0: st['bl'] = None
                elif self.mode == 'size': b = size_box(zr, lw + a * off[2], lh + a * off[3], self.size['anchor'], self.aim_y)
                else: b = blend_apply(b, off, a)
            se = st['se']
            if not (np.isfinite(b).all() and (se is None or (math.isfinite(se.x0) and math.isfinite(se.x1) and math.isfinite(se.d0) and math.isfinite(se.d1)))):
                del self.tr[tid]; continue   # a non-finite result or filter state: the id starts again on its next row (apply draws this row as it is)
            if st['out'] is not None: st['out1'], st['f1'], st['t1'] = st['out'], st['f'], st['t']
            st.update(f=fr, t=t, z=np.asarray(z, np.float64), out=b, Hacc=None, hs=Hs, dk=self.eff_key, k=st['k'] + 1); out[tid] = b
        for tid in list(self.tr):
            if tid in out: continue
            st = self.tr[tid]
            if fr - st['f'] > LOST_MAX: del self.tr[tid]; continue
            if H is not None: st['Hacc'] = H.copy() if st['Hacc'] is None else compose(H, st['Hacc'])   # applied at the id's next frame
        return out


class LostHold:
    """lost_same (11-16, judge A = U0, no APP): per output id its own copy [log w, log h, frames unseen, detector key, switch transition] of the
    Refiner's state after the id's last seen step (filtered size, st['dk'], st['tw'] as (old delta, progress) or None), + the camera log scale of
    every frame since (unified_srv.UnifiedStore.hold). Own copy: the Refiner's KeepID hand-over (HANDOVER_IOU) may move a lost id's filter state
    to another id. Reads the Refiner, never writes it (the drawn / gimbal rows stay the same).
    step(seen ids, H) once per frame right after ref.step; boxes(items) -> (lost boxes, converted, bad) for the lost tracks of the same frame."""
    __slots__ = ('ref', 'h', 'hold_max')

    def __init__(self, ref, hold_max=LOST_HOLD):
        self.ref = ref; self.h = {}; self.hold_max = int(hold_max)

    def step(self, seen, H):
        ls = log_scale(H)
        for tid in list(self.h):
            if tid in seen: continue
            e = self.h[tid]; e[2] += 1
            if e[2] > self.hold_max: del self.h[tid]; continue
            e[0] += ls; e[1] += ls
        for tid in seen:
            st = self.ref.tr.get(tid); se = st.get('se') if st is not None else None
            if se is not None and se.x0 is not None and math.isfinite(se.x0) and math.isfinite(se.x1):
                tw = st.get('tw'); self.h[tid] = [se.x0, se.x1, 0, st.get('dk'), (tw[0], tw[3]) if tw is not None else None]

    def boxes(self, items):
        """items [(output id, tracker lost box cxcywh, lost step)] -> ([box cxcywh per item], converted, bad). A box is the tracker's own box when the
        id has no held size, in mode 'off' (blend only), in body without a table, or when the result is not finite (bad). size: centre x and the
        gimbal aim point top + aim_y x h of the tracker box stay, the size is the held one (size_box, the seen rule). body: the running body anchor
        (aim / top / centre) and the held size around the box converted as the seen path converts a row no detection updated (z = the row): with
        the family of the detector of the id's last seen step (the tracker lost box is still in that detector's box definition: a detector change
        while the id is lost does not move it; review 11-16: pose <-> plain moved it 0.25 h), and a switch transition running on that step stays
        at its progress a ((1 - a) old delta + a delta). One table call per family (all of them in ONE call when they share it, the usual case);
        sw off (the first version) converts with the running family, as its seen path."""
        ref = self.ref; out = []; use = []
        for i, it in enumerate(items):
            z = np.array(it[1], np.float64).reshape(4); out.append(z)
            if it[0] in self.h and np.isfinite(z).all(): use.append(i)
        if not use or ref.mode not in ('size', 'body') or (ref.mode == 'body' and ref.table is None): return out, 0, 0
        n = bad = 0
        if ref.mode == 'size':
            an = ref.size['anchor']
            for i in use:
                e = self.h[items[i][0]]; z = out[i]; z[2:] = np.maximum(z[2:], 1.); b = size_box(z, e[0], e[1], an, ref.aim_y)
                if np.isfinite(b).all(): out[i] = b; n += 1
                else: bad += 1
            return out, n, bad
        Z = np.array([out[i] for i in use], np.float64).reshape(-1, 4); Z[:, 2:] = np.maximum(Z[:, 2:], 1.); an = ref.body.get('anchor', 'aim')
        sw = ref.body.get('sw', True); fam = [ref.family_of(self.h[items[i][0]][3] if sw else ref.eff_key) for i in use]
        if all(f == fam[0] for f in fam): D = ref.table.conv_delta(Z, fam[0])
        else:   # ids last seen under different detector families (one was lost before a detector change): one call per family
            D = np.empty((len(use), 4))
            for f in set(fam):
                m = np.array([f_ == f for f_ in fam]); D[m] = ref.table.conv_delta(Z[m], f)
        for k, i in enumerate(use):
            e = self.h[items[i][0]]; zr = Z[k]; d = D[k]
            if sw and e[4] is not None: d = (1.0 - e[4][1]) * e[4][0] + e[4][1] * d   # the seen transition of its last step, at that progress
            hc = zr[3] * math.exp(d[3]); cx = zr[0] + d[0] * zr[3]; cy = zr[1] + d[1] * zr[3]; top = cy - hc / 2; h_ = math.exp(e[1])
            tf = top + ref.aim_y * (hc - h_) if an == 'aim' else top if an == 'top' else cy - h_ / 2
            b = np.array([cx, tf + h_ / 2, math.exp(e[0]), h_])
            if np.isfinite(b).all(): out[i] = b; n += 1
            else: bad += 1
        return out, n, bad


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
    if time.time() - mt < 5:
        if BR['table'] is None: BR['tnote'] = '변환표 파일이 방금 바뀌어 5초 뒤에 읽음'   # (a table already read stays in use meanwhile)
        return BR['table']
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


def _hold_step(obj, res, H):
    """lost_same: LostHold of this Refiner (a new Refiner = a new one: mode change, reset), stepped with this frame's seen ids. Under BR['lock'].
    Its own error count: an error here never changes the drawn / gimbal rows."""
    try:
        ho = BR.get('hold')
        if ho is None or ho.ref is not obj: ho = BR['hold'] = LostHold(obj)
        ho.step(res, H)
    except Exception as e_:
        BR.pop('hold', None); BR['lost_err'] = repr(e_)[:160]; BR['lost_errors'] += 1


def lost_boxes(items):
    """lost_same (11-16): page_handler._lost_marks -> web/lost_marks.update, frame thread, after apply() of the same frame (which already carried
    the held sizes by this frame's camera scale): items [(output id, tracker lost box cxcywh, lost step)] -> [box cxcywh per item] in the running
    mode's box definition with the held size (LostHold.boxes). Switch off, mode off, no state yet: the tracker boxes. Never raises: an error
    gives the tracker boxes and is counted (lost_errors)."""
    t0 = time.perf_counter()
    try:
        with BR['lock']:
            obj = BR.get('obj'); ho = BR.get('hold')
            if not BR['lost_same'] or obj is None or ho is None or ho.ref is not obj or obj.mode != BR['mode']:
                return [np.array(it[1], np.float64).reshape(4) for it in items]
            out, n, bad = ho.boxes(items)
            if bad: BR['lost_errors'] += bad; BR['lost_err'] = '%d개 결과가 수가 아님: 추적기 상자로 그림' % bad
            BR['lost_n'] = n; BR['lost_frames'] += 1
            ms = (time.perf_counter() - t0) * 1000.0; BR['lost_ms'] += 0.05 * (ms - BR['lost_ms']); BR['lost_ms_max'] = max(BR['lost_ms_max'] * 0.999, ms)
            return out
    except Exception as e_:
        with BR['lock']: BR['lost_err'] = repr(e_)[:160]; BR['lost_errors'] += 1
        return [np.array(it[1], np.float64).reshape(4) for it in items]


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
                BR['why'] = (BR['tnote'] or '변환표 파일 없음') + ': 그대로 둠'; return rows   # the real reason (missing, unreadable, just copied)
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
            H = _warp()
            res = obj.step(items, float(t) if t is not None else time.perf_counter(), H, _app('args').det if _app('args') is not None else None, _new_ready())
            if BR['lost_same']: _hold_step(obj, res, H)   # 11-16 lost_same: the held sizes for the lost boxes of this frame (read only, never the rows)
            elif 'hold' in BR: BR.pop('hold', None)
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
    obj = BR.get('obj'); ho_ = BR.get('hold')   # read once: a lost_same toggle may pop 'hold' from another thread
    return dict(mode=BR['mode'], mode_name=MODE_KO.get(BR['mode']), blend_s=BR['blend_s'],
                modes=[dict(key=k, name=MODE_KO[k], available=(m is not None) if k == 'body' else True) for k in MODES],
                size=dict(SIZE), body=dict(BODY, **((m.body if m is not None else None) or {})), body_file=BR['tfile'], body_note=BR['tnote'],
                body_family=det_family(getattr(_app('args'), 'det', None)), body_family_name=FAMILY_KO.get(det_family(getattr(_app('args'), 'det', None))) if m is not None and m.person is not None else FAMILY_KO['pose'],
                body_switches=getattr(obj, 'switches', 0) if obj is not None else 0,
                ms=round(BR['ms'], 3), ms_max=round(BR['ms_max'], 3), ms_track=round(BR['ms_track'], 3),
                tracks=len(obj.tr) if obj is not None else 0, people=BR['n'], frames=BR['frames'], blends=BR['blends'], handovers=BR['handovers'], why=BR['why'],
                err=BR['err'], errors=BR['errors'], errors_all=BR['errors_all'],
                lost_same=bool(BR['lost_same']), lost_ms=round(BR['lost_ms'], 3), lost_ms_max=round(BR['lost_ms_max'], 3), lost_n=BR['lost_n'],   # 11-16 lost_same
                lost_held=len(ho_.h) if ho_ is not None else 0, lost_frames=BR['lost_frames'], lost_err=BR['lost_err'],
                lost_errors=BR['lost_errors'], lost_why=BR['lost_why'])


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
    if 'lost_same' in sel: qs.append({'lost_same': ['1' if sel['lost_same'] in (True, 1, '1', 'true', 'on') else '0']})   # 11-16
    return qs


def parse_request(q_):
    """q_: box_refine / box_blend / lost_same keys (parse_qs lists). Everything is checked first. -> (ok, text, apply)."""
    mode = None; blend = None; lost = None
    if 'lost_same' in q_:   # 11-16: lost box = seen box (display only), 1 / 0
        v_ = str(q_['lost_same'][0]).strip().lower()
        if v_ not in ('1', '0', 'true', 'false', 'on', 'off'): return False, '놓친 상자 = 보일 때 상자 값은 1 또는 0', None
        lost = v_ in ('1', 'true', 'on')
    if 'box_refine' in q_:
        v_ = str(q_['box_refine'][0]).strip().lower()
        if v_ not in MODES: return False, '상자 다듬기는 off, size, body 가운데 하나', None
        if v_ == 'body' and scan_table() is None: return False, '상체 네모 %s (models/track/%s*.npz)' % (BR['tnote'] or '변환표 파일 없음', TABLE_PREFIX), None
        mode = v_
    if 'box_blend' in q_:
        try: blend = float(q_['box_blend'][0])
        except (ValueError, IndexError): return False, '섞기 시간은 숫자(초)', None
        if not 0.0 <= blend <= BLEND_MAX: return False, '섞기 시간은 0 ~ %g 초 (0 = 끔)' % BLEND_MAX, None

    def apply_():
        with BR['lock']:   # a new mode makes apply() start a new state machine; a new blend time keeps every person's state
            if mode is not None: BR['mode'] = mode
            if blend is not None: BR['blend_s'] = blend
            if lost is not None:   # the held sizes start empty (each id gets one at its next seen frame)
                BR['lost_same'] = lost; BR.pop('hold', None); BR.update(lost_err='', lost_errors=0, lost_ms=0.0, lost_ms_max=0.0, lost_n=0, lost_why='')
            BR['err'] = ''; BR['errors'] = 0; BR['why'] = ''; BR['ms'] = 0.0; BR['ms_max'] = 0.0   # no status of the earlier choice stays on the page
            if BR['mode'] == 'body': scan_table()
        print('RULES set box_refine %s blend %g s' % (BR['mode'], BR['blend_s']), flush=True)
        if lost is not None: print('RULES set lost_same %d' % BR['lost_same'], flush=True)
        sel_ = _app('sel_save')
        if sel_ is not None: sel_(box_refine=BR['mode'], box_blend=BR['blend_s'], **(dict(lost_same=int(BR['lost_same'])) if lost is not None else {}))
    return True, '적용됨', apply_
