"""놓친 사람의 위치 기억 (GRU / Mamba). 기본값은 칼만 필터 그대로이고, 가중치 파일이 있을 때만 화면에서 고를 수 있다."""
import os
import sys
import json
import time
import numpy as np
import threading as _pos_th
from collections import deque as _pos_dq
from ..common.shared_values import PERSON_CLS, TRACKED


# ---- v32 (user 11-10): learned position predictor for LOST tracks (GRU / Mamba residual on Kalman + camera-motion warp). Default 'kalman' = no change.
POS_HIST, POS_MINOBS, POS_STEPS, POS_C, POS_S = 20, 5, 60, 0.5, float(np.log(1.1))
POS_NAMES = (('kalman', 'Kalman filter + 카메라 움직임 보정 (기본)'),
             ('gru_kpt', 'GRU 위치 기억 + 관절 (관절 내는 검출 모델과 함께)'), ('mamba_kpt', 'Mamba 위치 기억 + 관절 (관절 내는 검출 모델과 함께)'))   # 11-14: 관절 없는 GRU·Mamba 판은 지움(사용자)   # v40: 위치 기억 · 신뢰도. 11-14: 관절 판
POSP = dict(key='kalman', model=None, gen=0, show=False, boxes=[], boxes_t=0.0, avail={}, dir=None, err=0, used=0, lock=_pos_th.Lock())
_POS_Z4 = np.zeros(4); _POS_Z2 = np.zeros(2)


class PosGRU:
    """numpy forward of m4_model.GRUArm (GRU cell hidden 40 -> linear 4), one track at a time. Keys = GRUNumpy.save."""
    def __init__(self, w):
        self.w = {k: np.asarray(w[k], np.float64) for k in ('w_ih', 'w_hh', 'b_ih', 'b_hh', 'w_o', 'b_o')}; self.nh = self.w['w_hh'].shape[1]
    def init(self): return np.zeros(self.nh)
    def step(self, x, h):
        w = self.w; n = self.nh; gi = w['w_ih'] @ x + w['b_ih']; gh = w['w_hh'] @ h + w['b_hh']; sig = lambda a: 1. / (1. + np.exp(-a))
        r = sig(gi[:n] + gh[:n]); z = sig(gi[n:2 * n] + gh[n:2 * n]); c = np.tanh(gi[2 * n:] + r * gh[2 * n:]); h = (1. - z) * c + z * h
        return w['w_o'] @ h + w['b_o'], h


class PosMamba:
    """numpy forward of m4_model.MambaArm (one Mamba block as a one-step recurrence -> linear 4), one track at a time. Keys = the torch state_dict."""
    def __init__(self, w):
        g = lambda k: np.asarray(w[k], np.float64)
        self.ew, self.eb, self.n1, self.ip = g('emb.weight'), g('emb.bias'), g('norm.w'), g('in_proj.weight')
        self.cw, self.cb, self.xp, self.dw, self.db = g('conv_w'), g('conv_b'), g('x_proj.weight'), g('dt_proj.weight'), g('dt_proj.bias')
        self.nA, self.D, self.op, self.n2, self.hw, self.hb = -np.exp(g('A_log')), g('D'), g('out_proj.weight'), g('norm_f.w'), g('head.weight'), g('head.bias')
        self.di, self.dc = self.cw.shape; self.ds = self.nA.shape[1]; self.r = self.dw.shape[1]
    def init(self): return (np.zeros((self.di, self.dc - 1)), np.zeros((self.di, self.ds)))
    def step(self, x, st):
        buf, h = st; silu = lambda a: a / (1. + np.exp(-a)); rms = lambda a, w_: a / np.sqrt(np.mean(a * a) + 1e-5) * w_
        u = self.ew @ x + self.eb; xz = self.ip @ rms(u, self.n1); xi, z = xz[:self.di], xz[self.di:]
        win = np.concatenate([buf, xi[:, None]], 1); xc = silu((win * self.cw).sum(1) + self.cb)
        p = self.xp @ xc; dt = np.logaddexp(0., self.dw @ p[:self.r] + self.db); Bm = p[self.r:self.r + self.ds]; Cm = p[self.r + self.ds:]
        h = np.exp(dt[:, None] * self.nA) * h + dt[:, None] * Bm[None, :] * xc[:, None]
        y = ((h * Cm[None, :]).sum(1) + self.D * xc) * silu(z); u = u + self.op @ y
        return self.hw @ rms(u, self.n2) + self.hb, (win[:, 1:], h)


# ---- 11-14 (user: 관절을 읽도록 + 예측 이동·위치 기억 표시): keypoint arms trained on PoseTrack18 (close_range m5_kpt_motion.py, models/track/build_scripts/pos_kpt_posetrack18_build/).
# Input per seen frame (68): Kalman velocity 2, innovation 4, box change 4, log h / 100, log w / h, distance to the 4 picture edges 4 (all / box height),
# 17 keypoints relative to the box centre / box height x visible + visible flag, 0. While lost: the same box values of the plain Kalman path, keypoints 0, step / 15.
# Output: correction of the plain Kalman path (centre +-1 box height of the last seen box, size +-50 %) for 15 frames, then the Kalman filter alone.
# Keypoints come from detect_frame.KP (detector files with keypoint heads, kind six_kpt); without them the keypoint part is 0.
KP_NF, KP_HIST, KP_STEPS, KP_C, KP_S, KP_CLIP, KP_MINOBS = 68, 20, 15, 1.0, float(np.log(1.5)), 5.0, 4
KP_ROW0, KP_ROW1 = 16.0, 496.0   # camera rows inside the 640x512 frame canvas


class PosKpt:
    """numpy forward of m5_kpt_motion.Net: one recurrent cell (GRU or Mamba block) over the seen frames, then one step per lost frame."""
    kpt = True
    posetrack = False   # 11-14: True for files trained in PoseTrack keypoint order (index 1, 2 = head bottom / top, not the eyes our detector gives) -> those two inputs are 0

    def __init__(self, w):
        w = {k[5:] if k.startswith('cell.') else k: v for k, v in w.items()}
        if 'cell.weight_ih' in w:
            self.cell = PosGRU(dict(w_ih=w['cell.weight_ih'], w_hh=w['cell.weight_hh'], b_ih=w['cell.bias_ih'], b_hh=w['cell.bias_hh'], w_o=w['head.weight'], b_o=w['head.bias']))
        else: self.cell = PosMamba(w)
    def init(self): return self.cell.init()
    def step(self, x, st): return self.cell.step(x, st)


def _kp_boxfeat(v, b, prev, inn):
    h = max(b[3], 1.); w = max(b[2], 1.)
    edge = np.array([b[0] - w / 2, 640. - (b[0] + w / 2), b[1] - h / 2 - KP_ROW0, KP_ROW1 - (b[1] + h / 2)]) / h
    chg = np.r_[(b[:2] - prev[:2]) / h, np.log(np.maximum(b[2:4], 1.) / np.maximum(prev[2:4], 1.))]
    return np.r_[v / h, inn, chg, np.log(h / 100.), np.log(w / h), edge]


def _kp_lookup(box_xywh):
    """keypoints of the detection of the previous frame that matches this track box (IoU >= 0.5), from detect_frame.KP['prev'] -> (z xywh, kpts [17,3]) or (None, None)."""
    try:
        from ..detect.detect_frame import KP
        d = KP.get('prev')
    except Exception: return None, None
    if not d or not len(d['boxes']): return None, None
    b = d['boxes']; x1, y1 = box_xywh[0] - box_xywh[2] / 2, box_xywh[1] - box_xywh[3] / 2; x2, y2 = x1 + box_xywh[2], y1 + box_xywh[3]
    iw = np.clip(np.minimum(b[:, 2], x2) - np.maximum(b[:, 0], x1), 0, None); ih = np.clip(np.minimum(b[:, 3], y2) - np.maximum(b[:, 1], y1), 0, None); inter = iw * ih
    iou = inter / np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + box_xywh[2] * box_xywh[3] - inter, 1e-6); j = int(np.argmax(iou))
    if iou[j] < 0.5: return None, None
    r = b[j]; return np.array([(r[0] + r[2]) / 2, (r[1] + r[3]) / 2, r[2] - r[0], r[3] - r[1]], np.float64), d['kpts'][j]


def _kp_feat_seen(v, z, zprev, inn, kp, posetrack=False):
    f = _kp_boxfeat(v, z, zprev, inn); hz = max(z[3], 1.)
    if kp is None: kf = np.zeros(51)
    else:
        vis = (kp[:, 2] >= 0.5).astype(np.float64)
        if posetrack: vis[1:3] = 0.
        kf = np.c_[(kp[:, 0] - z[0]) / hz * vis, (kp[:, 1] - z[1]) / hz * vis, vis].ravel()
    return np.clip(np.r_[f, kf, 0.], -KP_CLIP, KP_CLIP)


def pos_predict_kpt(M, kf, pool, mean, cov):
    """Keypoint arms. Seen tracks collect inputs; a lost track gets the corrected plain Kalman path (camera warp applied to the path) written into its mean after the normal predict."""
    W = getattr(kf, 'warp', None); gen = POSP['gen']; todo = []
    for i, t in enumerate(pool):
        p = getattr(t, 'pk', None)
        if t.state == TRACKED:
            if p is None or p['gen'] != gen or p['j'] != 0: p = t.pk = dict(gen=gen, hist=_pos_dq(maxlen=KP_HIST), n=0, z=None, post=None, j=0)
            z, kp = _kp_lookup(mean[i, :4])
            if z is None: z = mean[i, :4].astype(np.float64)
            p['hist'].append(_kp_feat_seen(mean[i, 4:6], z, p['z'] if p['z'] is not None else z, getattr(t, 'pp_inn', _POS_Z4), kp, getattr(M, 'posetrack', False))); p['n'] += 1
            p['z'] = z; p['post'] = mean[i].copy()
        elif p is not None and p['gen'] == gen and 0 <= p['j'] < KP_STEPS:
            if p['j'] == 0:
                if p['n'] < KP_MINOBS: p['j'] = -1; continue
                st = M.init()
                for x in p['hist']: _, st = M.step(x, st)
                po = p['post']; p.update(st=st, v=po[4:6].copy(), c=po[:2].copy(), wh=po[2:4].copy(), vwh=po[6:8].copy(), pb=p['z'].copy(), h0=max(p['z'][3], 1.))
            todo.append((i, t, p))
    mean, cov = kf.multi_predict(mean, cov)
    for i, t, p in todo:
        keep = mean[i].copy()
        try:
            n_ = 2 if p['j'] == 0 else 1   # first call in state LOST: the frame before was already lost (step 1), this frame is step 2
            for _ in range(n_):
                s = p['j'] + 1
                if s > KP_STEPS: break
                if W is not None:
                    R = W[:2, :2]; p['c'] = R @ p['c'] + W[:2, 2]; p['v'] = R @ p['v']
                p['c'] = p['c'] + p['v']; wh = p['wh'] + (p['vwh'] if s == 1 else 0.); p['wh'] = wh
                b = np.r_[p['c'], wh]; hb = max(b[3], 1.); wb = max(b[2], 1.)
                edge = np.array([b[0] - wb / 2, 640. - (b[0] + wb / 2), b[1] - hb / 2 - KP_ROW0, KP_ROW1 - (b[1] + hb / 2)]) / hb
                chg = np.r_[(b[:2] - p['pb'][:2]) / hb, np.log(np.maximum(b[2:4], 1.) / np.maximum(p['pb'][2:4], 1.))]
                x = np.clip(np.r_[p['v'] / hb, _POS_Z4, chg, np.log(hb / 100.), np.log(wb / hb), edge, np.zeros(KP_NF - 17), s / KP_STEPS], -KP_CLIP, KP_CLIP)
                o, p['st'] = M.step(x, p['st'])
                if not np.isfinite(o).all(): raise ValueError('nan')
                q = np.r_[b[:2] + KP_C * p['h0'] * np.tanh(o[:2]), b[2:4] * np.exp(KP_S * np.tanh(o[2:]))]; p['pb'] = b; p['j'] = s
            if p['j'] <= KP_STEPS: mean[i, :4] = q; POSP['used'] += 1
        except Exception:
            mean[i] = keep; p['j'] = -1; POSP['err'] += 1
    return mean, cov


def pos_inn(z, p):
    """m4_model.innovation: observed box z (cx, cy, w, h) against the predicted state p."""
    h = max(p[3], 1.); return np.array([(z[0] - p[0]) / h, (z[1] - p[1]) / h, np.log(max(z[2], 1.) / max(p[2], 1.)), np.log(max(z[3], 1.) / h)])


def _pos_feat(v, whp, inn, flag, cam, m1, m2, cnt):
    """m4_model.feat for one track -> 16 numbers (lengths divided by the predicted box height, clipped to +-5)."""
    h = max(whp[1], 1.); w = max(whp[0], 1.); lo = lambda a: np.maximum(a, 1.0)
    return np.clip(np.r_[v / h, inn, flag, cam / h, (m1[:2] - m2[:2]) / h, np.log(lo(m1[2:4]) / lo(m2[2:4])), np.log(h / 100.), np.log(w / h), cnt], -5.0, 5.0)


def pos_scan():
    """Look for the weight files (cheap: one stat per file; a file is read again only when it changed and is older than 5 s = not being copied)."""
    d = POSP['dir'] or os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'models', 'track')
    with POSP['lock']:
        for key, cls in (('gru_kpt', PosKpt), ('mamba_kpt', PosKpt)):   # 11-14: box-only pos_gru / pos_mamba removed (user)
            p = os.path.join(d, 'pos_%s.npz' % key)
            if key.endswith('_kpt'):   # 11-14: newest pos_<arm>_kpt_*.npz (name carries data and date)
                c_ = sorted(f_ for f_ in os.listdir(d) if f_.startswith('pos_%s_' % key) and f_.endswith('.npz')) if os.path.isdir(d) else []
                if not c_: POSP['avail'].pop(key, None); continue
                p = os.path.join(d, c_[-1])
            try: mt = os.path.getmtime(p)
            except OSError: POSP['avail'].pop(key, None); continue
            a = POSP['avail'].get(key)
            if (a is not None and a['mt'] == mt) or time.time() - mt < 5: continue
            try:
                z = np.load(p); m = cls({k: z[k] for k in z.files}); m.posetrack = 'posetrack' in os.path.basename(p); o, _ = m.step(np.zeros(KP_NF if getattr(m, 'kpt', False) else 16), m.init())
                assert o.shape == (4,) and np.isfinite(o).all()
                note = ''
                try: note = str(json.load(open(p[:-4] + '.json', encoding='utf-8')).get('status', ''))[:120]
                except Exception: pass
                POSP['avail'][key] = dict(mt=mt, model=m, note=note or '가중치 파일 있음 (설명 없음)')
            except Exception as e:
                POSP['avail'][key] = dict(mt=mt, model=None, note='가중치 파일을 읽지 못함: %s' % type(e).__name__)


def pos_list():
    pos_scan(); out = []
    for key, name in POS_NAMES:
        a = POSP['avail'].get(key)
        if key == 'kalman': out.append(dict(key=key, name=name, available=True, note='학습 모델 없음'))
        else: out.append(dict(key=key, name=name, available=bool(a and a['model'] is not None), note=a['note'] if a else '가중치 없음 (학습 전)'))
    return out


def pos_select(key):
    """Called between two frames. Unknown / missing weights -> stays as it is. Every track restarts its history (gen)."""
    m = None
    if key != 'kalman':
        pos_scan(); a = POSP['avail'].get(key)
        if a is None or a['model'] is None: return False
        m = a['model']
    POSP.update(key=key, model=m, gen=POSP['gen'] + 1, err=0, used=0)
    print('POS predictor -> %s' % key, flush=True)
    return True


def pos_predict(kf, pool, mean, cov):
    """Replaces kf.multi_predict(mean, cov) for the tracked + lost pool. 'kalman': the very same call.
    Else m4_model.rollout frame by frame: tracked tracks only collect their input history (their prediction is NOT changed); a lost track's Kalman + warp
    prediction gets the bounded residual and the corrected box is written back into the Kalman mean (velocity, covariance untouched)."""
    M = POSP['model']
    _pos_trail_seen(pool, mean)
    if M is None:
        mean, cov = kf.multi_predict(mean, cov)
    elif getattr(M, 'kpt', False):   # 11-14
        mean, cov = pos_predict_kpt(M, kf, pool, mean, cov)
    else:
        W = getattr(kf, 'warp', None); gen = POSP['gen']; todo = []
        for i, t in enumerate(pool):
            p = getattr(t, 'pp', None)
            if t.state == TRACKED:
                if p is None or p['gen'] != gen or p['j'] != 0:
                    if p is not None and p['gen'] == gen: t.pp_inn = _POS_Z4            # found again: a new run starts (as in training)
                    p = t.pp = dict(gen=gen, hist=_pos_dq(maxlen=POS_HIST), n=1, m1=None, m2=None, x=None, j=0, st=None)
                elif p['x'] is not None: p['hist'].append(p['x']); p['n'] += 1               # last frame's prediction was followed by a detection
                p['m2'] = p['m1'] if p['m1'] is not None else mean[i, :4].copy(); p['m1'] = mean[i, :4].copy(); todo.append((i, t, p, False))
            else:
                if p is None or p['gen'] != gen or p['j'] < 0 or p['j'] >= POS_STEPS: continue
                if p['j'] == 0:      # first frame in state LOST: this row is the 1st lost prediction (made while still TRACKED) -> warm-up + step 1 now
                    if p['x'] is None or p['n'] < POS_MINOBS: p['j'] = -1; continue
                    keep = mean[i].copy()
                    try:
                        st = M.init()
                        for x in p['hist']: _, st = M.step(x, st)
                        o, st = M.step(p['x'], st); whp = np.maximum(mean[i, 2:4], 1.0)
                        if not np.isfinite(o).all(): raise ValueError('nan')
                        mean[i, :2] = mean[i, :2] + POS_C * whp[1] * np.tanh(o[:2]); mean[i, 2:4] = whp * np.exp(POS_S * np.tanh(o[2:]))
                        p.update(st=st, m2=p['m1'], m1=mean[i, :4].copy(), j=1, x=None); POSP['used'] += 1
                    except Exception:
                        mean[i] = keep; p['j'] = -1; POSP['err'] += 1; continue
                todo.append((i, t, p, True))
        c0 = mean[:, :2].copy()
        mean, cov = kf.multi_predict(mean, cov)
        for i, t, p, lost in todo:
            keep = mean[i].copy()
            try:
                cam = (W[:2, :2] @ c0[i] + W[:2, 2] - c0[i]) if W is not None else _POS_Z2
                whp = np.maximum(mean[i, 2:4], 1.0)
                if not lost: p['x'] = _pos_feat(mean[i, 4:6], whp, getattr(t, 'pp_inn', _POS_Z4), 1.0, cam, p['m1'], p['m2'], 0.0)
                else:
                    j = p['j'] + 1; o, st = M.step(_pos_feat(mean[i, 4:6], whp, _POS_Z4, 0.0, cam, p['m1'], p['m2'], (j - 1) / 60.), p['st'])
                    if not np.isfinite(o).all(): raise ValueError('nan')
                    mean[i, :2] = mean[i, :2] + POS_C * whp[1] * np.tanh(o[:2]); mean[i, 2:4] = whp * np.exp(POS_S * np.tanh(o[2:]))
                    p.update(st=st, m2=p['m1'], m1=mean[i, :4].copy(), j=j)
            except Exception:
                mean[i] = keep; p['j'] = -1; p['x'] = None; POSP['err'] += 1
        if POSP['err'] > 20:
            POSP.update(key='kalman', model=None, gen=POSP['gen'] + 1); print('POS predictor: more than 20 errors, back to kalman', flush=True)
    if POSP['show']:   # 11-14: + where the person was last seen (위치 기억), the predicted path since then (예측 이동), lost frames, predictor name
        out_ = []
        for i, t in enumerate(pool):
            if t.state == TRACKED or t.cls != PERSON_CLS: continue
            tr_ = getattr(t, 'pos_trail', None)
            if tr_ is not None:
                tr_['pts'].append((float(mean[i, 0]), float(mean[i, 1])))
                if len(tr_['pts']) > 90: del tr_['pts'][:-90]
            out_.append((int(t.track_id), mean[i, :4].copy(), dict(src=POSP['key'], last=tr_['last'] if tr_ else None, pts=list(tr_['pts']) if tr_ else [], n=len(tr_['pts']) if tr_ else 0)))
        POSP['boxes'] = out_
        POSP['boxes_t'] = time.perf_counter()
    return mean, cov


def _pos_trail_seen(pool, mean):
    """11-14: a seen person's last position (box centre and size) = the start of the predicted path drawn when it is lost."""
    if not POSP['show']: return
    for i, t in enumerate(pool):
        if t.state == TRACKED: t.pos_trail = dict(last=(float(mean[i, 0]), float(mean[i, 1]), float(mean[i, 2]), float(mean[i, 3])), pts=[])
