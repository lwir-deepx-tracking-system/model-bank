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
POS_NAMES = (('kalman', 'Kalman filter + 카메라 움직임 보정 (기본)'), ('gru', 'GRU 위치 기억 (신뢰도 출력은 재학습 후)'), ('mamba', 'Mamba 위치 기억 (신뢰도 출력은 재학습 후)'))   # v40: 위치 기억 · 신뢰도
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
        for key, cls in (('gru', PosGRU), ('mamba', PosMamba)):
            p = os.path.join(d, 'pos_%s.npz' % key)
            try: mt = os.path.getmtime(p)
            except OSError: POSP['avail'].pop(key, None); continue
            a = POSP['avail'].get(key)
            if (a is not None and a['mt'] == mt) or time.time() - mt < 5: continue
            try:
                z = np.load(p); m = cls({k: z[k] for k in z.files}); o, _ = m.step(np.zeros(16), m.init())
                assert o.shape == (4,) and np.isfinite(o).all()
                note = ''
                try: note = str(json.load(open(os.path.join(d, 'pos_%s.json' % key), encoding='utf-8')).get('status', ''))[:120]
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
    if M is None:
        mean, cov = kf.multi_predict(mean, cov)
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
    if POSP['show']:
        POSP['boxes'] = [(int(t.track_id), mean[i, :4].copy()) for i, t in enumerate(pool) if t.state != TRACKED and t.cls == PERSON_CLS]
        POSP['boxes_t'] = time.perf_counter()
    return mean, cov
