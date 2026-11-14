"""실시간 엔드투엔드 추적: FastTrackTr 장치 NPU 판 (2026-11-14, 사용자 "실시간에서 쓸 수 있게 해").

검출 모델 + 추적기 대신, 한 프레임마다 FastTrackTr 그래프 세 개(AE = 백본·인코더·인코더 머리, 디코더 층 0, 층 1)를 NPU 에서 돌려
사람 상자를 내고, 앞 프레임 디코더 출력(traj)을 다음 프레임에 넘긴다. 구조와 내보내기: research/fasttracktr_npu/, 학습:
close_range ftt_npu_train.py / ftt_trained_export.py (모델 폴더 INFO.json 에 출처).
번호 잇기(E2E['link']): 'iou' = 겹침 연결기 (IoU >= 0.3 이면 같은 번호, 30 프레임 못 보면 끝; research/fasttracktr_npu/ftt_stored_run.py 와 같음).
'id' (11-14) = FastTrackTr 의 번호 출력(id_words, 정규화 원형 손실로 다시 학습한 판부터 쓸모 있음: ftt_id_check.py) 의 코사인 + 겹침.
  방금까지 보이던 사람(2 프레임 이내): 겹침 >= 0.1 또는 코사인 >= 0.6 이면 후보, 점수 = 코사인 + 겹침.
  놓친 사람(150 프레임 = 약 5초까지): 코사인 >= 0.6 이고 마지막 자리에서 키 x min(0.5 + 0.05 x 놓친 프레임, 4) 안이면 후보 (KeepID 와 같은 커지는 범위).
  사람마다 번호 특징은 0.9/0.1 로 천천히 갱신한다. 'auto' = INFO.json 의 "id_link": true 이면 'id', 아니면 'iou'.
켜져 있는 동안 KeepID 는 쓰지 않는다(11-14, 겹치는 기능은 끔).

모델 파일: models/e2e/fasttracktr/INFO.json = {"label": 화면 이름, "files": {"ae": ..., "b0": ..., "b1": ...}, ...}
코어: AE 와 디코더 모두 NPU 1. 켜져 있는 동안 프레임 루프는 검출 모델을 돌리지 않는다(uses_own(), 11-14 사용자 "겹치는 거 다 꺼야지"); 페이지도 그 줄들을 잠근다.
"""
import json
import os
import time

import numpy as np

E2E = dict(want='off', on='off', eng=None, err=None, ms=[], label=None, link='auto', auto_link='iou')   # want: 화면에서 고른 것, on: 프레임 루프가 적용한 것
_DIR = [None]


def model_dir():
    if _DIR[0] is None:
        import sys
        _DIR[0] = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'models', 'e2e', 'fasttracktr')
    return _DIR[0]


def info():
    """-> (INFO dict, None) when the three files exist, else (None, reason)."""
    p = os.path.join(model_dir(), 'INFO.json')
    try: d = json.load(open(p, encoding='utf-8'))
    except (OSError, ValueError) as e: return None, '모델 파일 없음 (%s)' % type(e).__name__
    fs = d.get('files') or {}
    for k in ('ae', 'b0', 'b1'):
        if not fs.get(k) or not os.path.isfile(os.path.join(model_dir(), fs[k])): return None, '모델 파일 없음 (%s)' % k
    return d, None


def iou_m(A, B):
    if not len(A) or not len(B): return np.zeros((len(A), len(B)))
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = ix * iy; ar = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return inter / (ar(A)[:, None] + ar(B)[None, :] - inter + 1e-9)


class FTTLive:
    def __init__(self, dx, d, core=1, score=0.5, max_det=30, link_iou=0.3, max_missed=30, id_max_missed=150, id_cos=0.6):
        self.score, self.max_det, self.link_iou, self.max_missed = score, max_det, link_iou, max_missed
        self.id_max_missed, self.id_cos = id_max_missed, id_cos
        def eng(f):
            o = dx.InferenceOption(); o.devices = [0]; o.bound_option = getattr(dx.InferenceOption.BOUND_OPTION, 'NPU_%d' % core); o.use_ort = True
            return dx.InferenceEngine(os.path.join(model_dir(), f), o)
        fs = d['files']; self.ea = eng(fs['ae']); self.ebs = [eng(fs['b0']), eng(fs['b1'])]
        self.ae_out = [t['name'] for t in self.ea.get_output_tensors_info()]
        self.b_in = [[(t['name'], tuple(t['shape']), t['dtype']) for t in e.get_input_tensors_info()] for e in self.ebs]
        self.b_out = [[t['name'] for t in e.get_output_tensors_info()] for e in self.ebs]
        self.reset()

    def reset(self):
        self.traj = np.zeros((1, 300, 256), np.float32); self.tracks = {}; self.next_id = 1; self.n = 0

    def link_iou_only(self, dets):
        """The overlap linker: IoU >= link_iou keeps the number, a track ends after max_missed frames unseen. -> [(track id, det index)]"""
        live = [t for t, v in self.tracks.items() if self.n - v['last'] <= self.max_missed]
        M = iou_m(dets, np.array([self.tracks[t]['box'] for t in live]) if live else np.zeros((0, 4))); ud, ut = set(), set(); rows = []
        for di, ti in sorted(((i, j) for i in range(M.shape[0]) for j in range(M.shape[1]) if M[i, j] >= self.link_iou), key=lambda p: -M[p]):
            if di in ud or ti in ut: continue
            ud.add(di); ut.add(ti); self.tracks[live[ti]].update(box=dets[di], last=self.n); rows.append((live[ti], di))
        for di in range(len(dets)):
            if di not in ud: self.tracks[self.next_id] = dict(box=dets[di], last=self.n, emb=None); rows.append((self.next_id, di)); self.next_id += 1
        for t in [t for t, v in self.tracks.items() if self.n - v['last'] > self.max_missed]: del self.tracks[t]
        return rows

    def link_id(self, dets, emb):
        """11-14: number output (id_words, unit vectors) + overlap; lost people come back by the number output inside a growing range."""
        live = [t for t, v in self.tracks.items() if self.n - v['last'] <= self.id_max_missed]
        tb = np.array([self.tracks[t]['box'] for t in live]) if live else np.zeros((0, 4)); M = iou_m(dets, tb); cand = []
        for ti, t in enumerate(live):
            v = self.tracks[t]; miss = self.n - v['last']; te = v.get('emb')
            for di in range(len(dets)):
                c = float(emb[di] @ te) if te is not None else -1.0
                if miss <= 2:
                    if M[di, ti] >= 0.1 or c >= self.id_cos: cand.append((c + M[di, ti] if te is not None else M[di, ti], di, ti))
                elif te is not None and c >= self.id_cos:
                    b = v['box']; h = max(b[3] - b[1], 1.0); r = h * min(0.5 + 0.05 * miss, 4.0)
                    cd = np.hypot((dets[di, 0] + dets[di, 2] - b[0] - b[2]) / 2, (dets[di, 1] + dets[di, 3] - b[1] - b[3]) / 2)
                    if cd <= r: cand.append((c, di, ti))
        ud, ut, rows = set(), set(), []
        for s, di, ti in sorted(cand, key=lambda p: -p[0]):
            if di in ud or ti in ut: continue
            ud.add(di); ut.add(ti); v = self.tracks[live[ti]]
            e_ = emb[di] if v.get('emb') is None else 0.9 * v['emb'] + 0.1 * emb[di]
            v.update(box=dets[di], last=self.n, emb=e_ / (np.linalg.norm(e_) + 1e-9)); rows.append((live[ti], di))
        for di in range(len(dets)):
            if di not in ud: self.tracks[self.next_id] = dict(box=dets[di], last=self.n, emb=emb[di].copy()); rows.append((self.next_id, di)); self.next_id += 1
        for t in [t for t, v in self.tracks.items() if self.n - v['last'] > self.id_max_missed]: del self.tracks[t]
        return rows

    def step(self, gray480):
        """gray480: 480x640 uint8 picture -> rows [[x1, y1, x2, y2, id, score, 0, -1]] in 640x512 canvas coords (y + 16)."""
        self.n += 1; H, W = gray480.shape
        ao = dict(zip(self.ae_out, [np.asarray(v, np.float32) for v in self.ea.run([np.ascontiguousarray(np.repeat(gray480[None, :, :, None], 3, axis=3))])]))
        st = dict(kv=ao['kv4'].reshape(1, -1, 256), tgt=ao['tgt4'].reshape(1, -1, 256), ref_unact=ao['ref4'].reshape(1, -1, 4), traj=self.traj)
        res = {}
        for e, ins, outs in zip(self.ebs, self.b_in, self.b_out):
            out = e.run([np.ascontiguousarray(st[n].reshape(s).astype(dt)) for n, s, dt in ins]); res = dict(zip(outs, [np.asarray(v, np.float32) for v in out]))
            if 'tgt_out' in res: st['tgt'], st['ref_unact'] = res['tgt_out'], res['ref_out']
        self.traj = res['hs'].reshape(1, 300, 256)
        sc = 1 / (1 + np.exp(-res['logits'].reshape(-1))); bx = res['boxes'].reshape(-1, 4)
        xy = np.clip(np.c_[(bx[:, 0] - bx[:, 2] / 2) * W, (bx[:, 1] - bx[:, 3] / 2) * H, (bx[:, 0] + bx[:, 2] / 2) * W, (bx[:, 1] + bx[:, 3] / 2) * H], 0, [W, H, W, H])
        keep = np.flatnonzero((sc >= self.score) & (xy[:, 2] - xy[:, 0] > 2) & (xy[:, 3] - xy[:, 1] > 2)); keep = keep[np.argsort(-sc[keep])]; sel = []
        for k in keep:   # NMS 0.7
            if all(iou_m(xy[k:k + 1], xy[j:j + 1])[0, 0] < 0.7 for j in sel): sel.append(k)
            if len(sel) >= self.max_det: break
        dets = xy[sel]; emb = None
        if 'id_words' in res:
            e_ = res['id_words'].reshape(300, -1)[sel] if len(sel) else np.zeros((0, 1)); emb = e_ / (np.linalg.norm(e_, axis=1, keepdims=True) + 1e-9)
        rows = self.link_id(dets, emb) if (link_mode() == 'id' and emb is not None) else self.link_iou_only(dets)
        return [[float(dets[di][0]), float(dets[di][1]) + 16.0, float(dets[di][2]), float(dets[di][3]) + 16.0, int(tid), float(sc[sel[di]]), 0, -1] for tid, di in rows]


def link_mode():
    """'iou' or 'id' (E2E['link'] 'auto' follows the model's INFO.json "id_link")."""
    m = E2E.get('link', 'auto'); return E2E.get('auto_link', 'iou') if m == 'auto' else m


def apply(dx):
    """Called once per frame by the frame loop: switch on / off as chosen on the page (engines are made / dropped here, never from the web thread)."""
    if E2E['want'] == E2E['on']: return
    if E2E['want'] == 'ftt':
        d, why = info()
        if d is None: E2E.update(want='off', err=why); return
        try: E2E['eng'] = FTTLive(dx, d); E2E.update(on='ftt', err=None, label=d.get('label', 'FastTrackTr'), ms=[], auto_link='id' if d.get('id_link') else 'iou'); print('E2E live -> FastTrackTr', flush=True)
        except Exception as e: E2E.update(want='off', on='off', eng=None, err='엔진을 만들지 못함: %s' % repr(e)[:120]); print('E2E live failed', repr(e)[:200], flush=True)
    else:
        E2E.update(on='off', eng=None); print('E2E live -> off', flush=True)


def step(gray_canvas):
    """gray_canvas: the 640x512 frame canvas (camera rows 16..495). -> rows or None (off / error: the caller keeps the tracker rows)."""
    if E2E['on'] != 'ftt' or E2E['eng'] is None: return None
    t = time.perf_counter()
    try: rows = E2E['eng'].step(np.ascontiguousarray(gray_canvas[16:496]))
    except Exception as e:
        E2E.update(want='off', on='off', eng=None, err='실행 오류: %s' % repr(e)[:120]); print('E2E live error', repr(e)[:200], flush=True); return None
    E2E['ms'].append((time.perf_counter() - t) * 1e3); del E2E['ms'][:-50]
    return rows


def uses_own():
    """True while the end-to-end model makes the rows: the frame loop then skips the detector and KeepID."""
    return E2E['on'] == 'ftt' and E2E['eng'] is not None


def skip_detector():
    """True while the server detector and tracker are not used: the live end-to-end model is on, or a recording is replayed with
    stored end-to-end results (their boxes are drawn and scored, the server's are not: replay_eval EV.own_boxes)."""
    if uses_own(): return True
    try:
        from ..camera import replay_eval
        ev = replay_eval.EV
        return bool(ev is not None and getattr(ev, 'own_boxes', False))
    except Exception:
        return False


def _link_for(d):
    lk = E2E.get('link', 'auto')
    return lk if lk != 'auto' else ('id' if (d or {}).get('id_link') else 'iou')


def state():
    d, why = info()
    return dict(available=d is not None, why=why, label=(d or {}).get('label'), want=E2E['want'], on=E2E['on'], err=E2E['err'], link=E2E.get('link', 'auto'), link_used=_link_for(d),
                ms=round(float(np.mean(E2E['ms'])), 1) if E2E['ms'] else None)
