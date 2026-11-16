"""v2/scripts/v2_lib.py - box_refine v2 scoring (rules: v2/PREREG_V2.json).

The SERVER module (worktree tracking_server/src/track/box_refine.py, Refiner) runs frame by frame over the saved server-like tracker rows, so the
offline scoring and the server share one reset rule (missing frames > 30, warp-aware jump, id hand-over, per-id switch blend).
Per-row metrics and events reuse verify/combo/combo_lib (ref_eval.per_row definitions); new: lag against the current detection, track-start rows
from the server's own state segments, steps over the whole 1 s after a switch.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = V2.parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'verify' / 'combo'))
import prcommon as P      # noqa: E402
import ref_lib as R       # noqa: E402
import ref_eval as RE     # noqa: E402
import combo_lib as C     # noqa: E402

sys.path.insert(0, P.PATHS['worktree'])
from src.track import box_refine as BRM   # noqa: E402

DETS = ('near', 'far')
COMBO_A = (5.0, 1.0, 1.0)     # verify/combo (a): mc_s 5 Hz, beta_s 1, dcut 1 Hz on the GRU output's log size
GRU_FILE = Path(P.PATHS['worktree']) / 'models' / 'track' / 'box_refine_gru_tmotdance_20261115.npz'
SPEED_BINS = C.SPEED_BINS


def gru_model():
    z = np.load(GRU_FILE); return BRM.GRUNp({k: z[k] for k in z.files})


# ------------------------------------------------------------------------------------------------ inputs
def frame_items(ids):
    """per frame: [(tid, row j, z = detection when IoU(tracker row, detection) >= 0.3 else the tracker row (= box_refine.apply), kp, tracker row)]"""
    by_f = defaultdict(list)
    for tid in sorted(ids):
        e = ids[tid]; trk = e['trk']; z = e['z']
        iou = P.iou_pair(P.cxcywh2xyxy(trk), P.cxcywh2xyxy(z)); zu = np.where((iou >= 0.3)[:, None], z, trk)
        zu = np.c_[zu[:, :2], np.maximum(zu[:, 2:], 1.)]; tk = np.c_[trk[:, :2], np.maximum(trk[:, 2:], 1.)]
        for j, f in enumerate(e['f']): by_f[int(f)].append((tid, j, zu[j], e['kp'][j], tk[j]))
    return by_f


def stream_inputs(r, stream):
    """one sequence + stream -> dict(ids, by_f, W, n, keys, fps, sw)"""
    ids, S = R.rows_of(ROOT, r, stream); tr = np.load(ROOT / 'data' / 'tracks' / ('%s.npz' % r['name'])); n = S['n']
    keys = [DETS[int(s)] for s in tr['sw_src']] if stream == 'sw' else [stream] * n
    sw = tr['sw_frames'] if stream == 'sw' else np.zeros(0, np.int64)
    return dict(ids=ids, by_f=frame_items(ids), W=S['W'], n=n, keys=keys, fps=r['fps'], sw=sw, times=None)


# ------------------------------------------------------------------------------------------------ running the server module
def run(inp, mode, model=None, size=None, ca=None, euro=None):
    """the server Refiner over one stream -> ({tid: out [n_rows, 4]}, {tid: k [n_rows]} (rows since the server state started)).
    ca = (mc_s, beta_s, dcut): combo (a), a size-only One Euro after the output (state kept in the server state dict, so it follows hand-overs
    and resets with it; camera scale since the id's previous frame from the server state 'hs')."""
    obj = BRM.Refiner(mode, model, euro or BRM.EURO, 0.0, size); ids = inp['ids']; W = inp['W']; fps = inp['fps']; times = inp.get('times')
    out = {tid: np.zeros((len(e['f']), 4)) for tid, e in ids.items()}; kk = {tid: np.zeros(len(e['f']), np.int64) for tid, e in ids.items()}
    for f in range(inp['n']):
        H = W[f] if (f > 0 and f < len(W)) else None; its = inp['by_f'].get(f, ()); t = float(times[f]) if times is not None else f / fps
        res = obj.step([(tid, z, kp, zr) for tid, j, z, kp, zr in its], t, H, inp['keys'][f], True)
        for tid, j, *_ in its:
            st = obj.tr[tid]; b = res[tid]
            if ca is not None:
                se = st.get('ca')
                if se is None: se = st['ca'] = BRM.SizeEuro(*ca); st['ca_t'] = t
                lw, lh = se.step(math.log(max(b[2], 1.)), math.log(max(b[3], 1.)), t - st['ca_t'] if se.x0 is not None else 0.04, BRM.log_scale(st.get('hs')))
                st['ca_t'] = t; b = np.array([b[0], b[1], math.exp(lw), math.exp(lh)])
            out[tid][j] = b; kk[tid][j] = st['k']
    return out, kk


# ------------------------------------------------------------------------------------------------ records, metrics, events
def recs_of(r, stream, inp, outs, kk):
    """per id record like combo_lib (e: f, z, tgt, hg, src, vis, k, det = the raw detection; o: method -> [n,4])"""
    recs = []
    for tid, e in inp['ids'].items():
        ee = {k: e[k] for k in ('f', 'z', 'tgt', 'hg', 'src')}; ee['vis'] = (e['kp'][:, :, 2] >= 0.5).sum(1).astype(np.int16); ee['k'] = kk[tid]
        o = dict(raw=e['z'].copy(), tracker=e['trk'].copy())
        for m, om in outs.items(): o[m] = om[tid]
        recs.append(dict(name=r['name'], ds=r['ds'], split=r['split'], stream=stream, id=tid, fps=r['fps'], sw=inp['sw'], e=ee, o=o))
    return recs


AIM = R.AIM_Y


def aim_of(b): return np.stack([b[:, 0], b[:, 1] - b[:, 3] / 2 + AIM * b[:, 3]], 1)


def rows_metrics(rec, o):
    """combo_lib.rows_metrics + lag against the current detection along the target motion (centre, aim), aim lag vs target, size lag vs detection."""
    d = C.rows_metrics(rec, o); e = rec['e']; g = e['tgt']; ok = d['ok']; f = e['f']; n = len(f); z = e['z']
    gs = np.where(ok[:, None], g, 1.); ht = gs[:, 3]
    c1 = np.r_[False, np.diff(f) == 1]; j = np.flatnonzero(c1 & ok & np.r_[False, ok[:-1]])
    along_det = np.full(n, np.nan); aim_det = np.full(n, np.nan); aim_t = np.full(n, np.nan); sdet = np.full(n, np.nan)
    if len(j):
        v = gs[j, :2] - gs[j - 1, :2]; sp = np.hypot(*v.T); u = v / np.maximum(sp, 1e-9)[:, None]
        along_det[j] = np.sum((o[j, :2] - z[j, :2]) * u, 1) / ht[j]
        ao = aim_of(o); az = aim_of(z); at = aim_of(gs)
        aim_det[j] = np.sum((ao[j] - az[j]) * u, 1) / ht[j]; aim_t[j] = np.sum((ao[j] - at[j]) * u, 1) / ht[j]
        lt = np.log(gs[:, 3]); dl = lt[j] - lt[j - 1]
        sdet[j] = np.sign(dl) * (np.log(np.maximum(o[j, 3], 1e-3)) - np.log(np.maximum(z[j, 3], 1e-3)))
    d['along_det'] = along_det; d['aim_det'] = aim_det; d['aim_t'] = aim_t; d['sdet'] = sdet
    return d


def events(rec):
    """combo_lib.events with the track-start rows from the SERVER state segments (e['k']): start = rows 1-2, start5 = rows 1-4, later = rows >= 3."""
    ev, sb = C.events(rec); kk = rec['e']['k']
    ev['start'] = (kk == 1) | (kk == 2); ev['start5'] = (kk >= 1) & (kk <= 4); ev['later'] = kk >= 3
    return ev, sb


KEYS = C.KEYS + ('along_det', 'aim_det', 'aim_t', 'sdet')


def gather(recs, outs, evs):
    names = sorted({r['name'] for r in recs}); loc = {n: i for i, n in enumerate(names)}
    rows = defaultdict(list); seqv = []; idv = []; ev = defaultdict(list); sbv = []
    for k, (rec, o, (em, sb)) in enumerate(zip(recs, outs, evs)):
        d = rows_metrics(rec, o)
        for kk in KEYS: rows[kk].append(d[kk])
        n = len(rec['e']['f']); seqv.append(np.full(n, loc[rec['name']])); idv.append(np.full(n, k)); sbv.append(sb)
        for kk, v in em.items(): ev[kk].append(v)
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    return rows, np.concatenate(seqv), np.concatenate(idv), {k: np.concatenate(v) for k, v in ev.items()}, np.concatenate(sbv), names


def mean_(v):
    v = v[np.isfinite(v)]; return float(v.mean()) if len(v) else None


def summary(rows, ev, idv):
    s = C.summary(rows, ev, idv); ok = rows['ok']
    for a, b in SPEED_BINS:
        mm = ok & (rows['spd'] >= a) & (rows['spd'] < b)
        s['lag_%g-%g' % (a, b)].update(along_det=mean_(rows['along_det'][mm]), aim_det=mean_(rows['aim_det'][mm]), aim_t=mean_(rows['aim_t'][mm]))
    for a, b in C.SSPEED_BINS:
        mm = ok & (rows['sspd'] >= a) & (rows['sspd'] < b)
        s['slag_%g-%g' % (a, b)].update(sdet=mean_(rows['sdet'][mm]))
    for k, em in ev.items():   # the 1 s after a switch: largest steps too (the whole blend window, CHECK B)
        if k in s['events']:
            mm = em & ok; vh = rows['dh'][mm & np.isfinite(rows['dh'])]; va = rows['da'][mm & np.isfinite(rows['da'])]
            s['events'][k].update(da_max=float(va.max()), dh_max=float(vh.max()))
    return s


def sel_metrics(rows, ev, idv):
    """the val-A selection numbers (PREREG_V2 selection_valA) of one pooled dataset."""
    s = summary(rows, ev, idv)
    return dict(dh_p99=s['dh_p99'], tmax=s['dh_trackmax_med'], aim=s['aim'], d2a=s['d2a_rms'], lh=s['lh'], ce=s['ce'],
                slag=s['slag_0.03-99']['salong'], aimdet_fast=s['lag_0.1-99']['aim_det'], dh_p95=s['dh_p95'], da_p99=s['da_p99'],
                start_dh=s['events'].get('start', {}).get('dh_mean'), flip_dh_p95=s['events'].get('flip', {}).get('dh_p95'),
                sw1_dh_p99=s['events'].get('switch1s', {}).get('dh_p99'))
