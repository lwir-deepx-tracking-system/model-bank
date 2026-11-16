"""gru_v2/v3/scripts/g2_data.py - data of GRU refiner v3 = v2 with corrected targets (PREREG_GRU_V2.json amendment 2).
  python g2_data.py lut                          -> data/lut_v3.pkl : the verifier's lookup fit (verify/gen_outputs.fit_lut) on the CORRECTED train targets
  python g2_data.py train <versions> <workers>   -> data/train_v<k>.pkl : training pieces of the train split (per GT person; near / far / spliced, occluder,
                                                    flip, frame-rate change (half: every other frame; double: interpolated frames), random track starts)
  python g2_data.py rows valA|valB <workers>     -> data/rows_<split>.pkl : server-like tracker rows (data/tracks, streams near / far / sw) with the engine inputs
                                                    and the outputs of the fixed methods (raw, tracker, euro, conv, conv_euro, gru_v1)
"""
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import g2_lib as L
from g2_lib import P, PL, R

MAXLEN = 256


def fixed_tracks(S, k):
    """pos_lib.tracks_of(h2h) with the target of every frame that has a detection corrected by that detection's keypoints (g2_lib.fix_target)."""
    T = PL.tracks_of(S, k, 'h2h')
    for g, e in T.items():
        fs = [f for f in e['det'] if f in e['tgt']]
        if not fs: continue
        t2, _ = L.fix_target(np.array([e['tgt'][f] for f in fs]), np.array([e['det'][f][0] for f in fs]), np.array([e['det'][f][1] for f in fs]))
        for f, v in zip(fs, t2): e['tgt'][f] = v; e['tgt2'][f] = v
    return T


def kf_rows(ff, z, W):
    """tracker-row surrogate for the training pieces: the server Kalman (prcommon.KF + camera warp, as ref_series) over the raw detection boxes of one piece."""
    kf = P.KF(); mean = cov = None; lost = 0; out = np.zeros((len(ff), 4)); j = 0
    for f in range(int(ff[0]), int(ff[-1]) + 1):
        if mean is not None:
            m = mean.copy()
            if lost > 0: m[6:8] = 0
            mean, cov = kf.predict(m, cov)
            if W is not None and f < len(W): mean, cov = P.gmc_warp(mean, cov, W[f])
        if j < len(ff) and ff[j] == f:
            if mean is None or lost > P.LOST_MAX: mean, cov = kf.initiate(z[j])
            else: mean, cov = kf.update(mean, cov, z[j])
            out[j] = mean[:4]; lost = 0; j += 1
        elif mean is not None: lost += 1
    return out


def fit_lut():
    """= verify/gen_outputs.fit_lut (train split, both detectors and datasets pooled, median relation per detector-height x aspect cell, cells under 30 pairs
    from the nearest populated aspect cell of the row), on the corrected targets."""
    Z = []; T = []
    for r in [r for r in P.registry() if r['split'] == 'train']:
        S = PL.load_seq(L.ROOT, r); tg = P.xyxy2cxcywh(S['a']['tgt_h2h'])
        for k in ('near', 'far'):
            dg = S['a']['dg_' + k]; m = dg >= 0; z = P.xyxy2cxcywh(S['d'][k]['box'][m]); t, _ = L.fix_target(tg[dg[m]], z, S['d'][k]['kpt'][m]); Z.append(z); T.append(t)
    Z = np.concatenate(Z); T = np.concatenate(T); ok = np.isfinite(T[:, 0]); Z = Z[ok]; T = T[ok]
    e = np.stack([(T[:, 0] - Z[:, 0]) / Z[:, 3], (T[:, 1] - Z[:, 1]) / Z[:, 3], np.log(T[:, 2] / Z[:, 2]), np.log(T[:, 3] / Z[:, 3])], 1)
    lh = np.log(np.maximum(Z[:, 3], 1.)); ar = Z[:, 2] / np.maximum(Z[:, 3], 1.)
    bh = np.clip(np.searchsorted(L.LH_EDGES, lh, 'right') - 1, 0, len(L.LH_EDGES) - 2); ba = np.clip(np.searchsorted(L.AR_EDGES, ar, 'right') - 1, 0, len(L.AR_EDGES) - 2)
    tab = np.full((len(L.LH_EDGES) - 1, len(L.AR_EDGES) - 1, 4), np.nan); cnt = np.zeros(tab.shape[:2], int); glob = np.median(e, 0)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            m = (bh == i) & (ba == j); cnt[i, j] = m.sum()
            if m.sum() >= 30: tab[i, j] = np.median(e[m], 0)
    for i in range(tab.shape[0]):
        good = [j for j in range(tab.shape[1]) if np.isfinite(tab[i, j, 0])]
        for j in range(tab.shape[1]):
            if not np.isfinite(tab[i, j, 0]): tab[i, j] = tab[i, min(good, key=lambda g: abs(g - j))] if good else glob
    return tab, cnt


# ------------------------------------------------------------------------------------------------ frame-rate doubling (synthetic in-between frames)
def double(T, W, n, rng):
    """every frame f -> 2f; a new frame 2f+1 between f and f+1 where the person has a detection in both: mean box (cx, cy, log w, log h) + fresh noise of
    the track's own frame-to-frame noise size (robust sigma of the second difference / sqrt 6, halved variance), mean keypoints (visible in both),
    mean target. Warp of every half step = I + (W - I) / 2."""
    W2 = np.zeros((2 * n - 1, 2, 3)); W2[:] = np.eye(2, 3)
    for f in range(1, n):
        Hh = np.eye(2, 3) + (W[f] - np.eye(2, 3)) / 2; W2[2 * f - 1] = Hh; W2[2 * f] = Hh
    out = {}
    for g, e in T.items():
        det = e['det']; fs = sorted(det); o = dict(det={}, tgt={}, tgt2={}, hg={}, f1=2 * e['f1'])
        S = {f: np.r_[det[f][0][:2], np.log(np.maximum(det[f][0][2:], 1.))] for f in fs}
        d2 = [np.r_[(S[f - 1][:2] - 2 * S[f][:2] + S[f + 1][:2]) / max(det[f][0][3], 1.), S[f - 1][2:] - 2 * S[f][2:] + S[f + 1][2:]] for f in fs if f - 1 in S and f + 1 in S]
        sig = 1.4826 * np.median(np.abs(np.array(d2)), 0) / np.sqrt(6.) if len(d2) >= 5 else np.full(4, 0.01)
        for f in fs:
            o['det'][2 * f] = det[f]
            if f + 1 in det:
                (za, ka), (zb, kb) = det[f], det[f + 1]; h = max((za[3] + zb[3]) / 2, 1.); nz = rng.normal(0, 1, 4) * sig * np.sqrt(0.5)
                sm = (S[f] + S[f + 1]) / 2 + np.r_[nz[:2] * h, nz[2:]]; zm = np.r_[sm[:2], np.exp(sm[2:])]
                km = (ka + kb) / 2; km[:, :2] += nz[:2] * h; km[:, 2] = np.minimum(ka[:, 2], kb[:, 2]); o['det'][2 * f + 1] = (zm, km)
        for key in ('tgt', 'tgt2'):
            for f, v in e[key].items():
                o[key][2 * f] = v
                if f + 1 in e[key]:
                    b = e[key][f + 1]; o[key][2 * f + 1] = np.r_[(v[:2] + b[:2]) / 2, np.sqrt(np.maximum(v[2:] * b[2:], 1e-6))]
        for f, v in e['hg'].items():
            o['hg'][2 * f] = v
            if f + 1 in e['hg']: o['hg'][2 * f + 1] = (v + e['hg'][f + 1]) / 2
        out[g] = o
    return out, W2, 2 * n - 1


# ------------------------------------------------------------------------------------------------ training pieces
def build_train(args):
    v, nm = args; r = {x['name']: x for x in P.registry()}[nm]; S = PL.load_seq(L.ROOT, r); rng = np.random.default_rng([2077 + v, sum(map(ord, nm))])
    Tn = fixed_tracks(S, 'near'); Tf = fixed_tracks(S, 'far'); W = S['W']; n = S['n']; fps = r['fps']; rate = 'same'
    if rng.uniform() < 0.5:
        Tn, W2, _ = PL.augment(Tn, W, n, rng, flip=True); Tf, _, _ = PL.augment(Tf, W, n, rng, flip=True); W = W2
    u = rng.uniform()
    if u < 0.15:
        Tn, W2, n2 = PL.augment(Tn, W, n, rng, half=True); Tf, _, _ = PL.augment(Tf, W, n, rng, half=True); W = W2; n = n2; fps /= 2; rate = 'half'
    elif u < 0.45:
        Tn, W2, n2 = double(Tn, W, n, rng); Tf, _, _ = double(Tf, W, n, rng); W = W2; n = n2; fps *= 2; rate = 'double'
    ids = {}; tg = {}; eid = 0
    for g in sorted(set(Tn) | set(Tf)):
        det, src, sw = R.splice(Tn, Tf, g, rng, fps)
        if len(det) < 3: continue
        det = R.occlude(det, rng); tsrc = {f: (Tn if src[f] == 'near' else Tf)[g]['tgt'] for f in det}; fs = sorted(det)   # the target corrected with the detector in use on that frame
        if len(fs) > 12 and rng.uniform() < 0.5: fs = fs[int(rng.integers(1, len(fs) - 10)):]      # random track start
        for a in range(0, len(fs), MAXLEN):
            ff = fs[a:a + MAXLEN]
            if len(ff) < 3: continue
            ids[eid] = dict(f=np.array(ff, np.int64), z=np.array([det[f][0] for f in ff]), kp=np.array([det[f][1] for f in ff]))
            tg[eid] = np.array([tsrc[f].get(f, [np.nan] * 4) for f in ff], np.float32); eid += 1
    o = L.run_rows(ids, W, fps, n); out = []
    for i, e in ids.items():
        con = np.r_[False, np.diff(e['f']) == 1] & ~o[i]['start']
        out.append(dict(X=o[i]['X'].astype(np.float32), C=o[i]['C'].astype(np.float32), Ha=o[i]['Ha'].astype(np.float32), st=o[i]['start'], k=o[i]['k'].astype(np.int16),
                        con=con, tgt=tg[i], rate=rate, z=e['z'].astype(np.float32), trk=kf_rows(e['f'], e['z'], W).astype(np.float32)))
    return out


# ------------------------------------------------------------------------------------------------ server-like rows with the fixed methods
def build_rows(args):
    nm, split = args; r = {x['name']: x for x in P.registry()}[nm]; z = np.load(sorted((L.ROOT / 'export').glob('box_refine_gru_*.npz'))[-1]); v1 = R.RefNumpy({k: z[k] for k in z.files})
    tr = np.load(L.ROOT / 'data' / 'tracks' / ('%s.npz' % nm)); recs = []; W = None
    z2 = np.load(L.G2.parent / 'export' / 'box_refine_gru_v2_tmotdance_20261115.npz'); v2 = L.GRUNp({k: z2[k] for k in z2.files}); v2size = tuple(float(x) for x in z2['size_euro'])
    lut2 = (z2['lut_tab'].astype(np.float64), z2['lut_lh_cen'].astype(np.float64), z2['lut_ar_cen'].astype(np.float64)); lut3 = L.conv_delta.__defaults__
    for stream in ('near', 'far', 'sw'):
        ids, S = R.rows_of(L.ROOT, r, stream); W = S['W']; sw = tr['sw_frames'] if stream == 'sw' else np.zeros(0, np.int64)
        eo = L.run_rows(ids, W, r['fps'], S['n'])
        L.conv_delta.__defaults__ = lut2; e2 = L.run_rows(ids, W, r['fps'], S['n'], v2, v2size); L.conv_delta.__defaults__ = lut3   # GRU v2 as exported by the port agent (its own lookup)
        for i, e in ids.items():
            st = eo[i]['start']; cv = L.conv_hard(e['z']); e['tgt_old'] = e['tgt']; e['tgt'], e['fixed'] = L.fix_target(e['tgt'], e['z'], e['kp'])
            o = dict(raw=e['z'].copy(), tracker=e['trk'].copy(), euro=L.euro_rows(e, e['z'], st, W, r['fps'], L.EURO_V1), conv=cv,
                     conv_euro=L.euro_rows(e, cv, st, W, r['fps'], L.EURO_V1), gru_v1=R.run_ref(e, W, v1), gru_v2=e2[i]['out_f'])
            ee = {k: e[k] for k in ('f', 'z', 'tgt', 'hg', 'src', 'tgt_old', 'fixed', 'trk')}; ee['vis'] = (e['kp'][:, :, 2] >= 0.5).sum(1).astype(np.int16); ee['k'] = eo[i]['k']; ee['conv'] = cv
            ee['start'] = st; ee['kp'] = e['kp'].astype(np.float32)
            con = np.r_[False, np.diff(e['f']) == 1] & ~st
            recs.append(dict(name=nm, ds=r['ds'], split=split, stream=stream, id=i, fps=r['fps'], sw=sw, e=ee, o=o,
                             inp=dict(X=eo[i]['X'].astype(np.float32), C=eo[i]['C'].astype(np.float32), Ha=eo[i]['Ha'].astype(np.float32), st=st, k=eo[i]['k'].astype(np.int16), con=con)))
    return nm, W, recs, S['n']


def main():
    t0 = time.time(); D = L.G2 / 'data'; reg = P.registry()
    if sys.argv[1] == 'lut':
        tab, cnt = fit_lut(); (D / 'lut_v3.pkl').write_bytes(pickle.dumps(dict(tab=tab, cnt=cnt, lh_edges=L.LH_EDGES, ar_edges=L.AR_EDGES, fit='corrected targets (g2_lib.fix_target), train split'), protocol=4))
        np.set_printoptions(precision=3, suppress=True, linewidth=220); print('h ratio (target / detector) per detector-height row x aspect column:\n', np.exp(tab[:, :, 3])); print('dy / h:\n', tab[:, :, 1]); print('counts:\n', cnt); print('LUT_DONE', flush=True)
    elif sys.argv[1] == 'train':
        nv, nw = int(sys.argv[2]), int(sys.argv[3]); tr = [r['name'] for r in reg if r['split'] == 'train']
        with Pool(nw) as pool:
            for v in range(nv):
                pcs = [p for pp in pool.map(build_train, [(v, nm) for nm in tr]) for p in pp]; cnt = defaultdict(int)
                for p in pcs: cnt[p['rate']] += len(p['X'])
                (D / ('train_v%d.pkl' % v)).write_bytes(pickle.dumps(pcs, protocol=4))
                print('version %d pieces %d rows %d by rate %s starts %d (%.0f s)' % (v, len(pcs), sum(len(p['X']) for p in pcs), dict(cnt), sum(int(p['st'].sum()) for p in pcs), time.time() - t0), flush=True)
        print('TRAIN_DATA_DONE', flush=True)
    else:
        split, nw = sys.argv[2], int(sys.argv[3]); names = [r['name'] for r in reg if r['split'] == split]
        with Pool(nw) as pool: res = pool.map(build_rows, [(nm, split) for nm in names])
        recs = [x for _, _, rr, _ in res for x in rr]; Wd = {nm: W for nm, W, _, _ in res}; nd = {nm: n for nm, _, _, n in res}
        (D / ('rows_%s.pkl' % split)).write_bytes(pickle.dumps(dict(recs=recs, W=Wd, n=nd), protocol=4))
        print('ROWS_DONE %s ids %d rows %d state starts %d (%.0f s)' % (split, len(recs), sum(len(r['e']['f']) for r in recs), sum(int(r['e']['start'].sum()) for r in recs), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
