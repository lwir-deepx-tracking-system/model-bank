"""gru_v2/scripts/g2_data.py - data of GRU refiner v2 (PREREG_GRU_V2.json).
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
    Tn = PL.tracks_of(S, 'near', 'h2h'); Tf = PL.tracks_of(S, 'far', 'h2h'); W = S['W']; n = S['n']; fps = r['fps']; rate = 'same'
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
        det = R.occlude(det, rng); e = (Tn if g in Tn else Tf)[g]; fs = sorted(det)
        if len(fs) > 12 and rng.uniform() < 0.5: fs = fs[int(rng.integers(1, len(fs) - 10)):]      # random track start
        for a in range(0, len(fs), MAXLEN):
            ff = fs[a:a + MAXLEN]
            if len(ff) < 3: continue
            ids[eid] = dict(f=np.array(ff, np.int64), z=np.array([det[f][0] for f in ff]), kp=np.array([det[f][1] for f in ff]))
            tg[eid] = np.array([e['tgt'].get(f, [np.nan] * 4) for f in ff], np.float32); eid += 1
    o = L.run_rows(ids, W, fps, n); out = []
    for i, e in ids.items():
        con = np.r_[False, np.diff(e['f']) == 1] & ~o[i]['start']
        out.append(dict(X=o[i]['X'].astype(np.float32), C=o[i]['C'].astype(np.float32), Ha=o[i]['Ha'].astype(np.float32), st=o[i]['start'], k=o[i]['k'].astype(np.int16),
                        con=con, tgt=tg[i], rate=rate))
    return out


# ------------------------------------------------------------------------------------------------ server-like rows with the fixed methods
def build_rows(args):
    nm, split = args; r = {x['name']: x for x in P.registry()}[nm]; z = np.load(sorted((L.ROOT / 'export').glob('box_refine_gru_*.npz'))[-1]); v1 = R.RefNumpy({k: z[k] for k in z.files})
    tr = np.load(L.ROOT / 'data' / 'tracks' / ('%s.npz' % nm)); recs = []; W = None
    for stream in ('near', 'far', 'sw'):
        ids, S = R.rows_of(L.ROOT, r, stream); W = S['W']; sw = tr['sw_frames'] if stream == 'sw' else np.zeros(0, np.int64)
        eo = L.run_rows(ids, W, r['fps'], S['n'])
        for i, e in ids.items():
            st = eo[i]['start']; cv = L.conv_hard(e['z'])
            o = dict(raw=e['z'].copy(), tracker=e['trk'].copy(), euro=L.euro_rows(e, e['z'], st, W, r['fps'], L.EURO_V1), conv=cv,
                     conv_euro=L.euro_rows(e, cv, st, W, r['fps'], L.EURO_V1), gru_v1=R.run_ref(e, W, v1))
            ee = {k: e[k] for k in ('f', 'z', 'tgt', 'hg', 'src')}; ee['vis'] = (e['kp'][:, :, 2] >= 0.5).sum(1).astype(np.int16); ee['k'] = eo[i]['k']; ee['conv'] = cv
            ee['start'] = st; ee['kp'] = e['kp'].astype(np.float32)
            con = np.r_[False, np.diff(e['f']) == 1] & ~st
            recs.append(dict(name=nm, ds=r['ds'], split=split, stream=stream, id=i, fps=r['fps'], sw=sw, e=ee, o=o,
                             inp=dict(X=eo[i]['X'].astype(np.float32), C=eo[i]['C'].astype(np.float32), Ha=eo[i]['Ha'].astype(np.float32), st=st, k=eo[i]['k'].astype(np.int16), con=con)))
    return nm, W, recs, S['n']


def main():
    t0 = time.time(); D = L.G2 / 'data'; reg = P.registry()
    if sys.argv[1] == 'train':
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
