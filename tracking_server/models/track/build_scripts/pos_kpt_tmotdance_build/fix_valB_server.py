"""fix_valB_server.py (= gru_mismatch_20261116 gm_valB_server.py) - the model's own val-B score through the SERVER serving path (task step 1).

Per val-B sequence x detector (near / far) x gt person (the detections the training builder gives that person, pos_lib.tracks_of):
- seen frames: a server Track-like object goes through position_memory.pos_predict (the real function, PosKpt numpy of the given file) with the
  server's GKF (src.track.kalman.GKF, camera warp after predict), detect_frame.KP['prev'] = ALL detections of the previous frame of that detector
  (the server's _kp_lookup picks the best IoU >= 0.5), pp_inn = pos_inn(detection, prediction) and the GKF update (tracker_common.Track._absorb),
  lost tracks' size velocity zeroed before the predict (bytetrack.update), a new track after > 30 lost frames (ByteTrack buffer).
- every anchor of the training builder (seen frame a, >= 4 seen frames in the run, a target in the next 15 frames) gets a counterfactual loss from
  frame a+1: a copy of the track goes through pos_predict for frames a+1..a+15 with no detection (TRACKED at a+1, LOST from a+2), and a second copy
  through the plain GKF multi_predict (the 'kalman' choice of the server).
Score = the training score: mean over horizons 5, 10, 15 of the mean min(centre error / target height, 2), detector-definition target (tgt).
Usage: python fix_valB_server.py <weights npz> <out json> [workers] [--fix]
  --fix: the fixed builder (fix_lib) gives the anchors and targets (no target where the gt box is cut by the picture), and the
         server's picture rows (position_memory.KP_ROW0 / KP_ROW1) are set to the dataset's real picture rows (DanceTrack letterbox 76..436) for the
         replay; the server code is the model-bank working tree as it is at run time.
"""
import copy
import json
import sys
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prcommon as P   # noqa: E402
PR = Path(P.PATHS['root']); SRV = Path(P.PATHS['worktree'])
sys.path.insert(0, str(SRV))
import pos_lib as L    # noqa: E402
from src.common.shared_values import TRACKED, LOST   # noqa: E402
from src.detect import detect_frame as DF            # noqa: E402
from src.track import position_memory as PM          # noqa: E402
from src.track.kalman import GKF                     # noqa: E402

NPZ = None
FIXED = False


def clone(t, mean, cov):
    c = SimpleNamespace(state=t.state, cls=0, track_id=t.track_id, pp_inn=t.pp_inn.copy())
    p = getattr(t, 'pk', None)
    if p is not None:
        q = dict(p); q['hist'] = deque(p['hist'], maxlen=p['hist'].maxlen)
        for k in ('z', 'post', 'v', 'c', 'wh', 'vwh', 'pb'):
            if q.get(k) is not None: q[k] = np.array(q[k], copy=True)
        if 'st' in q: q['st'] = copy.deepcopy(q['st'])
        c.pk = q
    return c, mean.copy(), cov.copy()


def one_seq(nm):
    z = np.load(NPZ); M = PM.PosKpt({k: z[k] for k in z.files}); M.posetrack = 'posetrack' in Path(NPZ).name
    PM.POSP.update(key='gru_kpt', model=M, gen=PM.POSP['gen'] + 1, err=0, used=0, show=False)
    r = {x['name']: x for x in P.registry()}[nm]; S = L.load_seq(PR, r); W = S['W']; n = S['n']; out = []
    if FIXED:
        sys.path.insert(0, str(Path(__file__).resolve().parent)); import fix_lib as F
        PM.KP_ROW0, PM.KP_ROW1 = F.ROWS[r['ds']]
    for k in L.DETS:
        T = F.tracks_of(S, k) if FIXED else L.tracks_of(S, k, 'det'); d = S['d'][k]
        byf = {}
        for i, f in enumerate(d['fi']): byf.setdefault(int(f), []).append(i)
        def kp_of(f):
            ii = byf.get(f, [])
            return dict(boxes=d['box'][ii].astype(np.float32).reshape(-1, 4), kpts=d['kpt'][ii].astype(np.float32).reshape(-1, 17, 3))
        A = F.anchors(S, k, T, W, n) if FIXED else L.anchors(S, k, T, W, n)   # the training builder's anchors (gid, t) and targets
        want = {}
        for i, (g, t) in enumerate(zip(A['gid'], A['t'])): want[(int(g), int(t))] = i
        tgt = np.array(A['tgt'], np.float64) if len(A['tgt']) else np.zeros((0, 15, 4))
        Qg = np.full((len(tgt), 15, 4), np.nan); Qk = np.full((len(tgt), 15, 4), np.nan); nn = np.zeros(len(tgt), int)
        for g, e in T.items():
            det = e['det']; f0 = min(det); f1 = max(e['f1'], max(det))
            kf = GKF(); t = None; mean = cov = None; lost = 0
            for f in range(f0, f1 + 1):
                if t is not None:
                    m = mean.copy()
                    if t.state != TRACKED: m[0, 6:8] = 0
                    DF.KP['prev'] = kp_of(f - 1); kf.warp = W[f] if f < len(W) else None
                    mean, cov = PM.pos_predict(kf, [t], m, cov)
                if f in det:
                    zz = np.asarray(det[f][0], np.float64)
                    if t is None or lost > P.LOST_MAX:
                        t = SimpleNamespace(state=TRACKED, cls=0, track_id=g, pp_inn=np.zeros(4)); m0, c0 = kf.initiate(zz); mean = m0[None]; cov = c0[None]
                    else:
                        t.pp_inn = PM.pos_inn(zz, mean[0]); m1, c1 = kf.update(mean[0], cov[0], zz); mean = m1[None]; cov = c1[None]; t.state = TRACKED
                    lost = 0
                    i = want.get((g, f))
                    if i is not None:   # counterfactual loss from f + 1
                        tc, mc, cc = clone(t, mean, cov); kc = GKF(); mk, ck = mean.copy(), cov.copy()
                        for s in range(1, 16):
                            fr = f + s
                            m = mc.copy()
                            if tc.state != TRACKED: m[0, 6:8] = 0
                            DF.KP['prev'] = kp_of(fr - 1); kc.warp = W[fr] if fr < len(W) else None
                            mc, cc = PM.pos_predict(kc, [tc], m, cc); Qg[i, s - 1] = mc[0, :4]
                            mm = mk.copy()
                            if s >= 2: mm[0, 6:8] = 0
                            kc.warp = W[fr] if fr < len(W) else None; mk, ck = kc.multi_predict(mm, ck); Qk[i, s - 1] = mk[0, :4]
                            tc.state = LOST
                        nn[i] = tc.pk['n'] if getattr(tc, 'pk', None) is not None else 0
                elif t is not None:
                    t.state = LOST; lost += 1
        out.append(dict(seq=nm, ds=r['ds'], det=k, Qg=Qg, Qk=Qk, tgt=tgt, base=np.array(A['base'], np.float64).reshape(-1, 15, 4), n=nn,
                        real=np.array(A['real'], bool), h0=np.array(A['h0'], np.float64), err=PM.POSP['err']))
    return out


def m5(ce, sel):
    return float(np.mean([np.nanmean(np.where(sel, ce[:, h - 1], np.nan)) for h in (5, 10, 15)]))


def main():
    global NPZ, FIXED
    NPZ = sys.argv[1]; dst = Path(sys.argv[2]); workers = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 6
    FIXED = '--fix' in sys.argv
    t0 = time.time(); reg = [r['name'] for r in P.registry() if r['split'] == 'valB']
    with Pool(workers, initializer=_init, initargs=(NPZ, FIXED)) as pool:
        R = [x for xx in pool.map(one_seq, reg) for x in xx]
    cat = lambda k: np.concatenate([x[k] for x in R])
    Qg, Qk, tgt, base, real, nn = cat('Qg'), cat('Qk'), cat('tgt'), cat('base'), cat('real'), cat('n')
    ds = np.concatenate([[x['ds']] * len(x['tgt']) for x in R])
    ceG = L.ce_matrix(Qg, tgt); ceK = L.ce_matrix(Qk, tgt); ceB = L.ce_matrix(base, tgt)
    res = dict(weights=Path(NPZ).name, fix='--fix' in sys.argv, anchors=int(len(tgt)), short_hist=int((nn < PM.KP_MINOBS).sum()), pos_err=int(sum(x['err'] for x in R)),
               sec=round(time.time() - t0, 1), by_ds={})
    for d_ in ('TMOT', 'DT'):
        s = ds == d_
        res['by_ds'][d_] = dict(anchors=int(s.sum()), server_gru=m5(ceG, s), server_kalman=m5(ceK, s), builder_kalman=m5(ceB, s),
                                real_server_gru=m5(ceG, s & real), real_server_kalman=m5(ceK, s & real))
        print(d_, json.dumps(res['by_ds'][d_]), flush=True)
    dst.write_text(json.dumps(res, indent=1))
    np.savez_compressed(dst.with_suffix('.npz'), Qg=Qg.astype(np.float32), Qk=Qk.astype(np.float32), ds=ds)
    print('VALB_SERVER_DONE', json.dumps({k: v for k, v in res.items() if k != 'by_ds'}), flush=True)


def _init(npz, fixed=False):
    global NPZ, FIXED
    NPZ = npz; FIXED = fixed


if __name__ == '__main__':
    main()
