"""parity_both.py (PC): the tracking server's numpy code == the torch models it is exported from, on real val-B inputs.
1. position memory: server src.track.position_memory.pos_predict (PosKpt numpy, keypoint history from detect_frame.KP, the plain path with the camera warp)
   vs pos_lib PosNet (torch) on the same history and the server-exact lost-step inputs (prcommon.lost_path); 20 tracks x 14 lost frames per file.
2. box refiner: server src.track.box_refine.Refiner (numpy GRU + its own Kalman / camera features, frame by frame with missing frames)
   vs prcommon.ref_series + ref_lib RefNet (torch) over whole tracker-id rows of val-B.
Usage: python parity_both.py <root> <pos_gru.npz> <pos_mamba.npz> <box_refine.npz>   -> results/parity.json
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prcommon as P
import pos_lib as PL
import ref_lib as R

sys.path.insert(0, P.PATHS['worktree'])
from src.track import position_memory as PM      # noqa: E402
from src.track import box_refine as BRM          # noqa: E402
from src.detect import detect_frame as DF        # noqa: E402
from src.common.shared_values import TRACKED     # noqa: E402


def pos_parity(npz, arm, root):
    import torch
    z = np.load(npz); model = PM.PosKpt({k: z[k] for k in z.files}); net = PL.make_net(arm).double(); PL.load_npz_into(net, npz); net.eval()
    PM.POSP.update(key=arm, model=model, gen=PM.POSP['gen'] + 1, err=0, used=0, show=False)
    reg = {r['name']: r for r in P.registry()}; S = PL.load_seq(root, reg['dancetrack0099']); T = PL.tracks_of(S, 'near', 'det'); W = S['W']; worst = 0.; n_tr = 0

    class KF:   # the server's GKF, warp applied after predict
        warp = None
        def multi_predict(self, mean, cov):
            kf = P.KF(); out_m = mean.copy(); out_c = cov.copy()
            for i in range(len(mean)):
                out_m[i], out_c[i] = kf.predict(mean[i], cov[i])
                if self.warp is not None: out_m[i], out_c[i] = P.gmc_warp(out_m[i], out_c[i], self.warp)
            self.warp = None; return out_m, out_c
    for g in sorted(T)[:40]:
        det = T[g]['det']; fs = sorted(det)
        run = [f for i, f in enumerate(fs) if i < 25 and (i == 0 or fs[i - 1] == f - 1)]
        if len(run) < 8: continue
        kf = KF(); P_kf = P.KF(); t = SimpleNamespace(state=TRACKED, cls=0, track_id=1); m0, c0 = P_kf.initiate(det[run[0]][0]); mean = m0[None].copy(); cov = c0[None].copy()
        # seen frames: the server collects its features from the previous frame's detection (detect_frame.KP['prev'])
        for k, f in enumerate(run[1:]):
            zprev, kpp = det[run[k]]; DF.KP['prev'] = dict(boxes=np.array([P.cxcywh2xyxy(zprev)], np.float32), kpts=kpp[None].astype(np.float32))
            kf.warp = W[f]; mean, cov = PM.pos_predict(kf, [t], mean, cov)
            pred = mean[0].copy(); t.pp_inn = PM.pos_inn(det[f][0], pred); mean[0], cov[0] = P_kf.update(mean[0], cov[0], det[f][0])
        zl, kpl = det[run[-1]]; DF.KP['prev'] = dict(boxes=np.array([P.cxcywh2xyxy(zl)], np.float32), kpts=kpl[None].astype(np.float32))
        tl = run[-1] + 1; kf.warp = W[tl] if tl < len(W) else None; mean, cov = PM.pos_predict(kf, [t], mean, cov)   # last seen frame's features go in; the track is not matched here
        p = t.pk; hist = np.stack(list(p['hist'])); post = p['post'].copy(); z_last = p['z'].copy(); t_last = run[-1]
        t.state = 2; got = []
        for s in range(2, PM.KP_STEPS + 1):
            fr = t_last + s; kf.warp = W[fr] if fr < len(W) else None
            mean, cov = PM.pos_predict(kf, [t], mean, cov); got.append(mean[0, :4].copy())
        base, xl = P.lost_path(post, z_last, W, t_last, len(W))
        hx = np.zeros((1, P.KP_HIST, P.KP_NF)); hm = np.zeros((1, P.KP_HIST)); hx[0, P.KP_HIST - len(hist):] = hist; hm[0, P.KP_HIST - len(hist):] = 1
        with torch.no_grad():
            q = net(torch.as_tensor(hx), torch.as_tensor(hm), torch.as_tensor(xl[None]), torch.as_tensor(base[None]), torch.as_tensor([max(z_last[3], 1.)]))[0].numpy()
        d = float(np.abs(np.array(got) - q[1:len(got) + 1]).max()); worst = max(worst, d); n_tr += 1
        if n_tr >= 20: break
    return dict(tracks=n_tr, lost_frames=14, max_abs_px=worst)


def ref_parity(npz, root):
    import torch
    z = np.load(npz); m_np = BRM.GRUNp({k: z[k] for k in z.files}); net = R.make_net(m_np.n).double(); sd = net.state_dict()
    net.load_state_dict({k: torch.as_tensor(z[k], dtype=torch.float64) for k in sd}); net.eval()
    reg = [r for r in P.registry() if r['split'] == 'valB']; worst = 0.; rows = 0; ids_n = 0; feat_worst = 0.
    for r in reg[:4] + reg[-4:]:
        ids, S = R.rows_of(root, r, 'sw')
        for i, e in list(ids.items())[:25]:
            seg = R.segments(e)[0]; f = e['f'][seg]; zz = e['z'][seg]; kp = e['kp'][seg]
            if len(f) < 5: continue
            ref = BRM.Refiner('gru', m_np); got = {}
            fr = list(range(int(f[0]), int(f[-1]) + 1)); idx = {int(ff): j for j, ff in enumerate(f)}
            for ff in fr:   # the server sees every frame (its warp) and this id only on its rows
                items = [(1, zz[idx[ff]], kp[idx[ff]])] if ff in idx else []
                o = ref.step(items, ff / r['fps'], S['W'][ff] if ff < len(S['W']) else None)
                if ff in idx: got[ff] = o[1]
            det = {int(ff): (zz[j], kp[j]) for j, ff in enumerate(f)}; ser = P.ref_series(int(f[0]), int(f[-1]), det, S['W'])
            X = np.array([ser[int(ff)][0] for ff in f])[None]; Z = np.array([ser[int(ff)][1] for ff in f])[None]
            with torch.no_grad(): q, _ = net(torch.as_tensor(X), torch.as_tensor(Z), torch.ones(1, len(f), dtype=torch.bool))
            q = q[0].numpy(); d = max(float(np.abs(got[int(ff)] - q[j]).max()) for j, ff in enumerate(f)); worst = max(worst, d); rows += len(f); ids_n += 1
    return dict(ids=ids_n, rows=rows, max_abs_px=worst, note='server Refiner (numpy, own features frame by frame) vs offline ref_series + torch RefNet (float64)')


def main():
    root = Path(sys.argv[1]); res = {}
    for arm, f in (('gru_kpt', sys.argv[2]), ('mamba_kpt', sys.argv[3])):
        res['pos_' + arm] = dict(pos_parity(f, arm, root), file=Path(f).name); print('PARITY pos', arm, res['pos_' + arm], flush=True)
    res['box_refine'] = dict(ref_parity(sys.argv[4], root), file=Path(sys.argv[4]).name); print('PARITY refiner', res['box_refine'], flush=True)
    ok = all(v['max_abs_px'] < 1e-3 for v in res.values()); res['ok'] = ok
    (root / 'results' / 'parity.json').write_text(json.dumps(res, indent=1)); print('PARITY_OK' if ok else 'PARITY_FAIL', flush=True)


if __name__ == '__main__':
    main()
