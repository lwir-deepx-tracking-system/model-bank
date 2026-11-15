"""fix_export.py (= gru_mismatch_20261116 pf_export.py) - the val-A choice of fix_train.py -> the server file (float32 npz + json) and the parity check
server (model-bank position_memory.pos_predict, the fixed warp order, picture rows of the dataset) == fixed builder + torch model, exactly.
Usage: python fix_export.py <run dir> <arm> <out dir> <name>
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fix_lib as F   # noqa: E402
P = F.P; L = F.L
SRV = Path(P.PATHS['worktree'])   # paths.json 'worktree': the model-bank tracking_server folder
sys.path.insert(0, str(SRV))
from src.common.shared_values import TRACKED          # noqa: E402
from src.detect import detect_frame as DF             # noqa: E402
from src.track import position_memory as PM           # noqa: E402


def parity(npz, arm, seqname, k):
    import torch
    z = np.load(npz); model = PM.PosKpt({kk: z[kk] for kk in z.files}); net = L.make_net(arm).double(); L.load_npz_into(net, npz); net.eval()
    PM.POSP.update(key=arm, model=model, gen=PM.POSP['gen'] + 1, err=0, used=0, show=False)
    reg = {r['name']: r for r in P.registry()}; r = reg[seqname]; S = L.load_seq(F.PR, r); T = F.tracks_of(S, k); W = S['W']; n = S['n']
    y0, y1 = F.ROWS[r['ds']]; PM.KP_ROW0, PM.KP_ROW1 = y0, y1; worst = 0.; n_tr = 0; n_fb = 0

    class KF:   # the server's GKF: warp right after predict
        warp = None
        def multi_predict(self, mean, cov):
            kf = P.KF(); om = mean.copy(); oc = cov.copy()
            for i in range(len(mean)):
                om[i], oc[i] = kf.predict(mean[i], cov[i])
                if self.warp is not None: om[i], oc[i] = P.gmc_warp(om[i], oc[i], self.warp)
            self.warp = None; return om, oc
    try:
        for g in sorted(T)[:60]:
            det = T[g]['det']; fs = sorted(det)
            run = [fs[0]]   # the first gap-free stretch (at most 25 frames); parity_both.py's list also took frames after a gap
            for f in fs[1:25]:
                if f != run[-1] + 1: break
                run.append(f)
            if len(run) < 8 or run[-1] + 16 >= n: continue
            kf = KF(); Pk = P.KF(); t = SimpleNamespace(state=TRACKED, cls=0, track_id=1); m0, c0 = Pk.initiate(det[run[0]][0]); mean = m0[None].copy(); cov = c0[None].copy()
            fell_back = False
            for kk, f in enumerate(run[1:]):
                zp, kpp = det[run[kk]]; DF.KP['prev'] = dict(boxes=np.array([P.cxcywh2xyxy(zp)], np.float32), kpts=kpp[None].astype(np.float32))
                fell_back |= PM._kp_lookup(mean[0, :4])[0] is None   # server: posterior vs detection IoU < 0.5 -> the posterior box, no keypoints (the builder always has the detection)
                kf.warp = W[f]; mean, cov = PM.pos_predict(kf, [t], mean, cov)
                t.pp_inn = PM.pos_inn(det[f][0], mean[0]); mean[0], cov[0] = Pk.update(mean[0], cov[0], det[f][0])
            zl, kpl = det[run[-1]]; DF.KP['prev'] = dict(boxes=np.array([P.cxcywh2xyxy(zl)], np.float32), kpts=kpl[None].astype(np.float32))
            fell_back |= PM._kp_lookup(mean[0, :4])[0] is None
            if fell_back: n_fb += 1; continue
            tl = run[-1] + 1; kf.warp = W[tl]; mean, cov = PM.pos_predict(kf, [t], mean, cov)   # lost step 1 (still TRACKED)
            p = t.pk; hist = np.stack(list(p['hist'])); post = p['post'].copy(); z_last = p['z'].copy(); t_last = run[-1]
            t.state = 2; got = []
            for s in range(2, P.KP_STEPS + 1):
                kf.warp = W[t_last + s]; mean, cov = PM.pos_predict(kf, [t], mean, cov); got.append(mean[0, :4].copy())
            base, xl = F.lost_path_batch(post[None], z_last[None], W, np.array([t_last]), n, y0, y1, True)
            hx = np.zeros((1, P.KP_HIST, P.KP_NF)); hm = np.zeros((1, P.KP_HIST)); hx[0, P.KP_HIST - len(hist):] = hist; hm[0, P.KP_HIST - len(hist):] = 1
            # the server's seen history must be the fixed builder's (same rows)
            ser = F.pos_series(run[0], run[-1], {f: det[f] for f in run}, W, y0, y1); hb = np.array([ser[f][0] for f in run][-P.KP_HIST:])
            hd = float(np.abs(hb - hist).max())
            with torch.no_grad():
                q = net(torch.as_tensor(hx), torch.as_tensor(hm), torch.as_tensor(xl), torch.as_tensor(base), torch.as_tensor([max(z_last[3], 1.)]))[0].numpy()
            d = max(float(np.abs(np.array(got) - q[1:]).max()), hd); worst = max(worst, d); n_tr += 1
            if n_tr >= 20: break
    finally:
        PM.KP_ROW0, PM.KP_ROW1 = 16.0, 496.0
    return dict(seq=seqname, det=k, tracks=n_tr, lost_frames=14, max_abs=worst, skipped_lookup_fallback=n_fb)


def main():
    import torch
    run = Path(sys.argv[1]); arm = sys.argv[2]; out = Path(sys.argv[3]); name = sys.argv[4]; out.mkdir(parents=True, exist_ok=True)
    R = json.loads((run / ('pf_%s_RESULT.json' % arm)).read_text()); ck = torch.load(run / ('pf_%s_best.pt' % arm), map_location='cpu')
    edges = ck.get('edges', True); F.EDGES[0] = edges
    extra = {} if edges else dict(opt_no_edges=np.array([1.0], np.float32))   # server: the picture-edge inputs held at 5 for this file
    np.savez(out / (name + '.npz'), **{k: v.cpu().numpy().astype(np.float32) for k, v in ck['model'].items()}, **extra)
    sha = hashlib.sha256((out / (name + '.npz')).read_bytes()).hexdigest()
    par = [parity(out / (name + '.npz'), arm, 'dancetrack0099', 'near'), parity(out / (name + '.npz'), arm, R.get('tmot_seq', 'seq17'), 'far')]
    info = dict(status=('관절 입력 %s 위치 기억 (TMOT+DanceTrack) · 11-16 고침: 화면에 잘린 정답의 목표 뺌, 놓친 1단계 카메라 보정 순서, ' % ('GRU' if arm.startswith('gru') else 'Mamba')) + ('화면 가장자리 입력을 실제 화면 줄로' if edges else '화면 가장자리 입력 없음(공개 자료 카메라 배치를 배우지 않게)'),
                source='fix_train.py %s%s (pos_train.py config a recipe, fixed builder fix_lib.py), epoch %d of 30, val-A sel %.4f' % (arm, '' if edges else ' --no-edges', ck['epoch'], ck['sel']),
                inputs='프레임당 68개 (서버 position_memory 와 같은 식, 카메라 움직임 보정 포함). ' + ('화면 가장자리 = 실제 화면 (DanceTrack 은 위아래 회색 띠 안쪽)' if edges else '화면 가장자리 넷은 늘 5') + '. 관절 순서 = 우리 검출기(COCO)',
                output='15프레임 동안 보정 없는 Kalman 예측 상자에 더하는 보정(중심 ±1 키, 크기 ±50%)', warp_order='lost step s uses the camera warp of frame t+s (server 11-16)', edge_inputs=('picture edges (real picture rows)' if edges else 'none: the four edge inputs are always 5 (opt_no_edges = 1)'),
                kalman_valA=R['kalman_valA'], best=dict(epoch=R['best']['epoch'], sel=R['best']['sel']), parity=par, sha256=sha)
    (out / (name + '.json')).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    ok = all(p['max_abs'] < 1e-6 and p['tracks'] > 0 for p in par)
    print('EXPORT', name, 'epoch', ck['epoch'], 'sel %.4f' % ck['sel'], 'sha', sha[:12], 'parity', par, 'PARITY_OK' if ok else 'PARITY_FAIL', flush=True)


if __name__ == '__main__':
    main()
