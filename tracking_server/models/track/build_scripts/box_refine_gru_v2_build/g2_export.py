"""gru_v2/scripts/g2_export.py - export the val-A chosen GRU v2 run (results/valA_selection.json) as npz + json (round-1 style) and check numpy = torch.
Writes export/box_refine_gru_v2_tmotdance_20261115.npz / .json, export/EXPORT.json, results/parity.json."""
import hashlib
import json
import pickle

import numpy as np
import torch

import g2_lib as L
import g2_train as T

sel = json.loads((L.G2 / 'results' / 'valA_selection.json').read_text()); stem = sel['chosen_run']
ck = torch.load(L.G2 / 'runs' / (stem + '_best.pt'), map_location='cpu'); name = 'box_refine_gru_v2_tmotdance_20261115'
w = {k: v.numpy().astype(np.float32) for k, v in ck['model'].items()}
w.update(arch=np.array(ck['arch']), lut_tab=L.TAB.astype(np.float64), lut_lh_cen=L.LH_CEN, lut_ar_cen=L.AR_CEN, kc=np.array(L.KC), ks=np.array(L.KS), vb=np.array(L.VB),
         size_euro=np.array(sel['size_chosen'], np.float64), lost_max=np.array(L.LOST_MAX), jump_h=np.array(L.JUMP_H), clip=np.array(L.CLIP))
out = L.G2 / 'export'; np.savez(out / (name + '.npz'), **w); sha = hashlib.sha256((out / (name + '.npz')).read_bytes()).hexdigest()
res = json.loads((L.G2 / 'runs' / (stem + '_RESULT.json')).read_text())
meta = {"status": "상자 다듬기 GRU v2 (TMOT+DanceTrack, 우리 근거리·원거리 자세 검출 출력) · 장치 미확인", "lut": "변환표 = 학습 분할 표에서 검출 상자 170 px 이상 두 줄을 130-170 px 줄(그대로)로 바꾼 것 (g2_lutfix.py: 그 두 줄은 DanceTrack 큰 사람 1,659 / 300 쌍의 잘못된 목표로 상자를 0.86 / 0.64 배로 줄였음. 가중치는 다시 학습하지 않음)",
        "source": "pos_refine_20261115/gru_v2 g2_train.py %s, epoch %d of %d, val-A run loss %.5f" % (stem, res['best']['epoch'], res['epochs'], res['best']['val_loss']),
        "inputs": "프레임당 사람마다 78개: 속도 2, 예측과 검출 차이 2, 중심 변화 2, 크기 변화 2, log 키, 가로세로 비, 화면 가장자리 4, 관절 51, 놓친 프레임 수, 카메라 이동 2, 변환표 보정 4, 시작 표시, 시작 후 줄 수, 이전 출력과 변환 상자의 차이 4",
        "output": "변환표로 바꾼 상자(머리 위~골반)와 이전 출력 사이를 성분별 비율 g 로 섞고 작은 보정을 더함(중심 ±0.4 키, 크기 x/ 2.2). 첫 줄은 변환 상자 + 보정. 그 뒤 크기(log 가로, log 세로)만 One Euro (size_euro = mc_s, beta_s, dcut)",
        "arch": ck['arch'], "loss": dict(lam_j=ck['lam_j'], lam_l=ck['lam_l']), "size_euro": sel['size_chosen'],
        "choices": {k: dict(J=v['J'], gate_violation=v['viol'], epoch=v['epoch']) for k, v in sel['runs'].items()}, "chosen_feasible": sel['chosen_run_feasible'], "sha256": sha}
(out / (name + '.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
(out / 'EXPORT.json').write_text(json.dumps(dict(npz=name + '.npz', sha256=sha, run=stem), indent=1))
# parity: torch (float32, training forward) vs the numpy engine (float64, frame by frame, all ids of a frame in one batch) on val-A rows
recs = pickle.loads((L.G2 / 'data' / 'rows_valA.pkl').read_bytes()); Wd, nd, recs = recs['W'], recs['n'], recs['recs']
net = L.make_net(ck['arch']); net.load_state_dict(ck['model']); net.eval(); npnet = L.load_net(out / (name + '.npz')); worst = 0.; nrow = 0; worst_rel = 0.
for nm, st in (('seq12', 'far'), ('seq48', 'sw'), ('dancetrack0066', 'near'), ('dancetrack0023', 'sw')):
    rr = [r for r in recs if r['name'] == nm and r['stream'] == st]; ids = {r['id']: dict(f=r['e']['f'], z=r['e']['z'], kp=r['e']['kp']) for r in rr}
    o = L.run_rows(ids, Wd[nm], rr[0]['fps'], nd[nm], npnet, None); V = [dict(r['inp'], tgt=r['e']['tgt'].astype(np.float32)) for r in rr]
    with torch.no_grad():
        for a in range(0, len(V), 64):
            idx = list(range(a, min(a + 64, len(V)))); X, Cb, Ha, ST, M, G, OK, CON, K = T.batch(V, idx); q, _ = net(X, Cb, Ha, ST, M); q = q.numpy()
            for b, i in enumerate(idx):
                n = len(V[i]['X']); d = np.abs(q[b, :n] - o[rr[i]['id']]['out']); worst = max(worst, float(d.max())); worst_rel = max(worst_rel, float((d / o[rr[i]['id']]['out'][:, 3:4]).max())); nrow += n
json.dump(dict(rows=nrow, max_abs_px=worst, max_rel_height=worst_rel, note='torch float32 padded-batch forward on the cached float32 inputs vs numpy float64 engine frame by frame'), open(L.G2 / 'results' / 'parity.json', 'w'), indent=1)
print('EXPORT_DONE', stem, sha, 'parity rows %d max |diff| %.4g px, %.3g heights' % (nrow, worst, worst_rel), flush=True)
