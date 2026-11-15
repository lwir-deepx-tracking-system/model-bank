"""export_models.py: the val-A choices -> server files (float32 npz + json) in <root>/export.
  position memory: per arm the config (a scratch lr 2e-3 / b posetrack18 init lr 5e-4) and epoch with the lowest val-A sel -> pos_<arm>_tmotdance_20261115.npz
  box refiner: the lr with the lowest val-A loss -> box_refine_gru_tmotdance_20261115.npz (keys = torch GRUCell / Linear names, read by box_refine.GRUNp)
Usage: python -I export_models.py <root>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1]); runs = root / 'runs'; out = root / 'export'; out.mkdir(exist_ok=True); rep = {}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for arm in ('gru_kpt', 'mamba_kpt'):
    cand = []
    for cfg in ('a', 'b'):
        f = runs / ('pos_%s_%s_RESULT.json' % (arm, cfg))
        if f.exists(): r = json.loads(f.read_text()); cand.append((r['best']['sel'], cfg, r))
    if len(cand) < 2 and 'partial' not in sys.argv: print('SKIP', arm, 'runs not finished', flush=True); continue
    if not cand: continue
    sel, cfg, r = min(cand)
    ck = torch.load(runs / ('pos_%s_%s_best.pt' % (arm, cfg)), map_location='cpu')
    name = 'pos_%s_tmotdance_20261115' % arm
    np.savez(out / (name + '.npz'), **{k: v.cpu().numpy().astype(np.float32) for k, v in ck['model'].items()})
    info = dict(status=('관절 입력 %s 위치 기억 (TMOT+DanceTrack, 우리 근거리·원거리 자세 검출 출력으로 학습) · 장치 미확인' % ('GRU' if arm.startswith('gru') else 'Mamba')),
                source='pos_refine_20261115 pos_train.py %s config %s (%s), epoch %d of 30, val-A sel %.4f' % (arm, cfg, 'from scratch lr 2e-3' if cfg == 'a' else 'posetrack18 init lr 5e-4', ck['epoch'], sel),
                inputs='프레임당 68개 (서버 position_memory 와 같은 식, 카메라 움직임 보정 포함). 관절 순서 = 우리 검출기(COCO) 그대로 (posetrack 이름이 아니므로 관절 1, 2 를 지우지 않음)',
                output='15프레임 동안 보정 없는 Kalman 예측 상자에 더하는 보정(중심 ±1 키, 크기 ±50%)',
                choices={c: dict(sel=s, epoch=rr['best']['epoch']) for s, c, rr in cand}, kalman_valA=r['kalman_valA'], sha256=sha(out / (name + '.npz')))
    (out / (name + '.json')).write_text(json.dumps(info, ensure_ascii=False, indent=1)); rep[arm] = info; print('EXPORT', name, cfg, ck['epoch'], round(sel, 4), flush=True)
cand = []
if (root / 'results' / 'export.json').exists(): rep = {k: v for k, v in json.loads((root / 'results' / 'export.json').read_text()).items() if k not in rep}  | rep
for f in sorted(runs.glob('ref_lr*_RESULT.json')):
    r = json.loads(f.read_text()); cand.append((r['best']['val_loss'], f.name.replace('_RESULT.json', ''), r))
vl, stem, r = min(cand)
ck = torch.load(runs / (stem + '_best.pt'), map_location='cpu'); name = 'box_refine_gru_tmotdance_20261115'
np.savez(out / (name + '.npz'), **{k: v.cpu().numpy().astype(np.float32) for k, v in ck['model'].items()})
info = dict(status='상자 다듬기 GRU (TMOT+DanceTrack, 우리 근거리·원거리 자세 검출 출력과 둘을 바꾸는 장면으로 학습) · 장치 미확인',
            source='pos_refine_20261115 ref_train.py %s, epoch %d of 30, val-A loss %.5f (raw %.5f)' % (stem, ck['epoch'], vl, r['raw_valA'][0]),
            inputs='프레임당 70개: 자체 Kalman(카메라 움직임 보정) 속도 2, 예측과 검출 차이 4, 카메라 보정한 상자 변화 4, log 키, 가로세로 비, 화면 가장자리 4, 관절 51, 놓친 프레임 수, 카메라 이동 2',
            output='지금 검출 상자의 보정: 중심 ±0.5 키, 크기 x/ 2.5. 목표 = 몸 네모 하나의 정의(머리 위~골반, 사람 키 100 px 이상에서 잰 공개 자료 비율)',
            choices={c: dict(val_loss=v, epoch=rr['best']['epoch']) for v, c, rr in cand}, sha256=sha(out / (name + '.npz')))
(out / (name + '.json')).write_text(json.dumps(info, ensure_ascii=False, indent=1)); rep['box_refine'] = info; print('EXPORT', name, stem, ck['epoch'], round(vl, 5), flush=True)
(root / 'results' / 'export.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1)); print('EXPORT_DONE', flush=True)
