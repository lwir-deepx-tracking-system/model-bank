"""Test if masked fixed-capacity ID inputs match the original variable-size head."""
import json
import os
import sys
import types
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
upstream = Path(os.environ['MOTIP_UPSTREAM'])
sys.path.insert(0, str(upstream))
sys.modules['MultiScaleDeformableAttention'] = types.ModuleType('MultiScaleDeformableAttention')
os.chdir(upstream)
import models.deformable_detr.backbone as backbone
backbone.is_main_process = lambda: False
from models.motip import build
from utils.misc import yaml_to_dict
from configs.util import load_super_config

config = yaml_to_dict(str(upstream/'configs/r50_deformable_detr_motip_dancetrack.yaml'))
config = load_super_config(config, config['SUPER_CONFIG_PATH'])
config['DEVICE'] = 'cpu'
torch.set_num_threads(2)
model, _ = build(config)
state = torch.load(str(Path(os.environ['MOTIP_ARTIFACTS']) / 'motip_dancetrack_model_only.pth'),
                   map_location='cpu', weights_only=True)
model.load_state_dict(state['model'], strict=True)
model.eval()


def run(seq):
    # Model mutates the dictionary, so it must not share tensors with another run.
    seq = {k: v.clone() for k,v in seq.items()}
    seq = model.trajectory_modeling(seq)
    return model.id_decoder(seq, use_decoder_checkpoint=False)[0]


torch.manual_seed(22)
big = os.environ.get('MOTIP_ID_BIG') == '1'
TMAX, NMAX, QMAX = (29, 50, 50) if big else (4, 3, 4)
cases = [(1,1,1),(2,2,2),(10,5,5),(29,20,20),(29,50,50)] if big else [(1,1,1),(2,2,2),(3,1,3),(4,3,4)]
rows=[]
with torch.no_grad():
    for t,n,q in cases:
        base = dict(trajectory_features=torch.randn(1,1,t,n,256),
                    unknown_features=torch.randn(1,1,1,q,256),
                    trajectory_id_labels=torch.arange(n).view(1,1,1,n).expand(1,1,t,n).clone(),
                    trajectory_times=torch.arange(t).view(1,1,t,1).expand(1,1,t,n).clone(),
                    unknown_times=torch.full((1,1,1,q),t,dtype=torch.long),
                    trajectory_masks=torch.zeros(1,1,t,n,dtype=torch.bool),
                    unknown_masks=torch.zeros(1,1,1,q,dtype=torch.bool))
        padded = dict(trajectory_features=torch.zeros(1,1,TMAX,NMAX,256),
                      unknown_features=torch.zeros(1,1,1,QMAX,256),
                      trajectory_id_labels=torch.zeros(1,1,TMAX,NMAX,dtype=torch.long),
                      trajectory_times=torch.zeros(1,1,TMAX,NMAX,dtype=torch.long),
                      unknown_times=torch.full((1,1,1,QMAX),t,dtype=torch.long),
                      trajectory_masks=torch.ones(1,1,TMAX,NMAX,dtype=torch.bool),
                      unknown_masks=torch.ones(1,1,1,QMAX,dtype=torch.bool))
        for k in ('trajectory_features','trajectory_id_labels','trajectory_times','trajectory_masks'):
            padded[k][:,:,:t,:n] = base[k]
        for k in ('unknown_features','unknown_times','unknown_masks'):
            padded[k][:,:,:,:q] = base[k]
        expected = run(base)
        got = run(padded)[:,:,:,:q,:]
        error=float((expected-got).abs().max())
        rows.append(dict(history=t, tracks=n, candidates=q, max_abs_error=error,
                         matches=bool(torch.allclose(expected,got,rtol=1e-4,atol=1e-5))))
result = dict(scope='synthetic CPU ID logits only; not M1, full tracking, or accuracy',
              capacity=dict(history=TMAX, tracks=NMAX, candidates=QMAX), cases=rows,
              all_match=all(r['matches'] for r in rows))
(root/('motip_id_padding_big_preflight.json' if big else 'motip_id_padding_preflight.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
