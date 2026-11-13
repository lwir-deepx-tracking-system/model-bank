"""Fixed-capacity ID-head ONNX versus variable reference, synthetic CPU cases only."""
import json
import os
import sys
import types
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parent
upstream = Path(os.environ['MOTIP_UPSTREAM'])
sys.path.insert(0,str(upstream))
sys.modules['MultiScaleDeformableAttention'] = types.ModuleType('MultiScaleDeformableAttention')
os.chdir(upstream)
import models.deformable_detr.backbone as backbone
backbone.is_main_process=lambda:False
from models.motip import build
from utils.misc import yaml_to_dict
from configs.util import load_super_config

config=yaml_to_dict(str(upstream/'configs/r50_deformable_detr_motip_dancetrack.yaml'))
config=load_super_config(config,config['SUPER_CONFIG_PATH']);config['DEVICE']='cpu'
torch.set_num_threads(2)
model,_=build(config)
state=torch.load(str(Path(os.environ['MOTIP_ARTIFACTS']) / 'motip_dancetrack_model_only.pth'),map_location='cpu',weights_only=True)
model.load_state_dict(state['model'],strict=True); model.eval()
keys=['trajectory_features','unknown_features','trajectory_id_labels','trajectory_times',
      'unknown_times','trajectory_masks','unknown_masks']


class Wrapper(torch.nn.Module):
    def forward(self,*args):
        seq=dict(zip(keys,args));seq=model.trajectory_modeling(seq)
        return model.id_decoder(seq,use_decoder_checkpoint=False)[0]


# Register the underlying modules so ONNX export preserves them as model weights.
class RegisteredWrapper(torch.nn.Module):
    def __init__(self, parent):
        super().__init__(); self.trajectory=parent.trajectory_modeling; self.decoder=parent.id_decoder
    def forward(self,*args):
        seq=self.trajectory(dict(zip(keys,args)))
        return self.decoder(seq,use_decoder_checkpoint=False)[0]


wrapper=RegisteredWrapper(model).eval()
torch.manual_seed(22)
rows=[]
cases=[]
big = os.environ.get('MOTIP_ID_BIG') == '1'
TMAX,NMAX,QMAX = (29,50,50) if big else (4,3,4)
spec = [(1,1,1),(2,2,2),(10,5,5),(29,20,20),(29,50,50)] if big else [(1,1,1),(2,2,2),(3,1,3),(4,3,4)]
for t,n,q in spec:
    base=dict(trajectory_features=torch.randn(1,1,t,n,256),unknown_features=torch.randn(1,1,1,q,256),
              trajectory_id_labels=torch.arange(n).view(1,1,1,n).expand(1,1,t,n).clone(),
              trajectory_times=torch.arange(t).view(1,1,t,1).expand(1,1,t,n).clone(),
              unknown_times=torch.full((1,1,1,q),t,dtype=torch.long),
              trajectory_masks=torch.zeros(1,1,t,n,dtype=torch.bool),unknown_masks=torch.zeros(1,1,1,q,dtype=torch.bool))
    pad=dict(trajectory_features=torch.zeros(1,1,TMAX,NMAX,256),unknown_features=torch.zeros(1,1,1,QMAX,256),
             trajectory_id_labels=torch.zeros(1,1,TMAX,NMAX,dtype=torch.long),trajectory_times=torch.zeros(1,1,TMAX,NMAX,dtype=torch.long),
             unknown_times=torch.full((1,1,1,QMAX),t,dtype=torch.long),trajectory_masks=torch.ones(1,1,TMAX,NMAX,dtype=torch.bool),
             unknown_masks=torch.ones(1,1,1,QMAX,dtype=torch.bool))
    for k in ('trajectory_features','trajectory_id_labels','trajectory_times','trajectory_masks'):pad[k][:,:,:t,:n]=base[k]
    for k in ('unknown_features','unknown_times','unknown_masks'):pad[k][:,:,:,:q]=base[k]
    cases.append((t,n,q,base,pad))

suffix='big' if big else 'small'
output=(Path(os.environ['MOTIP_ARTIFACTS']) / f'motip_id_padded_{suffix}_probe.onnx')
result=dict(scope='synthetic fixed-capacity ID head only, no real sequence or NPU',capacity=[TMAX,NMAX,QMAX],success=False)
try:
    import onnx
    import onnxruntime as ort
    with torch.no_grad():
        sample=cases[-1][-1]
        torch.onnx.export(wrapper,tuple(sample[k] for k in keys),str(output),opset_version=17,input_names=keys,output_names=['id_logits'])
        onnx.checker.check_model(onnx.load(str(output)))
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        sess=ort.InferenceSession(str(output),sess_options=opts,providers=['CPUExecutionProvider'])
        for t,n,q,base,pad in cases:
            expected=wrapper(*(base[k] for k in keys)).numpy()
            available={k:pad[k].numpy() for k in keys}
            got=sess.run(None,{i.name:available[i.name] for i in sess.get_inputs()})[0][:,:,:,:q,:]
            rows.append(dict(history=t,tracks=n,candidates=q,max_abs_error=float(np.max(np.abs(expected-got))),
                             matches=bool(np.allclose(expected,got,rtol=1e-3,atol=1e-4))))
    result.update(success=all(x['matches'] for x in rows),cases=rows,output=str(output))
except Exception as exc:
    result.update(error_type=type(exc).__name__,error=str(exc)[:1000]);raise
finally:
    (root/f'motip_id_padded_onnx_{suffix}_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
