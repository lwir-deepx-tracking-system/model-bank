"""Compare 4 held frames: M1 INT8 backbone + float remainder against full float ONNX."""
import json
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort

root=Path(__file__).resolve().parent
base=Path(os.environ['MOTIP_ARTIFACTS'])
height=int(os.environ.get('MOTIP_PROBE_H','128'));width=int(os.environ.get('MOTIP_PROBE_W','160'))
data=np.load(base/f'motip_backbone_{height}x{width}_calib_probe.npz',allow_pickle=False)
npu_path = base/f'motip_npu_backbone_outputs_{height}x{width}.npz'
if not npu_path.exists() and (height,width)==(128,160): npu_path=base/'motip_npu_backbone_outputs.npz'
npu=np.load(npu_path,allow_pickle=False)
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
def sess(name):return ort.InferenceSession(str(base/name),sess_options=opts,providers=['CPUExecutionProvider'])
backbone=sess(f'motip_backbone_raw_{height}x{width}_probe.onnx')
remainder=sess(f'motip_detector_remainder_{height}x{width}_probe.onnx')
names=[x.name for x in backbone.get_outputs()]
mean=np.array([.485,.456,.406],np.float32).reshape(1,3,1,1)
std=np.array([.229,.224,.225],np.float32).reshape(1,3,1,1)
rows=[]
for ix in data['test_indices']:
    ix=int(ix);raw=data['images'][ix].transpose(2,0,1)[None].astype(np.float32)
    float_features=backbone.run(None,{'image':raw})
    quant_features=[];feature_errors=[]
    for j, ref in enumerate(float_features):
        got=npu[f'frame_{ix}_level_{j}'].reshape(ref.shape).astype(np.float32)
        quant_features.append(got)
        feature_errors.append(dict(mae=float(np.mean(np.abs(ref-got))),
                                   ref_abs_mean=float(np.mean(np.abs(ref))),
                                   max_abs=float(np.max(np.abs(ref-got)))))
    mask=np.zeros((1,height,width),np.bool_)
    def det(fs):
        d=dict(zip(names,fs));d['mask']=mask
        return remainder.run(None,{i.name:d[i.name] for i in remainder.get_inputs()})
    ref=det(float_features);got=det(quant_features)
    ref_scores=1/(1+np.exp(-ref[0]));got_scores=1/(1+np.exp(-got[0]))
    rows.append(dict(index=ix,feature_errors=feature_errors,
                     logit_mae=float(np.mean(np.abs(ref[0]-got[0]))),
                     box_mae_normalized=float(np.mean(np.abs(ref[1]-got[1]))),
                     float_candidates_05=int((ref_scores>.5).sum()),npu_candidates_05=int((got_scores>.5).sum()),
                     score_max_float=float(ref_scores.max()),score_max_npu=float(got_scores.max()),
                     same_threshold_queries=bool(np.array_equal(ref_scores>.5,got_scores>.5))))
result=dict(scope=f'four related held frames, {height}x{width}; NPU backbone + PC float remainder; not tracking or accuracy evaluation',
            calibration='8 frames; same sequences as test; 16-bit test conversion divide 256',frames=rows)
(root/f'motip_npu_vs_float_{height}x{width}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
