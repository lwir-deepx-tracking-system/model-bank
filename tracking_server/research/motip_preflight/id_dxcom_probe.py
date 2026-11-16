"""Compile the small synthetic ID-head fixture only; not a deployable model."""
import json
import os
import traceback
from pathlib import Path
import dx_com
import numpy as np
import torch

root=Path(__file__).resolve().parent
base=Path(os.environ['MOTIP_ARTIFACTS'])
aggressive=os.environ.get('MOTIP_AGGRESSIVE_PARTITIONING')=='1'
output=base/('dxcom_id_fixed_probe_v2_aggressive' if aggressive else 'dxcom_id_fixed_probe_v1')
output.mkdir(parents=True,exist_ok=False)
data=np.load(root/'motip_id_fixture.npz',allow_pickle=False)
keys=['trajectory_features','unknown_features','trajectory_id_labels','trajectory_times',
      'unknown_times','trajectory_masks','unknown_masks']


class One(torch.utils.data.Dataset):
    def __len__(self):return 1
    def __getitem__(self,index):return tuple(torch.from_numpy(data[k][0].copy()) for k in keys)


result=dict(scope='synthetic fixed two-track ID head, not production calibration',
            compiler=dx_com.__version__,aggressive_partitioning=aggressive,success=False)
try:
    torch.set_num_threads(2)
    loader=torch.utils.data.DataLoader(One(),batch_size=1,num_workers=0)
    dx_com.compile(model=str(base/'motip_id_fixed_probe.onnx'),output_dir=str(output),
                   dataloader=loader,calibration_num=1,calibration_method='ema',
                   quantization_device='cpu',opt_level=0,aggressive_partitioning=aggressive,gen_log=True,export_html=True)
    result.update(success=True,files=[p.name for p in output.rglob('*') if p.is_file()])
except Exception as exc:
    result.update(error_type=type(exc).__name__,error=str(exc)[:1500]);traceback.print_exc()
finally:
    (output/'probe_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
