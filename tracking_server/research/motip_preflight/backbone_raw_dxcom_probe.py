"""PTQ diagnostic on eight saved frames, 128x160 backbone only, not deployable."""
import json
import os
import traceback
from pathlib import Path
import dx_com
import numpy as np
import torch

base = Path(os.environ['MOTIP_ARTIFACTS'])
height = int(os.environ.get('MOTIP_PROBE_H','128'))
width = int(os.environ.get('MOTIP_PROBE_W','160'))
output = base / f'dxcom_backbone_raw_{height}x{width}_probe_v1'
output.mkdir(parents=True, exist_ok=False)
data = np.load(base / f'motip_backbone_{height}x{width}_calib_probe.npz', allow_pickle=False)


class Frames(torch.utils.data.Dataset):
    def __len__(self):
        return len(data['calibration_indices'])

    def __getitem__(self, index):
        i = data['calibration_indices'][index]
        return torch.from_numpy(data['images'][i].transpose(2,0,1).copy()).float()


result = dict(scope=f'{height}x{width} raw-input backbone; 8-frame diagnostic calibration only',
              compiler=dx_com.__version__, success=False)
try:
    torch.set_num_threads(2)
    loader = torch.utils.data.DataLoader(Frames(), batch_size=1, num_workers=0)
    dx_com.compile(model=str(base/f'motip_backbone_raw_{height}x{width}_probe.onnx'),
                   output_dir=str(output), dataloader=loader, calibration_num=8,
                   calibration_method='ema', quantization_device='cpu', opt_level=0,
                   gen_log=True, export_html=True)
    result.update(success=True, files=[p.name for p in output.rglob('*') if p.is_file()])
except Exception as exc:
    result.update(error_type=type(exc).__name__, error=str(exc)[:1500])
    traceback.print_exc()
finally:
    (output/'probe_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
