"""Synthetic calibration compile feasibility only, not an accurate deployable model."""
import json
import os
import traceback
from pathlib import Path
import dx_com
import numpy as np
import torch

aggressive = os.environ.get('MOTIP_AGGRESSIVE_PARTITIONING') == '1'
suffix = 'v2_aggressive' if aggressive else 'v1'
output = Path(os.environ['MOTIP_ARTIFACTS']) / ('dxcom_detector_128x160_probe_' + suffix)
output.mkdir(parents=True, exist_ok=False)
model = (Path(os.environ['MOTIP_ARTIFACTS']) / 'motip_detector_128x160_probe.onnx')


class One(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return (torch.from_numpy(np.zeros((3, 128, 160), dtype=np.float32)),
                torch.from_numpy(np.zeros((128, 160), dtype=np.bool_)))


result = dict(scope='synthetic 128x160 detector only, no production calibration',
              compiler=dx_com.__version__, aggressive_partitioning=aggressive, success=False)
try:
    torch.set_num_threads(2)
    loader = torch.utils.data.DataLoader(One(), batch_size=1, num_workers=0)
    dx_com.compile(model=str(model), output_dir=str(output), dataloader=loader,
                   calibration_num=1, calibration_method='ema', quantization_device='cpu',
                   opt_level=0, aggressive_partitioning=aggressive, gen_log=True, export_html=True)
    result.update(success=True, files=[p.name for p in output.rglob('*') if p.is_file()])
except Exception as exc:
    result.update(error_type=type(exc).__name__, error=str(exc)[:1500])
    traceback.print_exc()
finally:
    (output / 'probe_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
