"""Compile the tiny synthetic fixed-shape attention component only, not a model."""
import json
import traceback
from pathlib import Path
import os

import dx_com
import numpy as np
import torch

root = Path(__file__).resolve().parent
fixture = np.load(root / 'attention_reference_fixture.npz', allow_pickle=False)
output = (Path(os.environ['MOTIP_ARTIFACTS']) / 'dxcom_attention_probe_v1')
output.mkdir(parents=True, exist_ok=False)


class One(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return tuple(torch.from_numpy(fixture[k][0].copy()) for k in ('value', 'sampling', 'weights'))


result = dict(scope='synthetic tiny component only; no production calibration',
              compiler=dx_com.__version__, success=False)
try:
    loader = torch.utils.data.DataLoader(One(), batch_size=1, shuffle=False, num_workers=0)
    dx_com.compile(model=str(root / 'attention_reference_static.onnx'), output_dir=str(output),
                   dataloader=loader, calibration_num=1, calibration_method='ema',
                   quantization_device='cpu', opt_level=0, gen_log=True)
    files = sorted(p.name for p in output.rglob('*') if p.is_file())
    result.update(success=True, files=files)
except Exception as exc:
    result.update(error_type=type(exc).__name__, error=str(exc)[:1500])
    traceback.print_exc()
finally:
    (output / 'probe_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
