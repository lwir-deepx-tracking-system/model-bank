"""Safely extract only tensors from the official MOTIP checkpoint for the probes."""
import hashlib
import os
from pathlib import Path
import torch

base=Path(os.environ['MOTIP_ARTIFACTS'])
source=base/'motip_dancetrack_v01.pth'
target=base/'motip_dancetrack_model_only.pth'
expected='c642793c1f3d526aa8537d59bc6a3d201b44b16099076519ffad5ec379050800'
if target.exists():
    raise FileExistsError(f'Will not overwrite: {target}')
digest=hashlib.sha256()
with source.open('rb') as f:
    for chunk in iter(lambda:f.read(4*1024*1024),b''):
        digest.update(chunk)
if digest.hexdigest()!=expected:
    raise ValueError('Checkpoint differs from the verified v0.1 download')
checkpoint=torch.load(source,map_location='cpu',weights_only=True)
model=dict(checkpoint['model'])
assert len(model)==719 and all(isinstance(v,torch.Tensor) for v in model.values())
torch.save({'model':model},target)
print(f'Extracted {len(model)} tensors to {target}')
