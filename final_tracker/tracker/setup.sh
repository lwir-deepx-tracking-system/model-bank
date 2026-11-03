#!/bin/bash
# One-time setup on the DX-M1 (Orange Pi 5 Plus): build the native post-processing library and link the DXNN files into tracker/models/.
# Run from the final_tracker/tracker/ folder of a clone of the model bank:  bash setup.sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -shared -fPIC -o post.so post.cpp
ln -sf ../../dxnn/yolo26m_flir_rt480_a.dxnn models/NEWMRT480A.dxnn
ln -sf ../../dxnn/yolo26m_flir_rt480_b.dxnn models/NEWMRT480B.dxnn
ln -sf ../../dxnn/repvgg_tmot_camel_reid.dxnn models/RVGGA0_REID2C256.dxnn
python3 - <<'EOF'
import hashlib, json
from pathlib import Path
man = json.loads(Path('models/MANIFEST.json').read_text())
for n, h in man.items():
    if n.startswith('_'): continue
    got = hashlib.sha256(Path('models/%s.dxnn' % n).read_bytes()).hexdigest()
    print(n, 'OK' if got == h else 'MISMATCH', got[:12])
    assert got == h, n
EOF
echo "setup done: post.so built, models linked and checked"
