#!/bin/bash
# One-time setup on the DX-M1 (Orange Pi 5 Plus): build the native post-processing library and the camera helper, link the DXNN files into tracker/models/.
# Run from the final_tracker/tracker/ folder of a clone of the model bank:  bash setup.sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -shared -fPIC -o post.so post.cpp
ln -sf ../../dxnn/yolo26m_flir_rt480_a.dxnn models/NEWMRT480A.dxnn
ln -sf ../../dxnn/yolo26m_flir_rt480_b.dxnn models/NEWMRT480B.dxnn
ln -sf ../../dxnn/yolo26m_flir_whole480.dxnn models/NEWMRW480.dxnn
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
# Camera helper of the live server: needs the i3system camera SDK. Without the SDK only the live server is unavailable, file runs still work.
if [ -f /usr/local/include/i3system/i3system_TE.h ]; then
  g++ -O2 cam_pipe.cpp -o cam_pipe -D_LINUX -I/usr/local/include/i3system $(pkg-config --cflags --libs opencv4) -li3system_te_64 -li3system_usb_64 -li3system_imgproc_impl_64 -lpthread
  echo "cam_pipe built"
else
  echo "camera SDK not found (/usr/local/include/i3system): cam_pipe not built, the live server will not start"
fi
echo "setup done: post.so built, models linked and checked"
