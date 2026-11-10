#!/bin/bash
# One-time setup on the DX-M1 (Orange Pi 5 Plus): build the native post-processing library, the camera helper and the gimbal + PC GUI helper, link the DXNN files into tracker/models/.
# Run from the final_tracker/tracker/ folder of a clone of the model bank:  bash setup.sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -shared -fPIC -o post.so post.cpp
ln -sf ../../dxnn/yolo26m_flir_rt480_a_nms_480x384.dxnn models/NEWMRT480A.dxnn
ln -sf ../../dxnn/yolo26m_flir_rt480_b_nms_480x384.dxnn models/NEWMRT480B.dxnn
ln -sf ../../dxnn/yolo26m_flir_whole480_nms_640x480.dxnn models/NEWMRW480.dxnn
ln -sf ../../dxnn/repvgg_tmot_camel_reid.dxnn models/RVGGA0_REID2C256.dxnn
# Detectors the live server can switch to (model list of the web page and of the PC GUI). These DXNN files are in the bank's yolo26/ folder.
ln -sf ../../../yolo26/yolo26s_nms_640x480.dxnn models/S480.dxnn
ln -sf ../../../yolo26/yolo26m_flir_nms_640x640.dxnn models/M640.dxnn
ln -sf ../../../yolo26/yolo26s_nms_640x640.dxnn models/S6.dxnn
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
# Gimbal + PC GUI helper of the live server: ../gimbal/gimbal_bridge_v25.cpp with the team's control and GUI code in ../gimbal/team (unchanged copy, see team/SOURCE.txt).
# Needs OpenCV core/imgproc and GStreamer (gstreamer-1.0, gstreamer-app-1.0). Without it --gimbal and --gui-host are unavailable, so run/run_real_gui.sh will not start.
if pkg-config --exists opencv4 gstreamer-1.0 gstreamer-app-1.0; then
  G=../gimbal
  g++ -O2 -std=c++17 -pthread -I$G/team/include $(pkg-config --cflags opencv4 gstreamer-1.0 gstreamer-app-1.0) $G/gimbal_bridge_v25.cpp \
    $G/team/src/control/auto_tracking_controller.cpp $G/team/src/control/gimbal_command_executor.cpp $G/team/src/control/gimbal_device.cpp $G/team/src/control/pid_controller.cpp \
    $G/team/src/control/tracking_command_handler.cpp $G/team/src/gui/gui_protocol.cpp $G/team/src/gui/gui_sender.cpp $G/team/src/gui/gui_receiver.cpp $G/team/src/gui/gui_command_handler.cpp \
    $G/team/src/common/logger.cpp $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0) -lopencv_core -lopencv_imgproc -o $G/gimbal_bridge_v25
  echo "gimbal_bridge_v25 built"
else
  echo "OpenCV (opencv4) or GStreamer not found: gimbal_bridge_v25 not built, --gimbal and --gui-host will not start"
fi
echo "setup done: post.so built, models linked and checked"
