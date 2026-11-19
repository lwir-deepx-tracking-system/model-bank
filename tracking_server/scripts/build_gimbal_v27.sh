#!/bin/bash
# Build the v27 bridge without touching a running v27 process.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src/gimbal"
target="${1:-$ROOT/src/gimbal/gimbal_bridge_v27}"
tmp="$(mktemp "$(dirname "$target")/.build_gimbal_v27.XXXXXX")"
trap 'rm -f -- "$tmp"' EXIT
read -r -a media_flags <<< "$(pkg-config --cflags opencv4 gstreamer-1.0 gstreamer-app-1.0)"
read -r -a media_libs <<< "$(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0)"
g++ -O2 -std=c++17 -pthread -Iv27/include -Iteam/include "${media_flags[@]}" gimbal_bridge_v27.cpp \
  v27/src/control/auto_tracking_controller.cpp v27/src/control/gimbal_command_executor.cpp \
  team/src/control/gimbal_device.cpp team/src/control/pid_controller.cpp team/src/control/tracking_command_handler.cpp \
  team/src/gui/gui_protocol.cpp team/src/gui/gui_sender.cpp team/src/gui/gui_receiver.cpp \
  team/src/gui/gui_command_handler.cpp team/src/common/logger.cpp "${media_libs[@]}" \
  -lopencv_core -lopencv_imgproc -o "$tmp"
chmod 755 "$tmp"
mv -f -- "$tmp" "$target"
echo "gimbal_bridge_v27 built"
