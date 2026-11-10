#!/bin/bash
# Starts the live server on the DX-M1: real camera + real gimbal + web page + PC GUI link. Detector --det w480 = YOLO26m_(FLIR) 640x480 (one model on the whole frame).
# Gimbal control values tuned on the real gimbal with a virtual target on 2026-11-10 (record: ../gimbal/PID_TUNING_RECORD_20261110.md):
#   PD, Kp 30 (tracker default), Ki 0, Kd 6, deadband 0.06. Arrow keys: manual speed 20. Automatic tracking is capped at speed 10 (--gimbal-speed 10, a safety value).
# --live-wait 0: the page and the gimbal keys stay up even while the camera gives no frame. cam_pipe repairs a stalled camera itself (engine reset + reopen, about 30 s);
#   when cam_pipe ends, the server starts it again (at once after a clean end, after 60 s / 5 min / 15 min for repeated failures) and restarts a cam_pipe that stayed silent for 240 s.
# Board-specific values come from run/live.env (optional, not tracked by git; see run/live.env.example):
#   LWIR_LIVE_BIND   address of the web page. Default: the board's Tailscale IPv4 address. The page has no password and moves the gimbal.
#   LWIR_LIVE_PORT   port of the web page (default 8090)
#   LWIR_GUI_HOST    address of the team PC GUI (default 192.168.50.1)
#   LWIR_GIMBAL_DEV  serial device of the gimbal (default /dev/ttyUSB0)
# Normally started by run/keepalive.sh (cron). By hand, from any folder:  bash run/run_real_gui.sh   (messages: real_gui_1.out, results: real_gui_1/)
T="$(cd "$(dirname "$0")/.." && pwd)"
cd "$T" || exit 9
rm -rf real_gui_1
exec > real_gui_1.out 2>&1
[ -f run/live.env ] && . run/live.env
PORT="${LWIR_LIVE_PORT:-8090}"
BIND="${LWIR_LIVE_BIND:-$(ip -4 -o addr show tailscale0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)}"
GUI_HOST="${LWIR_GUI_HOST:-192.168.50.1}"
GIMBAL_DEV="${LWIR_GIMBAL_DEV:-/dev/ttyUSB0}"
[ -n "$BIND" ] || { echo "no page address: tailscale0 has no IPv4 address; set LWIR_LIVE_BIND in run/live.env"; exit 8; }
[ -x ../gimbal/gimbal_bridge_v25 ] || { echo "../gimbal/gimbal_bridge_v25 missing: run setup.sh (needs OpenCV and GStreamer)"; exit 7; }
pgrep -f "lwir_tracker_m1.py" > /dev/null && { echo "NPU busy"; exit 4; }
fuser "$GIMBAL_DEV" > /dev/null 2>&1 && { echo "gimbal busy"; exit 5; }
echo "START $(date +%T) page=$BIND:$PORT gui=$GUI_HOST gimbal=$GIMBAL_DEV"
exec python3 -B lwir_tracker_m1.py --tracker bytetrack --no-vit --det w480 --pipeline --class-gate --gmc --gmc-corners 300 --gmc-mindist 5 --n-init 3 --warm-min 45000 --edge-birth 0.5 \
  --relabel RVGGA0_REID2C256 --rl-async --gmc-thread --npu-depth 1 --confirm-gate gate_gru32_frozen.json --judge relink_judge_frozen.json --judge-crops 2 \
  --live-port "$PORT" --live-bind "$BIND" --live-seconds 604800 --live-wait 0 --out real_gui_1 \
  --gimbal "$GIMBAL_DEV" --gimbal-speed 10 --gimbal-bridge ../gimbal/gimbal_bridge_v25 --gimbal-opts "--controller pid --ki 0 --kd 6 --integral-limit 3 --deadband 0.06 --manual-speed 20" \
  --gui-host "$GUI_HOST" --gui-bind 0.0.0.0
