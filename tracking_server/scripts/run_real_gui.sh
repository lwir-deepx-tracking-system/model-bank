#!/bin/bash
# Starts the live server on the DX-M1: real camera + real gimbal + web page + PC GUI link. Detector --det w480 = YOLO26m_(FLIR) 640x480 (one model on the whole frame).
# Gimbal control values tuned on the real gimbal with a virtual target on 2026-11-10 (record: src/gimbal/PID_TUNING_RECORD_20261110.md):
#   PD, Kp 30 (tracker default), Ki 0, Kd 6, deadband 0.06. Arrow keys: manual speed 20.
# --live-wait 0: the page and the gimbal keys stay up even while the camera gives no frame. cam_pipe repairs a stalled camera itself (engine reset + reopen, about 30 s);
#   when cam_pipe ends, the server starts it again (at once after a clean end, after 60 s / 5 min / 15 min for repeated failures) and restarts a cam_pipe that stayed silent for 240 s.
# Board-specific values come from scripts/live.env (optional, not tracked by git; see scripts/live.env.example):
#   LWIR_LIVE_BIND   address of the web page. Default: the board's Tailscale IPv4 address. The page has no password and moves the gimbal.
#   LWIR_LIVE_PORT   port of the web page (default 8090)
#   LWIR_GUI_HOST    address of the team PC GUI (default 192.168.50.1)
#   LWIR_GIMBAL_DEV  serial device of the gimbal (default /dev/ttyUSB0)
#   LWIR_GIMBAL_SPEED_MAX     top speed of automatic following (default 100)
#   LWIR_GIMBAL_REACQUIRE_MS  how long the gimbal waits for a lost person before it gives up, in ms (default 10000)
# Normally started by scripts/keepalive.sh (cron). By hand, from any folder:  bash scripts/run_real_gui.sh   (messages: real_gui_1.out, results: real_gui_1/)
#
# 2026-11-11 (새 폴더 배치): 실행 파일은 start_tracking_server.py, 이 스크립트는 scripts/ 에 있다. 실행 폴더는 전처럼 실행 파일이 있는 폴더(tracking_server/)다.
#   판정기 파일은 models/track/ 에서, 짐벌 프로그램은 src/gimbal/ 에서 받는다.
# 2026-11-11 (장치의 실행 스크립트에 오늘 들어간 두 값을 옮김. 둘 다 짐벌 프로그램에 넘기는 --gimbal-opts 안에 있다):
#   --speed-max 100       자동으로 따라갈 때의 최고 속도를 10 에서 장치 한계 100 으로 올림. --gimbal-speed 10 이 넘기는 --speed-max 10 을 뒤에서 덮어쓴다.
#                         PD 값은 최고 속도 10 일 때 맞춘 것이다. 되돌리려면 scripts/live.env 에 LWIR_GIMBAL_SPEED_MAX=10.
#   --reacquire-ms 10000  사람을 놓친 뒤 다시 잡기를 기다리는 시간 10 초.
T="$(cd "$(dirname "$0")/.." && pwd)"
cd "$T" || exit 9
rm -rf real_gui_1
exec > real_gui_1.out 2>&1
[ -f scripts/live.env ] && . scripts/live.env
PORT="${LWIR_LIVE_PORT:-8090}"
BIND="${LWIR_LIVE_BIND:-$(ip -4 -o addr show tailscale0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)}"
GUI_HOST="${LWIR_GUI_HOST:-192.168.50.1}"
GIMBAL_DEV="${LWIR_GIMBAL_DEV:-/dev/ttyUSB0}"
SPEED_MAX="${LWIR_GIMBAL_SPEED_MAX:-100}"
REACQUIRE_MS="${LWIR_GIMBAL_REACQUIRE_MS:-10000}"
BRIDGE="${LWIR_GIMBAL_BRIDGE:-src/gimbal/gimbal_bridge_v25}"   # 짐벌 프로그램. v27(움직임 부드럽게, 시험용)을 쓰려면 scripts/live.env 에 LWIR_GIMBAL_BRIDGE=src/gimbal/gimbal_bridge_v27
EXTRA_OPTS="${LWIR_GIMBAL_EXTRA_OPTS:-}"                     # 짐벌 프로그램에 더 줄 옵션 (예: --deadband-exit 0.10 --axis-reversal 1)
[ -n "$BIND" ] || { echo "no page address: tailscale0 has no IPv4 address; set LWIR_LIVE_BIND in scripts/live.env"; exit 8; }
[ -x "$BRIDGE" ] || { echo "$BRIDGE missing: run scripts/setup.sh (needs OpenCV and GStreamer)"; exit 7; }
# 서버가 이미 떠 있으면 시작하지 않는다. 옛 이름(lwir_tracker_m1.py)으로 떠 있는 서버도 같은 NPU·카메라·짐벌을 쓰므로 함께 본다.
pgrep -f 'python3 -B ([s]tart_tracking_server|[l]wir_tracker_m1)\.py' > /dev/null && { echo "NPU busy"; exit 4; }
fuser "$GIMBAL_DEV" > /dev/null 2>&1 && { echo "gimbal busy"; exit 5; }
echo "START $(date +%T) page=$BIND:$PORT gui=$GUI_HOST gimbal=$GIMBAL_DEV speed_max=$SPEED_MAX reacquire_ms=$REACQUIRE_MS"
exec python3 -B start_tracking_server.py --tracker bytetrack --no-vit --det w480 --pipeline --class-gate --gmc --gmc-corners 300 --gmc-mindist 5 --n-init 3 --warm-min 45000 --edge-birth 0.5 \
  --relabel RVGGA0_REID2C256 --rl-async --gmc-thread --npu-depth 1 --confirm-gate models/track/gate_gru32_frozen.json --judge models/track/relink_judge_frozen.json --judge-crops 2 \
  --live-port "$PORT" --live-bind "$BIND" --live-seconds 604800 --live-wait 0 --out real_gui_1 \
  --gimbal "$GIMBAL_DEV" --gimbal-speed 10 --gimbal-bridge "$BRIDGE" --gimbal-opts "--controller pid --ki 0 --kd 6 --integral-limit 3 --deadband 0.06 --manual-speed 20 --speed-max $SPEED_MAX --reacquire-ms $REACQUIRE_MS $EXTRA_OPTS" \
  --gui-host "$GUI_HOST" --gui-bind 0.0.0.0
