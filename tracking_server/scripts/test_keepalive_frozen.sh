#!/bin/bash
# 11-15: keepalive.sh's "camera dead" reboot rung never fires while the camera's USB chip is frozen (cam_pipe v2.12 exit 8 / CAMERA_FROZEN; a reboot does not
# cut the power of its port), even with ~/lwir_keepalive.allow_reboot, and it says why once; with a camera that is dead for another reason it still fires.
# Runs the real watchdog loop with mock /healthz answers and stubbed actions, the same way as test_keepalive_replay.sh. Needs bash and python3 (or python).
set -eo pipefail
watchdog="$(dirname "${BASH_SOURCE[0]}")/keepalive.sh"
main_loop='^log "keepalive v2 started'
python_executable=$(command -v python3 || command -v python)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
FROZEN_LINE="CAMERA_FROZEN at a stall after 14964 frames (control libusb_timeout, ResetMainBoard refused): the camera's USB chip does not answer; only a power cut (unplug ~10 s, or the 5 V step on a switchable port) clears it, at 20:33:22"

run_case() (
  healthz_extra=$1 cam_log=$2
  # The definitions are inert. Replace host-facing functions before entering the loop.
  source <(sed "/$main_loop/,$ d" "$watchdog")
  T=$work/t; mkdir -p "$T/real_gui_1"; rm -f "$T/real_gui_1/cam_pipe.log"
  [ -n "$cam_log" ] && printf '%b' "$cam_log" > "$T/real_gui_1/cam_pipe.log"
  PAUSE=/dev/null/keepalive-test-pause
  ALLOW_REBOOT=$work/allow_reboot; : > "$ALLOW_REBOOT"   # put back on purpose: the frozen camera must still not reboot the board
  REBOOTS=$work/reboots; rm -f "$REBOOTS"
  poll=0
  sleep() { poll=$((poll + 1)); if [ "$poll" -gt 4 ]; then exit 0; fi; }
  up() { echo 7200; }
  log() { printf 'LOG %s\n' "$*"; }
  server_pid() { echo 1234; }
  npu_busy_other() { return 1; }
  prereq_ok() { return 0; }
  netd_ensure() { :; }   # 11-22: never start the real lwir_netd from a test
  experiment_running() { return 1; }
  cam_on_usb() { return 0; }
  crash_note() { :; }
  sync() { :; }
  systemctl() { echo "REBOOT $*"; exit 0; }
  stop_server() { printf 'RESTART %s\n' "$reason"; exit 0; }
  start_server() { return 1; }
  probe_addr() { echo 127.0.0.1; }
  python3() { "$python_executable" "$@"; }
  curl() {
    case "${!#}" in
      */healthz) printf '{"up":7000,"cam_idle":950.0,"out_idle":950.0,"frames_in":0,"q_len":0,"replay":null,"cam_fail":21,"gimbal_alive":true%s}\n' "$healthz_extra" ;;
      *) exit 2 ;;
    esac
  }
  # The watchdog deliberately uses test commands that can return nonzero in its loop.
  set +e
  source <(sed -n "/$main_loop/,$ p" "$watchdog")
)

fail() { printf '%s\n' "$*" >&2; exit 1; }
no_reboot() {   # name, healthz extra, cam_pipe.log
  local out; out=$(run_case "$2" "$3")
  grep -q '^REBOOT' <<< "$out" && fail "$1: rebooted although the camera is frozen: $out"
  [ "$(grep -c '^LOG no reboot: the camera' <<< "$out")" = 1 ] || fail "$1: expected one 'no reboot' line (said once in 10 min), got: $out"
  return 0
}
reboots() {
  local out; out=$(run_case "$2" "$3")
  grep -q '^REBOOT reboot' <<< "$out" || fail "$1: the camera-dead rung did not fire: $out"
  grep -q 'no reboot' <<< "$out" && fail "$1: said 'no reboot' for a camera that is not frozen: $out"
  return 0
}

no_reboot 'server says frozen' ',"cam_frozen":true' ''
no_reboot 'older server, cam_pipe.log ends frozen' '' "CAMERA 640 480\nFRAMES 14964 in 547.98 s\n$FROZEN_LINE\nDIAG open: op=4(send) result=-7(libusb_timeout)\nOPEN_FAILED\n"
reboots 'server says not frozen' ',"cam_frozen":false' ''
reboots 'older server, camera opened again after the freeze' '' "$FROZEN_LINE\nCAMERA 640 480\nOPEN_FAILED\n"
reboots 'older server, no log' '' ''
printf '%s\n' '5 watchdog frozen-camera checks passed'
