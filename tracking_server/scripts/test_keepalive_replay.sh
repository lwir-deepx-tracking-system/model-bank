#!/bin/bash
# Exercise the real watchdog loop with mock HTTP responses and restart actions.
set -eo pipefail
watchdog="$(dirname "${BASH_SOURCE[0]}")/keepalive.sh"
main_loop='^log "keepalive v2 started'
python_executable=$(command -v python3 || command -v python)

run_case() (
  expected=$1 replay_state=$2 input=$3
  # The definitions are inert. Replace host-facing functions before entering the loop.
  source <(sed "/$main_loop/,$ d" "$watchdog")
  PAUSE=/dev/null/keepalive-test-pause
  ALLOW_REBOOT=/dev/null/keepalive-test-reboot
  poll=0
  sleep() { poll=$((poll + 1)); if [ "$poll" -gt 4 ]; then exit 0; fi; }
  up() { echo 3600; }
  log() { :; }
  server_pid() { echo 1234; }
  npu_busy_other() { return 1; }
  prereq_ok() { return 0; }
  crash_note() { :; }
  stop_server() { printf '%s\n' "$reason"; exit 0; }
  start_server() { return 1; }
  probe_addr() { echo 127.0.0.1; }
  python3() { "$python_executable" "$@"; }
  curl() {
    case "${!#}" in
      */healthz)
        replay='"sample"' frames=42 queue=0 out_idle=$((30 + poll * 5))
        case "$input" in
          live) replay=null ;;
          advancing) frames=$((frames + poll)) ;;
          queued) queue=1 ;;
          healthy) out_idle=1 ;;
        esac
        printf '{"up":300,"cam_idle":0.2,"out_idle":%s,"replay":%s,"frames_in":%s,"q_len":%s,"gimbal_alive":true}\n' \
          "$out_idle" "$replay" "$frames" "$queue"
        ;;
      */replay)
        [ "$input" != live ] || { echo 'unexpected replay request for live camera'; exit 2; }
        printf '%s\n' "$replay_state"
        ;;
      *) exit 2 ;;
    esac
  }
  # The watchdog deliberately uses test commands that can return nonzero in its loop.
  set +e
  source <(sed -n "/$main_loop/,$ p" "$watchdog")
)

check() {
  local expected=$1 state=$2 input=$3 result
  result=$(run_case "$@")
  if [ "$expected" = restart ]; then
    [[ "$result" == 'tracker published nothing for 40 s while frames arrive' ]] || {
      printf 'Expected a tracker restart (%s), received: %s\n' "$input" "$result" >&2
      exit 1
    }
  else
    [ -z "$result" ] || { printf 'Unexpected restart (%s): %s\n' "$input" "$result" >&2; exit 1; }
  fi
}

paused='{"state":{"on":true,"busy":false,"name":"sample","paused":true,"seeking":false,"loop":false,"total":100,"pos":20}}'
ended='{"state":{"on":true,"busy":false,"name":"sample","paused":false,"seeking":false,"loop":false,"total":100,"pos":99}}'
playing=${paused/\"paused\":true/\"paused\":false}
check hold "$paused" still
check hold "$ended" still
check restart "$playing" still
check restart "${ended/\"loop\":false/\"loop\":true}" still
check restart "${paused/\"seeking\":false/\"seeking\":true}" still
check restart "${ended/sample/different}" still
check restart '{"state":null}' still
check restart "$ended" advancing
check restart "$ended" queued
check restart '' live
check hold "$playing" healthy
printf '%s\n' '11 watchdog replay and live checks passed'
