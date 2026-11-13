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
        replay='"sample"' frames=840 queue=0 out_idle=$((30 + poll * 5))
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

paused='{"state":{"on":true,"busy":false,"name":"sample","paused":true,"seeking":false,"loop":false,"total":100,"pos":20,"shown":21},"eval":{"on":true,"name":"sample","counts":{"received":21,"published":21,"server_skipped":0,"untagged":0}}}'
ended='{"state":{"on":true,"busy":false,"name":"sample","paused":false,"seeking":false,"loop":false,"total":100,"pos":99,"shown":100},"eval":{"on":true,"name":"sample","counts":{"received":100,"published":100,"server_skipped":0,"untagged":0}}}'
async_ended='{"state":{"on":true,"busy":false,"name":"sample","paused":false,"seeking":false,"loop":false,"total":280,"pos":null,"shown":280,"preview":"async"},"eval":{"on":true,"name":"sample","counts":{"received":280,"published":280,"server_skipped":0,"untagged":0}}}'
playing=${paused/\"paused\":true/\"paused\":false}
check hold "$paused" still
check hold "$ended" still
check hold "$async_ended" still
check hold "${paused/\"published\":21,\"server_skipped\":0/\"published\":20,\"server_skipped\":1}" still
check restart "${async_ended/\"published\":280/\"published\":279}" still
check restart "${paused/\"published\":21/\"published\":20}" still
check restart "${async_ended/\"received\":280,\"published\":280/\"received\":279,\"published\":279}" still
check restart "${async_ended/\"untagged\":0/\"untagged\":1}" still
check restart "${async_ended/\"eval\":/\"missing\":}" still
check restart "${async_ended/\"eval\":\{\"on\":true,\"name\":\"sample\"/\"eval\":\{\"on\":true,\"name\":\"different\"}" still
check restart "${async_ended/\"eval\":\{\"on\":true/\"eval\":\{\"on\":false}" still
check restart "$playing" still
check restart "${ended/\"loop\":false/\"loop\":true}" still
check restart "${paused/\"seeking\":false/\"seeking\":true}" still
check restart "${ended/sample/different}" still
check restart '{"state":null}' still
check restart "${async_ended/\"shown\":280/\"shown\":279}" still
check restart "${async_ended/\"pos\":null/\"pos\":278}" still
check restart "$async_ended" advancing
check restart "$async_ended" queued
check restart '' live
check hold "$playing" healthy
printf '%s\n' '22 watchdog replay and live checks passed'
