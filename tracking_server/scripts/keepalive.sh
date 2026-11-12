#!/bin/bash
# LWIR live server keep-alive v2 (2026-11-09). User requirement: once started, the tracker must never go off or stop.
# Runs forever; cron starts it every minute under flock, so there is never a second copy and a dead copy is replaced within a minute:
#   * * * * * flock -n $HOME/.lwir_keepalive.lock bash <clone>/tracking_server/scripts/keepalive.sh >> $HOME/lwir_keepalive.log 2>&1
# Every 5 s it reads GET /healthz of the live server (read-only; it never POSTs /gimbal/ping, which is the web page's safety heartbeat) and restarts the server when
#   - the python server process is missing (crash, killed, board rebooted),
#   - /healthz does not answer for 30 s (6 checks in a row),
#   - frames arrive but the tracker has published nothing for 30 s (main loop hung while the HTTP thread lives),
#   - gimbal_bridge is gone for 3 checks.
# The camera is repaired inside cam_pipe (ResetMainBoard + reopen, about 30 s) and the server backs off after repeated cam_pipe failures; this script does not touch the camera.
# Last rung, board reboot: only when ~/lwir_keepalive.allow_reboot exists, cam_pipe failed 3 rounds in a row (12 engine resets), no frame for 15 min, the board is up > 30 min,
# no camera experiment runs, at most 2 reboots per 24 h and 60 min apart. Counter in the home directory (not /tmp).
# Pause everything (deploys, experiments): create ~/lwir_keepalive.pause; delete it to resume. All timing uses /proc/uptime (the board has no RTC: the wall clock jumps after boot).
#
# Differences from the copy that runs on our board (the logic is otherwise the same):
#   - the server folder (tracking_server/) is taken from this script's location and the home folder from $HOME, not written into the script;
#   - the page address and port come from scripts/live.env or from the board's Tailscale address (same rule as scripts/run_real_gui.sh), not written into the script;
#   - stop_server also ends gimbal_bridge_v25 (the kernel shortens that process name to "gimbal_bridge_v"; the board copy only looks for "gimbal_bridge").
# 2026-11-11 (새 폴더 배치): 서버의 실행 파일 이름이 start_tracking_server.py 로 바뀌어 server_pid 가 그 이름을 찾는다. 이 스크립트와 live.env, run_real_gui.sh 는 scripts/ 에 있다.
#   옛 이름(lwir_tracker_m1.py)으로 떠 있는 서버는 이 스크립트가 찾지도 멈추지도 않는다: 옮기는 날에는 옛 keepalive 와 옛 서버를 먼저 멈춘 뒤 이 스크립트를 cron 에 건다.
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
H="${HOME:-/home/$(id -un)}"
T="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$T/scripts/live.env" ] && . "$T/scripts/live.env"
PORT="${LWIR_LIVE_PORT:-8090}"
PAUSE=$H/lwir_keepalive.pause
ALLOW_REBOOT=$H/lwir_keepalive.allow_reboot
REBOOTS=$H/lwir_keepalive.reboots      # one wall-clock epoch per line
CRASHLOG=$H/lwir_crash.log
START_GRACE=180     # s after a server start during which hang checks are off (model load)
RESTART_BUDGET=3    # restarts per 10 min before a 10 min stand-down

up() { cut -d. -f1 /proc/uptime; }
log() { echo "$(date '+%F %T') up=$(up) $*"; }
server_pid() { pgrep -f 'python3 -B [s]tart_tracking_server.py' | head -1; }
# address the page binds to: LWIR_LIVE_BIND, else the Tailscale IPv4 address (empty until Tailscale is up)
bind_addr() { if [ -n "${LWIR_LIVE_BIND:-}" ]; then echo "$LWIR_LIVE_BIND"; else ip -4 -o addr show tailscale0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1; fi; }
probe_addr() { local a; a=$(bind_addr); [ "$a" = "0.0.0.0" ] && a=127.0.0.1; echo "$a"; }
health() { local a; a=$(probe_addr); [ -n "$a" ] && curl -s -m 4 "http://$a:$PORT/healthz"; }
jget() { echo "$1" | python3 -I -c "import json,sys
try: d=json.load(sys.stdin); v=d.get('$2'); print('' if v is None else (int(v) if isinstance(v,bool) else v))
except Exception: print('')" 2>/dev/null; }
addr_up() { local a; a=$(bind_addr); [ -n "$a" ] && { [ "$a" = "0.0.0.0" ] || ip -4 -o addr show 2>/dev/null | grep -q " $a/"; }; }
# the server can only start when the NPU driver, the NPU daemon and the address the page binds to exist
prereq_ok() { [ -e /dev/dxrt0 ] && pgrep -x dxrtd > /dev/null && addr_up; }
experiment_running() { pgrep -x cam_pipe_exp > /dev/null || pgrep -x cam_diag > /dev/null; }
# 2026-11-11: 카메라(i3system 0547:0080)가 USB 에 아예 안 보이면 재부팅해도 살아나지 않는다(두 번 확인). 그때는 재부팅하지 않는다: 선을 다시 꽂아야 한다.
cam_on_usb() { grep -qs 0547 /sys/bus/usb/devices/*/idVendor; }
# 2026-11-11: 팀의 NPU 측정 프로그램이 도는 동안에는 서버를 켜지 않고 기다린다 (같이 돌면 NPU 서비스가 죽는다).
npu_busy_other() { pgrep -x lwir_benchmark > /dev/null; }

stop_server() {
  local pid; pid=$(server_pid)
  if [ -n "$pid" ]; then
    kill -TERM "$pid" 2>/dev/null
    for i in $(seq 1 25); do [ -z "$(server_pid)" ] && break; sleep 1; done
    if [ -n "$(server_pid)" ]; then log "server ignored SIGTERM for 25 s: SIGKILL (server only)"; kill -KILL "$(server_pid)" 2>/dev/null; sleep 1; fi
  fi
  for i in $(seq 1 40); do pgrep -x cam_pipe > /dev/null || break; sleep 1; done   # cam_pipe ends by itself once python is gone (PDEATHSIG, EPIPE)
  if pgrep -x cam_pipe > /dev/null; then
    log "cam_pipe still alive 40 s after the server: SIGTERM (clean CloseTE)"; pkill -TERM -x cam_pipe
    for i in $(seq 1 30); do pgrep -x cam_pipe > /dev/null || break; sleep 1; done
    if pgrep -x cam_pipe > /dev/null; then log "cam_pipe ignored SIGTERM for 30 s: SIGKILL as the last resort (camera may need an engine reset at the next open)"; pkill -KILL -x cam_pipe; sleep 1; fi
  fi
  for b in gimbal_bridge gimbal_bridge_v; do   # process names are cut to 15 characters: gimbal_bridge_v25 shows as gimbal_bridge_v
    if pgrep -x "$b" > /dev/null; then pkill -TERM -x "$b"; sleep 2; fi
  done
  for i in $(seq 1 20); do fuser /dev/dxrt0 > /dev/null 2>&1 || break; sleep 1; done
  sleep 3
}

start_server() {
  cd "$T" || return 1
  rm -rf real_gui_prev real_gui_prev.out
  [ -d real_gui_1 ] && mv real_gui_1 real_gui_prev
  [ -f real_gui_1.out ] && mv real_gui_1.out real_gui_prev.out
  setsid nohup bash scripts/run_real_gui.sh < /dev/null > /dev/null 2>&1 &
  for i in $(seq 1 90); do
    [ -n "$(health)" ] && return 0
    sleep 1
  done
  return 1
}

crash_note() {   # reason + last lines of the dying server for the record
  { echo "=== $(date '+%F %T') up=$(up) $1"; tail -n 4 "$T/real_gui_1.out" 2>/dev/null | cut -c1-200; grep -v CAMFPS "$T/real_gui_1/cam_pipe.log" 2>/dev/null | tail -n 3; } >> "$CRASHLOG"
}

log "keepalive v2 started (pid $$)"
fails=0; gbfails=0; hung=0; restarts=()
while true; do
  sleep 5
  if [ -e "$PAUSE" ]; then fails=0; gbfails=0; hung=0; continue; fi
  reason=""; reboot=0
  pid=$(server_pid)
  if [ -z "$pid" ]; then
    # The launcher retains its lock while it waits for DXRT between children.
    if ! flock -n "$T/real_gui_launch.lock" true; then continue; fi
    reason="server process missing"
  else
    h=$(health)
    if [ -n "$h" ]; then
      fails=0
      sup=$(jget "$h" up); cam_idle=$(jget "$h" cam_idle); out_idle=$(jget "$h" out_idle); cam_fail=$(jget "$h" cam_fail); gb=$(jget "$h" gimbal_alive)
      sup=${sup%.*}; cam_idle=${cam_idle%.*}; out_idle=${out_idle%.*}
      if [ -n "$sup" ] && [ "$sup" -gt $START_GRACE ]; then
        # frames flow but nothing is published: the tracker loop is stuck
        if [ -n "$cam_idle" ] && [ -n "$out_idle" ] && [ "$cam_idle" -lt 5 ] && [ "$out_idle" -gt 30 ]; then hung=$((hung + 1)); else hung=0; fi
        [ $hung -ge 2 ] && reason="tracker published nothing for ${out_idle} s while frames arrive"
        if [ -z "$reason" ]; then
          if [ "$gb" = "0" ]; then gbfails=$((gbfails + 1)); else gbfails=0; fi
          [ $gbfails -ge 3 ] && reason="gimbal_bridge missing"
        fi
        if [ -z "$reason" ] && [ -e "$ALLOW_REBOOT" ] && [ -n "$cam_fail" ] && [ "$cam_fail" -ge 3 ] && [ -n "$cam_idle" ] && [ "$cam_idle" -gt 900 ] && [ "$(up)" -gt 1800 ] && ! experiment_running && cam_on_usb; then
          now_epoch=$(date +%s); n24=0; last=0
          if [ "$(date +%Y)" -ge 2026 ] && [ -f "$REBOOTS" ]; then
            while read -r t; do [ -n "$t" ] && [ $((now_epoch - t)) -lt 86400 ] && n24=$((n24 + 1)) && last=$t; done < "$REBOOTS"
          fi
          if [ "$(date +%Y)" -ge 2026 ] && [ $n24 -lt 2 ] && [ $((now_epoch - last)) -gt 3600 ]; then reason="camera dead: $cam_fail cam_pipe failure rounds, no frame for ${cam_idle} s"; reboot=1; fi
        fi
      fi
    else
      fails=$((fails + 1))
      [ $fails -ge 6 ] && reason="no answer from /healthz for 30 s"
    fi
  fi
  [ -z "$reason" ] && continue
  if [ $reboot = 0 ] && [ -n "$reason" ] && npu_busy_other; then   # 팀 NPU 측정이 끝날 때까지 기다림 (다시 켜기 횟수에 넣지 않음)
    [ $(( $(up) % 60 )) -lt 5 ] && log "waiting: the team NPU benchmark is running ($reason)"
    continue
  fi
  if [ $reboot = 0 ] && ! prereq_ok; then   # right after a boot the NPU daemon or the page address may not be up yet: wait, do not count it as a restart
    [ $(( $(up) % 60 )) -lt 5 ] && log "waiting for the NPU driver / the page address before starting ($reason)"
    fails=0; continue
  fi

  now=$(up); recent=()
  for t in "${restarts[@]}"; do [ $((now - t)) -lt 600 ] && recent+=("$t"); done
  restarts=("${recent[@]}")
  if [ ${#restarts[@]} -ge $RESTART_BUDGET ]; then log "FAILED: $RESTART_BUDGET restarts in 10 min ($reason); standing down 10 min, then trying again"; crash_note "stand-down: $reason"; sleep 600; restarts=(); continue; fi
  restarts+=("$now")

  if [ $reboot = 1 ]; then
    log "REBOOT (last rung): $reason"; crash_note "reboot: $reason"; date +%s >> "$REBOOTS"; sync; systemctl reboot; sleep 180; continue
  fi
  [ -e "$PAUSE" ] && { log "pause file appeared, not restarting ($reason)"; continue; }   # re-check right before acting (race seen in the 11-10 injection test)
  log "restart: $reason"; crash_note "restart: $reason"
  stop_server
  if start_server; then log "server is up again (/healthz answers)"; else log "server gave no /healthz answer within 90 s after the start"; fi
  fails=0; gbfails=0; hung=0
done
