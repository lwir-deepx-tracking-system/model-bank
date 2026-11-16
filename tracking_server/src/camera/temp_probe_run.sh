#!/usr/bin/env bash
# temp_probe_run.sh (2026-11-14): one command on the device for the camera temperature test (cam_temp_probe.cpp in this folder).
# Kept behaviour: the server, cam_pipe and the replay code are not changed. The script only presses the page's own replay buttons
# (POST /replay/start, /replay/stop), so the server itself ends cam_pipe and later starts it again, exactly like a person on the page.
# Steps:
#   1. camera on USB? (lsusb 0547:0080) - else a clear message, exit 2. Probe binary present? - else exit 2.
#   2. is the device free? a replay that is not ours, a measurement recording, or the gimbal following someone -> exit 3, nothing changed
#   3. free the camera: start a short looping replay (the shortest usable label recording with >= 60 frames, or PROBE_REPLAY_SEQ)
#      at 15 fps, wait until it plays, then wait <= 60 s until no process holds ~/.cam_pipe.lock (read from /proc/locks: this
#      script never takes the lock itself, so it never makes cam_pipe fail)
#   4. run the probe (timeout 120 s; its stdout -> <outdir>/probe_stdout.txt, shown live)
#   5. ALWAYS (exit, Ctrl-C, hang-up, error): stop the probe if it still runs (it closes the camera on SIGTERM), then stop OUR replay
#      (only if the recording playing is still ours) -> the server starts cam_pipe again; then wait <= 90 s for camera frames
#   6. tar <outdir> -> /tmp/cam_temp_probe.tgz and print the summary lines (P1/P2/P3/P5)
# A replay started by an interrupted earlier run is recognised by the marker file ~/cam_temp_probe/replay_owner and reused/stopped.
# Usage: temp_probe_run.sh [--check] [outdir]          (default outdir ~/cam_temp_probe/out_<date>_<time>)
#   --check   read only: camera / server / lock / chosen recording; starts nothing (exit 0 = a run would start now)
# Env: PROBE_TELEMETRY=1 (probe phase P6; any other value = off), PROBE_REPLAY_SEQ (recording to loop), PROBE_SERVER (default
#      http://127.0.0.1:8090), PROBE_TIMEOUT_S (120), PROBE_BIN (~/cam_temp_probe/cam_temp_probe), PROBE_WAIT_BACK_S (90, 0 = skip),
#      PROBE_IGNORE_GIMBAL=1 (run even when the gimbal follows someone: starting a replay sends the gimbal a stop),
#      PROBE_TGZ (/tmp/cam_temp_probe.tgz), PROBE_LOCK_WAIT_S (60)
# Exit: 0 probe ran, 2 no camera / no probe binary / SDK found no camera, 3 device in use (nothing changed),
#       4 could not free the camera, 5 probe failed or timed out, 6 another run of this script is active
set -u
trap '' PIPE                     # a closed ssh pipe must not kill the script before it gives the camera back
export PYTHONIOENCODING=utf-8 LC_ALL=C.UTF-8

BASE="$HOME/cam_temp_probe"
BIN="${PROBE_BIN:-$BASE/cam_temp_probe}"
SERVER="${PROBE_SERVER:-http://127.0.0.1:8090}"
ORIGIN="${PROBE_ORIGIN:-http://localhost:8090}"
TIMEOUT_S="${PROBE_TIMEOUT_S:-120}"
WAIT_BACK_S="${PROBE_WAIT_BACK_S:-90}"
LOCK_WAIT_S="${PROBE_LOCK_WAIT_S:-60}"
FPS=15
LOCK="$HOME/.cam_pipe.lock"
OWNER="$BASE/replay_owner"
TGZ="${PROBE_TGZ:-/tmp/cam_temp_probe.tgz}"
USB_ID=0547:0080

CHECK=0
case "${1:-}" in
  --check) CHECK=1; shift ;;
  -h|--help) awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac
OUT="${1:-$BASE/out_$(date +%Y%m%d_%H%M%S)}"

LOG=""; PROBE_PID=""; PROBE_RC=""; STARTED=0; OUR_SEQ=""; CLEANED=0; INTERRUPTED=0; STOPPED_REPLAY=0
R_ON=0; R_BUSY=0; R_AVAIL=0; R_NAME="-"; R_MSG=""
mkdir -p "$BASE"

say() {
  local m; m="$(date +%H:%M:%S) $*"
  printf '%s\n' "$m" 2>/dev/null || true
  if [ -n "$LOG" ]; then printf '%s\n' "$m" >> "$LOG" 2>/dev/null || true; fi
}

# ---- server helpers (GET is read only; POST only /replay/start and /replay/stop) ----
http_get() { curl -s -m "${2:-10}" "$SERVER$1" 2>/dev/null; }

read_state() {   # sets R_ON R_BUSY R_AVAIL (0/1) R_NAME ('-' = none) R_MSG; returns 1 if the server does not answer
  local line flags
  line=$(http_get /replay 20 | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
s = d.get("state") or {}
m = (s.get("msg") or {}).get("text") or ""
print("%d %d %d\t%s\t%s" % (bool(s.get("on")), bool(s.get("busy")), bool(d.get("available")), s.get("name") or "-", m.replace("\t", " ").replace("\n", " ")))
' 2>/dev/null)
  [ -n "$line" ] || return 1
  IFS=$'\t' read -r flags R_NAME R_MSG <<< "$line"
  read -r R_ON R_BUSY R_AVAIL <<< "$flags"
  return 0
}

choose_seq() {   # the recording to loop: PROBE_REPLAY_SEQ if usable, else the shortest usable label recording with >= 60 frames
  http_get /replay 20 | python3 -c '
import json, os, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
items = [i for i in d.get("list") or [] if i.get("usable")]
want = os.environ.get("PROBE_REPLAY_SEQ") or ""
if want:
    c = [i for i in items if i.get("name") == want]
else:   # label recordings (PNG folders); the measurement recordings ("\uce21\uc815/...", one big gz file) are skipped
    c = [i for i in items if not str(i.get("name", "")).startswith("\uce21\uc815/") and (i.get("frames") or 0) >= 60]
    c.sort(key=lambda i: (i.get("frames") or 0, i.get("name")))
print("%s\t%s" % (c[0]["name"], c[0].get("frames")) if c else "")
sys.exit(0 if c else 3)
' 2>/dev/null
}

json_field() {   # json_field <path> <python expression over d>; empty on any error
  http_get "$1" 10 | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(eval(sys.argv[1], {}, {"d": d}))
except Exception:
    pass
' "$2" 2>/dev/null
}

urlenc() { python3 -c 'import sys, urllib.parse as u; print(u.quote(sys.argv[1], safe=""))' "$1"; }

post() {   # post <path?query> -> prints the HTTP code ("000" = no answer); body in $BASE/.last_post.json
  local code
  code=$(curl -s -m 20 -o "$BASE/.last_post.json" -w '%{http_code}' -X POST -H 'X-Gimbal: 1' -H "Origin: $ORIGIN" "$SERVER$1" 2>/dev/null)
  printf '%s\n' "${code:-000}"
}
post_text() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("text", ""))' "$BASE/.last_post.json" 2>/dev/null; }

lock_holder() {   # PID that holds ~/.cam_pipe.lock (from /proc/locks; never takes the lock), empty = free
  # Matched by inode: the device field of /proc/locks is not always the number stat shows (seen on WSL2). When the holder's open
  # files are readable it must have this very file open; otherwise (other user, or the locker already handed the fd to a child)
  # the inode match counts, which can only make the script wait, never run the probe next to cam_pipe.
  local ino real pid
  ino=$(stat -L -c '%i' "$LOCK" 2>/dev/null) || return 0
  real=$(readlink -f "$LOCK")
  for pid in $(awk -v i="$ino" '$2 != "->" { n = split($6, a, ":"); if (a[n] == i) print $5 }' /proc/locks 2>/dev/null); do
    if [ -r "/proc/$pid/fd" ] && ! ls -l "/proc/$pid/fd" 2>/dev/null | grep -qF -- " -> $real"; then continue; fi
    printf '%s\n' "$pid"; return 0
  done
}
pid_name() { ps -o comm= -p "$1" 2>/dev/null || echo '?'; }

# ---- the steps ----
FAIL_CODE=""; FAIL_MSG=""
fail() {   # fail <code> <message>: in a run, stop here (the exit trap gives back what was taken); with --check, note it and go on
  [ -z "$FAIL_CODE" ] && { FAIL_CODE=$1; FAIL_MSG=$2; }
  say "  STOP: $2"
  [ "$CHECK" = 1 ] || exit "$1"
}

wait_replay_on() {   # <= 90 s: the server ends cam_pipe (it waits up to 45 s for it), pauses 1.5 s, starts the replay program
  local t0=$SECONDS   # on failure STARTED stays 1: stop_our_replay checks the playing name and leaves other recordings alone
  while [ $((SECONDS - t0)) -lt 90 ]; do
    if read_state; then
      if [ "$R_ON" = 1 ] && [ "$R_BUSY" = 0 ]; then
        if [ "$R_NAME" = "$OUR_SEQ" ]; then say "replay playing: $R_NAME (after $((SECONDS - t0)) s)"; return 0; fi
        say "a different recording started meanwhile ($R_NAME): not ours, left alone"; return 1
      fi
      if [ "$R_ON" = 0 ] && [ "$R_BUSY" = 0 ]; then say "the server did not start the replay: $R_MSG"; return 1; fi
    fi
    sleep 1
  done
  say "the replay did not start within 90 s (server message: $R_MSG)"; return 1
}

wait_lock_free() {   # wait_lock_free <seconds>
  local t0=$SECONDS pid=""
  while [ $((SECONDS - t0)) -lt "$1" ]; do
    pid=$(lock_holder)
    if [ -z "$pid" ]; then say "camera lock free (after $((SECONDS - t0)) s)"; return 0; fi
    sleep 1
  done
  say "camera lock still held by pid $pid ($(pid_name "$pid")) after $1 s"; return 1
}

show_new_lines() {   # print probe stdout lines not shown yet; $1 = lines already shown -> echoes the new count
  local n
  n=$(wc -l < "$OUT/probe_stdout.txt" 2>/dev/null) || n=0
  if [ "$n" -gt "$1" ]; then
    sed -n "$(($1 + 1)),${n}p" "$OUT/probe_stdout.txt" | while IFS= read -r l; do say "  probe| $l" >&2; done
  fi
  printf '%s\n' "$n"
}

run_probe_once() {   # sets PROBE_RC; the probe runs in the background so a Ctrl-C or hang-up reaches the trap at once
  local shown=0 buf=()
  command -v stdbuf > /dev/null && buf=(stdbuf -oL)
  # timeout sends SIGTERM (the probe then closes the camera itself); SIGKILL 20 s later only if it ignores that
  timeout -s TERM -k 20 "$TIMEOUT_S" "${buf[@]}" "$BIN" "$OUT" > "$OUT/probe_stdout.txt" 2> "$OUT/probe_stderr.txt" 9>&- &
  PROBE_PID=$!
  while kill -0 "$PROBE_PID" 2>/dev/null; do
    sleep 1 & wait $!
    shown=$(show_new_lines "$shown")
  done
  wait "$PROBE_PID"; PROBE_RC=$?
  PROBE_PID=""
  show_new_lines "$shown" > /dev/null
}

stop_our_replay() {
  local t0=$SECONDS code warned=0
  while [ $((SECONDS - t0)) -lt 90 ]; do
    if ! read_state; then
      [ $warned = 0 ] && say "server does not answer; retrying the replay stop for up to 90 s"; warned=1; sleep 2; continue
    fi
    if [ "$R_BUSY" = 1 ]; then sleep 1; continue; fi
    if [ "$R_ON" = 0 ]; then say "no replay playing any more (nothing to stop)"; rm -f "$OWNER"; STARTED=0; return 0; fi
    if [ "$R_NAME" != "$OUR_SEQ" ]; then say "the recording playing now ($R_NAME) is not ours: left alone"; rm -f "$OWNER"; STARTED=0; return 0; fi
    code=$(post /replay/stop)
    if [ "$code" = 200 ]; then
      say "replay stopped: the server starts cam_pipe again (picture in 10-60 s)"; rm -f "$OWNER"; STARTED=0; STOPPED_REPLAY=1; return 0
    fi
    say "replay stop answered HTTP $code ($(post_text)); retrying"; sleep 2
  done
  say "WARNING: could not stop the replay '$OUR_SEQ' in 90 s. Press stop in the page's replay panel - the camera stays off until then."
  return 1
}

wait_camera_back() {   # informational: the server reads camera frames again (frames_in grows, cam_idle small, no replay)
  local t0=$SECONDS line rep idle fin fin0="" fail=""
  [ "$1" -gt 0 ] || return 0
  while [ $((SECONDS - t0)) -lt "$1" ]; do
    line=$(json_field /healthz '"%d %d %d %s" % (d.get("replay") is not None, int(d.get("cam_idle") or 0), int(d.get("frames_in") or 0), d.get("cam_fail"))')
    if [ -n "$line" ]; then
      read -r rep idle fin fail <<< "$line"
      if [ "$rep" = 0 ]; then
        [ -z "$fin0" ] && fin0=$fin
        if [ "$fin" -gt $((fin0 + 10)) ] && [ "$idle" -lt 2 ]; then say "camera back: the server reads camera frames again (after $((SECONDS - t0)) s)"; return 0; fi
      fi
    fi
    sleep 2
  done
  say "camera frames not back within $1 s (cam_fail ${fail:-?}); the server keeps retrying by itself. If the page stays dark, use its camera reset button."
}

summary() {
  [ -f "$OUT/probe_stdout.txt" ] || return 0
  say "---- summary ($OUT) ----"
  grep -E '^(NO_CAMERA_FOUND|CAMERA_BUSY|OPEN_FAILED|SAVE_FAILED|CAMERA |SHUTTER|P1_AGC_OFF|P2_CALCTEMP|P3_AGC_ON_CALCTEMP|P[0-9]_READ_FAILED|P5_STREAM|P6_TELEMETRY|CLOSED)' "$OUT/probe_stdout.txt" \
    | while IFS= read -r l; do say "  $l"; done
}

pack() {
  [ -d "$OUT" ] || return 0
  if tar -C "$(dirname "$OUT")" -czf "$TGZ.part" "$(basename "$OUT")" 2>/dev/null && mv -f "$TGZ.part" "$TGZ"; then
    say "packed $(du -h "$TGZ" | cut -f1) -> $TGZ (folder $(basename "$OUT"))"
  else
    say "tar failed for $OUT"
  fi
}

cleanup() {   # exit trap: runs once, also after Ctrl-C / hang-up / an error; further signals are ignored until it is done
  local rc=$? i
  [ "$CLEANED" = 1 ] && return; CLEANED=1
  trap '' INT TERM HUP
  if [ -n "$PROBE_PID" ] && kill -0 "$PROBE_PID" 2>/dev/null; then
    say "stopping the probe (SIGTERM: it closes the camera itself)"
    kill -TERM "$PROBE_PID" 2>/dev/null
    for i in $(seq 25); do kill -0 "$PROBE_PID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$PROBE_PID" 2>/dev/null; then
      say "probe still running after 25 s: SIGKILL"; pkill -KILL -P "$PROBE_PID" 2>/dev/null; kill -KILL "$PROBE_PID" 2>/dev/null
    fi
  fi
  [ "$STARTED" = 1 ] && stop_our_replay
  if [ "$INTERRUPTED" = 0 ] && [ "$STOPPED_REPLAY" = 1 ]; then wait_camera_back "$WAIT_BACK_S"; fi
  if [ -n "$LOG" ]; then summary; pack; fi
  say "finished (exit $rc)"
  exit "$rc"
}

# ---- preflight (the same checks for --check and for a run) ----
if [ "$CHECK" = 0 ]; then
  exec 9> "$BASE/.run.lock"
  if ! flock -n 9; then say "another temp_probe_run.sh is running (exit 6)"; exit 6; fi
fi
say "temperature probe runner (check only: $CHECK)"
if [ ! -x "$BIN" ]; then
  fail 2 "probe binary missing: $BIN (build it in ~/cam_temp_probe with the g++ line in cam_temp_probe.cpp)"
fi
if lsusb -d "$USB_ID" > /dev/null 2>&1; then
  say "camera on USB: $(lsusb -d "$USB_ID" | head -n 1)"
else
  fail 2 "camera not on USB (lsusb $USB_ID not found). Plug it in (USB3 port), wait about 10 s, run again."
fi

SERVER_UP=0; LEFTOVER=0; SEQ=""
if read_state; then
  SERVER_UP=1
  own=$(cat "$OWNER" 2>/dev/null || true)
  say "server answers: replay on=$R_ON busy=$R_BUSY name=$R_NAME available=$R_AVAIL"
  if [ "$R_BUSY" = 1 ]; then
    fail 3 "the server is switching the camera/replay right now (someone pressed a button); try again in a minute"
  elif [ "$R_ON" = 1 ]; then
    if [ -n "$own" ] && [ "$R_NAME" = "$own" ]; then
      LEFTOVER=1; say "the replay playing ($R_NAME) was started by an interrupted earlier run of this script: reusing it"
    else
      fail 3 "a replay is playing ($R_NAME): someone else may be using the device. Nothing changed. If it is not needed, press stop in the page's replay panel and run again."
    fi
  elif [ -n "$own" ] && [ "$CHECK" = 0 ]; then
    rm -f "$OWNER"   # stale marker: no replay plays
  fi
  meas=$(json_field /measure 'int(bool(d.get("on")))')
  [ "$meas" = 1 ] && fail 3 "a measurement recording is running on the page: someone is using the device. Nothing changed."
  auto=$(json_field /gimbal/people 'int((d.get("state") or {}).get("auto") or 0)')
  if [ -n "$auto" ] && [ "$auto" != 0 ]; then
    if [ "${PROBE_IGNORE_GIMBAL:-0}" = 1 ]; then say "the gimbal follows someone (auto=$auto); PROBE_IGNORE_GIMBAL=1: going on (the replay start stops the gimbal)"
    else fail 3 "the gimbal is following someone (auto=$auto); starting a replay would stop it. Stop following on the page, or set PROBE_IGNORE_GIMBAL=1."; fi
  fi
  if [ "$LEFTOVER" = 0 ]; then
    if [ "$R_AVAIL" != 1 ]; then
      fail 4 "replay is not available on this server (no --replay-root): cannot free the camera"
    else
      pick=$(choose_seq)
      SEQ=${pick%%$'\t'*}
      if [ -z "$SEQ" ]; then fail 4 "no usable recording to loop${PROBE_REPLAY_SEQ:+ (PROBE_REPLAY_SEQ=$PROBE_REPLAY_SEQ not in the list)}"
      else say "recording to loop while the probe runs: $SEQ (${pick#*$'\t'} frames, $FPS fps, repeat)"; fi
    fi
  fi
else
  holder=$(lock_holder)
  if [ -n "$holder" ]; then
    fail 4 "server at $SERVER does not answer and pid $holder ($(pid_name "$holder")) holds the camera lock"
  else
    say "server at $SERVER does not answer and nobody holds the camera lock: the probe runs directly (no replay needed)"
  fi
fi

if [ "$CHECK" = 1 ]; then
  holder=$(lock_holder)
  say "camera lock now: ${holder:+held by pid $holder ($(pid_name "$holder"))}${holder:-free}"
  [ "$SERVER_UP" = 1 ] && say "server input: $(json_field /healthz '"last frame %s s ago, cam_fail %s, camera-slot program alive %s (replay %s)" % (d.get("cam_idle"), d.get("cam_fail"), d.get("cam_pipe_alive"), d.get("replay"))')"
  if [ -z "$FAIL_CODE" ]; then say "CHECK OK: a run would start now"; exit 0; fi
  say "CHECK: a run would stop at: $FAIL_MSG (exit $FAIL_CODE)"; exit "$FAIL_CODE"
fi

# ---- the run ----
mkdir -p "$OUT"; LOG="$OUT/run_log.txt"
trap cleanup EXIT
trap 'INTERRUPTED=1; say "interrupted: giving the camera back"; exit 130' INT TERM HUP
say "output folder: $OUT"
if [ "${PROBE_TELEMETRY:-0}" = 1 ]; then export PROBE_TELEMETRY=1; say "PROBE_TELEMETRY=1: phase P6 on (stalls the stream ~2 s)"; else unset PROBE_TELEMETRY; fi

if [ "$SERVER_UP" = 1 ]; then
  if [ "$LEFTOVER" = 1 ]; then
    OUR_SEQ=$R_NAME; STARTED=1
  else
    OUR_SEQ=$SEQ; printf '%s\n' "$SEQ" > "$OWNER"
    STARTED=1     # set before the request: a Ctrl-C during it still ends in the stop (which checks the playing name first)
    code=$(post "/replay/start?seq=$(urlenc "$SEQ")&fps=$FPS&loop=1&backend=python")
    if [ "$code" != 200 ] && [ "$code" != 000 ]; then
      STARTED=0; rm -f "$OWNER"; say "replay start refused: HTTP $code ($(post_text))"; exit 4
    fi
    say "replay start: HTTP $code ($(post_text))"
  fi
  wait_replay_on || exit 4
fi
wait_lock_free "$LOCK_WAIT_S" || exit 4

say "running the probe (about 30-80 s). A person should stand 1-3 m in front of the camera, face visible, until it ends."
for attempt in 1 2 3; do
  run_probe_once
  [ "$PROBE_RC" = 7 ] || break       # 7 = the lock was taken between our check and the probe's start: wait and try again
  cat "$OUT/probe_stdout.txt" >> "$LOG" 2>/dev/null
  say "probe attempt $attempt: camera lock busy (exit 7)"; sleep 3; wait_lock_free 30 || break
done
case "$PROBE_RC" in
  0) say "probe finished (exit 0)"; exit 0 ;;
  2) say "probe: the SDK found no camera although lsusb lists it (exit 2)"; exit 2 ;;
  7) say "probe: camera lock stayed busy (exit 7)"; exit 4 ;;
  124) say "probe timed out after $TIMEOUT_S s (stopped with SIGTERM)"; exit 5 ;;
  *) say "probe failed (exit $PROBE_RC); see $OUT/probe_stderr.txt"; exit 5 ;;
esac
