#!/bin/bash
# 2026-11-11  Real gimbal + virtual person, THREE stages run one after another, a person reads the result in between (plan: PLAN_V27_HILS_20261111_KO.md).
#   bash run_v27_hil.sh check    looks only: files, plans, programs, server, pause file, who holds the gimbal. Changes nothing.
#   bash run_v27_hil.sh dry1 | dry2 | dry3
#                                no camera, no gimbal, the live server keeps running: the REAL bridge programs on a pseudo terminal + a pretend scene (1 to 2 min each).
#   bash run_v27_hil.sh stage1   THE GIMBAL MOVES. Identification: short timed moves for speed commands 1..100 on both axes (2 to 3 min, 5 min at most).
#   bash run_v27_hil.sh stage2   THE GIMBAL MOVES. PD candidates at speed cap 100, gimbal_bridge_v25 (7 to 8 min, 11 min at most). Needs plan_stage2.json (made from the stage 1 result).
#   bash run_v27_hil.sh stage3   THE GIMBAL MOVES. gimbal_bridge_v25 against v27 with the gains of stage 2 (10 to 12 min, 15 min at most). Needs plan_stage3.json.
# A stage pauses the keep-alive, stops the live server, runs the test and removes the pause file at the end, so that the keep-alive starts the server again within about a minute.
# IF THIS SCRIPT IS KILLED BY FORCE (kill -9) OR THE BOARD LOSES POWER DURING A STAGE, THE PAUSE FILE STAYS AND THE SERVER STAYS OFF (also after a reboot):
#   remove it by hand once you are sure no stage is running:   rm ~/lwir_keepalive.pause        (bash run_v27_hil.sh check  shows the file and who made it)
# STOP: the moment the test program has ended, for ANY reason, the first thing done is STOP written straight to the gimbal (send_stop.py). When the program has to be ended
#   (signal to this script, time limit) the order is (fourth round): SIGTERM to the program (it sends STOP itself and cuts its command line), STOP straight to the gimbal, and after 80 s without
#   an end SIGKILL to the program's whole session (program, bridge, camera program), then STOP again at once. A killed program closes its pseudo terminal, so the bridge's own STOP
#   can NOT reach the gimbal then (checked: review/test_tool_sigkill_linux.log); the STOP after a kill comes from send_stop.py alone.
# The live server is found by its exact command line AND its working folder and only that one is stopped. When ANY other process looks like a server (either file name anywhere in
# its command line, whatever folder), nothing is started. Earlier records are moved aside, never deleted.
# After the server is back, 예측 제어 and the detector choice are at their start values again (as after every server restart): this script does not set them.
# Fourth round (2026-11-12, third safety review): when the test program has to be ended, SIGTERM goes to it FIRST, then send_stop.py (was the other way round).
# Third round (2026-11-12, second safety review): send_stop.py is given 10 s (a time-out counts as "STOP not written": warning, exit code 7); the time-limit helper no longer
# signals this script when its sleep is killed at a normal end. BEFORE A STAGE (plan, section 6): stage 2 and 3 need 450 px free left-right and 250 px up-down; the background must not
# be a repeating pattern; the last 2 minutes of ~/lwir_keepalive.log must show no restart line.
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
MODE="${1:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
T="${HIL_ROOT:-$HOME/workspace/lwir_model_bank/tracking_server}"      # the server folder (same default as in pid_hil.py)
P="$HOME/lwir_keepalive.pause"
GRACE="${HIL_GRACE:-80}"                                                       # s the test program gets to end after SIGTERM (HIL_GRACE: only for testing this script)
cd "$HERE" || exit 9
# per stage: plan, record folder, the tool's own time limit (s), highest speed command the watchdog lets through
case "$MODE" in
  stage1|dry1) N=1; MAXS=300; SPD=100 ;;      # stage 1 has to send every command up to 100; each of its steps has its own guard = that step's speed
  stage2|dry2) N=2; MAXS=660; SPD=40 ;;       # following: Kp 30 x the largest error a block allows (240 px = 0.75) = 22.5, plus the D and lead terms -> 40 as this morning
  stage3|dry3) N=3; MAXS=900; SPD=40 ;;
  *) N=0; MAXS=0; SPD=0 ;;
esac
case "$MODE" in dry*) PLAN="plan_stage${N}_dry.json"; OUT="stage${N}_dry" ;; *) PLAN="plan_stage${N}.json"; OUT="stage${N}" ;; esac
LIMIT="${HIL_LIMIT:-$((MAXS + 120))}"                                          # this script's own time limit for the test program (HIL_LIMIT: only for testing this script)
TEST_CMD="${HIL_TEST_CMD:-python3 lead_hil.py --plan $PLAN --out $OUT --max-seconds $MAXS --spd-abort $SPD}"   # HIL_TEST_CMD: only for testing this script itself

# exact match: argv = python3 -B start_tracking_server.py ... and the process runs in the server folder
spid() {
  local p a
  for p in $(pgrep -f 'python3 -B [s]tart_tracking_server\.py'); do
    a="$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)"
    case "$a" in "python3 -B start_tracking_server.py "*) [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$T" ] && { echo "$p"; return; } ;; esac
  done
}
# loose match: every process whose command line names a server file (new or old name), except pid $1. One line each.
loose() {
  local p
  for p in $(pgrep -f '[s]tart_tracking_server\.py|[l]wir_tracker_m1\.py'); do
    [ "$p" = "${1:-}" ] || [ "$p" = "$$" ] || echo "pid $p in $(readlink /proc/$p/cwd 2>/dev/null): $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-110)"
  done
}
bridges() { pgrep -x gimbal_bridge; pgrep -x gimbal_bridge_v; }      # the kernel cuts process names to 15 characters: gimbal_bridge_v25 / _v27 show as gimbal_bridge_v
gimbal_dev() { ls /dev/serial/by-id/usb-Silicon_Labs_CP2102N_* 2>/dev/null | head -1; }
need() {
  local f ok=0
  for f in lead_hil.py pid_hil.py send_stop.py check_stop.py analyze_v27.py make_plan_v27.py plan_stage1.json plan_stage1_dry.json plan_stage2_dry.json plan_stage3_dry.json; do [ -f "$f" ] || { echo "MISSING file $HERE/$f"; ok=1; }; done
  for f in "$T/src/camera/cam_pipe" "$T/src/gimbal/gimbal_bridge_v25" "$T/src/gimbal/gimbal_bridge_v27"; do [ -x "$f" ] || { echo "MISSING program $f"; ok=1; }; done
  [ -f "$T/src/gimbal/gimbal_link.py" ] || { echo "MISSING $T/src/gimbal/gimbal_link.py (lead_shift is read from it)"; ok=1; }
  [ -n "${HIL_TEST_CMD:-}" ] || python3 -c 'import numpy, cv2' 2>/dev/null || { echo "python3 cannot import numpy and cv2"; ok=1; }
  return $ok
}
plan_note() { [ -f "$1" ] && python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))[0]; print("%s: stage %s, made %s, from %s%s" % (sys.argv[1], m.get("stage"), m.get("made"), m.get("from", "-"), "  ** ASSUMED NUMBERS: dry checks only **" if m.get("assumed") else ""))' "$1" 2>/dev/null || echo "$1: not there yet"; }

case "$MODE" in
check)
  need && echo "files and programs: all there"
  echo "server folder: $T"
  for f in plan_stage1.json plan_stage2.json plan_stage3.json; do plan_note "$f"; done
  echo "live server pid (exact match): $(spid)"
  echo "other processes that look like a server (a stage will NOT start while one is alive): $(o="$(loose "$(spid)")"; [ -n "$o" ] && echo && echo "$o" || echo none)"
  echo "camera program: $(pgrep -x cam_pipe | tr '\n' ' ')  bridge programs: $(bridges | tr '\n' ' ')"
  echo "pause file: $([ -e "$P" ] && echo "PRESENT ($(cat "$P" 2>/dev/null | head -1)) - while it is there the keep-alive does not start the server; if no stage is running remove it by hand: rm $P" || echo none)   keep-alive running: $(pgrep -f '[k]eepalive\.sh' | tr '\n' ' ')"
  G="$(gimbal_dev)"; echo "gimbal device: ${G:-no fixed name found} held by pid: $(fuser ${G:+"$G"} /dev/ttyUSB0 2>/dev/null | tr -s ' \n' ' ')"
  [ -f "$T/scripts/live.env" ] && grep -E '^LWIR_GIMBAL_(BRIDGE|EXTRA_OPTS|SPEED_MAX|REACQUIRE_MS|AIM_Y)' "$T/scripts/live.env"
  exit 0 ;;
dry1|dry2|dry3)
  need || exit 8
  [ -d "$OUT" ] && mv "$OUT" "${OUT}_before_$(date +%H%M%S)"
  [ -f "$OUT.log" ] && mv "$OUT.log" "${OUT}_before_$(date +%H%M%S).log"
  python3 lead_hil.py --plan "$PLAN" --out "$OUT" --dry --simCurve "1:158,2:160,5:172,10:355,20:600,50:800,100:900" --simL 0.12 --simTau 0.03 --max-seconds "$MAXS" --spd-abort "$SPD" > "$OUT.log" 2>&1; rc=$?
  python3 check_stop.py "$OUT" >> "$OUT.log" 2>&1; crc=$?
  tail -n 14 "$OUT.log"; echo "DRY EXIT $rc, check_stop $crc (log: $HERE/$OUT.log)   wanted: PLAN DONE, DRY EXIT 0, PASS"
  [ $rc = 0 ] && exit $crc; exit $rc ;;
stage1|stage2|stage3) ;;
*) echo "usage: bash run_v27_hil.sh check | dry1 | dry2 | dry3 | stage1 | stage2 | stage3      (stage1, stage2, stage3 move the gimbal)"; exit 1 ;;
esac

# ---------- a real stage ----------
exec 9> "$HERE/.run_v27_hil.lock"; flock -n 9 || { echo "another run_v27_hil.sh is running"; exit 10; }
echo $$ > "$HERE/.run_v27_hil.pid"            # to end a stage by hand:  kill -TERM "$(cat .run_v27_hil.pid)"   (STOP to the gimbal, the test ends, the pause file is removed)
need || exit 8
[ -f "$PLAN" ] || { echo "MISSING $PLAN: make it first (python3 make_plan_v27.py stage$N <folder of the stage before>). Nothing started."; exit 8; }
if [ -z "${HIL_TEST_CMD:-}" ]; then
  python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))[0]; sys.exit(0 if m.get("type") == "meta" and str(m.get("stage")) == sys.argv[2] and not m.get("assumed") else 1)' "$PLAN" "$N" \
    || { echo "$PLAN is not a stage $N plan made from a real result (it is marked as assumed, or belongs to another stage). Nothing started."; exit 8; }
fi
[ -e "$P" ] && { echo "pause file $P is already there ($(head -1 "$P" 2>/dev/null)): someone else paused the keep-alive, or an earlier stage was killed. Nothing started. If no stage is running: rm $P"; exit 7; }
o="$(loose "$(spid)")"                        # review: a server that the exact match does not find (other folder, other way of starting, old name) would fight for the camera and the gimbal
[ -n "$o" ] && { echo "a process that looks like a server is running and is NOT the live server of $T. Nothing touched, nothing started:"; echo "$o"; exit 6; }
# the records of earlier starts are kept (moved aside), never deleted
[ -f "${OUT}_run.log" ] && mv "${OUT}_run.log" "${OUT}_run_before_$(date +%H%M%S).log"
[ -d "$OUT" ] && mv "$OUT" "${OUT}_before_$(date +%H%M%S)"
exec > "${OUT}_run.log" 2>&1
HPID=""; SID=""; WPID=""; OWN_GIMBAL=0; STOP_RC=0; STOP_FAILED=0
stop_now() {                                  # STOP straight to the gimbal; only once the live server is gone (before that the gimbal is the server's)
  [ "$OWN_GIMBAL" = 1 ] || return 0
  local o; o="$(timeout -s KILL 10 python3 "$HERE/send_stop.py" ${HIL_STOP_DEV:+"$HIL_STOP_DEV"} 2>&1)"; STOP_RC=$?      # third round (second review): send_stop.py gets 10 s. Stuck longer (in opening the device) = ended and counted as NOT written; before, the script stood here for ever and the pause file stayed. KILL, not TERM: inside finish() this script ignores TERM and so would its child
  [ $STOP_RC = 124 ] || [ $STOP_RC = 137 ] && o="send_stop.py did not end within 10 s and was ended: STOP NOT WRITTEN $o"
  echo "STOP ($1) $(date +%T): $o"
  [ $STOP_RC = 0 ] || { STOP_FAILED=1; echo "WARNING: STOP WAS NOT WRITTEN TO THE GIMBAL ($1). GO TO THE GIMBAL."; }
}
alive() { [ -n "$(ps -o pid= -p "$1" 2>/dev/null)" ] && [ "$(ps -o stat= -p "$1" 2>/dev/null | cut -c1)" != Z ]; }
session_left() { [ -n "$SID" ] && pgrep -s "$SID" | tr '\n' ' '; }
sweep() {                                     # nothing that the test program started may stay behind (bridge, camera program): they are found by the program's own session, never by name
  local i left; left="$(session_left)"
  [ -z "$left" ] && { echo "no process of the test is left"; return; }
  echo "still alive after the test program: $(ps -o pid=,comm= -p $left 2>/dev/null | tr '\n' ';') waiting 5 s for them to end by themselves"
  for i in 1 2 3 4 5; do sleep 1; [ -z "$(session_left)" ] && break; done
  if [ -n "$(session_left)" ]; then
    echo "SIGTERM to: $(session_left) (the camera program closes the camera cleanly on SIGTERM; up to 45 s)"; pkill -TERM -s "$SID"
    for i in $(seq 1 45); do sleep 1; [ -z "$(session_left)" ] && break; done
  fi
  if [ -n "$(session_left)" ]; then echo "SIGKILL to: $(session_left) (the camera may need a reset at its next open)"; pkill -KILL -s "$SID"; sleep 1; fi
  stop_now "after the last process of the test was gone"
  left="$(session_left)"; [ -n "$left" ] && echo "LEFT BEHIND: $left" || echo "no process of the test is left"
}
end_tool() {                                  # the test program is still alive and has to end (signal to this script, time limit)
  local i
  kill -TERM "$HPID" 2>/dev/null; echo "SIGTERM to the test program $HPID $(date +%T): it sends STOP itself and cuts its command line; waiting up to $GRACE s"   # fourth round (third review, minor 5): first, so that the tool stops following at once (before, a stuck send_stop.py kept the tool following for up to 10 s)
  stop_now "after SIGTERM to the test program"     # fourth round: was before the SIGTERM ("before ending the test program"); the 10 s KILL limit stays
  for i in $(seq 1 "$GRACE"); do alive "$HPID" || break; sleep 1; done
  if alive "$HPID"; then
    echo "the test program ignored SIGTERM for $GRACE s: SIGKILL to its whole session $SID $(date +%T). The bridge cannot stop the gimbal now (its line to the gimbal went through the killed program): STOP follows at once"
    if [ -n "$SID" ]; then pkill -KILL -s "$SID"; else kill -KILL "$HPID" 2>/dev/null; fi
    stop_now "right after the kill"
    for i in 1 2 3 4 5; do alive "$HPID" || break; sleep 1; done
  fi
}
finish() {
  code=$?
  trap '' INT TERM HUP                        # the lines below must not be cut short
  [ -n "$WPID" ] && { pkill -P "$WPID" 2>/dev/null; kill "$WPID" 2>/dev/null; }
  if [ -n "$HPID" ]; then                     # the test was started and its end has not been handled yet
    alive "$HPID" && end_tool
    stop_now "right after the test program ended"      # before anything else
    sweep
  fi
  rm -f "$P"; echo "UNPAUSED $(date +%T) (pause file removed: $([ -e "$P" ] && echo NO || echo yes))"
  [ "$STOP_FAILED" = 1 ] && { echo "WARNING: AT LEAST ONE STOP COULD NOT BE WRITTEN TO THE GIMBAL (see above)."; [ "$code" = 0 ] && code=7; }
  echo "FINISHED code $code"
  exit "$code"
}
trap finish EXIT
trap 'echo "SIGNAL $(date +%T)"; exit 143' INT TERM HUP
echo "run_v27_hil.sh $MODE, pid $$, $(date '+%F %T')" > "$P"; echo "PAUSED $(date +%T)"
sleep 6                                       # the keep-alive looks at the pause file every 5 s: let it see the file before the server goes
pid="$(spid)"
if [ -n "$pid" ]; then
  echo "stopping the live server pid $pid: $(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-120)"
  kill -TERM "$pid"; for i in $(seq 1 30); do [ -z "$(spid)" ] && break; sleep 1; done
else echo "no live server found (exact match)"; fi
[ -n "$(spid)" ] && { echo "SERVER_STILL_ALIVE after 30 s (left alone)"; exit 2; }
o="$(loose "")"
[ -n "$o" ] && { echo "SERVER-LIKE PROCESS still alive (left alone), test not started:"; echo "$o"; exit 6; }
for i in $(seq 1 60); do pgrep -x cam_pipe > /dev/null || break; sleep 1; done      # the camera program ends by itself once the server is gone
pgrep -x cam_pipe > /dev/null && { echo "CAM_PIPE_STILL_ALIVE after 60 s (left alone)"; exit 3; }
for i in $(seq 1 15); do [ -z "$(bridges)" ] || { sleep 1; continue; }; break; done   # so does the bridge (its input closes)
[ -n "$(bridges)" ] && { echo "BRIDGE_STILL_ALIVE after 15 s (left alone): $(bridges | tr '\n' ' ')"; exit 4; }
G="$(gimbal_dev)"; h="$(fuser ${G:+"$G"} /dev/ttyUSB0 2>/dev/null | tr -s ' ')"
[ -n "$h" ] && { echo "GIMBAL DEVICE HELD by pid$h (left alone), test not started"; exit 5; }
sleep 3
OWN_GIMBAL=1
echo "HIL START $(date +%T): $TEST_CMD   (script time limit $LIMIT s)"
setsid $TEST_CMD &                            # its own session: everything it starts can be found again without going by names
HPID=$!
sleep 0.3; SID="$(ps -o sid= -p "$HPID" 2>/dev/null | tr -d ' ')"
[ "$SID" = "$(ps -o sid= -p $$ | tr -d ' ')" ] && { echo "note: the test program did not get its own session; only the program itself can be ended by force"; SID=""; }
( sleep "$LIMIT" && { echo "SCRIPT TIME LIMIT of $LIMIT s $(date +%T)"; kill -TERM $$; } ) &      # third round (second review): "&&": when the sleep is killed (the normal end kills it first) this helper sends nothing. With ";" it sent its signal in about 2 of 200 normal ends (exit code 143 instead of 0)
WPID=$!
wait "$HPID"; rc=$?
stop_now "right after the test program ended"    # before anything else, whatever the exit code
echo "HIL EXIT $rc $(date +%T)"
pkill -P "$WPID" 2>/dev/null; kill "$WPID" 2>/dev/null; WPID=""
sweep
HPID=""
if [ -z "${HIL_TEST_CMD:-}" ]; then
  python3 check_stop.py "$OUT"; crc=$?
  [ $rc = 0 ] && [ $crc != 0 ] && rc=11
  echo "next: python3 analyze_v27.py stage$N $OUT"
fi
[ $rc = 7 ] && echo "THE TEST PROGRAM SAYS ITS LAST EMERGENCY STOP WAS NOT WRITTEN, OR THAT THE VIEW WAS STILL MOVING AFTER THE LAST STOP, OR THAT THE END VIEW WAS NOT CHECKED (see its WARNING line in the log). The STOP lines above show whether send_stop.py reached the gimbal. GO TO THE GIMBAL."
exit $rc
