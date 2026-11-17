#!/bin/bash
# test_temp_probe_run.sh (2026-11-14): runs temp_probe_run.sh against a FAKE server, a fake cam_pipe and a fake probe (Linux: flock,
# /proc/locks). It never touches the real server, camera, lock or /tmp/cam_temp_probe.tgz: own HOME and tgz in a temp folder, server
# on port 18090 (PORT=... to change). The fake server copies the replay rules that matter here: POST /replay/start sets busy at once,
# ends "cam_pipe" (which releases the lock), then plays; POST /replay/stop starts "cam_pipe" again, which exits 7 and is retried while
# someone else holds the lock.
# Cases: normal run, Ctrl-C during the probe, hang-up during the probe, someone else's replay, measurement running, gimbal following,
#        leftover replay of an interrupted run, start refused, server cannot end cam_pipe, no camera, --check.
# usage: bash test_temp_probe_run.sh        (PASS/FAIL per check; exit 0 = all passed; about 2 minutes)
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$HERE/temp_probe_run.sh"
PORT=${PORT:-18090}
T=$(mktemp -d)
trap 'kill $(jobs -p) 2> /dev/null; pkill -f "fake_cam_pipe $T" 2> /dev/null; rm -rf "$T"' EXIT
mkdir -p "$T/bin"
FAILS=0

cat > "$T/fake_server.py" << 'EOF'
import json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
port, home, scen, log = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
lock = os.path.join(home, '.cam_pipe.lock')
playing = scen in ('other_replay', 'leftover')
S = dict(on=playing, busy=False, name={'other_replay': 'someone/else', 'leftover': 'rec_b'}.get(scen), msg=None, frames=0, cam=None,
         cam_frames=False, cam_want=not playing, cam_t0=0.0)
L = threading.Lock()
LIST = [dict(name='rec_a', frames=40, usable=True), dict(name='rec_b', frames=107, usable=True),
        dict(name='측정/20261110/MEASURE_x', frames=70, usable=True), dict(name='rec_c', frames=300, usable=True)]
def note(s):
    with open(log, 'a') as f: f.write(s + '\n')
CAM = 'import fcntl, sys, time\nf = open(sys.argv[1], "a")\ntry: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\nexcept OSError: sys.exit(7)\ntime.sleep(1e6)\n'
def cam_loop():   # one "cam_pipe" while no replay plays; it exits 7 while someone else holds the lock and is started again
    while True:
        with L:
            if S['cam_want'] and not S['on'] and (S['cam'] is None or S['cam'].poll() is not None):
                if S['cam'] is not None and S['cam'].returncode == 7: note('cam_pipe exit 7 (lock busy)')
                S['cam'] = subprocess.Popen([sys.executable, '-c', CAM, lock, 'fake_cam_pipe ' + home])
            S['cam_frames'] = S['cam'] is not None and S['cam'].poll() is None and time.time() - S['cam_t0'] > 1
            if S['on'] or S['cam_frames']: S['frames'] += 3
        time.sleep(0.2)
def switch(tgt):
    time.sleep(1.0)
    if tgt is not None:
        if scen == 'start_fail':
            with L: S.update(busy=False, msg='camera program did not end in 45 s'); return
        if scen in ('slow_cam', 'stuck_cam'):   # the replay plays while the old cam_pipe still holds the lock (3 s / for ever)
            with L: S.update(cam_want=False, on=True, name=tgt, busy=False, msg='playing ' + tgt)
            if scen == 'stuck_cam': return
            time.sleep(3)
        with L:
            S['cam_want'] = False
            if S['cam'] is not None and S['cam'].poll() is None: S['cam'].terminate(); S['cam'].wait()
        time.sleep(1.5)
        with L: S.update(on=True, name=tgt, busy=False, msg='playing ' + tgt)
    else:
        with L: S.update(on=False, name=None, msg='stopped', cam_t0=time.time(), cam_want=True)
        time.sleep(0.5)
        with L: S['busy'] = False
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, d):
        b = json.dumps(d, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def state(self):
        return dict(on=S['on'], busy=S['busy'], name=S['name'], msg=dict(text=S['msg'], at='00:00:00') if S['msg'] else None)
    def do_GET(self):
        p = urlsplit(self.path).path
        with L:
            if p == '/replay': self.send(200, dict(available=True, list=LIST, state=self.state()))
            elif p == '/measure': self.send(200, dict(on=scen == 'measure'))
            elif p == '/gimbal/people': self.send(200, dict(state=dict(auto=1 if scen == 'gimbal_auto' else 0)))
            elif p == '/healthz': self.send(200, dict(replay=S['name'] if S['on'] else None, cam_idle=0.0 if (S['on'] or S['cam_frames']) else 5.0, frames_in=S['frames'], cam_fail=0, cam_pipe_alive=True))
            else: self.send(404, {})
    def do_POST(self):
        u = urlsplit(self.path); q = parse_qs(u.query)
        if self.headers.get('X-Gimbal') != '1': self.send_error(403); return
        note('POST %s %s' % (u.path, json.dumps({k: v[0] for k, v in q.items()}, ensure_ascii=False, sort_keys=True)))
        with L:
            if u.path == '/replay/start':
                if S['busy']: self.send(409, dict(ok=False, text='busy')); return
                if scen == 'refuse': self.send(400, dict(ok=False, text='fps out of range')); return
                S['busy'] = True; tgt = q['seq'][0]
            elif u.path == '/replay/stop':
                if not S['on']: self.send(409, dict(ok=False, text='not playing')); return
                if S['busy']: self.send(409, dict(ok=False, text='busy')); return
                S['busy'] = True; tgt = None
            else: self.send(404, {}); return
        threading.Thread(target=switch, args=(tgt,), daemon=True).start()
        self.send(200, dict(ok=True, text='ok', state=self.state()))
threading.Thread(target=cam_loop, daemon=True).start()
ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
EOF

cat > "$T/fake_probe" << 'EOF'
#!/usr/bin/env python3
# fake cam_temp_probe: same lock rule (exit 7), same kind of stdout lines, writes one frame, ends cleanly on SIGTERM/SIGINT
import fcntl, os, signal, sys, time
sys.stdout.reconfigure(line_buffering=True)
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
f = open(os.path.join(os.environ['HOME'], '.cam_pipe.lock'), 'a')
try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError: print('CAMERA_BUSY cam_pipe holds the camera (exit 7)'); sys.exit(7)
stop = []
signal.signal(signal.SIGTERM, lambda *a: stop.append(1)); signal.signal(signal.SIGINT, lambda *a: stop.append(1))
t0 = time.time()
print('CAMERA 640 480'); print('SHUTTER d set_mode 1 set_value 1 calibration 1')
with open(os.path.join(out, 'p2_temp_0.u16'), 'wb') as g: g.write(b'\x00' * 640 * 480 * 2)
steps = ['P1_AGC_OFF ok 60 fail 0 call_mean_ms 2.0 call_max_ms 4.0', 'P2_CALCTEMP 0 frame_ms 12.0 pixel_ms 0.02 centre_raw 7200 centre_C 22.00',
         'P3_AGC_ON_CALCTEMP 0 read_ms 3.0 frame_ms 12.5', 'P5_STREAM frames 290 fps 29.0 longest_gap_ms 50.0 calctemp_calls 29 calctemp_max_ms 13.0']
for s in steps:
    for _ in range(int(float(os.environ.get('FAKE_STEP_S', '1')) * 10)):
        if stop: break
        time.sleep(0.1)
    if stop: break
    print(s)
print('CLOSED total_s %.1f stopped_by_signal %d' % (time.time() - t0, 1 if stop else 0))
EOF
chmod +x "$T/fake_probe"
printf '#!/bin/sh\n[ "${FAKE_CAM:-1}" = 1 ] || exit 1\necho "Bus 004 Device 003: ID 0547:0080 Anchor Chips, Inc."\n' > "$T/bin/lsusb"; chmod +x "$T/bin/lsusb"

start_server() {   # start_server <scenario>
  SC=$1; HOME_T="$T/home_$1"; mkdir -p "$HOME_T/cam_temp_probe"; : > "$T/posts_$1.txt"
  python3 "$T/fake_server.py" "$PORT" "$HOME_T" "$1" "$T/posts_$1.txt" & SRV=$!
  for _ in $(seq 50); do curl -s -m 1 "http://127.0.0.1:$PORT/healthz" > /dev/null && break; sleep 0.1; done
  sleep 1.5   # let the fake cam_pipe take the lock
}
stop_server() { kill "$SRV" 2> /dev/null; wait "$SRV" 2> /dev/null; pkill -f "fake_cam_pipe $HOME_T" 2> /dev/null; sleep 0.3; }
run() {   # run [args...] -> RC; output in $T/out_<scenario>.txt
  HOME="$HOME_T" PATH="$T/bin:$PATH" PROBE_SERVER="http://127.0.0.1:$PORT" PROBE_BIN="$T/fake_probe" PROBE_WAIT_BACK_S=20 PROBE_TGZ="$T/probe_$SC.tgz" \
    bash "$RUNNER" "$@" > "$T/out_$SC.txt" 2>&1
  RC=$?
}
check() {   # check <what> <shell condition, evaluated here>
  if eval "$2"; then echo "PASS $SC: $1"; else echo "FAIL $SC: $1"; FAILS=$((FAILS + 1)); sed 's/^/    | /' "$T/out_$SC.txt" | tail -n 30; fi
}
holder() { local i; i=$(stat -c '%i' "$HOME_T/.cam_pipe.lock") || return; awk -v i="$i" '$2 != "->" { n = split($6, a, ":"); if (a[n] == i) { print $5; exit } }' /proc/locks; }
posts() { grep -c "POST $1" "$T/posts_$SC.txt"; }
nothing_posted() { [ ! -s "$T/posts_$SC.txt" ]; }

# 1. normal run: replay of the shortest label recording with >= 60 frames (rec_b), probe, stop, camera back, tgz
start_server normal
run "$HOME_T/cam_temp_probe/out_t1"
check "exit 0" '[ "$RC" = 0 ]'
check "started rec_b at 15 fps, repeat, python tracker" 'grep -q "POST /replay/start {\"backend\": \"python\", \"fps\": \"15\", \"loop\": \"1\", \"seq\": \"rec_b\"}" "$T/posts_$SC.txt"'
check "stopped the replay once" '[ "$(posts /replay/stop)" = 1 ]'
check "probe lines shown" 'grep -q "probe| P5_STREAM" "$T/out_$SC.txt"'
check "summary printed" 'grep -q "^.* P2_CALCTEMP 0 frame_ms" "$T/out_$SC.txt"'
check "camera back reported" 'grep -q "camera back" "$T/out_$SC.txt"'
check "tgz has probe output and run log" 'tar -tzf "$T/probe_$SC.tgz" | grep -q out_t1/probe_stdout.txt && tar -tzf "$T/probe_$SC.tgz" | grep -q out_t1/run_log.txt'
check "owner marker removed" '[ ! -e "$HOME_T/cam_temp_probe/replay_owner" ]'
check "the fake cam_pipe holds the lock again" '[ -n "$(holder)" ] && ! pgrep -f "fake_probe $HOME_T" > /dev/null'
stop_server

# 2. Ctrl-C (SIGINT) during the probe: probe ends cleanly, replay stopped, exit 130
start_server sigint
( sleep 9; pkill -INT -f "temp_probe_run.sh $HOME_T/cam_temp_probe/out_t2" ) &
FAKE_STEP_S=4 run "$HOME_T/cam_temp_probe/out_t2"
check "exit 130" '[ "$RC" = 130 ]'
check "probe stopped by the signal and closed" 'grep -q "CLOSED .* stopped_by_signal 1" "$HOME_T/cam_temp_probe/out_t2/probe_stdout.txt"'
check "replay stopped once, marker removed" '[ "$(posts /replay/stop)" = 1 ] && [ ! -e "$HOME_T/cam_temp_probe/replay_owner" ]'
check "no probe left running" '! pgrep -f "fake_probe $HOME_T" > /dev/null'
stop_server

# 3. hang-up (SIGHUP, as when the ssh connection drops) during the probe
start_server sighup
( sleep 9; pkill -HUP -f "temp_probe_run.sh $HOME_T/cam_temp_probe/out_t3" ) &
FAKE_STEP_S=4 run "$HOME_T/cam_temp_probe/out_t3"
check "exit 130 and replay stopped" '[ "$RC" = 130 ] && [ "$(posts /replay/stop)" = 1 ]'
stop_server

# 4. someone else's replay plays / 5. measurement running / 6. gimbal following: abort, nothing posted
for s in other_replay measure gimbal_auto; do
  start_server $s; run
  check "exit 3, nothing posted" '[ "$RC" = 3 ] && nothing_posted'
  stop_server
done

# 7. leftover replay of an interrupted run (the marker names the playing recording): reused (no new start), stopped at the end
start_server leftover
echo rec_b > "$HOME_T/cam_temp_probe/replay_owner"
run "$HOME_T/cam_temp_probe/out_t7"
check "exit 0, no new start, one stop" '[ "$RC" = 0 ] && [ "$(posts /replay/start)" = 0 ] && [ "$(posts /replay/stop)" = 1 ]'
stop_server

# 8. replay start refused (HTTP 400): exit 4, no stop, marker removed
start_server refuse
run
check "exit 4, no stop, marker removed" '[ "$RC" = 4 ] && [ "$(posts /replay/stop)" = 0 ] && [ ! -e "$HOME_T/cam_temp_probe/replay_owner" ]'
stop_server

# 9. the server could not end cam_pipe (the switch fails): exit 4 with the server's message, probe never ran
start_server start_fail
run
check "exit 4 with the server's message, no probe" '[ "$RC" = 4 ] && grep -q "did not start the replay: camera program did not end" "$T/out_$SC.txt" && ! grep -q "probe|" "$T/out_$SC.txt"'
stop_server

# 9b. the old cam_pipe keeps the lock 3 s after the replay plays: the script waits for the lock, then runs the probe
start_server slow_cam
run "$HOME_T/cam_temp_probe/out_t9b"
check "exit 0 after waiting for the lock" '[ "$RC" = 0 ] && grep -Eq "camera lock free \(after [2-9] s\)" "$T/out_$SC.txt" && ! grep -q "CAMERA_BUSY" "$T/out_$SC.txt"'
stop_server

# 9c. the lock is never released: exit 4 after PROBE_LOCK_WAIT_S, probe never ran, our replay stopped again
start_server stuck_cam
PROBE_LOCK_WAIT_S=4 run
check "exit 4, holder named, no probe, replay stopped" '[ "$RC" = 4 ] && grep -q "camera lock still held by pid" "$T/out_$SC.txt" && ! grep -q "probe|" "$T/out_$SC.txt" && [ "$(posts /replay/stop)" = 1 ]'
stop_server

# 10. no camera on USB: exit 2 before any server request
start_server nocam
FAKE_CAM=0 run
check "exit 2, nothing posted, clear message" '[ "$RC" = 2 ] && nothing_posted && grep -q "camera not on USB" "$T/out_$SC.txt"'
stop_server

# 11. --check: read only, chooses rec_b (shortest label recording with >= 60 frames; the measurement recording is skipped)
start_server check
run --check
check "exit 0, rec_b chosen, nothing posted" '[ "$RC" = 0 ] && grep -q "recording to loop while the probe runs: rec_b (107 frames" "$T/out_$SC.txt" && nothing_posted'
stop_server

if [ "$FAILS" = 0 ]; then echo "ALL PASSED"; else echo "$FAILS FAILED"; fi
exit $((FAILS > 0))
