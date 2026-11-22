#!/bin/sh
# LWIR connect for macOS / Linux: find the Orange Pi tracking server and open its page (no typing of addresses).
# Order: 10.42.0.1:8090, orangepi.local, UDP beacon / discover (needs python3), link-local 169.254.42.1.
#   sh lwir_connect.sh            find and open        sh lwir_connect.sh --no-open   only print the address
PORT=8090
OPEN=1
[ "$1" = "--no-open" ] && OPEN=0

ok() {   # is this the LWIR server? (/netinfo of this version, /healthz of an older one)
  curl -s -m 3 "http://$1:$PORT/netinfo" 2>/dev/null | grep -q '"client_local"' && return 0
  curl -s -m 3 "http://$1:$PORT/healthz" 2>/dev/null | grep -q '"up"'
}

found=""
echo "1) 10.42.0.1 ..."
ok 10.42.0.1 && found=10.42.0.1
if [ -z "$found" ]; then
  echo "2) orangepi.local ..."
  ok orangepi.local && found=orangepi.local
fi
if [ -z "$found" ] && command -v python3 >/dev/null 2>&1; then
  echo "3) UDP beacon (5 s) ..."
  addr=$(python3 - <<'PY'
import json, socket, time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(('', 47890))
except OSError:
    sock.bind(('', 0))
sock.settimeout(0.5)
end = time.time() + 5
while time.time() < end:
    try:
        sock.sendto(b'LWIR_DISCOVER', ('255.255.255.255', 47890))
        data, _ = sock.recvfrom(512)
        message = json.loads(data.decode())
        if message.get('lwir') == 1:
            print(message['addr'])
            break
    except (OSError, ValueError):
        pass
PY
)
  [ -n "$addr" ] && ok "$addr" && found="$addr"
fi
if [ -z "$found" ]; then
  echo "4) link-local 169.254.42.1 ..."
  ok 169.254.42.1 && found=169.254.42.1
fi
if [ -z "$found" ]; then
  echo "Board not found. Plug the PC into the board's second (direct) LAN port and try again in 10-30 s."
  exit 2
fi
url="http://$found:$PORT/"
echo "Found: $url"
if [ "$OPEN" = 1 ]; then
  if command -v open >/dev/null 2>&1; then open "$url"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url"; fi
fi
