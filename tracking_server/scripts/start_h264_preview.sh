#!/bin/bash
# Optional preview only. Never terminates an existing port owner or tracker.
set -u
T="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$T/scripts/live.env" ] && . "$T/scripts/live.env"
BIND="${LWIR_LIVE_BIND:-$(ip -4 -o addr show tailscale0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)}"
PORT="${LWIR_LIVE_PORT:-8090}"
[ -n "$BIND" ] || { echo 'No Tailscale bind address'; exit 1; }
if ss -H -ltn 'sport = :18773' | grep -q .; then
  echo 'H.264 preview port 18773 is already in use; no process was changed.'
  exit 0
fi
cd "$T" || exit 1
LWIR_H264_BIND="$BIND" LWIR_LIVE_PORT="$PORT" nohup /usr/bin/python3 -B src/web/h264_preview_proxy.py >> h264_preview.out 2>&1 < /dev/null &
echo "H.264 preview started (PID $!); log: $T/h264_preview.out"
