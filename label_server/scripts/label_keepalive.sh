#!/bin/bash
# Keeps the label site on: cron runs this every minute; if nothing listens on the port it starts start_label_server.py
# with the line from README.md. Waits until the bind address exists (the server binds it) and the data disk is mounted.
# To stop the site on purpose: touch KEEP_OFF in the server folder  (remove the file to let it come back).
#
# 보드에서 돌던 파일과 다른 점 (하는 일은 같다): 서버 폴더, 라벨 폴더, 주소, 포트를 이 파일에 적지 않는다.
#   서버 폴더 = 이 파일이 있는 scripts 폴더의 바로 위 (start_label_server.py 가 있는 곳)
#   라벨 폴더 = LWIR_LABEL_ROOT, 없으면 서버 폴더 옆의 LWIR_labels
#   주소      = LWIR_LABEL_BIND, 없으면 이 보드의 Tailscale IPv4 주소 (tailscale0)
#   포트      = LWIR_LABEL_PORT, 없으면 8770
# 값을 바꾸려면 이 scripts 폴더에 label_server.env 를 둔다 (label_server.env.example 을 복사해 고친다. 저장소에는 올리지 않는다).
# KEEP_OFF 와 server.log 는 서버 폴더에 있다.
S="$(cd "$(dirname "$0")" && pwd)"
D="$(cd "$S/.." && pwd)"
[ -f "$D/start_label_server.py" ] || exit 0
[ -f "$D/KEEP_OFF" ] && exit 0
[ -f "$S/label_server.env" ] && . "$S/label_server.env"
ROOT="${LWIR_LABEL_ROOT:-$(cd "$D/.." && pwd)/LWIR_labels}"
PORT="${LWIR_LABEL_PORT:-8770}"
BIND="${LWIR_LABEL_BIND:-$(ip -4 -o addr show tailscale0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)}"
ss -ltn 2>/dev/null | grep -q ":$PORT " && exit 0
[ -n "$BIND" ] || exit 0
[ "$BIND" = "0.0.0.0" ] || ip -4 addr 2>/dev/null | grep -qF -- "$BIND/" || exit 0
if [ ! -d "$ROOT" ]; then
  echo "$(date '+%F %T') labels folder not found: $ROOT (set LWIR_LABEL_ROOT in scripts/label_server.env)"
  exit 0
fi
cd "$D" || exit 0
echo "$(date '+%F %T') label site not running -> start"
setsid nohup python3 start_label_server.py --root "$ROOT" --bind "$BIND" --port "$PORT" >> server.log 2>&1 < /dev/null &
