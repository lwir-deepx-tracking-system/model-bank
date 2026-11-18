#!/bin/bash
# Install the native ModelBank PC launcher on the current user's desktop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
DESKTOP="${DESKTOP:-$HOME/Desktop}"
mkdir -p "$DESKTOP"
TARGET="$DESKTOP/ModelBank Direct LAN.desktop"
cat > "$TARGET" <<EOF
[Desktop Entry]
Type=Application
Name=ModelBank 직결 LAN
Comment=랜선으로 연결한 Orange Pi의 ModelBank를 PC 앱 창에서 조작
Exec=$ROOT/direct_lan_pc/launch.sh
Path=$ROOT/direct_lan_pc
Icon=camera-video
Terminal=false
Categories=Graphics;Utility;
EOF
chmod +x "$TARGET"
if command -v gio >/dev/null 2>&1; then
  gio set "$TARGET" metadata::trusted true >/dev/null 2>&1 || true
fi
echo "$TARGET"
