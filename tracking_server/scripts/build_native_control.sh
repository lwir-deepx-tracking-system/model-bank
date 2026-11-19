#!/bin/bash
# Optional native calculations for the live gimbal control path.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if ! command -v g++ >/dev/null 2>&1 || ! command -v python3-config >/dev/null 2>&1; then
  echo "Native gimbal control skipped. The Python path is available."
  exit 0
fi
suffix="$(python3-config --extension-suffix)"
read -r -a includes <<< "$(python3-config --includes)"
target="$ROOT/src/gimbal/_native_control$suffix"
tmp="$(mktemp "$target.XXXXXX")"
trap 'rm -f -- "$tmp"' EXIT
g++ -O3 -std=c++17 -ffp-contract=off -shared -fPIC "${includes[@]}" \
  "$ROOT/src/gimbal/native_control.cpp" -o "$tmp"
chmod 755 "$tmp"
mv -f -- "$tmp" "$target"
echo "Native gimbal control built"
