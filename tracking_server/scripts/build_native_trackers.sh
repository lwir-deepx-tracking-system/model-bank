#!/bin/bash
# Optional CPython C++ tracker extensions. Safe to call during setup: missing
# source files or a missing compiler/config helper do not break Python tracking.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACK="$ROOT/src/track"
OUT="$TRACK/_native"
mkdir -p "$OUT"

if ! command -v python3-config >/dev/null 2>&1; then
  echo "native trackers skipped: python3-config unavailable; Python fallback remains enabled"
  exit 0
fi
if ! command -v g++ >/dev/null 2>&1; then
  echo "native trackers skipped: g++ unavailable; Python fallback remains enabled"
  exit 0
fi

EXT_SUFFIX="$(python3-config --extension-suffix 2>/dev/null || true)"
PY_INCLUDES="$(python3-config --includes 2>/dev/null || true)"
if [ -z "$EXT_SUFFIX" ] || [ -z "$PY_INCLUDES" ]; then
  echo "native trackers skipped: python3-config did not provide extension build flags"
  exit 0
fi
# shellcheck disable=SC2206
PY_INCLUDE_ARGS=($PY_INCLUDES)
built=0
failed=0
for stem in bytetrack botsort ocsort sort; do
  source="$TRACK/$stem.cpp"
  target="$OUT/_native_${stem}${EXT_SUFFIX}"
  if [ ! -f "$source" ]; then
    echo "native $stem skipped: $stem.cpp not present"
    continue
  fi
  if grep -q "^// LWIR_BYTETRACK_C_ABI:" "$source"; then
    echo "native $stem uses the separate C ABI; see src/track/build_bytetrack.py"
    continue
  fi
  tmp="$(mktemp "$OUT/.build_${stem}.XXXXXX")" || exit 1
  if g++ -O3 -DNDEBUG -std=c++17 -shared -fPIC "${PY_INCLUDE_ARGS[@]}" \
      "$source" -o "$tmp" && mv -f -- "$tmp" "$target"; then
    echo "native $stem built: ${target##*/}"
    built=$((built + 1))
  else
    echo "native $stem build failed; Python fallback remains enabled"
    rm -f -- "$tmp"
    failed=$((failed + 1))
  fi
done
echo "native tracker build summary: $built built, $failed failed"
# Optional acceleration never prevents the Python server from starting.
exit 0
