#!/bin/bash
set -e
T="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$T/app.py" "$@"
