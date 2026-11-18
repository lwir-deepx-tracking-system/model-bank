#!/bin/bash
# Orange Pi: run the same model-bank camera/tracker/gimbal in PC-only direct LAN mode.
set -e
export LWIR_GUI_MODE=direct
exec bash "$(dirname "$0")/run_real_gui.sh" "$@"
