#!/bin/bash
# WSL (dx_com env). YOLO26s 640x480 six-head DXNN (2026-11-09). The YOLO26m_(FLIR) 640x480 file was built earlier by claude_ft_deploy_o1_01/build_split6.py (window RW480) and is copied unchanged.
K="<YOUR_PATH>"; cd $K/claude_bank_640_01 || exit 9
PY="<YOUR_PATH>"
$PY -B build480.py yolo26s_nms_640x480 $K/claude_yolo26s_01/S_raw.pt > s480.log 2>&1; echo "s480 exit $?"; grep -E "PARITY|DONE|Error" s480.log | cut -c1-500 | tail -3
