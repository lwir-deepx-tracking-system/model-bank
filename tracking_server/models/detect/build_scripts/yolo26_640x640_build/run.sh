#!/bin/bash
K=<YOUR_PATH> cd $K/claude_bank_640_01 || exit 9
PY=<YOUR_PATH>
$PY -B build640.py yolo26s_nms_640x640 $K/claude_yolo26s_01/S_raw.pt > s.log 2>&1; echo "s exit $?"; grep -E "PARITY|DONE|Error" s.log | cut -c1-500 | tail -3
$PY -B build640.py yolo26m_flir_nms_640x640 $K/claude_ft_deploy_o1_01/NEWM/FT_raw.pt > m.log 2>&1; echo "m exit $?"; grep -E "PARITY|DONE|Error" m.log | cut -c1-500 | tail -3
