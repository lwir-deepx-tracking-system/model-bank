#!/bin/bash
# 2026-11-14: after the fixed FastTrackTr run (run_b4amp_fix.sh) ends: ID check on the last checkpoint (same test as ckpt0/ckpt1),
# then the device export (ONNX + calibration) for the WSL DX-COM compile on the PC.
cd /workspace/close_range_20261113
L=logs/ftt_b4amp_fix.log
while ! grep -q "^exit=" $L; do sleep 60; done
echo "$(date +%T) training ended: $(grep '^exit=' $L)"
C=$(ls -t ftt/fasttracktr/output/checkpoint_*.pth | head -1); echo "last checkpoint $C"
cp "$C" ftt/ckpt_b4amp_fix_last.pth
cd ftt/fasttracktr
export FTT_CODE=/workspace/close_range_20261113/ftt/fasttracktr FTT_KV_FROM=2 FTT_CHECK_FRAMES=400
V=/workspace/close_range_20261113/ftt/datasets/DanceTrack/val
/workspace/close_range_20261113/venv/bin/python -u /workspace/close_range_20261113/code/ftt_id_check.py /workspace/close_range_20261113/ftt/npu_r18_dt480_w033_b4amp_fix.yaml /workspace/close_range_20261113/ftt/ckpt_b4amp_fix_last.pth /workspace/close_range_20261113/runs/ftt_id_check_fixlast.json $V/* 2>&1 | grep -v DEBUG
echo "$(date +%T) id check exit=${PIPESTATUS[0]}"
/workspace/close_range_20261113/venv/bin/python -u /workspace/close_range_20261113/code/ftt_trained_export.py /workspace/close_range_20261113/ftt/npu_r18_dt480_w033_b4amp_fix.yaml /workspace/close_range_20261113/ftt/ckpt_b4amp_fix_last.pth /workspace/close_range_20261113/ftt/ftt_cal /workspace/close_range_20261113/ftt/export_fixlast 2>&1 | grep -v DEBUG
echo "$(date +%T) export exit=${PIPESTATUS[0]}"
echo AFTER_FIX_DONE
