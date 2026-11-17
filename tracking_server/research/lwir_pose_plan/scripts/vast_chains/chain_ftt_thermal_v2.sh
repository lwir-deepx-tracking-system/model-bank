#!/bin/bash
# 2026-11-15 (user: "알아서 해"): FastTrackTr thermal fine-tune after the pose arms R0/R1.
# Data: TMOT train thermal (SportsMOT slot) + ~30% grey DanceTrack; from ckpt_b4amp_fix_last.pth (6 epochs, ID-loss fix); 3 epochs, LR 5e-5.
# --epochs and --outputs-dir are given on the command line: main.py's defaults (35, ./output) override the yaml otherwise.
# Afterwards: the same ID check as before (DanceTrack val, not trained on) and the device export.
cd /workspace/close_range_20261113; L=logs/chain_ftt_thermal.log   # 11-15 v2: 1 warm-up epoch (settings study, RT-DETR uses ~2000 steps), every epoch checkpoint gets the ID check
while ! grep -q CHAIN_R0R1_DONE logs/chain_R0R1.log 2>/dev/null; do sleep 60; done
[ -d ftt/datasets/SportsMOT/train ] || { echo "no TMOT data $(date +%T)" >> $L; exit 1; }
echo "start $(date +%T) seqs: $(ls ftt/datasets/SportsMOT/train | wc -l)" >> $L
cd ftt
export FTT_CODE=/workspace/close_range_20261113/ftt/fasttracktr FTT_KV_FROM=2 FTT_AMP=1 FTT_ID_FIX=1 FTT_ASPECT=1
/workspace/close_range_20261113/venv/bin/python -u /workspace/close_range_20261113/code/ftt_npu_train.py /workspace/close_range_20261113/ftt/npu_r18_thermal1.yaml /workspace/close_range_20261113/ftt/datasets/ --epochs 3 --outputs-dir /workspace/close_range_20261113/ftt/out_thermal1 > /workspace/close_range_20261113/logs/ftt_thermal1.log 2>&1
echo "train exit=$? $(date +%T)" >> /workspace/close_range_20261113/$L
cd /workspace/close_range_20261113
C=$(ls -t ftt/out_thermal1/checkpoint_*.pth 2>/dev/null | head -1); [ -n "$C" ] || { echo "no checkpoint" >> $L; exit 1; }
cp "$C" ftt/ckpt_thermal1_last.pth; echo "last $C" >> $L
cd ftt/fasttracktr
V=/workspace/close_range_20261113/ftt/datasets/DanceTrack/val
for ck in /workspace/close_range_20261113/ftt/out_thermal1/checkpoint_*.pth; do b=$(basename $ck .pth); echo "id check $b" >> /workspace/close_range_20261113/$L
  FTT_CHECK_FRAMES=400 /workspace/close_range_20261113/venv/bin/python -u /workspace/close_range_20261113/code/ftt_id_check.py /workspace/close_range_20261113/ftt/npu_r18_thermal1.yaml $ck /workspace/close_range_20261113/runs/ftt_id_check_thermal1_$b.json $V/* 2>&1 | grep -v DEBUG | grep "^dance\|DONE" >> /workspace/close_range_20261113/$L
done
/workspace/close_range_20261113/venv/bin/python -u /workspace/close_range_20261113/code/ftt_trained_export.py /workspace/close_range_20261113/ftt/npu_r18_thermal1.yaml /workspace/close_range_20261113/ftt/ckpt_thermal1_last.pth /workspace/close_range_20261113/ftt/ftt_cal /workspace/close_range_20261113/ftt/export_thermal1 2>&1 | grep -v "DEBUG\|TracerWarning\|grid_" >> /workspace/close_range_20261113/$L
echo "CHAIN_FTT_THERMAL_DONE $(date +%T)" >> /workspace/close_range_20261113/$L
