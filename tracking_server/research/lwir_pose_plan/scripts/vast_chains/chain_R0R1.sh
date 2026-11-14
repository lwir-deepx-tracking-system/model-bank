#!/bin/bash
# 2026-11-14 (user: follow the literature study, research_transfer_20261114/RECOMMENDATION.md): arms R0 and R1 from experiment E.
# R0 = E's mixture trained 10 more epochs (control); R1 = the same + LLVIP-Pose thermal (eyes/ears flag 3 = ignore). Same settings as E.
# Starts after the FastTrackTr fixed run and its ID check / export (logs/after_fix_run.log ends with AFTER_FIX_DONE).
cd /workspace/close_range_20261113; L=logs/chain_R0R1.log
while ! grep -q AFTER_FIX_DONE logs/after_fix_run.log 2>/dev/null; do sleep 60; done
C="--weights runs/expE_mot17_v1/weights/best.pt --project /workspace/close_range_20261113/runs --epochs 10 --batch 16 --imgsz 640 --device 0 --partial-p 0.5 --workers 8 --seed 0 --wandb off --fast 1 --patience 4 --template code/template_points.npy --occlude-p 0.5 --thermal-look 1 --lr0 0.0002 --warmup-epochs 1"
echo "start R0 $(date +%T)" >> $L
venv/bin/python code/train_pose.py --data code/otp2_coco_mot17_vast.yaml --name R0_E_cont_v1 $C > logs/R0.log 2>&1 < /dev/null
echo "R0 exit=$? $(date +%T)" >> $L
venv/bin/python code/train_pose.py --data code/otp2_coco_mot17_llvippose_vast.yaml --name R1_llvippose_v1 $C > logs/R1.log 2>&1 < /dev/null
echo "R1 exit=$? $(date +%T)" >> $L
echo CHAIN_R0R1_DONE >> $L
