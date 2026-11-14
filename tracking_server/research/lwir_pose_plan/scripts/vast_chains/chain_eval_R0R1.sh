#!/bin/bash
# 2026-11-15: after R0/R1, the MOT17 occlusion check (MOT17-09/11 grey, never trained on). last.pt of every run (best.pt is chosen by a
# noisy box score); E is scored twice (best = the 4-epoch file on the device, last = its final epoch).
cd /workspace/close_range_20261113; L=logs/chain_eval_R0R1.log
while ! grep -q CHAIN_R0R1_DONE logs/chain_R0R1.log 2>/dev/null; do sleep 60; done
echo "start $(date +%T)" >> $L
venv/bin/python code/eval_mot17_check.py data/mot17_gray data/mot17_kpt code/template_points.npy runs/eval_mot17_check_R0R1.json E_best=runs/expE_mot17_v1/weights/best.pt E_last=runs/expE_mot17_v1/weights/last.pt R0_last=runs/R0_E_cont_v1/weights/last.pt R1_last=runs/R1_llvippose_v2/weights/last.pt > logs/eval_mot17_R0R1.log 2>&1
echo "exit=$? $(date +%T)" >> $L; echo EVAL_R0R1_DONE >> $L
