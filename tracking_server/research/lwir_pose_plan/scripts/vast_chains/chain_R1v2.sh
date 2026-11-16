#!/bin/bash
# 2026-11-15: R1 again with early stopping OFF (settings study: the stop rule adds the noisy body-box mAP, so runs stopped while the pose
# score was still rising and best.pt was worse than last.pt). Same data and settings as R0 otherwise; R0 also ran all 10 epochs.
cd /workspace/close_range_20261113; L=logs/chain_R0R1.log
C="--weights runs/expE_mot17_v1/weights/best.pt --project /workspace/close_range_20261113/runs --epochs 10 --batch 16 --imgsz 640 --device 0 --partial-p 0.5 --workers 8 --seed 0 --wandb off --fast 1 --patience 0 --template code/template_points.npy --occlude-p 0.5 --thermal-look 1 --lr0 0.0002 --warmup-epochs 1"
echo "start R1 v2 (patience 0) $(date +%T)" >> $L
venv/bin/python code/train_pose.py --data code/otp2_coco_mot17_llvippose_vast.yaml --name R1_llvippose_v2 $C > logs/R1v2.log 2>&1 < /dev/null
echo "R1v2 exit=$? $(date +%T)" >> $L
echo CHAIN_R0R1_DONE >> $L
