# Model card: LWIR-YOLOv8n-FT-10742-S101-E29

Completed YOLOv8n detector fully fine-tuned from the official COCO-pretrained yolov8n.pt on the FLIR ADAS v2 thermal training split.

## Checkpoint

- File: yolov8n_full_epoch30_last.pt
- SHA-256: 5cd07d31ed8861c46903d504e558460192f409bb3bb44179484c06e2759fb63e
- Size: 6,253,098 bytes
- Selection: last checkpoint from epoch index 29, the 30th/final epoch. This is not a validation-selected best checkpoint.
- Experiment: yolo_family_screen_20261019_v1; arm yolov8n_full_20261019_v1; completed 2026-10-19.

## Training

FLIR thermal, 16 classes; 10,742 training images and 1,144 source validation images. Full fine-tuning, input640, batch16, seed101, 30 fixed epochs, SGD, initial LR0.0025, final LR factor0.05, momentum0.9, weight decay0.0005, 3 warmup epochs and cosine schedule. Training-time validation was disabled. Ultralytics8.4.157, PyTorch2.9.1+cu128, RTX4090.

## Completed common detector evaluation

All six family-screen arms used the same 1,144-image exposed sourceVAL and common evaluator. YOLOv8n's batch16 evaluation ran out of memory; evaluation was recovered from the same saved checkpoint at batch4 without retraining or threshold changes.

| Metric | Value |
|---|---:|
| Overall mAP50:95 | 19.8696% |
| Person AP50:95 | 36.8542% |
| Car AP50:95 | 53.1387% |
| Overall mAP50 | 34.3712% |

The evaluator records confidence floor0.001, NMS IoU0.70 and max300 predictions. Source: RESULT_FULL6_v2_20261019.json and MANIFEST_FULL6_v2_20261019.json; hashes are in verification.json. This is single-seed detector evidence on reused sourceVAL, not an independent test or architecture-only/LWIR-causal result. Batch4 latency must not be directly ranked against batch16 evaluations. No YOLOv8n tracking performance is claimed.

## Contents and limits

This folder adds the PyTorch detector checkpoint and provenance only. No ONNX export, DEEPX conversion, model loading, inference or retraining was performed for this upload. Other model folders remain unchanged. Class count and preprocessing must match the training/evaluation pipeline.

No additional license is granted here. Upstream Ultralytics/model and FLIR data-use terms remain applicable. Raw images, training data, credentials and private filesystem paths are not included.
