# LWIR model bank

Detector checkpoints, deployment files and their provenance are available directly at the repository root. File prefixes distinguish the YOLO26s and YOLOv8n metadata.

## Download models

| Model | Files | Documentation |
|---|---|---|
| YOLOv8n Full FT, FLIR thermal, 16 classes, input640, seed101, final epoch30 | [PyTorch checkpoint](yolov8n_full_epoch30_last.pt) | [Model card](YOLOv8n_MODEL_CARD.md), [verification](YOLOv8n_verification.json) |
| Historical YOLO26s Full FT, 10,742 FLIR images, seed101, epoch index29 | [PyTorch checkpoint](best_full_ft_epoch29.pt), [ONNX](best_full_ft_epoch29.onnx), [INT8 DEEPX DXNN](best_full_ft_epoch29.dxnn) | [Model card](YOLO26s_MODEL_CARD.md), [handoff](YOLO26s_README.md), [verification](YOLO26s_verification.json) |

## YOLO26s deployment and tracking

- [DEEPX instructions](YOLO26s_DXNN_README.md), [conversion verification](YOLO26s_conversion_verification.json), [compiler configuration](YOLO26s_deepx_config.json), [metadata](YOLO26s_deepx_metadata.yaml).
- [Tracking profile](YOLO26s_tracking_profile.json). The ONNX/DXNN contain the detector only, with association, Kalman filtering and ID assignment external.
- Historical metrics for this exact YOLO26s checkpoint used standard XYAH ByteTrack. The project-priority XYWH+NSA candidate was evaluated on different checkpoints; do not transfer its results to this file.
- The historical detector-evaluation confidence was0.001. A separately prepared project reference preset uses0.1 and has not changed the application's default. ByteTrack high/low/new thresholds are0.25/0.10/0.25, buffer30 and match threshold0.80.
- The INT8 DXNN was compiled with256 LWIR calibration frames; raw calibration data are not shipped. Target-device validation remains outstanding.

## Layout change

Model assets and their associated documents were moved from nested models/ folders to the root without changing the binary contents. [ARTIFACT_RELOCATION.json](ARTIFACT_RELOCATION.json) records old/new paths and before/after SHA-256 for every prior tracked file. Text hashes can differ because links and file references were updated.

Old branch-relative URLs containing models/ or deepx/ must be updated to the root-level filenames above. Commit-pinned historical URLs remain available through Git history. The previous nested binary copies are not retained in the current tree.

These artifacts are bounded research/deployment records, not proof of independent generalization or a universal method winner. No model training, loading, inference, export or NPU execution was performed during this layout change. Upstream model/data-use terms continue to apply.
