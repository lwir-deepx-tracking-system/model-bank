# LWIR model bank

Detector checkpoints, deployment files and provenance are grouped in one top-level folder per model: [yolov8n/](yolov8n/) and [yolo26s/](yolo26s/). There is no additional nested model/deepx directory.

## Download models

| Model | Files | Documentation |
|---|---|---|
| YOLOv8n Full FT, FLIR thermal, 16 classes, input640, seed101, final epoch30 | [PyTorch checkpoint](yolov8n/yolov8n_full_epoch30_last.pt) | [Model card](yolov8n/MODEL_CARD.md), [verification](yolov8n/verification.json) |
| Historical YOLO26s Full FT, 10,742 FLIR images, seed101, epoch index29 | [PyTorch checkpoint](yolo26s/best_full_ft_epoch29.pt), [ONNX](yolo26s/best_full_ft_epoch29.onnx), [INT8 DEEPX DXNN](yolo26s/best_full_ft_epoch29.dxnn) | [Model card](yolo26s/MODEL_CARD.md), [handoff](yolo26s/README.md), [verification](yolo26s/verification.json) |

## YOLO26s deployment and tracking

- [DEEPX instructions](yolo26s/DXNN_README.md), [conversion verification](yolo26s/conversion_verification.json), [compiler configuration](yolo26s/deepx_config.json), [metadata](yolo26s/deepx_metadata.yaml).
- [Tracking profile](yolo26s/tracking_profile.json). The ONNX/DXNN contain the detector only, with association, Kalman filtering and ID assignment external.
- Historical metrics for this exact YOLO26s checkpoint used standard XYAH ByteTrack. The project-priority XYWH+NSA candidate was evaluated on different checkpoints; do not transfer its results to this file.
- The historical detector-evaluation confidence was0.001. A separately prepared project reference preset uses0.1 and has not changed the application's default. ByteTrack high/low/new thresholds are0.25/0.10/0.25, buffer30 and match threshold0.80.
- The INT8 DXNN was compiled with256 LWIR calibration frames; raw calibration data are not shipped. Target-device validation remains outstanding.

## Layout and compatibility

- yolov8n/: YOLOv8n PyTorch checkpoint, model card and verification.
- yolo26s/: historical YOLO26s PyTorch/ONNX/DXNN files, model cards and deployment/tracking metadata at one folder level.
- docs/: layout audit records shared by both models.

[Current relocation manifest](docs/ARTIFACT_RELOCATION_MODEL_FOLDERS_20261020.json) records every previous tracked file's old/new path and before/after SHA-256. [Earlier root-flatten audit](docs/ARTIFACT_RELOCATION_TO_ROOT_20261020.json) is retained unchanged as a historical record, not as the current download path list.

Update external branch-relative download URLs to the model-folder links above. Historical commit-pinned URLs remain available through Git history. There are no duplicate root-level or deeply nested binary copies in the current tree.

These are bounded research/deployment records, not proof of independent generalization or a universal method winner. No training, model loading, inference, export or NPU execution was performed during layout cleanup. Upstream model/data-use terms continue to apply.
