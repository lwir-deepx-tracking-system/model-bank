# Model card: LWIR-YOLO26s-FT-10742-S101-E29

## Model identity

- **Model name:** `LWIR-YOLO26s-FT-10742-S101-E29`
- **Architecture:** YOLO26s object detector, 16 classes
- **Checkpoint:** the completed historical seed-101 Full FT run, saved at epoch index 29 (the 30th/final training epoch)
- **Files:** `best_full_ft_epoch29.pt` and `best_full_ft_epoch29.onnx`
- **Source checkpoint SHA256:** `385621f65165f181421e1b4e4a209e32b7cb08d5f8258e95c54866439e1e2329`

## Training method

Initialized from the original COCO-pretrained `yolo26s.pt` checkpoint and fully fine-tuned on the FLIR ADAS v2 training split. All layers were trainable (`freeze=0`); batch normalization running statistics were kept in eval/frozen mode while affine parameters followed the full-FT policy. The recorded run used 10,742 training images, 640-pixel input, batch 16, 30 fixed epochs, seed 101, AdamW, and initial learning rate `3e-4`. The released checkpoint is from epoch index 29. It is a completed historic run, not the currently running input-representation experiment.

## Tracking method and Kalman filter

The ONNX graph contains the **detector only**. It does not contain track association, a Kalman filter, or ID assignment.

- **Tracker profile used for the historical scores below:** Ultralytics ByteTrack 8.4.157 with its standard `KalmanFilterXYAH`. The 8D state is `[cx, cy, aspect_ratio, height, vx, vy, v_aspect_ratio, v_height]` under a constant-velocity model. ByteTrack performs high- and low-confidence association and retains lost tracks for the configured buffer. Historical defaults: high threshold `0.25`, low threshold `0.10`, new-track threshold `0.25`, buffer `30`, match threshold `0.80`, score fusion enabled.
- **Current project tracking candidate:** Full FT detector + ByteTrack association logic using a custom `XYWH+NSA` Kalman update. The state is `[cx, cy, width, height, vx, vy, v_width, v_height]`; detection confidence scales measurement covariance by `max(1 - confidence, 0.05)`. It does not add ReID or camera-motion compensation. This is the current project-priority candidate, not a claim that its tracker is already embedded in this ONNX.
- **Important scope:** the historical HOTA/IDF1 figures for this exact checkpoint were produced with standard XYAH ByteTrack. The newer XYWH+NSA comparison used different completed Full FT/LoRA checkpoints (10,491 images, batch 4, seeds 303/404) on the already exposed DEV5 videos. Do not attribute those XYWH+NSA scores to this seed-101 checkpoint; this exact checkpoint has not been replayed and rescored with XYWH+NSA.

The newer DEV5 comparison motivates the FT+XYWH+NSA candidate: FT had fewer person false negatives and false positives than LoRA in both seeds, while HOTA still favored LoRA in both. It is a bounded engineering preference, not a universal winner or fresh independent test. See `tracking_profile.json` for the exact scope and numbers.

## Reference tracking metrics for this checkpoint

| Evaluation set | Person HOTA | Person IDF1 | Scope |
|---|---:|---:|---|
| Development | 0.6419 | 0.7987 | Existing project comparison; standard XYAH ByteTrack |
| Previously named held-out | 0.5005 | 0.6234 | Already exposed data; not a fresh independent test |

## Detector confidence threshold

The detector confidence cutoff used in the historical tracking evaluation is not recorded in the available run artifacts. The ByteTrack values below (`track_high_thresh=0.25`, `track_low_thresh=0.10`, `new_track_thresh=0.25`) are tracker association and track-initiation thresholds; they do not establish the YOLO inference confidence cutoff. The ONNX graph does not store a configurable runtime confidence threshold.

## ONNX interface

- Static input: float32 `[1, 3, 640, 640]`
- Output: float32 `[1, 300, 6]` (`xyxy`, confidence, class index) from YOLO26's end-to-end NMS-free detection head
- Exporter option: `nms=False`; this does not add external ByteTrack to the graph
- Export validation: ONNX checker passed and ONNX Runtime CPU successfully executed a zero-input smoke test; see `verification.json`

Use the same preprocessing and class mapping as the FLIR training/evaluation pipeline. The ONNX model must be followed by the separate tracker implementation selected for the application. This package does not change the application's default model or tracker.

## Provenance

- Existing project choice: internal LWIR research archive (not included in this package).
- Current FT + XYWH+NSA project-candidate analysis: internal LWIR research archive (not included in this package).
