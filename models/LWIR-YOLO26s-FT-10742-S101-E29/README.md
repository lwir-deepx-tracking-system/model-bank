# LWIR person-priority model handoff

**Model:** `LWIR-YOLO26s-FT-10742-S101-E29` (completed historical Full FT checkpoint; not the current training job).

- `best_full_ft_epoch29.pt` — source detector checkpoint copy
- `best_full_ft_epoch29.onnx` — ONNX export, ONNX checker + ONNX Runtime CPU smoke test passed
- `MODEL_CARD.md` — model name, training recipe, Kalman-filter/tracker description, metric provenance
- `tracking_profile.json` — current FT + XYWH+NSA project candidate and its bounded evidence
- `verification.json` — checkpoint and ONNX hashes, export settings, graph/runtime verification
- `export_and_verify.py` — reproducible export script

The ONNX file includes detection only. The historical scores for this exact checkpoint used standard ByteTrack with XYAH Kalman state. FT + XYWH+NSA is the current project-priority tracker candidate, based on separate seed303/404 checkpoints; its scores are not attributed to the packaged seed101 checkpoint. See `MODEL_CARD.md` before publishing.
