# LWIR person-priority model handoff

**Model:** `LWIR-YOLO26s-FT-10742-S101-E29` (completed historical Full FT checkpoint; not the current training job).

- `best_full_ft_epoch29.pt` — source detector checkpoint copy
- `best_full_ft_epoch29.onnx` — ONNX export, ONNX checker + ONNX Runtime CPU smoke test passed
- `MODEL_CARD.md` — model name, training recipe, Kalman-filter/tracker description, metric provenance
- `tracking_profile.json` — current FT + XYWH+NSA project candidate and its bounded evidence
- `verification.json` — checkpoint and ONNX hashes, export settings, graph/runtime verification

The ONNX file includes detection only. Historical controlled-evaluation scores used standard ByteTrack with XYAH Kalman state and a fixed association config: high/low/new-track thresholds `0.25/0.10/0.25`, buffer `30`, match threshold `0.80`, score fusion enabled. The evaluation protocol records detector prediction `conf=0.001`. A separately prepared project reference preset uses detector `conf=0.1` with the same ByteTrack settings, but it has not been applied to the application default. See `MODEL_CARD.md` for scope and provenance.

## DEEPX deployment artifact

An INT8 DXNN companion compiled from the existing ONNX detector is in this model folder (see [DEEPX companion](DXNN_README.md)). It uses 256 LWIR calibration frames; raw calibration data is not included. The compiler diagnostics, input contract, and the fact that target-device validation remains outstanding are documented in [DXNN_README.md](DXNN_README.md).
