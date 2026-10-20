# DEEPX DXNN companion

The YOLO26s folder includes the INT8 DEEPX compilation of the existing `LWIR-YOLO26s-FT-10742-S101-E29` ONNX detector. It is an additional deployment artifact; it does not replace the original `.pt` or `.onnx`, and it does not contain the ByteTrack tracker or Kalman filter.

## Conversion

- Source: `best_full_ft_epoch29.onnx` (SHA-256 `c91342f76eca4c7178720405b71764d151ec91356a22e003d46fe951092170d0`)
- Compiler: DEEPX DX-COM 2.4.1, `dx_com` Python package 2.4.1, x86-64 Linux
- Quantization: INT8 post-training quantization, EMA, 100 calibration steps, optimization level 0
- Calibration: 256 evenly spaced frames selected from the 10,742-image FLIR ADAS v2 thermal training split. All selected images were checked as replicated grayscale. Image data and labels are not included in this repository.
- `deepx_config.json` records the conversion preprocessing settings. Its dataset path is a placeholder; supply a local calibration-image directory to reproduce compilation.

## Input and application contract

The compiler reports an NPU input of `uint8 NHWC [1, 640, 640, 3]`. Convert the project's grayscale LWIR frame into three replicated channels and pad/letterbox to 640×640 with value 114 before NPU submission. Do not normalize by 255 or transpose to NCHW again: DX-COM fused those operations. The compiler did not fuse resize, color conversion, or batch expansion. For grayscale replicated channels, RGB/BGR order has no effect.

Detection confidence filtering, class mapping, ByteTrack association, and the Kalman update remain outside this `.dxnn`. Historical detector evaluation used confidence `0.001`; the separate prepared project preset uses `0.1`. Historical ByteTrack thresholds were high/low/new `0.25/0.10/0.25`, buffer 30, match threshold 0.80, and score fusion enabled. These settings are not embedded in the DXNN.

## Validation scope

DX-COM completed compilation and produced `best_full_ft_epoch29.dxnn`; see `conversion_verification.json` for hashes and compiler diagnostics. The compiler reported one NPU group and one CPU group, and warned that a pre-quantization graph rewrite yielded output cosine similarity 0.8506 before its TopK/NMS path; it skipped that warning as an error. No DEEPX NPU device or DXRT inference validation was available here. Therefore this upload proves artifact generation and integrity only; it does not establish INT8 accuracy, runtime speed, or tracking performance. Validate the exact preprocessing and detector outputs on the target DEEPX device before deployment.
