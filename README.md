# LWIR DXNN model bank

Deployment binaries are grouped by detector family. Family folders contain DXNN files only; all instructions and compilation provenance are collected here. Only existing, verified compiled artifacts are published.

## Downloads and completion status

| Family/model | Status | Artifact |
|---|---|---|
| YOLO26n | Not converted / no verified DXNN located | Not published |
| YOLO26s | Existing historical detector; DXNN compiled, NPU validation pending | [yolo26/yolo26s.dxnn](yolo26/yolo26s.dxnn) |
| YOLOX-Nano | No verified DXNN located; no new conversion started | Not published |
| YOLOX-s | No verified DXNN located; no new conversion started | Not published |
| YOLOv8n | Converted and compiled in this update; NPU validation pending | [yolov8/yolov8n.dxnn](yolov8/yolov8n.dxnn) |
| YOLOv8s | Not converted / no verified DXNN located | Not published |

No yolox/ folder is committed until an actual verified YOLOX DXNN exists. Git does not retain empty folders; no placeholder model files are used.

## Artifact identity and integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| yolo26s.dxnn | 10897723 | 1e82b415d86e9f26a549f5ee92af183f733879c77dafc40f7ff97adbf826b9f3 |
| yolov8n.dxnn | 3696192 | f11c80721fb88d7c4cb153b165d381e778e5bf8de1bc5302d32496baf70831d7 |

- YOLO26s: historical LWIR-YOLO26s-FT-10742-S101-E29 checkpoint, source PT SHA-256385621f65165f181421e1b4e4a209e32b7cb08d5f8258e95c54866439e1e2329. This is the earlier historical model, not the different YOLO26s checkpoint from the2026-10-19 six-model screen. The existing DXNN binary is unchanged.
- YOLOv8n: LWIR-YOLOv8n-FT-10742-S101-E29, source PT SHA-2565cd07d31ed8861c46903d504e558460192f409bb3bb44179484c06e2759fb63e; FLIR ADAS v2 thermal16 classes, 10,742 TRAIN images, full fine-tuning, seed101, input640, batch16, SGD initial LR0.0025 and30 fixed epochs. The source is final epoch30/zero-based29, not validation-selected best.
- YOLOv8n source ONNX: e6d02de05cc5b461ee0159fda746bc64226ff881283bbe05828c078d12bda36f (12250742 bytes). PT and ONNX intermediates remain local and are not included in the current repository tree.

## DX-M1 compilation

Both files use the existing DEEPX DX-COM2.4.1 environment (Python package reports2.4.1-rc.1), INT8 post-training quantization, EMA100 calibration steps and optimization level0. The intended target is DX-M1 using the existing SDK default hardware configuration. No new account, licence acceptance, paid resource, driver installation or security-setting change was performed.

YOLOv8n compilation ran on x86-64 Ubuntu WSL with torch 2.14.1+cpu, ONNX 1.23.2 and ONNX Runtime 1.31.0. Quantization was explicitly CPU-only; the active research-training GPU was not used.

Calibration used256 fixed, evenly spaced images from the10,742-image FLIR TRAIN list only: round(i*(N-1)/255), i=0..255. The source TRAIN-list SHA-256 is d40cb99963d55de372d0b46cfcc11652078b51d5d1c17321c3040621ed63f1d1. Images were verified replicated grayscale and letterboxed to640x640 using centered114 padding before calibration. The calibration-manifest SHA-256 is 78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34. Validation/TEST images and raw calibration data are not shipped.

## Runtime input and output

- Input contract: uint8 NHWC [1,640,640,3]. Replicate a grayscale LWIR frame to3 channels and letterbox to640x640 with centered114 padding before NPU submission. The compiler config fuses division by255 and HWC-to-CHW transpose. Do not apply those operations again. Resize/padding, color conversion and batch expansion remain external.
- YOLOv8n source ONNX input is float32 NCHW [1,3,640,640]. Its source output is [1,20,8400]: decoded center-x/center-y/width/height plus16 class scores, with no separate objectness channel. External class-aware NMS and inverse-letterbox coordinate restoration are required. Consult DXRT's actual tensor metadata on the board before wiring buffers; the compiled device layout has not been run here.
- The existing YOLO26s source ONNX output was [1,300,6] (xyxy, confidence, class index), an end-to-end detection head. Its output parser must not be reused for YOLOv8n without adapting the raw one-to-many/NMS path.
- Tracking, Kalman filtering, ID assignment and detector thresholds are external to both artifacts. No tracker implementation or default-application configuration is changed by this upload.

Class mapping for both16-class FLIR detectors:

| ID | Class |
|---:|---|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 4 | bus |
| 5 | train |
| 6 | truck |
| 7 | traffic light |
| 8 | fire hydrant |
| 9 | stop sign |
| 10 | dog |
| 11 | deer |
| 12 | skateboard |
| 13 | stroller |
| 14 | scooter |
| 15 | other vehicle |

## What was verified

YOLOv8n: source checkpoint identity/hash, unchanged source copy, ONNX checker, CPU zero-input execution with finite [1,20,8400] output, and raw PyTorch/ONNX output agreement on3 fixed TRAIN calibration samples (rtol1e-4, atol5e-3). DX-COM returned successfully and produced the hashed DXNN. Compilation elapsed 154.2 seconds.

The YOLOv8n compiler reported one NPU group and one CPU group, marked8 memory operations from NPU to CPU, and generated an empty error.log. It explicitly reported uint8 NHWC [1,640,640,3] input and fused Div(x=255.0) plus HWC-to-CHW transpose. This is a mixed NPU/CPU graph, not a claim that every operation executes on the NPU. The existing YOLO26s compiler previously warned about a0.8506 pre-quantization graph-rewrite cosine similarity around TopK/NMS and reported one NPU group plus one CPU group.

No physical DX-M1/DXRT device was detected. Neither new YOLOv8n nor the retained YOLO26s DXNN has been executed or accuracy/FPS/tracking-tested on NPU in this work. Compilation success is not a claim of board correctness, INT8 accuracy, speed or independent generalization. No training was started.

## Repository scope and history

Only README.md and the actual family DXNN artifacts are retained in the current tree. PT/ONNX binaries, model-specific JSON/YAML/Markdown and older layout records were removed from the current tree. Earlier Git commits still contain those historical files; no history rewrite or force push was performed. Update branch-relative download URLs to the links above. Local research originals and conversion intermediates are preserved.

Upstream Ultralytics/model, DEEPX SDK and FLIR data-use terms remain applicable; this README grants no additional licence. Official references: [DEEPX model compilation](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-COM/DX-COMPILER_Agent_Driven_Development/) and [Ultralytics DEEPX export](https://docs.ultralytics.com/integrations/deepx/).
