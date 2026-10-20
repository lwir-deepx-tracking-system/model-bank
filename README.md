# LWIR DXNN model bank

Deployment binaries are grouped by detector family. Family folders contain DXNN files only; all instructions and compilation provenance are collected here. Only existing, verified compiled artifacts are published.

## Downloads and completion status

| Family/model | Status | Artifact |
|---|---|---|
| YOLO26n | Compiled; NPU validation pending | [yolo26/yolo26n.dxnn](yolo26/yolo26n.dxnn) |
| YOLO26s | Compiled; NPU validation pending | [yolo26/yolo26s.dxnn](yolo26/yolo26s.dxnn) |
| YOLOX-Nano | Compiled; NPU validation pending | [yolox/yolox_nano.dxnn](yolox/yolox_nano.dxnn) |
| YOLOX-s | No verified DXNN located; no new conversion started | Not published |
| YOLOv8n | Converted and compiled in this update; NPU validation pending | [yolov8/yolov8n.dxnn](yolov8/yolov8n.dxnn) |
| YOLOv8s | Compiled; NPU validation pending | [yolov8/yolov8s.dxnn](yolov8/yolov8s.dxnn) |

No yolox/ folder is committed until an actual verified YOLOX DXNN exists. Git does not retain empty folders; no placeholder model files are used.

## Artifact identity and integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| yolo26s_ft_s101_e30_legacy_20261001.dxnn | 10897723 | 1e82b415d86e9f26a549f5ee92af183f733879c77dafc40f7ff97adbf826b9f3 |
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


## Published yolo26m_teacher_s202_e30

Artifact: [yolo26/yolo26m.dxnn](yolo26/yolo26m.dxnn) — 21991058 bytes; SHA-256 `fbda1f1ba2182090e1f2f4dc3a6306347cbeec9701633c7e21c47c4a34c2c03e`.

FLIR TRAIN4196, seed202, fixed30 epochs, AdamW, training480x640; separate teacher recipe. Source checkpoint SHA-256 `dca0f23d1a945b6a73b17ef050452b87e3e7d99265eb413519b5395db67c7736`; source ONNX SHA-256 `a36ce1ecfa49f798cc6ff977eea65501206681e068836317fe38caf67689713b`. Final checkpoint, not validation-selected best.

CPU-only DX-COM 2.4.1-rc.1, opt_level0, INT8 EMA100 steps; compilation 439.7s. ONNX checker, finite CPU execution, three TRAIN image raw PyTorch/ONNX parity and artifact hash passed. The DXNN has not been run on an NPU; board accuracy/FPS are unmeasured.

Calibration:256 fixed evenly spaced images from the full FLIR TRAIN10742 list only (no VAL/TEST); manifest SHA-256 `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34`. This deployment calibration pool is broader than this teacher's4196-image training subset. Input/output: uint8 NHWC [1,640,640,3]; confirm compiler artifact metadata; replicate grayscale to3 channels; centered LetterBox640x640 padding114 before submission. Source output [1, 20, 8400]: raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS. Follow per-model preprocessing; the old historical YOLO26s uses a different end-to-end head.
Division by255 and HWC-to-CHW transpose are fused; resize/padding and channel preparation remain external. Do not normalize a second time.


## Published yolo26n

Artifact: [yolo26/yolo26n.dxnn](yolo26/yolo26n.dxnn) — 3467688 bytes; SHA-256 `0b865a0d05fa2b53b2a52d54914f37c5abcaa54f41a1059c69ca747e03460bee`.

Common six-model screen: TRAIN10742, seed101, fixed30 epochs, SGD, batch16,640. Source checkpoint SHA-256 `dc6f25aea34bb67989fdff31ad2ce3ea993d4ae690ab22861f4b2240dc872fc9`; source ONNX SHA-256 `2ddda14087ac21a5365b7d94a40e60f95e6b83c1a74e271646eac540262e0400`. Final checkpoint, not validation-selected best.

CPU-only DX-COM 2.4.1-rc.1, opt_level0, INT8 EMA100 steps; compilation 157.8s. ONNX checker, finite CPU execution, three TRAIN image raw PyTorch/ONNX parity and artifact hash passed. The DXNN has not been run on an NPU; board accuracy/FPS are unmeasured.

Calibration:256 fixed evenly spaced images from the full FLIR TRAIN10742 list only (no VAL/TEST); manifest SHA-256 `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34`. Input/output: uint8 NHWC [1,640,640,3]; confirm compiler artifact metadata; replicate grayscale to3 channels; centered LetterBox640x640 padding114 before submission. Source output [1, 20, 8400]: raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS. Follow per-model preprocessing; the old historical YOLO26s uses a different end-to-end head.
Division by255 and HWC-to-CHW transpose are fused; resize/padding and channel preparation remain external. Do not normalize a second time.


## Published yolov8s

Artifact: [yolov8/yolov8s.dxnn](yolov8/yolov8s.dxnn) — 11857151 bytes; SHA-256 `c9783a3ce9765b35163bc876fab6a4b6332314dcab71c3fca282161a3ceaa79f`.

Common six-model screen: TRAIN10742, seed101, fixed30 epochs, SGD, batch16,640. Source checkpoint SHA-256 `53b640f03323a43b4751128108a762b8b4795b2f82834b391a2a19df4dbfb0eb`; source ONNX SHA-256 `146d4f64773af2d3aba40acd7b7433d9729d8819e556da8d6e8e963d4e3dddcf`. Final checkpoint, not validation-selected best.

CPU-only DX-COM 2.4.1-rc.1, opt_level0, INT8 EMA100 steps; compilation 250.1s. ONNX checker, finite CPU execution, three TRAIN image raw PyTorch/ONNX parity and artifact hash passed. The DXNN has not been run on an NPU; board accuracy/FPS are unmeasured.

Calibration:256 fixed evenly spaced images from the full FLIR TRAIN10742 list only (no VAL/TEST); manifest SHA-256 `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34`. Input/output: uint8 NHWC [1,640,640,3]; confirm compiler artifact metadata; replicate grayscale to3 channels; centered LetterBox640x640 padding114 before submission. Source output [1, 20, 8400]: raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS. Follow per-model preprocessing; the old historical YOLO26s uses a different end-to-end head.
Division by255 and HWC-to-CHW transpose are fused; resize/padding and channel preparation remain external. Do not normalize a second time.


## Published yolo26s

Artifact: [yolo26/yolo26s.dxnn](yolo26/yolo26s.dxnn) — 10895020 bytes; SHA-256 `7ac754185cc99e807bd824dfa617e8cc696f02a29c402076820967c906e7098b`.

Common six-model screen: TRAIN10742, seed101, fixed30 epochs, SGD, batch16,640. Source checkpoint SHA-256 `13346581137ebd1bea8e1e820c0b969b439d4859b3638685ad394fe6c116d4cc`; source ONNX SHA-256 `d021ac4097e31b3943d8029eb30be8cee500fe41ee11c308e36a8582181f3b90`. Final checkpoint, not validation-selected best.

CPU-only DX-COM 2.4.1-rc.1, opt_level0, INT8 EMA100 steps; compilation 247.4s. ONNX checker, finite CPU execution, three TRAIN image raw PyTorch/ONNX parity and artifact hash passed. The DXNN has not been run on an NPU; board accuracy/FPS are unmeasured.

Calibration:256 fixed evenly spaced images from the full FLIR TRAIN10742 list only (no VAL/TEST); manifest SHA-256 `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34`. Input/output: uint8 NHWC [1,640,640,3]; confirm compiler artifact metadata; replicate grayscale to3 channels; centered LetterBox640x640 padding114 before submission. Source output [1, 20, 8400]: raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS. Follow per-model preprocessing; the old historical YOLO26s uses a different end-to-end head.
Division by255 and HWC-to-CHW transpose are fused; resize/padding and channel preparation remain external. Do not normalize a second time.


## Published yolox_nano

Artifact: [yolox/yolox_nano.dxnn](yolox/yolox_nano.dxnn) — 1536206 bytes; SHA-256 `60333bccb6ec0b41dad61f10998fe5084b2aeb143596eebb16cc25f705ecd169`.

Common six-model screen: TRAIN10742, seed101, fixed30 epochs, SGD, batch16,640. Source checkpoint SHA-256 `64119cfcc291fcb436839cf29c11f3e089d7a3eee0580f5edce2049ebc35877f`; source ONNX SHA-256 `3c181badb88bff9a8d9c7f2788a3648eced42f24407a97ff2e27bd0ea045d271`. Final checkpoint, not validation-selected best.

CPU-only DX-COM 2.4.1-rc.1, opt_level0, INT8 EMA100 steps; compilation 144.1s. ONNX checker, finite CPU execution, three TRAIN image raw PyTorch/ONNX parity and artifact hash passed. The DXNN has not been run on an NPU; board accuracy/FPS are unmeasured.

Calibration:256 fixed evenly spaced images from the full FLIR TRAIN10742 list only (no VAL/TEST); manifest SHA-256 `a32a796e1cb2335e8f30d75cd3a52e4211ff6aa9dc80da9e5b106072906ef192`. Input/output: uint8 NHWC [1,640,640,3]; confirm compiler artifact metadata; replicate grayscale to3 channels; YOLOX top-left pad114640x640 BGR0..255, no normalization before submission. Source output [1, 8400, 21]: raw decoded xywh + objectness +16 class scores;8400 candidates, external objectness-times-class scoring and class-aware NMS. Follow per-model preprocessing; the old historical YOLO26s uses a different end-to-end head.
YOLOX uses BGR0..255 with top-left114 padding, no division by255; output includes separate objectness and16 class scores. External objectness-times-class scoring and NMS are required.


<!-- VERIFIED_IO_BEGIN -->
## Verified class count and tensor interfaces

Class count, output channel count, candidate count and number of output tensors are different quantities. The checked deployment exports use16 object classes. Each has one public output tensor; internal NPU-to-CPU graph boundaries may contain several tensors. The DXNN values below are read from the binary header graph_info and embedded CPU ONNX output ValueInfo, without executing an NPU.

| Model/artifact | nc | ONNX input | ONNX output | DXNN public outputs | Meaning / postprocessing |
|---|---:|---|---|---|---|
| [yolov8/yolov8n.dxnn](yolov8/yolov8n.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 20, 8400] | 1: output0 FLOAT [1, 20, 8400] | xywh4 + class16; boxes decoded; confidence filtering, xywh-to-xyxy and NMS external; internal NPU boundary tensors[4] |
| [yolo26/yolo26m.dxnn](yolo26/yolo26m.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 20, 8400] | 1: output0 FLOAT [1, 20, 8400] | xywh4 + class16; boxes decoded; confidence filtering, xywh-to-xyxy and NMS external; internal NPU boundary tensors[6] |
| [yolo26/yolo26n.dxnn](yolo26/yolo26n.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 20, 8400] | 1: output0 FLOAT [1, 20, 8400] | xywh4 + class16; boxes decoded; confidence filtering, xywh-to-xyxy and NMS external; internal NPU boundary tensors[6] |
| [yolov8/yolov8s.dxnn](yolov8/yolov8s.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 20, 8400] | 1: output0 FLOAT [1, 20, 8400] | xywh4 + class16; boxes decoded; confidence filtering, xywh-to-xyxy and NMS external; internal NPU boundary tensors[4] |
| [yolo26/yolo26s.dxnn](yolo26/yolo26s.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 20, 8400] | 1: output0 FLOAT [1, 20, 8400] | xywh4 + class16; boxes decoded; confidence filtering, xywh-to-xyxy and NMS external; internal NPU boundary tensors[6] |
| [yolox/yolox_nano.dxnn](yolox/yolox_nano.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 8400, 21] | 1: output0 FLOAT [1, 8400, 21] | xywh4 + objectness1 + class16; grid/stride/exp decoded; objectness*class, NMS external; internal NPU boundary tensors[9] |
| [yolo26/yolo26s_ft_s101_e30_legacy_20261001.dxnn](yolo26/yolo26s_ft_s101_e30_legacy_20261001.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 300, 6] | 1: output0 FLOAT [1, 300, 6] | xyxy4 + confidence + class index,300 selected detection slots; end-to-end decoder included; internal NPU boundary tensors[6] |

- New YOLO26n/s/m exports explicitly disable end-to-end selection to expose the one-to-many head. Their20 output fields are4 decoded box coordinates plus16 probabilities; no objectness and no DFL-bin channels in the public output. These actual YOLO26 ONNX graphs contain no DFL-named nodes.
- YOLOv8n/s include DFL decoding inside ONNX. The16-bin box distributions are reduced before the public output, so the public shape is [1,20,8400], not [1,80,8400].
- YOLOX exports explicitly enable decode_in_inference: grid offsets, stride scaling and exp width/height decoding are already in ONNX. [1,8400,21] contains4 decoded box coordinates,1 objectness and16 class probabilities. Do not decode grid/stride a second time. Multiply objectness by class probability before filtering/NMS.
- Historical YOLO26s [1,300,6] is a different end-to-end export:300 detection slots, six fields per slot. Six is not the class count. Apply confidence filtering and inverse letterbox; do not attach the new20-channel raw-head decoder.
- ONNX input is float32 NCHW [1,3,640,640], named images. Compiled NPU input is uint8 NHWC [1,640,640,3] per compiler guidance. UL input normalization by255 and layout transpose are fused. YOLOX has no division by255; its layout transpose is fused. External geometry and semantic channel transforms remain required per model.
- The public output0 is float32 according to embedded CPU ONNX metadata; verify actual DXRT buffer metadata on hardware. No claim of board execution, accuracy or FPS follows from these static checks.

Verified output class index mapping: 0=person, 1=bicycle, 2=car, 3=motorcycle, 4=bus, 5=train, 6=truck, 7=traffic light, 8=fire hydrant, 9=stop sign, 10=dog, 11=deer, 12=skateboard, 13=stroller, 14=scooter, 15=other vehicle. YOLOX COCO category IDs are one-based (index+1); detector output class indices are zero-based.

## Existing PC timing observations

These are historical pipeline timings, not a controlled detector-only benchmark and not DXNN/NPU latency. GTX1080Ti, torch2.7.0+cu118, Ultralytics8.4.157, FP32, batch1, square640,4 CPU threads,1924 frames after5 warmup predictions; conf0.1, IoU0.7, max_det300. Simultaneous load was not recorded; median/p95 are unavailable.

| Common-screen model | Mean predict + CPU transfer + ByteTrack (ms/frame) | Mean including read/decode (ms/frame) |
|---|---:|---:|
| YOLOv8s | 12.7453 | 39.4114 |
| YOLO26s | 14.8630 | 17.3547 |
| YOLOv8n / YOLO26n / YOLOX-Nano / YOLOX-s | Not measured under comparable local conditions | Not measured |

Derived from recorded total seconds divided by1924; excludes JSON writing. Read/decode differences can reflect I/O and cache order, so these values do not establish a fair pure-inference ranking. Sources: `yolo_family_tracking_report_only_20261019_v1/yolov8s_BYTE/manifest.json` and `.../yolo26s_BYTE/manifest.json` under the original research root. Recorded completion times:2026-10-19 23:02:14 and23:03:31 KST. No new GPU benchmark was run while RF-DETR training and CPU compilation were active. Remote RTX4090 batch4/16 evaluation timings are not substituted as local PC latency.
<!-- VERIFIED_IO_END -->
