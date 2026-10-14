# LWIR model bank

This repository stores detector checkpoints and their inference and tracking provenance.

## Available model

**[LWIR-YOLO26s-FT-10742-S101-E29](models/LWIR-YOLO26s-FT-10742-S101-E29/)** is a completed historical YOLO26s Full Fine-Tuning checkpoint (10,742 FLIR ADAS v2 training images, seed 101, 30 epochs; released checkpoint at epoch index 29). The folder contains the `.pt` checkpoint, ONNX export, model card, tracking profile, and verification manifest.

The ONNX contains the detector only. For the historical scores on this exact checkpoint, the tracker was Ultralytics ByteTrack with its standard XYAH Kalman filter. **The current project-priority tracking candidate is ByteTrack association with a custom XYWH + confidence-NSA Kalman update**, described in the model card. That tracker is separate from the ONNX, and the candidate's results came from different checkpoints; do not attribute them to this seed-101 model. The historical controlled-evaluation protocol records YOLO prediction `conf=0.001`. A separately prepared project reference preset uses `conf=0.1`, but has not been applied to the application default. The historical comparison held ByteTrack association settings common across model variants: high/low/new-track thresholds `0.25/0.10/0.25`, buffer `30`, and match threshold `0.80`. The same settings appear in the prepared preset, but are not a global application default.
