# 원본 가중치와 컴파일 ONNX

기준일은 2026-11-20입니다.

운영 등록 DXNN 35개(11-20에 지운 OTP2 외형 모델 제외) 중 27개는 원본 가중치와 최종 컴파일 ONNX를 모두 보관했습니다. 팀 보관소의 고유 파일 49개를 SHA256으로 다시 읽어 확인했습니다.

## 내려받기

- [모델별 파일 목록과 SHA256](SOURCE_ARTIFACTS.json)에서 DXNN 경로와 등록 SHA256을 찾습니다.
- 기존 팀 모델뱅크 보관소의 `~/lwir_model_staging/model_bank_registered_sources_20261120/`에 접근합니다. 사용하는 팀 계정의 권한이 필요합니다.
- 팀 보관소의 `INDEX.json`에서 `object` 경로를 내려받고 `filename`으로 저장합니다. 같은 가중치를 쓴 모델은 같은 보관 파일을 가리킵니다.
- 팀 보관소 `INDEX.json` SHA256은 `a37e29fc50c68c7e3f42618d7ffb7cb832282e70f717c5cbcff210573ac12cd4`입니다.

## 모델별 현황

| 등록 DXNN | 원본 가중치 | 컴파일 ONNX |
|---|---|---|
| `detect/yolo26m/yolo26m_flir_inside_20261110_2020_640x480.dxnn` | 보관 · epoch6.pt | 보관 · yolo26m_ft7_inside_640x480.onnx |
| `detect/yolo26m/yolo26m_flir_nms_640x480.dxnn` | 보관 · newm_full30_ema.pt | 보관 · NEWMRW480.onnx |
| `detect/yolo26m/yolo26m_flir_nms_640x640.dxnn` | 보관 · newm_full30_ema.pt | 보관 · yolo26m_flir_nms_640x640.onnx |
| `detect/yolo26m/yolo26m_flir_outside_20261110_patch_640x480.dxnn` | 보관 · epoch2.pt | 보관 · yolo26m_ft8_outside_patch_640x480.onnx |
| `detect/yolo26m/yolo26m_pose_ft16v7_reviewed264_e24_provisional_kpt_640x480.dxnn` | 보관 · best.pt | 보관 · yolo26m_pose_ft16v7_reviewed264_e24_provisional_kpt_640x480.onnx |
| `detect/yolo26m/yolo26m_pose_ft16v8_aug_e01_bodybox_provisional_kpt_640x480.dxnn` | 보관 · epoch01.pt | 보관 · yolo26m_pose_ft16v8_aug_e01_bodybox_provisional_kpt_640x480.onnx |
| `detect/yolo26m/yolo26m_pose_ft16v8_aug_e10_bodybox_provisional_kpt_640x480.dxnn` | 보관 · epoch10.pt | 보관 · yolo26m_pose_ft16v8_aug_e10_bodybox_provisional_kpt_640x480.onnx |
| `detect/yolo26m/yolo26m_pose_lwir1056_20261118_bodybox_kpt_640x480.dxnn` | 보관 · last.pt | 보관 · yolo26m_pose_lwir1056_20261118_bodybox_kpt_640x480.onnx |
| `detect/yolo26m/yolo26m_pose_lwir650_20261119_bodybox_kpt_640x480.dxnn` | 보관 · last.pt | 보관 · yolo26m_pose_lwir650_20261119_bodybox_kpt_640x480.onnx |
| `detect/yolo26m/yolo26m_pose_lwir650_20261119_temporal_e13_bodybox_kpt_640x480.dxnn` | 보관 · epoch03.pt | 보관 · yolo26m_pose_lwir650_20261119_temporal_e13_bodybox_kpt_640x480.onnx |
| `detect/yolo26m/yolo26m_pose_otp2cocomot_20261114_bodybox_occl_640x480.dxnn` | 미확보 | 미확보 |
| `detect/yolo26m/yolo26m_pose_otp2cocomot_20261114_bodybox_occl_kpt_640x480.dxnn` | 미확보 | 미확보 |
| `detect/yolo26m/yolo26m_pose_otp2cocomotllvip_20261115_bodybox_farc2_640x480.dxnn` | 미확보 | 미확보 |
| `detect/yolo26m/yolo26m_pose_otp2cocomotllvip_20261115_bodybox_farc2_kpt_640x480.dxnn` | 미확보 | 미확보 |
| `detect/yolo26s/yolo26s_nms_640x480.dxnn` | 보관 · last.pt | 보관 · yolo26s_nms_640x480.onnx |
| `detect/yolo26s/yolo26s_nms_640x640.dxnn` | 보관 · last.pt | 보관 · yolo26s_nms_640x640.onnx |
| `detect/yolo26s/yolo26s_nmsfree_npu_640x640.dxnn` | 보관 · best_full_ft_epoch29.pt | 보관 · vE3.onnx |
| `detect/yolo26x/yolo26x_flir_nms_640x480.dxnn` | 미확보 | 미확보 |
| `detect/yolov8/yolov8n_nms_640x640.dxnn` | 보관 · LWIR-YOLOv8n-FT-10742-S101-E29-640x640.pt | 보관 · LWIR-YOLOv8n-FT-10742-S101-E29-640x640.onnx |
| `detect/yolov8/yolov8s_nms_640x640.dxnn` | 보관 · last.pt | 보관 · LWIR-yolov8s-FT-10742-S101-E29-640x640.onnx |
| `detect/yolox/yolox_nano_nms_640x640.dxnn` | 보관 · latest_ckpt.pth | 보관 · LWIR-yolox-nano-FT-10742-S101-E29-640x640.onnx |
| `detect/yolox/yolox_s_nms_640x640.dxnn` | 보관 · latest_ckpt.pth | 보관 · LWIR-yolox-s-FT-10742-S101-E29-640x640.onnx |
| `e2e/fasttracktr/fasttracktr_r18_npu_dancetrack_tmot_20261115_thermal3_ae.dxnn` | 미확보 | 미확보 |
| `e2e/fasttracktr/fasttracktr_r18_npu_dancetrack_tmot_20261115_thermal3_b0.dxnn` | 미확보 | 미확보 |
| `e2e/fasttracktr/fasttracktr_r18_npu_dancetrack_tmot_20261115_thermal3_b1.dxnn` | 미확보 | 미확보 |
| `optimized/E10_KPT_480x384.dxnn` | 보관 · epoch10.pt | 보관 · E10_KPT_NHWC_PERSON_480x384.onnx |
| `optimized/LWIR1056_E10_480x384.dxnn` | 보관 · last.pt | 보관 · LWIR1056_E10_NHWC_PERSON_480x384.onnx |
| `optimized/LWIR650_E10_480x384.dxnn` | 보관 · last.pt | 보관 · LWIR650_E10_NHWC_PERSON_480x384.onnx |
| `optimized/LWIR650_TEMPORAL_E13_480x384.dxnn` | 보관 · epoch03.pt | 보관 · LWIR650_TEMPORAL_E13_NHWC_PERSON_480x384.onnx |
| `reid/repvgg_distill_inside_20261110_reid.dxnn` | 보관 · ep5.pt | 보관 · REPVGG_DISTILL_INSIDE_20261110.onnx |
| `reid/repvgg_ft_inside_20261110_reid.dxnn` | 보관 · ep6.pt | 보관 · REPVGG_FT_INSIDE_20261110.onnx |
| `reid/repvgg_ft_reviewed264_20261117_reid.dxnn` | 보관 · FINAL.pt | 보관 · REPVGG_REVIEWED264_20261117.onnx |
| `reid/repvgg_tmot_camel_reid.dxnn` | 보관 · final_ema.pt | 보관 · RVGGA0_REID2.onnx |
| `reid/vit_lite_distill_inside_20261111_reid.dxnn` | 보관 · student_folded.pt | 보관 · VIT_FT7IN_S.onnx |
| `reid/vit_lite_ft_inside_20261111_reid.dxnn` | 보관 · student_folded.pt | 보관 · VIT_FT7IN_D.onnx |

## DXNN 컴파일을 사용하지 않는 등록 항목

- `detect/yolo26x/yolo26x.pt`. 등록된 CPU 가중치를 Git LFS의 실제 객체와 대조해 보관했습니다.
- `track/box_body_table_20261116.npz`. 상자 보정 조회 테이블입니다.
- `track/pos_gru_elapsed_10s.npz`. 서버가 NPZ를 직접 읽습니다. 원본 .pt/.pth를 확인 중입니다.
- `track/pos_gru_kpt_tmotdance_20261116.npz`. 서버가 NPZ를 직접 읽습니다. 원본 .pt/.pth를 확인 중입니다.
- `track/pos_gru_near_kpt_tmotdance_20261116.npz`. 서버가 NPZ를 직접 읽습니다. 원본 .pt/.pth를 확인 중입니다.
- `track/pos_mamba_kpt_tmotdance_20261116.npz`. 서버가 NPZ를 직접 읽습니다. 원본 .pt/.pth를 확인 중입니다.

## 검증 기준

- 배포 DXNN 해시와 당시 컴파일 기록을 대조했습니다. 당시 ONNX 해시가 기록돼 있으면 원본 파일과 비교했습니다. 당시 해시가 없는 경우 기록된 컴파일 입력 경로, 결과 DXNN 해시와 보존 파일을 연결했습니다.
- 학습 체크포인트 사본이 원본과 해시가 같으면 같은 파일로 취급합니다. YOLOv8n의 경우 원래 학습 디렉터리 경로는 확인되지 않았으며 내보내기에 사용한 검증 사본을 보관했습니다.
- 이 목록은 파일 재현용입니다. 모델의 운영 검출 성능은 별도 결과를 따릅니다.
