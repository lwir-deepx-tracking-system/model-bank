# LWIR DXNN model bank


<!-- PERFORMANCE_TABLES_BEGIN -->
## 모델별 성능 · 실제 Markdown 표

아래 값은 기존 FP32 원본 모델의 검출·추적 평가입니다. INT8 DXNN의 정확도·속도를 측정한 값이 아닙니다. AP는 별도 AP50 열을 제외하면 AP50:95이며, 전체 mAP와 person AP는 다른 지표입니다. 지표 단위는 %, FP/FN/IDSW는 건수입니다. 표시값은 소수 넷째 자리로 반올림했습니다. 서로 다른 평가군을 합쳐 순위를 만들지 않습니다.

### 최근 검출 6종 · 동일 sourceVAL

TRAIN10742장, sourceVAL1144장,16클래스, seed101, Full FT30 epoch의 최종 체크포인트입니다. 입력640×640, AP confidence0.001, NMS IoU0.70, COCO maxDets[1,10,300]. YOLOv8n은 OOM 후 평가 batch4, 나머지는16이므로 아래 평가 batch를 지연시간 순위로 해석하지 않습니다.

| 모델 | 전체 mAP50:95 | 전체 AP50 | 전체 AP75 | person AP50:95 | car AP50:95 | AR300 | 평가 batch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8n | 19.8696 | 34.3712 | 19.2929 | 36.8542 | 53.1387 | 32.6315 | 4 |
| yolox-nano | 16.8664 | 31.2794 | 14.8047 | 31.6139 | 48.2667 | 27.6596 | 16 |
| yolo26n | 20.7318 | 35.6591 | 20.7991 | 38.0337 | 53.9829 | 34.3948 | 16 |
| yolov8s | 24.3720 | 40.6337 | 25.3877 | 42.2804 | 57.4554 | 34.7286 | 16 |
| yolox-s | 22.8154 | 38.6608 | 22.5710 | 39.9558 | 55.8228 | 30.8326 | 16 |
| yolo26s | 25.4758 | 41.9578 | 25.6621 | 44.9255 | 59.0974 | 36.6120 | 16 |

원문: `yolo_family_screen_20261019_v1/common_score_comparison_full6_v2_20261019.csv`, evaluator v3. sourceVAL은 이미 노출된 평가셋이며 독립 TEST 결과가 아닙니다.

### 최근 추적 · 2검출기 × 3 tracker

노출된 DEV5영상·1924프레임, 입력640×640, batch1, FP32, detector confidence0.10, NMS IoU0.70, max_det300. 검출 sourceVAL과 다른 데이터입니다. 최근 추적이 평가된 모델은 YOLOv8s와 YOLO26s뿐이며 나머지4종은 미평가입니다.

**person · GT boxes 2469**

| 모델 | tracker | HOTA | DetA | AssA | IDF1 | MOTA | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8s | BYTE | 62.1219 | 57.9567 | 66.6548 | 76.8591 | 69.9068 | 177 | 543 | 23 |
| yolov8s | OCSORT | 59.6343 | 57.8827 | 61.5119 | 74.0936 | 70.8789 | 155 | 542 | 22 |
| yolov8s | BOTSORT_GMC | 67.4590 | 63.6129 | 71.7691 | 77.9966 | 72.2155 | 201 | 467 | 18 |
| yolo26s | BYTE | 63.7377 | 59.8190 | 67.9860 | 78.9417 | 70.5549 | 191 | 518 | 18 |
| yolo26s | OCSORT | 62.9723 | 59.9617 | 66.2101 | 78.4777 | 71.3244 | 161 | 527 | 20 |
| yolo26s | BOTSORT_GMC | 67.9369 | 64.5543 | 71.7458 | 78.2812 | 71.5269 | 224 | 461 | 18 |


**car · GT boxes 6668**

| 모델 | tracker | HOTA | DetA | AssA | IDF1 | MOTA | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8s | BYTE | 43.5842 | 37.7524 | 50.8984 | 53.8226 | 39.6521 | 693 | 3238 | 93 |
| yolov8s | OCSORT | 41.6231 | 37.1677 | 47.1162 | 50.9204 | 39.0222 | 634 | 3322 | 110 |
| yolov8s | BOTSORT_GMC | 50.1605 | 40.3313 | 63.0404 | 60.8233 | 41.9316 | 693 | 3122 | 57 |
| yolo26s | BYTE | 47.9415 | 44.9303 | 51.9791 | 57.0719 | 48.3653 | 834 | 2497 | 112 |
| yolo26s | OCSORT | 47.6287 | 44.6058 | 51.6394 | 55.1503 | 48.2154 | 784 | 2577 | 92 |
| yolo26s | BOTSORT_GMC | 53.7894 | 47.9854 | 61.2350 | 62.3271 | 51.7846 | 808 | 2361 | 46 |


**truck · GT boxes 1180**

| 모델 | tracker | HOTA | DetA | AssA | IDF1 | MOTA | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8s | BYTE | 7.2595 | 9.7895 | 5.5417 | 11.4024 | -4.4915 | 189 | 1023 | 21 |
| yolov8s | OCSORT | 7.5302 | 9.1623 | 6.2953 | 12.4665 | -3.2203 | 165 | 1033 | 20 |
| yolov8s | BOTSORT_GMC | 10.3740 | 10.6709 | 10.3950 | 17.4198 | -2.7966 | 186 | 1019 | 8 |
| yolo26s | BYTE | 25.3696 | 24.1133 | 28.1567 | 32.0671 | -5.5932 | 445 | 778 | 23 |
| yolo26s | OCSORT | 19.2914 | 22.6585 | 17.0108 | 22.9188 | -4.0678 | 397 | 811 | 20 |
| yolo26s | BOTSORT_GMC | 22.9611 | 25.7977 | 21.3028 | 24.4676 | -1.2712 | 418 | 759 | 18 |

원문: `yolo_family_tracking_report_only_20261019_v1/COMPARISON.json`. TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`. HOTA/DetA/AssA/IDF1/MOTA는 높을수록, FP/FN/IDSW는 낮을수록 좋습니다.

### 검출·tracker 설정

| 항목 | 최근 검출6종 | 최근 추적2×3 | 역사 추적19조건 |
| --- | --- | --- | --- |
| 입력 H×W | 640×640 | 640×640 | 480×640 |
| detector confidence | 0.001 (AP) | 0.10 | 0.10 |
| NMS IoU | 0.70 | 0.70 | 0.70 |
| 최대 검출 수 | COCO maxDets[1,10,300] | 300 | 300 |
| batch | v8n4 / 나머지16 | 1 | 1 |
| 정밀도 | 집계 CSV만으로 미확인 | FP32 | FP32 |
| ByteTrack 구현 | 해당 없음 | Ultralytics BYTE | ByteTrack association + BOTrack XYWH/NSA wrapper |
| BYTE reset 횟수 | 해당 없음 | v8s BYTE7 /26s BYTE7; 다른 tracker 미확인 | 미확인 |
| reset 규약 | 해당 없음 | clip 시작 + source-frame gap | 미확인 |
| tracker frame_rate / class gating | 해당 없음 | 미확인 | 미확인 |


| ByteTrack 공통 파라미터 | 값 |
| --- | --- |
| track_high_thresh | 0.25 |
| track_low_thresh | 0.10 |
| new_track_thresh | 0.25 |
| match_thresh | 0.80 |
| track_buffer | 30 |
| fuse_score | true |


| tracker/평가군 | 추가 설정 |
| --- | --- |
| 최근 OC-SORT | delta_t3, inertia0.2, use_byte=false |
| 최근 BoT-SORT | GMC=sparseOptFlow, ReID=false, proximity_thresh0.5, appearance_thresh0.8 |
| 최근 BYTE | XYWH/NSA override 여부는 해당 runner 소스 미확보로 미확인 |
| 역사 BYTE wrapper | cx,cy,w,h 및 속도; Ultralytics XYWH Q/R; ReID=false, GMC=false |
| 역사 NSA | R_effective=max(1-matched_detection_confidence,0.05)*R_base |
| 최근 BYTE 두 모델 런타임 | Ultralytics8.4.157 / PyTorch2.7.0+cu118 |

AP용0.001과 추적 입력용0.10을 혼동하지 마세요. Detector confidence를0.25로 올리면 ByteTrack의 낮은 점수 association 후보가 제거될 수 있습니다. 동일 threshold 숫자여도 최근 BYTE와 역사 XYWH/NSA wrapper는 같은 구현·입력이 아닙니다. 모델 파일에 tracker 코드·설정은 포함되지 않습니다.

### PC latency · 기존 관측값

DESKTOP-AAO2NEF, i7-7700K, GTX1080Ti, torch2.7.0+cu118, Ultralytics8.4.157, CPU threads4, FP32, batch1,640×640,1924프레임, warmup5회. 총 기록 초÷프레임 수로 계산한 평균입니다.

| 모델 | 탐지+CPU 전송+ByteTrack 평균 ms/frame | 읽기·decode 포함 평균 ms/frame | 통제된 순수 추론 benchmark |
| --- | --- | --- | --- |
| yolov8n | 미측정 | 미측정 | 미측정 |
| yolox-nano | 미측정 | 미측정 | 미측정 |
| yolo26n | 미측정 | 미측정 | 미측정 |
| yolov8s | 12.7453 | 39.4114 | 미측정 |
| yolox-s | 미측정 | 미측정 | 미측정 |
| yolo26s | 14.8630 | 17.3547 | 미측정 |

두 관측값 모두 JSON 쓰기는 제외하며, median/p95·측정 당시 동시 부하·I/O 및 캐시 상태는 기록되지 않았습니다. 읽기 포함 결과의 차이를 모델 추론 속도 차이로 단정하지 않습니다. 기록 완료 시각은 v8s2026-10-19 23:02:14,26s23:03:31 KST입니다. 원문: `yolo_family_tracking_report_only_20261019_v1/{yolov8s_BYTE,yolo26s_BYTE}/manifest.json`. RF-DETR 학습과 CPU 컴파일이 실행 중이므로 새 GPU benchmark를 수행하지 않았습니다.

<details>
<summary>별도 참고: 최근 evaluator의 혼합 batch 시간 · PC latency와 비교하지 않음</summary>


| 모델 | 평가 batch | 기록 ms/image | 동일 조건 PC benchmark |
| --- | --- | --- | --- |
| yolov8n | 4 | 12.2254 | 미측정 |
| yolox-nano | 16 | 20.3782 | 미측정 |
| yolo26n | 16 | 9.0352 | 미측정 |
| yolov8s | 16 | 7.9904 | 미측정 |
| yolox-s | 16 | 17.7001 | 미측정 |
| yolo26s | 16 | 8.3163 | 미측정 |

이 값은 별도의 기존 evaluator 처리시간이며 batch1 순수 추론 지연이 아닙니다. YOLOv8n batch4와 나머지 batch16을 합쳐 속도 순위를 만들지 않습니다. 집계 CSV만으로 GPU명·정밀도·전체 전후처리 범위·warmup을 확정하지 않습니다.

</details>


<details>
<summary>역사 실험: native_input_seed202 · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 | person AP50 | car AP50:95 | 전체 mAP50:95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / clahe_full_ft | CLAHE · Full FT | 202 | 30 | 43.6077 | 75.4628 | 58.4576 | 미확인 |
| YOLO26s / clahe_lora_r8_head | CLAHE · LoRA r8+head | 202 | 30 | 40.9653 | 71.7479 | 55.0928 | 미확인 |
| YOLO26s / combined_full_ft | combined · Full FT | 202 | 30 | 43.9659 | 75.4729 | 58.5177 | 미확인 |
| YOLO26s / combined_lora_r8_head | combined · LoRA r8+head | 202 | 30 | 40.2628 | 70.7924 | 55.0202 | 미확인 |
| YOLO26s / gradient_full_ft | gradient · Full FT | 202 | 30 | 43.9518 | 75.4946 | 58.6052 | 미확인 |
| YOLO26s / gradient_lora_r8_head | gradient · LoRA r8+head | 202 | 30 | 41.3742 | 71.8697 | 55.0837 | 미확인 |
| YOLO26s / original_full_ft | original · Full FT | 202 | 30 | 44.1316 | 75.4051 | 58.8730 | 미확인 |
| YOLO26s / original_lora_r8_head | original · LoRA r8+head | 202 | 30 | 42.1545 | 73.3405 | 55.7646 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA | IDF1 | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| clahe_full_ft | person | 66.4821 | 75.5098 | 336 | 517 | 16 |
| clahe_full_ft | car | 48.6943 | 52.4884 | 1262 | 2763 | 65 |
| clahe_lora_r8_head | person | 66.4160 | 75.6505 | 339 | 550 | 20 |
| clahe_lora_r8_head | car | 47.9142 | 50.2463 | 1332 | 3097 | 61 |
| combined_full_ft | person | 63.3713 | 71.4743 | 251 | 502 | 30 |
| combined_full_ft | car | 51.1282 | 57.6633 | 1000 | 2918 | 63 |
| combined_lora_r8_head | person | 63.1893 | 71.6297 | 295 | 545 | 17 |
| combined_lora_r8_head | car | 47.8434 | 49.7910 | 1441 | 3056 | 85 |
| gradient_full_ft | person | 66.3299 | 74.4327 | 233 | 544 | 16 |
| gradient_full_ft | car | 51.9152 | 57.6897 | 1081 | 2661 | 74 |
| gradient_lora_r8_head | person | 67.9222 | 77.0734 | 210 | 542 | 18 |
| gradient_lora_r8_head | car | 48.7691 | 51.9627 | 1636 | 2948 | 73 |
| original_full_ft | person | 67.5832 | 76.6905 | 337 | 513 | 17 |
| original_full_ft | car | 52.0257 | 58.6102 | 1256 | 2734 | 82 |
| original_lora_r8_head | person | 66.5468 | 75.4382 | 228 | 545 | 26 |
| original_lora_r8_head | car | 48.8736 | 51.4959 | 1461 | 2998 | 62 |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>역사 실험: class_contrastive_staged3plus27 · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 | person AP50 | car AP50:95 | 전체 mAP50:95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / extension_continued_lora | gradient · continued LoRA | 506 | 3+27 | 41.7243 | 72.6721 | 55.4455 | 미확인 |
| YOLO26s / extension_supcon | gradient · LoRA+class SupCon | 506 | 3+27 | 41.6896 | 72.7156 | 55.2859 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA | IDF1 | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| extension_continued_lora | person | 70.6905 | 80.9895 | 107 | 558 | 12 |
| extension_continued_lora | car | 48.2851 | 50.8063 | 1867 | 2925 | 81 |
| extension_supcon | person | 69.6664 | 79.4563 | 99 | 549 | 18 |
| extension_supcon | car | 47.7719 | 50.0162 | 1898 | 2906 | 81 |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>역사 실험: id_bundle_seed707_extra30_raw · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 | person AP50 | car AP50:95 | 전체 mAP50:95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / id_id_supcon | ID-SupCon bundle 추가 학습 | 707 | 追加30 | 44.8057 | 76.5531 | 58.8270 | 미확인 |
| YOLO26s / id_continued_ft | continued Full FT 대조 | 707 | 追加30 | 44.6888 | 76.2608 | 58.8092 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA | IDF1 | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| id_id_supcon | person | 63.6071 | 71.1822 | 329 | 513 | 16 |
| id_id_supcon | car | 51.0136 | 57.3864 | 1096 | 2701 | 103 |
| id_continued_ft | person | 66.6170 | 75.4133 | 301 | 521 | 19 |
| id_continued_ft | car | 50.7996 | 58.0115 | 1085 | 2744 | 84 |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>역사 실험: teacher_kd_seed202_linear30_4090 · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 | person AP50 | car AP50:95 | 전체 mAP50:95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / kd30_student_ft | Full FT matched student 대조 | 202 | 30 | 44.4447 | 76.1630 | 58.8505 | 미확인 |
| YOLO26s / kd30_student_kd | feature KD student | 202 | 30 | 43.1643 | 76.0179 | 57.8921 | 미확인 |
| YOLO26m / kd30_teacher_m | Full FT teacher M | 202 | 30 | 49.0326 | 80.7519 | 62.5398 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA | IDF1 | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| kd30_student_ft | person | 70.7070 | 81.0319 | 201 | 526 | 15 |
| kd30_student_ft | car | 52.4678 | 57.8733 | 1100 | 2662 | 63 |
| kd30_student_kd | person | 65.7655 | 74.9420 | 332 | 525 | 20 |
| kd30_student_kd | car | 51.8699 | 56.8670 | 1069 | 2806 | 53 |
| kd30_teacher_m | person | 70.7814 | 79.6269 | 230 | 451 | 25 |
| kd30_teacher_m | car | 53.5707 | 59.2283 | 1135 | 2497 | 70 |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>역사 실험: teacher_kd_seed202_schedule60_4090 · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 | person AP50 | car AP50:95 | 전체 mAP50:95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / schedule60_student_ft_linear60 | Full FT linear60 · 4090 | 202 | 60 | 44.9319 | 76.9563 | 59.6659 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA | IDF1 | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| schedule60_student_ft_linear60 | person | 63.6633 | 71.8369 | 337 | 517 | 20 |
| schedule60_student_ft_linear60 | car | 51.4825 | 57.9357 | 825 | 2845 | 62 |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>역사 실험: matched_local1080Ti_v12 · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 | person AP50 | car AP50:95 | 전체 mAP50:95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / local60_v12_student_ft_linear60 | Full FT linear60 · 1080Ti | 202 | 60 | 44.8575 | 76.7890 | 59.5214 | 미확인 |
| YOLO26s / local60_v12_student_ft_cosine60 | Full FT cosine60 · 1080Ti | 202 | 60 | 44.7178 | 76.9162 | 59.5365 | 미확인 |
| YOLO26s / local60_v12_student_kd_linear60 | feature KD linear60 · 1080Ti | 202 | 60 | 44.2334 | 76.5013 | 59.3550 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA | IDF1 | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| local60_v12_student_ft_linear60 | person | 67.6725 | 77.5407 | 321 | 526 | 12 |
| local60_v12_student_ft_linear60 | car | 53.5025 | 60.9618 | 720 | 2744 | 59 |
| local60_v12_student_ft_cosine60 | person | 65.5681 | 74.8777 | 278 | 515 | 21 |
| local60_v12_student_ft_cosine60 | car | 49.8029 | 58.2127 | 794 | 3074 | 62 |
| local60_v12_student_kd_linear60 | person | 66.9051 | 76.2452 | 295 | 535 | 17 |
| local60_v12_student_kd_linear60 | car | 50.4331 | 55.0022 | 903 | 2774 | 70 |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>별도 반복 실험 · seed303/404 및 M/X 완료 범위</summary>

YOLO26s full-dose, TRAIN10491장, sourceVAL1144장, batch4, AdamW,30 epoch, 최종 epoch29. 아래 person AP50:95는 원문에 확인된 소수 둘째 자리 정밀도만 보존합니다.

| 모델 | 방법 | seed | TRAIN | epoch | person AP50:95 | 기타 지표 |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26s | Full FT | 303 | 10491 | 30 | 48.01 | AP50 / 전체 mAP / car / tracking 미확인 |
| YOLO26s | LoRA r8+head | 303 | 10491 | 30 | 45.10 | 동일 |
| YOLO26s | Full FT | 404 | 10491 | 30 | 47.68 | 동일 |
| YOLO26s | LoRA r8+head | 404 | 10491 | 30 | 45.35 | 동일 |


| 모델 | 확인된 상태 | person AP50:95 | person AP50 | 주의 |
| --- | --- | --- | --- | --- |
| YOLO26m | TRAIN4196 / seed202 / teacher Full FT30 완료 | 49.0326 | 80.7520 | 역사 Teacher/KD 블록; 최근6종과 별도 |
| YOLO26x | Full FT30 및 source 평가 완료 receipt | 미확인 | 미확인 | 로컬 최종 가중치 없음 / 추적 미평가 / YOLOX와 다름 |

원문: `lwir_nested_full_seed_repeat_20261014_v1`의 완료 checkpoint와 source 평가 요약. M의 정확값은 역사 CSV에 보존되어 있습니다. 최근26s 전체 mAP25.4758과 과거 person AP47–48은 다른 지표이므로 성능 하락으로 단정하지 않습니다.

</details>

<details>
<summary>역사19 runtime 참고 · 순수 추론 또는 통제 PC benchmark 아님</summary>

480×640, batch1, FP32,1924프레임, warmup5. source read/decode + predict + CPU transfer/tracking의 기존 합산 기록이며 JSON 쓰기는 제외합니다. 각 timing record의 추론 GPU명·개별 측정시각·median/p95는 미기록입니다. 학습에 사용한 GPU명을 추론 GPU명으로 대체하지 않습니다.

| 학습 블록 | 조건 | 읽기 포함 FPS | predict+track FPS | 읽기 포함 평균 ms/frame | predict+track 평균 ms/frame |
| --- | --- | --- | --- | --- | --- |
| native_input_seed202 | clahe_full_ft | 29.8011 | 41.9094 | 33.5559 | 23.8610 |
| native_input_seed202 | clahe_lora_r8_head | 36.8206 | 41.2531 | 27.1587 | 24.2406 |
| native_input_seed202 | combined_full_ft | 31.8534 | 35.1014 | 31.3939 | 28.4889 |
| native_input_seed202 | combined_lora_r8_head | 31.2442 | 34.3795 | 32.0060 | 29.0871 |
| native_input_seed202 | gradient_full_ft | 34.3921 | 38.1861 | 29.0764 | 26.1876 |
| native_input_seed202 | gradient_lora_r8_head | 34.1624 | 37.9124 | 29.2720 | 26.3766 |
| native_input_seed202 | original_full_ft | 40.7227 | 46.2036 | 24.5564 | 21.6434 |
| native_input_seed202 | original_lora_r8_head | 40.9993 | 46.5447 | 24.3907 | 21.4847 |
| class_contrastive_staged3plus27 | extension_continued_lora | 36.8445 | 40.9653 | 27.1411 | 24.4109 |
| class_contrastive_staged3plus27 | extension_supcon | 25.4244 | 41.5367 | 39.3324 | 24.0751 |
| id_bundle_seed707_extra30_raw | id_id_supcon | 39.1712 | 44.3536 | 25.5290 | 22.5461 |
| id_bundle_seed707_extra30_raw | id_continued_ft | 40.8876 | 46.4293 | 24.4573 | 21.5381 |
| teacher_kd_seed202_linear30_4090 | kd30_student_ft | 40.7308 | 46.2445 | 24.5514 | 21.6242 |
| teacher_kd_seed202_linear30_4090 | kd30_student_kd | 40.5708 | 46.1078 | 24.6483 | 21.6883 |
| teacher_kd_seed202_linear30_4090 | kd30_teacher_m | 34.8353 | 38.8658 | 28.7065 | 25.7296 |
| teacher_kd_seed202_schedule60_4090 | schedule60_student_ft_linear60 | 37.5953 | 42.4860 | 26.5990 | 23.5371 |
| matched_local1080Ti_v12 | local60_v12_student_ft_linear60 | 23.5144 | 44.2267 | 42.5271 | 22.6108 |
| matched_local1080Ti_v12 | local60_v12_student_ft_cosine60 | 37.9095 | 42.7153 | 26.3786 | 23.4108 |
| matched_local1080Ti_v12 | local60_v12_student_kd_linear60 | 40.2389 | 45.6569 | 24.8516 | 21.9025 |

원문은 위 역사19 selection evidence의 각 runtime/timing record입니다. 최근640×640 BYTE 관측값, 혼합 batch evaluator 값과 직접 순위를 만들지 않습니다.

</details>

표의 검증용 정밀 CSV/JSON과 PNG는 원본 Library ZIP에 보존했습니다. 저장소 현재 루트는 README.md와 모델 계열 폴더만 유지합니다. 학습 완료·검출 평가·추적 평가·DXNN 컴파일·실기 검증은 각각 다른 상태입니다.
<!-- PERFORMANCE_TABLES_END -->


Deployment binaries are grouped by detector family. Family folders contain DXNN files only; all instructions and compilation provenance are collected here. Only existing, verified compiled artifacts are published.

## Downloads and completion status

| Family/model | Status | Artifact |
|---|---|---|
| YOLO26n | Compiled; NPU validation pending | [yolo26/yolo26n.dxnn](yolo26/yolo26n.dxnn) |
| YOLO26s | Compiled; NPU validation pending | [yolo26/yolo26s.dxnn](yolo26/yolo26s.dxnn) |
| YOLOX-Nano | Compiled; NPU validation pending | [yolox/yolox_nano.dxnn](yolox/yolox_nano.dxnn) |
| YOLOX-s | Compiled; NPU validation pending | [yolox/yolox_s.dxnn](yolox/yolox_s.dxnn) |
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

The current root contains README.md and the yolo26/, yolox/, yolov8/ model-family folders. Family folders contain DXNN files only. Comparison PNGs remain in the original local/Library package and Git history; they were removed from the current tree at the user's request. PT/ONNX binaries, model-specific JSON/YAML/Markdown and older layout records were removed from the current tree. Earlier Git commits still contain those historical files; no history rewrite or force push was performed. Update branch-relative download URLs to the links above. Local research originals and conversion intermediates are preserved.

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
## Published yolox_s

Artifact: [yolox/yolox_s.dxnn](yolox/yolox_s.dxnn) — 9541952 bytes; SHA-256 `bd33a2b5cf0c38a537bed76dc09485bdcb94c517296c56c8c842cf677464efb6`.

Common six-model screen: TRAIN10742, seed101, fixed30 epochs, SGD, batch16,640. Source checkpoint SHA-256 `2c0ba3087733411e7d89443d27d621f0a482dacd56531f7bcbe213921a644070`; source ONNX SHA-256 `236d790348ad4b94a573326aa0ff8c3ff7739245f77293850d2633f08e0249c6`. Final checkpoint, not validation-selected best.

CPU-only DX-COM 2.4.1-rc.1, opt_level0, INT8 EMA100 steps; compilation 203.1s. ONNX checker, finite CPU execution, three TRAIN image raw PyTorch/ONNX parity and artifact hash passed. The DXNN has not been run on an NPU; board accuracy/FPS are unmeasured.

Calibration:256 fixed evenly spaced images from the full FLIR TRAIN10742 list only (no VAL/TEST); manifest SHA-256 `a32a796e1cb2335e8f30d75cd3a52e4211ff6aa9dc80da9e5b106072906ef192`. Input/output: uint8 NHWC [1,640,640,3]; confirm compiler artifact metadata; replicate grayscale to3 channels; YOLOX top-left pad114640x640 BGR0..255, no normalization before submission. Source output [1, 8400, 21]: raw decoded xywh + objectness +16 class scores;8400 candidates, external objectness-times-class scoring and class-aware NMS. Follow per-model preprocessing; the old historical YOLO26s uses a different end-to-end head.
YOLOX uses BGR0..255 with top-left114 padding, no division by255; output includes separate objectness and16 class scores. External objectness-times-class scoring and NMS are required.
## Published yolo26s_clahe_full_ft_s202_e30

Artifact: [yolo26/yolo26s_clahe_full_ft_s202_e30.dxnn](yolo26/yolo26s_clahe_full_ft_s202_e30.dxnn) — 10895043 bytes; SHA-256 `2ecdbb2f2106a52977d1f630376abca5076fc39b415a5e85982a64185440830e`.

Previously completed variant: native_input_seed202; method full_ft, seed202, recorded stage epochs30, training support4196, input representation clahe. This is a separate recipe and is not a row in the common six-model screen. Deployment export uses square640x640; no new accuracy evaluation of this deployment geometry was run. Original completion evidence: `D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json`.. Source checkpoint SHA-256 `aa4fba2b9dfc40aa9067d83f4c9f5f0e0057bc10f91da3ae8def30c12f349a48`; source ONNX SHA-256 `6fd687080cecd05a07b021fd4e0e63a38df04090c1f86ad3a1a4d28884a9b38d`. Final checkpoint, not validation-selected best.

CPU-only DX-COM 2.4.1-rc.1, opt_level0, INT8 EMA100 steps; compilation 228.1s. ONNX checker, finite CPU execution, three TRAIN image raw PyTorch/ONNX parity and artifact hash passed. The DXNN has not been run on an NPU; board accuracy/FPS are unmeasured.

Calibration:256 fixed evenly spaced images from the full FLIR TRAIN10742 list only (no VAL/TEST); manifest SHA-256 `5591747ff78e77e24aa2d4e373cafd660605088c35c20f53b628c0a3018658d1`. Input/output: uint8 NHWC [1,640,640,3]; confirm compiler artifact metadata; centered LetterBox640x640 padding114; external clahe semantic channel transform after padding and RGB conversion; see original representation implementation, do not replicate gray only. Source output [1, 20, 8400]: raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS. Follow per-model preprocessing; the old historical YOLO26s uses a different end-to-end head.
Division by255 and HWC-to-CHW transpose are fused; resize/padding and channel preparation remain external. Do not normalize a second time.
Adapter convolutions merged: 0. Parity compares the original unfused model against the merged/fused ONNX on three TRAIN images. Semantic channel assignment is external and occurs after padding and BGR-to-RGB conversion: R=original grayscale; G=CLAHE(clipLimit2.0,tileGrid8x8) for clahe/combined, otherwise grayscale; B=round(sqrt(clip(Sobel-magnitude/1443,0,1))*255) for gradient/combined, otherwise grayscale. Sobel uses CV_32F, ksize3. Compiler calibration PNGs reverse semantic RGB into BGR, and the configured BGR2RGB restores the correct semantic channel order. Do not submit three identical grayscale channels to these variants.


## Published yolo26s_clahe_lora_r8_head_s202_e30

Artifact: [yolo26/yolo26s_clahe_lora_r8_head_s202_e30.dxnn](yolo26/yolo26s_clahe_lora_r8_head_s202_e30.dxnn) — 10895048 bytes; SHA-256 `f62c0177880c0a0196bc03e4955b3b6108ab6086570ee79498bba717369f4d7d`.

Previously completed variant: native_input_seed202; method lora_r8_head, seed202, recorded stage epochs30, training support4196, input representation clahe. This is a separate recipe and is not a row in the common six-model screen. Deployment export uses square640x640; no new accuracy evaluation of this deployment geometry was run. Original completion evidence: `D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json`.. Source checkpoint SHA-256 `ff6d05c60c877b1a92b046c7433d19ff614d9e09a34a4e89fe2bef9ca42b6486`; source ONNX SHA-256 `3bd1d2b30598cfcd171dad155e00cb1ccc162297b80c80664bbae845907d0144`. Final checkpoint, not validation-selected best.

CPU-only DX-COM 2.4.1-rc.1, opt_level0, INT8 EMA100 steps; compilation 231.2s. ONNX checker, finite CPU execution, three TRAIN image raw PyTorch/ONNX parity and artifact hash passed. The DXNN has not been run on an NPU; board accuracy/FPS are unmeasured.

Calibration:256 fixed evenly spaced images from the full FLIR TRAIN10742 list only (no VAL/TEST); manifest SHA-256 `5591747ff78e77e24aa2d4e373cafd660605088c35c20f53b628c0a3018658d1`. Input/output: uint8 NHWC [1,640,640,3]; confirm compiler artifact metadata; centered LetterBox640x640 padding114; external clahe semantic channel transform after padding and RGB conversion; see original representation implementation, do not replicate gray only. Source output [1, 20, 8400]: raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS. Follow per-model preprocessing; the old historical YOLO26s uses a different end-to-end head.
Division by255 and HWC-to-CHW transpose are fused; resize/padding and channel preparation remain external. Do not normalize a second time.
Adapter convolutions merged: 74. Parity compares the original unfused model against the merged/fused ONNX on three TRAIN images. Semantic channel assignment is external and occurs after padding and BGR-to-RGB conversion: R=original grayscale; G=CLAHE(clipLimit2.0,tileGrid8x8) for clahe/combined, otherwise grayscale; B=round(sqrt(clip(Sobel-magnitude/1443,0,1))*255) for gradient/combined, otherwise grayscale. Sobel uses CV_32F, ksize3. Compiler calibration PNGs reverse semantic RGB into BGR, and the configured BGR2RGB restores the correct semantic channel order. Do not submit three identical grayscale channels to these variants.


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
| [yolox/yolox_s.dxnn](yolox/yolox_s.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 8400, 21] | 1: output0 FLOAT [1, 8400, 21] | xywh4 + objectness1 + class16; grid/stride/exp decoded; objectness*class, NMS external; internal NPU boundary tensors[9] |
| [yolo26/yolo26s_clahe_full_ft_s202_e30.dxnn](yolo26/yolo26s_clahe_full_ft_s202_e30.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 20, 8400] | 1: output0 FLOAT [1, 20, 8400] | xywh4 + class16; boxes decoded; confidence filtering, xywh-to-xyxy and NMS external; internal NPU boundary tensors[6] |
| [yolo26/yolo26s_clahe_lora_r8_head_s202_e30.dxnn](yolo26/yolo26s_clahe_lora_r8_head_s202_e30.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 20, 8400] | 1: output0 FLOAT [1, 20, 8400] | xywh4 + class16; boxes decoded; confidence filtering, xywh-to-xyxy and NMS external; internal NPU boundary tensors[6] |
| [yolo26/yolo26s_ft_s101_e30_legacy_20261001.dxnn](yolo26/yolo26s_ft_s101_e30_legacy_20261001.dxnn) | 16 | images: FLOAT [1, 3, 640, 640] | output0: FLOAT [1, 300, 6] | 1: output0 FLOAT [1, 300, 6] | xyxy4 + confidence + class index,300 selected detection slots; end-to-end decoder included; internal NPU boundary tensors[6] |

- New YOLO26n/s/m exports explicitly disable end-to-end selection to expose the one-to-many head. Their20 output fields are4 decoded box coordinates plus16 probabilities; no objectness and no DFL-bin channels in the public output. These actual YOLO26 ONNX graphs contain no DFL-named nodes.
- YOLOv8n/s include DFL decoding inside ONNX. The16-bin box distributions are reduced before the public output, so the public shape is [1,20,8400], not [1,80,8400].
- YOLOX exports explicitly enable decode_in_inference: grid offsets, stride scaling and exp width/height decoding are already in ONNX. [1,8400,21] contains4 decoded box coordinates,1 objectness and16 class probabilities. Do not decode grid/stride a second time. Multiply objectness by class probability before filtering/NMS.
- Historical YOLO26s [1,300,6] is a different end-to-end export:300 detection slots, six fields per slot. Six is not the class count. Apply confidence filtering and inverse letterbox; do not attach the new20-channel raw-head decoder.
- ONNX input is float32 NCHW [1,3,640,640], named images. Compiled NPU input is uint8 NHWC [1,640,640,3] per compiler guidance. UL input normalization by255 and layout transpose are fused. YOLOX has no division by255; its layout transpose is fused. External geometry and semantic channel transforms remain required per model.
- The public output0 is float32 according to embedded CPU ONNX metadata; verify actual DXRT buffer metadata on hardware. No claim of board execution, accuracy or FPS follows from these static checks.

Verified output class index mapping: 0=person, 1=bicycle, 2=car, 3=motorcycle, 4=bus, 5=train, 6=truck, 7=traffic light, 8=fire hydrant, 9=stop sign, 10=dog, 11=deer, 12=skateboard, 13=stroller, 14=scooter, 15=other vehicle. YOLOX COCO category IDs are one-based (index+1); detector output class indices are zero-based.

Read-only checkpoint audit:89 locally available checkpoints were loaded on CPU and all had nc16. The87 Ultralytics checkpoints carry the same explicit16-class names mapping and consistent detection-head nc; the two YOLOX state dictionaries have16 output channels in every classification head, with class order confirmed against the original training pack. The completed YOLO26x teacher checkpoint is unavailable locally, so its actual model object could not be inspected. The89 count includes54 additional candidates with smoke/stability/wrapper records and is not a count of89 formal experiments.

<!-- VERIFIED_IO_END -->
