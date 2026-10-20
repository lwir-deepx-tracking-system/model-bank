# LWIR DXNN Model Bank

학습을 마친 LWIR 탐지기의 **INT8 PTQ 컴파일 파일**을 계열별로 제공합니다. 성능표는 **원본 FP32 모델 평가**이며, DX-M1에서 양자화 후 정확도·속도·추적 성능은 아직 측정하지 않았습니다.

## 1. 모델 다운로드

현재 게시된 DXNN은 **26개**입니다. 아래 대표 모델과 추가 변형을 구분합니다.

| 계열 | 모델 | 학습 구분 | 파일 |
| --- | --- | --- | --- |
| yolo26 | YOLO26n | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolo26/yolo26n.dxnn) |
| yolo26 | YOLO26s | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolo26/yolo26s.dxnn) |
| yolo26 | YOLO26m teacher | teacher ·4196장 ·seed202 ·30ep | [DXNN](yolo26/yolo26m.dxnn) |
| yolov8 | YOLOv8n | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolov8/yolov8n.dxnn) |
| yolov8 | YOLOv8s | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolov8/yolov8s.dxnn) |
| yolox | YOLOX-Nano | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolox/yolox_nano.dxnn) |
| yolox | YOLOX-s | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolox/yolox_s.dxnn) |


<details>
<summary>추가 FT·LoRA·KD·입력 변형 다운로드 (19개)</summary>

| 모델 | 방법 / 입력 | seed | 기록 stage epoch | 파일 |
| --- | --- | --- | --- | --- |
| YOLO26s | full_ft / clahe | 202 | 30 | [yolo26s_clahe_full_ft_s202_e30.dxnn](yolo26/yolo26s_clahe_full_ft_s202_e30.dxnn) |
| YOLO26s | lora_r8_head / clahe | 202 | 30 | [yolo26s_clahe_lora_r8_head_s202_e30.dxnn](yolo26/yolo26s_clahe_lora_r8_head_s202_e30.dxnn) |
| YOLO26s | full_ft / combined | 202 | 30 | [yolo26s_combined_full_ft_s202_e30.dxnn](yolo26/yolo26s_combined_full_ft_s202_e30.dxnn) |
| YOLO26s | lora_r8_head / combined | 202 | 30 | [yolo26s_combined_lora_r8_head_s202_e30.dxnn](yolo26/yolo26s_combined_lora_r8_head_s202_e30.dxnn) |
| YOLO26s | full_ft / gradient | 202 | 30 | [yolo26s_gradient_full_ft_s202_e30.dxnn](yolo26/yolo26s_gradient_full_ft_s202_e30.dxnn) |
| YOLO26s | lora_r8_head / gradient | 202 | 30 | [yolo26s_gradient_lora_r8_head_s202_e30.dxnn](yolo26/yolo26s_gradient_lora_r8_head_s202_e30.dxnn) |
| YOLO26s | full_ft / original | 202 | 30 | [yolo26s_original_full_ft_s202_e30.dxnn](yolo26/yolo26s_original_full_ft_s202_e30.dxnn) |
| YOLO26s | lora_r8_head / original | 202 | 30 | [yolo26s_original_lora_r8_head_s202_e30.dxnn](yolo26/yolo26s_original_lora_r8_head_s202_e30.dxnn) |
| YOLO26s | continued LoRA / gradient | 506 | 27 | [yolo26s_extension_continued_lora_s506_e27.dxnn](yolo26/yolo26s_extension_continued_lora_s506_e27.dxnn) |
| YOLO26s | LoRA + class SupCon / gradient | 506 | 27 | [yolo26s_extension_supcon_s506_e27.dxnn](yolo26/yolo26s_extension_supcon_s506_e27.dxnn) |
| YOLO26s | Full FT + ID SupCon / original | 707 | 30 | [yolo26s_id_id_supcon_s707_e30.dxnn](yolo26/yolo26s_id_id_supcon_s707_e30.dxnn) |
| YOLO26s | continued Full FT / original | 707 | 30 | [yolo26s_id_continued_ft_s707_e30.dxnn](yolo26/yolo26s_id_continued_ft_s707_e30.dxnn) |
| YOLO26s | full_ft / original | 202 | 30 | [yolo26s_kd30_student_ft_s202_e30.dxnn](yolo26/yolo26s_kd30_student_ft_s202_e30.dxnn) |
| YOLO26s | Full FT + KD / original | 202 | 30 | [yolo26s_kd30_student_kd_s202_e30.dxnn](yolo26/yolo26s_kd30_student_kd_s202_e30.dxnn) |
| YOLO26s | full_ft / original | 202 | 60 | [yolo26s_schedule60_student_ft_linear60_s202_e60.dxnn](yolo26/yolo26s_schedule60_student_ft_linear60_s202_e60.dxnn) |
| YOLO26s | full_ft / original | 202 | 60 | [yolo26s_local60_v12_student_ft_linear60_s202_e60.dxnn](yolo26/yolo26s_local60_v12_student_ft_linear60_s202_e60.dxnn) |
| YOLO26s | full_ft / original | 202 | 60 | [yolo26s_local60_v12_student_ft_cosine60_s202_e60.dxnn](yolo26/yolo26s_local60_v12_student_ft_cosine60_s202_e60.dxnn) |
| YOLO26s | Full FT + KD linear60 / original | 202 | 60 | [yolo26s_local60_v12_student_kd_linear60_s202_e60.dxnn](yolo26/yolo26s_local60_v12_student_kd_linear60_s202_e60.dxnn) |
| YOLO26s | Full FT / historical10742 | 101 | 30 | [yolo26s_ft_s101_e30_legacy_20261001.dxnn](yolo26/yolo26s_ft_s101_e30_legacy_20261001.dxnn) |

기록 stage epoch는 추가 학습 단계 길이일 수 있습니다. 학습 블록·체크포인트 해시는 아래 컴파일 상세에 보존합니다. 역사 모델의 학습/평가480×640과 이 배포 export의640×640은 구분해야 합니다. CLAHE·gradient 변형은 외부 채널 변환이 필요합니다.

</details>

YOLO26x teacher는 학습 완료 근거가 있으나 로컬 최종 가중치가 없어 미게시입니다. **YOLO26x와 YOLOX는 다른 모델**입니다.

## 2. 같은 평가군의 핵심 성능

**↑ 높을수록 좋음 · ↓ 낮을수록 좋음 · <ins>굵은 밑줄</ins>은 같은 평가군/seed 내 지표별 최적값**입니다. 동률은 모두 강조합니다. 통제되지 않은 runtime과 서로 다른 평가군 사이에는 우열 표시를 하지 않습니다.

### 검출 · 최근6종

sourceVAL1144장,16클래스, TRAIN10742장, seed101, Full FT30epoch,640×640. AP/mAP 단위는 %이며 아래 값은 AP50:95입니다.

| 모델 | 전체 mAP ↑ | person AP ↑ | car AP ↑ |
| --- | --- | --- | --- |
| yolov8n | 19.8696 | 36.8542 | 53.1387 |
| yolox-nano | 16.8664 | 31.6139 | 48.2667 |
| yolo26n | 20.7318 | 38.0337 | 53.9829 |
| yolov8s | 24.3720 | 42.2804 | 57.4554 |
| yolox-s | 22.8154 | 39.9558 | 55.8228 |
| yolo26s | **<ins>25.4758</ins>** | **<ins>44.9255</ins>** | **<ins>59.0974</ins>** |

confidence0.001, NMS IoU0.70, maxDets[1,10,300]. 평가 batch는 v8n4(OOM 복구), 나머지16입니다. AP50/AP75/AR300 및 출처는 아래 상세표에 있습니다.

### 추적 · 최근 person

동일 exposed DEV5영상·1924프레임, GT person2469,640×640, batch1, FP32. **검출 sourceVAL과 다른 데이터**입니다.

| 모델 / tracker | HOTA ↑ | IDF1 ↑ | MOTA ↑ |
| --- | --- | --- | --- |
| yolov8s / BYTE | 62.1219 | 76.8591 | 69.9068 |
| yolov8s / OCSORT | 59.6343 | 74.0936 | 70.8789 |
| yolov8s / BOTSORT_GMC | 67.4590 | 77.9966 | **<ins>72.2155</ins>** |
| yolo26s / BYTE | 63.7377 | **<ins>78.9417</ins>** | 70.5549 |
| yolo26s / OCSORT | 62.9723 | 78.4777 | 71.3244 |
| yolo26s / BOTSORT_GMC | **<ins>67.9369</ins>** | 78.2812 | 71.5269 |
| yolox-nano / BYTE | 49.7061 | 59.1651 | 34.9534 |
| yolox-nano / OCSORT | 52.0710 | 65.5621 | 42.4058 |
| yolox-nano / BOTSORT_GMC | 54.2056 | 61.8107 | 36.2900 |
| yolox-s / BYTE | 57.9518 | 70.5412 | 58.2017 |
| yolox-s / OCSORT | 59.1224 | 72.8120 | 60.5913 |
| yolox-s / BOTSORT_GMC | 62.5672 | 72.5552 | 58.5662 |
| yolov8n /3 trackers | 미평가 | 미평가 | 미평가 |
| yolo26n /3 trackers | 미평가 | 미평가 | 미평가 |

DetA/AssA/FP/FN/IDSW와 car/truck은 상세표에 있습니다. YOLOX-Nano·YOLOX-s는 동일 frozen DEV·3 tracker로 추가 평가했습니다. YOLOv8n·YOLO26n은 미평가입니다. 추가 실행 시간은 RF-DETR와 compiler가 동작한 경합 관측이므로 통제 latency 비교에 사용하지 않습니다.

### PC latency · 기존 관측값

GTX1080Ti, FP32, batch1,640×640,1924프레임, warmup5, CPU threads4. **순수 추론 또는 동일 조건 통제 benchmark가 아닙니다.**

| 모델 | predict+track ms ↓ | 읽기 포함 ms ↓ | 통제 benchmark |
| --- | --- | --- | --- |
| yolov8s | 12.7453 | 39.4114 | 미측정 |
| yolo26s | 14.8630 | 17.3547 | 미측정 |
| 다른4종 | 미측정 | 미측정 | 미측정 |

CPU 전송·ByteTrack 포함, JSON 쓰기 제외. 총 초÷프레임 수의 평균이며 median/p95·동시 부하·캐시 상태는 미기록입니다. 측정 범위가 다른 mixed-batch evaluator·역사 runtime과 비교하지 않습니다.

### 핵심 설정

| 항목 | 값 |
| --- | --- |
| AP / 추적 입력 confidence | 0.001 /0.10 |
| NMS IoU / max_det | 0.70 /300 |
| ByteTrack high / low / new | 0.25 /0.10 /0.25 |
| ByteTrack match / buffer / fuse_score | 0.80 /30 /true |
| 최근 추적 입력 | 640×640 / batch1 / FP32 |
| 역사 추적 입력 | 480×640 / 별도 BOTrack XYWH·NSA wrapper |


## 3. 입력과 출력 해석

대표 모델의 배포 입력은 compiler 안내상 **uint8 NHWC [1,640,640,3]**입니다. grayscale을3채널로 구성하고 패딩114로 중앙 letterbox합니다. 특히 이 공통6종 실험은640×512 영상을(0,64)에 붙인640×640 캔버스를 사용하므로 **YOLOX도 이 중앙 캔버스를 먼저 구성**해야 합니다. 원본 평가의 YOLOX ValTransform은 이미640×640인 캔버스에 적용되어 추가 패딩을 하지 않습니다. UL의 /255 정규화와 HWC→CHW는 컴파일에 포함되고, YOLOX에는 /255 정규화가 없습니다. **resize·패딩·색/의미 채널 변환·tracker는 외부 처리**입니다.

YOLOX의 현재 게시본은 calibration을 표준 좌상단 패딩으로 준비했으므로 원본 실험과 일치하는 중앙 캔버스 calibration으로 교정 대기 중입니다. 가중치·출력 형상 검증과 calibration geometry의 일치는 별도 확인 항목입니다.

| 모델 / export | 공개 output0 | 필요한 후처리 |
| --- | --- | --- |
| 새 YOLO26n/s/m ·YOLOv8n/s | [1,20,8400] | decoded xywh4 + class16; xyxy 변환·confidence·NMS |
| YOLOX-Nano/s | [1,8400,21] | decoded xywh4 + objectness1 + class16; obj×class·NMS |
| historical YOLO26s | [1,300,6] | selected xyxy4 + confidence + class index; confidence·좌표 복원 |

16은 클래스 수이며 공개 출력 텐서는 모델당1개입니다. 출력은 DXNN 내부 CPU ONNX 메타데이터상 float32이며, NPU/CPU 혼합 그래프일 수 있습니다. **확장자 DXNN이 모든 내부 연산·입출력이 INT8이라는 뜻은 아닙니다.** 실제 DXRT 버퍼는 장치에서 재확인해야 합니다.

## 상세 근거

<details>
<summary>전체 성능표 · 추적 세부 · 설정 · 역사19 · seed 반복 · runtime</summary>

### 최근 검출 6종 · 동일 sourceVAL

TRAIN10742장, sourceVAL1144장,16클래스, seed101, Full FT30 epoch의 최종 체크포인트입니다. 입력640×640, AP confidence0.001, NMS IoU0.70, COCO maxDets[1,10,300]. YOLOv8n은 OOM 후 평가 batch4, 나머지는16이므로 아래 평가 batch를 지연시간 순위로 해석하지 않습니다.

| 모델 | 전체 mAP50:95 ↑ | 전체 AP50 ↑ | 전체 AP75 ↑ | person AP50:95 ↑ | car AP50:95 ↑ | AR300 ↑ | 평가 batch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8n | 19.8696 | 34.3712 | 19.2929 | 36.8542 | 53.1387 | 32.6315 | 4 |
| yolox-nano | 16.8664 | 31.2794 | 14.8047 | 31.6139 | 48.2667 | 27.6596 | 16 |
| yolo26n | 20.7318 | 35.6591 | 20.7991 | 38.0337 | 53.9829 | 34.3948 | 16 |
| yolov8s | 24.3720 | 40.6337 | 25.3877 | 42.2804 | 57.4554 | 34.7286 | 16 |
| yolox-s | 22.8154 | 38.6608 | 22.5710 | 39.9558 | 55.8228 | 30.8326 | 16 |
| yolo26s | **<ins>25.4758</ins>** | **<ins>41.9578</ins>** | **<ins>25.6621</ins>** | **<ins>44.9255</ins>** | **<ins>59.0974</ins>** | **<ins>36.6120</ins>** | 16 |

원문: `yolo_family_screen_20261019_v1/common_score_comparison_full6_v2_20261019.csv`, evaluator v3. sourceVAL은 이미 노출된 평가셋이며 독립 TEST 결과가 아닙니다.

### 최근 추적 · 4검출기 × 3 tracker

노출된 DEV5영상·1924프레임, 입력640×640, batch1, FP32, detector confidence0.10, NMS IoU0.70, max_det300. 검출 sourceVAL과 다른 데이터입니다. YOLOv8s·YOLO26s 기존 평가에 YOLOX-Nano·YOLOX-S의 동일 DEV 평가를 추가했습니다. YOLOv8n·YOLO26n은 미평가입니다.

**person · GT boxes 2469**

| 모델 | tracker | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | MOTA ↑ | FP ↓ | FN ↓ | IDSW ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8s | BYTE | 62.1219 | 57.9567 | 66.6548 | 76.8591 | 69.9068 | 177 | 543 | 23 |
| yolov8s | OCSORT | 59.6343 | 57.8827 | 61.5119 | 74.0936 | 70.8789 | **<ins>155</ins>** | 542 | 22 |
| yolov8s | BOTSORT_GMC | 67.4590 | 63.6129 | **<ins>71.7691</ins>** | 77.9966 | **<ins>72.2155</ins>** | 201 | 467 | 18 |
| yolo26s | BYTE | 63.7377 | 59.8190 | 67.9860 | **<ins>78.9417</ins>** | 70.5549 | 191 | 518 | 18 |
| yolo26s | OCSORT | 62.9723 | 59.9617 | 66.2101 | 78.4777 | 71.3244 | 161 | 527 | 20 |
| yolo26s | BOTSORT_GMC | **<ins>67.9369</ins>** | **<ins>64.5543</ins>** | 71.7458 | 78.2812 | 71.5269 | 224 | 461 | 18 |
| yolox-nano | BYTE | 49.7061 | 42.8354 | 57.7493 | 59.1651 | 34.9534 | 947 | 615 | 44 |
| yolox-nano | OCSORT | 52.0710 | 44.8164 | 60.5459 | 65.5621 | 42.4058 | 758 | 626 | 38 |
| yolox-nano | BOTSORT_GMC | 54.2056 | 45.3043 | 65.0205 | 61.8107 | 36.2900 | 958 | 583 | 32 |
| yolox-s | BYTE | 57.9518 | 53.6890 | 62.7458 | 70.5412 | 58.2017 | 538 | 469 | 25 |
| yolox-s | OCSORT | 59.1224 | 54.3652 | 64.4497 | 72.8120 | 60.5913 | 474 | 476 | 23 |
| yolox-s | BOTSORT_GMC | 62.5672 | 57.0430 | 68.9806 | 72.5552 | 58.5662 | 570 | **<ins>436</ins>** | **<ins>17</ins>** |
| yolov8n | BYTE / OCSORT / BOTSORT_GMC | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 |
| yolo26n | BYTE / OCSORT / BOTSORT_GMC | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 |


**car · GT boxes 6668**

| 모델 | tracker | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | MOTA ↑ | FP ↓ | FN ↓ | IDSW ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8s | BYTE | 43.5842 | 37.7524 | 50.8984 | 53.8226 | 39.6521 | 693 | 3238 | 93 |
| yolov8s | OCSORT | 41.6231 | 37.1677 | 47.1162 | 50.9204 | 39.0222 | **<ins>634</ins>** | 3322 | 110 |
| yolov8s | BOTSORT_GMC | 50.1605 | 40.3313 | **<ins>63.0404</ins>** | 60.8233 | 41.9316 | 693 | 3122 | 57 |
| yolo26s | BYTE | 47.9415 | 44.9303 | 51.9791 | 57.0719 | 48.3653 | 834 | 2497 | 112 |
| yolo26s | OCSORT | 47.6287 | 44.6058 | 51.6394 | 55.1503 | 48.2154 | 784 | 2577 | 92 |
| yolo26s | BOTSORT_GMC | **<ins>53.7894</ins>** | **<ins>47.9854</ins>** | 61.2350 | **<ins>62.3271</ins>** | **<ins>51.7846</ins>** | 808 | **<ins>2361</ins>** | **<ins>46</ins>** |
| yolox-nano | BYTE | 36.0034 | 28.8718 | 45.4500 | 42.5055 | 17.7115 | 1727 | 3648 | 112 |
| yolox-nano | OCSORT | 34.5187 | 28.4942 | 42.3818 | 39.9389 | 18.2663 | 1555 | 3769 | 126 |
| yolox-nano | BOTSORT_GMC | 38.4045 | 30.4109 | 49.5141 | 43.0836 | 19.5411 | 1750 | 3541 | 74 |
| yolox-s | BYTE | 45.1315 | 39.8609 | 51.9689 | 52.9756 | 38.4223 | 1215 | 2772 | 119 |
| yolox-s | OCSORT | 47.1963 | 39.6055 | 57.0445 | 57.0129 | 37.9274 | 1158 | 2844 | 137 |
| yolox-s | BOTSORT_GMC | 50.0540 | 41.7791 | 60.8175 | 58.3726 | 40.1770 | 1217 | 2681 | 91 |
| yolov8n | BYTE / OCSORT / BOTSORT_GMC | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 |
| yolo26n | BYTE / OCSORT / BOTSORT_GMC | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 |


**truck · GT boxes 1180**

| 모델 | tracker | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | MOTA ↑ | FP ↓ | FN ↓ | IDSW ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8s | BYTE | 7.2595 | 9.7895 | 5.5417 | 11.4024 | -4.4915 | 189 | 1023 | 21 |
| yolov8s | OCSORT | 7.5302 | 9.1623 | 6.2953 | 12.4665 | -3.2203 | **<ins>165</ins>** | 1033 | 20 |
| yolov8s | BOTSORT_GMC | 10.3740 | 10.6709 | 10.3950 | 17.4198 | -2.7966 | 186 | 1019 | **<ins>8</ins>** |
| yolo26s | BYTE | **<ins>25.3696</ins>** | 24.1133 | **<ins>28.1567</ins>** | **<ins>32.0671</ins>** | -5.5932 | 445 | 778 | 23 |
| yolo26s | OCSORT | 19.2914 | 22.6585 | 17.0108 | 22.9188 | -4.0678 | 397 | 811 | 20 |
| yolo26s | BOTSORT_GMC | 22.9611 | **<ins>25.7977</ins>** | 21.3028 | 24.4676 | -1.2712 | 418 | **<ins>759</ins>** | 18 |
| yolox-nano | BYTE | 13.8147 | 15.1813 | 12.7798 | 16.6250 | -0.1695 | 194 | 954 | 34 |
| yolox-nano | OCSORT | 13.3004 | 14.2011 | 12.6561 | 14.0396 | 1.0169 | 173 | 966 | 29 |
| yolox-nano | BOTSORT_GMC | 17.3259 | 16.4612 | 18.6357 | 22.2769 | **<ins>1.5254</ins>** | 197 | 932 | 33 |
| yolox-s | BYTE | 11.5744 | 12.8464 | 10.6141 | 15.9613 | -10.6780 | 278 | 984 | 44 |
| yolox-s | OCSORT | 10.0681 | 12.3869 | 8.4384 | 12.3153 | -9.4915 | 262 | 998 | 32 |
| yolox-s | BOTSORT_GMC | 13.2726 | 14.4290 | 12.5515 | 16.3658 | -8.3898 | 276 | 974 | 29 |
| yolov8n | BYTE / OCSORT / BOTSORT_GMC | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 |
| yolo26n | BYTE / OCSORT / BOTSORT_GMC | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 | 미평가 |

원문: `yolo_family_tracking_report_only_20261019_v1/COMPARISON.json`. TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`. HOTA/DetA/AssA/IDF1/MOTA는 높을수록, FP/FN/IDSW는 낮을수록 좋습니다.

### 검출·tracker 설정

| 항목 | 최근 검출6종 | 최근 추적4×3 | 역사 추적19조건 |
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

| 모델 | 탐지+CPU 전송+ByteTrack 평균 ms/frame ↓ | 읽기·decode 포함 평균 ms/frame ↓ | 통제된 순수 추론 benchmark |
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


| 모델 | 평가 batch | 기록 ms/image ↓ | 동일 조건 PC benchmark |
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

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 ↑ | person AP50 ↑ | car AP50:95 ↑ | 전체 mAP50:95 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / clahe_full_ft | CLAHE · Full FT | 202 | 30 | 43.6077 | 75.4628 | 58.4576 | 미확인 |
| YOLO26s / clahe_lora_r8_head | CLAHE · LoRA r8+head | 202 | 30 | 40.9653 | 71.7479 | 55.0928 | 미확인 |
| YOLO26s / combined_full_ft | combined · Full FT | 202 | 30 | 43.9659 | 75.4729 | 58.5177 | 미확인 |
| YOLO26s / combined_lora_r8_head | combined · LoRA r8+head | 202 | 30 | 40.2628 | 70.7924 | 55.0202 | 미확인 |
| YOLO26s / gradient_full_ft | gradient · Full FT | 202 | 30 | 43.9518 | **<ins>75.4946</ins>** | 58.6052 | 미확인 |
| YOLO26s / gradient_lora_r8_head | gradient · LoRA r8+head | 202 | 30 | 41.3742 | 71.8697 | 55.0837 | 미확인 |
| YOLO26s / original_full_ft | original · Full FT | 202 | 30 | **<ins>44.1316</ins>** | 75.4051 | **<ins>58.8730</ins>** | 미확인 |
| YOLO26s / original_lora_r8_head | original · LoRA r8+head | 202 | 30 | 42.1545 | 73.3405 | 55.7646 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA ↑ | IDF1 ↑ | FP ↓ | FN ↓ | IDSW ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| clahe_full_ft | person | 66.4821 | 75.5098 | 336 | 517 | **<ins>16</ins>** |
| clahe_full_ft | car | 48.6943 | 52.4884 | 1262 | 2763 | 65 |
| clahe_lora_r8_head | person | 66.4160 | 75.6505 | 339 | 550 | 20 |
| clahe_lora_r8_head | car | 47.9142 | 50.2463 | 1332 | 3097 | **<ins>61</ins>** |
| combined_full_ft | person | 63.3713 | 71.4743 | 251 | **<ins>502</ins>** | 30 |
| combined_full_ft | car | 51.1282 | 57.6633 | **<ins>1000</ins>** | 2918 | 63 |
| combined_lora_r8_head | person | 63.1893 | 71.6297 | 295 | 545 | 17 |
| combined_lora_r8_head | car | 47.8434 | 49.7910 | 1441 | 3056 | 85 |
| gradient_full_ft | person | 66.3299 | 74.4327 | 233 | 544 | **<ins>16</ins>** |
| gradient_full_ft | car | 51.9152 | 57.6897 | 1081 | **<ins>2661</ins>** | 74 |
| gradient_lora_r8_head | person | **<ins>67.9222</ins>** | **<ins>77.0734</ins>** | **<ins>210</ins>** | 542 | 18 |
| gradient_lora_r8_head | car | 48.7691 | 51.9627 | 1636 | 2948 | 73 |
| original_full_ft | person | 67.5832 | 76.6905 | 337 | 513 | 17 |
| original_full_ft | car | **<ins>52.0257</ins>** | **<ins>58.6102</ins>** | 1256 | 2734 | 82 |
| original_lora_r8_head | person | 66.5468 | 75.4382 | 228 | 545 | 26 |
| original_lora_r8_head | car | 48.8736 | 51.4959 | 1461 | 2998 | 62 |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>역사 실험: class_contrastive_staged3plus27 · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 ↑ | person AP50 ↑ | car AP50:95 ↑ | 전체 mAP50:95 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / extension_continued_lora | gradient · continued LoRA | 506 | 3+27 | **<ins>41.7243</ins>** | 72.6721 | **<ins>55.4455</ins>** | 미확인 |
| YOLO26s / extension_supcon | gradient · LoRA+class SupCon | 506 | 3+27 | 41.6896 | **<ins>72.7156</ins>** | 55.2859 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA ↑ | IDF1 ↑ | FP ↓ | FN ↓ | IDSW ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| extension_continued_lora | person | **<ins>70.6905</ins>** | **<ins>80.9895</ins>** | 107 | 558 | **<ins>12</ins>** |
| extension_continued_lora | car | **<ins>48.2851</ins>** | **<ins>50.8063</ins>** | **<ins>1867</ins>** | 2925 | **<ins>81</ins>** |
| extension_supcon | person | 69.6664 | 79.4563 | **<ins>99</ins>** | **<ins>549</ins>** | 18 |
| extension_supcon | car | 47.7719 | 50.0162 | 1898 | **<ins>2906</ins>** | **<ins>81</ins>** |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>역사 실험: id_bundle_seed707_extra30_raw · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 ↑ | person AP50 ↑ | car AP50:95 ↑ | 전체 mAP50:95 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / id_id_supcon | ID-SupCon bundle 추가 학습 | 707 | 追加30 | **<ins>44.8057</ins>** | **<ins>76.5531</ins>** | **<ins>58.8270</ins>** | 미확인 |
| YOLO26s / id_continued_ft | continued Full FT 대조 | 707 | 追加30 | 44.6888 | 76.2608 | 58.8092 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA ↑ | IDF1 ↑ | FP ↓ | FN ↓ | IDSW ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| id_id_supcon | person | 63.6071 | 71.1822 | 329 | **<ins>513</ins>** | **<ins>16</ins>** |
| id_id_supcon | car | **<ins>51.0136</ins>** | 57.3864 | 1096 | **<ins>2701</ins>** | 103 |
| id_continued_ft | person | **<ins>66.6170</ins>** | **<ins>75.4133</ins>** | **<ins>301</ins>** | 521 | 19 |
| id_continued_ft | car | 50.7996 | **<ins>58.0115</ins>** | **<ins>1085</ins>** | 2744 | **<ins>84</ins>** |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>역사 실험: teacher_kd_seed202_linear30_4090 · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 ↑ | person AP50 ↑ | car AP50:95 ↑ | 전체 mAP50:95 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / kd30_student_ft | Full FT matched student 대조 | 202 | 30 | 44.4447 | 76.1630 | 58.8505 | 미확인 |
| YOLO26s / kd30_student_kd | feature KD student | 202 | 30 | 43.1643 | 76.0179 | 57.8921 | 미확인 |
| YOLO26m / kd30_teacher_m | Full FT teacher M | 202 | 30 | **<ins>49.0326</ins>** | **<ins>80.7519</ins>** | **<ins>62.5398</ins>** | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA ↑ | IDF1 ↑ | FP ↓ | FN ↓ | IDSW ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| kd30_student_ft | person | 70.7070 | **<ins>81.0319</ins>** | **<ins>201</ins>** | 526 | **<ins>15</ins>** |
| kd30_student_ft | car | 52.4678 | 57.8733 | 1100 | 2662 | 63 |
| kd30_student_kd | person | 65.7655 | 74.9420 | 332 | 525 | 20 |
| kd30_student_kd | car | 51.8699 | 56.8670 | **<ins>1069</ins>** | 2806 | **<ins>53</ins>** |
| kd30_teacher_m | person | **<ins>70.7814</ins>** | 79.6269 | 230 | **<ins>451</ins>** | 25 |
| kd30_teacher_m | car | **<ins>53.5707</ins>** | **<ins>59.2283</ins>** | 1135 | **<ins>2497</ins>** | 70 |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>역사 실험: teacher_kd_seed202_schedule60_4090 · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 ↑ | person AP50 ↑ | car AP50:95 ↑ | 전체 mAP50:95 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / schedule60_student_ft_linear60 | Full FT linear60 · 4090 | 202 | 60 | 44.9319 | 76.9563 | 59.6659 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA ↑ | IDF1 ↑ | FP ↓ | FN ↓ | IDSW ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| schedule60_student_ft_linear60 | person | 63.6633 | 71.8369 | 337 | 517 | 20 |
| schedule60_student_ft_linear60 | car | 51.4825 | 57.9357 | 825 | 2845 | 62 |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>역사 실험: matched_local1080Ti_v12 · 별도 평가군</summary>

TRAIN4196장; 검출은 sourceVAL1144장, 추적은 별도 exposed DEV5영상1924프레임입니다. 아래 두 종류의 지표는 같은 데이터셋에서 계산되지 않았습니다. 입력·학습 stage·seed·장비가 다른 블록과 합쳐 순위를 만들지 않습니다.

| 모델 / 조건 | 방법 | seed | 기록 stage epoch | person AP50:95 ↑ | person AP50 ↑ | car AP50:95 ↑ | 전체 mAP50:95 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s / local60_v12_student_ft_linear60 | Full FT linear60 · 1080Ti | 202 | 60 | **<ins>44.8575</ins>** | 76.7890 | 59.5214 | 미확인 |
| YOLO26s / local60_v12_student_ft_cosine60 | Full FT cosine60 · 1080Ti | 202 | 60 | 44.7178 | **<ins>76.9162</ins>** | **<ins>59.5365</ins>** | 미확인 |
| YOLO26s / local60_v12_student_kd_linear60 | feature KD linear60 · 1080Ti | 202 | 60 | 44.2334 | 76.5013 | 59.3550 | 미확인 |

추적:480×640, batch1, FP32, ByteTrack + BOTrack XYWH/NSA wrapper. 역사 집계에는 DetA/AssA/MOTA가 없어 미확인으로 유지합니다.

| 조건 | class | HOTA ↑ | IDF1 ↑ | FP ↓ | FN ↓ | IDSW ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| local60_v12_student_ft_linear60 | person | **<ins>67.6725</ins>** | **<ins>77.5407</ins>** | 321 | 526 | **<ins>12</ins>** |
| local60_v12_student_ft_linear60 | car | **<ins>53.5025</ins>** | **<ins>60.9618</ins>** | **<ins>720</ins>** | **<ins>2744</ins>** | **<ins>59</ins>** |
| local60_v12_student_ft_cosine60 | person | 65.5681 | 74.8777 | **<ins>278</ins>** | **<ins>515</ins>** | 21 |
| local60_v12_student_ft_cosine60 | car | 49.8029 | 58.2127 | 794 | 3074 | 62 |
| local60_v12_student_kd_linear60 | person | 66.9051 | 76.2452 | 295 | 535 | 17 |
| local60_v12_student_kd_linear60 | car | 50.4331 | 55.0022 | 903 | 2774 | 70 |

원문: `lwir_kd_schedule_horizon_20261015_v1/evidence_snapshots/completed19_after_local_kd_linear60_20261016_v1/selection_evidence_v3.json`.

</details>

<details>
<summary>별도 반복 실험 · seed303/404 및 M/X 완료 범위</summary>

YOLO26s full-dose, TRAIN10491장, sourceVAL1144장, batch4, AdamW,30 epoch, 최종 epoch29. 아래 person AP50:95는 원문에 확인된 소수 둘째 자리 정밀도만 보존합니다.

| 모델 | 방법 | seed | TRAIN | epoch | person AP50:95 ↑ | 기타 지표 |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26s | Full FT | 303 | 10491 | 30 | **<ins>48.01</ins>** | AP50 / 전체 mAP / car / tracking 미확인 |
| YOLO26s | LoRA r8+head | 303 | 10491 | 30 | 45.10 | 동일 |
| YOLO26s | Full FT | 404 | 10491 | 30 | **<ins>47.68</ins>** | 동일 |
| YOLO26s | LoRA r8+head | 404 | 10491 | 30 | 45.35 | 동일 |


| 모델 | 확인된 상태 | person AP50:95 ↑ | person AP50 ↑ | 주의 |
| --- | --- | --- | --- | --- |
| YOLO26m | TRAIN4196 / seed202 / teacher Full FT30 완료 | 49.0326 | 80.7520 | 역사 Teacher/KD 블록; 최근6종과 별도 |
| YOLO26x | Full FT30 및 source 평가 완료 receipt | 미확인 | 미확인 | 로컬 최종 가중치 없음 / 추적 미평가 / YOLOX와 다름 |

원문: `lwir_nested_full_seed_repeat_20261014_v1`의 완료 checkpoint와 source 평가 요약. M의 정확값은 역사 CSV에 보존되어 있습니다. 최근26s 전체 mAP25.4758과 과거 person AP47–48은 다른 지표이므로 성능 하락으로 단정하지 않습니다.

</details>

<details>
<summary>역사19 runtime 참고 · 순수 추론 또는 통제 PC benchmark 아님</summary>

480×640, batch1, FP32,1924프레임, warmup5. source read/decode + predict + CPU transfer/tracking의 기존 합산 기록이며 JSON 쓰기는 제외합니다. 각 timing record의 추론 GPU명·개별 측정시각·median/p95는 미기록입니다. 학습에 사용한 GPU명을 추론 GPU명으로 대체하지 않습니다.

| 학습 블록 | 조건 | 읽기 포함 FPS ↑ | predict+track FPS ↑ | 읽기 포함 평균 ms/frame ↓ | predict+track 평균 ms/frame ↓ |
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


추가 YOLOX 원문: `yolox_dev_tracking_20261020_v1/COMPARISON.json` 및 hash manifest. 같은 frozen DEV·checkpoint·tracker 설정을 사용했으며 탐지1회 결과를3 tracker에 재사용했습니다. 추가 실행은 CPU threads1·batch1 FP32, RF-DETR·CPU compiler와 동시 실행한 GPU 관측이므로 해당 실행 시간을 기존 통제 latency로 비교하지 않습니다. 원본2종의 기존 성능값은 수정하지 않았습니다.

</details>

<details>
<summary>클래스16 · 실제 tensor metadata · 변형별 외부 전처리</summary>

### 클래스와 디코더

ONNX source input: images, float32 NCHW[1,3,640,640]. 새26 export는 end2end=False, xyxy=False, reg_max1이며 공개 출력은 이미 decoded xywh입니다. PyTorch/ONNX parity의 참조 모델도 같은 raw-head mode로 설정한 unfused 모델입니다. YOLOv8은 reg_max16의 DFL decoding을 graph 내부에서 마쳐 공개 출력에 DFL bin이 남지 않습니다. YOLOX는 decode_in_inference=True로 grid/stride/exp 복원이 graph에 포함돼 있어 다시 decode하지 않습니다. 역사26s는 별도 end-to-end selected head입니다.


| class index | 이름 |
| --- | --- |
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

YOLOX COCO category ID는 index+1이며 detector output index는0부터 시작합니다.

### 외부 의미 채널 변환

원본 uint8 grayscale을 letterbox한 다음 BGR→RGB 변환 후 의미 채널을 배정합니다. R=원본, G=CLAHE(clipLimit2.0,tileGrid8×8)는 clahe/combined일 때만, B=Sobel magnitude의 고정 sqrt 압축은 gradient/combined일 때만 사용하며 나머지 채널은 원본입니다. Sobel CV_32F, ksize3; gradient=round(sqrt(clip(magnitude/1443,0,1))*255). 최종 semantic RGB 순서를 그대로 공급해야 합니다.

이 전처리는 **DXNN에 포함되지 않습니다**. calibration PNG는 semantic RGB를 BGR로 뒤집어 저장하고 loader의 BGR2RGB로 복원했습니다. LoRA는 원본 forward와 병합/fusion/ONNX 결과의3개 TRAIN sample parity를 검증합니다. 단순히 가중치 파일만 교체하고 grayscale3채널을 넣으면 의미 채널 변형의 올바른 입력이 아닙니다.


| artifact | 공개 출력 수 | 공개 출력 메타데이터 | 내부 NPU 경계 tensor수 |
| --- | --- | --- | --- |
| yolo26m.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26n.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolov8s.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 4 |
| yolo26s.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolox_nano.dxnn | 1 | output0 [1, 8400, 21] FLOAT | 9 |
| yolox_s.dxnn | 1 | output0 [1, 8400, 21] FLOAT | 9 |
| yolo26s_clahe_full_ft_s202_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_clahe_lora_r8_head_s202_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_combined_full_ft_s202_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_combined_lora_r8_head_s202_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_gradient_full_ft_s202_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_gradient_lora_r8_head_s202_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_original_full_ft_s202_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_original_lora_r8_head_s202_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_extension_continued_lora_s506_e27.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_extension_supcon_s506_e27.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_id_id_supcon_s707_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_id_continued_ft_s707_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_kd30_student_ft_s202_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_kd30_student_kd_s202_e30.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_schedule60_student_ft_linear60_s202_e60.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_local60_v12_student_ft_linear60_s202_e60.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_local60_v12_student_ft_cosine60_s202_e60.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolo26s_local60_v12_student_kd_linear60_s202_e60.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolov8n.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 4 |
| yolo26s_ft_s101_e30_legacy_20261001.dxnn | 1 | output0 [1, 300, 6] FLOAT | 6 |

DXNN header graph_info와 embedded CPU ONNX ValueInfo를 읽은 정적 확인입니다. 내부 NPU 경계 tensor 여러 개를 공개 출력 tensor 수와 혼동하지 않습니다.

</details>

<details>
<summary>체크포인트·해시·calibration·모델별 컴파일 상세</summary>

### 공통 컴파일 조건

기존 x86-64 Ubuntu WSL, DEEPX DX-COM2.4.1(package2.4.1-rc.1), DX-M1 SDK default(M1A), optimization0, CPU INT8 PTQ, EMA calibration100steps. torch2.14.1+cpu, ONNX1.23.2, ORT1.31.0. Source export: Windows torch2.7.0+cu118, Ultralytics8.4.157, ONNX1.18.0, ORT1.19.2; static batch1,640×640, opset17. YOLOX source commit6ddff4824372906469a7fae2dc3206c7aa4bbaee.

Calibration은 전체 FLIR TRAIN10742 목록에서 round(i*(N-1)/255), i=0..255로 뽑은256장입니다. TRAIN-list SHA-256 `d40cb99963d55de372d0b46cfcc11652078b51d5d1c17321c3040621ed63f1d1`; VAL/TEST는 사용하지 않았습니다.4196장 등 부분집합 학습 모델에는 calibration pool과 학습 subset이 다릅니다. Source hash, ONNX checker·finite CPU execution·3개 TRAIN raw parity, DX-COM 종료·DXNN hash를 확인했습니다. NPU 보드 정확도·FPS 검증과는 다릅니다.

### yolo26m.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | LWIR-YOLO26m-TEACHER-4196-S202-E29 |
| 학습 방법 / seed / stage epochs | full_ft /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_tracking_recovery_20261015_v1\recovered\m\runs\teacher_m\train\weights\epoch29.pt |
| DXNN bytes | 21991058 |
| DXNN SHA-256 | `fbda1f1ba2182090e1f2f4dc3a6306347cbeec9701633c7e21c47c4a34c2c03e` |
| source PT SHA-256 | `dca0f23d1a945b6a73b17ef050452b87e3e7d99265eb413519b5395db67c7736` |
| source ONNX SHA-256 | `a36ce1ecfa49f798cc6ff977eea65501206681e068836317fe38caf67689713b` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 439.723 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.002960205078125 / score 5.945563316345215e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

M teacher 원본 학습4196장·seed202·AdamW·480×640·batch4·30epoch·2367updates와 배포640×640을 구분합니다.

### yolo26n.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | LWIR-yolo26n-FT-10742-S101-E29 |
| 학습 방법 / seed / stage epochs | full_ft /101 /30 |
| TRAIN support | 10742 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\yolo_family_screen_20261019_v1\full_n\yolo26n\yolo26n_full_epoch30_last.pt |
| DXNN bytes | 3467688 |
| DXNN SHA-256 | `0b865a0d05fa2b53b2a52d54914f37c5abcaa54f41a1059c69ca747e03460bee` |
| source PT SHA-256 | `dc6f25aea34bb67989fdff31ad2ce3ea993d4ae690ab22861f4b2240dc872fc9` |
| source ONNX SHA-256 | `2ddda14087ac21a5365b7d94a40e60f95e6b83c1a74e271646eac540262e0400` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 157.786 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00510406494140625 / score 1.0251998901367188e-05 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolov8s.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | LWIR-yolov8s-FT-10742-S101-E29 |
| 학습 방법 / seed / stage epochs | full_ft /101 /30 |
| TRAIN support | 10742 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\yolo_family_screen_20261019_v1\full_s\yolov8s\last.pt |
| DXNN bytes | 11857151 |
| DXNN SHA-256 | `c9783a3ce9765b35163bc876fab6a4b6332314dcab71c3fca282161a3ceaa79f` |
| source PT SHA-256 | `53b640f03323a43b4751128108a762b8b4795b2f82834b391a2a19df4dbfb0eb` |
| source ONNX SHA-256 | `146d4f64773af2d3aba40acd7b7433d9729d8819e556da8d6e8e963d4e3dddcf` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 250.115 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00274658203125 / score 6.407499313354492e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | LWIR-yolo26s-FT-10742-S101-E29 |
| 학습 방법 / seed / stage epochs | full_ft /101 /30 |
| TRAIN support | 10742 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\yolo_family_screen_20261019_v1\full_s\yolo26s\last.pt |
| DXNN bytes | 10895020 |
| DXNN SHA-256 | `7ac754185cc99e807bd824dfa617e8cc696f02a29c402076820967c906e7098b` |
| source PT SHA-256 | `13346581137ebd1bea8e1e820c0b969b439d4859b3638685ad394fe6c116d4cc` |
| source ONNX SHA-256 | `d021ac4097e31b3943d8029eb30be8cee500fe41ee11c308e36a8582181f3b90` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 247.393 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.01220703125 / score 9.775161743164062e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolox_nano.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | LWIR-yolox-nano-FT-10742-S101-E29 |
| 학습 방법 / seed / stage epochs | full_ft /101 /30 |
| TRAIN support | 10742 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\yolo_family_screen_20261019_v1\full_n\yolox-nano_audit_20261019_v1\latest_ckpt.pth |
| DXNN bytes | 1536206 |
| DXNN SHA-256 | `60333bccb6ec0b41dad61f10998fe5084b2aeb143596eebb16cc25f705ecd169` |
| source PT SHA-256 | `64119cfcc291fcb436839cf29c11f3e089d7a3eee0580f5edce2049ebc35877f` |
| source ONNX SHA-256 | `3c181badb88bff9a8d9c7f2788a3648eced42f24407a97ff2e27bd0ea045d271` |
| calibration manifest SHA-256 | `a32a796e1cb2335e8f30d75cd3a52e4211ff6aa9dc80da9e5b106072906ef192` |
| compile elapsed seconds | 144.065 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.0077362060546875 / score 2.199411392211914e-05 |
| 공개 source output | [1, 8400, 21]; raw decoded xywh + objectness +16 class scores;8400 candidates, external objectness-times-class scoring and class-aware NMS |

### yolox_s.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | LWIR-yolox-s-FT-10742-S101-E29 |
| 학습 방법 / seed / stage epochs | full_ft /101 /30 |
| TRAIN support | 10742 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\yolo_family_screen_20261019_v1\full_s\yolox-s\latest_ckpt.pth |
| DXNN bytes | 9541952 |
| DXNN SHA-256 | `bd33a2b5cf0c38a537bed76dc09485bdcb94c517296c56c8c842cf677464efb6` |
| source PT SHA-256 | `2c0ba3087733411e7d89443d27d621f0a482dacd56531f7bcbe213921a644070` |
| source ONNX SHA-256 | `236d790348ad4b94a573326aa0ff8c3ff7739245f77293850d2633f08e0249c6` |
| calibration manifest SHA-256 | `a32a796e1cb2335e8f30d75cd3a52e4211ff6aa9dc80da9e5b106072906ef192` |
| compile elapsed seconds | 203.103 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00958251953125 / score 1.6033649444580078e-05 |
| 공개 source output | [1, 8400, 21]; raw decoded xywh + objectness +16 class scores;8400 candidates, external objectness-times-class scoring and class-aware NMS |

### yolo26s_clahe_full_ft_s202_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | native_input_seed202 |
| 학습 방법 / seed / stage epochs | full_ft /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | clahe; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895043 |
| DXNN SHA-256 | `2ecdbb2f2106a52977d1f630376abca5076fc39b415a5e85982a64185440830e` |
| source PT SHA-256 | `aa4fba2b9dfc40aa9067d83f4c9f5f0e0057bc10f91da3ae8def30c12f349a48` |
| source ONNX SHA-256 | `6fd687080cecd05a07b021fd4e0e63a38df04090c1f86ad3a1a4d28884a9b38d` |
| calibration manifest SHA-256 | `5591747ff78e77e24aa2d4e373cafd660605088c35c20f53b628c0a3018658d1` |
| compile elapsed seconds | 228.113 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.0021820068359375 / score 7.867813110351562e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_clahe_lora_r8_head_s202_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | native_input_seed202 |
| 학습 방법 / seed / stage epochs | lora_r8_head /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | clahe; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895048 |
| DXNN SHA-256 | `f62c0177880c0a0196bc03e4955b3b6108ab6086570ee79498bba717369f4d7d` |
| source PT SHA-256 | `ff6d05c60c877b1a92b046c7433d19ff614d9e09a34a4e89fe2bef9ca42b6486` |
| source ONNX SHA-256 | `3bd1d2b30598cfcd171dad155e00cb1ccc162297b80c80664bbae845907d0144` |
| calibration manifest SHA-256 | `5591747ff78e77e24aa2d4e373cafd660605088c35c20f53b628c0a3018658d1` |
| compile elapsed seconds | 231.225 |
| 병합 adapter 수 | 74 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00250244140625 / score 7.957220077514648e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_combined_full_ft_s202_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | native_input_seed202 |
| 학습 방법 / seed / stage epochs | full_ft /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | combined; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895046 |
| DXNN SHA-256 | `32694d910ffef52366ff7113535b02a841a4014e303634db0df586898dc44ccc` |
| source PT SHA-256 | `02c8f0af7c4d08d31512b8e0c2aa2eae5d97cbed4f22fb6258e443dc7a8487d9` |
| source ONNX SHA-256 | `49d1b4259694676d8030457a6e44f3898d341b9be8d10ab4da9300e884e95240` |
| calibration manifest SHA-256 | `284992956b2f5ceb7c2f60eda3e07b6e090908b1b063b3a28048d8f6de82a581` |
| compile elapsed seconds | 206.247 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00152587890625 / score 6.377696990966797e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_combined_lora_r8_head_s202_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | native_input_seed202 |
| 학습 방법 / seed / stage epochs | lora_r8_head /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | combined; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895051 |
| DXNN SHA-256 | `d042d26f37ffe8d028315737fc2d315323d1110991cc89f7e817f51603ebbf80` |
| source PT SHA-256 | `406b87dfcfd74ac1cdd3d68d693dc8eea9b8b6a5e8ad65471aa9e9a40d45f081` |
| source ONNX SHA-256 | `bea4cb87696a8facfe233cf814b5be548194fef8f1fda7a6b55f4ebde62dfce8` |
| calibration manifest SHA-256 | `284992956b2f5ceb7c2f60eda3e07b6e090908b1b063b3a28048d8f6de82a581` |
| compile elapsed seconds | 252.266 |
| 병합 adapter 수 | 74 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.0039520263671875 / score 7.420778274536133e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_gradient_full_ft_s202_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | native_input_seed202 |
| 학습 방법 / seed / stage epochs | full_ft /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | gradient; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895046 |
| DXNN SHA-256 | `6aeef0d7ce3043c7ec64b07a93ecad440a5cc947a84625ad74439695fb1f27e3` |
| source PT SHA-256 | `352cb5e4dd1b7db8532dd9836472be3e3d85fa12f8f827eff7b58df874842df2` |
| source ONNX SHA-256 | `7d5647f0c9f88e634e2286bdd148a588fa014b9de526db162f5e054eab8a32aa` |
| calibration manifest SHA-256 | `df709ba7b8ee24639e7338f87c0b35adf5054807d68a20dbe0a0bf87caf6487b` |
| compile elapsed seconds | 207.379 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.0029296875 / score 1.0579824447631836e-05 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_gradient_lora_r8_head_s202_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | native_input_seed202 |
| 학습 방법 / seed / stage epochs | lora_r8_head /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | gradient; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895051 |
| DXNN SHA-256 | `c7fa6f4e148c57e22eb005a7be9c999a08418e089f814601785a2e313e9e2c0d` |
| source PT SHA-256 | `dc19e4efd2110323daf6d13eb0487ad41a7f8e380152fc321cdfd0f7abf7bfb8` |
| source ONNX SHA-256 | `d8b856e52f2cc53ffa15d2bb856fd0c9f8ac371ea014accb6acd97c31a296809` |
| calibration manifest SHA-256 | `df709ba7b8ee24639e7338f87c0b35adf5054807d68a20dbe0a0bf87caf6487b` |
| compile elapsed seconds | 214.566 |
| 병합 adapter 수 | 74 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.003509521484375 / score 8.948147296905518e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_original_full_ft_s202_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | native_input_seed202 |
| 학습 방법 / seed / stage epochs | full_ft /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895046 |
| DXNN SHA-256 | `2b35f4be8db888e822b8399b34eaa9252bc43468afdd7ab5a18c1b5cf80d8644` |
| source PT SHA-256 | `bf3282ed2dfeaa88bfae746dde09020ae9c6a8639fe3f9408464cc041065c5f5` |
| source ONNX SHA-256 | `e7204602d37b5629c9ac4d130184ce718292d2ebdd471787589e50d5d16c36d8` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 199.123 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00162506103515625 / score 4.202127456665039e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_original_lora_r8_head_s202_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | native_input_seed202 |
| 학습 방법 / seed / stage epochs | lora_r8_head /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895051 |
| DXNN SHA-256 | `464eddbccd29bd0622050646189d73e53225b1d3b3ac1bdaa698fbfa48a7e8f7` |
| source PT SHA-256 | `00f662c753030394c7eec089627c858d69624d5ed68c7320bfce45e0d4547bcc` |
| source ONNX SHA-256 | `33381866d940aec318af864a2205c3d3a44c4b6e8cd990ffb21682560ac98b04` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 221.595 |
| 병합 adapter 수 | 74 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.002838134765625 / score 1.2889504432678223e-05 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_extension_continued_lora_s506_e27.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | class_contrastive_staged3plus27 |
| 학습 방법 / seed / stage epochs | continued LoRA /506 /27 |
| TRAIN support | 4196 |
| 입력 representation | gradient; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895054 |
| DXNN SHA-256 | `4171bac7c3e4790c5c504d528d77feb2e1d9d5390220f350bf55dcaa02b51571` |
| source PT SHA-256 | `5f130b03ba986bcc47d1daf9520e36911551157f4b6afc08edad2c864cda4667` |
| source ONNX SHA-256 | `fb2c704a19e9251460dc880a82aa8af1836f6aece63f1c23bb657a923d02f6ed` |
| calibration manifest SHA-256 | `df709ba7b8ee24639e7338f87c0b35adf5054807d68a20dbe0a0bf87caf6487b` |
| compile elapsed seconds | 204.650 |
| 병합 adapter 수 | 74 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.0023193359375 / score 8.519738912582397e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_extension_supcon_s506_e27.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | class_contrastive_staged3plus27 |
| 학습 방법 / seed / stage epochs | LoRA + class SupCon /506 /27 |
| TRAIN support | 4196 |
| 입력 representation | gradient; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895046 |
| DXNN SHA-256 | `256ba6905f7dbba92cb30d9499b4bac2dc92be0ad5943bb725892569c7bd895a` |
| source PT SHA-256 | `4ea7e9208a9c6cc1b7f8c619e9ab64b20188aa0719fea861a71c11a72cb6b994` |
| source ONNX SHA-256 | `110f47c788b2c853b6fd1e91b8f03cef3b3c14e828390f3117fcce6f6e98e62b` |
| calibration manifest SHA-256 | `df709ba7b8ee24639e7338f87c0b35adf5054807d68a20dbe0a0bf87caf6487b` |
| compile elapsed seconds | 213.488 |
| 병합 adapter 수 | 74 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00238037109375 / score 9.573996067047119e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_id_id_supcon_s707_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | id_bundle_seed707_extra30_raw |
| 학습 방법 / seed / stage epochs | Full FT + ID SupCon /707 /30 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895042 |
| DXNN SHA-256 | `c3de647894faebf785403b3450b211b98482d5d4f4ab77878ed8868f2116b376` |
| source PT SHA-256 | `1c173843e90152ecaf1a55ce214a4653b4b31eb747a8a0443bd99c9c8c5997e2` |
| source ONNX SHA-256 | `a77748333ff0fb491f8c159ec97612f516f113ade57ece53d5011df4e38c41b4` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 202.779 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00115966796875 / score 5.5730342864990234e-05 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_id_continued_ft_s707_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | id_bundle_seed707_extra30_raw |
| 학습 방법 / seed / stage epochs | continued Full FT /707 /30 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895045 |
| DXNN SHA-256 | `8db9e19d39b70798d66f4341caae3e8580555e63bc2deb30bc6953a3b1401ed7` |
| source PT SHA-256 | `70ba24668e3a7b4216015659945b4f56431f85f8ca0defb9414e31123c842b0c` |
| source ONNX SHA-256 | `0594f71fd7e5ad4c1b8b39f2a9c23957bf49a6adbe471f85da5562269da11e20` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 237.502 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00249481201171875 / score 1.0132789611816406e-05 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_kd30_student_ft_s202_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | teacher_kd_seed202_linear30_4090 |
| 학습 방법 / seed / stage epochs | full_ft /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895045 |
| DXNN SHA-256 | `dcff9adee8a55389137933ccc82c291b3e6e934aae596606d8338dc43f839368` |
| source PT SHA-256 | `8b8a40a34fff457558f548ac8abc05ea226ffd7c9a1737871878dfc66cba35f0` |
| source ONNX SHA-256 | `27f6bb8728e7af7bd861ab11b0751aae79ed6387d1cd150b990cebe616fa1c58` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 219.311 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.001842498779296875 / score 4.678964614868164e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_kd30_student_kd_s202_e30.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | teacher_kd_seed202_linear30_4090 |
| 학습 방법 / seed / stage epochs | Full FT + KD /202 /30 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10890245 |
| DXNN SHA-256 | `47410681251bc107262ee36323a068f9b18bd0197177fd9d096799c3a35d2572` |
| source PT SHA-256 | `5cd33f55d28b5041d9c43aa88e8e1cd16ec1dc485788c970e08cc0432ca075fc` |
| source ONNX SHA-256 | `272e9900d83795ca3dba2f975f05a1c92a2725cecf6f835fad925b388dc1efe5` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 198.556 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.000946044921875 / score 3.769993782043457e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_schedule60_student_ft_linear60_s202_e60.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | teacher_kd_seed202_schedule60_4090 |
| 학습 방법 / seed / stage epochs | full_ft /202 /60 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895060 |
| DXNN SHA-256 | `3d354c78259d6e67c8fb92ee79f428d15a473abbc6d853f092abd282ff14cce9` |
| source PT SHA-256 | `9053374a124c2bce4f6f5ca1a128cd4fe498c3e8202af33fb3306d6b18ba1afe` |
| source ONNX SHA-256 | `5947addaa925a08f3ce39e3e63d0ed2c60498d238e48c213445ca6f718021214` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 208.701 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00152587890625 / score 1.823902130126953e-05 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_local60_v12_student_ft_linear60_s202_e60.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | matched_local1080Ti_v12 |
| 학습 방법 / seed / stage epochs | full_ft /202 /60 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895061 |
| DXNN SHA-256 | `772ea829e9d5ed7afb52728b2b7aead81ba11327a57dbbba0d21e6f6edf39f88` |
| source PT SHA-256 | `2f98faf042e2a14e69f2a0823e86494d74b0f856bf9ea1c0133bfc993e6c4196` |
| source ONNX SHA-256 | `0f3c611748a09a67ca53863e6c5711421de9b4eff540a0806bedeaed2d407f2d` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 209.667 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00164794921875 / score 4.112720489501953e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_local60_v12_student_ft_cosine60_s202_e60.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | matched_local1080Ti_v12 |
| 학습 방법 / seed / stage epochs | full_ft /202 /60 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10895061 |
| DXNN SHA-256 | `576da41a86c9ef332d9103ac9fdcfbda55e81c3f3fddca37f5d2a02e0be02e59` |
| source PT SHA-256 | `c6d5b0f159df651233aa7c7f0d5e11c0742b9151ee31664fd30eff51fcb876fd` |
| source ONNX SHA-256 | `f5f3e74be46df23c7589770608851badfc3c574f1d78ceac0c674eec384fc44b` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 205.514 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.00152587890625 / score 5.304813385009766e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_local60_v12_student_kd_linear60_s202_e60.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | matched_local1080Ti_v12 |
| 학습 방법 / seed / stage epochs | Full FT + KD linear60 /202 /60 |
| TRAIN support | 4196 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\lwir_kd_schedule_horizon_20261015_v1\evidence_snapshots\completed19_after_local_kd_linear60_20261016_v1\selection_evidence_v3.json |
| DXNN bytes | 10890261 |
| DXNN SHA-256 | `7415528737b672eec22ce23095741cba56fbcdcfdfafacea802162c245f8aade` |
| source PT SHA-256 | `32c5168629a7c281d16e8466959c8462a000d482dad4ceb6ac774616c6f6385c` |
| source ONNX SHA-256 | `7b44c96a3f2f678156256b3d6be4bc5a48bd216cac02daa39bb60cb4027512c5` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 193.818 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.0012054443359375 / score 8.046627044677734e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolov8n.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | yolov8n |
| 학습 방법 / seed / stage epochs | full_ft /101 /30 |
| TRAIN support | 10742 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\yolo_family_screen_20261019_v1\full_n\yolov8n\yolov8n_full_epoch30_last.pt |
| DXNN bytes | 3696192 |
| DXNN SHA-256 | `f11c80721fb88d7c4cb153b165d381e778e5bf8de1bc5302d32496baf70831d7` |
| source PT SHA-256 | `5cd07d31ed8861c46903d504e558460192f409bb3bb44179484c06e2759fb63e` |
| source ONNX SHA-256 | `e6d02de05cc5b461ee0159fda746bc64226ff881283bbe05828c078d12bda36f` |
| calibration manifest SHA-256 | `78b52315e77e0156a04bfb31f82d7106f9f71f18c8d4c0aafe3b0addd2375e34` |
| compile elapsed seconds | 154.234 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.001678466796875 / score 4.351139068603516e-06 |
| 공개 source output | [1, 20, 8400]; raw decoded xywh +16 class scores, 8400 candidates; no separate objectness; external class-aware NMS |

### yolo26s_ft_s101_e30_legacy_20261001.dxnn

| 항목 | 값 |
| --- | --- |
| 학습 | historical YOLO26s FT /10742장 /seed101 /30epoch /AdamW lr3e-4 /BN stats frozen |
| DXNN bytes | 10897723 |
| DXNN SHA-256 | `1e82b415d86e9f26a549f5ee92af183f733879c77dafc40f7ff97adbf826b9f3` |
| source PT SHA-256 | `385621f65165f181421e1b4e4a209e32b7cb08d5f8258e95c54866439e1e2329` |
| source ONNX SHA-256 | `c91342f76eca4c7178720405b71764d151ec91356a22e003d46fe951092170d0` |
| compiler note | one NPU + one CPU group; graph-rewrite cosine0.8506 warning near TopK/NMS; board validation pending |

### 보존 범위

로컬 원본과 중간 ONNX/캘리브레이션 자료는 보존하며, 저장소 계열 폴더에는 DXNN만 둡니다. PNG/정밀 CSV/JSON은 원본 Library ZIP `libfile_473f586f45ac8191bef1087765316cc2`(SHA-256 `be592eed621b340e112782f81e340eed5c49f5a6f9690dce36c44ba3c47bbe74`)과 로컬에 보존했습니다. 이전 Git 이력의 파일은 삭제하지 않았으며 force push도 하지 않았습니다.

Ultralytics/model, DEEPX SDK 및 FLIR 이용 조건은 별도로 적용됩니다. [DEEPX compiler](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-COM/DX-COMPILER_Agent_Driven_Development/) · [Ultralytics DEEPX](https://docs.ultralytics.com/integrations/deepx/).

</details>
