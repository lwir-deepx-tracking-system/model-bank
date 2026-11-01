# LWIR DXNN Model Bank

## YOLO26x 열영상 PT 모델

사람 검출 및 검수용 슈도 라벨 후보에 사용하는 Full FT 모델 `yolo26/yolo26x.pt`입니다(2026-11-01, FLIR 학습 10,491장, 사람 AP50:95 53.2791%). **PT는 DXNN이 아니며 DX-M1 컴파일·실행을 검증하지 않았습니다.** 이전 YOLO26x(4,196장으로 학습, 69 epoch째 가중치, 45.3642%)는 삭제했고 git 기록 `0928ac9`에 남아 있습니다. **이전 모델과 데이터 증강·학습 캔버스가 다릅니다** — [yolo26x.pt 상세](#yolo26xpt)를 보십시오.


학습을 마친 LWIR 탐지기의 **INT8 PTQ 컴파일 파일**을 계열별로 제공합니다. 성능표는 **원본 FP32 모델 평가**이며, DX-M1에서 양자화 후 정확도·속도·추적 성능은 아직 측정하지 않았습니다. 예외로 **YOLO26m-RT·YOLO26m-CS는 DX-M1 INT8 실측값**을 아래 [DX-M1 INT8 실측](#dx-m1-int8-실측--yolo26m-rt--yolo26m-cs)에 따로 적었습니다.

## 1. 모델 다운로드

현재 게시된 DXNN은 단일 모델 **5개**(2026-11-01 기존 YOLO26m 교사 `yolo26m.dxnn` 삭제, git 기록 `1e957ad`에 남음), NPU 2코어용 2파일 패키지 **2종(YOLO26m-RT, YOLO26m-CS · 파일 4개)**, 사람 재식별용 **ViT-lite1 1개**이고, 이와 별도로 슈도 라벨용 PT **YOLO26x 1개**가 있습니다(`yolo26x_train_metadata.json`은 컴파일이 아니라 학습 메타데이터). 각 DXNN 옆의 `*_compile_metadata.json`은 DXNN 파일 안에 DX-COM이 저장한 컴파일 메타데이터(header · graph_info · compile_config · rmap_info)를 그대로 추출한 것입니다.

| 계열 | 모델 | 학습 구분 | 파일 |
| --- | --- | --- | --- |
| yolo26 | YOLO26s | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolo26/yolo26s.dxnn) · [JSON](yolo26/yolo26s_compile_metadata.json) |
| yolo26 | YOLO26m-RT (E08) | fine-tuned · 2타일 512×384 · NPU0/NPU1 | [A DXNN](yolo26/yolo26m_rt_a.dxnn) · [A JSON](yolo26/yolo26m_rt_a_compile_metadata.json) · [B DXNN](yolo26/yolo26m_rt_b.dxnn) · [B JSON](yolo26/yolo26m_rt_b_compile_metadata.json) |
| yolo26 | YOLO26m-CS (KDF) | two-tower 0.75 · KD 최종 · 480×640 · NPU0/NPU1 | [A DXNN](yolo26/yolo26m_cs_a.dxnn) · [A JSON](yolo26/yolo26m_cs_a_compile_metadata.json) · [B DXNN](yolo26/yolo26m_cs_b.dxnn) · [B JSON](yolo26/yolo26m_cs_b_compile_metadata.json) |
| yolo26 | YOLO26x (PT · 슈도라벨) | Full FT ·10491장 ·seed202 ·30ep(best 29) · DXNN 없음 | [PT](yolo26/yolo26x.pt) · [JSON](yolo26/yolo26x_train_metadata.json) |
| reid | ViT-lite1 (person re-ID) | SOLIDER 선생님 → FLIR 사람 crop KD 학생 · 128×64 · NPU2 | [DXNN](reid/vit_lite1_reid.dxnn) · [JSON](reid/vit_lite1_reid_compile_metadata.json) |
| yolov8 | YOLOv8n | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolov8/yolov8n.dxnn) · [JSON](yolov8/yolov8n_compile_metadata.json) |
| yolov8 | YOLOv8s | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolov8/yolov8s.dxnn) · [JSON](yolov8/yolov8s_compile_metadata.json) |
| yolox | YOLOX-Nano | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolox/yolox_nano.dxnn) · [JSON](yolox/yolox_nano_compile_metadata.json) |
| yolox | YOLOX-s | 공통6종 ·10742장 ·seed101 ·30ep | [DXNN](yolox/yolox_s.dxnn) · [JSON](yolox/yolox_s_compile_metadata.json) |


> 2026-10-22 저장소 경량화: YOLO26n, YOLO26s 추가 변형 28개, `experiments/yolo26m_two_core`를 제거하고 git 기록에서도 정리했습니다 (clone 약 449MB → 약 43MB). 아래 성능·컴파일 기록에 남은 해당 이름은 기록용입니다. 정리 전 저장소 전체는 별도 백업으로 보관합니다.

YOLO26x와 YOLOX는 다른 모델입니다. YOLO26x는 아래 PT 모델로 별도 제공합니다.

### DX-M1 INT8 실측 · YOLO26m-RT / YOLO26m-CS

두 모델은 각각 A·B 두 DXNN을 NPU_0·NPU_1에 동시에 올려 한 프레임을 처리합니다. 공개 출력은 `boxes`·`scores` raw 텐서이며, decode·NMS는 외부에서 합니다.

| 모델 | 입력 | A · B 역할 | 합치는 방법 |
| --- | --- | --- | --- |
| YOLO26m-RT (E08) | uint8 [1,512,384,3] × 2 | 640×512 프레임의 열 0:384 · 256:640 | 타일별 Decoder(512,384) → ownership join → NMS |
| YOLO26m-CS (KDF) | uint8 [1,480,640,3] × 2 | 같은 프레임(행 16:496)을 두 tower에 입력 | 두 raw 출력 평균 → Decoder(480,640) 1회 → NMS |

DX-M1 (Orange Pi 5 Plus, DXRT 3.4.2, NPU 1000MHz), FLIR ADAS v2 `video_thermal_test` 8영상 · 3749프레임 · GT person 12323. **위 FP32 표(sourceVAL·DEV5)와 데이터가 달라 직접 비교하지 않습니다.** 검출은 confidence 0.001 · NMS IoU 0.70 · max 300, COCO maxDets[1,10,300], 16클래스 예측 중 **GT가 있는 10클래스의 평균**(person·bike·car·motor·truck·light·hydrant·sign·dog·other vehicle; bike 113개·dog 25개는 AP≈0). 추적은 person만(프레임별 class 0 필터), IoU 0.5, TrackEval HOTA/DetA/AssA · motmetrics IDF1/MOTA/IDSW(TrackEval CLEAR 기준이면 MOTA 0.03~0.08 낮고 IDSW 3~6 많음).

| 모델 | mAP50:95 ↑ | AP50 ↑ | AP75 ↑ | person AP ↑ | car AP ↑ | AR300 ↑ | 검출 ms p50 / p95 ↓ | NPU ms (A / B) ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26m-RT | 17.2538 | 30.6676 | 17.1007 | 46.6323 | 50.4572 | 32.4809 | 25.77 / 29.53 | 11.77 / 11.78 |
| YOLO26m-CS | 16.2227 | 28.7995 | 15.8576 | 46.9800 | 51.5364 | 31.0124 | 28.24 / 32.29 | 12.60 / 12.76 |

검출 ms는 같은 프로세스에서 프레임마다 번갈아 잰 8비트 프레임 → 최종 검출까지의 시간입니다(JPEG 읽기 제외, 파이프라인 없이 제출 → 대기 → 디코드 직렬). 아래 추적 표의 처리 주기는 파이프라인(다음 프레임 선제출) 실행이라 검출 ms의 역수보다 짧으며, **두 표의 속도는 서로 비교하지 않습니다.**

| 모델 / tracker | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | MOTA ↑ | IDSW ↓ | 처리 주기 ms (fps) ↓ | 지연 p50 / p95 ms ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26m-RT / rebind + ViT-lite1 | 54.0168 | 50.4523 | 58.5011 | 65.0132 | 54.8649 | 114 | 19.64 (50.9) | 31.43 / 41.37 |
| YOLO26m-CS / ByteTrack | 54.4108 | 49.6273 | 60.2450 | 64.1369 | 53.3880 | 117 | 23.54 (42.5) | 38.72 / 44.67 |
| YOLO26m-CS / rebind | 53.9474 | 48.9952 | 60.0443 | 64.7884 | 50.4828 | 104 | 23.47 (42.6) | 38.77 / 44.88 |
| YOLO26m-CS / rebind + ViT-lite1 | 53.6826 | 48.8877 | 59.5922 | 64.3460 | 50.3043 | 109 | 23.47 (42.6) | 39.15 / 45.60 |

추적은 검출·추적·ViT를 파이프라인으로 돌린 실행입니다(ViT-lite1은 NPU_2). 같은 파이프라인 조건의 ByteTrack(ViT 없음)은 RT 55.0fps · CS 44.4fps이고 실행마다 ±4fps 정도 흔들리므로, **ViT를 켜도 처리 주기는 달라지지 않습니다**(ViT는 NPU_2 코어 사용률 약 3%).

**ViT-lite1 (`reid/vit_lite1_reid.dxnn`)**: 입력 uint8 NHWC [1,128,64,3](사람 crop을 128×64로 resize, gray→RGB 복제), 출력 `embedding` [1,128](호스트에서 L2 정규화), 1.46MB, NPU 101 layer, INT8 보정 128장(EMA). M1 NPU_2에서 crop당 NPU 3.0ms(준비 포함 5.9ms). 추적기는 잃어버린 track과 새 검출의 코사인 ≥ 0.6이면 예전 ID를 되살립니다. **현재 M1 INT8 실측에서는 재식별에 기여하지 못합니다**: 기준선 0.45~0.70 전부에서 ViT 없는 rebind(HOTA 53.95)보다 낮거나 같고, 그 차이는 전부 "외형 비교에 실패한 검출을 움직임 재연결에서 제외하는 hybrid veto" 때문입니다(veto 제거 시 ViT 없음과 동일하게 수렴하는 것으로 오프라인 재현됨, M1 실측 진행 중). 학습 상세(층 수·데이터 수)는 학습 PC 기록을 참조하십시오. 점수는 M1이 저장한 출력(PRED/TRACKS)을 GT로 채점한 값입니다. 실행 기록·채점 스크립트·demo 영상은 M1의 runtime `test/chatgpt/yolobytetrack` 작업 폴더 `results/m1_yolo26m_rt_cs_20261023/`에 있습니다(`results/`는 git 추적 제외).

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

YOLOX-Nano/S는 원본 실험과 같은 중앙640×640 캔버스로 calibration을 교정한 게시본입니다. Source checkpoint는 그대로이며 이전 calibration 버전은 Git 이력에 보존했습니다.

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
| yolov8s.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 4 |
| yolo26s.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 6 |
| yolox_nano.dxnn | 1 | output0 [1, 8400, 21] FLOAT | 9 |
| yolox_s.dxnn | 1 | output0 [1, 8400, 21] FLOAT | 9 |
| yolov8n.dxnn | 1 | output0 [1, 20, 8400] FLOAT | 4 |

DXNN header graph_info와 embedded CPU ONNX ValueInfo를 읽은 정적 확인입니다. 내부 NPU 경계 tensor 여러 개를 공개 출력 tensor 수와 혼동하지 않습니다.

</details>

<details>
<summary>체크포인트·해시·calibration·모델별 컴파일 상세</summary>

### 공통 컴파일 조건

기존 x86-64 Ubuntu WSL, DEEPX DX-COM2.4.1(package2.4.1-rc.1), DX-M1 SDK default(M1A), optimization0, CPU INT8 PTQ, EMA calibration100steps. torch2.14.1+cpu, ONNX1.23.2, ORT1.31.0. Source export: Windows torch2.7.0+cu118, Ultralytics8.4.157, ONNX1.18.0, ORT1.19.2; static batch1,640×640, opset17. YOLOX source commit6ddff4824372906469a7fae2dc3206c7aa4bbaee.

Calibration은 전체 FLIR TRAIN10742 목록에서 round(i*(N-1)/255), i=0..255로 뽑은256장입니다. TRAIN-list SHA-256 `d40cb99963d55de372d0b46cfcc11652078b51d5d1c17321c3040621ed63f1d1`; VAL/TEST는 사용하지 않았습니다.4196장 등 부분집합 학습 모델에는 calibration pool과 학습 subset이 다릅니다. Source hash, ONNX checker·finite CPU execution·3개 TRAIN raw parity, DX-COM 종료·DXNN hash를 확인했습니다. NPU 보드 정확도·FPS 검증과는 다릅니다.

### yolo26m.dxnn (삭제됨)

2026-11-01 삭제했습니다. 4,196장으로 학습한 기존 YOLO26m 교사(source PT SHA-256 `dca0f23d…7736`, DXNN SHA-256 `fbda1f1b…c03e`)이며, 파일과 상세 표는 git 기록 `1e957ad`에 남아 있습니다.

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
| DXNN bytes | 1536223 |
| DXNN SHA-256 | `9ab36237847c1e4a704af6677adf038d9f16fa03cb7b8cd3304c95b47c62fe46` |
| source PT SHA-256 | `64119cfcc291fcb436839cf29c11f3e089d7a3eee0580f5edce2049ebc35877f` |
| source ONNX SHA-256 | `3c181badb88bff9a8d9c7f2788a3648eced42f24407a97ff2e27bd0ea045d271` |
| calibration manifest SHA-256 | `381f90326244ee6007cd1b10aa9193c9a9545debf78101725fd8e2678f7facd9` |
| compile elapsed seconds | 125.533 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.0050048828125 / score 2.047419548034668e-05 |
| 공개 source output | [1, 8400, 21]; raw decoded xywh + objectness +16 class scores;8400 candidates, external objectness-times-class scoring and class-aware NMS |

### yolox_s.dxnn

| 항목 | 값 |
| --- | --- |
| 실험 식별 | LWIR-yolox-s-FT-10742-S101-E29 |
| 학습 방법 / seed / stage epochs | full_ft /101 /30 |
| TRAIN support | 10742 |
| 입력 representation | original; deployment export640×640 |
| 학습/완료 출처 | D:\LIG-Runs\flir_yolo26\yolo_family_screen_20261019_v1\full_s\yolox-s\latest_ckpt.pth |
| DXNN bytes | 9543889 |
| DXNN SHA-256 | `e1349efb61ed4f229807a31517688c35c9d45a7b2ee474ccebe8aaf8871168f7` |
| source PT SHA-256 | `2c0ba3087733411e7d89443d27d621f0a482dacd56531f7bcbe213921a644070` |
| source ONNX SHA-256 | `236d790348ad4b94a573326aa0ff8c3ff7739245f77293850d2633f08e0249c6` |
| calibration manifest SHA-256 | `381f90326244ee6007cd1b10aa9193c9a9545debf78101725fd8e2678f7facd9` |
| compile elapsed seconds | 161.499 |
| 병합 adapter 수 | 0 |
| 3-sample parity | allclose rtol1e-4 atol5e-3; max abs box 0.0062255859375 / score 1.0013580322265625e-05 |
| 공개 source output | [1, 8400, 21]; raw decoded xywh + objectness +16 class scores;8400 candidates, external objectness-times-class scoring and class-aware NMS |

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

### yolo26x.pt

DXNN이 아닌 PyTorch 체크포인트입니다. 위 공통 컴파일 조건은 적용되지 않습니다.

| 항목 | 값 |
| --- | --- |
| 실험 식별 | LWIR-YOLO26x-FT-10491-S202-E29 |
| 학습 방법 / seed / stage epochs | full_ft /202 /30 (best epoch 29) |
| TRAIN support | 10491 (`ladder_n10491_full_eligible.txt`, SHA-256 `a3fad7d3bf74…`; 검증 장면 연결 후보 영상 2개 251장 제외) |
| 입력 representation | original; 학습 정사각 640×640, 평가·추론 480×640 |
| 초기 가중치 | COCO 사전학습 yolo26x.pt (SHA-256 `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`) |
| PT bytes | 118376293 |
| PT SHA-256 | `e0e8a30cd6dfb7219b9501ae793389d4875fc7e4ab2904fd5e0cc1e49e4b4538` |
| person AP50:95 / AP50 | 53.2791 / 84.5572 (sourceVAL1144장, 사람 GT4470, 480×640, conf0.001, max_det300, maxDets100) |
| TMOT R@P95 (개발 진단) | 04-18 0.7225 · 04-20 0.6832 (사람 GT와 IoU0.5; 독립 검증 아님) |
| 정밀도 | 1~4 epoch FP32, 5~30 epoch AMP+TF32 (RTX 4090) |

**데이터 증강·학습 설정 변경 (이전 YOLO26x · 이전 YOLO26m 교사 대비)**

| 항목 | 이전 YOLO26x (4,196장, 삭제) | 이전 YOLO26m 교사 (4,196장, 삭제된 `yolo26m.dxnn`의 원본) | **yolo26x.pt** |
| --- | --- | --- | --- |
| TRAIN | 4196 | 4196 | **10491** |
| 학습 캔버스 | 고정 480×640 | 고정 480×640 | **정사각 640×640** (mosaic 사용 위해) |
| mosaic | 0 | 0 | **1.0, 마지막 10 epoch 끔** |
| scale | 0.5 | 0.5 | **0.75** (0.25~1.75배, 작은 사람 보강) |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 | 0.015 / 0.7 / 0.4 | **0 / 0** / 0.4 (흑백 복제 입력이라 색조·채도는 효과 없음) |
| translate / fliplr | 0.1 / 0.5 | 0.1 / 0.5 | 0.1 / 0.5 (같음) |
| mixup·회전·shear·perspective | 0 | 0 | 0 (같음) |
| LR 감소 | 선형 | 선형 | **cosine** |
| epochs / batch | 60(+추가) / 1 | 30 / 4 | 30 / 8 |
| warmup_bias_lr | **0.1** (AdamW에서 epoch1 mAP 0.009) | 0.0 | 0.0 |
| lr0 / lrf / warmup / nbs | 3e-4 / 0.01 / 3 / 64 | 같음 | 같음 |

이전 YOLO26x가 m보다 낮았던 주원인은 warmup_bias_lr 0.1과 batch 1로 보입니다(기록 근거 가설). 위 변경은 데이터·증강·캔버스·LR 감소를 함께 바꿨으므로 **어느 변경이 얼마나 기여했는지는 분리되지 않았습니다.** 같은 설정의 YOLO26m 재학습은 person AP50:95 50.7486으로 이전 m 교사(48.98, 같은 평가기)보다 높았습니다.

**전처리**: FLIR COCO JSON→YOLO txt 변환 시 같은 이미지의 완전 중복 상자 병합(TRAIN2·VAL8). FLIR 범주 16개를 위 클래스 표로 대응(bike→bicycle, motor→motorcycle, light→traffic light, hydrant→fire hydrant, sign→stop sign); COCO 헤드는 클래스 이름 기준으로 행을 다시 배치(COCO에 없는 deer·stroller·scooter·other vehicle 4개는 새로 학습). 입력은 8-bit grayscale을 3채널 복제, /255만 적용(평균·표준편차 정규화 없음, CLAHE·gradient 없음), 패딩114 중앙 letterbox. BatchNorm은 running mean/var를 COCO 값으로 고정하고 affine만 학습. 16-bit 원본은 카메라별 8-bit 변환이 필요하며 이 저장소는 제공하지 않습니다.

**사용**: `git lfs install` 후 clone, 파일 크기 118376293 bytes 확인. Ultralytics로 `YOLO('yolo26/yolo26x.pt').predict(img, imgsz=(480,640), rect=False, conf=0.001, max_det=300)`; `img`는 위 전처리를 거친 480×640×3 uint8. 상자는 패딩·크기 조정을 되돌려 원본 좌표로 복원합니다. TMOT에서 정밀도95%가 되는 임계값은 0.46~0.48이었고, 슈도 라벨은 사람 검수가 필요합니다. 학습·검증은 프로젝트 보관 Ultralytics 8.4.157 수정본을 사용했습니다.

### 보존 범위

로컬 원본과 중간 ONNX/캘리브레이션 자료는 보존하며, 저장소 계열 폴더에는 DXNN과 그 컴파일 메타데이터 JSON만 둡니다(예외: 슈도 라벨용 `yolo26/yolo26x.pt`와 학습 메타데이터 JSON). PNG/정밀 CSV/JSON은 원본 Library ZIP `libfile_473f586f45ac8191bef1087765316cc2`(SHA-256 `be592eed621b340e112782f81e340eed5c49f5a6f9690dce36c44ba3c47bbe74`)과 로컬에 보존했습니다. 이전 Git 이력의 파일은 삭제하지 않았으며 force push도 하지 않았습니다.

Ultralytics/model, DEEPX SDK 및 FLIR 이용 조건은 별도로 적용됩니다. [DEEPX compiler](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-COM/DX-COMPILER_Agent_Driven_Development/) · [Ultralytics DEEPX](https://docs.ultralytics.com/integrations/deepx/).

</details>
