# LWIR YOLO26x — FLIR 전체 학습 데이터, 30 epoch

`yolo26x_flir_full30.pt`는 FLIR 열영상 전체 학습 데이터로 COCO 사전학습 YOLO26x를 Full Fine-Tuning한 검출기다. 이전 E69 모델(2026-11-01 삭제, git 기록에는 남음)을 대체하는 **슈도 라벨 후보 생성용 기본 모델**이다. 사람 검수가 필요한 후보를 만드는 용도이며 추적기, 확정 GT, ONNX 또는 DXNN은 포함하지 않는다. DX-M1에서 바로 실행할 수 있는 것으로 간주하면 안 된다.

## 데이터와 전처리

**라벨 변환 (FLIR COCO JSON → YOLO txt)**
- 원본: FLIR ADAS v2 `images_thermal_train` 10,742장(상자 175,038개), `images_thermal_val` 1,144장(상자 16,688개). 열영상만 쓰고 RGB 영상은 쓰지 않는다.
- 같은 이미지 안의 완전히 같은 중복 상자는 하나로 합쳤다(train 2개, val 8개).
- FLIR 범주 16개를 YOLO 클래스 0~15로 옮겼다. FLIR 이름을 COCO 이름에 맞췄다(bike→bicycle, motor→motorcycle, light→traffic light, hydrant→fire hydrant, sign→stop sign). deer, stroller, scooter, other vehicle은 COCO에 없는 클래스다.
- COCO 사전학습 헤드는 클래스 **이름 기준으로 행을 다시 배치**해 시작했다(COCO에 있는 12개 클래스는 해당 COCO 출력에서 시작, 없는 4개는 새로 학습).

**학습 이미지 선택**
- 검증 영상과 같은 장면일 가능성이 있는 학습 영상 2개(`2SReBn5LtAkL5HMj2` 55장, `jNQtRj6NGycZDEXpe` 196장, 총 251장)를 뺐다. 이 제외는 알려진 후보만 걸러낸 것이며, 나머지 학습 영상이 검증 장면과 완전히 독립임을 보장하지 않는다.
- 남은 **10,491장 전체**를 썼다(131개 영상). 프레임 간격 제한·클래스 균형 조정은 하지 않았다. E69는 이 중 프레임 간격 30 이상으로 뽑은 4,196장 부분집합이었다.
- 검증 1,144장은 학습에 쓰지 않았다.

**입력 전처리 (학습·추론 공통)**
- FLIR 열영상 JPEG는 8-bit 흑백(640×512)이다. 흑백 한 장을 **세 채널에 그대로 복제**한다. CLAHE, 경사(gradient) 같은 추가 변환은 하지 않는다(코드가 세 채널이 같은지 검사한다).
- 값은 0~255를 0~1로 나누기만 한다. 평균·표준편차 정규화는 없다.
- 학습: 비율을 유지해 **640×640**에 맞추고 남는 곳을 값 114로 채운다(모자이크를 쓰기 위한 정사각 경로).
- 검증·추론: 비율을 유지해 **480×640**에 맞추고 값 114로 가운데 패딩한다. `predict_person.py`가 이 처리를 한다. 상자는 패딩과 크기 조정을 되돌려 원본 좌표로 돌려준다.
- 16-bit 원시 열영상은 FLIR와 같은 8-bit 변환을 거친 뒤 넣어야 한다. 카메라마다 변환이 달라 이 저장소는 그 변환을 제공하지 않는다.

**BatchNorm**
- 모든 BatchNorm 층을 평가 모드로 두어 **running mean/var(이동 통계)는 COCO 값으로 고정**했다. BN의 scale·shift 가중치는 학습했다.

## 학습

- 초기 모델: COCO 사전학습 YOLO26x(SHA-256 `9fdd44a3…7e92`). 16개 클래스, 사람은 class 0.
- 증강: 모자이크 1.0(마지막 10 epoch 끔), 크기 변화 scale 0.75(0.25~1.75배, 작은 사람 보강), 좌우 반전 0.5, 이동 0.1, 밝기 0.4. 색조·채도(흑백 영상이라 효과 없음)·mixup·회전은 끔.
- 최적화: AdamW lr0 3e-4, 코사인으로 30 epoch 동안 0.01배까지 감소, warm-up 3 epoch, **warmup_bias_lr 0**, batch 8 / nbs 64, BN 통계 고정, seed 202. 1~4 epoch FP32, 5 epoch부터 AMP+TF32(RTX 4090).
- E69가 m보다 낮았던 주원인으로 보이는 설정 차이(warmup_bias_lr 0.1, batch 1)를 m 학습 설정에 맞췄다. 이것은 기록으로 확인한 가설이며, 같은 설정으로 다시 학습하자 X가 m보다 높아졌다.
- 최고 체크포인트: **29 epoch**. 30 epoch 이후 학습률 재시작으로 10 epoch를 더 학습했지만 개선이 없어 채택하지 않았다.

## 평가

| 모델 | FLIR 사람 AP50:95 | AP50 | TMOT 04-18 R@P95 | TMOT 04-20 R@P95 |
| --- | --- | --- | --- | --- |
| **이 모델 (FULL30)** | **53.279%** | **84.557%** | **0.723** | **0.683** |
| YOLO26x E69 | 45.364% | 78.097% | 0.557 | 0.638 |
| YOLO26m teacher epoch29 | 48.977% | — | 0.686 | 0.653 |
| YOLO26m 같은 설정 전체 학습 30 epoch | 50.749% | 82.396% | 0.708 | 0.677 |

- FLIR: source validation **1,144장 / 사람 GT 4,470개**. 8-bit 흑백, 비율 유지 480×640 크기 조정 + 값 114 패딩, 세 채널 복제. `conf=0.001`, `max_det=300`, COCO `maxDets=100`. 사람 클래스 점수이며 16개 클래스 평균이 아니다.
- TMOT: 04-18 15개, 04-20 16개 시퀀스, 사람 GT와 IoU 0.5 일대일 매칭. R@P95는 정밀도 95% 이상을 유지하는 점수 임계값 중 최대 재현율. 정밀도 95%에서 누락 상자는 04-18 6,016개 / 04-20 9,500개로, m teacher(6,818 / 10,408)보다 12% / 9% 적다.
- 두 평가 모두 모델 선택·조정에 쓴 **개발 자료**다. TMOT는 독립 검증이 아니다. 실제 촬영 영상에서의 정답 품질, M1 정확도·속도는 검증되지 않았다. 작은 사람, 가림, 오탐·누락은 사람이 확인해야 한다.

## 사용법

Git LFS를 설치한 뒤 저장소를 받는다. 실제 파일 크기가 **118,376,293 bytes**인지 확인한다.

```bash
git lfs install
git clone https://github.com/lwir-deepx-tracking-system/model-bank.git
cd model-bank/models/LWIR-YOLO26x-FT-FLIR-FULL30
python -m pip install ultralytics opencv-python pillow numpy
python predict_person.py thermal_8bit.png --conf 0.25 --device cpu
# CUDA 환경: --device 0
```

결과는 JSON이며 `xywh`는 **원본 이미지 기준 좌상단 x, y와 너비, 높이**다. `--conf 0.25`는 예시이며 최적 임계값으로 검증된 값이 아니다(TMOT에서 정밀도 95%가 되는 임계값은 0.46~0.48이었다). 16-bit 입력은 카메라별 변환을 확인한 후 8-bit로 준비한다.

학습·검증에는 프로젝트에서 보관한 Ultralytics **8.4.157 기반 수정 코드**를 사용했다. 일반 배포판의 버전·결과 동등성은 보장하지 않는다.

## 파일 무결성

- SHA-256: `e0e8a30cd6dfb7219b9501ae793389d4875fc7e4ab2904fd5e0cc1e49e4b4538`
- 클래스: `person, bicycle, car, motorcycle, bus, train, truck, traffic light, fire hydrant, stop sign, dog, deer, skateboard, stroller, scooter, other vehicle`.
