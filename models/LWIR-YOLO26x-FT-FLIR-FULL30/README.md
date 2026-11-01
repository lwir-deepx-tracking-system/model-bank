# LWIR YOLO26x — FLIR 전체 학습 데이터, 30 epoch

`yolo26x_flir_full30.pt`는 FLIR 열영상 전체 학습 데이터로 COCO 사전학습 YOLO26x를 Full Fine-Tuning한 검출기다. [E69 모델](../LWIR-YOLO26x-FT-FLIR-E69/)을 대체하는 **슈도 라벨 후보 생성용 기본 모델**이다. 사람 검수가 필요한 후보를 만드는 용도이며 추적기, 확정 GT, ONNX 또는 DXNN은 포함하지 않는다. DX-M1에서 바로 실행할 수 있는 것으로 간주하면 안 된다.

## 학습

- 초기 모델: COCO 사전학습 YOLO26x(SHA-256 `9fdd44a3…7e92`). 16개 클래스, 사람은 class 0.
- 학습 데이터: FLIR ADAS v2 열영상 train 중 적격 이미지 **10,491장 전체**(영상 2개, 251장 제외). E69는 4,196장 부분집합이었다.
- 입력: 정사각 640 letterbox 학습 경로(모자이크를 쓰기 위해). 검증·선택은 아래 480×640 입력으로 한다.
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
