# LWIR YOLO26x — FLIR fine-tuned, E69

`yolo26x_flir_ft_e69.pt`는 FLIR 열영상으로 Full Fine-Tuning한 YOLO26x 검출기다. 사람 검출 후보와 **사람 검수가 필요한 슈도 라벨**을 만드는 용도다. 추적기, 확정 GT, ONNX 또는 DXNN은 포함하지 않는다. 이 파일을 DX-M1에서 바로 실행할 수 있는 것으로 간주하면 안 된다.

## 학습과 평가

- 초기 모델: COCO 사전학습 YOLO26x. 16개 클래스로 FLIR ADAS v2 열영상 train 4,196장 학습. 사람은 class 0.
- 검증 입력: 8-bit 흑백 영상, 가로세로 비율을 유지해 480×640으로 크기 조정 및 값 114로 패딩, 같은 이미지를 세 채널에 복제. YOLO 입력을 0~1로 정규화.
- 기준 학습 60 epoch 후 가중치 기반 별도 추가 학습을 진행했다. 추가 학습은 AdamW, batch 1, seed 202, LR 0.000008. 추가 학습 도중 중단되어 2번째 추가 epoch의 optimizer·EMA 상태에서 이어갔으며 난수 스트림을 완전히 복원하지 못했다. 총 74 epoch까지 완료했고, 최고 체크포인트는 **총 69 epoch**(기존 60 + 추가 9)다.
- 사람 COCO AP: FLIR source validation **1,144장 / 사람 GT 4,470개**에서 AP50:95 **45.364%**, AP50 **78.097%**. 기존 60 epoch 모델은 각각 45.106%, 77.665%였다. 후보 생성 시 `conf=0.001`, `max_det=300`; 채점은 COCO `maxDets=100`. 이것은 사람 클래스 점수이며 전체 16개 클래스 평균이 아니다.
- 이 검증 세트는 모델 선택에 사용된 개발 자료다. 독립 영상에서의 일반화, 실제 촬영 영상의 정답 생성 품질, M1 정확도와 속도는 검증되지 않았다. 작은 사람, 가림, 오탐 및 누락을 사람이 확인해야 한다.

## 사용법

Git LFS를 설치한 뒤 저장소를 받는다. Git LFS 포인터가 아니라 실제 파일 크기가 **118,361,445 bytes**인지 확인한다.

```bash
git lfs install
git clone https://github.com/lwir-deepx-tracking-system/model-bank.git
cd model-bank/models/LWIR-YOLO26x-FT-FLIR-E69
python -m pip install ultralytics opencv-python pillow numpy
python predict_person.py thermal_8bit.png --conf 0.25 --device cpu
# CUDA 환경: --device 0
```

결과는 JSON이며 `xywh`는 **원본 이미지 기준 좌상단 x, y와 너비, 높이**다. `--conf 0.25`는 사용 예시이지, 최적의 오탐·누락 절충으로 검증된 값이 아니다. AP 계산 시 사용한 0.001은 평가 후보 하한으로, 화면 표시나 최종 라벨 채택 임계값과 구분한다. 16-bit 입력은 카메라별 변환을 확인한 후 8-bit로 준비한다.

학습·검증에는 프로젝트에서 보관한 Ultralytics **8.4.157 기반 수정 코드**와 PyTorch 2.7.0+cu118을 사용했다. 위 설치 명령은 사용을 시작하는 예이며, 일반 배포판의 버전·결과 동등성은 보장하지 않는다. 클래스 전체를 사용하려면 모델의 `names`를 확인한다.

## 파일 무결성

- SHA-256: `10f9462d845672e73732cfd66fe444a7f785f5db2f87160ec2ec5fbc29f4c94a`
- 클래스: `person, bicycle, car, motorcycle, bus, train, truck, traffic light, fire hydrant, stop sign, dog, deer, skateboard, stroller, scooter, other vehicle`.
