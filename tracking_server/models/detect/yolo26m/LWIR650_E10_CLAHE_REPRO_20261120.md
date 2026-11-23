# YOLO26m LWIR650 상체 E10 재현

GUI 모델 키는 `yolo26m_pose_lwir650_20261119_bodybox_kpt_640x480`입니다. 10에폭의 `last.pt`에서 내보낸 상체 박스와 관절 17개 모델입니다. 이전 11/17 검수264 E10과는 별도 학습입니다.

## 팀 보관본

- 아카이브 `LWIR650_E10_CLAHE_REPRO_20261120.tar`
- SHA256 `cc47cd3c711e76cc1f0a509bdeba765fd91956b7dee4065e9d91ec507df8d69f`
- 크기 907,253,760바이트, 멤버 8,128개
- 내부 `metadata/SHA256_INDEX.json`의 SHA256 `954ee1e8a8cd71e6342e2ae118058fcdedff1e33689d46c080c6494e75060f4c`

팀 장치의 접근 제한 Model Bank 보관 위치에서 아카이브 이름으로 찾습니다. `sha256sum LWIR650_E10_CLAHE_REPRO_20261120.tar` 결과를 위 값과 비교한 뒤 빈 폴더에 압축을 풉니다. 내부 인덱스에는 각 파일의 상대 경로, 크기와 SHA256이 있습니다. 장치에서 모든 멤버의 크기와 해시를 검증했습니다.

| 분할 | 장수 | 구성 |
| --- | ---: | --- |
| 학습 | 650 | 검수 세트 상체 211장과 4명 영상 439장의 이미지·라벨·무시 영역 |
| 당시 검증 | 934 | OTP2 이미지·라벨 |
| 별도 시험 | 2,151 | OTP2 이미지·라벨과 목록. E10 당시 평가 기록 없음 |

학습·검증·시험 사이에서 파일 바이트와 디코딩한 픽셀이 같은 이미지는 0장이었습니다. 이 검사는 유사 프레임이나 같은 촬영 장면까지 판정하지 않습니다. 검증·시험 목록의 파일별 해시는 재현 자료를 만들 때 계산했습니다.

## 코드, 학습과 평가

`metadata/LWIR650_E10_CODE_AND_MANIFESTS.zip`에 실제 `launch_training.py`, `train_pose_upper.py`, `pose_partial.py`, 열화상 증강, 라벨·무시 영역·미검수 관절 처리, 원래 CLI와 최종 인자가 있습니다. `TRAIN_PROTOCOL.json`과 `RUN_MANIFEST.json`에 당시 가중치·코드·데이터의 해시가 있으며 `TRAIN.log`에 Python 3.11.18, Ultralytics 8.4.184, PyTorch 2.7.1+cu118을 기록했습니다.

시작 가중치는 `weights/C_control_v1_epoch2.pt`이며 SHA256은 `78525f5e987b5a5a429f19dbfd30e8ce610047148794488dc77ecbc7fb6d4ebb`입니다. `weights/E10_last.pt`의 SHA256은 `907af7c22a74c0a5bd65d0f70089233eb819fde1102428031301128e097d8a9c`입니다.

당시 PC에서는 `REPRODUCE_CONTROL.ps1`로 원본 해시를 점검하고 학습 인자를 출력할 수 있습니다. 다른 컴퓨터에서는 `pose.yaml`과 목록의 이미지 경로, `--data`, `--weights`, `--project`, `--name`, `--must-keep` 경로를 압축을 푼 위치에 맞춥니다. 원본 학습 지원 파일을 `PYTHONPATH`에 넣습니다. 학습은 seed 0, 배치 4, 10에폭, AdamW, 초기 학습률 `2e-5`, AMP 끔으로 실행했습니다.

학습 직후 OTP2 검증셋의 mAP50–95는 박스 46.28%, 관절 78.35%였습니다. 당시 평가 명령과 결과는 `historical_code/evaluation/`에 있습니다. 기존 학습에는 CLAHE가 없습니다. 비교할 때는 같은 시작 가중치, 데이터, 학습 설정과 검증 절차로 한쪽 학습 로더에만 CLAHE를 추가합니다. CLAHE 대조 학습과 별도 시험 2,151장의 평가는 아직 실행하지 않았습니다.
