# LWIR 모델 뱅크

열화상(LWIR) 카메라로 **사람을 찾고 따라가는** 팀 프로젝트의 저장소입니다. 서버 두 개와, 그 서버가 쓰는 모델 파일이 들어 있습니다. 장치는 DX-M1(Orange Pi 5 Plus) 한 대입니다.

| 서버 | 포트 | 화면에서 하는 일 |
|---|---|---|
| [추적 서버](#1-추적-서버-tracking_server) | 8090 | 실시간 검출·추적 보기, 녹화 재생, 정답과 비교, 지연·출력 간격 측정 |
| [라벨 서버](#2-라벨-서버-label_server) | 8770 | 녹화에서 사람 박스와 ID를 고치고 프레임별 검수 상태를 저장 |

**라벨 검수 → 추적 평가**

8770에서 팀이 전달한 주소를 열고 녹화를 고릅니다. 사람 박스와 ID를 고친 뒤 프레임을 **확인함**으로 표시하고 **저장**합니다. 8090의 **녹화 재생**에서 같은 녹화를 열면 **정답과 비교**에 오검출, 놓침, ID 변경, IDF1, MOTA가 나옵니다. [8770 편집 화면과 사용법](#2-라벨-서버-label_server) · [8090 재생 및 채점](tracking_server/README.md#9-웹-화면)

두 서버는 각각 켜고 끌 수 있습니다. 라벨 서버는 카메라와 NPU를 쓰지 않습니다.

> 이어서 일하는 사람이나 AI 작업자는 [HANDOFF.md](HANDOFF.md)(지금 상태, 다음 할 일, 규칙)를 먼저 읽으세요.

## 1. 추적 서버 (`tracking_server/`)

열화상 영상에서 사람을 찾고, 같은 사람에게 같은 번호를 유지하고, 웹 화면에서 고른 사람을 짐벌로 따라갑니다.

![8090 추적 서버 GUI의 영상, 재생 조작, 정답 비교, 설정 화면](tracking_server/images/interface_8090.png)

*8090 추적 서버의 PC 미리보기입니다. 영상과 채점 수치는 합성 예시입니다.* [화면 사용법과 인터페이스](tracking_server/README.md#웹-인터페이스-빠른-사용법)

<details>
<summary>추적 파이프라인과 단계별 처리 그림</summary>

![파이프라인과 실시간 처리](tracking_server/images/pipeline_realtime_20261111.png)

> 그림의 숫자는 2026-11-11 에 장치에서 짧게 읽은 값입니다. 검출을 NPU 코어 0·1 에 번갈아 넣는 방식은 장치 시험에서 기준을 넘지 못해 꺼 두었고, 지금은 코어 0 하나로 돕니다.

</details>

**하는 일**

1. 카메라에서 16비트 640×480 영상을 받습니다.
2. YOLO26m 으로 사람 상자를 찾습니다 (NPU 코어 0).
3. 온도 범위 밖, 화면 끝의 점수 낮은 상자를 버립니다.
4. 앞 장의 사람과 짝지어 번호를 유지합니다 (BoT-SORT, Kalman filter).
5. 외형을 비교해 돌아온 사람에게 옛 번호를 줍니다 (KeepID, NPU 코어 2).
6. 고른 사람의 머리를 화면 가운데로 오게 짐벌에 속도 명령을 보냅니다 (150 ms 예측 + PID).

### 처음 설치하고 켜기

Orange Pi에서 처음 내려받을 때 실행합니다. 이미 `~/model-bank`가 있다면 [기존 저장소 업데이트](#기존-저장소-업데이트)를 참고하세요.

```bash
git clone https://github.com/lwir-deepx-tracking-system/model-bank.git "$HOME/model-bank"
cd "$HOME/model-bank"
```

저장소로 들어간 뒤 [빌드와 서버 실행](#빌드와-서버-실행)을 진행합니다.
- 화면에서 할 수 있는 것: 사람을 눌러 따라가기, 손으로 짐벌 움직이기, 검출·외형 모델 바꾸기, Filter 값 바꾸기, 측정 저장, 카메라 초기화.
- 팀 PC GUI 에도 영상과 상자를 보냅니다.
- 꺼지면 스스로 다시 켜는 스크립트(`scripts/keepalive.sh`)가 있습니다.

**자세한 안내:** [tracking_server/README.md](tracking_server/README.md) — 설정, 거르는 곳, 짐벌이 멈추는 경우, 장치에서 잰 속도, 끄는 법

### 기존 저장소 업데이트

`destination path 'model-bank' already exists and is not an empty directory`가 나오면 `~/model-bank` 폴더가 이미 있습니다. 먼저 사용하려는 저장소 폴더로 들어가 주소와 브랜치, 수정 파일을 확인합니다.

```bash
cd "$HOME/model-bank"
git rev-parse --show-toplevel
git remote get-url origin
git status --short --branch
```

`~/model-bank` 안에서 다시 클론했다면 안쪽 저장소인 `~/model-bank/model-bank`로 들어가 위 Git 확인 명령을 실행합니다. 아래부터는 선택한 저장소의 최상위 폴더에서 실행합니다.

`origin`이 `lwir-deepx-tracking-system/model-bank`를 가리키고, 현재 브랜치가 `main`이며 수정 파일이 없으면 아래를 실행합니다.

```bash
git pull --ff-only origin main
git log -3 --oneline
```

다른 브랜치에 있거나 수정 파일이 표시되면 현재 작업을 먼저 확인합니다. 기존 폴더가 Git 저장소가 아니라면 다른 이름으로 내려받을 수 있습니다.

```bash
git clone https://github.com/lwir-deepx-tracking-system/model-bank.git "$HOME/model-bank-new"
```

업데이트 후에는 같은 폴더에서 [빌드와 서버 실행](#빌드와-서버-실행)을 진행합니다. 실행 중인 추적 서버의 Python 변경은 서버를 다시 시작할 때 적용됩니다. [추적 서버 시작과 종료](tracking_server/README.md#8-꺼지면-다시-켜기-끄기)

### 빌드와 서버 실행

1. 처음 설치하거나 바이너리·모델 파일이 바뀌었다면, 저장소 최상위 폴더에서 준비 스크립트를 실행합니다.

   ```bash
   bash tracking_server/scripts/setup.sh
   ```

   `cam_pipe built` 뒤에는 브리지 컴파일이 진행되는 동안 출력이 뜸할 수 있습니다. 실시간 카메라와 기본 PC GUI 구성은 `cam_pipe built`, `gimbal_bridge_v25 built`, 마지막 `setup done`을 확인합니다.

2. 기존 추적 서버가 꺼져 있을 때 같은 폴더에서 실행합니다.

   ```bash
   bash tracking_server/scripts/run_real_gui.sh
   ```

3. 브라우저에서 `http://<장치 주소>:8090/`을 엽니다. 명령 직후 셸 프롬프트가 돌아왔다면 그 실행은 종료된 상태입니다. 같은 저장소에서 최근 로그와 8090 포트를 확인합니다. `LWIR_LIVE_PORT`를 바꿨다면 해당 포트를 확인합니다.

   ```bash
   tail -n 60 tracking_server/real_gui_1.out
   ss -ltn 'sport = :8090'
   ```

   `LISTEN`이 보이면 브라우저로 추적 화면을 확인합니다. 로그에 `NPU busy`가 있으면 기존 서버 프로세스부터 확인합니다. 로그에 `gimbal busy`가 있으면 [짐벌 포트 점유 확인과 서버 재시작](tracking_server/README.md#gimbal-busy-해결)을 참고하세요. 다른 종료 이유는 [추적 서버 빠른 시작](tracking_server/README.md#2-빠른-시작)에 정리했습니다.

영상 테두리에서 짐벌 위치까지 보려면 [v27 빌드와 8090 실행](tracking_server/README.md#v27로-실행하고-짐벌-위치-표시-확인)을 참고하세요.

## 2. 라벨 서버 (`label_server/`)

열화상 녹화에 자동으로 붙인 사람 상자를, 사람이 브라우저에서 고치는 사이트입니다. 팀원은 받은 주소 하나만 열면 됩니다.

![8770 라벨 편집 화면과 사람 박스 및 ID 수정 도구](label_server/images/interface_8770.png)

*8770 편집 화면의 PC 미리보기. 영상, 박스, ID는 합성 예시입니다.*

**하는 일**

- 첫 화면은 녹화(시퀀스) 목록입니다. 프레임 수, 확인한 프레임, 검수 상태, 지금 열어 둔 사람이 보입니다.
- 녹화를 고르면 편집 화면이 열립니다. 상자를 지우고, 옮기고, 새로 그리고, 번호를 바꾸고, 쓸모없는 프레임을 삭제 표시합니다.
- 저장하면 그 녹화 폴더에 `tracks_edited_mot.txt`(고친 라벨)와 `review_state.json`(검수 상태)이 생깁니다. 원래 라벨은 바꾸지 않고, 덮어쓰기 전 파일은 매번 백업합니다.
- 한 녹화는 한 사람만 작업합니다. 다른 사람이 먼저 저장했으면 덮어쓸지 묻습니다.
- 실시간 화면의 측정 버튼으로 찍은 녹화를 30초마다 스스로 목록에 넣습니다.
- 녹화가 많을 때 어느 장부터 라벨할지 고르는 도구(`scripts/pick_frames.py`)가 있습니다.

**켜기**

```bash
cd model-bank/label_server
python3 start_label_server.py --root <라벨 폴더> --bind <주소> --port 8770
```

- 켜지면 `address: http://<주소>:8770/?k=<접속 키>` 줄이 나옵니다. 이 주소를 통째로 팀원에게 줍니다.
- 파이썬 3.10 이상의 표준 라이브러리만으로 돕니다.
- 접속 키 하나로만 막는 http 사이트입니다. 팀만 닿는 주소(사설 VPN 등)에 붙입니다.
- 실제 녹화와 라벨은 이 저장소에 없습니다.

**자세한 안내:** [label_server/README.md](label_server/README.md) — 라벨 폴더 모양, 계속 켜 두기, 접속 키, 서버 없이 쓰기

## 3. 모델

추적 서버가 쓰는 모델은 `tracking_server/models/` 에 있습니다. 모두 DX-M1 NPU 용 `.dxnn`(INT8)입니다.

| 일 | 시작 모델 | 고를 수 있는 수 |
|---|---|---|
| 검출 | YOLO26m_(FLIR) 640x480 | 9개 |
| 외형 비교 | RepVGG_distillation_(TMOT,CAMEL)_20261103 | 5개 |

- 모델 목록, 파일 이름, 입력·출력, 학습 자료, 측정값: [tracking_server/models/README.md](tracking_server/models/README.md)
- 실내·실외 fine-tuning 모델과 실내 외형 모델은 시험용입니다. 장치에서 정확도를 재지 않았습니다.
- `detect/yolo26x/yolo26x.pt` 는 Git LFS 파일입니다. 필요하면 `git lfs install` 뒤에 clone 하세요.

## 4. 무엇이 어디에 있나

폴더 이름은 밖에서 안으로 `서버 / 종류 / 일` 순서입니다. 일 이름은 어디서든 같습니다: `detect`(검출), `track`(추적), `reid`(외형 비교).

**추적 서버**
- 켜는 파일: [`tracking_server/start_tracking_server.py`](tracking_server/start_tracking_server.py)
- 코드 (일별 폴더): [`tracking_server/src/`](tracking_server/src/)
- 카메라 프로그램: [`tracking_server/src/camera/`](tracking_server/src/camera/)
- 짐벌 제어 프로그램, 팀 코드 사본, 제어값 기록: [`tracking_server/src/gimbal/`](tracking_server/src/gimbal/)
- 8090 웹 화면: [`tracking_server/page/`](tracking_server/page/)
- 준비·켜기·다시 켜기 스크립트: [`tracking_server/scripts/`](tracking_server/scripts/)

**모델**
- 검출 모델 (YOLO26, YOLOv8, YOLOX): [`tracking_server/models/detect/`](tracking_server/models/detect/)
- 외형 비교(re-ID) 모델: [`tracking_server/models/reid/`](tracking_server/models/reid/)
- 추적용 작은 파일 (확인 판정기, 번호 잇기 판정기, 위치 기억): [`tracking_server/models/track/`](tracking_server/models/track/)
- 모델 파일 확인값(sha256): [`tracking_server/models/MANIFEST.json`](tracking_server/models/MANIFEST.json)

**라벨 서버**
- 켜는 파일: [`label_server/start_label_server.py`](label_server/start_label_server.py)
- 서버 부품: [`label_server/src/`](label_server/src/)
- 편집 화면: [`label_server/page/`](label_server/page/)
- 계속 켜 두기, 장 고르기 도구: [`label_server/scripts/`](label_server/scripts/)

## 5. 옛 경로로 찾아온 분께 (2026-11-11 에 옮김)

새 자리는 모두 `tracking_server/` 아래입니다.

<details>
<summary>옛 자리 → 새 자리</summary>

**모델**
- `yolo26/yolo26m_flir_nms_*.dxnn` → `models/detect/yolo26m/`
- `yolo26/yolo26m_flir_inside_nms_640x480.dxnn` → `models/detect/yolo26m/yolo26m_flir_inside_20261110_2020_640x480.dxnn` (같은 파일, 이름에 날짜·시각)
- `yolo26/yolo26s_*.dxnn` → `models/detect/yolo26s/`
- `yolo26/yolo26x.pt`, `yolo26x_train_metadata.json` → `models/detect/yolo26x/`
- `yolo26/*_build/` → `models/detect/build_scripts/`
- `yolov8/`, `yolox/` → `models/detect/yolov8/`, `models/detect/yolox/`
- `final_tracker/dxnn/repvgg_tmot_camel_reid.dxnn` → `models/reid/`
- `final_tracker/dxnn/yolo26m_flir_whole480_nms_640x480.dxnn` → 없앰. 같은 파일이 `models/detect/yolo26m/yolo26m_flir_nms_640x480.dxnn`
- `final_tracker/dxnn/yolo26m_flir_rt480_*_nms_480x384.dxnn` (타일 2장) → 없앰
- `final_tracker/compile_metadata/` → 모델 파일 옆
- `final_tracker/tracker/gate_gru32_frozen.json`, `relink_judge_frozen.json` → `models/track/`
- `final_tracker/tracker/models/MANIFEST.json` → `models/MANIFEST.json`

**코드와 스크립트**
- `final_tracker/tracker/lwir_tracker_m1.py` → `start_tracking_server.py` + `src/`
- `final_tracker/tracker/native_post.py`, `post.cpp` → `src/detect/`
- `final_tracker/tracker/relabel_online.py`, `relink_online_judge.py` → `src/reid/`
- `final_tracker/tracker/cam_pipe.cpp` → `src/camera/`
- `final_tracker/tracker/live_page.html` → `page/`
- `final_tracker/tracker/setup.sh`, `run/` → `scripts/`
- `final_tracker/gimbal/` → `src/gimbal/`

**글과 그림**
- `final_tracker/reid_gru_io.md` → `models/reid/README.md`, `models/track/README.md`
- `final_tracker/README.md` → `README.md`
- `tensor_requirements.md` → 없앰. 모델별 입력·출력은 `models/README.md`
- 옛 그림 (`final_tracker/images/` 의 4장) → 없앰. 지금 그림은 `images/pipeline_realtime_20261111.png` 한 장

</details>

없앤 파일과 옛 README 의 긴 기록(지난 측정표, 지운 모델의 기록, 옛 그림)은 **git 기록에 남아 있습니다.** 옛 커밋을 checkout 하면 그대로 받을 수 있습니다.

## 6. 주의

- 이 저장소는 **공개**입니다. 키, 장치 주소, 카메라 영상, 라벨 자료는 올리지 않습니다.
- 추적 서버 화면에는 비밀번호가 없고 짐벌을 움직입니다. 믿을 수 있는 기기만 닿는 주소에서 쓰세요.
- NPU 측정 프로그램을 돌리기 전에는 추적 서버를 먼저 꺼 주세요 ([끄는 법](tracking_server/README.md#8-꺼지면-다시-켜기-끄기)).
