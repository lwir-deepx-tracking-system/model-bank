# 최종 추적 시스템 — 열화상 사람 추적 (DX-M1 INT8)

640×480 열화상 영상에서 **사람을 찾고, 같은 사람에게 같은 번호를 유지**합니다. 쓰는 방법은 두 가지입니다.

| 쓰는 방법 | 무엇을 하나 | 어디를 보나 |
| --- | --- | --- |
| **실시간 서버** | 카메라 영상을 추적해 웹 화면으로 보여 주고, 짐벌과 PC GUI를 연결 | [5절](#5-실시간-서버-카메라--웹-화면) |
| **파일 실행** | 저장된 프레임 폴더를 추적해 결과 파일로 남김 | [6절](#6-파일-실행) |

처음이면 3절(준비물) → 4절(설치) → 5절 또는 6절 순서로 보면 됩니다.

## 1. 폴더 구조

```
final_tracker/
├── README.md                이 문서
├── reid_gru_io.md           외형 비교(re-ID) 모델과 GRU 확인 게이트의 입력·출력
├── dxnn/                    DXNN 모델 4개 (2절)
├── compile_metadata/        모델 컴파일 기록. 참고용, 실행에는 필요 없음
├── images/                  이 문서의 그림
├── tracker/                 실행 코드 (여기서 실행)
│   ├── setup.sh             설치: 빌드, 모델 연결, 파일 확인 (4절)
│   ├── lwir_tracker_m1.py   실행 파일. 파일 실행과 실시간 서버 둘 다 이 파일
│   ├── native_post.py, post.cpp      박스 디코드·NMS (setup.sh가 post.so로 빌드)
│   ├── relabel_online.py, relink_online_judge.py   번호 되돌리기 코드
│   ├── gate_gru32_frozen.json        학습된 GRU 확인 게이트
│   ├── relink_judge_frozen.json      학습된 재연결 판정기
│   ├── cam_pipe.cpp         카메라 원본을 읽어 추적기로 넘김. 카메라가 멈추면 스스로 다시 엶
│   ├── live_page.html       웹 화면
│   ├── run/                 서버 스크립트
│   │   ├── run_real_gui.sh      서버 켜기
│   │   ├── keepalive.sh         꺼지면 다시 켜기 (cron)
│   │   ├── live.env.example     장치별 설정 예
│   │   └── gui_model_client.py  PC GUI 명령 시험
│   ├── models/              setup.sh가 모델을 연결하는 곳, MANIFEST.json(sha256)
│   ├── test300/             300프레임 시험 기록
│   └── CHANGES_FROM_RESEARCH_SCRIPT.md   연구용 스크립트와의 차이
└── gimbal/                  짐벌 제어와 PC GUI 연결
    ├── gimbal_bridge_v25.cpp    연결 프로그램 (setup.sh가 빌드)
    ├── team/                    팀 런타임 소스의 고치지 않은 사본 (출처: team/SOURCE.txt)
    ├── test_*.py                시험 스크립트
    └── PID_TUNING_RECORD_20261110.md   짐벌 제어값 조정 기록
```

- 파일 실행은 이 폴더 안의 파일만 있으면 됩니다.
- 실시간 서버의 모델 바꾸기 목록만 모델뱅크 `../yolo26/` 폴더의 DXNN 세 개를 더 씁니다.
- `team/SOURCE.txt`는 연결 프로그램을 `gimbal_bridge.cpp`라고 부르지만 여기서는 `gimbal_bridge_v25.cpp`입니다.

## 2. 모델과 처리 흐름

| 파일 | 하는 일 | 쓰는 곳 | 도는 곳 |
| --- | --- | --- | --- |
| `dxnn/yolo26m_flir_rt480_a_nms_480x384.dxnn` | YOLO26m_(FLIR) 사람 검출, 화면 왼쪽 타일 (열 0~384) | 파일 실행 | NPU 코어 0 |
| `dxnn/yolo26m_flir_rt480_b_nms_480x384.dxnn` | YOLO26m_(FLIR) 사람 검출, 화면 오른쪽 타일 (열 256~640) | 파일 실행 | NPU 코어 1 |
| `dxnn/yolo26m_flir_whole480_nms_640x480.dxnn` | YOLO26m_(FLIR) 사람 검출, 자르지 않은 640×480 한 장 | 실시간 서버 | NPU 코어 0 |
| `dxnn/repvgg_tmot_camel_reid.dxnn` | RepVGG-A0_(TMOT,CAMEL) 사람 외형 비교. 끊겼던 사람에게 예전 번호를 돌려줄 때 씀 | 둘 다 | NPU 코어 2 |
| `tracker/gate_gru32_frozen.json` | GRU 확인 게이트. 새 번호를 줄지 정함 | 둘 다 | CPU |
| `tracker/relink_judge_frozen.json` | 재연결 판정기. 예전 번호로 되돌릴지 정함 | 둘 다 | CPU |

- DXNN은 모두 **INT8**입니다(입력 uint8, 출력 float32).
- 타일 A와 B는 같은 가중치를 타일 위치별로 두 번 컴파일한 것이라 **둘 다** 있어야 합니다.
- 검출기 출력은 `tensor_requirements.md`의 YOLO26 분할 출력 6개(박스 3개 + 클래스 3개, C = 16)입니다. 실행 코드가 알아서 합칩니다(`--no-split6`은 예전 두 출력 DXNN용).
- 외형 비교 모델과 게이트의 입력·출력 형식은 [`reid_gru_io.md`](reid_gru_io.md)에 있습니다.

![파이프라인: 검출기 2개와 외형 모델이 NPU 코어 세 개에서 돌고, 추적·게이트·판정기는 CPU에서 도는 흐름](images/pipeline_m1.png)

모델별 입력과 출력, 장치, 시간:

![최종 추적 시스템 구조: 모델별 입력과 출력, 장치, 시간](images/system_structure.png)

한 프레임의 처리 순서와 각 단계를 맡은 함수 이름·줄 번호(`tracker/lwir_tracker_m1.py` 기준):

![코드 구조와 단계별 실제 이미지](images/code_structure.png)

## 3. 준비물

**공통**

- DX-M1(Orange Pi 5 Plus), DXRT 3.4.2. 확인한 환경: Ubuntu 22.04, Python 3.10, g++ 11.4.
- Python 패키지: numpy, opencv(cv2), scipy, `dx_engine`.
- **NPU가 비어 있어야 합니다.** 다른 NPU 작업이 돌고 있으면 시작 때 멈춥니다.

**실시간 서버에만 필요**

- i3system 열화상 카메라(640×480, 16비트, USB)와 카메라 SDK(`/usr/local/include/i3system`, `libi3system_*`). 없으면 파일 실행만 됩니다.
- 짐벌(시리얼, 기본 `/dev/ttyUSB0`).
- OpenCV(`opencv4`)와 GStreamer(`gstreamer-1.0`, `gstreamer-app-1.0`)가 `pkg-config`로 잡혀야 합니다. PC GUI로 보내는 영상은 하드웨어 인코더 `mpph264enc`를 씁니다.
- 웹 페이지 주소. 기본은 장치의 Tailscale 주소(`tailscale0`)입니다.
- `fuser`, `pgrep`, `curl`, `flock`, cron.

## 4. 설치 (한 번만)

```bash
git clone https://github.com/lwir-deepx-tracking-system/model-bank.git
cd model-bank/final_tracker/tracker
bash setup.sh
```

끝에 모델 일곱 개가 `OK`로 나오고 `setup done`이 보이면 됩니다. `setup.sh`가 하는 일:

- `post.so`를 빌드하고, 모델 파일을 연결하고, 파일이 맞는지(sha256) 확인합니다.
- 카메라 SDK가 있으면 `cam_pipe`를 빌드합니다.
- OpenCV와 GStreamer가 있으면 `gimbal_bridge_v25`를 빌드합니다.

## 5. 실시간 서버 (카메라 + 웹 화면)

카메라 영상을 실시간으로 추적하고 결과를 웹 페이지로 보여 줍니다. 화면에는 사람(상자와 번호)만 그립니다. 차와 트럭도 추적은 하지만 그리지 않습니다.

### 5-1. 켜기

```bash
cd model-bank/final_tracker/tracker
bash run/run_real_gui.sh
```

브라우저에서 `http://<페이지 주소>:8090/`을 엽니다. **페이지에 비밀번호가 없고 짐벌을 움직입니다.** 믿을 수 있는 망의 주소만 주십시오.

- 앞에서 계속 돕니다. 메시지는 `real_gui_1.out`, 결과는 `real_gui_1/`에 남습니다.
- **손으로 켜면 이전 `real_gui_1` 폴더를 지웁니다**(`keepalive.sh`가 켤 때는 `real_gui_prev`로 옮겨 둡니다).
- 스크립트 안의 구성: 검출기 `--det w480`(YOLO26m_(FLIR) 640×480 한 장), ByteTrack + GRU 확인 게이트 + 판정기 + RepVGG-A0_(TMOT,CAMEL).
- `--live-seconds 604800`: 7일 뒤 스스로 끝나고 `keepalive.sh`가 다시 켭니다.
- `--warm-min 45000`은 우리 카메라에서 잰 값입니다. 다른 카메라에서는 0(끔)에서 시작하십시오.

장치마다 다른 값은 `cp run/live.env.example run/live.env` 뒤 고칩니다.

| 값 | 뜻 | 기본 |
| --- | --- | --- |
| `LWIR_LIVE_BIND` | 웹 페이지 주소 | 장치의 Tailscale 주소 |
| `LWIR_LIVE_PORT` | 웹 페이지 포트 | 8090 |
| `LWIR_GUI_HOST` | PC GUI 주소 | 192.168.50.1 |
| `LWIR_GIMBAL_DEV` | 짐벌 시리얼 장치 | `/dev/ttyUSB0` |

바로 끝나면 종료 코드를 봅니다.

| 종료 코드 | 뜻 |
| --- | --- |
| 4 | 추적기가 이미 돌고 있음 (NPU 사용 중) |
| 5 | 짐벌 장치를 다른 프로그램이 씀 |
| 7 | `gimbal_bridge_v25` 없음 (`setup.sh` 다시) |
| 8 | 페이지 주소 없음 (`run/live.env`에 `LWIR_LIVE_BIND`) |
| 9 | 폴더 이동 실패 |

### 5-2. 꺼지면 다시 켜기 (`run/keepalive.sh`)

crontab에 한 줄을 넣습니다(`<클론 경로>`는 `model-bank`를 받은 곳).

```
* * * * * flock -n $HOME/.lwir_keepalive.lock bash <클론 경로>/final_tracker/tracker/run/keepalive.sh >> $HOME/lwir_keepalive.log 2>&1
```

- 5초마다 `/healthz`를 읽습니다. 서버 프로세스가 없거나, 30초 동안 답이 없거나, 프레임은 오는데 30초 동안 결과가 안 나오거나, 짐벌 프로그램이 사라지면 서버를 다시 켭니다.
- 장치가 재부팅돼도 cron이 다시 시작합니다.
- 10분에 3번 다시 켜면 10분 쉽니다.
- 기록은 `~/lwir_keepalive.log`, 죽을 때의 마지막 메시지는 `~/lwir_crash.log`입니다.
- **잠시 멈추기**(배포, 실험): `touch ~/lwir_keepalive.pause`. 다시 켜기: `rm ~/lwir_keepalive.pause`.
- 장치 재부팅 단계는 `~/lwir_keepalive.allow_reboot` 파일이 있을 때만 켜집니다(카메라가 15분 동안 죽어 있고 여러 조건이 맞을 때, 하루 2번까지). 파일이 없으면 재부팅하지 않습니다.

### 5-3. 살아 있는지 확인

```bash
curl -s -m 4 http://<페이지 주소>:8090/healthz
```

`/healthz`(GET, 읽기 전용)는 `up`, `cam_idle`, `out_idle`, `frames_in`, `cam_fail`, `cam_pipe_alive`, `gimbal_alive`, `rss_mb` 등을 돌려줍니다. **스크립트에서 `/gimbal/ping`을 보내지 마십시오.** 그것은 웹 페이지가 살아 있다는 안전 신호입니다.

### 5-4. 끄기

```bash
touch ~/lwir_keepalive.pause
kill -TERM $(pgrep -f 'python3 -B [l]wir_tracker_m1.py')
```

- `cam_pipe`는 서버가 끝나면 스스로 끝납니다. **`cam_pipe`를 `kill -9`로 끄지 마십시오**(카메라가 다음에 안 열릴 수 있습니다).
- 서버를 강제로 끄면 결과 파일이 남지 않습니다.

### 5-5. 웹 화면

- 사람을 클릭하면 짐벌이 그 사람을 따라갑니다. 페이지가 400 ms마다 신호를 보내고, **페이지를 닫으면 짐벌이 멈춥니다.**
- 방향키(화면의 버튼 또는 키보드)는 누르는 동안만 짐벌을 움직입니다.
- 검출 모델 바꾸기, 거름 기준(온기 기준, 가장자리 신뢰도), 추적기 종류, 측정 시작/끝(이름·메모, 영상 저장 선택)이 있습니다.
- 짐벌 제어값은 화면에 글자로만 보이고 화면에서 바꿀 수 없습니다. `run/run_real_gui.sh`에서 고칩니다.

화면 윗줄의 뜻:

| 표시 | 뜻 |
| --- | --- |
| `TRACK`, `CAM` | 추적 처리와 카메라의 초당 프레임 수 |
| `delay` | 카메라 프레임을 받은 뒤 화면에 그리기까지의 시간 |
| `compute` | 프레임 제출부터 추적 끝까지의 시간. `delay`와 함께 **다음 카메라 프레임을 기다리는 시간이 들어 있어** 순수 계산 시간이 아님 |
| `persons` | 지금 화면의 사람 수 |
| `newID` | 지금까지 새로 만든 사람 번호 수 |
| `relink` | 판정기가 예전 번호로 되돌린 수 |

모델 바꾸기 목록(모두 NPU 코어 0):

| `--det` | 모델 | DXNN |
| --- | --- | --- |
| `w480` (기본) | YOLO26m_(FLIR) 640×480 | `dxnn/yolo26m_flir_whole480_nms_640x480.dxnn` |
| `s480` | YOLO26s 640×480 | `../yolo26/yolo26s_nms_640x480.dxnn` |
| `m640` | YOLO26m_(FLIR) 640×640 | `../yolo26/yolo26m_flir_nms_640x640.dxnn` |
| `s6` | YOLO26s 640×640 | `../yolo26/yolo26s_nms_640x640.dxnn` |

- `s480`, `m640`, `s6`은 이 추적기 구성으로 점수를 잰 적이 없습니다(8절 성능 표에 없음).
- 코드에는 `--det ft1`, `ft9`, `s640`도 있지만 그 DXNN은 모델뱅크에 없습니다(연구용).
- 추적기 종류: `gate` = ByteTrack + GRU 확인 게이트(기본), `confirm3` = ByteTrack + 3프레임 확인, `rebind` = 재결합 추적기. 뒤의 둘은 비교용이고 성능 표에 없습니다.

### 5-6. 카메라가 멈췄을 때

- `cam_pipe`가 스스로 카메라를 다시 엽니다(약 30초, 그동안 화면은 멈춤).
- `cam_pipe`가 끝나면 서버가 다시 띄웁니다(연속 실패 시 60초, 5분, 15분 간격).
- `cam_pipe`가 240초 동안 조용하면 서버가 끄고 다시 띄웁니다.

### 5-7. 짐벌 제어값

PD 제어, Kp 30 · Ki 0 · Kd 6 · 불감대 0.06, 자동 추적 속도 최대 10(안전값), 방향키 속도 20. 2026-11-10에 실제 짐벌과 가상 표적으로 조정했고 기록은 `gimbal/PID_TUNING_RECORD_20261110.md`에 있습니다(기록에 나오는 `pid_sim` 스크립트는 모델뱅크에 없습니다).

### 5-8. PC GUI 연결

`gimbal_bridge_v25`가 PC GUI(`pc-application`)와 다음 포트로 통신합니다.

| 방향 | 내용 | 포트 |
| --- | --- | --- |
| 장치 → PC | 영상 (H.264/RTP) | UDP 5000 |
| 장치 → PC | 추적 상자 | UDP 5001 |
| PC → 장치 | 명령 | TCP 5002 |

PC GUI 없이 모델 수 묻기와 모델 바꾸기 명령을 시험하려면 `run/gui_model_client.py <주소> <포트> count|tcount|change:<번호>|wait:<초>`를 씁니다.

### 5-9. 측정 저장

- 측정은 `<측정 폴더>/<날짜>/MEASURE_<시각>_<번호>[_<이름>]/`에 `frames.json`으로 남습니다. 측정 폴더 기본값은 `/mnt/lwir_data/results/live_measure`(`--measure-dir`)입니다.
- 영상 저장을 켜면 16비트 원본이 `frames.raw.gz`(무손실, 1분에 약 400 MB)로 함께 남습니다.
- `--live-seconds`가 지나면 서버가 스스로 끝나고 결과 파일을 남깁니다. 메모리를 아끼려고 오래된 기록은 버리므로 마지막 부분만 남습니다.

## 6. 파일 실행

### 6-1. 입력

- 프레임 이미지가 들어 있는 폴더 하나입니다. 그 안에 `coco.json`이 있어야 합니다.
- `coco.json`의 `images[]`마다 `id`, `file_name`(폴더 기준 상대 경로), `extra_info.video_id`(영상 구분. 영상이 바뀌면 추적이 새로 시작)가 있으면 됩니다. 정답 박스는 필요 없습니다.
- 이미지는 `file_name` 순서로 처리합니다.
- **영상은 640×480, 8비트 흑백(카메라 원본 크기)이 표준입니다.** FLIR 시험 영상 같은 640×512 8비트 흑백도 받습니다(검증한 형태). 그 밖의 크기, 16비트 영상, 크기가 섞인 입력은 어느 프레임이 왜 안 되는지 알려 주고 멈춥니다.
- 16비트 카메라 원본은 먼저 8비트로 줄여야 합니다(이 코드는 줄이지 않습니다).
- 안에서는 640×480 프레임을 640×512 캔버스의 가운데(행 16~496)에 넣고, 위아래 16행은 첫 행·마지막 행으로 채웁니다. 타일, 좌표 기준, 학습한 설정이 FLIR 시험에서 검증한 것과 같아집니다.

### 6-2. 명령

`final_tracker/tracker` 폴더에서 실행합니다.

```bash
python3 -B lwir_tracker_m1.py \
  --tracker bytetrack --no-vit --det rt480 --pipeline --preload \
  --class-gate --gmc --gmc-corners 300 --gmc-mindist 5 --n-init 3 \
  --relabel RVGGA0_REID2C256 --rl-async --gmc-thread --npu-depth 2 \
  --confirm-gate gate_gru32_frozen.json \
  --judge relink_judge_frozen.json --judge-crops 2 \
  --data /영상/프레임/폴더 --out 결과폴더이름
```

- `--data`는 필수이고, `--out`은 `tracker/` 아래에 새로 만들 폴더 이름입니다(이미 있으면 멈춥니다).
- 옵션은 그대로 복사해서 쓰면 최종 구성입니다. 각 옵션의 뜻은 루트 `README.md`의 4-4절에 있습니다.

### 6-3. 결과

`결과폴더/TRACKS.jsonl`이 최종 결과입니다. 한 줄이 한 프레임입니다.

```
{"image_id", "coco_image_id", "video_id", "tracker_reset",
 "tracks_xyxy_id_conf_class_detectionindex": [[x1, y1, x2, y2, 번호, 신뢰도, 클래스, 검출 번호], ...]}
```

- 좌표는 **입력 영상 기준**입니다(640×480 입력이면 640×480 영상의 좌표, 640×512 입력이면 640×512 영상의 좌표).
- 번호는 영상 안에서만 유일합니다.
- 클래스는 0 = 사람, 2 = 차, 6 = 트럭입니다. 사람 번호만 게이트·판정기로 다듬습니다.
- `RESULT.json`에는 프레임 간격, 지연, 모델 해시, 사용한 옵션이 들어 있습니다.

## 7. 옵션

파일 실행 명령의 옵션은 루트 `README.md`의 4-4절에 있습니다. 아래는 검출기 선택과 실시간 서버 옵션입니다.

| 옵션 | 뜻 |
| --- | --- |
| `--det` | 검출기. `rt480` = YOLO26m_(FLIR) 타일 2장(코어 0·1), `w480` = YOLO26m_(FLIR) 한 장(코어 0), `s480`·`m640`·`s6` = 5-5절 표 |
| `--live-port` | 웹 페이지 포트. 주지 않으면 파일 실행입니다 |
| `--live-bind` | 웹 페이지 주소. 기본값 `127.0.0.1`은 이 장치에서만 열립니다 |
| `--live-seconds` | 실행 시간(초). 기본 600 |
| `--live-wait` | 카메라 첫 프레임을 기다리는 시간(초). 기본 90, 0 = 포기하지 않음(카메라가 없어도 페이지와 짐벌 키는 살아 있음) |
| `--live-map` | 16비트 → 8비트 변환. `pct`(기본) = 밝기 1~99% 구간을 천천히 따라감, `fixed` = 0~65535 고정 |
| `--live-scale` | 화면 배율. 기본 1.5 |
| `--warm-min` | 상자 안 16비트 원본 밝기가 이 값보다 낮으면 사람 검출을 버립니다. 0 = 끔 |
| `--edge-birth`, `--edge-px` | 화면 가장자리(`--edge-px` 이내, 기본 4)에서 새 번호를 만들 때 필요한 신뢰도 |
| `--gimbal`, `--gimbal-bridge`, `--gimbal-speed`, `--gimbal-opts` | 짐벌 시리얼 장치, 연결 프로그램 경로, 자동 추적 속도 상한, 연결 프로그램에 넘길 옵션 |
| `--gui-host`, `--gui-bind` | PC GUI 주소와 명령을 받을 주소 |
| `--measure-dir` | 측정 저장 폴더. 기본 `/mnt/lwir_data/results/live_measure` |

## 8. 성능

모두 FLIR 시험 영상 8편 3,749프레임, 사람, DX-M1 INT8, 파일 실행 실측입니다. 실제 카메라 영상은 정답이 없어 점수를 재지 않았습니다.

![최종 결과 요약: IDF1·오검출·미검출(작은 사람/큰 사람)·번호 바뀜·속도, mAP·HOTA 그래프, 쓴 모델과 부품](images/final_summary_m1.png)

### 8-1. 부품을 더했을 때 (검출기 `rt480`)

| 구성 | IDF1 | 오검출 | 미검출 | 번호 바뀜 | 속도 |
| --- | --- | --- | --- | --- | --- |
| 기준: YOLO26m_(FLIR) INT8 + ByteTrack | 0.6567 | 1,466 | 4,010 | 111 | 52.6 fps |
| 이전 확정: + SameClass_connect + OpticalFlow-Fast + 3FrameConfirm + KeepID | 0.6897 | 908 | 4,214 | 58 | 74.5 fps |
| **최종 (이 폴더)**: 게이트 + 판정기 | 0.6932 | 650 | 4,514 | 31 | 71.8 fps |

### 8-2. 검출기별 (게이트와 판정기 모두 켬)

영상의 행 16~496 구간 채점, 2026-11-06 같은 날 측정입니다. 속도는 실행마다 달라집니다(8-1의 71.8 fps는 다른 날 측정).

| `--det` | 검출기 | AP50 | AP50-95 | HOTA | IDF1 | AssA | 번호 바뀜 | 오검출 | 미검출 | 속도 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rt480` | YOLO26m_(FLIR) 타일 2장 | 0.7636 | 0.4571 | 0.5645 | 0.6932 | 0.6535 | 31 | 650 | 4,514 | 78.8 fps |
| `w480` | YOLO26m_(FLIR) 한 장 | 0.7651 | 0.4583 | 0.5541 | 0.6783 | 0.6266 | 39 | 767 | 4,416 | 62.5 fps |
| `s640` | YOLO26s 한 장 | 0.7233 | 0.4232 | 0.5516 | 0.6776 | 0.6350 | 51 | 1,043 | 4,433 | 70.1 fps |

- `s640` DXNN은 출력이 `tensor_requirements.md` 형식(`boxes`, `scores`)과 달라 모델뱅크에 올리지 않았고, 표의 값은 기록입니다.
- `s640`은 위 파일 실행에서 640×512 프레임 전체를 넣었고, 실시간 서버에서는 카메라 640×480을 넣습니다. 나머지 두 구성은 어느 쪽이든 행 16~496만 봅니다.
- **한 장 구성에서는 판정기가 점수를 내립니다.** 번호 바뀜은 어느 쪽이든 줄어듭니다.

| `--det` | 지표 | 판정기 끔 | 판정기 켬 |
| --- | --- | --- | --- |
| `w480` | HOTA | 0.5629 | 0.5541 |
| `w480` | IDF1 | 0.6852 | 0.6783 |
| `w480` | AssA | 0.6456 | 0.6266 |
| `s640` | HOTA | 0.5535 | 0.5516 |
| `s640` | IDF1 | 0.6771 | 0.6776 |
| `s640` | AssA | 0.6392 | 0.6350 |

## 9. 한계와 확인 범위

### 9-1. 꼭 알아둘 것

- 오검출과 번호 바뀜은 줄지만 **미검출이 늘어납니다**(늘어난 504건 중 472건이 키 32px 미만의 작은 사람). IDF1 이득은 +0.35점으로 작고 통계적으로 확정되지 않았습니다.
- 게이트와 판정기는 고정 교차로 영상(5~9 fps)으로 학습했습니다. 다른 카메라·프레임률에서 같은 이득이 난다는 보장은 없습니다.
- 실제 카메라에서 **다른 사람이 번호를 이어받는 경우가 보고되었습니다.** 실제 카메라는 약 24 fps 근거리입니다. 학습 영상과의 차이가 주는 영향은 재지 않았습니다.
- 프레임 하나의 지연은 p50 38.5 ms, p95 54.7 ms로 30 fps 조건(33 ms)을 만족하지 않습니다. 처리량(프레임 간격 p95 20.1 ms)은 만족합니다.
- 정답이 없는 영상에서는 결과가 맞는지 확인할 방법이 없으니 결과를 정답으로 쓰지 마십시오.
- 다른 프로그램이 NPU를 쓰거나 NPU 서비스가 다시 시작되면 서버가 멈출 수 있습니다(2026-11-04에 한 번 관찰).
- 2026-11-10에 장치 전체가 두 번 응답하지 않았습니다. 원인은 아직 찾지 못했습니다.

### 9-2. 확인한 것

- 8절의 값은 **2026-11-06 코드로** 잰 것입니다. 그때 `rt480`, `w480` 파일 실행의 TRACKS.jsonl이 이전 결과와 바이트 단위로 같았습니다.
- 2026-11-10, 우리 장치에서 실제 카메라·짐벌로 `run_real_gui.sh` 구성을 돌렸습니다.
  - 서버를 `kill -9`로 죽이면 8초 안에 다시 켜졌습니다.
  - 멈춤 파일이 있으면 다시 켜지 않았습니다.
  - 두 번째 `cam_pipe`는 종료 코드 7로 거절됐습니다.
  - 장치가 갑자기 재시작된 뒤 부팅 44초 만에 서버가 답했습니다.
  - 웹 화면은 실제 휴대폰 터치와 짐벌 이동을 빼고 확인했습니다.

### 9-3. 확인하지 않은 것

- **2026-11-10 코드의 파일 실행이 이전 결과와 같은지.** 새 기능은 모두 기본값이 꺼짐이지만 다시 돌려 보지 않았습니다.
- **8시간 연속 실행.**
- 재부팅 단계. 실제로 돌려 본 기록이 없습니다.
- **이 폴더를 새로 받아 `setup.sh`로 빌드하고 켜 보는 시험.** 모델뱅크의 `run/` 스크립트는 우리 장치의 사본과 조금 다릅니다(폴더·주소를 스크립트 위치와 `live.env`에서 읽음). 이 형태로는 장치에서 돌려 보지 않았습니다.

자세한 근거와 한계는 루트 `README.md`의 4절과 "상세 근거"에 있습니다.
