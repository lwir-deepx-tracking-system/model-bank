# 실시간 추적 서버 (`tracking_server/`, 포트 8090)

열화상 카메라 영상에서 **사람을 찾고, 같은 사람에게 같은 번호를 유지하고, 고른 사람을 짐벌로 따라가는** 서버입니다. 장치는 DX-M1(Orange Pi 5 Plus) 한 대입니다.

![파이프라인과 실시간 처리](images/pipeline_realtime_20261111.png)

> 그림 읽는 법: 숫자는 2026-11-11 에 장치에서 짧게 읽은 값이고, 막대 길이는 이해를 돕는 그림입니다(실제 단계별 시간은 7절). 그림의 "코어 0·1 교대"는 시험한 방식이며, 기준을 넘지 못해 지금은 코어 0 하나로 돕니다. 사진의 상자와 번호는 검수한 라벨로 그린 예시입니다.

## 목차

1. [한눈에 보기](#1-한눈에-보기)
2. [빠른 시작](#2-빠른-시작)
3. [폴더](#3-폴더)
4. [설정](#4-설정-scriptsliveenv)
5. [거르는 곳](#5-거르는-곳)
6. [짐벌이 멈추는 경우](#6-짐벌이-멈추는-경우)
7. [속도: 장치에서 잰 값](#7-속도-장치에서-잰-값)
8. [꺼지면 다시 켜기, 끄기](#8-꺼지면-다시-켜기-끄기)
9. [웹 화면](#9-웹-화면)
10. [모델](#10-모델)
11. [확인한 것과 확인하지 않은 것](#11-확인한-것과-확인하지-않은-것)

## 1. 한눈에 보기

| 단계 | 하는 일 | 어디서 |
|---|---|---|
| 1. 영상 받기 | 16비트 640×480 영상 | CPU |
| 2. 검출 | YOLO26m 으로 사람 상자 | **NPU 코어 0** |
| 3. 검출 뒤 Filter | 온도·가장자리로 상자 거르기 | CPU |
| 4. 추적 | 번호 유지 (BoT-SORT, Kalman) | CPU |
| 5. 다시 알아보기 | 외형으로 옛 번호 (KeepID) | **NPU 코어 2** + CPU |
| 6. 조준 | 머리를 가운데로, PID | CPU |

- 신경망 두 개(검출, 외형)는 모두 `.dxnn`(INT8)으로 DX-M1 NPU 에서 돕니다. CPU 에서 도는 신경망은 없습니다.
- 조준은 150 ms 앞을 내다보고(예측 제어) 짐벌에 속도 명령을 보냅니다.
- 코드 자리: 1 `src/camera/`, 2 `src/detect/`, 3·4 `src/track/`, 5 `src/reid/`, 6 `src/gimbal/`.

## 2. 빠른 시작

```bash
git clone https://github.com/lwir-deepx-tracking-system/model-bank.git
bash model-bank/tracking_server/scripts/setup.sh         # 한 번만: 빌드, 모델 연결과 확인
bash model-bank/tracking_server/scripts/run_real_gui.sh  # 켜기
```

브라우저에서 `http://<장치 주소>:8090/` 을 엽니다.

- **화면에는 비밀번호가 없고 짐벌을 움직입니다.** 믿을 수 있는 기기만 닿는 주소에서 쓰세요.
- `setup.sh` 끝에 `setup done` 이 나오면 준비가 된 것입니다.
- 장치마다 다른 값은 `scripts/live.env.example` 을 `scripts/live.env` 로 복사해 고칩니다 (4절).
- 서버를 다시 켜면 예측 제어는 꺼진 채로 켜집니다. 화면에서 다시 켭니다.

<details>
<summary>준비물</summary>

- DX-M1(Orange Pi 5 Plus), DXRT 3.4.2, Ubuntu 22.04, Python 3.10, g++
- Python 패키지: numpy, opencv(cv2), scipy, `dx_engine`
- i3system 열화상 카메라와 카메라 SDK(`/usr/local/include/i3system`). 없으면 카메라 프로그램을 빌드하지 않습니다.
- 짐벌(직렬 장치), OpenCV(`opencv4`), GStreamer(`gstreamer-1.0`, `gstreamer-app-1.0`). 없으면 짐벌 프로그램을 빌드하지 않습니다.
- `fuser`, `pgrep`, `curl`, `flock`, cron
- NPU 를 다른 프로그램이 쓰고 있지 않아야 합니다.

</details>

<details>
<summary>setup.sh 가 하는 일</summary>

1. `src/detect/post.cpp` → `post.so` 빌드
2. `models/links/<내부 이름>.dxnn` 링크 만들기 (`models/MANIFEST.json` 의 `_links` 표대로)
3. 모델 파일을 확인값(sha256)으로 확인. 서버가 쓰는 모델이 없거나 다르면 이름을 알려 주고 멈춥니다.
4. 카메라 SDK 가 있으면 `src/camera/cam_pipe` 빌드
5. OpenCV 와 GStreamer 가 있으면 `src/gimbal/gimbal_bridge_v25` 빌드

빌드한 파일과 `models/links/` 는 git 에 올라가지 않습니다.

</details>

<details>
<summary>켜자마자 끝날 때</summary>

`real_gui_1.out` 과 끝나는 값을 봅니다.

| 끝나는 값 | 뜻 |
|---|---|
| 4 | 서버가 이미 떠 있음 |
| 5 | 짐벌 장치를 다른 프로그램이 씀 |
| 7 | 짐벌 프로그램이 없음 (`setup.sh` 다시) |
| 8 | 페이지 주소 없음 (`LWIR_LIVE_BIND`) |
| 9 | 폴더 이동 실패 |

</details>

## 3. 폴더

```
tracking_server/
├── start_tracking_server.py   서버를 켜는 파일
├── models/                    모델 파일
│   ├── detect/                검출 모델
│   ├── track/                 추적용 파일
│   └── reid/                  외형 비교 모델
├── src/                       서버 코드
│   ├── app.py                 본체
│   ├── detect/                검출
│   ├── track/                 추적, 검출 뒤 Filter
│   ├── reid/                  외형 비교
│   ├── camera/                카메라
│   ├── gimbal/                짐벌
│   ├── web/                   8090 화면 처리
│   └── common/                같이 쓰는 값
├── page/                      8090 화면
├── scripts/                   준비, 켜기, 다시 켜기
└── images/                    이 글의 그림
```

같은 일은 어디서든 같은 이름입니다: `detect`(검출), `track`(추적), `reid`(외형 비교).

<details>
<summary>폴더별 파일</summary>

**`src/detect/`**
- `npu_engine.py` NPU 에 모델 올리기·돌리기
- `detect_frame.py` 프레임 → 상자
- `model_scan.py` 검출 모델 파일 스스로 찾기
- `zoom_redetect.py` 작은 사람 확대 재검출
- `native_post.py` + `post.cpp` 검출기 출력을 푸는 빠른 계산
- `frame_feed.py` 번갈아 넣기 (꺼짐)

**`src/track/`**
- `trackers.py` 추적기, `kalman.py` Kalman filter
- `camera_motion.py` 카메라 움직임 추정
- `position_memory.py` 놓친 사람 위치 기억
- `gate.py` 확인 판정기
- `confirm_time.py` 새 사람 확인을 시간으로
- `warm_p95.py` 온도 Filter 숫자 계산
- `stitch.py` 끊긴 추적 잇기
- `stabilizer.py` 화면 흔들림 보정
- `gt_score.py` 정답과 비교한 점수 계산 (놓침, 잘못 그린 상자, 번호 바뀜, IDF1, MOTA)

**`src/reid/`**
- `reid_models.py` 외형 모델 목록과 실행 중 바꾸기
- `reid_embed.py` 사람 조각에서 특징 뽑기
- `relabel_online.py` KeepID 규칙
- `relink_online_judge.py` 번호 잇기 판정기

**`src/camera/`**
- `cam_pipe.cpp` 카메라 프로그램
- `camera_link.py` 그 프로그램을 띄우고 읽기, 카메라 초기화
- `camera_extra.py` 카메라 기록
- `replay_cam.py` 저장된 녹화를 카메라처럼 보내기
- `replay_link.py` 화면의 "녹화 재생": 녹화 목록, 카메라 ↔ 녹화 바꾸기, 재생 중 짐벌 막기
- `replay_eval.py` 재생 중 정답과 비교: 라벨 읽기, 프레임과 PNG 짝 맞추기, 결과 파일

**`src/gimbal/`**
- `gimbal_bridge_v25.cpp` 짐벌 + PC GUI 프로그램
- `gimbal_bridge_v27.cpp` + `v27/` + `sim/` 시험용 판
- `team/` 팀 제어·GUI 코드 사본 (출처 `team/SOURCE.txt`)
- `gimbal_link.py` 짐벌 프로그램을 띄우고 명령 보내기
- `test_*.py` 시험
- `PID_TUNING_RECORD_20261110.md` 제어값 기록

**`src/web/`**
- `page_handler.py` 주소별 처리와 화면 그림
- `page_settings.py` 화면에서 바꾸는 설정, 모델 바꾸기
- `measure.py` 측정 버튼
- `system_buttons.py` 재부팅 버튼

**`src/common/`**
- `shared_values.py` 고정 값, `state.py` 공유 값 보관함
- `settings.py` 실행 중 바뀌는 설정
- `stage_timing.py` 단계별 시간 재기
- `result_summary.py` 결과 요약
- `device_wait.py` 다른 측정 기다리기
- `pc_dry_run.py`, `research_*.py` PC 에서 돌려 보기와 연구용 기능

**`page/`**
- `live_page.html` 서버가 읽는 한 파일
- `html/` `css/` `js/` 조각, `parts.json` 조각 순서
- `build_page.py` 합치는 도구

**`scripts/`**
- `setup.sh` 준비, `run_real_gui.sh` 켜기
- `keepalive.sh` 꺼지면 다시 켜기
- `live.env.example` 장치별 설정 예
- `gui_model_client.py` PC GUI 명령 시험

</details>

## 4. 설정 (`scripts/live.env`)

| 값 | 기본 |
|---|---|
| `LWIR_LIVE_BIND` | 장치의 Tailscale 주소 |
| `LWIR_LIVE_PORT` | 8090 |
| `LWIR_GUI_HOST` | 192.168.50.1 |
| `LWIR_GIMBAL_DEV` | 고정 이름, 없으면 `/dev/ttyUSB0` |
| `LWIR_GIMBAL_AIM_Y` | 0.08 (머리) |
| `LWIR_GIMBAL_SPEED_MAX` | 100 |
| `LWIR_GIMBAL_REACQUIRE_MS` | 10000 |
| `LWIR_CONFIRM_MS` | 130 |
| `LWIR_GIMBAL_BRIDGE` | `gimbal_bridge_v25` |
| `LWIR_GIMBAL_EXTRA_OPTS` | 없음 |
| `LWIR_REPLAY_ROOT` | 라벨 사이트 녹화 폴더 |

뜻:

- `LWIR_LIVE_BIND`, `LWIR_LIVE_PORT` — 웹 화면 주소와 포트
- `LWIR_GUI_HOST` — 팀 PC GUI 주소
- `LWIR_GIMBAL_DEV` — 짐벌 직렬 장치
- `LWIR_GIMBAL_AIM_Y` — 사람 상자의 어느 높이를 화면 가운데에 둘지 (위에서부터 비율. 0.08 = 머리, 0.3 = 가슴)
- `LWIR_GIMBAL_SPEED_MAX` — 자동으로 따라갈 때의 최고 속도
- `LWIR_GIMBAL_REACQUIRE_MS` — 놓친 사람을 기다리는 시간 (ms)
- `LWIR_CONFIRM_MS` — 새 사람을 화면에 내기 전에 확인하는 시간 (ms). 0 = 끔 (늘 3장)
- `LWIR_GIMBAL_BRIDGE` — 짐벌 프로그램 (`src/gimbal/` 아래)
- `LWIR_GIMBAL_EXTRA_OPTS` — 짐벌 프로그램에 더 줄 옵션
- `LWIR_REPLAY_ROOT` — 화면의 "녹화 재생"이 고르는 녹화 폴더 (기본 `/mnt/lwir_data/label_web/LWIR_labels`, 서버 옵션 `--replay-root`). 폴더가 없으면 재생을 쓸 수 없습니다

시험·측정용 값(단계별 시간 재기, 카메라 시간표, 번갈아 넣기 등)은 `scripts/live.env.example` 아래쪽에 설명이 있고, 모두 기본 꺼짐입니다.

**짐벌 제어값**
- PD 제어 (Kp 30, Ki 0, Kd 6), 불감대 0.06, 손으로 움직일 때 속도 20.
- 최고 속도 10 일 때 맞춘 값입니다 (`src/gimbal/PID_TUNING_RECORD_20261110.md`). 최고 속도 100 에 맞춘 값은 아직 없습니다.

**v27 (시험용 판)**
- 움직임을 부드럽게 하려는 판입니다. `src/gimbal/V27_CHANGES.md` 의 명령으로 빌드합니다.
- 옵션 예: `--deadband-exit 0.10 --axis-reversal 1`
- 실제 짐벌에서 흔들림이 줄어드는지는 아직 재지 않았습니다.

**조준 높이**
- `LWIR_GIMBAL_AIM_Y` 를 바꾸면 화면의 세로 오차 표시 기준(`page/js/04_ping_fps_aim.js` 의 `AIM_Y`)도 같이 바꿉니다.

**새 사람 확인 시간 (`LWIR_CONFIRM_MS`)**
- 장 수 = 130 × 잰 초당 장 수 ÷ 1000 을 반올림.
- 초당 22~27장이면 3장(전과 같음), 28장을 넘으면 4장. 느려져도 3장보다 작아지지 않습니다.
- 서버를 켠 채로 화면 "Filter 값 → 새 사람 확인 시간"에서 바꿀 수 있습니다 (`POST /rules/set?confirm_ms=160`). 화면에서 바꾼 값은 저장되어 다음에 켤 때 먼저 쓰입니다.
- 지금 값은 화면의 "지금 적용된 값"에 나옵니다.
- 코드 `src/track/confirm_time.py`, PC 시험 `python -m unittest src.track.test_confirm_time`.

## 5. 거르는 곳

영상 한 장에서 상자가 버려지는 곳을 순서대로 적었습니다.

**① 점수 · 겹침** — 점수 낮은 상자, 같은 사람에 겹친 상자를 버림. 바꾸는 곳: `scripts/run_real_gui.sh`

**② 온도 Filter** — 사람 상자 안의 16비트 원본 값(상위 5 % 지점)이 정한 범위 밖이면 버림. 바꾸는 곳: 화면 "Filter 값"

**③ 종류** — 사람과 차를 서로 짝짓지 않음. 바꾸는 곳: `scripts/run_real_gui.sh`

**④ 가장자리 Filter** — 화면 끝에 닿은 상자 중 점수가 낮은 것은 새 번호를 받지 못함. 바꾸는 곳: 화면 "Filter 값"

**⑤ 새 사람 확인 (130 ms)** — 130 ms 동안 이어서 잡히지 않은 것은 화면에 나오지 않음. 바꾸는 곳: 화면 "Filter 값" 또는 `LWIR_CONFIRM_MS`

**⑥ 불감대** — 조준점이 가운데 가까이 있으면 짐벌이 움직이지 않음. 바꾸는 곳: `scripts/run_real_gui.sh`

## 6. 짐벌이 멈추는 경우

짐벌 속도 명령은 스스로 꺼지지 않습니다. 아래 경우에 누가 멈추는지 정리했습니다 (2026-11-12, 코드로 확인).

| 무슨 일 | 멈출 때까지 |
|---|---|
| 서버 처리가 멈춤 | 약 0.3초 |
| 그림이 이전 장과 똑같음 | 다음 장 |
| 서버 프로그램이 죽음 | 바로 |
| 짐벌 프로그램이 죽음 | 바로 |
| 웹 화면을 닫음 (손으로 움직이던 중) | 1.5초 |

누가 멈추나:

- **서버 처리가 멈춤 (새 상자가 안 옴)** — 짐벌 프로그램이 250 ms 넘은 영상은 쓰지 않고 멈춥니다.
- **영상은 오는데 그림이 이전 장과 똑같음** — 서버가 그 장은 짐벌에 사람을 보내지 않습니다. 고른 사람은 다시 찾는 시간 동안 유지합니다. 기록에 `FROZEN`, `/healthz` 에 `frozen_frames`.
- **서버 프로그램이 죽음** — 짐벌 프로그램이 입력이 끊기면 멈춤 명령을 보내고 끝납니다.
- **짐벌 프로그램이 죽음** — 서버가 짐벌 장치에 멈춤 명령을 직접 씁니다.
- **웹 화면을 닫음** — 1.5초 동안 화면 신호가 없으면 멈춥니다.

**아직 막지 못한 것**

- 짐벌 장치 명령은 두 축이 속도 하나를 같이 씁니다. 가로로 빠르게 따라갈 때 세로 오차가 불감대를 조금 넘으면, 세로도 그 속도로 움직일 수 있습니다 (v25, v27 모두). 고치려면 짐벌 프로그램을 바꿔 실제 짐벌로 시험해야 합니다.
- 짐벌 프로그램이 죽지 않고 멈춰 버리거나, 장치 전원이 꺼지면 마지막 명령이 남습니다.

## 7. 속도: 장치에서 잰 값

장치(DX-M1, 진짜 NPU)에서 저장된 실외 녹화를 재생해 180초씩 3번 잰 값입니다 (2026-11-12, 짐벌 없음). "도착 → 결과"의 중간값입니다.

**초당 25장**

| 방식 | 중간값 | 버린 장 |
|---|---|---|
| 지금 방식 (코어 0) | 72 ms | 0 |
| 번갈아 넣기 (코어 0·1) | 66 ms | 0.5 % |

**초당 30장**

| 방식 | 중간값 | 버린 장 |
|---|---|---|
| 지금 방식 (코어 0) | 63 ms | 0~3장 |
| 번갈아 넣기 (코어 0·1) | 75 ms | 5 % |

- 늦을 때(p95)는 초당 25장에서 86 ms 대 85 ms, 초당 30장에서 87~90 ms 대 100 ms 였습니다.
- **번갈아 넣기는 켜지 마세요.** 초당 25장에서 중간값은 줄지만 장을 버리고, 30장에서는 느려집니다.

**한 장의 단계별 시간** (초당 25장, 지금 방식, 중간값)

| 단계 | 시간 |
|---|---|
| 도착 → NPU 에 넣음 | 3 ms |
| NPU 검출 (코어 0) | 30 ms |
| 상자 풀기 + 다음 장 넣기 | 12 ms |
| 추적 전 준비 | 8 ms |
| 추적 | 3 ms |
| 번호 잇기 (코어 2 + CPU) | 2 ms |
| 내보내기 | 12 ms |

- 추적 전 준비 8 ms 가운데 온도 Filter 가 7 ms 였습니다 (사람이 최대 9명인 녹화).
- 내보내기 12 ms 는 짐벌 전송과 화면용 그림(색 그림 6 ms, JPEG 5 ms)입니다.

**위 측정 뒤에 고친 것** (PC 에서 결과가 똑같은지 확인, 장치에서는 아직 안 잼)

- **온도 Filter 숫자를 가벼운 계산으로** — 같은 값이 나오는 방법 (`src/track/warm_p95.py`, 시작할 때 스스로 같은지 확인). 상자 하나 136 → 28 μs (PC).
- **짐벌 명령을 화면 그림보다 먼저 보냄** — 짐벌 쪽 약 6 ms 빨라질 것으로 봄.
- **보는 사람이 없으면 화면 그림을 1초에 한 장만** — 그때 약 11 ms 빠질 것으로 봄.

## 8. 꺼지면 다시 켜기, 끄기

crontab 에 한 줄을 넣으면 서버가 꺼졌을 때 스스로 다시 켜집니다 (`<클론 경로>` 는 저장소를 받은 곳).

```
* * * * * flock -n $HOME/.lwir_keepalive.lock bash <클론 경로>/tracking_server/scripts/keepalive.sh >> $HOME/lwir_keepalive.log 2>&1
```

- **다시 켜기를 잠시 멈춤:** `touch ~/lwir_keepalive.pause`
- **다시 돌림:** `rm ~/lwir_keepalive.pause`
- **서버 끄기:** 위처럼 멈춘 뒤
  `kill -TERM $(pgrep -f 'python3 -B [s]tart_tracking_server.py')`
- **기록 보기:** `~/lwir_keepalive.log`, 꺼질 때의 마지막 메시지는 `~/lwir_crash.log`

주의:

- 카메라 프로그램(`cam_pipe`)은 서버가 끝나면 스스로 끝납니다. `kill -9` 로 끄지 마세요 (카메라가 다음에 안 열릴 수 있습니다).
- **NPU 측정 프로그램을 돌리기 전에는 서버를 먼저 꺼 주세요.** 같이 돌면 NPU 서비스가 죽어 서버가 꺼집니다. 측정 프로그램(`lwir_benchmark`)이 도는 동안에는 다시 켜기가 기다립니다.

<details>
<summary>다시 켜기 규칙 자세히</summary>

- 5초마다 `/healthz` 를 읽습니다. 서버 프로세스가 없거나, 30초 동안 답이 없거나, 프레임은 오는데 30초 동안 결과가 안 나오거나, 짐벌 프로그램이 3번 연속 없으면 다시 켭니다. 시작 뒤 180초 동안은 멈춤 검사를 하지 않습니다.
- 다시 켤 때 이전 `real_gui_1/` 을 `real_gui_prev/` 로 옮겨 둡니다. 손으로 켜면 이전 `real_gui_1/` 을 지웁니다.
- 10분에 3번 다시 켜면 10분 쉽니다.
- 장치 재부팅 단계는 `~/lwir_keepalive.allow_reboot` 파일이 있을 때만 켜집니다 (카메라 프로그램이 3번 연속 실패, 15분 동안 프레임 없음, 켜진 지 30분 넘음, 하루 2번까지, 60분 간격).
- **카메라가 USB 에서 아예 보이지 않으면 재부팅하지 않습니다** (재부팅으로는 살아나지 않고, 선을 다시 꽂아야 합니다).

</details>

## 9. 웹 화면

화면에서 할 수 있는 것: 영상 보기, 사람을 눌러 따라가기, 손으로 짐벌 움직이기, 검출 모델·추적기·외형 모델·위치 기억 고르기, Filter 값 바꾸기, 측정 저장, 카메라 초기화, 장치 재부팅, 저장된 녹화 재생.

화면을 고칠 때는 `page/live_page.html` 을 손으로 고치지 않습니다.

1. `page/html/`, `page/css/`, `page/js/` 에서 조각을 고칩니다 (어느 조각에 무엇이 있는지는 [page/README.md](page/README.md)).
2. `page/` 에서 `python3 build_page.py` 로 다시 만들고, `python3 build_page.py --check` 로 확인합니다.
3. 조각과 `live_page.html` 을 함께 올립니다. 서버를 다시 켤 필요는 없습니다.

**녹화 재생** (2026-11-12)

라벨 사이트의 녹화 하나를 카메라 대신 서버에 넣어, 검출·추적을 화면에서 봅니다. 화면 "녹화 재생" 구역에서 고릅니다.

- 목록: 녹화 폴더(`LWIR_REPLAY_ROOT`) 아래에서 `images/` 에 번호 PNG(`000001.png` …)가 있는 폴더. 이름, 장 수, 8/16 비트가 나옵니다.
- 고를 것: 초당 장 수(5~30, 기본 25), 반복(끄면 한 번 튼 뒤 저절로 카메라로).
- 재생 중에는 영상 맨 위 줄이 붉은 바탕에 `REPLAY <이름>` 으로 바뀝니다. `/healthz` 의 `replay` 에 이름이 나옵니다.
- 앞에서부터 약 650장(400 MB)까지만 씁니다. 처음 불러오는 데 몇 초 걸립니다.
- 재생 멈춤 → 카메라 프로그램을 다시 시작합니다. 영상은 10~60초 뒤.
- 재생 중에는 카메라 초기화 버튼을 받지 않습니다.

재생 중 짐벌:

- 시작할 때 짐벌을 멈춥니다 (고른 사람도 놓음).
- 짐벌 프로그램에 가는 줄에 사람을 싣지 않습니다. 녹화 속 사람을 따라갈 수 없습니다. PC GUI 는 그림을 계속 받습니다.
- 영상 누르기와 손 이동은 거절합니다 ("녹화 재생 중에는 짐벌을 움직이지 않습니다").
- 멈춘 뒤에는 사람을 다시 누를 때까지 아무도 따라가지 않습니다.

알아 둘 것:

- **8비트 녹화**는 값이 0~255 를 늘린 것이라 온도 Filter(16비트 값, 기본 45000)에 사람이 걸러질 수 있습니다. 화면에 주의 글이 나옵니다. 볼 때만 온도 Filter 를 잠시 끄세요. 서버가 Filter 를 스스로 바꾸지 않습니다.
- 재생 중 측정도 됩니다. 측정 결과(`summary.json` 의 `replay`, `README_KO.txt`)에 녹화 이름이 적혀 카메라 녹화와 섞이지 않습니다.

**정답과 비교** (2026-11-12)

검수한 정답 라벨이 있는 녹화를 재생하면, 서버가 내보낸 사람 상자를 프레임마다 정답과 비교해 "녹화 재생" 구역의 "정답과 비교"에 1초마다 보여 줍니다.

- 정답 = 녹화 폴더의 `tracks_edited_mot.txt` (라벨 사이트에서 고친 라벨). 자동 라벨 `tracks_mot.txt` 만 있으면 채점하지 않고 그렇다고 적습니다.
- 채점하는 프레임 = 라벨 파일에 줄이 있고, `review_state.json` 의 `reviewed` 에 있고, `removed`·`purged` 에 없는 프레임. `review_state.json` 이 없으면 라벨이 있는 모든 프레임 ("검수 상태 없음").
- 숫자는 두 칸입니다: **새 채점**(무시 구역 뺌), **옛 채점**(무시 구역 없음). 맨 위가 **잘못 그린 상자** 입니다.
- 줄: 잘못 그린 상자, 정답 사람, 찾음, 놓침(작은 사람 32 px 미만 / 그 밖), 찾은 비율, 맞은 비율, 번호 바뀜, IDF1, MOTA, 무시 구역 상자.
- **숫자 처음부터** 단추 = 지금부터 다시 셉니다 (`POST /replay/eval/reset`).
- **정답 상자 그리기** = 화면 그림에만 정답 상자를 얇은 점선으로 그립니다 (흰색 = 사람, 분홍 = 무시 구역). 짐벌에는 가지 않습니다. 보는 사람 모두 같은 그림입니다 (`POST /replay/eval/draw?on=0|1`).
- 재생을 멈추거나 다른 녹화로 바꾸면 (서버가 끝날 때도) 결과를 `<결과 폴더>/replay_eval_<이름>_<시각>.json` 에 씁니다: 모든 숫자, 규칙, 라벨 파일 이름과 sha256, 채점한·건너뛴 프레임 수, 채점한 프레임 목록(서버 프레임 이름, PNG 번호, 바퀴).

채점 규칙 (사용자 규칙):

1. 사람만: 정답 클래스 0 과 서버가 내보낸 사람 상자. 서버 상자는 화면에 보이고 짐벌에 가는 것과 같은 상자, 640x480 좌표 (서버 안의 +16 줄은 뺌).
2. 짝짓기: CLEAR MOT 방식 (Bernardin & Stiefelhagen 2008, py-motmetrics 와 같음). IoU 0.5 이상만, 하나는 하나와만. 앞 프레임에서 짝이었던 정답 사람과 서버 번호가 아직 IoU 0.5 이상이면 그대로 짝짓고, 나머지만 헝가리안 방법 (scipy).
3. 새 채점: 클래스 1 상자 = 무시 구역. 왼쪽·오른쪽 가장자리 4 px 안에 닿고 보이는 폭이 24 px 보다 좁은 사람도 무시 구역. 짝이 없는 서버 상자가 무시 구역 하나 안에 넓이의 반 이상 들어가면(IoA 0.5 이상) 잘못 그린 상자가 아닙니다. 무시 구역은 놓침이 아닙니다.
4. 옛 채점: 클래스 0 상자만 정답, 가장자리 규칙 없음, 무시 구역 없음.
5. 번호 바뀜: 정답 사람 하나에 짝지어진 서버 번호가, 그 사람이 마지막으로 짝지어졌던 번호와 다를 때 1.
6. IDF1: 채점한 모든 프레임에 걸쳐 정답 번호와 서버 번호를 한 번에 하나씩 짝지어 셉니다.
7. MOTA = 1 − (놓침 + 잘못 그린 상자 + 번호 바뀜) ÷ 정답 사람.
8. 재생 시작 직후 추적 기록이 없는 프레임도 넣어 셉니다. 숫자를 좋게 하려고 바꾼 것은 없습니다.

참고 논문:

- MOTA, 번호 바뀜: K. Bernardin, R. Stiefelhagen, "Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics", EURASIP Journal on Image and Video Processing, 2008.
- IDF1: E. Ristani, F. Solera, R. Zou, R. Cucchiara, C. Tomasi, "Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking", ECCV Workshops 2016.

프레임 짝 맞추기:

- 재생 카메라(`replay_cam.py --tag`)가 프레임마다 머리에 PNG 번호(파일 이름의 숫자)와 바퀴 번호를 적어 보냅니다. 서버는 그 번호로 정답을 찾습니다. 받은 프레임 수를 세지 않습니다.
- 재생 카메라가 건너뛴 프레임, 서버가 건너뛴 프레임, 번호를 모르는 프레임은 채점하지 않고 수만 보여 줍니다.
- 반복 재생은 바퀴마다 계속 채점합니다. 바퀴 하나를 영상 하나처럼 보고 더합니다 (정답 번호와 서버 번호 모두 바퀴마다 새로).
- 재생 프로그램은 카메라 설정(`LWIR_CAM_STAMP`)과 상관없이 늘 프레임 머리를 붙여 띄웁니다. 진짜 카메라를 띄우는 길은 바뀌지 않았습니다.

알아 둘 것:

- 화면과 결과 파일의 숫자는 CLEAR MOT 짝짓기로 셉니다. 그래서 논문과 py-motmetrics 숫자와 견줄 수 있습니다. 결과 파일의 `per_frame_hungarian` 에는 앞 프레임 짝을 보지 않고 프레임마다 헝가리안만으로 센 숫자도 적습니다 (화면에는 없음). 사람이 겹칠 때 이쪽이 번호 바뀜을 더 셉니다.
- 무거운 계산은 아닙니다 (프레임마다 작은 헝가리안 두세 번). 재생하지 않을 때는 아무것도 하지 않습니다.

<details>
<summary>주소 목록</summary>

**읽기 (GET)**

- `/` — 웹 화면
- `/stream`, `/snap.jpg` — 상자와 번호를 그린 영상, 그림 한 장. 보는 사람이 없으면 그림은 1초에 한 장만 새로 만듭니다.
- `/healthz` — 살아 있는지, 지연 값, 멈춘 그림 수(`frozen_frames`), 만든 그림 수(`pictures`), 보는 사람 수(`viewers`)
- `/model` — 검출 모델 목록과 지금 쓰는 모델
- `/rules` — 지금 설정 (추적기, Filter 값, 새 사람 확인, 외형 모델, 위치 기억, 짐벌 프로그램에 넘긴 값 `gimbal_cfg`)
- `/perf` — 그래프용 초당 기록
- `/measure`, `/measure/<번호>` — 측정 상태, 지난 측정 하나
- `/gimbal/people` — 짐벌 상태와 지금 화면의 사람 상자
- `/replay` — 재생할 수 있는 녹화 목록(이름, 장 수, 8/16 비트)과 재생 상태. 목록은 60초 동안 기억합니다 (`?refresh=1` 은 다시 훑기). `eval` 에 정답과 비교한 숫자
- `/replay/eval` — 정답과 비교한 숫자만 (재생 중이면 지금 것, 끝났으면 마지막 재생의 것)

**바꾸기 (POST, 머리말 `X-Gimbal: 1` 이 있어야 함)**

- `/model/set?id=<번호>` — 검출 모델 바꾸기
- `/rules/set?...` — 추적기, Filter 값(`confirm_ms` 포함), 외형 모델, 위치 기억, 예측 제어(`lead=1`) 등 바꾸기
- `/measure/start`, `/measure/stop`, `/measure/note` — 측정 시작, 끝, 메모
- `/camera/reset` — 카메라 초기화
- `/replay/start?seq=<이름>&fps=<5~30>&loop=<0|1>`, `/replay/stop` — 녹화 재생 시작, 멈춤(카메라로)
- `/replay/eval/reset`, `/replay/eval/draw?on=<0|1>` — 정답과 비교 숫자를 처음부터, 화면 그림에 정답 상자 그리기
- `/system/reboot?confirm=REBOOT` — 장치 재부팅
- `/gimbal/click`, `/gimbal/move`, `/gimbal/stop` — 사람 고르기, 손으로 움직이기, 멈추기
- `/gimbal/ping` — 화면이 살아 있다는 안전 신호. **스크립트에서 보내지 마세요**

**그 밖**

- 측정은 `<측정 폴더>/<날짜>/MEASURE_<시각>_<번호>/` 에 남습니다 (기본 `/mnt/lwir_data/results/live_measure`, 옵션 `--measure-dir`).
- 팀 PC GUI 와는 짐벌 프로그램이 통신합니다: 영상 H.264/RTP → UDP 5000, 추적 상자 → UDP 5001, PC 의 명령 ← TCP 5002.

</details>

## 10. 모델

모델 표와 측정값은 [models/README.md](models/README.md)에 있습니다.

- 시작 모델: 검출 YOLO26m_(FLIR) 640x480, 외형 비교 RepVGG_distillation_(TMOT,CAMEL)_20261103. 둘 다 `.dxnn`(DX-M1 NPU 용 INT8)입니다.
- 검출 모델과 외형 모델은 서버를 켠 채로 화면에서 바꿉니다. 마지막 선택은 `last_selection.json` 에 적혀 다음에 켤 때 되살아납니다.
- `models/detect/` 아래에 새 검출 모델 파일을 넣으면 서버가 스스로 찾아 화면 목록에 올립니다 (입력 640×480 또는 640×640, 분할 출력 6개 또는 `[1,300,6]`).
- 위치 기억(GRU, Mamba) 가중치는 1 epoch 시험용입니다. **쓰지 마세요.**

## 11. 확인한 것과 확인하지 않은 것

**확인한 것**

- **나누기 전 한 파일짜리 서버와 동작이 같음** (2026-11-11) — PC 에서 가짜 카메라·NPU·짐벌로 나란히 돌려 비교, 독립 검사 3번
- **장치에서 설치와 실행이 됨** (2026-11-11) — DX-M1 에서 `setup.sh`, 켜기, 영상, 짐벌 연결, 검출·외형 모델 바꾸기
- **꺼지면 다시 켜짐** (2026-11-11) — 장치에서 서버를 끄고 1분 안에 다시 켜지는 것을 확인
- **단계별 시간, 번갈아 넣기 판정** (2026-11-12) — 장치에서 저장된 실외 녹화 재생, 진짜 NPU (7절)
- **7절 "고친 것"과 멈춘 그림 멈춤이 결과를 바꾸지 않음** (2026-11-12) — PC 가짜 장치로 고치기 전·후 714장 비교, 추적 결과와 짐벌 명령이 똑같음
- **녹화 재생** (2026-11-12) — PC 에서 가짜 카메라·NPU·짐벌로만 확인: 16·8비트 녹화 재생, 재생 중 짐벌 줄에 사람 없음, 누르기·이동 거절, 카메라로 돌아오기, 재생을 안 쓰면 전과 같은 결과. 장치에서는 아직 안 돌려 봄
- **정답과 비교** (2026-11-12) — PC 에서 가짜 카메라·NPU·짐벌로만 확인: 채점 코드가 따로 만든 계산·py-motmetrics 와 같은 숫자, 정답을 아는 합성 녹화에서 화면 숫자 = 저장된 추적 결과로 따로 센 숫자, 건너뛴 프레임이 있어도 PNG 짝이 맞음, 재생을 안 쓰면 전과 같은 결과. 장치와 진짜 검출 모델로는 아직 안 돌려 봄

**확인하지 않은 것**

- 실제 짐벌로 사람을 따라갈 때의 흔들림 (v25, v27 모두)
- 최고 속도 100 에 맞는 제어값
- 7절 "고친 것"이 장치에서 실제로 몇 ms 줄이는지
- 몇 시간 이상 연속 실행, 재부팅 뒤 자동 시작
- 우리 카메라 영상에서의 검출·추적 점수 (정답이 없어 재지 않음)
- 시험용 모델(실내·실외 fine-tuning, 실내 외형 모델)의 장치 정확도
- 녹화 재생을 장치에서 쓰는 것 (진짜 카메라로 돌아올 때 카메라가 다시 열리는지 포함)

**알아 둘 것**

- 카메라가 USB 에서 사라지는 일이 있습니다 (2026-11-11). 선을 다시 꽂아야 합니다. 카메라를 USB3 허브 포트에 꽂아 두면 화면의 카메라 초기화가 전원을 끊어 살릴 수 있습니다.
- 확인 판정기와 번호 잇기 판정기는 공개 열화상 영상으로 학습했습니다. 우리 카메라에서 같은 이득이 난다는 확인은 없습니다.
- 지난 측정표와 옛 구성의 기록은 git 기록의 옛 README 에 있습니다.
