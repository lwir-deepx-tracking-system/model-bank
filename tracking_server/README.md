# 실시간 추적 서버 (`tracking_server/`, 포트 8090)

열화상 카메라 영상에서 **사람을 찾고, 같은 사람에게 같은 번호를 유지하고, 고른 사람을 짐벌로 따라가는** 서버입니다. 장치는 DX-M1(Orange Pi 5 Plus) 한 대입니다.

![파이프라인과 실시간 처리](images/pipeline_realtime_20261111.png)

> 그림 읽는 법: 숫자는 2026-11-11 에 장치에서 짧게 읽은 값입니다. 가운데 막대의 길이는 이해를 돕기 위한 그림이고 단계별 시간은 아직 재지 않았습니다. 검출을 NPU 코어 0·1 에 번갈아 넣는 방식은 만들어 시험하는 중이며, 지금 기본은 코어 0 하나입니다. 사진의 상자와 번호는 검수한 라벨로 그린 예시입니다.

## 1. 한눈에 보기

| 단계 | 하는 일 | 어디서 | 코드 |
|---|---|---|---|
| 1. 영상 받기 | 카메라에서 16비트 640×480 영상을 받음 | CPU | `src/camera/` |
| 2. 검출 | YOLO26m 으로 사람 상자를 찾음 | NPU 코어 0 | `src/detect/` |
| 3. 검출 뒤 Filter | 온도 범위 밖인 상자, 화면 끝의 점수 낮은 상자를 버림 | CPU | `src/track/` |
| 4. 추적 | 앞 장의 사람과 짝지어 번호를 유지 (BoT-SORT, Kalman filter) | CPU | `src/track/` |
| 5. 다시 알아보기 | 외형 특징으로 같은 사람인지 확인, 돌아온 사람에게 옛 번호 (KeepID) | NPU 코어 2 + CPU | `src/reid/` |
| 6. 조준 | 고른 사람의 머리를 화면 가운데로: 150 ms 앞을 내다보고 PID 로 짐벌 속도 명령 | CPU | `src/gimbal/` |

장치에서 읽은 값(2026-11-11): 장 사이 약 45 ms(초당 22~24장), 한 장 계산 합계 약 58 ms, 영상 도착부터 결과까지 중간값 약 75 ms.

## 2. 빠른 시작

```bash
git clone https://github.com/lwir-deepx-tracking-system/model-bank.git
bash model-bank/tracking_server/scripts/setup.sh         # 한 번만: 빌드, 모델 연결과 확인
bash model-bank/tracking_server/scripts/run_real_gui.sh  # 켜기
```

브라우저에서 `http://<장치 주소>:8090/` 을 엽니다.

- **화면에는 비밀번호가 없고 짐벌을 움직입니다.** 믿을 수 있는 기기만 닿는 주소에서 쓰세요.
- `setup.sh` 끝에 `setup done` 이 나오면 준비가 된 것입니다.
- 장치마다 다른 값(주소, 짐벌 장치 등)은 `scripts/live.env.example` 을 `scripts/live.env` 로 복사해 고칩니다(4절).

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
| 7 | 짐벌 프로그램이 없음 (`scripts/setup.sh` 다시) |
| 8 | 페이지 주소 없음 (`scripts/live.env` 에 `LWIR_LIVE_BIND`) |
| 9 | 폴더 이동 실패 |

</details>

## 3. 폴더

```
tracking_server/
├── start_tracking_server.py   서버를 켜는 파일
├── models/                    모델 파일
│   ├── detect/                검출 모델 (yolo26m, yolo26s, yolo26x, yolov8, yolox)
│   ├── track/                 추적용 파일 (위치 기억, 확인 판정기, 번호 잇기 판정기)
│   └── reid/                  외형 비교 모델 (RepVGG, ViT-lite)
├── src/                       서버 코드
│   ├── app.py                 본체: 옵션 → 준비 → 프레임 반복 → 결과
│   ├── detect/                검출
│   ├── track/                 추적
│   ├── reid/                  외형 비교
│   ├── camera/                카메라 프로그램과 읽는 코드
│   ├── gimbal/                짐벌 제어 프로그램, 팀 제어 코드 사본, 시험
│   ├── web/                   8090 요청 처리, 화면 설정, 측정 저장
│   └── common/                같이 쓰는 값
├── page/                      8090 화면 (한 파일 + 소스 조각 + 합치는 도구)
├── scripts/                   준비, 켜기, 다시 켜기
└── images/                    이 글의 그림
```

같은 일은 어디서든 같은 이름입니다: `detect`(검출), `track`(추적), `reid`(외형 비교).

<details>
<summary>폴더별 파일</summary>

| 폴더 | 파일 |
|---|---|
| `src/detect/` | `npu_engine.py` NPU 에 모델 올리기·돌리기, `detect_frame.py` 프레임 → 상자, `model_scan.py` 검출 모델 파일 스스로 찾기, `zoom_redetect.py` 작은 사람 확대 재검출, `native_post.py` + `post.cpp` 검출기 출력을 푸는 빠른 계산 |
| `src/track/` | `trackers.py` 추적기, `kalman.py` Kalman filter, `camera_motion.py` 카메라 움직임 추정, `position_memory.py` 놓친 사람 위치 기억, `gate.py` 확인 판정기, `confirm_time.py` 새 사람 확인을 시간(ms)으로 계산, `stitch.py` 끊긴 추적 잇기, `stabilizer.py` 화면 흔들림 보정 |
| `src/reid/` | `reid_models.py` 외형 모델 목록과 실행 중 바꾸기, `reid_embed.py` 사람 조각에서 특징 뽑기, `relabel_online.py` KeepID 규칙, `relink_online_judge.py` 번호 잇기 판정기 |
| `src/camera/` | `cam_pipe.cpp` 카메라 프로그램, `camera_link.py` 그 프로그램을 띄우고 읽기·카메라 초기화 |
| `src/gimbal/` | `gimbal_bridge_v25.cpp` 짐벌 + PC GUI 프로그램, `team/` 팀 제어·GUI 코드 사본(출처 `team/SOURCE.txt`), `gimbal_link.py` 그 프로그램을 띄우고 명령 보내기, `test_*.py` 시험, `PID_TUNING_RECORD_20261110.md` 제어값 기록, `gimbal_bridge_v27.cpp` + `v27/` + `sim/` 시험용 판(4절) |
| `src/web/` | `page_handler.py` 주소별 처리, `page_settings.py` 화면에서 바꾸는 설정·모델 바꾸기, `measure.py` 측정 버튼, `system_buttons.py` 재부팅 버튼 |
| `src/common/` | `shared_values.py` 고정 값, `state.py` 공유 값 보관함, `settings.py` 실행 중 바뀌는 설정, `result_summary.py` 결과 요약, `device_wait.py` 다른 측정 기다리기, `pc_dry_run.py`·`research_*.py` PC 에서 돌려 보기와 연구용 기능 |
| `page/` | `live_page.html` 서버가 읽는 한 파일, `html/` `css/` `js/` 조각, `build_page.py` 합치는 도구, `parts.json` 조각 순서 |
| `scripts/` | `setup.sh` 준비, `run_real_gui.sh` 켜기, `keepalive.sh` 꺼지면 다시 켜기, `live.env.example` 장치별 설정 예, `gui_model_client.py` PC GUI 명령 시험 |

</details>

## 4. 설정 (`scripts/live.env`)

| 값 | 뜻 | 기본 |
|---|---|---|
| `LWIR_LIVE_BIND` | 웹 화면 주소 | 장치의 Tailscale 주소 |
| `LWIR_LIVE_PORT` | 웹 화면 포트 | 8090 |
| `LWIR_GUI_HOST` | 팀 PC GUI 주소 | 192.168.50.1 |
| `LWIR_GIMBAL_DEV` | 짐벌 직렬 장치 | 고정 이름을 먼저 찾고, 없으면 `/dev/ttyUSB0` |
| `LWIR_GIMBAL_AIM_Y` | 사람 상자의 어느 높이를 화면 가운데에 둘지 (위에서부터의 비율) | 0.08 = 머리 (0.3 = 가슴) |
| `LWIR_GIMBAL_SPEED_MAX` | 자동으로 따라갈 때의 최고 속도 | 100 |
| `LWIR_GIMBAL_REACQUIRE_MS` | 놓친 사람을 기다리는 시간 (ms) | 10000 |
| `LWIR_CONFIRM_MS` | 새 사람을 화면에 내기 전에 확인하는 시간 (ms). 0 = 끔 (늘 3장) | 130 |
| `LWIR_GIMBAL_BRIDGE` | 짐벌 프로그램 | `src/gimbal/gimbal_bridge_v25` |
| `LWIR_GIMBAL_EXTRA_OPTS` | 짐벌 프로그램에 더 줄 옵션 | 없음 |

- 짐벌 제어값: PD 제어(Kp 30, Ki 0, Kd 6), 불감대 0.06, 손으로 움직일 때 속도 20. 이 값은 최고 속도 10 일 때 맞춘 것입니다(`src/gimbal/PID_TUNING_RECORD_20261110.md`). 최고 속도 100 에 맞춘 값은 아직 없습니다.
- `gimbal_bridge_v27` 은 움직임을 부드럽게 하려는 **시험용 판**입니다. `src/gimbal/V27_CHANGES.md` 의 명령으로 먼저 빌드하고, 옵션 예는 `--deadband-exit 0.10 --axis-reversal 1` 입니다. 실제 짐벌에서 흔들림이 줄어드는지는 아직 재지 않았습니다.
- `LWIR_GIMBAL_AIM_Y` 를 바꾸면 화면의 세로 오차 표시 기준(`page/js/04_ping_fps_aim.js` 의 `AIM_Y`)도 같이 바꿉니다.
- `LWIR_CONFIRM_MS` 는 "새 사람 확인"을 장 수가 아니라 시간으로 적은 값입니다. 서버가 초당 장 수를 재서 장 수를 다시 계산합니다: 장 수 = 130 × 초당 장 수 ÷ 1000 을 반올림. 초당 22~27장이면 3장(전과 같음), 초당 28장을 넘으면 4장이 됩니다. 느려져도 3장보다 작아지지는 않습니다. 지금 값은 화면의 "지금 적용된 값"에 나옵니다. 코드는 `src/track/confirm_time.py`, PC 시험은 `python -m unittest src.track.test_confirm_time` 입니다. 장치에서는 아직 돌려 보지 않았습니다.

## 5. 거르는 곳

영상 한 장에서 상자가 버려지는 곳을 순서대로 적었습니다.

| 번호 | 이름 | 무엇을 버리나 | 어디서 바꾸나 |
|---|---|---|---|
| ① | 점수 · 겹침 | 점수 낮은 상자, 같은 사람에 겹친 상자 | `scripts/run_real_gui.sh` |
| ② | 온도 Filter | 사람 상자 안의 16비트 원본 값(상위 5% 지점)이 정한 범위 밖인 것 | 화면 "Filter 값" |
| ③ | 종류 | 사람과 차를 서로 짝짓는 것 | `scripts/run_real_gui.sh` |
| ④ | 가장자리 Filter | 화면 끝에 닿은 상자 중 점수가 낮은 것은 새 번호를 받지 못함 | 화면 "Filter 값" |
| ⑤ | 새 사람 확인 (130 ms) | 130 ms 동안(초당 22~27장이면 3장) 이어서 잡히지 않은 것은 화면에 나오지 않음 | `scripts/live.env` 의 `LWIR_CONFIRM_MS` |
| ⑥ | 불감대 | 조준점이 가운데 가까이 있으면 짐벌이 움직이지 않음 | `scripts/live.env` |

## 6. 꺼지면 다시 켜기, 끄기

crontab 에 한 줄을 넣으면 서버가 꺼졌을 때 스스로 다시 켜집니다(`<클론 경로>` 는 저장소를 받은 곳).

```
* * * * * flock -n $HOME/.lwir_keepalive.lock bash <클론 경로>/tracking_server/scripts/keepalive.sh >> $HOME/lwir_keepalive.log 2>&1
```

| 하고 싶은 일 | 방법 |
|---|---|
| 다시 켜기를 잠시 멈춤 | `touch ~/lwir_keepalive.pause` |
| 다시 돌림 | `rm ~/lwir_keepalive.pause` |
| 서버 끄기 | 위처럼 멈춘 뒤 `kill -TERM $(pgrep -f 'python3 -B [s]tart_tracking_server.py')` |
| 기록 보기 | `~/lwir_keepalive.log`, 꺼질 때의 마지막 메시지는 `~/lwir_crash.log` |

- 카메라 프로그램(`cam_pipe`)은 서버가 끝나면 스스로 끝납니다. `kill -9` 로 끄지 마세요(카메라가 다음에 안 열릴 수 있습니다).
- **NPU 측정 프로그램을 돌리기 전에는 위 방법으로 서버를 먼저 꺼 주세요.** 같이 돌면 NPU 서비스가 죽어 서버가 꺼집니다. 측정 프로그램(`lwir_benchmark`)이 도는 동안에는 다시 켜기가 기다립니다.

<details>
<summary>다시 켜기 규칙 자세히</summary>

- 5초마다 `/healthz` 를 읽습니다. 서버 프로세스가 없거나, 30초 동안 답이 없거나, 프레임은 오는데 30초 동안 결과가 안 나오거나, 짐벌 프로그램이 3번 연속 없으면 다시 켭니다. 시작 뒤 180초 동안은 멈춤 검사를 하지 않습니다.
- 다시 켤 때 이전 `real_gui_1/` 을 `real_gui_prev/` 로 옮겨 둡니다. 손으로 켜면 이전 `real_gui_1/` 을 지웁니다.
- 10분에 3번 다시 켜면 10분 쉽니다.
- 장치 재부팅 단계는 `~/lwir_keepalive.allow_reboot` 파일이 있을 때만 켜집니다(카메라 프로그램이 3번 연속 실패, 15분 동안 프레임 없음, 켜진 지 30분 넘음, 하루 2번까지, 60분 간격). **카메라가 USB 에서 아예 보이지 않으면 재부팅하지 않습니다**(재부팅으로는 살아나지 않고, 선을 다시 꽂아야 합니다).

</details>

## 7. 웹 화면

화면에서 할 수 있는 것: 영상 보기, 사람을 눌러 따라가기, 손으로 짐벌 움직이기, 검출 모델·추적기·외형 모델·위치 기억 고르기, Filter 값 바꾸기, 측정 저장, 카메라 초기화, 장치 재부팅.

화면을 고칠 때는 `page/live_page.html` 을 손으로 고치지 않습니다.

1. `page/html/`, `page/css/`, `page/js/` 에서 조각을 고칩니다(어느 조각에 무엇이 있는지는 [page/README.md](page/README.md)).
2. `page/` 에서 `python3 build_page.py` 로 다시 만들고, `python3 build_page.py --check` 로 확인합니다.
3. 조각과 `live_page.html` 을 함께 올립니다. 서버를 다시 켤 필요는 없습니다.

<details>
<summary>주소 목록</summary>

읽기 (GET):

| 주소 | 주는 것 |
|---|---|
| `/` | 웹 화면 |
| `/stream`, `/snap.jpg` | 상자와 번호를 그린 영상, 그림 한 장 |
| `/healthz` | 살아 있는지와 지연 값 |
| `/model` | 검출 모델 목록과 지금 쓰는 모델 |
| `/rules` | 지금 설정 (추적기, Filter 값, 외형 모델, 위치 기억, 짐벌 프로그램에 넘긴 값 `gimbal_cfg`) |
| `/perf` | 그래프용 초당 기록 |
| `/measure`, `/measure/<번호>` | 측정 상태, 지난 측정 하나 |
| `/gimbal/people` | 짐벌 상태와 지금 화면의 사람 상자 |

바꾸기 (POST, 머리말 `X-Gimbal: 1` 이 있어야 함):

| 주소 | 하는 일 |
|---|---|
| `/model/set?id=<번호>` | 검출 모델 바꾸기 |
| `/rules/set?...` | 추적기, Filter 값, 외형 모델, 위치 기억 등 바꾸기 |
| `/measure/start`, `/measure/stop`, `/measure/note` | 측정 시작, 끝, 메모 |
| `/camera/reset` | 카메라 초기화 |
| `/system/reboot?confirm=REBOOT` | 장치 재부팅 |
| `/gimbal/click`, `/gimbal/move`, `/gimbal/stop` | 사람 고르기, 손으로 움직이기, 멈추기 |
| `/gimbal/ping` | 화면이 살아 있다는 안전 신호. **스크립트에서 보내지 마세요** |

측정은 `<측정 폴더>/<날짜>/MEASURE_<시각>_<번호>/` 에 남습니다(기본 `/mnt/lwir_data/results/live_measure`, 옵션 `--measure-dir`).
팀 PC GUI 와는 짐벌 프로그램이 통신합니다: 영상 H.264/RTP → UDP 5000, 추적 상자 → UDP 5001, PC 의 명령 ← TCP 5002.

</details>

## 8. 모델

모델 표와 측정값은 [models/README.md](models/README.md)에 있습니다.

- 켜기 스크립트의 시작 모델: 검출 YOLO26m_(FLIR) 640x480, 외형 비교 RepVGG_distillation_(TMOT,CAMEL)_20261103.
- 검출 모델과 외형 모델은 서버를 켠 채로 화면에서 바꿉니다. 마지막 선택은 `last_selection.json` 에 적혀 다음에 켤 때 되살아납니다.
- `models/detect/` 아래에 새 검출 모델 파일을 넣으면 서버가 스스로 찾아 화면 목록에 올립니다(입력 640×480 또는 640×640, 분할 출력 6개 또는 `[1,300,6]`).
- 위치 기억(GRU, Mamba) 가중치는 1 epoch 시험용입니다. **쓰지 마세요.**

## 9. 확인한 것과 확인하지 않은 것

| 확인한 것 | 날짜 | 방법 |
|---|---|---|
| 나누기 전 한 파일짜리 서버와 동작이 같음 | 2026-11-11 | PC 에서 가짜 카메라·NPU·짐벌로 나란히 돌려 비교 (독립 검사 3번) |
| 장치에서 설치와 실행이 됨 | 2026-11-11 | DX-M1 에서 `setup.sh`, 켜기, 영상, 짐벌 연결, 검출·외형 모델 바꾸기 |
| 꺼지면 다시 켜짐 | 2026-11-11 | 장치에서 서버를 끄고 1분 안에 다시 켜지는 것을 확인 |

| 확인하지 않은 것 |
|---|
| 실제 짐벌로 사람을 따라갈 때의 흔들림 (v25, v27 모두) |
| 최고 속도 100 에 맞는 제어값 |
| 몇 시간 이상 연속 실행, 재부팅 뒤 자동 시작 |
| 우리 카메라 영상에서의 검출·추적 점수 (정답이 없어 재지 않음) |
| 시험용 모델(실내·실외 fine-tuning, 실내 외형 모델)의 장치 정확도 |
| 단계별 처리 시간 |

알아 둘 것

- 카메라가 USB 에서 사라지는 일이 있습니다(2026-11-11). 그때는 선을 다시 꽂아야 합니다. 카메라를 전원을 끊을 수 있는 포트에 꽂아 두면 화면의 카메라 초기화로 살릴 수 있습니다.
- 확인 판정기와 번호 잇기 판정기는 공개 열화상 영상으로 학습했습니다. 우리 카메라에서 같은 이득이 난다는 확인은 없습니다.
- 지난 측정표와 옛 구성의 기록은 git 기록의 옛 README 에 있습니다.
