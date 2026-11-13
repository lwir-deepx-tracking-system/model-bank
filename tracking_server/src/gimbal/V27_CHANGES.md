# gimbal_bridge_v27 — 무엇을 왜 고쳤고 어떻게 시험했나 (2026-11-11)

기준은 zip(`LWIR_run_code_as_is_20261111.zip`)의 `gimbal_bridge_v25.cpp`와 `team/`입니다.
아래 11-11 기록 당시에는 `gimbal_bridge_v25.cpp`와 `team/`을 바꾸지 않았습니다. **11-12 예측 제어 수정에서는 공용 `Track` 전달 구조와 PID에 호환 확장을 추가했습니다.** 기존 두 인자 PID 호출과 예측을 끈 제어 경로는 유지합니다.

## 11-12: 관측과 예측을 분리

목표는 선택한 사람을 중앙 부근에 유지하면서 불필요한 시작·정지·왕복을 줄이는 것입니다. 이번 변경은 제어 입력 오류 수정이며, 실제 모터의 최적 이득을 찾았다는 뜻은 아닙니다.

- 이전에는 Python이 예측량만큼 박스를 옮겼습니다. 그 결과를 C++에서 다시 미분해 예측값 변경까지 속도로 계산했고, PC GUI에 전달되는 박스도 옮겨졌습니다.
- 이제 관측 박스는 그대로이며 `F/G`의 끝에 `LEAD id dx dy control_epoch`를 별도로 붙입니다. v27의 `LEAD_CONTROL 1`과 `CONTROL_EPOCH n`을 받은 경우에만 사용합니다. 구형 브리지는 관측만 받습니다.
- P/I에는 예측 오차를 줄 수 있지만 D와 불감대 판단은 실제 관측을 씁니다. 관측 PID가 요청하지 않는 출발·방향 반전은 예측만으로 만들지 않습니다. 두 축이 공유하는 속도 때문에 중앙의 약한 축까지 빨리 움직이는 반례를 막습니다.
- 같은 ID를 다시 선택해도 새로운 제어 세대의 **새 프레임**만 예측에 쓸 수 있습니다. 이전 선택의 전송 중인 예측은 무시합니다.
- 예측은 실제 짝이 맞는 성공한 GMC 결과만 사용합니다. 실패/누락, 10ms 미만 간격, 250ms 이상 공백, 반복·역행 시각에서는 이력을 다시 시작합니다. 속도 필터는 시간상수 120ms, 예측 벡터는 최대 60px입니다. 이 값은 보수적 초기값이며 실기 최적값은 아닙니다.

검증: Windows에서 Python 28개, C++ 관측·예측/실행기/기존 동작 시험 3종 통과. 실제 OpenCV의 합성 연속 9장에서는 추적기에 주는 GMC 행렬이 수정 전과 정확히 같았습니다. Orange Pi의 별도 임시 폴더에서 v27을 빌드하고 가상 직렬 포트로 기존 브리지 17개, 옵션 4그룹, 새 F/G·세대·잘못된 접미사 3그룹을 통과했습니다. **운영 8090과 실제 모터는 변경하지 않았습니다.**

추가 CPU 시험 (`CORE`는 아래 빌드 절과 동일):

```bash
c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub test_v27_lead.cpp $CORE -o test_v27_lead
./test_v27_lead
c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub test_v27_lead_executor.cpp $CORE -o test_v27_lead_executor
./test_v27_lead_executor
python3 test_v27_lead_protocol.py ./gimbal_bridge_v27
```

남은 확인은 실제 영상의 GMC 품질, 센서 시각과 도착 시각 차이, 모터의 최소 속도·지연에 따른 중심 오차와 화면 이동량입니다. 기존 `hil_test/lead_hil.py`는 옛 박스 이동 방식이므로 새 예측 경로의 실증 근거로 쓰지 않습니다. 근거리 박스 흔들림과 머리 위치 오차는 이 제어 수정만으로 없앨 수 없습니다.

## 넣는 파일 (모두 `tracking_server/src/gimbal/` 아래)

- **`gimbal_bridge_v27.cpp`** — v25 사본 + 옵션 5개. v25에서 바뀐 줄은 1줄(executor 생성), 나머지는 추가
- **`v27/include/control/smoothing_config.hpp`** — 새 설정 구조체 (팀의 `app_config.hpp`를 고치지 않으려고 따로 둠)
- **팀 파일 사본, `v27` 표시된 곳만 수정:**
  - `v27/include/control/auto_tracking_controller.hpp`
  - `v27/src/control/auto_tracking_controller.cpp`
- **팀 파일 사본, `v27` 표시된 곳만 수정:**
  - `v27/include/control/gimbal_command_executor.hpp`
  - `v27/src/control/gimbal_command_executor.cpp`
- **`test_v27_control.cpp`** — 제어 단위 시험 (짐벌·브리지 없이)
- **`test_v27_options.py`** — 브리지 옵션 시험 (가상 단말, 기존 `test_gimbal_bridge.py` 옆에 둠)
- **가상 짐벌 폐루프 시뮬레이터, OpenCV 없는 PC용 대체 헤더:**
  - `sim/sim.cpp`
  - `sim/stub/opencv2/core.hpp`

**`v27/` 폴더가 필요한 이유:** 고친 로직은 브리지가 아니라 팀의 `AutoTrackingController`, `GimbalCommandExecutor` 안에 있습니다. 팀의 `TrackingCommandHandler`, `GuiCommandHandler`가 `GimbalCommandExecutor`를 클래스 이름으로 직접 쓰기 때문에 브리지 파일 하나로는 옮길 수 없습니다. 그래서 v27 빌드에서만 고친 사본 4개를 팀 파일 **대신** 쓰도록 했습니다(include 경로에서 `v27/include`를 `team/include`보다 앞에 둠).

## 무엇을 왜

각 옵션의 괄호 안은 기본값입니다. 기본값이면 v25 와 같이 동작합니다.

- **`--far-gain <1..4>` (1), `--far-start-px` (64), `--far-full-px` (160), `--brake-px-per-spd <px>` (0)** — 관측 오차(640 폭 픽셀)가 far-start에서 far-full로 커지는 동안 속도 요구를 far-gain배까지 키우고, brake가 0보다 크면 속도를 (|오차| − 불감대)/brake 이하로 묶습니다. 기본은 둘 다 꺼짐(팀 규칙 그대로). 실제 짐벌 + 가상 표적(11-13): brake 5는 도착 전에 너무 일찍 느려져 100 px 이동의 복귀가 늦어짐, `--speed-rate 150 --far-gain 2`는 지나침 없이 따라가는 동안의 지연이 조금 줄었으나 시험 도구가 자주 중단되어 비교가 부족함. (2026-11-13)
- **`--aim-screen-y <0.1..0.9>` (0.5)** — 조준점을 화면 높이의 이 비율(위에서부터)에 둡니다. 0.5는 가운데(예전과 같음), 0.333은 머리를 화면 위 1/3에 둡니다. 가로는 항상 가운데입니다. (2026-11-13)
- **`--deadband-exit <e>` (0)**
  - 바꾼 것: 불감대 히스테리시스: 불감대에 들어온 축은 오차가 `e`를 넘어야 다시 움직임
  - 이유: 목표가 불감대 경계에 있으면 출발·정지를 되풀이함
- **`--axis-reversal <0|1>` (0)**
  - 바꾼 것: 한 축이 방향을 바꾸면 그 축만 멈추고 다른 축은 계속 움직임. 속도는 움직이는 축의 요구만으로 정함
  - 이유: 두 축이 속도 하나를 공유해서 오차가 작은 축이 먼저 지나치고, 그때 v25는 두 축을 모두 멈춤 (가로 오차가 크게 남아 있어도)
- **`--reversal-join-speed <SPD>` (5)**
  - 바꾼 것: 멈춘 축은 공유 속도가 이 값 이하일 때 다시 움직임
  - 이유: 높은 속도로 갑자기 출발하지 않게
- **`--reversal-hold-ms <ms>` (500)**
  - 바꾼 것: 멈춘 축은 늦어도 이 시간 뒤에는 다시 움직임
  - 이유: 다른 축이 오래 빠르게 움직이면 무기한 기다리게 되는 것을 막음 (단위 시험에서 발견)
- **`--stop-rate <SPD/s>` (0)**
  - 바꾼 것: 불감대에 들어갈 때 즉시 STOP 대신 이 비율로 감속한 뒤 STOP
  - 이유: 높은 속도에서 불감대에 들어가면 화면이 툭 멈춤

목표 놓침, 데이터 지연, 생존 신호 끊김, 장치 오류, 수동 조작에서의 정지는 v25와 같이 **즉시** 나갑니다.

## 빌드 (`tracking_server/src/gimbal/` 에서)

브리지 (장치, OpenCV + GStreamer 필요). `-Iv27/include`가 반드시 앞에 오고, 팀의 `auto_tracking_controller.cpp`, `gimbal_command_executor.cpp`는 넣지 않습니다.

```bash
g++ -O2 -std=c++17 -pthread -Iv27/include -Iteam/include $(pkg-config --cflags opencv4 gstreamer-1.0 gstreamer-app-1.0) gimbal_bridge_v27.cpp v27/src/control/auto_tracking_controller.cpp v27/src/control/gimbal_command_executor.cpp team/src/control/gimbal_device.cpp team/src/control/pid_controller.cpp team/src/control/tracking_command_handler.cpp team/src/gui/gui_protocol.cpp team/src/gui/gui_sender.cpp team/src/gui/gui_receiver.cpp team/src/gui/gui_command_handler.cpp team/src/common/logger.cpp $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0) -lopencv_core -lopencv_imgproc -o gimbal_bridge_v27
```

단위 시험과 시뮬레이터:

- OpenCV가 없는 PC에서는 `-Isim/stub`을 붙입니다.
- 장치에서는 빼고 `$(pkg-config --cflags opencv4)`와 `-lopencv_core`를 붙입니다.

```bash
CORE="v27/src/control/auto_tracking_controller.cpp v27/src/control/gimbal_command_executor.cpp team/src/control/pid_controller.cpp team/src/common/logger.cpp team/src/pipeline/runtime_control.cpp"
```
```bash
c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub test_v27_control.cpp $CORE -o test_v27_control && ./test_v27_control
```
```bash
c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub sim/sim.cpp $CORE -o sim_v27 && ./sim_v27 200 0.10 9 1
```

브리지 시험 (장치에서, 빌드한 뒤):

```bash
python3 test_gimbal_bridge.py ./gimbal_bridge_v27
```
```bash
python3 test_v27_options.py ./gimbal_bridge_v27
```

켤 때 예:

- `--gimbal-bridge <gimbal_bridge_v27 경로>`
- `--gimbal-opts` 에 `--deadband-exit 0.10 --axis-reversal 1` 추가.

## 어떻게 시험했나

PC(macOS, clang, OpenCV 없음 → `sim/stub`)에서 한 것:

- **단위 시험 `test_v27_control`: 9개 모두 통과**, 3번 반복해도 같음. 기본값이 팀 규칙 그대로인지 3개(0.08에서 다시 움직임, 반전 시 두 축 STOP, 불감대 즉시 STOP)와 새 동작 6개.
- **ASan, TSan**으로 같은 시험: 오류·경고 0.
- **문법 검사**: `gimbal_bridge_v27.cpp`와, v27 헤더를 앞에 둔 팀의 `tracking_command_handler.cpp`, `gui_command_handler.cpp`를 대체 헤더로 `-fsyntax-only -Wall -Wextra` → 오류·경고 0 (v25도 같은 조건에서 0).
- **시뮬레이션 `sim_v27`** (실제 executor + 가상 짐벌, 9초 × 2회).
  - 가정: SPD 1 = 0.5 °/s, 30 fps, 지연 120 ms, Kp 30 / Kd 6 / 불감대 0.06 / 상한 100.
  - 목표: 오른쪽 15°·위 5°에서 시작, 3~6초에 4 °/s 이동.

시뮬레이션 결과:

- **v25 규칙 (옵션 없음)**
  - 출발·정지 반복 7 / 7
  - 가장 큰 순간 속도 변화 (°/s) 6.0 / 6.0
  - 평균 오차 4.3°
- **`--deadband-exit 0.10`**
  - 출발·정지 반복 2 / 3
  - 가장 큰 순간 속도 변화 (°/s) 6.0 / 6.0
  - 평균 오차 4.3~4.4°
- **`--axis-reversal 1`**
  - 출발·정지 반복 6 / 7
  - 가장 큰 순간 속도 변화 (°/s) 1.0 / 3.5
  - 평균 오차 4.1~4.3°
- **셋 다 (`--stop-rate 200` 포함)**
  - 출발·정지 반복 1 / 1
  - 가장 큰 순간 속도 변화 (°/s) 1.0 / 1.0
  - 평균 오차 4.2°

## 11-12 정상 영상의 한 프레임 관측 누락

`--observation-grace-ms 100`을 주면 선택한 사람의 **발행된 추적 행**이 한 프레임 빠졌을 때 마지막 이동 명령을 잠깐 유지합니다. 기본 0은 기존 즉시 정지입니다. 검출기 자체의 누락과 새 ID 확인·필터에서 빠진 행을 이 기능만으로 구별하지는 않습니다.

- 실제 입력 시각과 RAW16이 짝을 이룬 정상 라이브 결과만 `OBS 1`을 받습니다. Python과 새 v27을 함께 갱신해야 합니다. 새 브리지는 `OBSERVATION_STATUS 1`로 이를 알립니다.
- 이미 움직이며 같은 ID를 연속 관측한 경우에만 허용합니다. 기한은 직전 관측 간격의 1.5배와 설정 상한 중 짧은 값이고, 기존 입력 만료 시간도 넘지 않습니다. 22fps이면 약 68ms이며 무조건 100ms 기다리는 설정이 아닙니다.
- 유예 중에는 마지막 명령을 그대로 유지하며 가속·반전·다른 ID 선택을 하지 않습니다. 다음 연속 프레임에 같은 ID가 돌아오면 새 관측으로 즉시 계산하고 이전 예측은 버립니다. 다른 사람만 보여도 그 사람 좌표로 전환하지 않습니다.
- 두 번째 누락, 영상 동결·오래된 입력·재생·카메라 초기화·STOP·연결 끊김에는 정지합니다. 중간 프레임이 전송 슬롯에서 사라져 확인할 수 없을 때도 정지 후 새 관측으로 계산합니다.
- 무효 입력 표지는 Python과 C++의 최신 프레임 슬롯에서 덮어써지지 않도록 별도로 보존합니다. `/healthz`의 `observation_status`, `observation_healthy`, `observation_good`, `observation_bad`로 실제 경로를 확인할 수 있습니다.

PC Python 37개와 C++ 새 검사 26개·기존 검사 40개를 통과했습니다. Orange Pi 별도 빌드와 F/G 가상 직렬 포트 검사도 통과했습니다. 기존 유예 0의 동작 검사 3종을 함께 통과했습니다. 이후 8090에 적용하고 저속 가상 표적의 실제 짐벌 진단을 수행했습니다. [실기 기록과 남은 범위](hil_test/OBSERVATION_GAP_20261112_KO.md). 실제 사람 급이동의 화면 안정성 증명은 아닙니다.

```bash
python3 test_v27_observation_grace.py ./gimbal_bridge_v27
```

장치의 `scripts/live.env`에서 기존 `LWIR_GIMBAL_EXTRA_OPTS` 값에 `--observation-grace-ms 100`을 더합니다. 기존 옵션을 지우지 않습니다. 다시 시작한 뒤 `/rules`의 `gimbal_cfg.extra`와 `/healthz`의 capability를 함께 확인합니다. 되돌리려면 값을 0으로 바꿔 다시 시작합니다.

## 최초 작성 시 하지 않은 것, 한계

- **브리지 전체 빌드와 `test_v27_options.py`, `test_gimbal_bridge.py`는 돌리지 못했습니다** (PC에 OpenCV·GStreamer 없음). 장치에서 빌드한 뒤 두 시험을 먼저 돌려야 합니다.
- **실제 짐벌에서 시험하지 않았습니다.** 시뮬레이션의 SPD 환산값과 지연은 가정이고 관성·백래시가 없습니다. 위 결과의 수치는 경향만 봅니다.
- `--axis-reversal 1` 단독은 2회 중 1회에서 효과가 작았습니다(대기 시간이 끝나 높은 속도로 다시 합류). `--deadband-exit`와 같이 쓰는 것을 권합니다.
- `--stop-rate`는 지금 이득(Kp 30)에서는 불감대 진입 속도가 이미 1~2 SPD라 효과가 거의 없습니다.
- 기본값은 경험값입니다(이론적 최적화나 안정성 증명 없음).
- 장치 주소, 폴더 경로, 짐벌 장치 번호는 어느 파일에도 넣지 않았습니다. 장치는 `--device` 옵션으로 받습니다(v25와 같음).
- 장치의 v26(`--slow-pulse`)은 포함하지 않았습니다. v27은 v25에서 갈라진 것입니다.
