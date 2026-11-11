# gimbal_bridge_v27 — 무엇을 왜 고쳤고 어떻게 시험했나 (2026-11-11)

기준은 zip(`LWIR_run_code_as_is_20261111.zip`)의 `gimbal_bridge_v25.cpp`와 `team/`입니다.
**`gimbal_bridge_v25.cpp`와 `team/`은 한 글자도 바꾸지 않았습니다.** 새 옵션을 주지 않으면 v27은 v25와 똑같이 동작합니다.

## 넣는 파일 (모두 `tracking_server/src/gimbal/` 아래)

| 파일 | 내용 |
|---|---|
| `gimbal_bridge_v27.cpp` | v25 사본 + 옵션 5개. v25에서 바뀐 줄은 1줄(executor 생성), 나머지는 추가 |
| `v27/include/control/smoothing_config.hpp` | 새 설정 구조체 (팀의 `app_config.hpp`를 고치지 않으려고 따로 둠) |
| `v27/include/control/auto_tracking_controller.hpp`, `v27/src/control/auto_tracking_controller.cpp` | 팀 파일 사본, `v27` 표시된 곳만 수정 |
| `v27/include/control/gimbal_command_executor.hpp`, `v27/src/control/gimbal_command_executor.cpp` | 팀 파일 사본, `v27` 표시된 곳만 수정 |
| `tests/test_v27_control.cpp` | 제어 단위 시험 (짐벌·브리지 없이) |
| `tests/test_v27_options.py` | 브리지 옵션 시험 (가상 단말, 기존 `test_gimbal_bridge.py` 옆에 둠) |
| `sim/sim.cpp`, `sim/stub/opencv2/core.hpp` | 가상 짐벌 폐루프 시뮬레이터, OpenCV 없는 PC용 대체 헤더 |

**`v27/` 폴더가 필요한 이유:** 고친 로직은 브리지가 아니라 팀의 `AutoTrackingController`, `GimbalCommandExecutor` 안에 있습니다. 팀의 `TrackingCommandHandler`, `GuiCommandHandler`가 `GimbalCommandExecutor`를 클래스 이름으로 직접 쓰기 때문에 브리지 파일 하나로는 옮길 수 없습니다. 그래서 v27 빌드에서만 고친 사본 4개를 팀 파일 **대신** 쓰도록 했습니다(include 경로에서 `v27/include`를 `team/include`보다 앞에 둠).

## 무엇을 왜

| 옵션 (기본값 = v25 동작) | 바꾼 것 | 이유 |
|---|---|---|
| `--deadband-exit <e>` (0) | 불감대 히스테리시스: 불감대에 들어온 축은 오차가 `e`를 넘어야 다시 움직임 | 목표가 불감대 경계에 있으면 출발·정지를 되풀이함 |
| `--axis-reversal <0\|1>` (0) | 한 축이 방향을 바꾸면 그 축만 멈추고 다른 축은 계속 움직임. 속도는 움직이는 축의 요구만으로 정함 | 두 축이 속도 하나를 공유해서 오차가 작은 축이 먼저 지나치고, 그때 v25는 두 축을 모두 멈춤 (가로 오차가 크게 남아 있어도) |
| `--reversal-join-speed <SPD>` (5) | 멈춘 축은 공유 속도가 이 값 이하일 때 다시 움직임 | 높은 속도로 갑자기 출발하지 않게 |
| `--reversal-hold-ms <ms>` (500) | 멈춘 축은 늦어도 이 시간 뒤에는 다시 움직임 | 다른 축이 오래 빠르게 움직이면 무기한 기다리게 되는 것을 막음 (단위 시험에서 발견) |
| `--stop-rate <SPD/s>` (0) | 불감대에 들어갈 때 즉시 STOP 대신 이 비율로 감속한 뒤 STOP | 높은 속도에서 불감대에 들어가면 화면이 툭 멈춤 |

목표 놓침, 데이터 지연, 생존 신호 끊김, 장치 오류, 수동 조작에서의 정지는 v25와 같이 **즉시** 나갑니다.

## 빌드 (`tracking_server/src/gimbal/` 에서)

브리지 (장치, OpenCV + GStreamer 필요). `-Iv27/include`가 반드시 앞에 오고, 팀의 `auto_tracking_controller.cpp`, `gimbal_command_executor.cpp`는 넣지 않습니다.

```bash
g++ -O2 -std=c++17 -pthread -Iv27/include -Iteam/include $(pkg-config --cflags opencv4 gstreamer-1.0 gstreamer-app-1.0) gimbal_bridge_v27.cpp v27/src/control/auto_tracking_controller.cpp v27/src/control/gimbal_command_executor.cpp team/src/control/gimbal_device.cpp team/src/control/pid_controller.cpp team/src/control/tracking_command_handler.cpp team/src/gui/gui_protocol.cpp team/src/gui/gui_sender.cpp team/src/gui/gui_receiver.cpp team/src/gui/gui_command_handler.cpp team/src/common/logger.cpp $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0) -lopencv_core -lopencv_imgproc -o gimbal_bridge_v27
```

단위 시험과 시뮬레이터 (OpenCV가 없는 PC에서는 `-Isim/stub`을 붙이고, 장치에서는 빼고 `$(pkg-config --cflags opencv4)`와 `-lopencv_core`를 붙임):

```bash
CORE="v27/src/control/auto_tracking_controller.cpp v27/src/control/gimbal_command_executor.cpp team/src/control/pid_controller.cpp team/src/common/logger.cpp team/src/pipeline/runtime_control.cpp"
```
```bash
c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub tests/test_v27_control.cpp $CORE -o test_v27_control && ./test_v27_control
```
```bash
c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub sim/sim.cpp $CORE -o sim_v27 && ./sim_v27 200 0.10 9 1
```

브리지 시험 (장치에서, 빌드한 뒤):

```bash
python3 tests/test_gimbal_bridge.py ./gimbal_bridge_v27
```
```bash
python3 tests/test_v27_options.py ./gimbal_bridge_v27
```

켤 때 예: `--gimbal-bridge <gimbal_bridge_v27 경로>` 와 `--gimbal-opts` 에 `--deadband-exit 0.10 --axis-reversal 1` 추가.

## 어떻게 시험했나

PC(macOS, clang, OpenCV 없음 → `sim/stub`)에서 한 것:

- **단위 시험 `test_v27_control`: 9개 모두 통과**, 3번 반복해도 같음. 기본값이 팀 규칙 그대로인지 3개(0.08에서 다시 움직임, 반전 시 두 축 STOP, 불감대 즉시 STOP)와 새 동작 6개.
- **ASan, TSan**으로 같은 시험: 오류·경고 0.
- **문법 검사**: `gimbal_bridge_v27.cpp`와, v27 헤더를 앞에 둔 팀의 `tracking_command_handler.cpp`, `gui_command_handler.cpp`를 대체 헤더로 `-fsyntax-only -Wall -Wextra` → 오류·경고 0 (v25도 같은 조건에서 0).
- **시뮬레이션 `sim_v27`** (실제 executor + 가상 짐벌, 9초 × 2회). 가정: SPD 1 = 0.5 °/s, 30 fps, 지연 120 ms, Kp 30 / Kd 6 / 불감대 0.06 / 상한 100. 목표: 오른쪽 15°·위 5°에서 시작, 3~6초에 4 °/s 이동.

| 설정 | 출발·정지 반복 | 가장 큰 순간 속도 변화 (°/s) | 평균 오차 |
|---|---|---|---|
| v25 규칙 (옵션 없음) | 7 / 7 | 6.0 / 6.0 | 4.3° |
| `--deadband-exit 0.10` | 2 / 3 | 6.0 / 6.0 | 4.3~4.4° |
| `--axis-reversal 1` | 6 / 7 | 1.0 / 3.5 | 4.1~4.3° |
| 셋 다 (`--stop-rate 200` 포함) | 1 / 1 | 1.0 / 1.0 | 4.2° |

## 하지 않은 것, 한계

- **브리지 전체 빌드와 `test_v27_options.py`, `test_gimbal_bridge.py`는 돌리지 못했습니다** (PC에 OpenCV·GStreamer 없음). 장치에서 빌드한 뒤 두 시험을 먼저 돌려야 합니다.
- **실제 짐벌에서 시험하지 않았습니다.** 시뮬레이션의 SPD 환산값과 지연은 가정이고 관성·백래시가 없습니다. 표의 수치는 경향만 봅니다.
- `--axis-reversal 1` 단독은 2회 중 1회에서 효과가 작았습니다(대기 시간이 끝나 높은 속도로 다시 합류). `--deadband-exit`와 같이 쓰는 것을 권합니다.
- `--stop-rate`는 지금 이득(Kp 30)에서는 불감대 진입 속도가 이미 1~2 SPD라 효과가 거의 없습니다.
- 기본값은 경험값입니다(이론적 최적화나 안정성 증명 없음).
- 장치 주소, 폴더 경로, 짐벌 장치 번호는 어느 파일에도 넣지 않았습니다. 장치는 `--device` 옵션으로 받습니다(v25와 같음).
- 장치의 v26(`--slow-pulse`)은 포함하지 않았습니다. v27은 v25에서 갈라진 것입니다.
