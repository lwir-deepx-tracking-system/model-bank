# AUTO 표적 중심 정렬

`gui.command.handler: tracking`은 PC AUTO(ID), MANUAL, 키보드 MOVE를 하나의
짐벌 실행 경계에 연결한다. 수동 전용 핸들러는 제거했으며 `deferred`는 통신 전용으로 유지한다.
기존 `handler: manual` 설정은 `handler: tracking`으로 변경해야 한다.
HOME과 키 해제는 T=135 STOP을 사용한다. HOME은 정지 요청 후 NOT_READY이며 물리 원점 이동/완료를 보고하지 않는다.

## 책임과 실행 구조

- `PidController`: 축 하나의 P/I/D 계산·미분 필터·적분 제한을 담당한다. 시계나 장치에 접근하지 않는다.
- `AutoTrackingController`: 원본 영상 BBox 검증, 중심 오차, 축별 PID 상태, Deadband,
  공통 SPD를 계산한다. 캡처 시각과 프레임 ID를 명시적으로 받으며 TCP/Serial/스레드 접근은 없다.
  배선 부호는 장치 어댑터에서 한 번만 적용한다.
- `ControlThread`: Tracking 결과를 executor의 최신 결과 mailbox로 전달한다.
  Tracking→Control queue는 최신 한 개만 보존한다. Serial과 선택 ID는 소유하지 않는다.
- `GimbalCommandExecutor`: 전용 스레드에서 선택 ID, AUTO/MANUAL 상태, 명령 세대,
  watchdog, 모든 장치 I/O를 소유한다. 이 경로는 기존 `TargetSelector`를 사용하지 않는다.
  계산된 AUTO MOVE를 별도 큐에 보관하지 않아 모드 전환 후 재생될 명령이 없다.
  연결 해제/모델 전환/종료 요청은 대기 요청보다 우선 처리하고, 일반 GUI 요청도 AUTO 출력보다
  먼저 처리한다. 진행 중인 Serial 쓰기는 선점하지 않으며 설정된 쓰기 deadline으로 제한한다.
- `TrackingCommandHandler`: TCP 명령·ACK·STATUS를 executor API에 대응시킨다.
  응답은 요청의 상태 전환/정지 제출이 실행된 다음 생성한다. ACK는 물리 동작 완료를 의미하지 않는다.
- `IGimbalDevice`: 기존 수동 `move(x,y)`와 AUTO `move_with_speed(x,y,spd)`를 제공한다.
  Waveshare 구현만 T=141/135 인코딩과 Serial을 다룬다.
- AppConfig만 YAML을 읽고 검증한다. 처리 객체에는 typed config를 전달한다.

## 설정

기존 `config/runtime.yaml`의 모델·GUI 설정을 유지하면서 다음 항목을 선택한다.
공유 기본 설정은 `control.enabled: true`, `gui.command.handler: tracking`으로 AUTO와 수동 조작을 활성화한다.

```yaml
control:
  enabled: true

  auto:
    controller: p          # p / pi / pd / pid
    pid_x: {kp: 30.0, ki: 0.0, kd: 0.0}
    pid_y: {kp: 30.0, ki: 0.0, kd: 0.0}
    deadband_x: 0.04
    deadband_y: 0.04
    speed_max: 100
    speed_rate: 60.0        # SPD/초. 안전 STOP은 제한 우회

  gimbal:
    device: /dev/ttyUSB0
    speed: 50              # 수동 방향키 고정 속도

gui:
  command:
    enabled: true
    handler: tracking
```

기본 YAML에는 운용·튜닝 항목만 노출한다. 장치별 고정값과 세부 항목은 생략하며,
AppConfig가 코드 기본값으로 채운다. 기존 설정 키는 계속 지원하므로 필요할 때만 추가하면 된다.

| 생략한 항목 | 기본값 |
| --- | --- |
| `control.heartbeat_timeout_ms` / `poll_interval_ms` | 1500 / 25 ms |
| `auto.period_ms` / `input_timeout_ms` / `reacquire_timeout_ms` | 25 / 250 / 1000 ms |
| `auto.speed_min` / `min_confidence` | 1 / 0.5 |
| `auto.fov_x_deg` / `fov_y_deg` | 0 / 0 (정규화 오차) |
| 각 축 `integral_limit` / `derivative_filter_ms` | 10 / 100 ms |
| `gimbal.baud_rate` / `write_timeout_ms` | 115200 / 100 ms |
| `gimbal.speed_limit` / `acceleration` | 100 / 50 |
| `gimbal.home_x_deg` / `home_y_deg` | 0 / 0 |
| `gimbal.x_sign` / `y_sign` | 1 / 1 |

미분 필터·적분 제한 등도 튜닝할 수 있다. 조정 시 해당 축의 inline map에 키를 추가하거나
일반 YAML map 형태로 풀어서 작성한다.

숫자는 실기 튜닝 전 임시값이다. 기본 수동 속도 `gimbal.speed`는 50,
AUTO 상한 `auto.speed_max`는 100으로 서로 독립적이다.
`gimbal.speed_limit`은 모든 MOVE 명령에 적용하는 별도의 소프트웨어 상한(기본 100)이며,
수동 속도와 AUTO 상한 모두 이 값을 넘을 수 없다. 실제 장치의 물리 최대속도나
지원 범위를 실측했다는 의미는 아니다. `auto` 생략 시에도 AUTO 상한 기본값은 100이다.
SPD는 장치 명령 단위이고 °/s가 아니다. P의 기본 Kp=30은 그대로이므로,
AUTO 상한을 100으로 올려도 계산 결과가 30 이하이면 실제 명령도 30 이하이다.

FOV 두 값이 0이면 중심 오차를 영상 반폭/반높이로 나눈 정규화 오차를 쓴다.
두 FOV를 모두 설정하면 pinhole 변환으로 rad 오차를 계산하며 Deadband도 rad 단위다.
두 방식은 Kp/Deadband를 따로 튜닝해야 한다. 축별 Deadband 이내(경계 포함)인 축은
방향과 출력을 0으로 하고 해당 축의 I/D 상태를 초기화한다. 움직일 축의 PID 출력 절댓값 중
최댓값으로 공통 SPD를 계산한다. P 모드의 기본 계수는 기존 공통 Kp=30과 같다.
양축이 정지이면 최소속도를 적용하지 않고 STOP한다.

영상 좌표는 아래쪽이 +Y이고, PC 수동 명령 좌표는 위쪽이 +Y이다.
영상 좌표계의 축별 제어 출력을 `u_x`, `u_y`라 할 때
`X = sign(u_x)`, `Y = -sign(u_y)`로 변환한다(Deadband 밖의 축).
P 모드에서 위쪽 표적에는 수동 위 방향키와 같은 Y=+1, 아래쪽 표적에는 Y=-1을 요청한다.
PI/PD/PID에서는 누적·미분 출력으로 총 출력 부호가 오차 부호와 달라질 수 있으므로,
최종 방향은 오차가 아닌 총 출력으로 결정한다.
이 변환은 계산기에만 적용하며 수동 입력과 `gimbal.x_sign` / `gimbal.y_sign`은 유지한다.
장치 배선 부호는 이 좌표 변환 이후 장치 어댑터에서 한 번 적용한다.


SPD는 `speed_rate`(SPD/초)로 변화량을 제한한다. 첫 구동은 최소 유효속도에서 시작하며
명령은 정수로 반올림된다. 최소속도 진입과 정수 양자화는 연속 ramp의 예외다.
축 방향이 반전되면 양축 STOP 후 다음 제어주기에 최소속도에서 다시 시작한다.
안전 STOP과 Deadband STOP은 변화율 제한을 기다리지 않는다.

## P / PI / PD / PID 선택

`control.auto.controller`는 통신의 `gui.command.handler`와 독립적인 제어기 선택이다.
선택값은 `p`, `pi`, `pd`, `pid`이며 생략하면 `p`이다.
`p`는 P만, `pi`는 P+I, `pd`는 P+D, `pid`는 P+I+D를 사용한다.
사용하지 않는 항은 YAML에 계수가 있어도 출력에 포함하지 않는다.
따라서 매 비교 때 Ki/Kd를 0으로 덮어쓸 필요 없이 `controller`만 바꿀 수 있다.

기본 Ki/Kd는 미튜닝 상태인 0이다. PI/PD/PID의 추가 항을 시험하려면 해당 축의 계수를
먼저 설정해야 하며, 선택값만 바꾸고 Ki/Kd가 모두 0이면 결과는 P와 같다.
`pid_x`, `pid_y`로 축별 계수를 설정한다. 기존 `auto.kp` YAML도 두 축 Kp의 공통 기본값으로
읽으며, 축별 Kp가 있으면 그 값을 우선한다. Kp는 양수, Ki/Kd는 0 이상이어야 한다.
정규화 오차/FOV 각도 오차 간에는 세 이득의 단위가 달라지므로 함께 재튜닝한다.

- I항은 새 측정의 `Ki * error * dt`를 누적한다. `integral_limit`은 누적 오차 자체가
  아닌 I항 출력의 최대 절댓값(SPD 명령 단위)이다.
- 출력 포화를 더 키우는 적분은 동결한다. 공통 SPD·변화율 제한을 거친 실제 제출 명령도
  계산기에 돌려주어, 아직 요청 속도에 도달하지 못한 동안 같은 방향의 적분 누적을 막는다.
  반대 방향으로 누적값을 줄이는 적분은 포화 조건이 허용하면 수행한다.
  정수 SPD 반올림에 따른 0.5 단위 이내 잔차는 속도 제한으로 간주하지 않는다.
- 이 피드백은 명령값이며 실제 모터 속도/각도 측정값이 아니다. 공통 SPD의 축 간 결합은 남는다.
- D항은 서로 다른 프레임의 캡처 시각 차이로 계산하고, `derivative_filter_ms` 시정수의
  1차 저역통과 필터를 적용한다. 필터 시정수는 양수여야 한다.
- 동일 frame ID는 계산 결과를 재사용한다. 제어 tick이 반복돼도 I/D를 다시 갱신하지 않는다.
  역순 ID·역행 시각은 STOP 출력과 상태 초기화를 유발한다. timeout 이상의 측정 간격은
  첫 측정처럼 취급하여 I/D 과도 출력을 방지한다.
- 첫 측정에서는 I/D가 0이다. 선택 대상·모델 세대·영상 크기 변경, 상실·수동 전환·권한 해제·
  연결 종료·장치 오류·Runtime 종료에서는 이력을 초기화한다. 재획득 시 과거 상태를 재사용하지 않는다.
- 방향 반전 STOP은 제어 이력을 유지하면서 제출 출력 0을 피드백한다.
  기존 반전 정책대로 다음 주기부터 최소속도로 다시 시작한다.

## 상태 전이와 입력 계약

| 이벤트 | 상태/출력 | 응답 |
| --- | --- | --- |
| 시작·새 연결 | MANUAL / 선택 없음 / 정지 | 최초 STATUS |
| AUTO(ID) | 연결·heartbeat·Runtime·최신 표적 검증 후 이전 이동 STOP, 선택 등록 | 성공 시 ACK OK + STATUS |
| ID 0, int 범위 밖 ID, 현재 유효 결과에 없는 ID | 기존 선택 유지, 새 AUTO 거절 | INVALID_TARGET |
| heartbeat 만료·장치 오류·모델 전환 중 AUTO | AUTO 거절 | NOT_READY |
| 선택 Track 상실, 예측 BBox, stale/무효 입력 | 즉시 STOP, AUTO / Lost / Hold로 재획득 대기 | STATUS |
| 대기 기한 내 같은 ID·같은 모델 세대의 유효 결과 | AUTO 재개 | STATUS |
| 재획득 기한 만료 | MANUAL / 선택 해제 / 정지. 뒤늦은 결과로 자동 재개하지 않음 | STATUS |
| MANUAL | AUTO 권한·선택 무효화 후 STOP. 이미 MANUAL인 수동 이동은 유지 | ACK OK + 변경 시 STATUS |
| 유효 키보드 MOVE, MOVE(0,0) 포함 | AUTO 무효화·STOP 후 MANUAL 명령 적용 | ACK 없음, 변경 시 STATUS |
| 연결 종료·PING 만료 | 선택 해제·MANUAL·STOP. 재연결 시 이동 재생 없음 | 연결 유지 시 STATUS |
| 모델 전환·모델 세대 변경 | STOP·선택 해제. ID 재사용에 따른 다른 객체 추종 방지 | STATUS |
| Serial 실패 | 최선의 STOP 시도, Error 고정·선택 해제. 자동 복구 없음 | STATUS, 이후 모드 요청 NOT_READY |
| Runtime/Control 입력 종료 | 이동 정지·권한 영구 해제 | 종료 |

AUTO ID는 wire uint32를 손실 없이 파싱하고, 내부 tracker가 표현할 수 없는 ID는 거절한다.
선택은 `(model_generation, track_id)`에 묶인다. 명령 세대는 모델 세대와 별개로 관리한다.
모드·선택 변경 및 상실 때 이전 AUTO 권한이 무효화된다. 계산과 전송은 같은 스레드에서
수행하고 실행 직전 Runtime 상태와 권한을 재확인한다.

프레임의 `captured_at`은 steady clock 기준이다. 프레임 ID는 같은 모델 세대에서 증가해야
하며 역순·중복 결과는 최신 입력을 덮어쓰지 않는다. 미래/미지정 시각, 오래된 프레임,
bypass, 다른 모델 세대, 빈 영상/CV_16UC1 이외의 입력은 AUTO에 사용하지 않는다.
BBox의 모든 좌표는 유한수이고 크기는 양수, 중심은 원본 영상 내부여야 한다.
confidence는 설정 하한 이상 및 1 이하이어야 한다. `Track.predicted`가 true이면 거절한다.

`input_timeout_ms`는 캡처 시각부터 계산한다. stale 또는 표적 상실을 감지한 시점부터
`reacquire_timeout_ms` 동안만 재획득을 허용한다. 이 기한이 지난 결과는 재개시키지 않는다.
watchdog은 프레임 수신과 독립적으로 `min(period_ms, poll_interval_ms)` 간격으로 확인한다.
따라서 STOP 제출은 감지 주기와 진행 중인 bounded Serial 쓰기 시간의 영향을 받는다.
물리 정지 완료와 실제 STOP 지연은 하드웨어 검증 대상이다.

## PC 상태와 측정

PC `ControlProtocol.h`의 기존 8바이트 STATUS 형식을 유지한다.
Mode: Manual=0, Auto=1. Target: None=0, Tracking=1, Lost=2.
Gimbal: Idle=0, ManualMoving=1, AutoTracking=2, Hold=4, Error=5.
AUTO에서 이동 중이면 AutoTracking, 중앙 정렬 또는 재획득 대기 중이면 Hold이다.
Error code는 장치 오류 시 1, 그 외 0이며 selected ID는 big endian uint32다.
상태가 변경될 때 명령 응답 또는 TCP tick에서 STATUS를 전송한다.

`metrics.csv`의 `control_submit`은 mailbox 제출까지의 시간이다. 실제 제어 계산 또는
물리 짐벌 동작 완료 시간으로 해석하면 안 된다. Debug 제어 로그에는 시각, frame ID,
선택 ID, 계산 오차, 총 축 출력(ux/uy), I항(ix/iy), D항(dx/dy), 명령 방향/SPD,
모델·명령 세대 및 STOP 사유가 기록된다.

## 검증 범위

`test_pid_controller`: 제어기 선택별 항 활성화, 적분 제한·포화·속도 제한 시 적분 동결,
캡처 시각 기반 미분과 필터, 중복 프레임 재사용, 대상·모델 변경 및 재설정,
축별 Deadband 초기화, PID 출력 부호의 장치 방향 변환, YAML 선택·검증을 다룬다.

`test_auto_control`: 계산·Deadband 축 제외·SPD 제한·FOV·입력 검증, 프레임 없는 watchdog,
상실/재획득/기한 만료, MANUAL 전환, 모델 세대 무효화, heartbeat, 장치 실패,
속도 ramp·방향 반전·bypass를 명시적 제어 fixture와 기록 장치로 검증한다.

`test_auto_control_e2e`: 실제 loopback TCP AUTO → ControlThread Tracking 입력 →
단일 실행 스레드 → 실제 Waveshare 인코더 → PTY Serial 출력을 검증한다.
STATUS, 프레임 중단 STOP, 재획득, 키보드 override, 연결 해제/재접속도 포함한다.
장치 명령 인코딩·쓰기 deadline 시험과 deferred 회귀 시험을 유지한다.

실제 짐벌의 축 방향, 최소속도, 동시 축 구동, 축 0 정지, STOP 반응 및 중심 정렬 성능은
미검증이다. DXNN 로딩, NPU 추론, tensor 검증 및 NPU benchmark도 미검증이다.
CPU 제어 fixture와 PTY 시험은 이 하드웨어 검증을 대체하지 않는다.
