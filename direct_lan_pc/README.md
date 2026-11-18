# ModelBank 직결 LAN PC 앱

이 모드는 **model-bank만** 사용합니다. Orange Pi에서 기존 카메라·검출·추적·짐벌·녹화 재생·측정 코드를 실행하고, PC에서는 Python/Tk 독립 앱 창을 띄웁니다. Runtime 저장소, 브라우저, WebKit, Tailscale, 인터넷, 8090 웹 포트는 새 모드에 필요하지 않습니다. 기존 8090 웹 실행은 그대로 남아 있으며, 두 모드를 동시에 켜지는 않습니다(카메라와 NPU는 한 실행만 사용).

Orange Pi의 `--direct-lan-bind` 실행은 지정한 유선 IP에서 PC 한 대의 IP만 받는 직결 TCP 통신을 엽니다. 요청은 길이가 붙은 JSON, 영상은 길이가 붙은 JPEG 바이트입니다. 화면용 웹 서비스는 열지 않습니다. 기존 model-bank 명령 검증을 프로세스 안에서 재사용합니다.

## 준비

- Orange Pi와 PC를 랜선으로 직접 연결합니다. 인터넷이나 공유기는 필요 없습니다.
- 이 설치의 기본 주소는 **PC `192.168.77.1/24`**, **Orange Pi `192.168.77.2/24`**, 통신 포트 `8765`입니다. 다른 주소를 쓴다면 양쪽 실행 옵션도 함께 바꿉니다.
- 유선 고정 주소를 처음 설정할 때는 각 장치에서 `sudo bash tracking_server/scripts/setup_direct_ip.sh <유선 인터페이스> <주소/24>`를 실행합니다. 이 스크립트는 전용 NetworkManager 프로필만 만들거나 고치며, 다른 네트워크의 기본 경로를 바꾸지 않습니다. 이 작업 환경의 PC 프로필은 `enp3s0`에 `192.168.77.1/24`로 이미 준비되어 있습니다. 케이블이 없어 현재 연결 검사는 보류 상태입니다.
- Orange Pi에는 기존 `tracking_server/scripts/setup.sh`가 준비한 model-bank 모델, 카메라 프로그램, 짐벌 브리지가 있어야 합니다. `scripts/live.env`의 장치별 설정은 기존 실행과 같이 읽습니다.
- PC에는 Python 3, Tkinter, Pillow가 필요합니다. Debian/Ubuntu: `sudo apt install python3-tk python3-pil`.
- 랜선이 연결되지 않은 지금은 실제 영상·짐벌·NPU 동작을 검증할 수 없습니다.

## 실행 방법

기존 8090 실행/keepalive가 Orange Pi의 카메라·NPU를 쓰고 있다면 먼저 **그 실행을 정상 종료**하고, keepalive가 다시 켜지지 않도록 장치의 기존 `lwir_keepalive.pause` 절차를 적용합니다. 다른 작업자의 실행을 강제로 종료하지 않습니다. 그런 다음 Orange Pi의 model-bank 저장소에서:

```bash
bash tracking_server/scripts/run_direct_lan.sh
```

이 스크립트는 기존 `run_real_gui.sh`의 모델·짐벌·재생 설정을 그대로 사용하면서 새 옵션 `--direct-lan-bind 192.168.77.2 --direct-lan-peer 192.168.77.1 --direct-lan-port 8765`를 줍니다. 기존 웹 실행의 `--live-port 8090`, Tailscale 바인딩, 다른 PC GUI로의 H.264 송출은 주지 않습니다. 장치 설정 파일에서 주소를 바꾸려면 `LWIR_DIRECT_LAN_BIND`, `LWIR_DIRECT_LAN_PEER`, `LWIR_DIRECT_LAN_PORT`를 지정합니다.

PC에서:

```bash
./direct_lan_pc/launch.sh
```

PC 바탕화면 아이콘을 설치하려면:

```bash
bash tracking_server/scripts/desktop/install_direct_lan.sh
```

아이콘 이름은 **ModelBank 직결 LAN**입니다. Orange Pi 주소를 바꾼 경우 PC에서는 `./direct_lan_pc/launch.sh --board <주소> --port <포트>`로 실행합니다. 종료할 때 PC 창을 닫으면 짐벌 정지 명령을 보냅니다. Orange Pi 실행은 기존과 같이 `Ctrl+C`로 정상 종료합니다.

## PC 앱 기능

- 실시간 JPEG 영상, 같은 프레임에 속한 사람 상자·ID 선택, 화면 우선 전송의 상자 그리기, 놓친 사람 예측 상자, 짐벌 추종·원점·정지·방향 조작
- 검출 모델, 자동 검출 모델 선택, 두 모델 상주, NPU 투입 방식, 엔드투엔드, Tracker/엔진, 외형 모델·KeepID, 위치 기억, 상자 다듬기, Filter와 표시 옵션
- 카메라 초기화, 추적 프로그램 재시작, Orange Pi 재부팅
- 실시간 측정 시작·종료, 영상 저장, 지난 측정 열기·이름/메모 수정
- 녹화 목록과 재생 Tracker 선택, 시작·정지·일시정지·재개·한 장 이동·위치 이동·속도·반복, 정답 상자와 현재/위치/최종 채점 결과
- 최근 성능 값, 시간별 FPS 그래프, 자세한 처리 기록과 현재 장치 상태

기능별 사용 가능 여부는 Orange Pi의 모델 파일·브리지 버전·현재 카메라/재생 상태에 따릅니다. 장치가 명령을 거절하면 앱 아래쪽에 이유가 표시됩니다. 새 앱의 배치와 그래프 모양은 기존 웹 화면과 다르지만, 처리와 설정은 model-bank의 같은 코드를 사용합니다.

## 확인 상태

- PC에서 직결 명령의 요청·응답, JPEG 바이너리, 기존 설정 명령 경로, 표시 프레임의 ID 소유권 전달과 사진 먼저 모드의 프레임 짝을 5개 모의 시험했습니다.
- PC 앱 창을 실제 Tk 화면과 모의 영상/상태로 띄워 확인했습니다.
- 기존 웹 영상 관련 회귀 시험을 실행했습니다. 알려진 `test_live_preview_prepare` 모형 함수 오류 한 건은 기반 브랜치에도 있습니다.
- **아직 확인하지 못한 것:** 랜선 연결, Orange Pi의 새 코드 설치/실행, 실제 카메라·NPU·짐벌, 1:1 전송 지연과 프레임률. 이 항목은 케이블을 연결할 수 있을 때 함께 확인합니다.
