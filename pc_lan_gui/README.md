# PC 직접 연결 GUI

이 폴더는 기존 `tracking_server/page/`의 화면 디자인을 PC에서 사용하는
버전입니다. **Orange Pi에서는 C++ Runtime을 실행**하고, PC에서 이 GUI를
실행합니다. 화면은 Gtk/WebKit2에 띄우며, PC와 Runtime은 영상 UDP 5000,
추적 메타데이터 UDP 5001, 명령 TCP 5002로 직접 통신합니다. Python 8090
웹서버와 HTTP·MJPEG·WebSocket 서버를 새로 열지 않습니다.

## 실행

PC에 Python GI의 Gtk3·WebKit2 4.0·GStreamer 1.0·GstVideo 1.0,
GStreamer H.264 디코더, Pillow가 필요합니다. Runtime YAML의 영상·메타데이터
목적지를 PC의 LAN 주소로, 명령 수신 주소를 Orange Pi의 LAN 주소로 설정합니다.

```bash
./pc_lan_gui/launch_lan_gui.sh <Runtime의 LAN 주소>
```

또는 `LWIR_BOARD_IP` 환경 변수에 주소를 지정할 수 있습니다. 실행 스크립트는
중복 실행을 막고 PC의 `~/.local/state/lwir-gui/gui.log`에 로그를 남깁니다.
주소는 저장소에 넣지 않습니다. 이 폴더의 코드는 PC에서만 실행합니다.

## 현재 연결된 기능

- 영상과 추적 상자·ID 표시, 화면 클릭 대상 선택, 수동 이동·정지
- Runtime에서 받은 검출 모델·Tracker 목록과 실제 변경 완료 결과
- 활성 Tracker 정책 조회와 온도·가장자리·확인 시간·GMC 변경
- ByteTrack과 사후 ReID 구성 간 KeepID 선택
- PC에서 실제 받은 영상·메타데이터의 초당 수신 수 그래프

온도·가장자리 Filter 타일은 아래 **Filter 값**에 임계값을 먼저 입력한 뒤
켜고 끌 수 있습니다. 정책 변경은 Tracker를 재생성하므로 Track ID와
Gallery가 초기화됩니다.

예측 제어·예측 상자, 실행 중 ReID 모델 변경, 측정 세션, 녹화 재생,
카메라·시스템 명령은 현재 C++ Runtime LAN 계약에 없습니다. 화면은 이 항목을
사용 불가로 표시하거나 실패를 돌려줍니다. Runtime 단계별 지연도 이 계약으로
전송되지 않으므로 그래프에 가짜 값을 표시하지 않습니다. 영상과 메타데이터의
프레임 단위 동기화 및 장시간 프레임 속도는 아직 측정하지 않았습니다.

## 코드와 확인

`desktop_gui.py`는 영상·메타데이터를 받고 화면을 띄웁니다.
`lan_protocol.py`는 기존 Runtime 바이너리 명령을 사용합니다.
`desktop_routes.py`는 화면 조작을 그 명령에 연결합니다. `page/`는 원본
디자인을 복사한 PC용 화면이며, [화면 수정법](page/README.md)에 따라
조각을 고친 뒤 HTML을 다시 만듭니다. 원본 8090 서버 화면은 변경하지 않습니다.

```bash
python3 pc_lan_gui/page/build_page.py --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=pc_lan_gui \
  python3 -m unittest discover -s pc_lan_gui -p 'test_*.py' -q
```

이 버전은 `pc-application`의 `feat/python-runtime-integration` 브랜치
`4b8094b`에서 옮겼습니다. 이 저장소에서는 공개 주소 기본값을 제거했습니다.
