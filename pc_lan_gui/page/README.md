# PC LAN GUI 화면 조각

이 화면은 `tracking_server/page/` 디자인을 PC 직접 연결 GUI에 맞게 복사한
별도 버전입니다. Orange Pi의 8090 서버는 이 폴더를 읽지 않습니다.
PC GUI의 영상 입력과 명령은 `../desktop_gui.py`와 `../desktop_routes.py`가
기존 Runtime UDP/TCP 연결로 처리합니다.

화면을 고칠 때는 `html/`, `css/`, `js/`의 해당 조각을 수정한 뒤 아래 명령을
실행합니다. 생성된 `live_page.html`을 직접 수정하지 않습니다.

```bash
python3 pc_lan_gui/page/build_page.py
python3 pc_lan_gui/page/build_page.py --check
```

`parts.json`이 조각 순서를 정합니다. 조각과 생성된 HTML을 함께 커밋합니다.
원본 8090 서버 화면의 구조 설명은
[`tracking_server/page/README.md`](../../tracking_server/page/README.md)에
있습니다. PC용으로 바뀐 명령 연결과 사용 가능 범위는
[`pc_lan_gui/README.md`](../README.md)에 있습니다.
