# lwir_tracker_m1.py 와 연구용 m1_track_eval_v11.py 의 차이

알고리즘과 출력은 같고, 아래 네 가지 환경 가정을 제거하고 한 가지(5번)를 추가했습니다.

1. M1 작업 폴더가 특정 git 브랜치(`test/chatgpt/yolobytetrack`)인지 확인하던 줄을 제거했습니다.
2. `--data`를 주지 않으면 연구용 폴더(`claude_flir_video_thermal_test`)를 쓰던 기본값을 없애고 `--data`를 필수로 했습니다.
3. `native_post.py`와 `post.so`를 연구용 작업 폴더가 아니라 이 스크립트 옆 폴더에서 읽도록 했습니다. `post.so`는 `post.cpp`를 `setup.sh`가 M1에서 빌드합니다.
4. 연구용 `runtime.py`(torch를 불러옴)에서 `join_tiles` 함수만 스크립트 안으로 복사했습니다(코드 동일).

M1에서 연구용 스크립트와 이 스크립트의 출력(TRACKS.jsonl, TRACKS_BASE.jsonl)이 바이트 단위로 같음을 확인했습니다(README 참고).
5. 검출기 DXNN의 출력이 모델뱅크 `tensor_requirements.md`의 분할 출력 6개(`box_s8/16/32`, `cls_s8/16/32`)로 바뀌어서, 이를 박스 `[1,4,N]`·점수 `[1,16,N]`로 합치는 몇 줄(`--split6`, 기본 켜짐)을 넣었습니다. 합친 뒤의 계산은 같아서, 이전 두 출력 DXNN으로 낸 결과와 바이트 단위로 같은 추적 결과가 나왔습니다.
6. 입력 크기: 표준 입력을 카메라 원본 640×480으로 하고, 640×512(FLIR 시험 영상)도 받습니다. 640×480은 640×512 캔버스의 행 16~496에 넣어 쓰고(위아래 16행은 가장자리 행 복제), 출력 좌표는 입력 영상 기준으로 되돌립니다(y - 16). 이전에는 `--preload` 경로에 크기 검사가 없었고 일반 경로는 이유 없이 멈췄습니다. 640×512 입력의 출력은 이전과 같습니다.
7. 실시간 서버: `--live-port`, `--live-bind`, `--live-seconds`, `--live-cam`, `--live-map`, `--live-scale`와 검출기 선택 `--det w480`, `--det s640`, 카메라 읽기 프로그램 `cam_pipe.cpp`를 추가했습니다. `--live-port`를 주지 않은 파일 실행의 출력은 바뀌지 않았습니다(M1에서 `--det rt480`의 TRACKS.jsonl이 이전 결과와 바이트 단위로 같음을 확인). 실시간 서버는 `--data`가 필요 없고, `--live-bind` 기본값은 이 장치에서만 열리는 `127.0.0.1`입니다.
