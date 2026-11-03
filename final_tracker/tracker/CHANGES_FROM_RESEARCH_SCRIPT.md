# lwir_tracker_m1.py 와 연구용 m1_track_eval_v11.py 의 차이

알고리즘과 출력은 같고, 아래 네 가지 환경 가정을 제거하고 한 가지(5번)를 추가했습니다.

1. M1 작업 폴더가 특정 git 브랜치(`test/chatgpt/yolobytetrack`)인지 확인하던 줄을 제거했습니다.
2. `--data`를 주지 않으면 연구용 폴더(`claude_flir_video_thermal_test`)를 쓰던 기본값을 없애고 `--data`를 필수로 했습니다.
3. `native_post.py`와 `post.so`를 연구용 작업 폴더가 아니라 이 스크립트 옆 폴더에서 읽도록 했습니다. `post.so`는 `post.cpp`를 `setup.sh`가 M1에서 빌드합니다.
4. 연구용 `runtime.py`(torch를 불러옴)에서 `join_tiles` 함수만 스크립트 안으로 복사했습니다(코드 동일).

M1에서 연구용 스크립트와 이 스크립트의 출력(TRACKS.jsonl, TRACKS_BASE.jsonl)이 바이트 단위로 같음을 확인했습니다(README 참고).
5. 검출기 DXNN의 출력이 모델뱅크 `tensor_requirements.md`의 분할 출력 6개(`box_s8/16/32`, `cls_s8/16/32`)로 바뀌어서, 이를 박스 `[1,4,N]`·점수 `[1,16,N]`로 합치는 몇 줄(`--split6`, 기본 켜짐)을 넣었습니다. 합친 뒤의 계산은 같아서, 이전 두 출력 DXNN으로 낸 결과와 바이트 단위로 같은 추적 결과가 나왔습니다.
