# 추적기 C++ 이식 안내

이 폴더의 Python 구현이 비교 기준입니다. 첫 이식은 **ByteTrack의 IoU 계산부터** 시작하고, 같은 입력에서 결과가 맞으면 ByteTrack 전체로 넓힙니다. Python 서버가 기존처럼 프레임과 설정을 관리하고, C++ 결과를 Python 추적기와 같은 형식으로 돌려주면 8090 웹 화면과 팀 PC GUI가 그대로 받을 수 있습니다.

## 파일과 작업 순서

| 순서 | 파일 | 이식 범위 |
|---|---|---|
| 1 | [`tracker_common.py`](tracker_common.py), [`kalman.py`](kalman.py) | IoU·박스 변환, Kalman 예측·갱신, 비용 행렬 매칭을 각각 비교합니다. |
| 2 | [`bytetrack.py`](bytetrack.py) | 높은 점수/낮은 점수 연결, 새 트랙 확인, 분실·만료, ID 할당 순서. `confirm3`는 별도 구현이 아니며, `rebind`는 같은 클래스의 추가 모드입니다. |
| 3 | [`botsort.py`](botsort.py) | ByteTrack 공통 상태를 바탕으로 IoU 비용과 외형 특징을 결합합니다. |
| 4 | [`alt_tracker.py`](alt_tracker.py), [`sort.py`](sort.py), [`ocsort.py`](ocsort.py), [`biou.py`](biou.py) | `AltTracker` 공통 흐름을 옮기고 SORT, OC-SORT, C-BIoU의 연결 차이를 각각 검증합니다. |
| 5 | [`lostfirst.py`](lostfirst.py), [`gate.py`](gate.py) | 이전·현재 프레임 외형 요청과 학습 기반 확인 경로를 별도로 비교합니다. |

- [`trackers.py`](trackers.py)는 기존 import 경로를 유지하는 파일입니다. 구현을 추가할 때는 각 클래스 파일에서 시작합니다.
- [`camera_motion.py`](camera_motion.py), [`stabilizer.py`](stabilizer.py), [`position_memory.py`](position_memory.py), [`confirm_time.py`](confirm_time.py)는 보정·상태 보조 코드입니다. 처음에는 Python에서 기존 계산값을 넘깁니다.
- [`warm_p95.py`](warm_p95.py)는 검출 후 온도 필터, [`stitch.py`](stitch.py)는 추적 결과 잇기, [`gt_score.py`](gt_score.py)는 평가용입니다. 먼저 추적기 출력 동등성을 맞추고 필요할 때 각각 분리해 옮깁니다.

## 서버와 맞출 입력·출력

호출 지점은 [`app.py`](../app.py)의 `APP.tracker.update(...)`입니다. 웹의 추적기 변경과 실시간 영상 초기화는 [`page_settings.py`](../web/page_settings.py)의 `mk_tracker_()`를 사용합니다. 오프라인 실행에는 별도 생성 경로가 있으므로 두 모드를 모두 지원할 때 각각 연결합니다.

```text
검출·필터 → arr → tracker.update(...) → 추적 행 → KeepID(설정 시) → live_publish()

입력 arr [N, 6] = [x1, y1, x2, y2, conf, cls]
출력 한 행      = [x1, y1, x2, y2, id, conf, cls, det_idx]
```

- 입력은 필터를 지난 `float32` 배열입니다. `det_idx`는 **그 프레임의 최종 `arr`에서 0부터 센 행 번호**입니다. 내부에서 점수순 정렬이나 높은/낮은 점수 그룹 분리를 해도 원래 인덱스를 유지합니다. 빈 검출 프레임도 `(0, 6)`으로 호출해 추적기 프레임 번호와 분실 기간이 진행되게 합니다.
- 실시간/녹화 재생의 추적 좌표는 640×512 캔버스의 픽셀 `XYXY`입니다. 640×480 카메라 영상은 위에서 16픽셀 떨어진 곳에 들어갑니다. C++에는 서버가 준 좌표를 그대로 넘기고 출력 좌표도 그 기준으로 돌려줍니다. 원본 영상 좌표와의 변환은 기존 발행부가 맡습니다.
- `id`, `cls`, `det_idx`는 Python에서 정수여야 합니다. ID를 `float32`에 담지 않습니다. 좌표와 점수뿐 아니라 출력 순서도 유지합니다. 예를 들어 ByteTrack은 활성화된 트랙만 `tracked` 목록 순서대로 내보냅니다.
- `PRED_*.jsonl`은 좌표 변환과 반올림을 거친 `p`의 저장값입니다. 온도 필터와 다른 옵션이 바꾼 최종 `arr`와 일치한다고 가정할 수 없습니다. 프레임 단위 비교에는 **`update` 직전의 `arr`**를 따로 기록하거나 그와 동일함을 확인한 캐시를 씁니다.

## 상태와 연결에서 놓치기 쉬운 부분

- `tracked`, `lost`, `frame_id`, `next_id`, `n_init`, `new_this_frame`, `lost_this_frame`은 프레임 사이에 유지되며 일부는 Python 호출부가 직접 읽거나 바꿉니다. Kalman 평균·공분산, `is_activated`, 마지막 검출과 외형 이력도 비교합니다. 래퍼가 Python에 상태를 노출할 경우 같은 트랙을 두 곳에서 따로 갱신하지 않도록 소유권을 정합니다.
- GUI에서 추적기를 바꾸면 프레임 사이에 새 인스턴스를 만들고 KeepID를 초기화합니다. 영상 전환·탐색은 기존 재생 경로가 정한 초기화 시점을 따릅니다. 실시간 입력이 1초 넘게 끊겼을 때는 활성·분실 목록을 비우되 현재 인스턴스의 ID 카운터는 유지합니다. 시간 기반 확인은 `confirm_time.apply()`가 `n_init`을 바꾸므로 생성 시 한 번만 읽으면 동작이 달라집니다.
- GMC의 변환은 이전 프레임에서 현재 프레임으로 갑니다. 기존 코드는 미확정 트랙을 미리 보정하고, 활성·분실 풀은 `GKF.multi_predict()`에서 대기 중인 변환을 소비합니다. 경계를 바꿀 때 같은 트랙에 같은 변환을 두 번 적용하지 않습니다.
- BoT-SORT의 외형 입력은 `{det_idx: L2 정규화 특징}`입니다. GUI의 `repvgg` 경로는 `app.py`에서 `isinstance(..., BotSortTracker)`를 확인합니다. 래퍼 적용 후에도 타입 검사 또는 그와 동등한 기능 판별이 유지되어야 합니다. LostFirst와 `rebind`의 후보별 외형 콜백 및 비동기 결과는 프레임·검출 행과 맞춥니다. KeepID는 추적기 뒤에서 동작하며, 최종 화면 ID는 KeepID 뒤의 값을 비교해야 합니다.
- 매칭은 [`tracker_common.py`](tracker_common.py)의 비용 제한과 순서를 따릅니다. SciPy와 다른 솔버는 동점에서 다른 짝을 낼 수 있습니다. 빈 행렬, `1×1`, 임계값과 같은 비용, 직사각 행렬, 동점, 매칭 후 미배정 목록을 먼저 맞춥니다. Python의 Kalman 상태와 `Track` 박스는 `float64`를 쓰므로 처음부터 일괄 `float32`로 바꾸지 않습니다.

## C++ 자동 선택 규약

- 실시간 GUI는 `bytetrack.cpp`, `botsort.cpp`, `ocsort.cpp`, `sort.cpp`를 찾습니다. `confirm3`와 `rebind`는 `bytetrack.cpp`를 공유합니다.
- 소스를 이 폴더에 넣고 저장소 최상위에서 `bash tracking_server/scripts/setup.sh`를 실행합니다. 추적기만 빌드할 때는 `bash tracking_server/scripts/build_native_trackers.sh`를 사용합니다. Orange Pi의 `g++`, 서버 Python 버전에 맞는 개발 헤더와 `python3-config`가 필요합니다. 재시작하면 빌드된 C++를 우선 사용합니다.
- C++ 소스는 CPython 확장 모듈 `_native_<이름>`을 제공해야 합니다. 예를 들어 `bytetrack.cpp`의 초기화 함수는 `PyInit__native_bytetrack`입니다. `API_VERSION = 1`과 `create_tracker(mode, kwargs)`를 내보냅니다. 여러 모드는 `SUPPORTED_MODES`에 선언합니다. `confirm3`는 `mode='bytetrack'`과 `n_init` 설정을, `rebind`는 `mode='rebind'`를 받습니다. 나머지 모드는 파일 이름과 같습니다.
- 팩토리는 `update(arr, det_feats, feat_fn)`와 위의 상태 필드를 갖는 객체를 반환합니다. 서버가 `n_init`, `kf`, `tracked`, `lost`를 변경하는 경로도 지원해야 합니다. 빌드나 로드가 실패하면 Python을 사용합니다. 메뉴는 사용 가능 구현을, 현재 값은 실제 실행 중인 구현을 보여 줍니다.
- 일부 계산만 옮길 때는 Python 추적기 내부에서 `ctypes`로 별도 공유 라이브러리를 호출할 수도 있습니다. [후처리 어댑터 예](../detect/native_post.py). 전체 추적기를 메뉴에 C++로 표시하려면 위 모듈 규약을 따릅니다.
- 8090은 기존 [`live_publish()`](../web/page_handler.py)를 사용합니다. 팀 PC GUI도 캡처 프레임 번호와 영상 타임스탬프를 유지한 기존 전송 경로를 씁니다. [팀 PC 통신 형식](../gimbal/team/docs/pc_communication.md)

## 팀 runtime의 C++ 코드

팀의 [runtime C++ 추적기](https://github.com/lwir-deepx-tracking-system/runtime/tree/0fe016f7edd069900457a746dc4cc787569c5ae6/src/tracking)에는 ByteTrack과 OC-SORT가 있습니다. `track(vector<Detection>)`과 `reset()`을 사용합니다. 재사용하려면 `XYWH`↔`XYXY`를 변환하고, 검출 인덱스 `det_idx`와 서버가 읽는 추적 상태를 연결하는 어댑터가 필요합니다. 임계값, Kalman 상태 표현, 새 트랙 확인과 클래스 구분도 Python판과 맞춰서 검증합니다.

## 넘기기 전 확인

1. **PC에서 단위 비교.** 같은 배열로 Python과 C++의 IoU, 비용·짝, Kalman 평균·공분산을 대조합니다. 좌표·점수 허용 오차는 비교 전에 정하고, 오차가 임계값을 넘어 실제 매칭을 바꿨는지 확인합니다. 현재 확인 시간 테스트는 `tracking_server` 폴더에서 `python3 -m unittest src.track.test_confirm_time`으로 실행할 수 있습니다.
2. **프레임별 비교.** 같은 시퀀스, 같은 검출 순서·점수, 같은 설정으로 Python판과 C++판을 각각 실행합니다. 프레임 번호, 빈 입력, 타임스탬프·초기화 이벤트, GMC와 외형 입력을 함께 고정합니다. 출력 행 수·순서·ID·클래스·`det_idx`, 활성/분실 상태가 일치해야 합니다. 좌표·점수는 정한 허용 오차 안에서 비교합니다. 먼저 달라지는 프레임과 상태를 기록합니다.
3. **서버 연결 확인.** 카메라와 짐벌 없이 PC의 모의 입력 또는 저장 영상으로 한 사람·두 사람, 교차·가림·재등장, 무검출, 재생 전환·탐색, 추적기 변경을 확인합니다. KeepID 사용 시에는 추적기 직후와 최종 출력 양쪽을 봅니다. 웹 화면의 박스·ID가 같은 프레임에 나오는지 확인하고, 팀 PC GUI도 사용한다면 별도로 영상과 메타데이터를 맞춰 봅니다.
4. **M1 속도 비교.** 결과가 맞은 뒤 같은 녹화·모델·옵션으로 추적 계산 시간, 전체 출력 간격, 프레임 처리 지연의 중앙값·p95를 비교합니다. 출력 개수와 건너뛴 프레임도 함께 기록합니다.

인계에는 기준 커밋, 옮긴 파일과 빌드 명령, 선택 방법, 비교에 사용한 입력·설정, 처음 다른 프레임 또는 일치한 프레임 수, M1 측정 여부를 남깁니다. 이 문서는 이식 절차이며 현재 추적기 구현은 Python입니다.

## 팀 C++ ByteTrack 비교

팀 [runtime](https://github.com/lwir-deepx-tracking-system/runtime)의 ByteTrack을 Python ByteTrack과 비교합니다. 빌드는 팀 저장소 커밋 `0fe016f7edd069900457a746dc4cc787569c5ae6`의 Git 객체를 읽습니다. 팀 코드는 모델뱅크에 복사해 넣지 않습니다. 빌드 중에만 실제 매칭된 검출의 행 번호를 결과로 전달하도록 연결합니다.

```sh
python tracking_server/src/track/build_team_bytetrack.py /path/to/runtime /path/to/team_bytetrack.so
python -m tracking_server.src.track.check_team_bytetrack /path/to/team_bytetrack.so --cache /path/to/exact_detections.jsonl
```

Windows에서는 출력 확장자를 `.dll`로 바꾸고 필요하면 `--cxx`로 MinGW `g++` 경로를 지정합니다. 시험용 캐시에는 프레임마다 `tracker_input_xyxy_conf_class`와 `tracker_reset`이 있어야 합니다. 두 구현 모두 같은 검출을 프레임 순서대로 받고, 빈 프레임도 업데이트하며, 시퀀스 경계에서만 ID를 초기화합니다.

서버의 녹화 재생에서 C++ ByteTrack을 선택할 수 있습니다. ARM64 라이브러리를 장치의 `tracking_server/models/links/team_bytetrack.so`에 설치한 뒤 서버를 다시 시작합니다. 다른 경로는 `--replay-bytetrack-library` 또는 `LWIR_REPLAY_BYTETRACK_LIBRARY`로 지정합니다. 녹화 재생 패널에서 **C++ ByteTrack**을 고르면 영상의 상자·ID와 평가에 적용합니다. 기본 선택은 현재 추적기입니다.

C++ 경로는 최종 필터를 거친 검출 행을 받아 독립 추적 상태를 유지합니다. 녹화 전환이나 새 반복 회차에서 상태를 초기화하고, 일시정지 중에는 마지막 결과를 사용합니다. 기존 추적기와 별도로 실행되므로 추적기 단독 시간과 서버 전체 지연은 구분해서 해석합니다. 비교용 ByteTrack의 점수 문턱과 XYWH Kalman은 Python 쪽에 맞췄지만, 확인·매칭 규칙과 출력은 완전히 같지 않습니다. 출력 수가 늘거나 줄어도 정답을 대조하지 않으면 정확도가 개선됐는지 판단할 수 없습니다.

재생 화면에서 **끝까지 다시 계산**을 누르면 선택한 추적기로 첫 장부터 한 번 재생합니다. 슬라이더로 이동하면 해당 장에서 멈추고 첫 장부터 그 장까지 채점한 누적값을 표시합니다. 아직 그 위치까지 채점하지 않았다면 다시 계산해 결과를 만들 수 있습니다. **최종 보기**는 재생 위치를 유지한 채 같은 녹화와 추적기의 완주 결과를 표시합니다.
