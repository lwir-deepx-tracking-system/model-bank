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

실시간 서버의 공통 추적기 팩토리는 아래 CPython 확장을 찾습니다. 저장소 최상위에서 `bash tracking_server/scripts/build_native_trackers.sh`로 추적기만 빌드하거나 `bash tracking_server/scripts/setup.sh`로 장치용 구성 요소와 함께 빌드합니다. Orange Pi의 `g++`, 서버 Python에 맞는 개발 헤더와 `python3-config`가 필요합니다. 빌드 결과는 이 폴더의 `_native/`에 Python 버전과 운영체제에 맞는 확장자로 설치합니다.

| C++ 소스 | CPython 모듈 | 서버가 넘기는 `mode` |
|---|---|---|
| `bytetrack.cpp`의 CPython 구현 | `_native_bytetrack` | `bytetrack` (`confirm3` 포함), `rebind` |
| `botsort.cpp` | `_native_botsort` | `botsort` |
| `ocsort.cpp` | `_native_ocsort` | `ocsort` |
| `sort.cpp` | `_native_sort` | `sort` |

- 모듈 초기화 함수 이름은 `PyInit__native_<이름>`입니다. `API_VERSION = 1`과 `create_tracker(mode, kwargs)`를 내보냅니다. `SUPPORTED_MODES`를 생략하면 파일 이름과 같은 모드만 선택합니다. `confirm3`는 `bytetrack` 모드와 기존 `n_init` 설정을 받으므로 위치 재결합을 지원하려면 `rebind`도 따로 선언합니다. 현재 저장소의 `bytetrack.cpp`는 아래 절의 팀 C ABI 구현입니다. 이 파일은 공통 빌드에서 건너뛰며 별도 빌드와 어댑터를 사용합니다.
- 팩토리가 반환하는 추적기는 `update(arr, det_feats=None, feat_fn=None)`와 위에 적힌 상태 필드를 제공합니다. 서버가 `n_init`, `kf`, `tracked`, `lost`를 읽고 바꾸는 경로도 지원합니다. 각 출력 행은 `[x1, y1, x2, y2, id, conf, cls, det_idx]` 형식이며 최종 검출 배열의 행 번호, 좌표 기준, 출력 순서를 보존합니다. 빈 입력도 프레임 진행에 포함합니다.
- 빌드 후 처음 불러오는 확장은 다음 추적기 생성 때 우선 사용합니다. 소스보다 오래됐거나 규약에 맞지 않는 확장은 Python으로 돌아갑니다. 한번 불러온 확장 파일을 같은 이름으로 교체하면 현재 객체는 계속 기존 코드로 동작합니다. 공통 로더는 파일의 경로와 디스크 식별값을 비교해 재시작이 필요하다고 표시하고, 새 객체에는 재시작 전까지 Python을 사용합니다. 빌드 스크립트처럼 임시 파일을 만든 뒤 완성본으로 교체합니다.
- `backend_status()`는 다음 객체에 사용 가능한 구현과 재시작 필요 여부를 제공합니다. `backend_of()`는 현재 객체가 실제 쓰는 구현을 반환합니다. 둘은 실행 중 확장을 교체했을 때 서로 다를 수 있습니다.
- 일부 계산만 옮길 때는 Python 추적기 내부에서 `ctypes`로 별도 공유 라이브러리를 호출할 수도 있습니다. [후처리 어댑터 예](../detect/native_post.py). 전체 추적기를 메뉴에 C++로 표시하려면 위 모듈 규약을 따릅니다.
- 8090은 기존 [`live_publish()`](../web/page_handler.py)를 사용합니다. 팀 PC GUI도 캡처 프레임 번호와 영상 타임스탬프를 유지한 기존 전송 경로를 씁니다. [팀 PC 통신 형식](../gimbal/team/docs/pc_communication.md)

## C++ SORT·OC-SORT·BoT-SORT

위 규약대로 만든 CPython 확장 세 개입니다. Python 클래스(`sort.py`, `ocsort.py`, `botsort.py`와 공통 `alt_tracker.py`, `bytetrack.py`)를 한 단계씩 옮겼고, Python이 비교 기준입니다.

| 파일 | 내용 |
|---|---|
| [`native_tracker_core.hpp`](native_tracker_core.hpp) | 공통 구현. Kalman, IoU, 배정, 트랙 수명, Python에 보이는 트랙·추적기 객체 |
| [`sort.cpp`](sort.cpp), [`ocsort.cpp`](ocsort.cpp), [`botsort.cpp`](botsort.cpp) | 종류만 정하고 공통 구현을 포함하는 모듈 3개 |
| [`test_native_trackers.py`](test_native_trackers.py) | Python과 프레임별 대조 시험 |

**빌드와 선택**

```sh
bash tracking_server/scripts/build_native_trackers.sh
cd tracking_server && python3 -m unittest src.track.test_native_trackers
```

- 빌드에는 `g++`와 `python3-config`만 씁니다. numpy 헤더는 필요 없습니다.
- 빌드된 확장이 있으면 서버를 다시 시작한 뒤 Tracker 메뉴의 SORT·OC-SORT·BoT-SORT가 C++로 표시되고 그 구현을 씁니다. `setup.sh`도 이 빌드를 실행합니다.
- **끄는 법:** 서버 환경에 `LWIR_NATIVE_TRACKERS=0`을 넣으면 확장이 있어도 Python을 씁니다.
- `native_tracker_core.hpp`가 확장보다 새 파일이면 다시 빌드할 때까지 Python을 씁니다.
- 시험은 확장이 없으면 건너뜁니다. 건너뛴 결과를 통과로 적지 않습니다.

**Python과 같게 맞춘 것**

- 입력 `[N, 6]`과 출력 `[x1, y1, x2, y2, id, conf, cls, det_idx]`, 출력 순서, 빈 프레임의 프레임 진행.
- 서버가 읽고 바꾸는 상태: `tracked`, `lost`, `frame_id`, `next_id`, `n_init`, `kf`, `new_this_frame`, `lost_this_frame`, 트랙의 `mean`, `cov`, `state`, `is_activated`, `last_det`, `feat`, `xyxy`, `xywh`. 서버가 붙이는 임의 속성도 받습니다.
- 카메라 움직임 보정: `tracker.kf.warp`를 예측 뒤 한 번 적용하고 비웁니다. OC-SORT는 저장한 관측에도 적용합니다.
- 배정은 SciPy `linear_sum_assignment`와 같은 순서·동점 규칙으로 옮겼습니다. float32 비용의 문턱 비교는 실행 중인 numpy 버전의 규칙(1.x와 2.x가 다름)을 따릅니다.
- BoT-SORT 외형 비용, `set_feat`, `attach_embedding`, `find`. 외형 입력 `{det_idx: 특징}`은 float32와 float64 모두 받습니다.

**Python을 그대로 부르는 부분**

- 가장자리 출생 규칙: `EDGE_BIRTH`가 0보다 클 때만 Python `edge_ok()`를 부릅니다(흔들림 보정 여백 포함).
- 위치 예측기(`POSP`)나 예측 상자 표시가 켜져 있으면 그 프레임의 예측 단계는 Python `pos_predict()`가 계산합니다.

**확인한 것 (2026-11-13, PC의 Linux x86-64와 Orange Pi)**

- CPython 3.8 + numpy 1.24, CPython 3.11 + numpy 2.4에서 각각 빌드하고 대조 시험 10개를 통과했습니다.
- 합성 검출 열(사람 이동·가림·미검출·낮은 점수·오검출·빈 프레임) 300프레임 묶음으로 ID·클래스·`det_idx`·출력 순서는 완전 일치, 상자·점수·Kalman 상태는 `1e-6` 안에서 비교했습니다.
- 조건: 카메라 움직임, 외형 특징, 클래스 구분, 확인 프레임 수 변경, 입력 끊김 초기화, 가장자리 규칙, 위치 예측기.
- 배정 단위 비교 16,000건(동점과 문턱과 같은 비용 포함) 일치.
- Orange Pi(aarch64, CPython 3.10 + numpy 2.2)에서 빌드하고 같은 시험을 통과했습니다. 같은 3,000프레임 묶음에서 다른 행 0개, 프레임당 SORT 0.54→0.016ms, OC-SORT 2.14→0.028ms, BoT-SORT 0.67→0.023ms였습니다. 줄어드는 시간은 프레임당 최대 약 2ms이고 정확도는 Python과 같습니다.
- 3,000프레임 한 묶음(카메라 움직임 포함)에서 세 추적기 모두 다른 행 0개. 프레임당 계산 시간은 PC에서 SORT 0.92→0.02ms, OC-SORT 4.19→0.03ms, BoT-SORT 2.17→0.04ms였습니다. 2만 4천 프레임 반복에서 메모리 증가는 없었습니다.

**확인하지 못한 것**

- 8090 서버에서 실제로 이 추적기로 바꿔 돌린 결과, 실제 카메라·녹화 검출로의 비교, 서버 전체 지연에 미치는 효과.
- 정답과 대조한 정확도. 이 작업은 Python과 같은 결과를 내는지만 봤습니다.
- 부동소수 계산 순서가 달라 `1e-6`보다 작은 차이는 남을 수 있습니다. 비용이 문턱과 그만큼 가까운 드문 경우에는 짝이 달라질 수 있습니다.
- C-BIoU, LostFirst, 학습 확인 판정기, ByteTrack의 `rebind`는 옮기지 않았습니다.

## 팀 runtime의 C++ 코드

팀의 [runtime C++ 추적기](https://github.com/lwir-deepx-tracking-system/runtime/tree/0fe016f7edd069900457a746dc4cc787569c5ae6/src/tracking)에는 ByteTrack과 OC-SORT가 있습니다. `track(vector<Detection>)`과 `reset()`을 사용합니다. 재사용하려면 `XYWH`↔`XYXY`를 변환하고, 검출 인덱스 `det_idx`와 서버가 읽는 추적 상태를 연결하는 어댑터가 필요합니다. 임계값, Kalman 상태 표현, 새 트랙 확인과 클래스 구분도 Python판과 맞춰서 검증합니다.

## 넘기기 전 확인

1. **PC에서 단위 비교.** 같은 배열로 Python과 C++의 IoU, 비용·짝, Kalman 평균·공분산을 대조합니다. 좌표·점수 허용 오차는 비교 전에 정하고, 오차가 임계값을 넘어 실제 매칭을 바꿨는지 확인합니다. 현재 확인 시간 테스트는 `tracking_server` 폴더에서 `python3 -m unittest src.track.test_confirm_time`으로 실행할 수 있습니다.
2. **프레임별 비교.** 같은 시퀀스, 같은 검출 순서·점수, 같은 설정으로 Python판과 C++판을 각각 실행합니다. 프레임 번호, 빈 입력, 타임스탬프·초기화 이벤트, GMC와 외형 입력을 함께 고정합니다. 출력 행 수·순서·ID·클래스·`det_idx`, 활성/분실 상태가 일치해야 합니다. 좌표·점수는 정한 허용 오차 안에서 비교합니다. 먼저 달라지는 프레임과 상태를 기록합니다.
3. **서버 연결 확인.** 카메라와 짐벌 없이 PC의 모의 입력 또는 저장 영상으로 한 사람·두 사람, 교차·가림·재등장, 무검출, 재생 전환·탐색, 추적기 변경을 확인합니다. KeepID 사용 시에는 추적기 직후와 최종 출력 양쪽을 봅니다. 웹 화면의 박스·ID가 같은 프레임에 나오는지 확인하고, 팀 PC GUI도 사용한다면 별도로 영상과 메타데이터를 맞춰 봅니다.
4. **M1 속도 비교.** 결과가 맞은 뒤 같은 녹화·모델·옵션으로 추적 계산 시간, 전체 출력 간격, 프레임 처리 지연의 중앙값·p95를 비교합니다. 출력 개수와 건너뛴 프레임도 함께 기록합니다.

인계에는 기준 커밋, 옮긴 파일과 빌드 명령, 선택 방법, 비교에 사용한 입력·설정, 처음 다른 프레임 또는 일치한 프레임 수, M1 측정 여부를 남깁니다. 기존 실시간 추적 프리셋은 Python이며, 아래 C++ ByteTrack은 별도 선택입니다.

## C++ ByteTrack 설치와 비교

팀 [runtime](https://github.com/lwir-deepx-tracking-system/runtime)의 ByteTrack을 Python ByteTrack과 비교합니다. 입출력을 바꾼 C++ 파일 4개는 [bytetrack_runtime](bytetrack_runtime/)에 있으며, 변경 줄에 `// LWIR I/O:` 표시가 있습니다. 빌드는 이 파일들을 직접 쓰고, 나머지 의존 파일은 팀 runtime의 `0fe016f7edd069900457a746dc4cc787569c5ae6`에서 읽습니다.

```sh
python tracking_server/src/track/build_bytetrack.py /path/to/runtime tracking_server/models/links/bytetrack.so
python -m tracking_server.src.track.check_bytetrack tracking_server/models/links/bytetrack.so --cache /path/to/exact_detections.jsonl
```

Windows에서는 출력 확장자를 `.dll`로 바꾸고 필요하면 `--cxx`로 MinGW `g++` 경로를 지정합니다. 시험용 캐시에는 프레임마다 `tracker_input_xyxy_conf_class`와 `tracker_reset`이 있어야 합니다. 두 구현 모두 같은 검출을 프레임 순서대로 받고, 빈 프레임도 업데이트하며, 시퀀스 경계에서만 ID를 초기화합니다.

### 하건님 원본 입출력 수정 전후

기준은 팀 runtime의 `0fe016f7edd069900457a746dc4cc787569c5ae6`입니다. 아래에서 `+`는 추가한 줄입니다. 원본의 `track(const std::vector<Detection>&) → std::vector<Track>` 호출은 그대로 사용합니다.

```diff
--- a/include/common/detection.hpp
+++ b/include/common/detection.hpp
@@ -19,2 +19,3 @@
     std::string class_name;
+    int source_index = -1;
 };
--- a/include/tracking/bytetrack_tracker.hpp
+++ b/include/tracking/bytetrack_tracker.hpp
@@ -34 +34,2 @@
     int class_id = -1;
+    int source_index = -1;
--- a/include/common/track.hpp
+++ b/include/common/track.hpp
@@ -30,2 +30,3 @@
     bool predicted = false;
+    int source_index = -1;
 };
--- a/src/tracking/bytetrack_tracker.cpp
+++ b/src/tracking/bytetrack_tracker.cpp
@@ -481,2 +481,3 @@
     t.class_id = det.class_id;
+    t.source_index = det.source_index;
     t.frame_id = frame_id_;
@@ -505 +506,2 @@
         det->class_id = d.class_id;
+        det->source_index = d.source_index;
@@ -772,2 +774,3 @@
         out.confidence = t->score;
+        out.source_index = t->frame_id == frame_id_ ? t->source_index : -1;
         tracks.push_back(out);
```

**입력**은 8090의 `[x1, y1, x2, y2, conf, cls]`를 원본의 `Detection`으로 옮깁니다. `x=x1`, `y=y1`, `width=x2-x1`, `height=y2-y1`로 계산하고, `source_index=i`에는 **필터를 거친 이번 프레임 검출 배열**에서의 0부터 시작하는 행 번호를 넣습니다. 입력 배열의 열은 6개 그대로입니다.

**출력**은 원본이 계산한 `Track` 박스를 `[x, y, x+width, y+height, track_id, confidence, class_id, source_index]` 순서로 돌려줍니다. 예를 들어 검출 배열의 세 번째 박스가 트랙 17과 매칭되면 `track_id=17`이고 출력의 마지막 열은 `source_index=2`입니다. 트래커가 추정한 박스를 사용하므로 입력 박스와 출력 박스의 좌표는 달라질 수 있습니다. 현재 프레임에 대응하는 검출이 없는 행은 `source_index=-1`이며, 이 서버 연결에서는 예측 행이나 유효한 검출 번호가 없는 행을 오류로 처리합니다. [좌표 변환과 행 검사](bytetrack.cpp#L41)

위 diff의 주석은 [ByteTrack 매칭·반환](bytetrack_runtime/src/tracking/bytetrack_tracker.cpp#L482), [Detection 입력](bytetrack_runtime/include/common/detection.hpp#L20), [STrack 상태](bytetrack_runtime/include/tracking/bytetrack_tracker.hpp#L35), [Track 출력](bytetrack_runtime/include/common/track.hpp#L31)에서 바로 볼 수 있습니다. 좌표 변환은 모델뱅크의 [서버 연결부](bytetrack.cpp#L51)에 있습니다. 빌드 뒤에는 위의 `check_bytetrack`으로 검출 행 번호를 확인합니다.

### 라이브러리 연결과 재생

소스 파일은 [`bytetrack.py`](bytetrack.py)와 [`bytetrack.cpp`](bytetrack.cpp)입니다. 이 C++ 소스는 위의 CPython 확장과 다른 C ABI를 사용하므로 전용 명령으로 빌드합니다.

1. Orange Pi용 ARM64 **`bytetrack.so`**를 `tracking_server/models/links/`에 넣습니다. 완성된 파일을 임시 이름으로 복사한 뒤 그 자리에서 이름을 바꾸면 복사 도중 읽히지 않습니다. 구버전 설치의 `team_bytetrack.so`도 인식합니다.
2. 이 버전 서버에서는 새로 넣은 라이브러리의 ABI를 확인해 메뉴를 자동으로 엽니다. `.cpp` 소스만 복사해서는 실행되지 않습니다. 라이브러리 경로를 따로 지정할 때만 `--replay-bytetrack-library` 또는 `LWIR_REPLAY_BYTETRACK_LIBRARY`를 씁니다.
3. 카메라 화면에서 **Tracker → ByteTrack** 줄을 누르면 C++ 라이브러리가 있을 때 C++로 시작합니다(없거나, 녹화 재생 중이거나, 따라가는 중이거나, 브리지가 C++ ID를 받지 못하면 Python). **ByteTrack 옵션 → C++**을 직접 골라도 됩니다. C++에서는 화면, 클릭 선택, 짐벌에 같은 C++ 박스와 ID를 전달합니다. 함께 제공한 v25/v27 브리지를 다시 빌드하고 서버를 다시 시작해야 합니다. 브리지가 새 연결 규약을 보고하지 않으면 화면 비교만 허용하고 클릭을 막습니다. **Python**을 고르면 원래 추적 경로로 돌아갑니다.
4. 녹화 재생 중에 **Tracker → ByteTrack → C++**을 고르고 확인하면 해당 녹화를 처음부터 다시 재생합니다. 시작 전에는 **녹화 재생 → 추적기**에서 엔진을 지정할 수 있습니다. 카메라와 녹화의 엔진 선택은 서로 독립입니다.
5. 카메라에서 Python ByteTrack을 쓸 때 **위치 재결합**과 **새 ID 확인 시간(ms)**을 따로 바꿀 수 있습니다. `0 ms`는 추가 확인을 끕니다. 재생 중에는 평가 조건을 유지하기 위해 이 옵션을 잠급니다.

C++ 경로는 최종 필터를 거친 검출 행을 받아 독립 추적 상태를 유지합니다. 녹화 전환이나 새 반복 회차에서 상태를 초기화하고, 일시정지 중에는 마지막 결과를 사용합니다. 카메라 비교에서는 기존 추적도 별도로 계속 실행되므로, 이 화면의 전체 지연으로 C++ 단독 교체의 속도를 평가할 수 없습니다. 비교용 ByteTrack의 점수 문턱과 XYWH Kalman은 Python 쪽에 맞췄지만, 확인·매칭 규칙과 출력은 완전히 같지 않습니다. 출력 수가 늘거나 줄어도 정답을 대조하지 않으면 정확도가 개선됐는지 판단할 수 없습니다.

### 화면·클릭·짐벌이 공유하는 출력

- [`live_publish()`](../web/page_handler.py)는 현재 선택한 엔진의 출력 행 하나를 화면과 [`live_tracks.py`](../gimbal/live_tracks.py)에 넘깁니다. 일반 추적기 팩토리를 통해 교체한 BoT-SORT·OC-SORT·SORT도 같은 경로를 사용합니다. 별도의 C++ 짐벌 로직을 추적기마다 작성하지 않습니다.
- 웹 클릭은 [`03_gimbal_control.js`](../../page/js/03_gimbal_control.js) → `/gimbal/click` → `live_tracks.click()` 순서입니다. 클릭한 상자의 ID를 골라 해당 프레임의 `F`/`G` 명령을 먼저 전달하고 `A ID`를 보냅니다. 기존 좌표 변환, 속도와 제어법은 유지합니다.
- 엔진·추적 상태가 바뀌면 기존 선택을 해제합니다. 브리지가 `H 번호`를 받아 정지를 처리한 뒤 `HALTED 번호`를 응답해야 새 선택을 허용합니다. 오래된 결과, 전환 중, 녹화 재생에서는 클릭으로 짐벌을 움직이지 않습니다. 서버가 처음 켜지거나 브리지가 재시작될 때도 같은 확인을 합니다.
- C++ ByteTrack 전용 경로에 Python의 KeepID·추가 확인 시간·위치 재결합이 적용되는 것은 아닙니다. 화면은 해당 옵션을 구분합니다. 공유 출력 연결만으로 추적기의 ID 변경이나 오매칭이 없어지는 것은 아닙니다.
- 이미 읽은 `bytetrack.so`를 교체하면 실행 중 객체는 예전 코드를 유지합니다. GUI의 **서버 재시작 필요** 안내를 따라 다시 시작해야 새 파일을 읽습니다. 새 공유 라이브러리는 임시 파일로 완전히 복사한 뒤 이름을 바꿔 교체합니다. 실행 중 파일을 직접 덮어쓰지 않습니다.

재생 화면에서 **끝까지 다시 계산**을 누르면 선택한 추적기로 첫 장부터 한 번 재생합니다. 슬라이더로 이동하면 해당 장에서 멈추고 첫 장부터 그 장까지 채점한 누적값을 표시합니다. 아직 그 위치까지 채점하지 않았다면 다시 계산해 결과를 만들 수 있습니다. **최종 보기**는 재생 위치를 유지한 채 같은 녹화와 추적기의 완주 결과를 표시합니다.
