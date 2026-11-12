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

## C++을 Python에 연결하는 방법

1. 공통 계산 하나를 C++17 공유 라이브러리(`.so`)로 만들고 Python 어댑터에서 `ctypes`로 호출합니다. 현재 프로젝트의 [`native_post.py`](../detect/native_post.py)와 [`setup.sh`](../../scripts/setup.sh)에서 로드·빌드 방식을 볼 수 있습니다. 이식 대상의 `.so`는 검출 후처리 `post.so`와 별도로 둡니다.
2. `extern "C"` 함수에 배열 자료형·개수·출력 버퍼 크기·실패 반환값을 명시합니다. Python에서 연속 배열을 넘기고 `argtypes`와 `restype`를 지정합니다. C++이 다음 호출까지 쓸 입력은 자체 소유 버퍼에 보관합니다. C++ 예외를 Python 호출 경계 밖으로 내보내지 않습니다.
3. 전체 추적기로 확장하면 Python 객체 하나가 C++ 인스턴스 하나의 생성·갱신·종료를 맡게 합니다. 첫 통합은 Python판과 C++판을 실행 설정으로 고를 수 있게 하고, 프레임을 처리하는 도중에 한 인스턴스의 구현을 바꾸지 않습니다. 실시간 GUI에는 기존 추적기 이름과 설정 키를 유지합니다.
4. 8090 브라우저는 [`live_publish()`](../web/page_handler.py)가 그린 `/stream`을 받습니다. 팀 PC GUI도 기존 전송 경로를 씁니다. 웹에서 박스나 ID를 별도로 계산할 필요가 없고, 팀 PC 전송 시에는 캡처 프레임 번호와 영상 타임스탬프를 유지합니다. [팀 PC 통신 형식](../gimbal/team/docs/pc_communication.md)

## 넘기기 전 확인

1. **PC에서 단위 비교.** 같은 배열로 Python과 C++의 IoU, 비용·짝, Kalman 평균·공분산을 대조합니다. 좌표·점수 허용 오차는 비교 전에 정하고, 오차가 임계값을 넘어 실제 매칭을 바꿨는지 확인합니다. 현재 확인 시간 테스트는 `tracking_server` 폴더에서 `python3 -m unittest src.track.test_confirm_time`으로 실행할 수 있습니다.
2. **프레임별 비교.** 같은 시퀀스, 같은 검출 순서·점수, 같은 설정으로 Python판과 C++판을 각각 실행합니다. 프레임 번호, 빈 입력, 타임스탬프·초기화 이벤트, GMC와 외형 입력을 함께 고정합니다. 출력 행 수·순서·ID·클래스·`det_idx`, 활성/분실 상태가 일치해야 합니다. 좌표·점수는 정한 허용 오차 안에서 비교합니다. 먼저 달라지는 프레임과 상태를 기록합니다.
3. **서버 연결 확인.** 카메라와 짐벌 없이 PC의 모의 입력 또는 저장 영상으로 한 사람·두 사람, 교차·가림·재등장, 무검출, 재생 전환·탐색, 추적기 변경을 확인합니다. KeepID 사용 시에는 추적기 직후와 최종 출력 양쪽을 봅니다. 웹 화면의 박스·ID가 같은 프레임에 나오는지 확인하고, 팀 PC GUI도 사용한다면 별도로 영상과 메타데이터를 맞춰 봅니다.
4. **M1 속도 비교.** 결과가 맞은 뒤 같은 녹화·모델·옵션으로 추적 계산 시간, 전체 출력 간격, 프레임 처리 지연의 중앙값·p95를 비교합니다. 출력 개수와 건너뛴 프레임도 함께 기록합니다.

인계에는 기준 커밋, 옮긴 파일과 빌드 명령, 선택 방법, 비교에 사용한 입력·설정, 처음 다른 프레임 또는 일치한 프레임 수, M1 측정 여부를 남깁니다. 이 문서는 이식 절차이며 현재 추적기 구현은 Python입니다.
