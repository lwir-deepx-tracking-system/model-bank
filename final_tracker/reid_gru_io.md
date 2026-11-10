# 재식별 모델 2개와 GRU 확인 게이트의 입력·출력 (2026-11-10)

사람 외형 비교(re-ID) 모델 **RepVGG-A0_(TMOT,CAMEL)**, **ViT-lite_(FLIR)** 과 트랙 확인용 **GRU32 확인 게이트**가 무엇을 받고 무엇을 내는지 정리한 문서입니다.
값은 DXNN 안의 컴파일 메타데이터(`*_compile_metadata.json`의 `rmaps.npu_0.inputs/outputs`)와 실행 코드 `final_tracker/tracker/lwir_tracker_m1.py`에서 그대로 옮겼습니다. 장치에서 다시 재지는 않았습니다.

## 한눈에 보기

| 모델 | 파일 | 입력 | 출력 | 실행 위치 |
| --- | --- | --- | --- | --- |
| RepVGG-A0_(TMOT,CAMEL) INT8 | `final_tracker/dxnn/repvgg_tmot_camel_reid.dxnn` (main) | `images` uint8 NHWC `[1,128,64,3]` | `embedding` float32 `[1,128]` | NPU 코어 2 (전부 NPU, 29 layer) |
| ViT-lite_(FLIR) INT8 | `reid/vit_lite1_reid.dxnn` (**main에는 없음**, 아래 참고) | `images` uint8 NHWC `[1,128,64,3]` | `embedding` float32 `[1,128]` | NPU 코어 2 (101 layer) + 마지막 단계만 CPU |
| GRU32 확인 게이트 | `final_tracker/tracker/gate_gru32_frozen.json` (main) | float `[N,11,19]` (트랙 N개 × 11프레임 × 특징 19개) | float `[N]`, 0~1 확률 | M1 CPU (numpy), NPU 사용 안 함 |

두 재식별 모델은 입력·출력 모양과 이름이 같아서 추적기에서 파일만 바꿔 끼울 수 있습니다.

## 1. RepVGG-A0_(TMOT,CAMEL) INT8 — `repvgg_tmot_camel_reid.dxnn`

| 항목 | 값 |
| --- | --- |
| 입력 이름 / 모양 / 형식 | `images` / `[1,128,64,3]` (높이 128, 너비 64, 채널 3) / uint8, NHWC |
| 입력 만드는 법 | 1) 8비트 흑백 프레임에서 사람 상자를 프레임 안으로 자름(가로·세로 2 px 미만이면 건너뜀) 2) `cv2.resize(..., (64,128), INTER_LINEAR)`로 비율 무시하고 늘림(letterbox 없음) 3) `GRAY2RGB`로 같은 값을 3채널에 복제 |
| 호스트 정규화 | 없음. 0~255 uint8 그대로 넣습니다(/255, 평균·표준편차 처리 안 함) |
| 출력 이름 / 모양 / 형식 | `embedding` / `[1,128]` / float32 |
| 출력 뜻 | 사람 외형을 나타내는 128차원 벡터. 모델 출력은 정규화되어 있지 않습니다 |
| 호스트 후처리 | `e / max(‖e‖, 1e-12)`로 L2 정규화 → 두 벡터의 내적이 코사인 유사도(−1~1, 클수록 같은 사람) |
| batch | 1 (크롭 한 장씩) |
| 컴파일 | DX-COM 2.4.1-rc.1, ONNX 입력 `images` `[1,3,128,64]`, 보정 256장 EMA, ppu 없음 |
| 파일 | 7,451,967 bytes, sha256 `dc19d7862a77ffa1d7c7f71f84283e86c33267d9e741b59070b839cb1921ff53` |
| 시간 | 크롭 한 장 NPU 코어 2에서 2.3 ms |

## 2. ViT-lite_(FLIR) INT8 — `vit_lite1_reid.dxnn`

| 항목 | 값 |
| --- | --- |
| 입력 | RepVGG-A0_(TMOT,CAMEL)과 같습니다: `images` uint8 NHWC `[1,128,64,3]`, 같은 크롭·resize·3채널 복제, 호스트 정규화 없음 |
| 출력 | `embedding` float32 `[1,128]`, 호스트에서 L2 정규화 후 코사인 비교 |
| 내부 구성 | NPU가 `[1,128,1,1]`을 내고, DXNN 안의 CPU 단계가 `[1,128]`로 펴서 `embedding`으로 내보냅니다. 사용자는 `embedding`만 받으면 됩니다 |
| 컴파일 | DX-COM 2.4.1-rc.1, ONNX 입력 `images` `[1,3,128,64]`, 보정 128장 EMA |
| 파일 | 1,463,301 bytes, sha256 `db9ca9f1a71fa3e994ccf55daeed4b927578aa772c0b72927f68014fd27c0cef` |
| 시간 | 크롭 한 장 NPU 코어 2에서 3.0 ms (준비 포함 5.9 ms) |
| 위치 | 2026-11-06 정리(`d686dba`) 때 main에서 빠졌습니다. `final-tracker-20261103`, `input-640x480` 브랜치의 `reid/vit_lite1_reid.dxnn`과 `reid/vit_lite1_reid_compile_metadata.json`에 그대로 있습니다 |
| 주의 | M1 INT8 실측에서 재식별에 도움이 되지 않아 최종 추적기는 RepVGG-A0_(TMOT,CAMEL)을 씁니다 |

## 3. GRU32 확인 게이트 — `gate_gru32_frozen.json`

DXNN이 아니라 가중치가 들어 있는 JSON이고, 추적기가 numpy로 직접 계산합니다.

| 항목 | 값 |
| --- | --- |
| 역할 | 새로 생긴 사람 트랙을 화면에 내보낼지(확정할지) 정합니다. 이미 확정된 트랙은 건드리지 않습니다 |
| 입력 모양 | `[N, 11, 19]` float. N = 이번 프레임에 판정할 트랙 수, 11 = 과거 10프레임 + 현재 프레임(오래된 것이 앞), 19 = 프레임당 특징 수 |
| 기록이 짧을 때 | 앞쪽을 0 벡터로 채웁니다. 이때 19번째 특징 `valid`도 0이라 "빈 칸"임을 모델이 압니다 |
| 출력 | `[N]` float, 0~1. 그 트랙이 진짜 사람일 확률 |
| 출력 쓰는 법 | 0.3 이상이면 트랙 확정(`thr_out` 0.3). `thr_low`는 0이라 쓰지 않습니다 |
| 구조 | GRU 1층(은닉 32, 시작 상태 0) → 마지막 은닉 상태 → 선형 1개 → sigmoid. 가중치 5,121개 |
| JSON 키 | `hidden` 32, `n_features` 19, `window` 11, `thr_out`, `thr_low`, `weights`(`gru.weight_ih_l0` `[96,19]`, `gru.weight_hh_l0` `[96,32]`, `gru.bias_ih_l0` `[96]`, `gru.bias_hh_l0` `[96]`, `head.weight` `[1,32]`, `head.bias` `[1]`) |
| 시간 | 프레임당 평균 0.18 ms, 호출 한 번 0.58 ms (M1 CPU) |

### 프레임당 특징 19개 (이 순서 그대로)

상자 좌표는 640×512 캔버스 기준입니다(640×480 영상은 위쪽 16 px 아래에 놓입니다). h = 상자 높이, w = 상자 너비.

| 번호 | 이름 | 값 |
| --- | --- | --- |
| 1 | matched | 이번 프레임에 검출과 매칭되면 1, 아니면 0 |
| 2 | matched_low | 낮은 점수 검출과 매칭되면 1 |
| 3 | conf | 매칭된 검출의 신뢰도 |
| 4 | h | h / 480 |
| 5 | aspect | w / h |
| 6 | logh | ln(h) |
| 7 | speed | 직전 프레임 대비 중심 이동 거리 / h (최대 5) |
| 8 | kf_iou | 칼만 예측 상자와 검출 상자의 IoU |
| 9 | overlap | 다른 사람 트랙 상자와의 최대 IoU |
| 10 | contrast | (상자 안 평균 − 상자 둘레 띠 평균) / (띠 표준편차 + 1), −8~8 |
| 11 | texture | 상자 안 표준편차 / 64 (최대 4) |
| 12 | hits | 누적 매칭 수(최대 30) / 10 |
| 13 | misses | 누적 놓침 수(최대 30) / 10 |
| 14 | cmiss | 연속 놓침 수(최대 30) / 10 |
| 15 | age | 태어난 뒤 프레임 수(최대 60) / 30 |
| 16 | edge | 영상 가장자리까지 가장 가까운 거리 / h, −1~2 |
| 17 | lost_iou | 끊긴(lost) 사람 트랙의 예측 상자와의 최대 IoU |
| 18 | lost_dist | 끊긴 사람 트랙 중심까지 가장 가까운 거리 / h (최대 5, 없으면 5) |
| 19 | valid | 실제 기록이면 1, 채움 칸이면 0 |

특징 순서·개수(19)와 창 길이(11)는 추적기 코드와 JSON이 일치해야 하며, 다르면 추적기가 시작할 때 멈춥니다.
