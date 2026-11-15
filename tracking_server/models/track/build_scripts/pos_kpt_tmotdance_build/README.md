# 위치 기억 GRU·Mamba (관절 입력, TMOT+DanceTrack, 2026-11-15) 만든 코드

`models/track/pos_{gru,mamba}_kpt_tmotdance_20261115.{npz,json}` 를 만든 스크립트입니다.

1. `paths.example.json` 을 `paths.json` 으로 복사하고 자료 위치를 적습니다.
2. `det_run.py`: 근거리·원거리 자세 검출기를 장치와 같은 방식(같은 출력 갈래, 서버 NMS, 점수 0.10 초과)으로 TMOT·DanceTrack 프레임에 돌립니다.
3. `gmc_run.py`: 카메라 움직임(프레임 사이 변환)을 구합니다.
4. `assoc_build.py`, `track_run.py`: 정답 상자와 검출을 잇고, 사람마다 시계열을 만듭니다.
5. `pos_train.py`: GRU·Mamba 를 학습합니다(설정 a, 처음부터 lr 2e-3, 30 epoch, val-A 로 선택).
6. `export_models.py`: 서버가 읽는 npz·json 으로 내보냅니다. `parity_both.py`: 서버 numpy 계산과 학습 torch 계산을 비교합니다.
7. `pos_eval.py`: val-B 로 Kalman·옛 파일·새 파일을 비교합니다.

공통 함수는 `pos_lib.py`, `prcommon.py` 에 있습니다. 입력 68개를 만드는 식은 서버 `src/track/position_memory.py` 와 같습니다.

## 11-16 고친 판 (`fix_*.py` → `models/track/pos_gru_kpt_tmotdance_20261116`)

11-15 GRU 는 우리 카메라에서 놓친 사람 상자를 줄이고 위로 올렸습니다. 원인은 위 스크립트의 학습 자료 만들기였습니다(자세한 숫자는 `models/README.md` 위치 기억 항목).

- `fix_lib.py`: `pos_lib`·`prcommon` 을 그대로 가져다 쓰고 세 가지만 바꿉니다. (1) 화면 가장자리를 실제 화면 줄로 잽니다(TMOT 16~496, DanceTrack 은 회색 띠 안쪽 76~436). (2) 정답 상자가 화면 가장자리(2 px)에 닿은 프레임은 목표를 만들지 않습니다(잘린 정답에 키 변환표를 쓰면 더 작고 높은 상자가 목표가 됨). (3) 놓친 s 장째는 그 프레임(t+s)의 카메라 움직임을 씁니다(서버 11-16 과 같음). `EDGES[0] = False` 이면 가장자리 입력 넷을 늘 5 로 둡니다(새 파일). 옛 설정으로 돌리면 `pos_lib.anchors` 와 값이 똑같습니다(36,060개, 차이 0).
- `fix_train.py <출력 폴더> gru_kpt --no-edges`: `pos_train.py` 설정 a 와 같은 학습(처음부터 lr 2e-3, 30 epoch, 판 4개, seed 0, val-A 로 선택).
- `fix_export.py <출력 폴더> gru_kpt <내보낼 폴더> pos_gru_kpt_tmotdance_20261116`: npz·json 으로 내보내고(가장자리 입력 없이 학습했으면 npz 에 `opt_no_edges` = 1), 서버 `pos_predict` 와 학습 계산을 비교합니다(서버가 검출 대신 Kalman 상자를 쓰게 되는 경우, 곧 둘의 IoU 가 0.5 미만인 트랙은 세어서 뺌: 옛 코드에도 있는 차이).
- `fix_valB.py <옛 npz> <새 npz> <결과 json> --no-edges`: val-B 에서 옛 파일과 새 파일을 같은 사람·같은 프레임, 잘린 정답을 뺀 같은 목표로 비교합니다(규칙: 새 ≤ 옛 + 0.005).
- `fix_valB_server.py <npz> <결과 json> [일꾼 수] [--fix]`: val-B 를 학습 코드가 아니라 서버 `position_memory.pos_predict`(GKF, 앞 프레임 검출 전부를 관절 찾기에 줌)로 다시 잽니다.
- `paths.json` 에 `root`(data/assoc, data/det, data/gmc 가 있는 폴더)를 더 적습니다.
