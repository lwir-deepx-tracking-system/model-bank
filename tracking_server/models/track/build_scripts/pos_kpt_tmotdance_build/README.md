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

11-15 GRU 는 우리 카메라에서 놓친 사람 상자를 줄이고 위로 올렸습니다. 가장 큰 원인은 화면 가장자리 입력이 공개 자료 카메라의 사람 배치를 배운 것이고(새 파일은 가장자리 입력 없음), 위 스크립트의 학습 자료 만들기 잘못 둘(아래 (1)(2))은 작았습니다(둘만 고친 판: Kalman 까지 차이의 41 %). 자세한 숫자와 "눈가림 시험이 아님" 설명은 `models/README.md` 위치 기억 항목.

- `fix_lib.py`: `pos_lib`·`prcommon` 을 그대로 가져다 쓰고 세 가지만 바꿉니다. (1) 화면 가장자리를 실제 화면 줄로 잽니다(TMOT 16~496, DanceTrack 은 회색 띠 안쪽 76~436). (2) 정답 상자가 화면 가장자리(2 px)에 닿은 프레임은 목표를 만들지 않습니다(잘린 정답에 키 변환표를 쓰면 더 작고 높은 상자가 목표가 됨). (3) 놓친 s 장째는 그 프레임(t+s)의 카메라 움직임을 씁니다(서버 11-16 과 같음). `EDGES[0] = False` 이면 가장자리 입력 넷을 늘 5 로 둡니다(새 파일). 옛 설정으로 돌리면 `pos_lib.anchors` 와 값이 똑같습니다(36,060개, 차이 0).
- `fix_train.py <출력 폴더> gru_kpt --no-edges`: `pos_train.py` 설정 a 와 같은 학습(처음부터 lr 2e-3, 30 epoch, 판 4개, seed 0, val-A 로 선택).
- `fix_export.py <출력 폴더> gru_kpt <내보낼 폴더> pos_gru_kpt_tmotdance_20261116 [우리 카메라 결과 한 줄]`: npz·json 으로 내보내고(가장자리 입력 없이 학습했으면 npz 에 `opt_no_edges` = 1, 새 카메라 보정 순서로 학습했으면 `opt_warp_fix` = 1 — 이 표시가 없는 파일은 서버가 옛 순서로 돌림), 서버 `pos_predict` 와 학습 계산을 비교합니다(서버가 검출 대신 Kalman 상자를 쓰게 되는 경우, 곧 둘의 IoU 가 0.5 미만인 트랙은 세어서 뺌: 옛 코드에도 있는 차이; 관절 없는 큰 사람의 Kalman 규칙은 신경망 비교에서 끔). json 의 `status`(화면에 앞 120자)는 끝에 `장치 미확인` 을 붙입니다. 11-16 검토 뒤 같은 가중치(`runs/fix2` best)로 다시 내보냄: sha256 ea542076…, 비교 값은 처음과 같음(2.8e-7 · 7.6e-7 px).
- **남은 학습 자료 문제(고치지 않음):** 목표는 정답 상자를 키 하나로 찾는 변환표(`assoc_build.py` 의 `tgt_det_*`, 머리~골반 방식)인데, 보이는 관절이 없는 64 px 이상 사람에게 검출기는 대개 전신 상자를 냅니다. 그 목표는 검출 상자의 0.55~0.69배라 모델이 그런 사람을 줄여 그립니다. 고치려면 그 프레임 검출기의 상자 방식(관절 있음 = 머리~골반, 없음 = 전신)대로 목표를 만들거나 방식이 다른 목표를 빼고 다시 학습하고, npz 에 `opt_box_mode_targets` = 1 을 넣습니다(그때 서버의 Kalman 규칙이 꺼짐).
- 이 폴더만으로 돌리려면 `m4_model.py`(`../box_refine_tmotdance_build/` 에 사본)를 이 폴더나 `../ref_code/` 에, 분할 목록 `splits.json`(또는 `../PREREG_POS_REFINE_20261115.json`)을 두어야 합니다(저장소에는 없음).
- `fix_valB.py <옛 npz> <새 npz> <결과 json> --no-edges`: val-B 에서 옛 파일과 새 파일을 같은 사람·같은 프레임, 잘린 정답을 뺀 같은 목표로 비교합니다(규칙: 새 ≤ 옛 + 0.005).
- `fix_valB_server.py <npz> <결과 json> [일꾼 수] [--fix]`: val-B 를 학습 코드가 아니라 서버 `position_memory.pos_predict`(GKF, 앞 프레임 검출 전부를 관절 찾기에 줌)로 다시 잽니다.
- `paths.json` 에 `root`(data/assoc, data/det, data/gmc 가 있는 폴더)를 더 적습니다.
