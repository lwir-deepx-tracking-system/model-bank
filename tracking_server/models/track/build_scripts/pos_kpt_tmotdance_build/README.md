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
