상자 다듬기 학습 GRU 2판 (2026-11-15, 서버 모드 box_refine=gru2): 학습, 고르기, 채점, 내보내기, 눈으로 보기, 서버 비교 스크립트 사본.
가중치: ../../box_refine_gru_v2_tmotdance_20261115.npz (+ .json). 결과 요약: REPORT.md (끝의 Addendum 이 눈으로 본 결함과 변환표 고침), 서버가 계산할 것: SERVER_NOTES.md.

작업 폴더(공개 저장소 밖)의 배치 그대로 돌립니다:
  <작업 폴더>/scripts/            첫 판 스크립트 (../box_refine_tmotdance_build/ 의 prcommon, pos_lib, ref_lib, ref_eval, det_run, gmc_run, track_run ...) + paths.json
                                  (paths.json.example 을 paths.json 으로 복사해 자기 경로를 적음)
  <작업 폴더>/verify/combo/       combo_lib.py (../box_refine_tmotdance_build/v2/combo_lib_verify.py)
  <작업 폴더>/verify/gen_outputs.py  여기의 lut_fit_gen_outputs.py (fit_lut: 학습 분할 변환표 -> verify/lut.pkl; 이것을 gru_v2/data/lut_train.pkl 로 복사)
  <작업 폴더>/v2/data/{det,gmc}/  장치 녹화의 검출·카메라 움직임 (../box_refine_tmotdance_build/v2/dev_det.py)
  <작업 폴더>/gru_v2/scripts/     이 폴더의 g2_*.py
  <작업 폴더>/gru_v2/PREREG_GRU_V2.json  미리 적은 규칙 (학습과 val-B 숫자보다 먼저 씀)

순서:
  g2_lutfix.py                    변환표의 검출 상자 170 px 이상 두 줄을 130-170 px 줄로 (data/lut_train_fix.pkl; g2_lib 가 이 파일을 읽음)
  g2_lutcheck.py                  (근거) 학습 분할에서 줄마다 목표 / 검출 비율, 골반 관절이 상자 안 어디에 있는지
  g2_data.py train 6 7            학습 조각          g2_grid.py -> g2_train.py   8개 학습 (PC CPU)
  g2_data.py rows valA 7          val-A 줄           g2_eval.py valA             고르기 (GRU, 크기 필터, 맞춘 One Euro)
  g2_export.py                    npz + json, numpy = torch
  g2_data.py rows valB 7          val-B 줄           g2_eval.py valB             규칙 P1-P5 (한 번; 변환표 고친 뒤 같은 모델로 한 번 더)
  g2_device.py <녹화 ...>         장치 속도 녹화 (정답 없음)
  g2_eyecheck.py <출력 폴더> <npz> <raw16 폴더> <녹화 ...>   추적기 상자(흰색)와 2판 출력(초록)을 그린 그림 (그림은 저장소에 넣지 않음)
  g2_server_parity.py <tracking_server> <fixture.npz> <g2 scripts dir> <npz> <rows_valB.pkl> [녹화 ...]   서버 모드 = g2_lib 인지 모든 줄에서 비교 + 서버 시험용 조각(src/track/box_refine_gru2_fixture.npz:
                                  val-B 두 구간의 검출 상자·관절과 g2_lib 출력, 정답 없음)
  g2_cost.py, g2_server_cost.py <tracking_server> <npz> [gru2|body]   PC 계산 시간 (엔진만 / 서버 Refiner.step)

상체 네모로 통일 (서버 모드 box_refine=body, 11-16): 변환표는 같은 npz 의 lut_tab, 크기 필터 값은 서버 BODY (0.4 Hz, beta 0 = val-A 에서 맞춘 변환표 + One Euro 의 크기 값).
  g2_body.py <tracking_server> <npz> <rows_valB.pkl> <이름> [녹화 ...]   서버 Refiner 를 그대로 돌려 추적기 상자 / body / gru2 숫자 (val-B, 장치 속도 녹화):
                                  추적기 상자 대비 키·폭(크기 구간별), 키 변화, 조준점 떨림, 뒤처짐, 트랙 첫 줄들, 검출 모델 바뀌는 프레임의 변화.
                                  body_c(중심 고정), body5(5 Hz, beta 3)는 비교용
  g2_body_eye.py ...              가까운(장치 녹화) / 중간(다른 장치 녹화 구간, 검출을 그 프레임만 다시 돌림) / 먼(공개 열화상 val-B) 사람 그림

저장소의 가중치 파일은 다시 학습한 판(v3b: 고친 목표로 변환표와 가중치를 다시 만든 것)입니다. 작업 폴더의 gru_v2/v3b/scripts 에서 같은 스크립트의 사본으로 만들었고
(이 폴더의 v3b/ 가 그 사본; 목표 고치기 g2_lib.fix_target,
PREREG_GRU_V2.json 의 amendment 2, 3), 내보낸 파일 이름은 같습니다. g2_server_parity.py 의 <g2 scripts dir> 에 그 폴더의 scripts 를 주면 그 판의 기준 코드와 비교합니다.

주의: 실제 선택(g2_gate_j4_l4, 29 epoch, 크기 5 Hz beta 0)은 고치기 전 변환표로 한 것입니다. 고친 뒤에는 다시 고르지 않고 같은 가중치를 다시 내보냈습니다.
REPORT.md 와 SERVER_NOTES.md 는 첫 2판(변환표 고치기 전·후)의 보고입니다. body / gru2(v3b 파일) / 추적기 상자 숫자는 tables_body.md (g2_body.py 출력).
