상체 네모로 통일 2판 (2026-11-16, 서버 모드 box_refine=body): 표 고치기, 사람 전체 상자 고정 비율, 크기 필터 고르기, 채점, 눈으로 보기, 비용 스크립트 사본.
결과 파일: ../../box_body_table_20261116.npz (+ .json: 만든 규칙, 칸 수, sha256). 미리 적은 규칙: PREREG_BODY_V2.json (val-A / val-B / 장치 녹화 숫자보다 먼저 씀).

작업 폴더(공개 저장소 밖)의 배치 그대로 돌립니다:
  <작업 폴더>/scripts/                 첫 판 스크립트 (../box_refine_tmotdance_build/ 의 prcommon, pos_lib, ref_lib, ref_eval, track_run, det_run, gmc_run ...) + paths.json
  <작업 폴더>/verify/combo/combo_lib.py (../box_refine_tmotdance_build/v2/combo_lib_verify.py)
  <작업 폴더>/v2/scripts/dev_det.py     (../box_refine_tmotdance_build/v2/dev_det.py), v2/data/{det,gmc}/ 장치 녹화의 검출·카메라 움직임
  <작업 폴더>/gru_v2/v3b/scripts/       g2_lib.py, g2_device.py, g2_data.py (../box_refine_gru_v2_build/v3b/), v3b/data/ (lut_v3.pkl, rows_valA.pkl, rows_valB.pkl)
  <작업 폴더>/body_v2/scripts/          이 폴더의 b2_*.py,  body_v2/PREREG_BODY_V2.json

순서 (모두 PC CPU, 저장소의 tracking_server 를 paths.json 의 worktree 로 읽음):
  b2_train_look.py           (학습 분할만) 변환표 칸마다 쌍 수, 순서열 수, 흔들림, 검출기 자신의 골반 위치; 사람 전체 상자 -> 몸 네모 관계
  b2_table.py stats          (학습 분할만) N 의 근거: 쌍 하나의 퍼짐 0.078 log, 설계 효과 16
  b2_table.py fit            고친 표 + 사람 전체 상자 고정 비율 -> body_v2/data/table_v2.npz, results/table_v2.json
  b2_eval.py valA 6          크기 필터 격자 35점, 미리 적은 규칙으로 고름 -> results/valA_grid.json (1.5 Hz, beta 0.3)
  b2_export.py <models/track> 20261116   표 파일 + json
  b2_pushed_parity.py        채점의 'body_pushed' 흉내 = 11-16 에 올린 서버 코드 (val-B 6 흐름 32,391 줄, 2.3e-13 px)
  b2_eval.py valB 1.5 0.3    val-B 한 번 (규칙 B1-B7) -> results/valB.json, tables_valB.md
  b2_yolo.py imgs|rec        일반 검출기 YOLO26x FLIR 를 학습 분할 표본 474장과 장치 녹화 3개에 (fp32 CPU)
  b2_plain_check.py          (학습 분할) 고정 비율을 YOLO26x 자신의 상자에 대 봄 (1,890쌍: 키 0.99, 폭 0.99 배)
  b2_eval.py device 1.5 0.3 <녹화 3개>   장치 속도 녹화 한 번 (규칙 D1-D4, 일반 검출기 흐름은 보고만)
  b2_eye.py                  눈으로 보기 판 (가까이·중간: 장치 녹화, 공개 val-B, 우리 카메라 야외 먼 사람 낱장)
  b2_cost.py                 PC 비용 (사람 10 / 20 / 30명, 한 가닥)
  b2_pos_parity.py           위치 기억 numpy = torch (정리 뒤에도 같음)
