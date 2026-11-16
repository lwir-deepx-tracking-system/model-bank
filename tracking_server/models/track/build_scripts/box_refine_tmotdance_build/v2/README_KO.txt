[11-16 기록용] 서버 box_refine 는 11-16 에 off / size / body 만 남았고 Refiner 의 행은 (id, 검출, 추적기 행) 세 값입니다(전에는 (id, 검출, 관절, 추적기 행)). 이 폴더의 스크립트로 지금 서버를 다시 채점하려면 행에서 관절을 빼야 합니다. euro / gru 숫자는 기록입니다.

상자 다듬기 v2 (2026-11-15): '크기만 부드럽게'(box_refine=size) 값 고르기와 채점 스크립트 사본.

작업 폴더(공개 저장소 밖)의 배치 그대로 돌립니다:
  <작업 폴더>/scripts/        이 폴더 위의 첫 판 스크립트(prcommon, ref_lib, ref_eval, det_run, gmc_run, track_run ...) + paths.json
  <작업 폴더>/verify/combo/   combo_lib.py  (여기서는 combo_lib_verify.py 라는 이름으로 넣음, 독립 검증 단계의 지표 코드)
  <작업 폴더>/v2/scripts/     이 폴더의 *.py
  <작업 폴더>/v2/PREREG_V2.json  미리 적은 규칙 (val-B 숫자보다 먼저 씀)

순서: v2_tune.py (val-A 로 값 고르기) -> 서버 box_refine.SIZE 에 적기 -> v2_valB.py valB (한 번) ->
      dev_det.py + v2_device.py (장치 녹화, 정답 없음) -> v2_cpu.py (PC 에서 apply() 시간).
채점은 서버 모듈 src/track/box_refine.py 의 Refiner 를 프레임마다 그대로 돌리므로 서버와 같은 초기화 규칙을 씁니다.
