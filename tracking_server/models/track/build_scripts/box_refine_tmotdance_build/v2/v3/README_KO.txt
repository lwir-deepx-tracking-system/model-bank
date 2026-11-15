상자 다듬기 v3 (2026-11-15 밤) - 채점 스크립트 사본. 원본과 결과는 LIG-Data pos_refine_20261115/v2/v3/.
- PREREG_V3.json: 숫자를 보기 전에 적은 규칙 ('크기만'의 조준점 기준점, val-A 고르기, val-B 규칙 P1-P7, 이 계열에서 val-B 네 번째).
- v3_lib.py: v2_lib 를 그대로 쓰고, 채점 조준점(V3_AIM, 기본 0.37)과 서버 Refiner 의 aim_y, 섞기 시간을 넘김.
- v3_tune.py: val-A 에서 조준점 기준점의 크기 필터 고르기 (python v3_tune.py 3) -> results/aim_valA.json.
- v3_valB.py: val-B (python v3_valB.py valB 0.37 3, 보고만: valB 0.08 3) -> results/aim_valB_aim<조준점>.json. 위·아래 가장자리 움직임과
  바뀌는 프레임 + 다음 2 프레임(P7b, 보고만)도 냄.
결과: val-A 선택 = 조준점, 추적기 크기, 5 Hz, beta 3 (v2 와 같은 필터). val-B: TMOT 의 P7 하나만 못 맞춤, 조준점 값은 추적기와 같음.
