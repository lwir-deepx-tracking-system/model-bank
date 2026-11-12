# 짐벌 시험 도구 (`hil_test/`)

**11-12 실제 저속 비교 완료:** 새 경로로 OFF/150ms,60ms/OFF,화면60ms/장면60ms 세 쌍을 실행했습니다. 중심 유지와 화면 이동량을 함께 개선한 운영값은 아직 채택하지 않았습니다. [수치·조건·소실 검사](PREDICTIVE_TUNING_20261112_KO.md)를 먼저 확인합니다. 운영 서버 복구와 제어 튜닝 완료는 다릅니다.

**11-12 예측 제어 변경 주의:** `lead_hil.py`는 옛 박스 이동 방식입니다. 새 `LEAD ... control_epoch` 경로의 성능을 이 도구로 검증했다고 주장하지 않습니다. 새 경로의 CPU·가상 포트 검증과 남은 실기 조건은 [V27_CHANGES.md](../V27_CHANGES.md)를 확인합니다.

새 제어의 첫 비교 조건과 필요한 기록은 [예측 끔/켬 비교 규약](PREDICTIVE_AB_CHECK_KO.md)을 따릅니다. `analyze_v27.py`는 좌표 없는 기록을 대상 놓침에 포함하고, 전체 기록 시간의 놓침 비율·최장 연속 놓침과 관측 구간의 양축 중심 오차를 구분합니다. `C_lost`는 기존 채점 구간의 표본 비율, `lost_time_ratio`는 워밍업을 포함한 전체 기록 시간의 비율입니다. 빈 좌표를 먼저 제거한 과거 수치와 동일한 집계가 아닙니다.

실제 짐벌에 가상의 사람을 따라가게 해서 짐벌의 특성, 제어값, v25 대 v27 을 재는 도구입니다. **먼저 [`PLAN_V27_HILS_20261111_KO.md`](PLAN_V27_HILS_20261111_KO.md) 를 읽으세요.** 멈춤 규칙, 시작 조건, 옆에 선 사람이 알 것(6절)이 모두 거기 있습니다.

- 사람이 짐벌 옆에 있을 때만 돌립니다.
- PC 가상 짐벌 검증 뒤 실제 1단계를 한 번 실행했지만 영상 변위 한계를 넘어 중단했습니다. 정지와 서버 복귀를 확인했으며 2·3단계는 미실행입니다. 자세한 실패 기록은 루트 `HANDOFF.md` 5절에 보존합니다. 이전의 실행 가능 검토를 현재 실기 완료나 재시도 승인으로 해석하지 않습니다.

## 파일

- `OBSERVATION_GAP_20261112_KO.md` — 한 프레임 관측 누락 보완의 배포·실기 결과와 실패 기록
- `observation_gap_hil.py` — 같은 PD/저속 조건의 관측 누락·복귀·반전 시험. 기본은 합성 카메라와 가상 직렬입니다. 실제 시험에는 `--execute --attended`와 현장 사람이 필요합니다. `--single-condition 100`은 단독 기능 진단이며 짝지은 성능 비교가 아닙니다. 출력은 저장소 밖의 새 비공개 폴더로 지정합니다.

새 관측 누락 도구를 모의 실행할 때의 예입니다. `<저장소>`는 `tracking_server`를 포함하는 최상위 폴더입니다.

```bash
python3 observation_gap_hil.py --repo <저장소> --bridge <새_v27_실행파일> --out <저장소_밖_새_결과폴더>
```

- `PLAN_V27_HILS_20261111_KO.md` — 계획, 규칙, 판정 기준 (실행 전에 적어 둔 것)
- `CHANGES_FROM_M1_TOOLS_KO.md` — 장치에 있던 옛 도구와 다른 점
- `run_v27_hil.sh` — 실행 스크립트 (단계, 시간 한계, 끝날 때 멈춤)
- `pid_hil.py`, `lead_hil.py` — 짐벌 프로그램을 띄우고 가상의 사람을 움직이며 재는 도구
- `make_plan_v27.py` — 앞 단계 결과로 다음 단계 계획 만들기
- `analyze_v27.py` — 결과 표와 판정
- `send_stop.py`, `check_stop.py` — 짐벌에 직접 멈춤 보내기, 마지막이 멈춤으로 끝났는지 확인
- `dry_bridge_standin.py` — PC 확인용 가짜 짐벌 프로그램 (장치에서는 쓰지 않음)
- `plan_stage1.json` — 1단계 계획

## 돌리는 순서 (장치에서)

이 폴더의 파일을 장치의 한 폴더에 두고 그 폴더에서 실행합니다. 서버 폴더는 기본 `~/workspace/lwir_model_bank/tracking_server` 이고 `HIL_ROOT` 로 바꿉니다.

```bash
bash run_v27_hil.sh check          # 보기만
bash run_v27_hil.sh dry1           # 짐벌 없이 실제 v25·v27 프로그램 확인 (dry2, dry3 도)
setsid nohup bash run_v27_hil.sh stage1 > /dev/null 2>&1 &
tail -f stage1_run.log
python3 analyze_v27.py stage1 stage1      # 표를 사람이 읽는다
python3 make_plan_v27.py stage2 stage1    # 다음 단계 계획
```

- 2·3단계도 같은 식입니다 (`stage2`, `stage3`). 단계마다 로그 끝에 `NOT WRITTEN`, `WARNING`, `END VIEW NOT CHECKED` 가 없는지 보고, 하나라도 있으면 다음 단계로 가지 않습니다.
- 손으로 끝내기: `kill -TERM "$(cat .run_v27_hil.pid)"`
- 스크립트를 `kill -9` 했거나 전원이 나갔으면: 시험이 안 도는 것을 확인하고 `python3 send_stop.py`, 그리고 `rm ~/lwir_keepalive.pause` (안 지우면 서버가 다시 켜지지 않습니다).
- 서버가 다시 켜지면 예측 제어는 꺼진 채로 켜집니다.
