# 짐벌 시험 도구 (`hil_test/`)

**11-12 예측 제어 변경 주의:** `lead_hil.py`는 옛 박스 이동 방식입니다. 새 `LEAD ... control_epoch` 경로의 성능을 이 도구로 검증했다고 주장하지 않습니다. 새 경로의 CPU·가상 포트 검증과 남은 실기 조건은 [V27_CHANGES.md](../V27_CHANGES.md)를 확인합니다.

실제 짐벌에 가상의 사람을 따라가게 해서 짐벌의 특성, 제어값, v25 대 v27 을 재는 도구입니다. **먼저 [`PLAN_V27_HILS_20261111_KO.md`](PLAN_V27_HILS_20261111_KO.md) 를 읽으세요.** 멈춤 규칙, 시작 조건, 옆에 선 사람이 알 것(6절)이 모두 거기 있습니다.

- 사람이 짐벌 옆에 있을 때만 돌립니다.
- PC 에서 가짜 짐벌로 확인했고, 독립 안전 검토 3번을 거쳤습니다 (셋째 판정: 세 단계 모두 돌려도 됨). 실제 짐벌 숫자는 아직 없습니다.

## 파일

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
