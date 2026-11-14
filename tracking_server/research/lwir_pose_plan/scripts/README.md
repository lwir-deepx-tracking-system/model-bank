# 열화상 자세 모델 학습 스크립트 (2026-11-15 판)

계획과 설정의 이유는 상위 폴더 `../README.md`. 장치에 있는 실험 E 를 만든 당시 판은 `models/detect/build_scripts/yolo26m_pose_otp2_bodybox_build/` 에 그대로 둔다(이 폴더는 그 뒤의 판).
자료(녹화 그림, 라벨, 가림 조각 묶음)는 넣지 않는다. 모두 Ultralytics 8.4.184, Python 3.11 기준.

## 학습

| 파일 | 하는 일 |
| --- | --- |
| `pose_partial.py` | Ultralytics 를 그 프로세스 안에서만 고친다: 화면 밖으로 나간 관절은 "안 보임(1)"으로 남김, 보임 출력은 "보임(2)"으로 학습, 머리~엉덩이 몸 상자(뼈대 틀 맞추기), 부분 잘라 확대(PartialCrop), 가림(Occlude v3/v4), 열화상 화질 흉내(ThermalLook), 무시 표시(3: 자료에 없는 칸은 위치·보임·몸 상자 계산에서 모두 뺌), 상자만 있는 사람 지우지 않기(keep_unlabelled) |
| `train_pose.py` | 학습 실행. `--office-bank --bank-share --heavy-ramp --keep-unlabelled` 는 가림 v4. 실행 폴더에 RUN_MANIFEST.json(인자, 판, sha256) |
| `skeleton_box.py`, `template_points.npy` | 뼈대 틀과 몸 상자 계산 |
| `otp2_coco_mot17.yaml` | 실험 E·R0 자료 (OTP2 + 흑백 COCO + 흑백 MOT17) |
| `otp2_coco_mot17_llvippose_vast.yaml` | R1 자료 (+ LLVIP-Pose) |
| `llvip_pose_vast.yaml` | LLVIP-Pose 만 (무시 표시 시험용) |

## 자료 만들기

| 파일 | 하는 일 |
| --- | --- |
| `coco_prep.py` | COCO 사람 자세를 흑백으로 (관절 없는 사람은 상자만 두어 배경으로 배우지 않게) |
| `mot17_det_prep.py`, `mot17_kpt_label.py` | MOT17 흑백 + 큰 자세 모델로 가짜 관절, 가려진 관절은 뼈대 틀로 채움 |
| `llvip_pose_prep.py` | LLVIP-Pose(PoseTrack 순서) → COCO 17점, 눈·귀는 무시 표시(3), 열화상 640x512 흑백 |
| `build_occluder_bank.py` | 사람 없는 사무실 16비트 녹화에서 책상·모니터·의자 같은 조각을 잘라 가림 v4 용 묶음(.npz) |

## 시험·채점

| 파일 | 하는 일 |
| --- | --- |
| `test_ignore_flag.py` | 무시 표시가 모든 보강을 지나도 그대로인지, 손실이 정상인지 |
| `test_occlude_bank.py` | 가림 v4: 덮인 보임 관절은 모두 안 보임으로, 무시 표시·상자는 안 바뀜, 가장자리 부드러움, v4 를 끄면 v3 와 같음 |
| `eval_mot17_check.py` | MOT17-09/11 흑백(학습 안 함): 보이는 정도별 찾은 비율, 몸 상자 기준 검증 |
| `indoor_score_v2.py`, `indoor_score_v2_check.py` | 우리 실내 녹화 채점: 상자 하나 = 사람 하나, (a) 보이는 상자 중심이 예측 안 + 보이는 넓이 50% 이상, 또는 (b) 예측 상자 70% 이상이 보이는 상자 안 + 보이는 넓이 25% 이상 덮음. 크기·보이는 정도·화면 가장자리별, 녹화 단위 부트스트랩. check 는 독립 재채점 |
| `llvip_transfer_check.py` | 컬러에서 찾은 관절을 열화상에 옮긴 것 vs 열화상에 바로 돌린 것 (LLVIP-Pose 시험용) |

## Vast 에서 쓴 실행 순서 (`vast_chains/`)
- `chain_R0R1.sh` → (조기 멈춤 문제로) `chain_R1v2.sh`: R0(대조), R1(+LLVIP-Pose), 둘 다 실험 E 에서, 조기 멈춤 끔.
- `chain_eval_R0R1.sh`: E(장치판·마지막판), R0, R1 의 MOT17 가림 비교 (모두 last.pt).
- `chain_ftt_thermal_v2.sh`, `after_fix_run.sh`: FastTrackTr 열화상 학습과 그 뒤 번호 확인·장치 내보내기 (FastTrackTr 스크립트는 `models/e2e/fasttracktr/build_scripts/`).
