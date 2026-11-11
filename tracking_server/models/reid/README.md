# 외형 비교(re-ID) 모델 (`models/reid/`)

파일 목록·학습 자료·상태는 [../README.md](../README.md) 3절에 있습니다. 여기는 다섯 모델이 무엇을 받고 무엇을 내는지 적은 글입니다. 모양은 모델 파일 안의 값(2026-11-11 에 읽음)이고, 조각 만드는 법은 코드 `src/reid/reid_embed.py` 에서 옮겼습니다. 장치에서 다시 재지는 않았습니다.

## 입력과 출력 (다섯 모델 모두 같음)

| 항목 | 값 |
| --- | --- |
| 입력 이름 / 모양 / 형식 | `images` / `[1,128,64,3]` (높이 128, 너비 64, 채널 3) / uint8, NHWC |
| 입력 만드는 법 | 1) 8비트 흑백 프레임에서 사람 상자를 프레임 안으로 자름 2) 비율을 무시하고 64×128 로 늘림(선형 보간, 여백 채우기 없음) 3) 같은 값을 3채널에 복제 |
| 정규화 | 없음. 0~255 uint8 그대로 넣습니다 |
| 출력 이름 / 모양 / 형식 | `embedding` / `[1,128]` / float32 |
| 출력 뜻 | 사람 외형을 나타내는 숫자 128개. 모델 출력은 길이가 맞춰져 있지 않습니다 |
| 서버가 하는 뒤처리 | `e / max(‖e‖, 1e-12)` 로 길이를 1 로 맞춤 → 두 벡터의 내적이 코사인 유사도(−1~1, 클수록 같은 사람) |
| 한 번에 넣는 수 | 1 (조각 한 장씩) |
| 보정 | 256장 EMA |

모양과 이름이 같아서 서버를 켠 채로 화면에서 모델을 바꿀 수 있습니다. 바꿀 때 서버는 그동안 모은 외형 특징을 모두 버립니다(서로 다른 모델의 특징은 비교하지 않음).

## 구조별 차이

| 구조 | 파일 | 파일 크기 | 도는 곳 |
| --- | --- | --- | --- |
| RepVGG | `repvgg_tmot_camel_reid.dxnn`, `repvgg_distill_inside_20261110_reid.dxnn`, `repvgg_ft_inside_20261110_reid.dxnn` | 약 7.45 MB | 전부 NPU |
| ViT-lite | `vit_lite_distill_inside_20261111_reid.dxnn`, `vit_lite_ft_inside_20261111_reid.dxnn` | 약 1.46 MB | NPU + 마지막 단계만 파일 안의 CPU 부분. 받는 쪽은 `embedding` 만 받으면 됩니다 |

`repvgg_tmot_camel_reid_compile_metadata.json` 은 RepVGG_distillation_(TMOT,CAMEL)_20261103 파일 안의 컴파일 기록을 꺼낸 것입니다. 다른 네 파일의 것은 꺼내 두지 않았습니다.

번호 잇기 판정기(`../track/relink_judge_frozen.json`)는 RepVGG_distillation_(TMOT,CAMEL)_20261103 의 점수로 맞춘 것입니다. 다른 외형 모델과 함께 썼을 때의 결과는 확인되지 않았습니다.

확인 판정기(GRU)의 입력·출력은 [../track/README.md](../track/README.md)에 있습니다.
