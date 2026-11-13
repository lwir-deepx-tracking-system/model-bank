"""외형(재식별) 모델 목록과, 실행 중에 그 모델을 바꾸는 절차."""
from pathlib import Path


# ---- v41 (user 11-11): the appearance (re-ID) model is picked on the page and changed while running (KeepID and BoT-SORT 외형 비교 share ONE model:
# both already ran on the same engine on NPU core 2; two different models would need two more engines on that core and two feature spaces) ----
REID_JUDGE_FIT = 'RVGGA0_REID2C256'   # the 번호 잇기 판정기 (relink_judge_frozen.json) was fitted on this model's scores; with any other model it is not validated
REID_INFO = {'RVGGA0_REID2C256': ('RepVGG_distillation_(TMOT,CAMEL)_20261103', 'RepVGG', 'teacher SOLIDER · 공개 열화상 데이터로 학습'),
             'REPVGG_DISTILL_INSIDE_20261110': ('RepVGG_distillation_(실내)_20261110', 'RepVGG', 'teacher SOLIDER · 우리 카메라 실내로 학습 · 장치 정확도 미측정'),
             'REPVGG_FT_INSIDE_20261110': ('RepVGG_fine-tuning_(실내)_20261110', 'RepVGG', '우리 카메라 실내로 학습 · 장치 정확도 미측정'),
             'VIT_LITE_DISTILL_INSIDE_20261111': ('ViT-lite_distillation_(실내)_20261111', 'ViT-lite', 'teacher SOLIDER · 우리 카메라 실내로 학습 · 장치 정확도 미측정'),
             'VIT_LITE_FT_INSIDE_20261111': ('ViT-lite_fine-tuning_(실내)_20261111', 'ViT-lite', '우리 카메라 실내로 학습 · 장치 정확도 미측정'),
             'REPVGG_BODYBOX_OTP2_20261113': ('RepVGG_근거리 몸 네모_(OTP2 신원)_20261113', 'RepVGG', '근거리 몸 네모 검출 모델과 짝 · 잘림·가림 조각으로 학습 · 장치 정확도 미측정')}
def reid_catalog(models_dir, man, start_key=''):
    """-> the appearance models the page may show: the known names above (+ the model this run was started with), each with available / why.
    A file counts as installed when models/<key>.dxnn exists AND MANIFEST.json has its sha256 (the sha itself is checked at the change, not here).
    MANIFEST.json holds no tensor shapes, so the input size and the output are read from the engine itself when it is loaded."""
    out = []
    for k in list(REID_INFO) + ([start_key] if start_key and start_key not in REID_INFO else []):
        name, folder, note = REID_INFO.get(k, (k, '', ''))
        try: has = (Path(models_dir) / (k + '.dxnn')).exists()
        except OSError: has = False
        why = '' if (has and k in man) else ('장치에 파일이 없습니다' if not has else 'MANIFEST.json 에 이 파일의 확인값이 없습니다')
        out.append(dict(key=k, name=name, folder=folder, note=note, available=not why, why=why, judge_fit=k == REID_JUDGE_FIT, start=k == start_key))
    return out
def reid_swap(key, old, make, drain):
    """Change of the appearance model between two frames. old = the objects in use now (returned untouched on any failure).
    1) make(key) builds and test-runs the NEW engines while the old ones are still loaded and in use; any error -> old kept.
    2) drain() collects the old engine's crops still in flight and drops every stored appearance feature (features of two models must never be compared).
    -> (ok, objects to use from now on, text for the page)"""
    try: new = make(key)
    except Exception as e: return False, old, '새 모델을 올리지 못해 이전 모델 그대로: ' + repr(e)[:160]
    try: drain()
    except Exception as e:
        del new; return False, old, '이전 특징을 비우지 못해 이전 모델 그대로: ' + repr(e)[:160]
    return True, new, '바뀜'
