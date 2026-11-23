"""Reviewed box definitions for replay metrics; stored JSON only, no geometry guesses."""

import hashlib
import json
import math
from pathlib import Path

from ..common.state import APP
from ..track import e2e_live

SIDECAR = "replay_box_contract.json"
SCHEMA = "reviewed-replay-boxes-v1"
FULL = "full_body"
UPPER = "head_to_hip"
APPROX = "approx_upper_body"
# pose_partial.BOX_ONLY. Set once before scoring and independent of predictions.
BOX_ONLY = {"top": -0.009, "height": 0.543, "width": 0.709, "cx": -0.006}
KNOWN_FULL_GT = frozenset({"outdoor_bandfill_20261111/seq_016"})
KNOWN_UPPER_MODELS = frozenset({
    "yolo26m_pose_ft16v8_aug_e10_bodybox_provisional_kpt_640x480",
    "yolo26m_pose_lwir1056_20261118_bodybox_kpt_640x480",
    "yolo26m_pose_lwir650_20261119_bodybox_kpt_640x480",
    "yolo26m_pose_lwir650_20261119_temporal_e13_bodybox_kpt_640x480",
    "yolo26m_pose_ft16v7_reviewed264_e24_provisional_kpt_640x480",
    "yolo26m_pose_otp2cocomot_20261114_bodybox_occl_kpt_640x480",
    "yolo26m_pose_otp2cocomotllvip_20261115_bodybox_farc2_kpt_640x480",
})


def live_model():
    app = vars(APP)
    keys = tuple(str(key) for key in (app.get("det_names") or ()))
    manifest = app.get("man") or {}
    return {
        "source": "live_e2e" if e2e_live.uses_own() else "server_detector",
        "mode": getattr(app.get("args"), "det", None),
        "model_keys": keys,
        "model_sha256": tuple(manifest.get(key) for key in keys),
    }


def stored_model(meta):
    # Older caches did not declare a detector contract. Never borrow the active model.
    source = meta.get("offline_run")
    key = source.get("model_key") if isinstance(source, dict) else None
    return {
        "source": "stored_tracks",
        "mode": meta.get("key"),
        "model_keys": (key,) if isinstance(key, str) and key else (),
        "model_sha256": (),
        "tracks_sha256": meta.get("mot_sha256"),
    }


def prediction_definition(model):
    keys = model["model_keys"]
    if (model["source"] in ("server_detector", "stored_tracks") and keys
            and all(key.removeprefix("bank:") in KNOWN_UPPER_MODELS for key in keys)):
        return UPPER
    return None


def project_upper(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) * 0.5 + BOX_ONLY["cx"] * w
    top = y1 + BOX_ONLY["top"] * h
    half = BOX_ONLY["width"] * w * 0.5
    clipped = max(0., cx - half), max(0., top), min(640., cx + half), min(480., top + BOX_ONLY["height"] * h)
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError("추정 상자를 만들 수 없는 원본 사람 박스가 있습니다.")
    return clipped


def _read_reviewed_boxes(sidecar, name, lab, prediction):
    raw = sidecar.read_bytes()
    doc = json.loads(raw)
    if (not isinstance(doc, dict) or doc.get("schema") != SCHEMA
            or doc.get("recording") != name or doc.get("human_reviewed") is not True):
        raise ValueError("검수 완료된 상자 계약 형식이 아닙니다.")
    if not lab.get("review_sha256") or (doc.get("source_mot_sha256"), doc.get("source_review_sha256")) != (
            lab["label_sha256"], lab["review_sha256"]):
        raise ValueError("정답 MOT 또는 프레임 검수 상태가 상자 계약과 일치하지 않습니다.")
    source = doc.get("source_box_definition")
    resolved = doc.get("box_definition")
    if source not in (FULL, UPPER) or resolved not in (FULL, UPPER):
        raise ValueError("정답 상자의 범위가 명시되지 않았습니다.")
    if name in KNOWN_FULL_GT and source != FULL:
        raise ValueError("이 녹화의 원본 MOT는 전신 정답입니다.")
    if resolved != prediction:
        raise ValueError("검출 모델과 정답의 상자 범위가 일치하지 않습니다.")
    if lab.get("bad_lines"):
        raise ValueError("정답에 유효하지 않은 행이 있습니다.")
    declared = doc.get("boxes", [])
    if not isinstance(declared, list):
        raise ValueError("검수 상자 목록 형식이 잘못되었습니다.")
    uses_parts = source != resolved
    if uses_parts and (source, resolved) != (FULL, UPPER):
        raise ValueError("전신 정답의 머리–골반 수동 검수만 지원합니다.")
    if not uses_parts and declared:
        raise ValueError("동일 범위 MOT 계약에는 별도 사람 상자를 사용할 수 없습니다.")

    original = {}
    for frame in lab["frames"]:
        for row in lab["gt"].get(frame, ()):
            if row[5] != 0:
                continue
            key = (frame, row[0])
            if key in original:
                raise ValueError("같은 검수 프레임의 사람 ID가 중복되었습니다.")
            original[key] = row[1:5]
    manual = {}
    for entry in declared:
        if not isinstance(entry, dict) or entry.get("reviewed") is not True:
            raise ValueError("명시적으로 검수된 사람 상자가 아닙니다.")
        frame, person = entry.get("frame"), entry.get("id")
        coords = entry.get("head_to_hip_xyxy")
        if type(frame) is not int or type(person) is not int or not isinstance(coords, list) or len(coords) != 4:
            raise ValueError("검수 사람 상자의 프레임, ID 또는 좌표 형식이 잘못되었습니다.")
        key = (frame, person)
        if key in manual or key not in original:
            raise ValueError("검수 사람 상자가 원본 MOT의 정확한 프레임과 ID에 대응하지 않습니다.")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in coords):
            raise ValueError("검수 사람 상자에 유효하지 않은 값이 있습니다.")
        l, t, r, b = coords
        old_l, old_t, old_r, old_b = original[key]
        if not (0 <= l < r <= 640 and 0 <= t < b <= 480
                and max(old_l, 0) - 0.02 <= l and r <= min(old_r, 640) + 0.02
                and max(old_t, 0) - 0.02 <= t and b <= min(old_b, 480) + 0.02):
            raise ValueError("검수 사람 상자가 화면 또는 원본 사람 범위를 벗어났습니다.")
        manual[key] = tuple(coords)
    if uses_parts and set(manual) != set(original):
        raise ValueError("채점 대상 전체 사람의 머리–골반 상자 검수가 필요합니다.")
    gt = lab["gt"]
    if uses_parts:
        gt = {f: [(row[0], *manual[(f, row[0])], row[5]) if row[5] == 0 else row
                  for row in gt.get(f, ())] for f in lab["frames"]}
    return source, resolved, hashlib.sha256(raw).hexdigest(), gt


def resolve(rec_dir, name, lab, predictor, *, strict, mode="verified"):
    prediction = prediction_definition(predictor)
    result = {
        "score_allowed": False,
        "evaluation_mode": mode,
        "approximate": False,
        "source_gt_box_definition": FULL if name in KNOWN_FULL_GT else None,
        "scored_gt_box_definition": None,
        "prediction_box_definition": prediction,
        "source_mot_sha256": lab.get("label_sha256"),
        "source_review_sha256": lab.get("review_sha256"),
        "reviewed_box_contract_sha256": None,
        "why": None,
    }
    sidecar = Path(rec_dir) / SIDECAR
    if not lab["ok"]:
        result["why"] = lab["why"]
    elif mode == APPROX:
        if name not in KNOWN_FULL_GT or prediction != UPPER or lab["bad_lines"]:
            result["why"] = "확인된 전신 MOT와 머리–골반 검출 모델에서만 추정 상체 GT를 사용할 수 있습니다."
        else:
            try:
                gt = {}
                for f in lab["frames"]:
                    rows = lab["gt"].get(f, ())
                    people = [row[0] for row in rows if row[5] == 0]
                    if len(people) != len(set(people)):
                        raise ValueError("동일 프레임의 원본 사람 ID가 중복되어 추정 평가를 할 수 없습니다.")
                    gt[f] = [(row[0], *project_upper(row[1:5]), row[5]) if row[5] == 0 else row for row in rows]
                result.update(score_allowed=True, approximate=True, scored_gt_box_definition="projected_head_to_hip",
                              projection=dict(BOX_ONLY), source_annotation_coverage=lab["review"])
                return result, gt
            except ValueError as exc:
                result["why"] = str(exc)
    elif sidecar.is_file():
        try:
            source, resolved, digest, gt = _read_reviewed_boxes(sidecar, name, lab, prediction)
            result.update(score_allowed=True, source_gt_box_definition=source,
                          scored_gt_box_definition=resolved, reviewed_box_contract_sha256=digest)
            return result, gt
        except (OSError, UnicodeError, ValueError, OverflowError) as exc:
            result["why"] = str(exc)
    elif not strict and name not in KNOWN_FULL_GT:
        # Direct synthetic evaluator use. The server's prepare() always enables strict mode.
        result["score_allowed"] = True
    elif name in KNOWN_FULL_GT and prediction == UPPER:
        result["why"] = "전신 정답과 모델의 머리–골반 상자 기준이 다릅니다. 동일 범위의 수동 검수 정답이 없어 채점하지 않습니다."
    else:
        result["why"] = "동일 범위의 정답과 검출 상자 및 프레임 검수를 확인하지 못해 채점하지 않습니다."
    return result, lab["gt"]
