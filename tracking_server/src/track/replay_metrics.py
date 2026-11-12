"""Sequence-level replay metrics for person detections and tracks.

Input rows are project-filtered XYXY boxes:
  gt/tracks: [id, x1, y1, x2, y2]
  detections: [score, x1, y1, x2, y2]

HOTA follows TrackEval's single-sequence HOTA implementation (alpha=.05:.05:.95).
AP is COCO-style 101-point interpolated AP over IoU=.50:.05:.95, for the
available post-NMS/project-filtered candidate detections. It is not an official
COCO benchmark result for this project. The caller should run this off the
playback path; this module does no I/O and has no playback/thread management.

Primary references:
  TrackEval: https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/hota.py
  COCO API:  https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


_HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)
_AP_IOUS = np.arange(0.50, 0.96, 0.05)
_RECALL_SAMPLES = np.linspace(0.0, 1.0, 101)
_MAX_DETS_PER_FRAME = 100  # COCO's default maxDets[2]


def _rows(rows: Iterable[Sequence[float]], width: int, name: str) -> np.ndarray:
    if rows is None:
        raise ValueError(f"{name} rows cannot be None")
    out = np.asarray(list(rows), dtype=np.float64)
    if out.size == 0:
        return np.empty((0, width), dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != width:
        raise ValueError(f"{name} rows must have shape (N, {width})")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} rows must contain only finite values")
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise XYXY IoU; degenerate boxes have zero area."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.maximum(0.0, br - tl)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def _hota(frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float | None:
    gt_ids = sorted({row[0] for gt, _, _ in frames for row in gt})
    tr_ids = sorted({row[0] for _, tr, _ in frames for row in tr})
    gt_map = {v: i for i, v in enumerate(gt_ids)}
    tr_map = {v: i for i, v in enumerate(tr_ids)}
    n_gt = sum(len(gt) for gt, _, _ in frames)
    if n_gt == 0:
        return None

    # TrackEval's global alignment term: per frame, normalize each IoU by the
    # sum of its row and column similarities minus itself, then accumulate.
    potential = np.zeros((len(gt_ids), len(tr_ids)), dtype=np.float64)
    gt_count = np.zeros(len(gt_ids), dtype=np.float64)
    tr_count = np.zeros(len(tr_ids), dtype=np.float64)
    frame_data = []
    for gt, tr, _ in frames:
        if len(set(gt[:, 0])) != len(gt) or len(set(tr[:, 0])) != len(tr):
            return None  # TrackEval requires unique IDs per frame.
        gids = np.asarray([gt_map[x] for x in gt[:, 0]], dtype=np.int64)
        tids = np.asarray([tr_map[x] for x in tr[:, 0]], dtype=np.int64)
        sim = _iou(gt[:, 1:5], tr[:, 1:5])
        frame_data.append((gids, tids, sim))
        if sim.size:
            denom = sim.sum(axis=0)[None, :] + sim.sum(axis=1)[:, None] - sim
            normalized = np.divide(sim, denom, out=np.zeros_like(sim), where=denom > np.finfo(float).eps)
            potential[np.ix_(gids, tids)] += normalized
        # TrackEval counts ID occurrences even on frames where the opposite
        # side is empty; those occurrences belong in the association union.
        np.add.at(gt_count, gids, 1.0)
        np.add.at(tr_count, tids, 1.0)

    alignment = np.divide(
        potential,
        gt_count[:, None] + tr_count[None, :] - potential,
        out=np.zeros_like(potential),
        where=(gt_count[:, None] + tr_count[None, :] - potential) > 0.0,
    )
    tp = np.zeros(len(_HOTA_ALPHAS), dtype=np.float64)
    fn = np.zeros_like(tp)
    fp = np.zeros_like(tp)
    pair_matches = [Counter() for _ in _HOTA_ALPHAS]

    for gids, tids, sim in frame_data:
        ng, nt = len(gids), len(tids)
        if ng == 0:
            fp += nt
            continue
        if nt == 0:
            fn += ng
            continue
        score = alignment[np.ix_(gids, tids)] * sim
        rr, cc = linear_sum_assignment(-score)
        assigned_iou = sim[rr, cc]
        for k, alpha in enumerate(_HOTA_ALPHAS):
            keep = assigned_iou >= alpha - np.finfo(float).eps
            mr, mc = rr[keep], cc[keep]
            nm = len(mr)
            tp[k] += nm
            fn[k] += ng - nm
            fp[k] += nt - nm
            if nm:
                pair_matches[k].update(zip(gids[mr], tids[mc]))

    assa = np.zeros_like(tp)
    for k, matches in enumerate(pair_matches):
        assa[k] = sum(n * n / max(1.0, gt_count[g] + tr_count[t] - n)
                      for (g, t), n in matches.items()) / max(1.0, tp[k])
    deta = tp / np.maximum(1.0, tp + fn + fp)
    return float(100.0 * np.sqrt(deta * assa).mean())


def _ap_for_iou(frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]], threshold: float) -> float | None:
    total_gt = sum(len(gt) for gt, _, _ in frames)
    if total_gt == 0:
        return None
    scored = []
    for frame_index, (gt, _, det) in enumerate(frames):
        if not len(det):
            continue
        # COCO evaluates no more than maxDets candidates per image/category.
        order = np.argsort(-det[:, 0], kind="stable")[:_MAX_DETS_PER_FRAME]
        for local_rank, di in enumerate(order):
            scored.append((float(det[di, 0]), frame_index, local_rank, det[di, 1:5]))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched = [np.zeros(len(gt), dtype=bool) for gt, _, _ in frames]
    tp = np.zeros(len(scored), dtype=np.float64)
    fp = np.zeros(len(scored), dtype=np.float64)
    for i, (_, fi, _, box) in enumerate(scored):
        gt = frames[fi][0]
        if len(gt):
            ious = _iou(box[None, :], gt[:, 1:5])[0]
            # COCO uses the highest-IoU eligible unmatched GT. Its scan
            # accepts an equal IoU later in the stable ground-truth order.
            eligible = np.flatnonzero((ious >= threshold - np.finfo(float).eps) & ~matched[fi])
            if len(eligible):
                best = eligible[len(eligible) - 1 - np.argmax(ious[eligible[::-1]])]
                matched[fi][best] = True
                tp[i] = 1.0
                continue
        fp[i] = 1.0
    if not len(scored):
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(float).eps)
    # COCO's monotonically non-increasing precision envelope and 101 recall
    # samples. np.searchsorted implements the recall-point lookup.
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, _RECALL_SAMPLES, side="left")
    sampled = np.zeros_like(_RECALL_SAMPLES)
    valid = indices < len(precision_envelope)
    sampled[valid] = precision_envelope[indices[valid]]
    return float(sampled.mean())


def _ap_metrics(frames: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]) -> tuple[float | None, float | None]:
    if any(det is None for _, _, det in frames):
        return None, None
    concrete = [(gt, tr, det) for gt, tr, det in frames]
    ap50 = _ap_for_iou(concrete, 0.50)
    aps = [_ap_for_iou(concrete, float(t)) for t in _AP_IOUS]
    if ap50 is None or any(v is None for v in aps):
        return None, None
    return 100.0 * ap50, 100.0 * float(np.mean(aps))


def compute(frames: Sequence[tuple[Iterable[Sequence[float]], Iterable[Sequence[float]], Iterable[Sequence[float]] | None]]) -> dict[str, float | None]:
    """Compute sequence HOTA, AP50, and COCO-style AP50:95 in percent.

    ``frames`` must contain one complete, ordered sequence. Track IDs are
    interpreted as sequence-global IDs. Empty detections (``[]``) are valid;
    unavailable detections (``None`` on any frame) make both AP values None.
    HOTA can still be computed when detection candidates are unavailable.
    Empty/ignored boxes should already have been filtered by the caller.
    """
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
    for index, item in enumerate(frames):
        if len(item) != 3:
            raise ValueError(f"frame {index} must be a (gt, tracks, detections) tuple")
        gt_raw, tr_raw, det_raw = item
        gt = _rows(gt_raw, 5, f"frame {index} gt")
        tr = _rows(tr_raw, 5, f"frame {index} tracks")
        det = None if det_raw is None else _rows(det_raw, 5, f"frame {index} detections")
        prepared.append((gt, tr, det))
    hota = _hota([(gt, tr, np.empty((0, 5)) if det is None else det) for gt, tr, det in prepared])
    ap50, mean_ap = _ap_metrics(prepared)
    return {
        "hota": hota,
        "ap50": None if ap50 is None else float(ap50),
        "map": None if mean_ap is None else float(mean_ap),
    }
