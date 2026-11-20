"""Suppress a weak full-height duplicate of a well-observed upper-body box.

Which detectors the rule runs for is a switch (2026-11-20, ID-rules audit):
  named  (default) only the three detectors it was written for on 11-18 / 11-19 (NAMED_DETS) = the behaviour before the switch
  kpt    every detector whose file has keypoint heads (model_scan.KPT_DETS), whatever its name
  off    no detector
Start value: LWIR_NESTED_DUP=named|kpt|off. While running: POST /rules/set?nested_dup=named|kpt|off (not kept over a restart);
GET /rules shows nested_dup, nested_dup_active (does it run for the detector in use) and nested_dup_dropped (boxes removed).

What the rule removes: the LARGER box of a pair, never the small one. A large box is dropped only when all of this holds:
its own head and shoulders are not seen (at most one of the 7 upper keypoints >= .35); exactly one small box with a well seen
face (3 of 5 face points and a shoulder >= .50) lies in it (74 % covered, 48 px or taller), is 1.30 to 2.15 times shorter, has
its top within 15 % of its height of the large box's top, its centre within 16 % of its width, and a score at least as high;
and no second well seen face lies in the upper 45 % of the large box (that would be a person standing behind).
"""
import os

import numpy as np

from ..common import frame_math

NAMED_DETS = ('bank:yolo26m_pose_lwir1056_20261118_bodybox_kpt_640x480',
              'bank:yolo26m_pose_lwir650_20261119_bodybox_kpt_640x480',
              'bank:yolo26m_pose_lwir650_20261119_temporal_e13_bodybox_kpt_640x480')
MODES = ('named', 'kpt', 'off')


def _start_mode():
    mode = os.environ.get('LWIR_NESTED_DUP', '').strip().lower()
    return mode if mode in MODES else 'named'


RULE = {'mode': _start_mode(), 'dropped': 0}


def applies(det, kpt_dets):
    """Does the rule run for detector `det` (APP.args.det) under the switch? kpt_dets = model_scan.KPT_DETS."""
    mode = RULE['mode']
    if mode == 'off':
        return False
    if mode == 'kpt':
        return det in kpt_dets
    return det in NAMED_DETS


def set_mode(mode):
    """-> True when taken. The frame loop reads the mode once per frame, so the change lands between two frames."""
    if mode not in MODES:
        return False
    if mode != RULE['mode']:
        RULE['mode'] = mode; RULE['dropped'] = 0
    return True


def state(det, kpt_dets):
    return dict(nested_dup=RULE['mode'], nested_dup_active=applies(det, kpt_dets), nested_dup_dropped=int(RULE['dropped']))


def keep_person_poses(boxes, poses):
    """Return one mask for aligned person boxes (xyxy, score) and COCO poses."""
    keep = np.ones(len(boxes), dtype=bool)
    if len(boxes) < 2:
        return keep
    native = frame_math.nested_people(boxes, poses)
    if native is not None:
        return native

    xyxy = boxes[:, :4]
    size = xyxy[:, 2:] - xyxy[:, :2]
    good = (size > 0).all(axis=1) & np.isfinite(boxes).all(axis=1) & np.isfinite(poses[:, :7]).all(axis=(1, 2))
    upper = poses[:, :7]
    confident = upper[:, :, 2] >= .50
    trusted = good & (confident[:, :5].sum(axis=1) >= 3) & confident[:, 5:].any(axis=1)
    weak = good & ((upper[:, :, 2] >= .35).sum(axis=1) <= 1)
    strong_rows = np.flatnonzero(trusted)
    if not len(strong_rows) or not weak.any():
        return keep

    upper_extent = upper[strong_rows, :5, :2]
    face_visible = confident[strong_rows, :5]
    faces = (upper_extent * face_visible[:, :, None]).sum(axis=1) / face_visible.sum(axis=1)[:, None]
    centers_x = (xyxy[:, 0] + xyxy[:, 2]) * .5
    weak_rows = np.flatnonzero(weak)
    large = xyxy[weak_rows, None, :]
    small = xyxy[None, strong_rows, :]
    width, height = size[strong_rows].T
    large_height = size[weak_rows, 1, None]
    intersection = np.maximum(0, np.minimum(small[:, :, 2:], large[:, :, 2:]) -
                              np.maximum(small[:, :, :2], large[:, :, :2]))
    coverage = np.prod(intersection, axis=2) / (width * height)
    height_ratio = large_height / height
    x_distance = np.abs(centers_x[strong_rows] - centers_x[weak_rows, None])
    nested = (
        (height >= 48) & (height_ratio >= 1.30) & (height_ratio <= 2.15) &
        (coverage >= .74) & (x_distance <= .16 * width) &
        (np.abs(small[:, :, 1] - large[:, :, 1]) <= .15 * height) &
        (boxes[strong_rows, 4] >= boxes[weak_rows, 4, None])
    )
    # One coherent head is necessary. A second head in that upper region can
    # be an occluded neighbor, so leave the larger box available to association.
    upper_witness = ((faces[:, 0] >= large[:, :, 0]) & (faces[:, 0] <= large[:, :, 2]) &
                     (faces[:, 1] >= large[:, :, 1]) & (faces[:, 1] <= large[:, :, 1] + .45 * large_height))
    duplicate = ((nested & upper_witness).sum(axis=1) == 1) & (upper_witness.sum(axis=1) == 1)
    keep[weak_rows[duplicate]] = False
    return keep


def suppress_nested_people(detections, keypoints, person_cls):
    """Apply the same selector to current rows and the corresponding pose cache."""
    if keypoints is None or len(detections) < 2:
        return detections, keypoints
    person = detections[:, 5] == person_cls
    person_boxes = detections[person, :5]
    points = keypoints['kpts']
    count = len(person_boxes)
    if count < 2 or points.shape != (count, 17, 3) or not np.array_equal(person_boxes[:, :4], keypoints['boxes']):
        return detections, keypoints
    kept = keep_person_poses(person_boxes, points)
    if kept.all():
        return detections, keypoints
    result = np.ones(len(detections), dtype=bool)
    result[person] = kept
    return detections[result], {**keypoints, 'boxes': keypoints['boxes'][kept], 'kpts': points[kept]}
