"""Suppress a weak full-height duplicate of a well-observed upper-body box."""
import numpy as np

from ..common import frame_math

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
