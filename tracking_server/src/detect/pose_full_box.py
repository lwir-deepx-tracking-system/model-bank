"""Project a head-to-hip E10 detection into the visible extent of a full-person box."""
from math import hypot, isfinite

DETECTOR = 'bank:yolo26m_pose_lwir650_20261119_bodybox_kpt_640x480'
TEMPORAL_DETECTOR = 'bank:yolo26m_pose_lwir650_20261119_temporal_e13_bodybox_kpt_640x480'
# 12 matched people in the local seq_002 frames 227, 228 and 311.
# This is a standing-person prior for occlusion, not a new trained pose head.
FULL_HEIGHT_PER_UPPER = 1.77
POINT_SCORE = 0.55


def available(app):
    return getattr(getattr(app, 'args', None), 'det', None) in (DETECTOR, TEMPORAL_DETECTOR)


def mode(app):
    return getattr(app, 'SEL', {}).get('pose_box', 'upper') if available(app) else None


def project_box(box, pose, width, height, y_offset=0):
    """Return (xyxy, evidence) in the same canvas as the input, or None."""
    x1, y1, x2, y2 = map(float, box)
    h, w = y2 - y1, x2 - x1
    if not all(isfinite(v) for v in (x1, y1, x2, y2)) or min(w, h) <= 0:
        return None

    def point(index):
        x, y, score = map(float, pose[index])
        if (score >= POINT_SCORE and isfinite(score)
                and x1 - w <= x <= x2 + w and y1 - .15 * h <= y <= y1 + 2.15 * h
                and 0 <= x <= width and y_offset <= y <= y_offset + height):
            return x, y
        return None

    hips, knees, ankles = [], [], []
    for hip_i, knee_i, ankle_i in ((11, 13, 15), (12, 14, 16)):
        hip, knee, ankle = point(hip_i), point(knee_i), point(ankle_i)
        if hip is None or not y1 + .38 * h <= hip[1] <= y2 + .24 * h:
            continue
        hips.append(hip)
        if (knee is None or knee[1] < hip[1] - .08 * h
                or not .13 * h <= hypot(knee[0] - hip[0], knee[1] - hip[1]) <= .74 * h):
            continue
        knees.append(knee)
        if (ankle is not None and ankle[1] >= knee[1] - .10 * h
                and .13 * h <= hypot(ankle[0] - knee[0], ankle[1] - knee[1]) <= .74 * h):
            ankles.append(ankle)

    pad = .05 * h
    if ankles:
        bottom = max(y2, max(y for _, y in knees + ankles) + pad)
        evidence = 'ankle'
    elif knees:
        bottom = max(y2, *(y + .43 * h for _, y in knees))
        evidence = 'knee'
    else:
        hip_y = sum(y for _, y in hips) / len(hips) if hips else y2
        hip_y = max(y2 - .12 * h, min(y2 + .12 * h, hip_y))
        bottom = max(y2, hip_y + (FULL_HEIGHT_PER_UPPER - 1) * h)
        evidence = 'standing_prior'
    xs = [x for x, _ in knees + ankles]
    result = (max(0., min([x1, *(x - pad for x in xs)])),
              max(float(y_offset), y1),
              min(float(width), max([x2, *(x + pad for x in xs)])),
              min(float(y_offset + height), bottom))
    return (result, evidence) if result[0] < result[2] and result[1] < result[3] else None


def project_tracks(rows, detections, keypoints, width, height, y_offset):
    """Use only the unique pose attached to each track's current detection index."""
    if not len(rows) or detections is None or not keypoints:
        return rows
    boxes, points = keypoints['boxes'], keypoints['kpts']
    if boxes.shape != (len(points), 4) or points.shape != (len(boxes), 17, 3):
        return rows
    pose_by_box = {}
    for box, pose in zip(boxes, points):
        key = tuple(box)
        pose_by_box[key] = None if key in pose_by_box else pose
    projected = []
    for row in rows:
        pose = None
        if len(row) >= 8 and int(row[6]) == 0:
            index = int(row[7])
            if 0 <= index < len(detections) and int(detections[index][5]) == 0:
                pose = pose_by_box.get(tuple(detections[index][:4]))
        if pose is None:
            projected.append(row)
            continue
        # The detection's pose is expressed in canvas coordinates. Keep its
        # full extent while including the tracker-smoothed upper-body edges.
        result = project_box(detections[index][:4], pose, width, height, y_offset)
        if result is None:
            projected.append(row)
            continue
        (x1, y1, x2, y2), _ = result
        box = (max(0., min(x1, float(row[0]))), max(float(y_offset), min(y1, float(row[1]))),
               min(float(width), max(x2, float(row[2]))), min(float(y_offset + height), max(y2, float(row[3]))))
        projected.append([*box, *row[4:]])
    return projected
