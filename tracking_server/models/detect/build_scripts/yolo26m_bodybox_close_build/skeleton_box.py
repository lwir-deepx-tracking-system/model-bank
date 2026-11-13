"""A stable person box from the RELATIONS between keypoints (user requirement, 2026-11-13), not from the visible pixels.

Idea: every person is a scaled, rotated, shifted copy of one template skeleton (17 COCO keypoints in a body frame: origin = middle of the torso,
+y = from hips to shoulders, unit = torso length). Fit that template to whatever keypoints are available (weighted least-squares similarity
transform). The box is a fixed rectangle of the template (head top to hips, shoulders plus a margin) carried along by the fit, so it does not
shrink when part of the body leaves the picture or is hidden - only the fit gets less certain.

- 2 or more usable points: scale, rotation and position come from the points.
- exactly 1 point: only the position can be fitted; scale and rotation must come from memory (earlier frames of the same person).
- Limbs move relative to the torso, so they get small weights; head and torso points get weight 1.

build_template(): the template is the per-keypoint median over people with all 17 points labelled, each brought into the body frame.
"""
import numpy as np

KP = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_sh', 'r_sh', 'l_elb', 'r_elb', 'l_wri', 'r_wri', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ank', 'r_ank']
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
WEIGHT = np.array([1, 1, 1, 1, 1, 1, 1, .3, .3, .15, .15, 1, 1, .3, .3, .15, .15], float)
HEAD, SH, HIP = [0, 1, 2, 3, 4], [5, 6], [11, 12]


def body_frame(k):
    """(17, 2) pixel keypoints of a fully labelled person -> the same points in the body frame (image y points down; frame y points up the torso)."""
    s, h = k[SH].mean(0), k[HIP].mean(0)
    up = s - h; L = float(np.hypot(*up))
    if L < 1e-6: return None
    uy = up / L; ux = np.array([-uy[1], uy[0]])          # ux: to the picture-right when the person is upright (uy = (0, -1) -> ux = (1, 0))
    d = k - (s + h) / 2
    return np.stack([d @ ux, d @ uy], 1) / L


def template_from_points(t, n=0):
    """Template dict from the (17, 2) body-frame points (e.g. loaded from template_points.npy)."""
    t = np.asarray(t, float)
    half_w = float(np.abs(t[SH, 0]).max()) + 0.25        # shoulders plus a margin for the upper arms
    top = float(t[HEAD, 1].max()) + 0.35                 # above eyes/ears: top of the head
    bottom = float(t[HIP, 1].min()) - 0.10
    return dict(points=t, box=(-half_w, bottom, half_w, top), n=n)


def build_template(people):
    """people: iterable of (17, 2) arrays (all points labelled). Returns dict(points=(17, 2), box=(x0, y0, x1, y1) in the body frame, n=...)."""
    fr = [f for f in (body_frame(np.asarray(p, float)) for p in people) if f is not None]
    t = np.median(np.stack(fr), 0)
    m = t[FLIP].copy(); m[:, 0] *= -1                    # left/right symmetric
    t = (t + m) / 2
    return template_from_points(t, len(fr))


def fit(template, k, use, memory=None):
    """Fit the template to keypoints k (17, 2) using the boolean mask `use`. Returns dict(scale, R (2x2), t (2,), n, rms) or None.
    memory = dict(scale, R) from earlier frames, needed when only one point is usable (and used as is in that case)."""
    idx = np.flatnonzero(use); n = len(idx)
    if n == 0: return None
    P = template['points'][idx].copy(); P[:, 1] *= -1     # template to image orientation (y down)
    Q = np.asarray(k, float)[idx]; w = WEIGHT[idx] / WEIGHT[idx].sum()
    if n == 1:
        if memory is None: return None
        s, R = memory['scale'], memory['R']
        return dict(scale=s, R=R, t=Q[0] - s * R @ P[0], n=1, rms=0.0)
    mp, mq = (w[:, None] * P).sum(0), (w[:, None] * Q).sum(0)
    Pc, Qc = P - mp, Q - mq
    H = (w[:, None] * Pc).T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, d]) @ U.T                    # rotation only, no mirror
    var = float((w * (Pc ** 2).sum(1)).sum())
    if var < 1e-9: return None
    s = float((S * np.array([1.0, d])).sum() / var)
    if s <= 0: return None
    t = mq - s * R @ mp
    res = Q - (s * (R @ P.T).T + t)
    return dict(scale=s, R=R, t=t, n=n, rms=float(np.sqrt((w * (res ** 2).sum(1)).sum())))


def box_of(template, f):
    """Axis-aligned box (x1, y1, x2, y2) in pixels: the template rectangle's centre moved by the fit, its size scaled by the fit."""
    x0, y0, x1, y1 = template['box']
    c = np.array([(x0 + x1) / 2, -(y0 + y1) / 2])         # image orientation
    cx, cy = f['scale'] * f['R'] @ c + f['t']
    w, h = f['scale'] * (x1 - x0), f['scale'] * (y1 - y0)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def aim_point(template, f, chest_drop=0.25):
    """The point to keep in the middle of the picture (user, 2026-11-13: "chest and face in view is best"): halfway between the head point and
    the chest (chest = `chest_drop` torso lengths below the shoulder line), i.e. about the base of the neck. With this point centred the face is
    above and the chest below it, so both stay in the picture as long as the person is not closer than the picture can hold."""
    t = template['points']
    head = t[HEAD].mean(0); chest = t[SH].mean(0) - np.array([0.0, chest_drop])
    p = (head + chest) / 2 * np.array([1.0, -1.0])
    return f['scale'] * f['R'] @ p + f['t']


def head_point(template, f):
    """Where the fit puts the head (mean of the template's five face points) - usable even when no face point was seen."""
    p = template['points'][HEAD].mean(0) * np.array([1.0, -1.0])
    return f['scale'] * f['R'] @ p + f['t']
