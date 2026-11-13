"""Training changes for "estimate the body from whatever part is visible" (close_range_20261113/PLAN.md, v3).

Stock Ultralytics (read in 8.4.184) throws away every keypoint that an augmentation pushes outside the image: the visibility flag becomes 0
("not labelled") and the coordinate is clipped to the border, so a model never learns where the unseen part of a cut-off person is.
`install()` changes that behaviour for the current Python process only (nothing is written into site-packages):

1. Keypoints that leave the image keep their true coordinates and get flag 1 ("labelled, not visible") instead of 0.
   Patched: RandomPerspective.apply_keypoints, Instances.clip.
2. A person keeps its label when a smaller share of its box remains (stock 10 %, here AREA_THR) - needed for "only an arm is in view".
   Patched: RandomPerspective.box_candidates.
3. Loss (PoseLoss26.calculate_keypoints_loss):
   - the visibility output is trained against "flag == 2" (really visible), not "flag != 0", so it separates seen from estimated points;
   - location (OKS) and uncertainty (RLE sigma) losses still use every labelled point, visible or hidden (stock behaviour);
   - the OKS size term uses the larger of the visible box and the extent of all labelled keypoints. With the clipped box alone a far-away
     hidden point of a heavily cropped person saturates the OKS loss and gets no gradient.
4. PartialCrop: an extra augmentation that zooms into a random labelled body part, so much of the body falls outside the picture.

Flags: 0 = not labelled (ignored), 1 = labelled but not visible (hidden or outside the image), 2 = visible.
Mosaic must stay off (mosaic=0, close_mosaic=0): mosaic tiles hide parts of people without changing their flags, and closing mosaic rebuilds
the transform list, which would drop PartialCrop.
"""
import cv2
import numpy as np
import torch

HIDDEN = 1.0
AREA_THR = 0.03
_INSTALLED = [False]

# ---- body-box mode (user decision 2026-11-13: "let the model learn the stable box itself") ------------------------------------------------
# When BODY['template'] is set, the box label of every person is no longer the visible part but the BODY BOX: the fixed rectangle of the
# template skeleton (head top to hips, tools/skeleton_box.py) fitted to that person's labelled keypoints. It is NOT clipped to the picture, so a
# person who is cut off keeps the same box and the network has to regress it from whatever it sees. People with fewer than 2 labelled keypoints
# (or a degenerate fit) keep their original box. Validation uses the same body box clipped to the picture, because the stock validator clips
# predictions to the picture as well.
BODY = dict(template=None)


def _sb():
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for p in (here, here.parent / 'tools'):
        if str(p) not in sys.path: sys.path.insert(0, str(p))
    import skeleton_box
    return skeleton_box


def set_body_box(template_points_npy):
    """Switch body-box mode on with the template stored by tools/skeleton_box_eval.py (template_points.npy)."""
    SB = _sb()
    BODY['template'] = SB.template_from_points(np.load(str(template_points_npy)))


def to_body_boxes(bboxes, keypoints, size=None):
    """bboxes (N, 4) xyxy pixels, keypoints (N, 17, 3) pixels -> body boxes (N, 4). size=(w, h) only limits absurd fits."""
    T = BODY['template']
    if T is None or keypoints is None or len(bboxes) == 0: return bboxes
    SB = _sb(); out = np.asarray(bboxes, np.float32).copy()
    lim = 6.0 * max(size) if size else 1e9
    for i, k in enumerate(np.asarray(keypoints, float)):
        use = k[:, 2] > 0
        if use.sum() < 2: continue
        f = SB.fit(T, k[:, :2], use)
        if f is None: continue
        b = SB.box_of(T, f)
        if not np.isfinite(b).all() or (b[3] - b[1]) < 4 or (b[2] - b[0]) < 4 or (b[3] - b[1]) > lim: continue
        out[i] = b
    return out


class BodyBoxVal:
    """Validation labels: body box clipped to the picture (placed right after LetterBox in the validation transforms)."""

    def __call__(self, labels):
        inst = labels.get('instances')
        if inst is None or len(inst) == 0 or inst.keypoints is None or BODY['template'] is None: return labels
        inst.convert_bbox(format='xyxy')
        h, w = labels['img'].shape[:2]
        if inst.normalized: inst.denormalize(w, h)
        b = to_body_boxes(inst.bboxes, inst.keypoints, (w, h))
        c = b.copy(); c[:, [0, 2]] = c[:, [0, 2]].clip(0, w); c[:, [1, 3]] = c[:, [1, 3]].clip(0, h)
        ok = ((c[:, 2] - c[:, 0]) > 2) & ((c[:, 3] - c[:, 1]) > 2)
        inst.bboxes[ok] = c[ok]
        return labels


def _outside(xy_x, xy_y, w, h):
    return (xy_x < 0) | (xy_y < 0) | (xy_x > w) | (xy_y > h)


class PartialCrop:
    """With probability p: zoom by a factor in `zoom` into one labelled keypoint (or box centre) of a random person and put that point at a random
    place in the middle half of the picture. Boxes are clipped to the picture (visible part); keypoints keep true coordinates, flag 1 if outside.
    A person stays labelled if at least `min_share` of its box area is still inside, or a visible keypoint is still inside and the box is > 8 px."""

    def __init__(self, p=0.5, zoom=(1.4, 4.0), min_share=0.15, seed=None):
        # Randomness comes from Python's `random` module, like the stock Ultralytics transforms: the data loader seeds it differently in every
        # worker process. (A private generator stored here would be copied with the same state into every worker.) `seed` is kept for callers.
        self.p, self.zoom, self.min_share = p, zoom, min_share

    def __call__(self, labels):
        import random as _r
        inst = labels.get('instances')
        if inst is None or len(inst) == 0 or inst.keypoints is None or _r.random() >= self.p:
            return labels
        assert not inst.normalized and inst.keypoints.shape[-1] == 3, 'PartialCrop must come right after RandomPerspective (absolute pixels, x y flag)'
        inst.convert_bbox(format='xyxy')
        img = labels['img']; h, w = img.shape[:2]
        box = inst.bboxes.astype(np.float32).copy(); kp = inst.keypoints.astype(np.float32).copy()
        i = _r.randrange(len(box))
        cand = kp[i][(kp[i, :, 2] == 2) & ~_outside(kp[i, :, 0], kp[i, :, 1], w, h)]
        fx, fy = (cand[_r.randrange(len(cand)), :2] if len(cand) else ((box[i, 0] + box[i, 2]) / 2, (box[i, 1] + box[i, 3]) / 2))
        z = _r.uniform(*self.zoom)
        tx, ty = _r.uniform(0.25, 0.75) * w, _r.uniform(0.25, 0.75) * h
        M = np.array([[z, 0, tx - z * fx], [0, z, ty - z * fy]], np.float32)
        out = cv2.warpAffine(img, M, dsize=(w, h), borderValue=(114, 114, 114, 114))
        labels['img'] = out if out.ndim == 3 else out[..., None]
        full = box * z + np.array([M[0, 2], M[1, 2], M[0, 2], M[1, 2]], np.float32)
        clip = full.copy(); clip[:, [0, 2]] = clip[:, [0, 2]].clip(0, w); clip[:, [1, 3]] = clip[:, [1, 3]].clip(0, h)
        kp[..., 0] = kp[..., 0] * z + M[0, 2]; kp[..., 1] = kp[..., 1] * z + M[1, 2]
        out_k = _outside(kp[..., 0], kp[..., 1], w, h)
        kp[..., 2][out_k & (kp[..., 2] > 0)] = HIDDEN
        cw, ch = clip[:, 2] - clip[:, 0], clip[:, 3] - clip[:, 1]
        share = cw * ch / np.maximum((full[:, 2] - full[:, 0]) * (full[:, 3] - full[:, 1]), 1e-6)
        seen = ((kp[..., 2] == 2) & ~out_k).any(1)
        keep = (cw > 8) & (ch > 8) & ((share >= self.min_share) | seen)
        from ultralytics.utils.instance import Instances
        boxes_out = full if BODY['template'] is not None else clip        # body-box mode: the box is not cut at the picture edge
        new = Instances(boxes_out[keep], inst.segments[keep] if len(inst.segments) == len(keep) else inst.segments[:0], kp[keep], bbox_format='xyxy', normalized=False)
        labels['instances'] = new; labels['cls'] = labels['cls'][keep]
        return labels


def add_partial_crop(dataset, p=0.5, zoom=(1.4, 4.0), seed=None):
    """Insert PartialCrop right after the first block of the training transforms (Mosaic / CopyPaste / RandomPerspective)."""
    from ultralytics.data.augment import Compose, RandomPerspective
    t = dataset.transforms
    assert isinstance(t, Compose) and isinstance(t.transforms[0], Compose) and any(isinstance(x, RandomPerspective) for x in t.transforms[0].transforms), \
        'unexpected transform layout: %r' % (t,)
    assert not any(isinstance(x, PartialCrop) for x in t.transforms)
    t.insert(1, PartialCrop(p=p, zoom=zoom, seed=seed))
    return dataset


def install():
    """Patch Ultralytics in this process (idempotent)."""
    if _INSTALLED[0]: return
    from ultralytics.data.augment import RandomPerspective
    from ultralytics.utils.instance import Instances
    from ultralytics.utils import loss as L
    from ultralytics.utils.ops import xyxy2xywh

    def apply_keypoints(self, keypoints, M, size):
        n, nkpt = keypoints.shape[:2]
        if n == 0: return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1).copy()
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T
        xy = xy[:, :2] / xy[:, 2:3]
        out_mask = _outside(xy[:, 0], xy[:, 1], size[0], size[1])
        visible[out_mask & (visible[:, 0] > 0)] = HIDDEN
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    stock_clip = Instances.clip

    def clip(self, w, h, preserve_obb=False):
        k = None if self.keypoints is None or self.keypoints.shape[-1] != 3 else self.keypoints.copy()
        stock_clip(self, w, h, preserve_obb)
        if k is not None:
            flag = k[..., 2].copy()
            flag[_outside(k[..., 0], k[..., 1], w, h) & (flag > 0)] = HIDDEN
            self.keypoints[..., :2] = k[..., :2]; self.keypoints[..., 2] = flag

    stock_cand = RandomPerspective.box_candidates

    def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        return stock_cand(box1, box2, wh_thr=wh_thr, ar_thr=ar_thr, area_thr=min(area_thr, AREA_THR), eps=eps)

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, stride_tensor, target_bboxes, pred_kpts):
        kpts_loss = 0; kpts_obj_loss = 0; rle_loss = 0
        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = keypoints[masks.nonzero(as_tuple=True)[0], target_gt_idx[masks]]
            gt_kpt[..., :2] /= stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks][:, None, None]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            kpt_mask = gt_kpt[..., 2] != 0
            big = torch.finfo(gt_kpt.dtype).max
            x, y = gt_kpt[..., 0], gt_kpt[..., 1]
            ex = (x.masked_fill(~kpt_mask, -big).amax(1) - x.masked_fill(~kpt_mask, big).amin(1)).clamp(min=0)
            ey = (y.masked_fill(~kpt_mask, -big).amax(1) - y.masked_fill(~kpt_mask, big).amin(1)).clamp(min=0)
            extent = torch.where(kpt_mask.sum(1) >= 2, ex * ey, torch.zeros_like(ex)).unsqueeze(1)
            area = torch.maximum(area, extent)
            pred_kpt = pred_kpts[masks]
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask).clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], (gt_kpt[..., 2] == 2).float())
        return kpts_loss, kpts_obj_loss, rle_loss

    stock_apply_instances = RandomPerspective.apply_instances

    def apply_instances(self, labels, params=None):
        """Body-box mode only: replace the box by the body box before the affine move, never clip it; a person stays labelled while at least
        AREA_THR of the body box is inside the picture or a visible keypoint is inside. Otherwise the stock code runs."""
        if BODY['template'] is None or labels['instances'].keypoints is None:
            return stock_apply_instances(self, labels, params)
        cls = labels['cls']
        instances = labels.pop('instances')
        instances.convert_bbox(format='xyxy')
        instances.denormalize(*params['orig_shape'][::-1])
        M = params['M']; w, h = params['size']
        body = to_body_boxes(instances.bboxes, instances.keypoints, params['orig_shape'][::-1])
        bboxes = self.apply_bboxes(body, M)
        keypoints = self.apply_keypoints(instances.keypoints, M, params['size'])
        clipped = bboxes.copy(); clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, w); clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, h)
        cw, ch = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
        share = cw * ch / np.maximum((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]), 1e-6)
        seen = ((keypoints[..., 2] == 2) & ~_outside(keypoints[..., 0], keypoints[..., 1], w, h)).any(1)
        keep = ((cw > 2) & (ch > 2) & (share >= AREA_THR)) | seen
        labels['instances'] = Instances(bboxes[keep], instances.segments[keep] if len(instances.segments) == len(keep) else instances.segments[:0],
                                        keypoints[keep], bbox_format='xyxy', normalized=False)
        labels['cls'] = cls[keep]
        return labels

    RandomPerspective.apply_keypoints = apply_keypoints
    RandomPerspective.apply_instances = apply_instances
    Instances.clip = clip
    RandomPerspective.box_candidates = staticmethod(box_candidates)
    L.PoseLoss26.calculate_keypoints_loss = calculate_keypoints_loss
    _INSTALLED[0] = True


def add_body_box_val(dataset):
    """Validation set: insert BodyBoxVal right after LetterBox (body-box mode only)."""
    if BODY['template'] is None: return dataset
    from ultralytics.data.augment import Compose, LetterBox
    t = dataset.transforms
    assert isinstance(t, Compose) and isinstance(t.transforms[0], LetterBox), 'unexpected validation transform layout: %r' % (t,)
    if not any(isinstance(x, BodyBoxVal) for x in t.transforms): t.insert(1, BodyBoxVal())
    return dataset


def make_trainer(partial_p=0.5, zoom=(1.4, 4.0), body_box_template=None):
    """PoseTrainer whose training dataset has PartialCrop. The saved checkpoint is a stock PoseModel (no dependency on this file).
    body_box_template = path of template_points.npy switches body-box mode on (see BODY above)."""
    install()
    if body_box_template is not None: set_body_box(body_box_template)
    from ultralytics.models.yolo.pose import PoseTrainer

    class PartialPoseTrainer(PoseTrainer):
        def build_dataset(self, img_path, mode='train', batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == 'train':
                assert float(self.args.mosaic) == 0.0 and int(self.args.close_mosaic) == 0, 'run with mosaic=0 close_mosaic=0 (see module text)'
                add_partial_crop(ds, p=partial_p, zoom=zoom)
            else:
                add_body_box_val(ds)
            return ds

    return PartialPoseTrainer
