"""Training-only variation of local person/background contrast in rendered LWIR."""

from copy import deepcopy
import random

import numpy as np


# Sample a band with equal probability for each person, then sample within it.
GAIN_BANDS = ((0.35, 0.60), (0.80, 1.10), (1.25, 1.60))


class PersonThermalContrast:
    def __init__(self, probability, rng=None):
        if not 0 <= probability <= 1:
            raise ValueError('thermal contrast probability must be between 0 and 1')
        self.probability = probability
        self.rng = rng

    def __call__(self, labels):
        instances = labels.get('instances')
        image = labels['img']
        if not self.probability or instances is None or not len(instances):
            return labels
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            return labels
        gray = image if image.ndim == 2 else image[..., 0]
        if image.ndim == 3 and np.any(image != gray[..., None]):
            return labels
        rng = random if self.rng is None else self.rng
        if rng.random() >= self.probability:
            return labels

        height, width = gray.shape
        pixel_instances = deepcopy(instances)
        pixel_instances.convert_bbox(format='xyxy')
        pixel_instances.denormalize(width, height)
        boxes = np.asarray(pixel_instances.bboxes)
        boxes = boxes[np.isfinite(boxes).all(axis=1)]
        boxes = np.rint(np.clip(boxes, 0, [width, height, width, height])).astype(int)
        boxes = boxes[((boxes[:, 2:] - boxes[:, :2]) >= 6).all(axis=1)]
        if not len(boxes):
            return labels

        cover = np.zeros(gray.shape, np.uint16)
        for left, top, right, bottom in boxes:
            cover[top:bottom, left:right] += 1
        valid = gray != 114  # The neutral letterbox and crop fill stays intact.
        semantic = labels.get('semantic_mask')
        if semantic is not None:
            semantic = np.asarray(semantic)
            if semantic.shape != gray.shape:
                raise ValueError('thermal contrast ignore mask must match the current image')
            valid &= semantic == 0  # 1 is ignored, 255 is padding.
        output = image.copy()
        for left, top, right, bottom in boxes:
            w, h = right - left, bottom - top
            margin = max(4, round(0.2 * min(w, h)))
            ring = np.s_[max(0, top - margin):min(height, bottom + margin),
                         max(0, left - margin):min(width, right + margin)]
            background = gray[ring][(cover[ring] == 0) & valid[ring]]
            if background.size < 16:
                continue
            region = np.s_[top:bottom, left:right]
            pixels = gray[region].astype(np.float32)
            exclusive = (cover[region] == 1) & valid[region]
            x = (np.arange(w, dtype=np.float32) + 0.5) * (2 / w) - 1
            y = (np.arange(h, dtype=np.float32) + 0.5) * (2 / h) - 1
            radius = x[None, :] ** 2 + y[:, None] ** 2
            residual = pixels - np.median(background)
            center = residual[(radius < 0.34) & exclusive]
            if center.size < 9:
                continue
            contrast = float(np.median(center))
            strength = abs(contrast)
            if strength < 6:
                continue
            # Use only pixels consistent with the sign of the central thermal contrast.
            # Fade out towards the box border and leave overlaps between people untouched.
            foreground = np.clip((np.sign(contrast) * residual - 3) / max(4, strength * 0.55), 0, 1)
            feather = np.clip((1 - radius) / 0.4, 0, 1)
            gain = rng.uniform(*rng.choice(GAIN_BANDS))
            change = np.clip((gain - 1) * residual * foreground * feather, -35, 35)
            adjusted = np.clip(np.rint(pixels + change * exclusive), 0, 255).astype(np.uint8)
            output[region] = adjusted if image.ndim == 2 else adjusted[..., None]
        labels['img'] = output
        return labels


def add_person_thermal_contrast(dataset, probability):
    """Insert after geometry and before synthetic occluders or camera look."""
    if not probability:
        return dataset
    transforms = dataset.transforms
    for index, transform in enumerate(transforms.transforms):
        if transform.__class__.__name__ == 'PartialCrop':
            transforms.insert(index + 1, PersonThermalContrast(probability))
            return dataset
    raise ValueError('thermal contrast requires the pose PartialCrop training pipeline')


def with_person_thermal_contrast(trainer, probability):
    if not 0 <= probability <= 1:
        raise ValueError('thermal contrast probability must be between 0 and 1')
    if not probability:
        return trainer

    class ThermalContrastTrainer(trainer):
        def build_dataset(self, img_path, mode='train', batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == 'train':
                add_person_thermal_contrast(dataset, probability)
            return dataset

    return ThermalContrastTrainer
