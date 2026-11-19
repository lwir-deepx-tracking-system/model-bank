from types import SimpleNamespace
import unittest

import numpy as np
from ultralytics.data.augment import Compose
from ultralytics.utils.instance import Instances

from thermal_contrast import PersonThermalContrast, with_person_thermal_contrast


class FixedGain:
    def __init__(self, gain):
        self.gain = gain

    def random(self):
        return 0.0

    def choice(self, bands):
        return next(band for band in bands if band[0] <= self.gain <= band[1])

    def uniform(self, low, high):
        return self.gain


class ThermalContrastTests(unittest.TestCase):
    def sample(self):
        gray = np.full((72, 104), 100, np.uint8)
        gray[18:54, 20:42] = 150
        gray[18:54, 64:86] = 65
        gray[:5] = 114
        image = np.repeat(gray[..., None], 3, axis=2)
        boxes = np.array([[20, 18, 42, 54], [64, 18, 86, 54]], np.float32)
        keypoints = np.zeros((2, 17, 3), np.float32)
        keypoints[0, 0] = (28, 24, 2)
        instances = Instances(boxes, np.empty((0, 0, 2), np.float32), keypoints,
                              bbox_format='xyxy', normalized=False)
        return dict(img=image, instances=instances, cls=np.zeros((2, 1), np.float32))

    def test_gain_changes_person_contrast_in_both_directions(self):
        for gain, warmer, cooler in ((0.40, 120, 86), (1.0, 150, 65), (1.50, 175, 48)):
            with self.subTest(gain=gain):
                labels = self.sample()
                image = labels['img']
                boxes, points = labels['instances'].bboxes.copy(), labels['instances'].keypoints.copy()
                result = PersonThermalContrast(1, FixedGain(gain))(labels)
                self.assertEqual((int(result['img'][35, 31, 0]), int(result['img'][35, 75, 0])), (warmer, cooler))
                np.testing.assert_array_equal(result['img'][:5], image[:5])
                np.testing.assert_array_equal(result['img'][:, 47:59], image[:, 47:59])
                np.testing.assert_array_equal(result['instances'].bboxes, boxes)
                np.testing.assert_array_equal(result['instances'].keypoints, points)
                self.assertEqual(image[35, 31, 0], 150)

    def test_box_overlap_stays_unchanged(self):
        labels = self.sample()
        labels['img'][18:54, 42:64] = 130
        labels['instances'].bboxes[0, 2] = 57
        labels['instances'].bboxes[1, 0] = 46
        overlap = labels['img'][18:54, 46:57].copy()
        PersonThermalContrast(1, FixedGain(0.4))(labels)
        np.testing.assert_array_equal(labels['img'][18:54, 46:57], overlap)

    def test_disabled_and_nonthermal_inputs_keep_original_array(self):
        labels = self.sample()
        original = labels['img']
        self.assertIs(PersonThermalContrast(0)(labels)['img'], original)
        original[12, 12, 1] = 200
        self.assertIs(PersonThermalContrast(1, FixedGain(1.5))(labels)['img'], original)

    def test_ignore_and_padding_masks_leave_pixel_and_label_untouched(self):
        labels = self.sample()
        before = labels['img'].copy()
        semantic = np.zeros(before.shape[:2], np.uint8)
        semantic[29:33, 28:35] = 1
        semantic[36:40, 70:75] = 255
        labels['semantic_mask'] = semantic
        result = PersonThermalContrast(1, FixedGain(0.4))(labels)
        np.testing.assert_array_equal(result['img'][semantic != 0], before[semantic != 0])
        self.assertIs(result['semantic_mask'], semantic)
        self.assertFalse(np.array_equal(result['img'], before))

    def test_trainer_inserts_augmentation_before_occlusion_in_train_only(self):
        class PartialCrop:
            pass

        class Occlude:
            pass

        class Base:
            def build_dataset(self, img_path, mode='train', batch=None):
                return SimpleNamespace(transforms=Compose([PartialCrop(), Occlude()]))

        self.assertIs(with_person_thermal_contrast(Base, 0), Base)
        trainer = with_person_thermal_contrast(Base, 0.45)()
        train = trainer.build_dataset('', 'train')
        validation = trainer.build_dataset('', 'val')
        self.assertIsInstance(train.transforms.transforms[1], PersonThermalContrast)
        self.assertIsInstance(train.transforms.transforms[2], Occlude)
        self.assertEqual(len(validation.transforms.transforms), 2)


if __name__ == '__main__':
    unittest.main()
