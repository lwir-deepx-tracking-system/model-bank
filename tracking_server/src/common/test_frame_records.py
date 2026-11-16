"""frame_records: the compact per-frame records must give the same file text as the old nested lists, and far fewer objects for
the garbage collector. Run from tracking_server: python -m unittest src.common.test_frame_records"""
import gc
import json
import unittest

import numpy as np

from src.common import frame_records as fr


def old_pred(pc, image_id):
    return dict(image_id=image_id, xyxy=np.round(pc[:, :4], 2).tolist(), conf=np.round(pc[:, 4], 5).tolist(), cls=pc[:, 5].astype(int).tolist())


def old_tracks(rows, dy, image_id):
    return dict(image_id=image_id, coco_image_id=3, video_id=1, tracker_reset=False,
                tracks_xyxy_id_conf_class_detectionindex=[[round(float(v) - (dy if j in (1, 3) else 0.0), 2) for j, v in enumerate(r[:4])]
                                                          + [int(r[4]), round(float(r[5]), 5), int(r[6]), int(r[7])] for r in rows])


def new_pred(pc, image_id):
    a, b, c = fr.pred_arrays(pc)
    return dict(image_id=image_id, xyxy=a, conf=b, cls=c)


def new_tracks(rows, dy, image_id):
    return dict(image_id=image_id, coco_image_id=3, video_id=1, tracker_reset=False,
                tracks_xyxy_id_conf_class_detectionindex=fr.track_array(rows, dy))


def dumps(r):
    return json.dumps(r, separators=(',', ':'))


class SameText(unittest.TestCase):
    def test_random_frames_give_identical_lines(self):
        rng = np.random.default_rng(7)
        for n in (0, 1, 5, 300):
            pc = np.column_stack([rng.uniform(-50, 700, (n, 4)), rng.uniform(0, 1, n), rng.integers(0, 16, n)]).astype(np.float32)
            # values that sit on rounding halves and exact integers
            if n >= 5: pc[:5, :4] = [[0.125, 1.005, 2.675, 100.0], [-0.005, 3.335, 7.5, 8.25], [1e-7, 639.995, 0, 1], [12.345, 6.785, 0.015, 9.995], [2.5, 3.5, 4.5, 5.5]]
            self.assertEqual(dumps(old_pred(pc, 'f%d' % n)), dumps(fr.plain(new_pred(pc, 'f%d' % n))))
            rows = [list(rng.uniform(-20, 660, 4)) + [int(rng.integers(1, 9999)), float(rng.uniform(0, 1)), int(rng.integers(0, 16)), int(rng.integers(-1, 300))] for _ in range(n)]
            if n >= 2: rows[0][:4] = [0.125, 17.005, 2.675, 100.0]; rows[1][5] = 0.123455
            for dy in (16.0, 0.0, 64.0):
                self.assertEqual(dumps(old_tracks(rows, dy, 'g')), dumps(fr.plain(new_tracks(rows, dy, 'g'))))

    def test_plain_leaves_old_records_and_extra_keys_alone(self):
        rec = dict(image_id='x', xyxy=[[1.0, 2.0, 3.0, 4.0]], conf=[0.5], cls=[0], zoom=[1])
        self.assertEqual(fr.plain(rec), rec)
        pc = np.array([[1, 2, 3, 4, .5, 0]], np.float32)
        r = new_pred(pc, 'x'); r['zoom'] = [1]
        self.assertEqual(fr.plain(r)['zoom'], [1])


class FewerObjects(unittest.TestCase):
    def test_the_collector_has_far_less_to_walk(self):
        rng = np.random.default_rng(1)
        pc = np.column_stack([rng.uniform(0, 640, (150, 4)), rng.uniform(0, 1, 150), rng.integers(0, 16, 150)]).astype(np.float32)
        rows = [list(rng.uniform(0, 640, 4)) + [7, 0.9, 0, 3] for _ in range(4)]
        gc.collect(); base = len(gc.get_objects())
        keep = [(old_pred(pc, i), old_tracks(rows, 16.0, i)) for i in range(200)]
        old_n = len(gc.get_objects()) - base
        del keep; gc.collect(); base = len(gc.get_objects())
        keep = [(new_pred(pc, i), new_tracks(rows, 16.0, i)) for i in range(200)]
        new_n = len(gc.get_objects()) - base
        self.assertGreater(old_n, 20 * new_n, (old_n, new_n))


if __name__ == '__main__':
    unittest.main()
