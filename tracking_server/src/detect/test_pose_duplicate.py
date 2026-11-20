"""The nested-duplicate rule of pose detectors (pose_duplicate.py) and the switch that says for which detectors it runs.

Run from the tracking_server folder:  python -m unittest src.detect.test_pose_duplicate

The rule is checked on made-up boxes (x1, y1, x2, y2, score) and COCO poses (17 x (x, y, score)); no camera data.
A person "with a seen face" = the 7 upper keypoints (nose, eyes, ears, shoulders) at one point with score .75;
a "headless" box = all keypoint scores 0. The Python rule is the reference; when the C++ helper is built
(liblwir_frame_math.so) the same boxes must give the same answer.
"""
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from ..common import frame_math
from . import detect_frame, pose_duplicate

NAMED = pose_duplicate.NAMED_DETS[1]
GENERAL = 'bank:yolo26m_pose_ft16v8_aug_e10_bodybox_provisional_kpt_640x480'   # the general detector: not one of the three names
NO_KPT = 'bank:yolo26m_flir_640x480'


def scene(*people):
    """people: (x1, y1, x2, y2, score, face (x, y) or None) -> boxes [n, 5] float32, poses [n, 17, 3] float32"""
    boxes = np.array([p[:5] for p in people], np.float32)
    poses = np.zeros((len(people), 17, 3), np.float32)
    for row, p in enumerate(people):
        if p[5] is not None:
            poses[row, :7] = [p[5][0], p[5][1], .75]
    return boxes, poses


def keep(boxes, poses):
    with patch.object(frame_math, '_nested', None):
        reference = pose_duplicate.keep_person_poses(boxes, poses)
    native = pose_duplicate.keep_person_poses(boxes, poses)   # the C++ helper when it is built, else the same Python rule
    np.testing.assert_array_equal(native, reference)
    return reference.tolist()


class RuleTests(unittest.TestCase):
    def test_headless_full_height_box_around_an_upper_body_box_is_removed(self):
        """One person, two boxes: the upper-body box with the face (kept) and a weaker box 1.67 times taller around it (removed)."""
        boxes, poses = scene((24, 24, 68, 84, .9, (44, 35)), (21, 22, 70, 122, .4, None))
        self.assertEqual(keep(boxes, poses), [True, False])

    def test_it_is_the_large_box_that_goes_whatever_the_order(self):
        boxes, poses = scene((21, 22, 70, 122, .4, None), (24, 24, 68, 84, .9, (44, 35)), (300, 100, 340, 200, .8, (320, 115)))
        self.assertEqual(keep(boxes, poses), [False, True, True])

    def test_two_people_side_by_side_are_kept(self):
        apart, _ = scene((100, 100, 150, 220, .9, (125, 118)), (170, 102, 222, 224, .8, (196, 120)))
        touching, _ = scene((100, 100, 150, 220, .9, (125, 118)), (140, 102, 192, 224, .8, (166, 120)))
        for boxes in (apart, touching):
            for faces in (((125, 118), (boxes[1, 0] + 26, 120)), ((125, 118), None), (None, None)):   # both seen, one turned away, both turned away
                people = [(*boxes[i, :5].tolist(), faces[i]) for i in range(2)]
                self.assertEqual(keep(*scene(*people)), [True, True], (boxes.tolist(), faces))

    def test_a_person_half_hidden_behind_another_is_kept_when_a_second_face_is_seen(self):
        """The large box has two faces in its upper part: it may be the person behind, so it stays."""
        boxes, poses = scene((24, 24, 68, 84, .9, (44, 35)), (21, 22, 70, 122, .4, None), (31, 29, 58, 68, .8, (35, 39)))
        self.assertEqual(keep(boxes, poses), [True, True, True])

    def test_child_standing_in_front_of_an_adult_both_kept(self):
        """The child's box lies fully inside the adult's box, but its top is far below the adult's top: not the same head."""
        for adult_face in ((140, 70), None):
            boxes, poses = scene((100, 50, 180, 290, .5, adult_face), (118, 170, 162, 290, .8, (140, 190)))
            self.assertEqual(keep(boxes, poses), [True, True], adult_face)

    def test_child_at_the_adults_head_height_KNOWN_LIMIT(self):
        """A small person whose box starts at the top of the large box and sits in its middle (a carried child, a person
        sitting right in front): when the large box's own head and shoulders are NOT seen and its score is not higher, the
        rule cannot tell this from a duplicate and removes the LARGE box (the adult is then not tracked; the child is)."""
        boxes, poses = scene((100, 50, 180, 290, .5, None), (118, 56, 162, 196, .8, (140, 80)))
        self.assertEqual(keep(boxes, poses), [False, True])
        boxes, poses = scene((100, 50, 180, 290, .5, (140, 70)), (118, 56, 162, 196, .8, (140, 80)))   # the adult's face is seen
        self.assertEqual(keep(boxes, poses), [True, True])
        boxes, poses = scene((100, 50, 180, 290, .9, None), (118, 56, 162, 196, .8, (140, 80)))        # the adult's score is higher
        self.assertEqual(keep(boxes, poses), [True, True])

    def test_height_ratio_limits_as_written(self):
        """The rule starts at a height ratio of 1.30. The nested pairs seen live on 2026-11-20 had 1.18 to 1.33 (audit,
        verify_code v03): a pair at 1.25 is NOT removed. Changing 1.30 means changing the C++ helper too (frame_math.cpp)."""
        small = (100, 100, 160, 230, .9, (130, 120))   # 130 px tall
        for ratio, removed in ((1.20, False), (1.25, False), (1.35, True), (2.10, True), (2.30, False)):
            boxes, poses = scene(small, (98, 98, 162, 98 + 130 * ratio, .4, None))
            self.assertEqual(keep(boxes, poses), [True, not removed], ratio)

    def test_small_or_weaker_upper_body_box_does_not_remove_anything(self):
        boxes, poses = scene((24, 24, 54, 64, .9, (39, 32)), (21, 22, 57, 90, .4, None))     # the small box is 40 px tall (< 48)
        self.assertEqual(keep(boxes, poses), [True, True])
        boxes, poses = scene((24, 24, 68, 84, .3, (44, 35)), (21, 22, 70, 122, .4, None))    # the large box has the higher score
        self.assertEqual(keep(boxes, poses), [True, True])

    def test_rows_and_pose_cache_stay_aligned(self):
        boxes, poses = scene((24, 24, 68, 84, .9, (44, 35)), (21, 22, 70, 122, .4, None))
        car = np.array([[200, 200, 300, 260, .7, 2]], np.float32)
        detections = np.vstack([np.hstack([boxes[1:], [[0]]]), car, np.hstack([boxes[:1], [[0]]])]).astype(np.float32)
        cache = dict(boxes=detections[[0, 2], :4].copy(), kpts=poses[[1, 0]].copy())
        with patch.object(frame_math, '_nested', None):
            out, kept = pose_duplicate.suppress_nested_people(detections, cache, 0)
        np.testing.assert_array_equal(out, detections[1:])
        np.testing.assert_array_equal(kept['boxes'], detections[2:, :4])
        np.testing.assert_array_equal(kept['kpts'], poses[:1])


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(pose_duplicate.RULE)
        self.kpt = {NAMED, GENERAL}

    def tearDown(self):
        pose_duplicate.RULE.clear(); pose_duplicate.RULE.update(self.saved)

    def test_default_is_the_behaviour_before_the_switch(self):
        with patch.dict(os.environ):
            os.environ.pop('LWIR_NESTED_DUP', None)
            self.assertEqual(pose_duplicate._start_mode(), 'named')
        pose_duplicate.RULE['mode'] = 'named'
        for det in pose_duplicate.NAMED_DETS:
            self.assertTrue(pose_duplicate.applies(det, self.kpt))
        for det in (GENERAL, NO_KPT, 'w480', 'bank:yolo26m_pose_lwir650_20261119_bodybox_kpt_640x480_copy'):
            self.assertFalse(pose_duplicate.applies(det, self.kpt), det)

    def test_kpt_follows_the_keypoint_heads_not_the_name(self):
        self.assertTrue(pose_duplicate.set_mode('kpt'))
        self.assertTrue(pose_duplicate.applies(GENERAL, self.kpt))
        self.assertTrue(pose_duplicate.applies(NAMED, self.kpt))
        self.assertFalse(pose_duplicate.applies(NO_KPT, self.kpt))
        self.assertFalse(pose_duplicate.applies(NAMED, set()))   # a file listed without keypoint heads gives no poses: nothing to judge

    def test_off_and_bad_values(self):
        self.assertTrue(pose_duplicate.set_mode('off'))
        self.assertFalse(any(pose_duplicate.applies(det, self.kpt) for det in (NAMED, GENERAL, NO_KPT)))
        self.assertFalse(pose_duplicate.set_mode('on'))
        self.assertEqual(pose_duplicate.RULE['mode'], 'off')

    def test_start_value_from_the_environment(self):
        for value, mode in (('kpt', 'kpt'), ('OFF', 'off'), ('named', 'named'), ('1', 'named'), ('', 'named')):
            with patch.dict(os.environ, {'LWIR_NESTED_DUP': value}):
                self.assertEqual(pose_duplicate._start_mode(), mode)

    def detect(self, det):
        """detect_frame.detect on one frame whose detector output is the one-person-two-boxes scene above"""
        boxes, poses = scene((24, 24, 68, 84, .9, (44, 35)), (21, 22, 70, 122, .4, None))
        rows = np.hstack([boxes, np.zeros((2, 1), np.float32)])

        def raw_detect(ra, rb):
            detect_frame.KP['cur'] = dict(boxes=rows[:, :4].copy(), kpts=poses.copy()) if det in self.kpt else detect_frame.KP['cur']
            return rows.copy()

        app = NS(args=NS(det=det, min_person_h=0.0), _detect=raw_detect)
        saved = dict(detect_frame.KP)
        try:
            with patch.object(detect_frame, 'APP', app), patch.object(detect_frame, 'KPT_DETS', self.kpt), \
                    patch.object(frame_math, '_nested', None):
                out = detect_frame.detect(None, None)
                return len(out), len(detect_frame.KP['cur']['boxes']) if det in self.kpt else None
        finally:
            detect_frame.KP.clear(); detect_frame.KP.update(saved)

    def test_frame_loop_obeys_the_switch(self):
        want = {'named': {NAMED: 1, GENERAL: 2, NO_KPT: 2}, 'kpt': {NAMED: 1, GENERAL: 1, NO_KPT: 2}, 'off': {NAMED: 2, GENERAL: 2, NO_KPT: 2}}
        for mode, by_det in want.items():
            pose_duplicate.set_mode(mode)
            for det, rows in by_det.items():
                n_rows, n_poses = self.detect(det)
                self.assertEqual(n_rows, rows, (mode, det))
                if n_poses is not None:
                    self.assertEqual(n_poses, rows, (mode, det))   # the pose cache loses the same row
            dropped = sum(2 - rows for rows in by_det.values())
            self.assertEqual(pose_duplicate.state(GENERAL, self.kpt),
                             dict(nested_dup=mode, nested_dup_active=mode == 'kpt', nested_dup_dropped=dropped))


if __name__ == '__main__':
    unittest.main()
