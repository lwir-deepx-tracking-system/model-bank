"""Compare each native frame kernel with the established Python path."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import frame_math
from .state import APP
from ..detect import detect_frame
from ..detect import pose_duplicate
from ..gimbal import live_tracks
from ..reid import relink_online_judge as judge
from ..track import box_refine
from ..track import position_memory as position


@unittest.skipUnless(shutil.which('g++'), 'g++ is needed to build frame_math.cpp')
class FrameMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        source = Path(frame_math.__file__).with_suffix('.cpp')
        output = Path(folder.name) / 'frame_math.so'
        subprocess.run(['g++', '-O3', '-std=c++17', '-ffp-contract=off', '-shared', '-fPIC',
                        '-o', str(output), str(source)], check=True)
        cls.native = ctypes.CDLL(str(output))

    def setUp(self):
        library = patch.object(frame_math, 'lib', self.native)
        library.start()
        self.addCleanup(library.stop)
        keepid = patch.object(frame_math, '_keepid', frame_math.bind(
            'keepid_geometry', [frame_math.f64, frame_math.i64, frame_math.i32, ctypes.c_int,
                                frame_math.f64, ctypes.c_int, ctypes.c_double, frame_math.f64,
                                frame_math.u8]))
        keepid.start()
        self.addCleanup(keepid.stop)
        native_position = patch.object(frame_math, '_position', frame_math.bind(
            'position_seen', [frame_math.f64, frame_math.f64, frame_math.f64, frame_math.u8,
                              ctypes.c_int, frame_math.f32, frame_math.f32, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, frame_math.f64, frame_math.f64]))
        native_position.start()
        self.addCleanup(native_position.stop)
        p = ctypes.POINTER(ctypes.c_void_p)
        kernels = (
            ('_nested', 'nested_people', [frame_math.f32, frame_math.f32, ctypes.c_int, frame_math.u8]),
            ('_merge', 'merge_heads', [p, frame_math.i32, frame_math.i32, ctypes.c_int, p]),
            ('_anchors', 'pose_anchors', [frame_math.f32, ctypes.c_int, frame_math.f32,
                                         ctypes.c_int, frame_math.i64]),
            ('_body', 'body_deltas', [frame_math.f64, frame_math.f64, ctypes.c_int, frame_math.f64,
                                     ctypes.c_int, frame_math.f64, ctypes.c_int, frame_math.f64]),
            ('_motion', 'motion_chain', [frame_math.f64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_double, ctypes.c_double, frame_math.f64]),
        )
        for attribute, name, args in kernels:
            kernel = patch.object(frame_math, attribute, frame_math.bind(name, args))
            kernel.start()
            self.addCleanup(kernel.stop)

    def test_keepid_geometry_and_boundary_fallback(self):
        rng = np.random.default_rng(651901)
        for count in (4, 17, 60):
            boxes = rng.uniform(-112, 608, (2 * count, 4))
            boxes[:, 2:] = boxes[:, :2] + rng.uniform(1, 120, boxes[:, 2:].shape)
            boxes = np.round(boxes, 2).tolist()
            D = {i * 13: box for i, box in enumerate(boxes)}
            rows = [[*box, (1 << 54) + i // 2, .55, 0, i * 13 if i % 3 else -1]
                    for i, box in enumerate(boxes[:count])]
            fast = frame_math.keepid_geometry(rows, D, judge.NEAR_H)
            self.assertTrue(any(g is not None for g in fast))
            for row, metrics in zip(rows, fast):
                feature = np.array([1., 0.])
                slow = judge.make_obs(10, row, D, rows, feature)
                native = judge.make_obs(10, row, D, rows, feature, geometry=metrics)
                for k in slow:
                    if k == 'm1':
                        self.assertIs(native[k], slow[k])
                    elif k == 'crop_h' and not slow['in_P']:
                        self.assertTrue(np.isnan(native[k]))
                    else:
                        self.assertEqual(native[k], slow[k], k)

        boxes = {i: [1000. + i, 50., 1010. + i, 60.] for i in range(32)}
        boxes[4] = [0., 0., 2., 10.]
        boxes[13] = [20., 0., 22., 10.]  # exactly two heights away
        rows = [[0., 0., 2., 10., 1, .9, 0, 4]]
        self.assertEqual(frame_math.keepid_geometry(rows, boxes, 2.), [None])
        self.assertEqual(judge.make_observations(1, rows, boxes, {})[1]['n_near_det'], 1)
        boxes[13] = np.array(boxes[13], np.float32)
        self.assertIsNone(frame_math.keepid_geometry(rows, boxes, 2.))

    def test_keepid_public_ids_and_frozen_features(self):
        columns = ['miou_new', 'n_near_trk_new', 'n_near_det_new', 'm1_cos_mean5',
                   'conf_last5', 'iou_last', 'rank_d']
        # Use the real feature names in pair_features(), as a string list
        # keeps the test independent from the deployed weights file.
        with patch.object(judge, 'load_spec', return_value=(
            {'model': {}, 'features': columns, 'decision': {'threshold': .5}}, 'fixture')):
            slow, fast = (judge.OnlineJudge('fixture', compact_ids=True) for _ in range(2))
        for session in (slow, fast):
            session.keep_rows = True
        for frame in range(25):
            ids = (10, 20, 30, 40) if frame < 12 else (50, 20, 30, 40)
            boxes = {}
            for i in range(45):
                x = float(i * 13)
                boxes[i] = [x, 20., x + 30., 95.]
            rows = [[*boxes[i], ident, .9, 0, i] for i, ident in enumerate(ids)]
            self.assertGreater(sum(g is not None for g in frame_math.keepid_geometry(rows, boxes, 2.)), 0)
            def embed(keys):
                return {k: np.array([1., 0.]) for k in keys}
            with patch.object(judge, 'predict_json', side_effect=lambda model, rows: np.full(len(rows), .9)):
                with patch.object(frame_math, '_keepid', None):
                    expected = slow.step(rows, boxes, embed, frame)
                actual = fast.step(rows, boxes, embed, frame)
            self.assertEqual(actual, expected)
        self.assertTrue(fast.events)
        self.assertEqual(fast.events, slow.events)
        self.assertEqual(fast.stats, slow.stats)
        self.assertEqual(fast.id_map, slow.id_map)
        for a, b in zip(fast.rows_log, slow.rows_log, strict=True):
            self.assertEqual(a[:4], b[:4])
            np.testing.assert_array_equal(a[4], b[4])
            np.testing.assert_array_equal(a[5], b[5])

    @unittest.skipUnless(frame_math._float32_with_float64_scalar, 'requires NumPy 2 scalar promotion')
    def test_position_batch_matches_previous_pose_and_features(self):
        rng = np.random.default_rng(72)
        boxes = np.array([[100.125, 43.533, 153.88, 117.92], [2, 20, 12, 30],
                          [100.125, 43.533, 153.88, 117.92], [5, 20, 10, 30]], np.float32)
        kpts = rng.uniform(0, 150, (4, 17, 3)).astype(np.float32)
        kpts[..., 2] = np.resize(np.array([.49999997, .5, .50000006], np.float32), (4, 17))
        poses = dict(boxes=boxes, kpts=kpts)
        states = np.array([[127., 79., 52., 70., -2., 4.], [7., 25., 10., 10., 1., -3.],
                           [1024., 45., 1., 0., 80., -7.], [7.5, 25., 5., 10., 0., 0.]])
        previous = [states[2, :4].copy(), None, states[0, :4].copy(), None]
        innovations = rng.normal(size=(len(states), 4))
        for pose_result in (poses, None):
            for posetrack, edges in ((False, True), (True, False)):
                z, features = frame_math.position_seen(states, innovations, previous, pose_result,
                                                       posetrack, edges)
                with patch.dict(detect_frame.KP, prev=pose_result):
                    for i, row in enumerate(states):
                        z_old, kp = position._kp_lookup(row)
                        if z_old is None:
                            z_old = row[:4]
                        zp = z_old if previous[i] is None else previous[i]
                        feat = position._kp_feat_seen(row[4:6], z_old, zp, innovations[i], kp,
                                                      posetrack, edges)
                        np.testing.assert_array_equal(z[i], z_old)
                        np.testing.assert_array_equal(features[i], feat)
        kpts[0, 0, 0] = np.nan
        self.assertIsNone(frame_math.position_seen(states, innovations, previous, poses, False, True))

    def test_nested_boxes_leave_occluded_neighbor_when_two_faces_are_seen(self):
        boxes = np.array([[24, 24, 68, 84, .9], [21, 22, 70, 122, .4],
                          [31, 29, 58, 68, .8], [100, 0, 300, 110, .6]], np.float32)
        poses = np.zeros((len(boxes), 17, 3), np.float32)
        poses[0, :7] = [44., 35., .75]
        for two_faces in (False, True):
            if two_faces:
                poses[2, :7] = [35., 39., .75]
            native = pose_duplicate.keep_person_poses(boxes, poses)
            with patch.object(frame_math, '_nested', None):
                reference = pose_duplicate.keep_person_poses(boxes, poses)
            np.testing.assert_array_equal(native, reference)
            self.assertEqual(native[1], two_faces)

    def test_ordinary_head_merge_preserves_stride_and_frame_ownership(self):
        rng = np.random.default_rng(7)
        inputs = [rng.normal(size=(1, c, n)).astype(np.float32)
                  for c in (4, 16, 51) for n in (67, 34, 4)]
        packed = frame_math.merge_heads(inputs, inputs)
        for i in range(3):
            np.testing.assert_array_equal(packed[i], np.concatenate(inputs[i*3:i*3+3], axis=2))
        saved = packed[0].copy()
        inputs[0].fill(2)
        np.testing.assert_array_equal(packed[0], saved)
        inputs[7][0, 0, 0] = np.inf
        self.assertIsNone(frame_math.merge_heads(inputs, inputs))

    def test_ordinary_anchor_search_keeps_first_matching_corner(self):
        rng = np.random.default_rng(775)
        dec = rng.uniform(30, 500, (20, 1024)).astype(np.float32)
        dec[4] = rng.uniform(0, 1, 1024)
        dec[:5, 76] = dec[:5, 3]
        dec[4, 3] -= np.float32(5e-7)
        selected = np.r_[76, np.arange(300, 400)]
        rows = np.zeros((len(selected), 6), np.float32)
        rows[:, :2] = (dec[:2, selected] - dec[2:4, selected] * .5).T
        rows[:, 4] = dec[4, selected]
        anchors = frame_math.pose_anchors(dec, rows)
        np.testing.assert_array_equal(anchors, detect_frame._kp_anchors_loop(dec, rows))
        self.assertEqual(anchors[0], 3)

    def test_body_table_conversion_clamps_and_matches_original(self):
        rng = np.random.default_rng(61)
        tab = box_refine.BodyTable(dict(tab=rng.uniform(-.3, .1, (5, 4, 4)),
                                      lh_cen=np.log([25., 55., 92., 126., 200.]),
                                      ar_cen=np.array([.3, .4, .6, .8])))
        boxes = np.zeros((80, 4))
        boxes[:, 2:] = rng.uniform(4, 650, (80, 2))
        boxes[:2, 2:] = [[1, 1], [77, 200]]
        values = tab.conv_delta(boxes)
        with patch.object(frame_math, '_body', None):
            expected = tab.conv_delta(boxes)
        np.testing.assert_array_equal(values, expected)

    def test_motion_chain_wraps_and_preserves_oldest_matching_step(self):
        rng = np.random.default_rng(17)
        hist = frame_math.MotionHistory(8)
        for i in range(13):
            hist.append((i, i + 1, rng.normal(0, .1, (2, 3)) + np.eye(2, 3)))
        hist.append((10, 11, np.eye(2, 3)))  # duplicate start, first one wins
        for begin, end in ((10, 11), (6, 13), (4, 8), (6, 19), (8, 8)):
            with patch.dict(vars(APP), CTL={'motion_hist': list(hist)}):
                expected = live_tracks._motion(begin, end)
            with patch.dict(vars(APP), CTL={'motion_hist': hist}):
                got = live_tracks._motion(begin, end)
            if expected is None:
                self.assertIsNone(got)
            else:
                np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)
