"""Parity and buffer-lifetime tests for native split detection."""
from ctypes import c_float
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from . import detect_frame, dual_tile
from .split_native import NativeTiles, Timing


@unittest.skipUnless(shutil.which('g++'), 'g++ is needed to build post.cpp')
class SplitPostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).parent
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        path = Path(cls.directory.name)
        subprocess.run(['g++', '-O3', '-shared', '-fPIC', '-o', str(path / 'post.so'),
                        str(source / 'post.cpp')], check=True)
        wrapper = path / 'native_post.py'
        shutil.copyfile(source / wrapper.name, wrapper)
        spec = importlib.util.spec_from_file_location('split_post_test_library', wrapper)
        cls.post = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.post)

    def setUp(self):
        backend = dual_tile.PairBackend.__new__(dual_tile.PairBackend)
        backend.native_post = self.post.prepare_split_heads
        # An SDK need not report the heads in stride order.
        backend.names = tuple(reversed(dual_tile.HEAD_ORDER))
        backend.head_indices = tuple(backend.names.index(name) for name in dual_tile.HEAD_ORDER)
        backend.shapes = {}
        self.tiles = [dict(), dict()]
        for stride, box, score, pose in zip(dual_tile.STRIDES, dual_tile.BOXES,
                                            dual_tile.CLASSES, dual_tile.POSES):
            h, w = dual_tile.H // stride, dual_tile.TW // stride
            for name, shape, value in ((box, (1, 4, h, w), 0.5), (score, (1, 1, h, w), -14),
                                       (pose, (1, h, w, 51), 0)):
                backend.shapes[name] = shape
                for tile in self.tiles:
                    tile[name] = np.full(shape, value, np.float32)
        self.backend = backend
        app = patch.dict(vars(APP), native_post=self.post, fast_nms=self.post.fast_nms,
                         cv2=self.post.cv2, det_yoff=16)
        kp = patch.dict(detect_frame.KP, cur=None, prev=None)
        for context in (app, kp):
            context.start()
            self.addCleanup(context.stop)

    def waited(self):
        return [[tile[name] for name in self.backend.names] for tile in self.tiles]

    def assert_parity(self):
        waited = self.waited()
        legacy_rows = dual_tile.detect_legacy(self.backend.pack_legacy(waited))
        legacy_points = detect_frame.KP['cur']
        outputs = self.backend.pack(waited, owned=True)
        native_rows = detect_frame._detect(outputs, None)
        current = detect_frame.KP['cur']
        np.testing.assert_array_equal(native_rows, legacy_rows)
        np.testing.assert_array_equal(current['boxes'], legacy_points['boxes'])
        np.testing.assert_array_equal(current['kpts'][..., :2], legacy_points['kpts'][..., :2])
        np.testing.assert_allclose(current['kpts'][..., 2], legacy_points['kpts'][..., 2],
                                   rtol=0, atol=1.2e-7)
        self.assertIs(detect_frame.KP['prev'], legacy_points)
        np.testing.assert_array_equal(detect_frame._detect(outputs, None), native_rows)
        return native_rows, current['kpts']

    def test_sparse_dense_and_empty(self):
        empty, points = self.assert_parity()
        self.assertEqual((empty.shape, points.shape), ((0, 6), (0, 17, 3)))
        rng = np.random.default_rng(37)
        for count in (26, 2000):
            for tile in self.tiles:
                for box, score, pose in zip(dual_tile.BOXES, dual_tile.CLASSES, dual_tile.POSES):
                    tile[box][...] = rng.uniform(0.05, 5.5, tile[box].shape)
                    tile[pose][...] = rng.normal(0, 4, tile[pose].shape)
                    logits = tile[score].ravel()
                    logits.fill(-14)
                    logits[rng.choice(logits.size, min(count, logits.size), replace=False)] = (
                        rng.uniform(-7, 4, min(count, logits.size)))
            rows, _ = self.assert_parity()
            self.assertGreater(len(rows), 20)

    def test_near_tie_uses_first_anchor_and_applies_both_offsets(self):
        right = self.tiles[1]
        box = right[dual_tile.BOXES[0]][0, :, 0]
        score = right[dual_tile.CLASSES[0]][0, 0, 0]
        pose = right[dual_tile.POSES[0]][0, 0]
        box[:, 9], box[:, 10] = [1, 1, 2, 2], [2, 1, 1, 2]
        score[9], score[10] = np.float32(2.197219), np.float32(2.197225)
        pose[9, 0], pose[10, 0] = 7, 23
        # Reject an otherwise valid left anchor at the left/right boundary.
        self.tiles[0][dual_tile.CLASSES[0]][0, 0, 0, 40] = 10
        rows, points = self.assert_parity()
        self.assertEqual(rows.shape, (1, 6))
        self.assertEqual(points[0, 0, 0], (7 + 9.5) * 8 + dual_tile.RIGHT)
        self.assertEqual(points[0, 0, 1], 20)
        # Greater score difference selects the actual surviving NMS anchor.
        score[9] = 2.1971
        _, points = self.assert_parity()
        self.assertEqual(points[0, 0, 0], (23 + 10.5) * 8 + dual_tile.RIGHT)

    def test_nonfinite_head_and_python_fallback_ownership(self):
        name = dual_tile.POSES[2]
        head = self.tiles[1][name]
        result = self.backend.pack(self.waited())
        head.flat[0] = np.nan
        self.assertEqual(result.arrays[-1].flat[0], 0)
        with self.assertRaisesRegex(RuntimeError, f'{name} on core 1'):
            self.backend.pack(self.waited(), owned=True)


class SplitSdkOutputTests(unittest.TestCase):
    def test_owned_wait_buffers_survive_another_job(self):
        sdk = NativeTiles.__new__(NativeTiles)
        sdk.timing = Timing()
        sdk.layout = [('box', 1, 4, (1, 4), 16)]
        sdk.output_size = 16
        sdk.buffers = [np.empty(16, np.uint8) for _ in range(2)]
        sdk.views = [sdk.tensor_views(buf) for buf in sdk.buffers]
        sdk.handle = 1

        def wait(_, left, right, size, timing):
            for pointer, value in ((left, timing._obj.job_id[0]), (right, timing._obj.job_id[1])):
                array = np.ctypeslib.as_array((c_float * (size // 4)).from_address(pointer))
                array.fill(value)
            return 0

        sdk.lib = SimpleNamespace(lwir_split_wait=wait)
        sdk.timing.job_id[:] = [1, 2]
        first = sdk.wait([1, 2], ['box'], owned=True)
        sdk.timing.job_id[:] = [3, 4]
        second = sdk.wait([3, 4], ['box'], owned=True)
        for i in (0, 1):
            np.testing.assert_array_equal(first[i][0], np.full((1, 4), i + 1))
            np.testing.assert_array_equal(second[i][0], np.full((1, 4), i + 3))
