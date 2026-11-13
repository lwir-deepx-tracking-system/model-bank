"""Frame-by-frame comparison of the C++ SORT / OC-SORT / BoT-SORT extensions with the Python trackers.

Run from the tracking_server folder after `bash scripts/build_native_trackers.sh`:
    python3 -m unittest src.track.test_native_trackers
Without the built extensions every test is skipped (the server then uses Python anyway).

Both implementations get the same synthetic detections, frame by frame: people walking, hidden for a
while, missed detections, low scores, false boxes, empty frames, optional camera motion and appearance
features. Compared each frame: output rows (count, order, ID, class, detection row exactly; box and
score within TOL), the tracked / lost lists, the ID counter and the per-frame new / lost lists.
"""
import sys
import unittest

import numpy as np

from ..common import settings
from . import native_backend
from .botsort import BotSortTracker
from .bytetrack import crop_requests
from .kalman import GKF, gmc_warp
from .ocsort import OCSortTracker
from .position_memory import POSP
from .sort import SortTracker
from .stabilizer import STAB_M
from .tracker_common import linear_assignment

TOL = 1e-6
PYTHON = {'sort': SortTracker, 'ocsort': OCSortTracker}


def make_pair(kind, **kwargs):
    """(python tracker, native tracker) for one kind, or None when the extension is not built."""
    if kind == 'botsort':
        py = lambda: BotSortTracker('bytetrack', **kwargs)
    else:
        py = lambda: PYTHON[kind](**kwargs)
    native = native_backend.create(kind, kind, kwargs, py)
    if native_backend.backend_of(native) != 'C++':
        return None
    return py(), native


def scenario(seed, frames=300, camera=False, features=False, classes=(0,), crowd=6, feature_dtype=np.float32):
    """Yield (arr [N,6] float32, warp or None, {det_idx: feature} or None) per frame."""
    rng = np.random.default_rng(seed)
    people, serial = [], 0
    for frame in range(frames):
        warp = None
        if camera:
            angle = rng.normal(0, 0.004)
            warp = np.array([[np.cos(angle), -np.sin(angle), rng.normal(0, 2.5)],
                             [np.sin(angle), np.cos(angle), rng.normal(0, 1.5)]])
            for p in people:
                p['c'] = warp[:, :2] @ p['c'] + warp[:, 2]
        if len(people) < crowd and rng.random() < 0.2:
            serial += 1
            h = rng.uniform(14, 150)
            base = rng.normal(size=32)
            people.append(dict(c=np.array([rng.uniform(0, 640), rng.uniform(40, 480)]), wh=np.array([h * rng.uniform(0.3, 0.5), h]),
                               v=rng.normal(0, 2.5, 2), cls=int(rng.choice(classes)), hidden=0, serial=serial, base=base / np.linalg.norm(base)))
        rows, owners = [], []
        for p in people:
            p['c'] = p['c'] + p['v'] + rng.normal(0, 0.5, 2)
            if rng.random() < 0.03: p['v'] = rng.normal(0, 3.5, 2)
            if p['hidden'] == 0 and rng.random() < 0.03: p['hidden'] = int(rng.integers(1, 45))
            if p['hidden'] > 0:
                p['hidden'] -= 1
                continue
            if rng.random() < 0.08: continue
            wh = p['wh'] * (1 + rng.normal(0, 0.03, 2)); c = p['c'] + rng.normal(0, 0.8, 2)
            score = float(np.clip(rng.normal(0.6, 0.25), 0.02, 0.99))
            if rng.random() < 0.02: score = float(rng.choice([0.10, 0.25]))   # exactly on a threshold
            rows.append([c[0] - wh[0] / 2, c[1] - wh[1] / 2, c[0] + wh[0] / 2, c[1] + wh[1] / 2, score, p['cls']]); owners.append(p)
        for _ in range(rng.poisson(0.4)):   # false boxes
            x, y, w, h = rng.uniform(0, 600), rng.uniform(16, 440), rng.uniform(4, 60), rng.uniform(8, 120)
            rows.append([x, y, x + w, y + h, float(rng.uniform(0.05, 0.7)), int(rng.choice(classes))]); owners.append(None)
        if rng.random() < 0.02:   # zero-width box: must be ignored
            rows.append([100.0, 100.0, 100.0, 150.0, 0.9, 0]); owners.append(None)
        people = [p for p in people if -80 < p['c'][0] < 720 and -80 < p['c'][1] < 590 and rng.random() > 0.004]
        if rng.random() < 0.02: rows, owners = [], []   # empty frame
        order = rng.permutation(len(rows))
        arr = np.asarray([rows[i] for i in order], np.float32).reshape(-1, 6)
        owners = [owners[i] for i in order]
        feats = None
        if features and len(arr):
            best = [int(j) for j in np.argsort(-arr[:, 4]) if int(arr[j, 5]) == 0 and arr[j, 4] >= 0.25 and arr[j, 3] - arr[j, 1] >= 32.0][:4]
            feats = {}
            for j in best:
                f = (owners[j]['base'] if owners[j] is not None else rng.normal(size=32)) + rng.normal(0, 0.25, 32)
                feats[j] = (f / np.linalg.norm(f)).astype(feature_dtype)
        yield arr, warp, feats


def server_camera_step(tracker, warp):
    """What app.py does with --gmc before update: pending warp for the pool, unconfirmed tracks warped now."""
    tracker.kf.warp = warp
    for t in tracker.tracked:
        if not t.is_activated and t.mean is not None:
            m, c = gmc_warp(t.mean[None].copy(), t.cov[None], warp)
            t.mean, t.cov = m[0], c[0]


def end_all_tracks(tracker):
    """What app.py does after an input gap of over one second."""
    for t in tracker.tracked + tracker.lost: t.state = 3
    tracker.tracked = []; tracker.lost = []


class NativeTrackerComparison(unittest.TestCase):
    def pair(self, kind, **kwargs):
        made = make_pair(kind, **kwargs)
        if made is None:
            self.skipTest('C++ %s extension is not built' % kind)
        return made

    def same_tracks(self, where, a, b):
        self.assertEqual([t.track_id for t in a], [t.track_id for t in b], where)
        for x, y in zip(a, b):
            self.assertEqual((x.state, bool(x.is_activated), x.cls, x.idx, x.end_frame, x.start_frame, x.feat_frame),
                             (y.state, bool(y.is_activated), y.cls, y.idx, y.end_frame, y.start_frame, y.feat_frame), where)
            np.testing.assert_allclose(y.mean, x.mean, rtol=0, atol=TOL, err_msg=where)
            np.testing.assert_allclose(y.cov, x.cov, rtol=1e-9, atol=TOL, err_msg=where)
            self.assertEqual(x.feat is None, y.feat is None, where)
            if x.feat is not None:
                self.assertEqual(x.feat.dtype, y.feat.dtype, where)
                np.testing.assert_allclose(y.feat, x.feat, rtol=0, atol=1e-5, err_msg=where)
            self.assertEqual(x.last_det[:2], y.last_det[:2], where)
            np.testing.assert_allclose(y.last_det[2], x.last_det[2], rtol=0, atol=TOL, err_msg=where)

    def run_pair(self, py, native, frames, camera=False, gap_at=(), n_init_at=None, requests=False):
        rows_total = 0
        if camera:
            py.kf = GKF(); native.kf = GKF()
        for i, (arr, warp, feats) in enumerate(frames):
            where = 'frame %d' % (i + 1)
            if warp is not None:
                server_camera_step(py, warp); server_camera_step(native, warp)
            if i in gap_at:
                end_all_tracks(py); end_all_tracks(native)
            if n_init_at and i in n_init_at:
                py.n_init = native.n_init = n_init_at[i]
            a = py.update(arr, feats, None)
            b = native.update(arr, feats, None)
            self.assertEqual(len(a), len(b), where)
            for ra, rb in zip(a, b):
                self.assertEqual(ra[4:5] + ra[6:], rb[4:5] + rb[6:], where)             # ID, class, detection row
                self.assertTrue(all(type(v) is int for v in rb[4:5] + rb[6:]), where)
                np.testing.assert_allclose(rb[:4] + rb[5:6], ra[:4] + ra[5:6], rtol=0, atol=TOL, err_msg=where)
            self.assertEqual((py.frame_id, py.next_id), (native.frame_id, native.next_id), where)
            self.same_tracks(where + ' tracked', py.tracked, native.tracked)
            self.same_tracks(where + ' lost', py.lost, native.lost)
            self.assertEqual([t.track_id for t in py.new_this_frame], [t.track_id for t in native.new_this_frame], where)
            self.assertEqual([t.track_id for t in py.lost_this_frame], [t.track_id for t in native.lost_this_frame], where)
            if requests:
                qa = crop_requests(py, 3, 6, 0.25, 32.0, True); qb = crop_requests(native, 3, 6, 0.25, 32.0, True)
                self.assertEqual([(q['kind'], q['track_id'], q['frame'], q['idx'], q['src']) for q in qa],
                                 [(q['kind'], q['track_id'], q['frame'], q['idx'], q['src']) for q in qb], where)
            rows_total += len(a)
        self.assertGreater(rows_total, 0)
        return rows_total

    def test_assignment_matches_python(self):
        made = self.pair('sort', n_init=3, class_gate=False, radius=2.0)
        native_la = sys.modules['_native_sort'].linear_assignment
        rng = np.random.default_rng(7)
        for case in range(4000):
            n, m = int(rng.integers(0, 9)), int(rng.integers(0, 9))
            if case % 3 == 0: cost = rng.integers(0, 4, (n, m)) / 4.0             # many ties
            elif case % 3 == 1: cost = np.round(rng.random((n, m)), 1)           # ties and threshold-equal costs
            else: cost = rng.random((n, m))
            cost = cost.astype(np.float32 if case % 2 else np.float64)
            for thresh in (0.5, 0.7, 0.8, 0.95):
                want = linear_assignment(cost.copy(), thresh)
                got = native_la(cost, thresh)
                self.assertEqual(([tuple(p) for p in want[0]], list(want[1]), list(want[2])), (list(got[0]), list(got[1]), list(got[2])),
                                 'case %d thresh %s\n%r' % (case, thresh, cost))
        del made

    def test_sort(self):
        for seed in range(6):
            py, native = self.pair('sort', n_init=3, class_gate=False, radius=2.0)
            self.run_pair(py, native, scenario(seed))

    def test_ocsort(self):
        for seed in range(6):
            py, native = self.pair('ocsort', n_init=3, class_gate=False, radius=2.0)
            self.run_pair(py, native, scenario(100 + seed))

    def test_botsort_without_appearance(self):
        for seed in range(4):
            py, native = self.pair('botsort', n_init=2, class_gate=False, radius=2.0, sim=0.5, app=False)
            self.run_pair(py, native, scenario(200 + seed), requests=True)

    def test_botsort_with_appearance(self):
        for seed, dtype in ((0, np.float32), (1, np.float32), (2, np.float64)):
            py, native = self.pair('botsort', n_init=2, class_gate=False, radius=2.0)
            self.run_pair(py, native, scenario(300 + seed, features=True, feature_dtype=dtype), requests=True)
            self.assertEqual(getattr(py, 'app_used', 0), native.app_used)

    def test_camera_motion(self):
        for kind, kw in (('sort', dict(n_init=3)), ('ocsort', dict(n_init=3)), ('botsort', dict(n_init=2))):
            for seed in range(3):
                py, native = self.pair(kind, class_gate=False, radius=2.0, **kw)
                self.run_pair(py, native, scenario(400 + seed, camera=True, features=kind == 'botsort'), camera=True)

    def test_class_gate_confirm_frames_and_input_gap(self):
        for kind in ('sort', 'ocsort', 'botsort'):
            for seed, n_init in ((0, 3), (1, 5), (2, 2)):
                py, native = self.pair(kind, n_init=n_init, class_gate=True, radius=2.0)
                self.run_pair(py, native, scenario(500 + seed, classes=(0, 2, 6)), gap_at=(90, 200), n_init_at={120: 4, 220: 3})

    def test_edge_birth_rule_and_stabilizer_margins(self):
        old = settings.EDGE_BIRTH, settings.EDGE_PX, STAB_M[0]
        try:
            for kind in ('sort', 'ocsort', 'botsort'):
                settings.EDGE_BIRTH, settings.EDGE_PX, STAB_M[0] = 0.5, 30.0, None
                py, native = self.pair(kind, n_init=3, class_gate=False, radius=2.0)
                self.run_pair(py, native, scenario(600))
                STAB_M[0] = (12.0, 8.0, 20.0, 6.0)
                py, native = self.pair(kind, n_init=3, class_gate=False, radius=2.0)
                self.run_pair(py, native, scenario(601))
        finally:
            settings.EDGE_BIRTH, settings.EDGE_PX, STAB_M[0] = old

    def test_position_predictor_switches(self):
        class FakeModel:
            def init(self): return 0
            def step(self, x, state): return np.array([0.2, -0.1, 0.05, 0.02]), state + 1

        old = dict(POSP)
        try:
            for kind in ('ocsort', 'botsort'):
                POSP.update(model=None, show=True)
                py, native = self.pair(kind, n_init=3, class_gate=False, radius=2.0)
                self.run_pair(py, native, scenario(700, frames=150))
                POSP.update(key='test', model=FakeModel(), gen=POSP['gen'] + 1, show=False, err=0, used=0)
                py, native = self.pair(kind, n_init=3, class_gate=False, radius=2.0)
                self.run_pair(py, native, scenario(701, frames=200, camera=True), camera=True)
                self.assertGreater(POSP['used'], 0)
                self.assertEqual(POSP['err'], 0)
        finally:
            POSP.clear(); POSP.update(old)

    def test_server_fields_and_late_feature(self):
        py, native = self.pair('botsort', n_init=2, class_gate=False, radius=2.0)
        for name in native_backend._REQUIRED_TRACKER_ATTRIBUTES:
            self.assertTrue(hasattr(native, name), name)
        arr = np.array([[100, 100, 140, 200, 0.9, 0]], np.float32)
        for tracker in (py, native):
            tracker.update(arr); tracker.update(arr)
            tracker.n_init_base = 2                                    # confirm_time.apply stores this
            feat = np.arange(8, dtype=np.float32)
            self.assertIsNone(tracker.attach_embedding(1, 2, feat, 'new'))
            self.assertIsNone(tracker.attach_embedding(99, 2, feat, 'new'))
            self.assertEqual(tracker.find(1).track_id, 1)
            self.assertIsNone(tracker.find(99))
        np.testing.assert_allclose(native.find(1).feat, py.find(1).feat, rtol=0, atol=1e-6)
        self.assertEqual(native.find(1).feat_frame, 2)
        self.assertEqual(native.update(np.empty((0, 6), np.float32)), [])
        with self.assertRaises(ValueError):
            native.update(np.zeros((2, 4), np.float32))
        with self.assertRaises(TypeError):
            native.tracked = [object()]


if __name__ == '__main__':
    unittest.main()
