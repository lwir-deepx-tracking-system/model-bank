"""Check live KeepID history compaction against the full replay history."""

import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from ..gimbal.live_tracks import _appearance
from . import relink_online_judge as judge


class LiveObservationHistoryTests(unittest.TestCase):
    def test_long_gaps_in_appearance_keep_old_clean_gallery_and_original_age(self):
        full = []
        live = judge._LiveObservationHistory()
        for t in range(240):
            feature = np.array([1.0, t/250.0]) if t < 8 or t % 37 == 0 else None
            box = [10.0+t/10.0, 20.0, 50.0+t/10.0, 110.0]
            row = [*box, 8, .45+t/1000.0, 0, 0]
            people = {0: box}
            if t >= 8:
                people[1] = [box[0]+2, *box[1:]]
            observation = judge.make_obs(t, row, people, [row], feature)
            observation['fp'] = feature
            full.append(observation)
            live.append(observation)

        self.assertEqual(len(live), 12)
        self.assertEqual(live.total, len(full))
        self.assertEqual(live.first_t, 0)
        self.assertEqual([o['t'] for o in live], list(range(228, 240)))
        for k in ('m1', 'fp'):
            for clean in (False, True):
                expected = judge.gallery(full, k, clean)
                actual = judge.gallery(live, k, clean)
                self.assertEqual([o['t'] for o in expected], [o['t'] for o in actual])
                for a, b in zip(expected, actual):
                    self.assertIs(a, b)
        self.assertEqual([o['t'] for o in judge.gallery(live, 'm1', True)], list(range(3, 8)))
        box = [33., 21., 73., 109.]
        new = judge.make_obs(244, [*box, 9, .7, 0, 0], {0: box}, [], np.array([1., .1]))
        expected = judge.pair_features([new], full, 244, 244, [], False)
        actual = judge.pair_features([new], live, 244, 244, [], False)
        self.assertEqual(expected.keys(), actual.keys())
        np.testing.assert_array_equal(list(expected.values()), list(actual.values()))

    def test_async_birth_relink_retirement_and_follow_match_full_history(self):
        features = ['cand_len', 'cand_age', 'cand_born_first_frame', 'm1_cand_n5', 'm1_cand_nclean',
                    'm1_cand_feat_age', 'm1_cos_clean5', 'conf_last5', 'cv_vx_h', 'fp_cand_n5']
        spec = {'model': {}, 'features': features, 'decision': {'threshold': .5}}
        with patch.object(judge, 'load_spec', return_value=(spec, 'synthetic')):
            full = judge.OnlineJudge('fixture', crops=2, compact_ids=True)
            live = judge.OnlineJudge('fixture', crops=2, compact_ids=True)
        full._history_type = list
        sessions = (full, live)
        crop_logs = [[], []]
        for relinker in sessions:
            relinker.keep_rows = True

        def score(_, X):
            return np.where(X[:, 0] > 24, .9, .2)

        with patch.object(judge, 'predict_json', side_effect=score):
            for t in range(430):
                ids = [1, 2] if t < 401 else ([3, 2] if t < 407 else [1, 3, 2])
                rows, boxes = [], {}
                for k, raw_id in enumerate(ids):
                    x = {1:55.0, 2:260.0, 3:57.0}[raw_id] + (t % 10)/5.0
                    box = [x, 20., x+45., 120.]
                    rows.append([*box, raw_id, .8, 0, k])
                    boxes[k] = box
                if 4 < t < 400:
                    boxes[17] = [56.0, 20., 99.0, 120.]
                results = []
                appearances = []
                for i, relinker in enumerate(sessions):
                    def crop(ks, i=i, t=t):
                        crop_logs[i].append((t, 'sync', tuple(ks)))
                        return {k:np.array([1., .01*(k+t)]) for k in ks if t < 10 or t % 13 != 0}
                    def defer(ks, i=i, t=t):
                        crop_logs[i].append((t, 'submit', tuple(ks)))
                        def collect():
                            crop_logs[i].append((t, 'collect', tuple(ks)))
                            return {k:np.array([1., .01*(k+t)]) for k in ks if t < 25 or t % 29 == 0}
                        return collect
                    results.append(relinker.step(rows, boxes, crop, frame=str(t), embed_async=defer))
                    with patch.dict(vars(APP), relabeler=relinker):
                        appearances.append([_appearance(row[4]) for row in results[-1]])
                self.assertEqual(results[0], results[1])
                for a, b in zip(*appearances):
                    if a is None:
                        self.assertIsNone(b)
                    else:
                        np.testing.assert_array_equal(a[0], b[0])
                        self.assertEqual(a[1], b[1])

        self.assertEqual(crop_logs[0], crop_logs[1])
        self.assertEqual(full.stats, live.stats)
        self.assertEqual(full.events, live.events)
        self.assertTrue(full.events)
        self.assertGreater(full.stats['retired_reappearances'], 0)
        self.assertEqual(full.id_map, live.id_map)
        self.assertEqual(full.retired, live.retired)
        self.assertEqual(len(full.rows_log), len(live.rows_log))
        for old_row, new_row in zip(full.rows_log, live.rows_log):
            self.assertEqual(old_row[:4], new_row[:4])
            np.testing.assert_array_equal(old_row[4], new_row[4])
            np.testing.assert_array_equal(old_row[5], new_row[5])
        self.assertGreaterEqual(len(full.hist[1]), 400)
        self.assertEqual(len(live.hist[1]), 12)
        for relinker in sessions:
            relinker.reset()
            self.assertFalse(relinker.hist)
            self.assertIsNone(relinker.deferred)
