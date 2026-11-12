import unittest

from .replay_metrics import compute


class ReplayMetricsTests(unittest.TestCase):
    def test_perfect_single_frame(self):
        box = [11, 0, 0, 10, 10]
        det = [0.9, 0, 0, 10, 10]
        got = compute([([box], [box], [det])])
        self.assertAlmostEqual(got["hota"], 100.0)
        self.assertAlmostEqual(got["ap50"], 100.0)
        self.assertAlmostEqual(got["map"], 100.0)

    def test_high_score_duplicate_hurts_ap_and_detection_hota(self):
        gt = [[1, 0, 0, 10, 10]]
        tracks = [[1, 0, 0, 10, 10], [2, 0, 0, 10, 10]]
        # A high-score misplaced duplicate comes before the correct detection.
        dets = [[0.99, 20, 20, 30, 30], [0.98, 0, 0, 10, 10]]
        got = compute([(gt, tracks, dets)])
        self.assertLess(got["ap50"], 100.0)
        self.assertLess(got["hota"], 100.0)

    def test_missing_detections_make_ap_unavailable(self):
        box = [1, 0, 0, 10, 10]
        got = compute([([box], [box], None)])
        self.assertIsNotNone(got["hota"])
        self.assertIsNone(got["ap50"])
        self.assertIsNone(got["map"])

    def test_empty_predictions_are_valid_zero_scores(self):
        box = [1, 0, 0, 10, 10]
        got = compute([([box], [], [])])
        self.assertEqual(got["hota"], 0.0)
        self.assertEqual(got["ap50"], 0.0)
        self.assertEqual(got["map"], 0.0)

    def test_track_id_switch_reduces_hota_not_detection_ap(self):
        gt = [[7, 0, 0, 10, 10]]
        f1 = (gt, [[4, 0, 0, 10, 10]], [[0.9, 0, 0, 10, 10]])
        f2 = (gt, [[9, 0, 0, 10, 10]], [[0.9, 0, 0, 10, 10]])
        got = compute([f1, f2])
        self.assertLess(got["hota"], 100.0)
        self.assertAlmostEqual(got["ap50"], 100.0)

    def test_unmatched_id_occurrences_count_in_association_union(self):
        gt = [[7, 0, 0, 10, 10]]
        matched = (gt, [[4, 0, 0, 10, 10]], [[0.9, 0, 0, 10, 10]])
        missed = (gt, [], [])
        got = compute([matched, missed])
        # DetA=.5 and AssA=.5, hence HOTA=.5 for this one-ID example.
        self.assertAlmostEqual(got["hota"], 50.0)


if __name__ == "__main__":
    unittest.main()
