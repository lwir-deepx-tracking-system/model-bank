"""Ambiguous relink candidates must not steal a public identity."""
import unittest
from .relink_online_judge import mutual_best_pairs


class MutualRelinkTests(unittest.TestCase):
    def test_only_mutual_unique_best_pairs_survive(self):
        pairs = [(.95, 1, 7), (.94, 2, 7), (.9, 2, 8), (.8, 1, 8)]
        self.assertEqual(mutual_best_pairs(pairs), [pairs[0]])

    def test_tie_on_either_side_is_withheld(self):
        for pairs in [[(.9, 1, 7), (.9, 1, 8)], [(.9, 1, 7), (.9, 2, 7)]]:
            self.assertEqual(mutual_best_pairs(pairs), [])

    def test_unambiguous_independent_pairs_are_preserved(self):
        pairs = [(.95, 1, 7), (.91, 2, 8)]
        self.assertEqual(mutual_best_pairs(pairs), pairs)
