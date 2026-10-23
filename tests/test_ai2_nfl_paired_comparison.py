"""Paired historical simulation comparison tests."""
import unittest
import pandas as pd
from ai2_nfl_paired_comparison import compare_matched_predictions


class PairedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.sim = pd.DataFrame({
            "slate_id": ["week6", "week6"], "ID": ["1", "2"],
            "actual_dk": [10., 20.], "sim_mean": [11., 18.],
            "covered_50": [True, False], "covered_80": [True, True]})
        self.emp = pd.DataFrame({
            "slate_id": ["week6", "week6"], "ID": ["2", "1"],
            "actual_dk": [20., 10.], "baseline_mean": [15., 16.],
            "covered_50": [False, True], "covered_80": [True, False]})

    def test_comparison_aligns_by_player_not_order(self):
        paired, summary = compare_matched_predictions(self.sim, self.emp)
        self.assertEqual(len(paired), 2)
        self.assertAlmostEqual(summary.iloc[0].mean_absolute_error, 1.5)
        self.assertAlmostEqual(summary.iloc[1].mean_absolute_error, 5.5)
        self.assertTrue((paired.sim_improvement > 0).all())

    def test_different_player_coverage_rejected(self):
        with self.assertRaises(ValueError):
            compare_matched_predictions(self.sim, self.emp.iloc[:1])

    def test_different_truth_rejected(self):
        self.emp.loc[0, "actual_dk"] = 99.
        with self.assertRaises(ValueError):
            compare_matched_predictions(self.sim, self.emp)

    def test_duplicate_player_rejected(self):
        with self.assertRaises(ValueError):
            compare_matched_predictions(pd.concat([self.sim, self.sim]), self.emp)


if __name__ == "__main__":
    unittest.main()
