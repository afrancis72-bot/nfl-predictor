"""Unit tests for simulation distribution diagnostics."""
import unittest
import numpy as np
import pandas as pd
from ai2_nfl_distribution_audit import distribution_diagnostics, distribution_summary


class DistributionAuditTests(unittest.TestCase):
    def setUp(self):
        self.players = pd.DataFrame({"ID": ["1", "2"], "Name": ["A", "B"], "Position": ["WR", "RB"]})
        self.draws = np.tile([[10., 20.], [20., 30.]], (100, 1))
        self.actuals = pd.DataFrame({"ID": ["2", "1"], "actual_dk": [25., 15.]})

    def test_aligns_by_id_not_row_order(self):
        result = distribution_diagnostics(self.players, self.draws, self.actuals)
        self.assertEqual(result.actual_dk.tolist(), [15., 25.])
        self.assertAlmostEqual(distribution_summary(result)["mean_absolute_error"], 0)
        self.assertTrue(result.covered_80.all())

    def test_missing_actual_fails_closed(self):
        with self.assertRaises(ValueError):
            distribution_diagnostics(self.players, self.draws, self.actuals.iloc[:1])

    def test_duplicate_ids_fail_closed(self):
        with self.assertRaises(ValueError):
            distribution_diagnostics(self.players, self.draws, pd.concat([self.actuals, self.actuals]))

    def test_mismatched_game_fails_closed(self):
        players = self.players.copy()
        players["game"] = ["AAA@BBB", "CCC@DDD"]
        actuals = self.actuals.copy()
        actuals["game"] = ["CCC@DDD", "XXX@YYY"]
        with self.assertRaises(ValueError):
            distribution_diagnostics(players, self.draws, actuals)

    def test_nonfinite_draws_fail_closed(self):
        bad = self.draws.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            distribution_diagnostics(self.players, bad, self.actuals)


if __name__ == "__main__":
    unittest.main()
