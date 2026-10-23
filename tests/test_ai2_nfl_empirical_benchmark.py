"""Empirical fantasy distribution benchmark checks."""
import unittest
import pandas as pd
from ai2_nfl_empirical_benchmark import walkforward_empirical_distribution


class EmpiricalBenchmarkTests(unittest.TestCase):
    def test_future_breakout_not_in_prior_distribution(self):
        rows = []
        for week, yards in [(1, 20), (2, 30), (3, 40), (4, 250)]:
            rows.append(dict(season=2026, week=week, season_type="REG",
                             player_name="Example WR", player_id="p1",
                             recent_team="AAA", position="WR",
                             rushing_yards=0, rushing_tds=0,
                             receiving_yards=yards, receptions=5,
                             receiving_tds=0))
        result = walkforward_empirical_distribution(pd.DataFrame(rows), 2026, 4, 4)
        self.assertEqual(len(result), 1)
        self.assertLess(result.iloc[0].baseline_mean, 10)
        self.assertGreater(result.iloc[0].actual_partial_dk, 30)
        self.assertFalse(bool(result.iloc[0].covered_80))

    def test_insufficient_history_skipped(self):
        rows = [dict(season=2026, week=w, season_type="REG",
                     player_name="New WR", recent_team="AAA", position="WR",
                     rushing_yards=0, rushing_tds=0, receiving_yards=50,
                     receptions=4, receiving_tds=0) for w in [3, 4]]
        self.assertTrue(walkforward_empirical_distribution(pd.DataFrame(rows), 2026, 4, 4).empty)


if __name__ == "__main__":
    unittest.main()
