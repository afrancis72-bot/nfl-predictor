"""Tests for the partial offensive DK scoring audit."""
import unittest
import math
import pandas as pd
from props_data import dk_classic_points, walkforward_dk_offense


class PartialDKScoringTests(unittest.TestCase):
    def test_bonus_and_ppr(self):
        box = dict(passing_yards=300, passing_tds=2, interceptions=1,
                   rushing_yards=100, rushing_tds=1, receiving_yards=100,
                   receptions=5, receiving_tds=1)
        # QB applicable stats: 12 + 8 - 1 + 10 + 6 + 3 + 3 bonus
        self.assertAlmostEqual(dk_classic_points(box, "QB"), 41.0)

    def test_missing_stats_not_assumed_zero(self):
        self.assertTrue(math.isnan(dk_classic_points(dict(passing_yards=300), "QB")))

    def test_receiver_without_passing_columns_is_valid(self):
        box = dict(rushing_yards=0, rushing_tds=0,
                   receiving_yards=100, receptions=5, receiving_tds=1)
        self.assertAlmostEqual(dk_classic_points(box, "WR"), 24.0)

    def test_walkforward_never_uses_target_week_for_prediction(self):
        def row(week, yards):
            return dict(season=2026, week=week, season_type="REG",
                        player_name="Test Receiver", recent_team="AAA",
                        position="WR", passing_yards=0, passing_tds=0,
                        interceptions=0, rushing_yards=0, rushing_tds=0,
                        receiving_yards=yards, receptions=5, receiving_tds=0)
        raw = pd.DataFrame([row(3, 40), row(4, 60), row(5, 200)])
        result = walkforward_dk_offense(raw, 2026, first_week=5, last_week=5)
        self.assertEqual(len(result), 1)
        self.assertLess(result.iloc[0]["predicted_partial_dk"], 15)
        self.assertAlmostEqual(result.iloc[0]["actual_partial_dk"], 28.0)


if __name__ == "__main__":
    unittest.main()
