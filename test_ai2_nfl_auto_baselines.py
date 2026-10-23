"""Regression tests for the automated NFL historical projection workflow."""
import unittest
import pandas as pd
from ai2_nfl_auto_baselines import build_historical_baselines

class HistoricalBaselineTests(unittest.TestCase):
    def test_real_stats_are_not_mislabeled_verified(self):
        weekly = pd.DataFrame([{
            "season": 2026, "week": 6, "season_type": "REG",
            "player_name": "Sample Quarterback", "player_display_name": "Sample Quarterback",
            "player_id": "test-1", "recent_team": "NYJ", "position": "QB",
            "attempts": 32, "completions": 20, "passing_yards": 250,
            "passing_tds": 2, "passing_interceptions": 1,
            "carries": 4, "rushing_yards": 22, "rushing_tds": 0,
            "targets": 0, "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
        }])
        salaries = [{"ID": "42", "Name": "Sample Quarterback", "TeamAbbrev": "NYJ",
                     "Position": "QB", "Salary": "6000", "Status": "",
                     "Game Info": "NYJ@MIA 10/25/2026"}]
        result = build_historical_baselines(salaries, weekly, 2026, 7)
        row = result.iloc[0]
        self.assertEqual(row["match_status"], "matched")
        self.assertAlmostEqual(row["historical_partial_dk"], 19.2)
        self.assertAlmostEqual(row["pass_attempts_pg"], 32)
        self.assertFalse(bool(row["weekly_projection_valid"]))
        self.assertFalse(bool(row["opportunity_verified"]))
        self.assertIn("sparse", row["readiness_reason"])

    def test_alias_and_matchup_consistency(self):
        from ai2_nfl_auto_baselines import _team
        self.assertEqual(_team("JAC"), "JAX")
        self.assertEqual(_team("WSH"), "WAS")
        self.assertEqual(_team("LA"), "LAR")
        from ai2_nfl_matchups import opponent_from_game_info
        self.assertEqual(opponent_from_game_info("JAC", "JAX@MIA 1:00PM ET"), "MIA")
        self.assertIsNone(opponent_from_game_info("NYJ", "JAX@MIA 1:00PM ET"))

    def test_out_player_is_blocked_even_when_history_matches(self):
        weekly = pd.DataFrame([{
            "season": 2026, "week": 6, "season_type": "REG",
            "player_name": "Sample Quarterback", "recent_team": "NYJ", "position": "QB",
            "attempts": 32, "passing_yards": 250, "passing_tds": 2,
            "passing_interceptions": 1, "carries": 4, "rushing_yards": 22,
            "rushing_tds": 0,
        }])
        salaries = [{"ID": "42", "Name": "Sample Quarterback", "TeamAbbrev": "NYJ",
                     "Position": "QB", "Salary": "6000", "Status": "OUT"}]
        row = build_historical_baselines(salaries, weekly, 2026, 7).iloc[0]
        self.assertIn("blocked", row["readiness_reason"])

if __name__ == "__main__":
    unittest.main()
