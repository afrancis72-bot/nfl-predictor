"""Offline checks for post-slate market challenge grading."""
import unittest
from ai2_market_challenge import freeze_simulation
from ai2_market_grading import grade_market_challenge, summarize_market_challenge

class MarketGradingTests(unittest.TestCase):
    def setUp(self):
        self.frozen = freeze_simulation(
            "week7", [{"player_id": "p1", "market_key": "player_rush_yds",
                       "samples": [90, 100]}], "2026-10-22T18:00:00Z")
        self.props = [{"player_id": "p1", "market_key": "player_rush_yds",
                       "period": "FULL", "line": 65.5}]
    def test_model_more_accurate(self):
        rows = grade_market_challenge(self.frozen, self.props, [
            {"player_id": "p1", "market_key": "player_rush_yds", "actual": 92}])
        result = summarize_market_challenge(rows)
        self.assertEqual(result["observations"], 1)
        self.assertEqual(result["simulation_mae"], 3)
        self.assertEqual(result["market_line_mae"], 26.5)
        self.assertEqual(result["simulation_closer_count"], 1)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
    def test_missing_results_not_counted(self):
        self.assertEqual(summarize_market_challenge(
            grade_market_challenge(self.frozen, self.props, []))["observations"], 0)
    def test_duplicate_actuals_rejected(self):
        actual = {"player_id": "p1", "market_key": "player_rush_yds", "actual": 92}
        with self.assertRaises(ValueError):
            grade_market_challenge(self.frozen, self.props, [actual, actual])
if __name__ == "__main__":
    unittest.main()
