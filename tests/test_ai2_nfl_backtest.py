"""Historical NFL research metrics: leakage, lineup, and portfolio checks."""
import unittest
from ai2_nfl_backtest import (score_projection_history,
                              evaluate_lineup_history, evaluate_portfolio_profit)

class TestBacktest(unittest.TestCase):
    def record(self, **updates):
        result = dict(slate_id="2026-W01", player_id="101",
                      kickoff_at="2026-09-13T17:00:00Z",
                      prediction_asof="2026-09-13T16:30:00Z",
                      projected_dk_points=20, actual_dk_points=15)
        result.update(updates)
        return result

    def test_calibration_metrics(self):
        metrics = score_projection_history([self.record(),
            self.record(player_id="102", projected_dk_points=10, actual_dk_points=15)])
        self.assertEqual(metrics.observations, 2)
        self.assertEqual(metrics.mae, 5)
        self.assertEqual(metrics.rmse, 5)
        self.assertEqual(metrics.bias, 0)

    def test_post_lock_prediction_rejected(self):
        with self.assertRaises(ValueError):
            score_projection_history([self.record(prediction_asof="2026-09-13T17:00:00Z")])

    def test_duplicate_player_slate_rejected(self):
        with self.assertRaises(ValueError):
            score_projection_history([self.record(), self.record()])

    def test_timezone_required(self):
        with self.assertRaises(ValueError):
            score_projection_history([self.record(prediction_asof="2026-09-13T16:00:00")])

    def test_missing_actual_rejected(self):
        positions = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "DST", "WR")
        lineup = [{"ID": str(i), "Position": p, "TeamAbbrev": "T"+str(i),
                   "Salary": 5500, "opponent": "OTHER"} for i,p in enumerate(positions)]
        with self.assertRaises(ValueError):
            evaluate_lineup_history(lineup, {})

    def test_portfolio_net_roi(self):
        result = evaluate_portfolio_profit([120, 130, 150], [0, 2, 6], 1)
        self.assertEqual(result["net_profit"], 5)
        self.assertAlmostEqual(result["roi"], 5/3)
        self.assertAlmostEqual(result["cash_rate"], 2/3)

    def test_no_fake_payout(self):
        with self.assertRaises(ValueError):
            evaluate_portfolio_profit([100], [-1], 1)

if __name__ == "__main__":
    unittest.main()
