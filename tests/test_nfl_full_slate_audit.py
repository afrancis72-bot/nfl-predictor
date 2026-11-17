"""No-network acceptance checks for historical NFL forecast reporting."""
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.audit_nfl_full_slates import (
    build_report, game_cluster_lift_ci, metric_summary, provisional_gates,
)


def sample():
    return pd.DataFrame([
        {"season": 2026, "week": 4, "game": "BUF@MIA", "player": "Player One",
         "team": "BUF", "position": "QB", "opponent": "MIA",
         "actual_partial_dk": 10.0, "history_mean_partial_dk": 14.0,
         "simulated_mean_dk": 11.0, "simulated_p10_dk": 5.0,
         "simulated_p90_dk": 20.0},
        {"season": 2026, "week": 4, "game": "JAX@TEN", "player": "Player Two",
         "team": "JAX", "position": "TE", "opponent": "TEN",
         "actual_partial_dk": 20.0, "history_mean_partial_dk": 12.0,
         "simulated_mean_dk": 18.0, "simulated_p10_dk": 12.0,
         "simulated_p90_dk": 24.0},
    ])


class FullSlateAuditTests(unittest.TestCase):
    def test_exact_prediction_accuracy_and_coverage(self):
        result = metric_summary(sample())
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["simulator_mae"], 1.5)
        self.assertEqual(result["historical_baseline_mae"], 6)
        self.assertEqual(result["simulator_improvement_pct"], 75)
        self.assertEqual(result["within_3_dk_pct"], 100)
        self.assertEqual(result["actual_below_p90_pct"], 100)
        self.assertEqual(result["actual_inside_p10_p90_pct"], 100)

    def test_bootstrap_resamples_games_not_player_rows(self):
        data = sample()
        a = game_cluster_lift_ci(data, resamples=200)
        b = game_cluster_lift_ci(data, resamples=200)
        self.assertEqual(a, b)
        self.assertEqual(a["games"], 2)
        self.assertGreater(a["lift_pct_95_ci"][0], 0)

    def test_missing_quantiles_cannot_be_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Missing benchmark columns"):
            metric_summary(sample().drop(columns="simulated_p10_dk"))

    def test_invalid_p10_p90_never_passes(self):
        data = sample()
        data.loc[0, "simulated_p10_dk"] = 99
        with self.assertRaisesRegex(ValueError, "P10 is above P90"):
            metric_summary(data)

    def test_research_gate_never_approves_paid_production(self):
        gates = provisional_gates(sample(), skipped_games=1)
        self.assertFalse(gates["research_diagnostic_pass"])
        self.assertFalse(gates["production_approved"])
        self.assertFalse(gates["checks"]["at_least_400_matched_player_games"])

    @patch("scripts.audit_nfl_full_slates.evaluate")
    def test_full_slate_evaluation_never_caps_games(self, mock_evaluate):
        observations = sample()
        mock_evaluate.side_effect = [(observations.assign(season=2025, week=9), []),
                                     (observations.copy(), [])]
        schedule = pd.DataFrame([
            {"season": 2025, "week": 9, "game_type": "REG",
             "home_score": 7, "away_score": 14, "weekday": "Sunday"},
            {"season": 2025, "week": 9, "game_type": "REG",
             "home_score": 14, "away_score": 7, "weekday": "Sunday"},
            {"season": 2026, "week": 4, "game_type": "REG",
             "home_score": 7, "away_score": 14, "weekday": "Sunday"},
            {"season": 2026, "week": 4, "game_type": "REG",
             "home_score": 14, "away_score": 7, "weekday": "Sunday"},
        ])
        report, rows = build_report(pd.DataFrame(), schedule,
                                    pd.DataFrame(), n_sims=400)
        self.assertEqual(len(rows), 4)
        self.assertEqual(mock_evaluate.call_count, 2)
        for call in mock_evaluate.call_args_list:
            self.assertIsNone(call.kwargs["games_per_week"])
            self.assertEqual(call.kwargs["defense_weight"], 0.0)
            self.assertEqual(call.kwargs["market_weight"], 0.0)
        self.assertFalse(report["diagnostic_gates"]["production_approved"])
        self.assertEqual(report["by_opponent_2026"]["MIA"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
