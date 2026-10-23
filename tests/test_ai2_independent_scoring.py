"""Regression checks: betting-market fields must not influence NFL scoring priors."""
import inspect
import unittest

from dfs_stats_engine import _team_scoring_budget, simulate_stat_driven_dfs


class IndependentScoringBudgetTests(unittest.TestCase):
    def test_historical_rates_determine_touchdowns(self):
        td, fg = _team_scoring_budget(1.6, 0.8)
        self.assertAlmostEqual(td, 2.4)
        self.assertAlmostEqual(fg, 1.65)

    def test_higher_historical_tds_increase_budget(self):
        low, _ = _team_scoring_budget(0.8, 0.3)
        high, _ = _team_scoring_budget(2.0, 1.0)
        self.assertGreater(high, low)

    def test_scoring_budget_rejects_nan_without_propagating_it(self):
        import math
        td, fg = _team_scoring_budget(float("nan"), float("nan"))
        self.assertTrue(math.isfinite(td))
        self.assertTrue(math.isfinite(fg))

    def test_scoring_budget_is_deterministic_and_bounded(self):
        first = _team_scoring_budget(1.6, 0.8)
        self.assertEqual(first, _team_scoring_budget(1.6, 0.8))
        td, fg = _team_scoring_budget(100, 100, expected_field_goals=100)
        self.assertEqual(td, 4.8)
        self.assertEqual(fg, 3.5)

    def test_scoring_budget_has_no_market_parameter(self):
        params = inspect.signature(_team_scoring_budget).parameters
        self.assertEqual(
            list(params),
            ["historical_pass_tds", "historical_rush_tds", "expected_field_goals"],
        )

    def test_simulator_source_does_not_read_market_implied_points(self):
        source = inspect.getsource(simulate_stat_driven_dfs)
        self.assertNotIn("team_implied_points", source)
        self.assertNotIn("implied_points", source)
        self.assertIn("historical_rates_only", source)


if __name__ == "__main__":
    unittest.main()
