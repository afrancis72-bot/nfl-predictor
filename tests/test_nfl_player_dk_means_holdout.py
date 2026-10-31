"""Historical player fantasy-point benchmark never builds pool from target week."""
import unittest
import pandas as pd
from scripts.backtest_nfl_player_dk_means import pregame_game_pool,summary


class NflMeansHoldoutTests(unittest.TestCase):
    def test_does_not_include_target_week_or_old_qb(self):
        r=pd.DataFrame([
            dict(player="AAA QB",team="AAA",position="QB",
                 pass_attempts_pg=31,games_played=5,
                 latest_observed_season=2026,latest_observed_week=5),
            dict(player="AAA RB",team="AAA",position="RB",
                 carries_pg=14,targets_pg=3,games_played=4,
                 latest_observed_season=2026,latest_observed_week=6),
            dict(player="BBB QB",team="BBB",position="QB",
                 pass_attempts_pg=30,games_played=6,
                 latest_observed_season=2026,latest_observed_week=5),
            dict(player="LEAKED",team="AAA",position="WR",
                 targets_pg=15,games_played=6,
                 latest_observed_season=2026,latest_observed_week=6),
        ])
        pool=pregame_game_pool(r,"AAA","BBB",2026,6)
        self.assertIn("AAA QB",pool["Name"].tolist())
        self.assertNotIn("LEAKED",pool["Name"].tolist())
        self.assertNotIn("AAA RB",pool["Name"].tolist())

    def test_historical_comparison_metrics_do_not_require_winning_lineups(self):
        x=pd.DataFrame({
            "actual_partial_dk":[8.,22.,10.],
            "history_mean_partial_dk":[10.,12.,11.],
            "simulated_mean_dk":[9.,19.,12.],
            "simulated_p90_dk":[20.,30.,21.]})
        report=summary(x)
        self.assertEqual(report["n"],3)
        self.assertAlmostEqual(report["sim_mean_mae"],2.)
        self.assertEqual(report["simulated_p90_actual_coverage_pct"],100.)
        self.assertEqual(summary(pd.DataFrame()),{"n":0})


if __name__=="__main__":
    unittest.main()
