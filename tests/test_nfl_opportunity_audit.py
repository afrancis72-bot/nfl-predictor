"""No-network NFL opportunity audit: exact stats, uncertainty and leakage."""
import unittest

import numpy as np
import pandas as pd

from ai2_nfl_opportunity_audit import (
    STATS, extract_player_stats, opportunity_report, stat_summary,
)


def example(**kw):
    args=dict(
        player="Test QB",team="AAA",position="QB",game="AAA@BBB",
        season=2026,week=9,
        observed=dict(attempts=30,completions=19,passing_yards=220,
                      passing_tds=2,interceptions=0,carries=5,
                      rushing_yards=40,rushing_tds=1),
        prior=dict(latest_observed_season=2026,latest_observed_week=8,
                   pass_attempts_pg=31,completions_pg=18,pass_yards_pg=210,
                   passing_tds_pg=1.1,interceptions_pg=.5,
                   carries_pg=4,rushing_yards_pg=25,rushing_tds_pg=.2),
        simulated_stats=dict(pass_attempts=np.array([28,30,32,34]),
                             completions=np.array([17,19,21,23]),
                             passing_yards=np.array([190,210,230,250]),
                             passing_tds=np.array([1,2,3,2]),
                             interceptions=np.array([0,0,1,1]),
                             carries=np.array([2,3,4,5]),
                             rushing_yards=np.array([10,20,30,40]),
                             rushing_tds=np.array([0,0,0,1])),
        n_sims=4,
    )
    args.update(kw)
    return args


class OpportunityAuditTests(unittest.TestCase):
    def test_stat_by_stat_simulated_and_prior_are_not_confused(self):
        rows=extract_player_stats(**example())
        self.assertEqual(len(rows),len(STATS["QB"]))
        attempts=next(x for x in rows if x["stat"]=="pass_attempts")
        self.assertEqual(attempts["actual"],30.)
        self.assertEqual(attempts["predicted_mean"],31.)
        self.assertEqual(attempts["prior_baseline"],31.)
        self.assertEqual(attempts["predicted_p10"],28.6)
        self.assertEqual(attempts["predicted_p90"],33.4)
        tds=next(x for x in rows if x["stat"]=="rushing_tds")
        self.assertEqual(tds["predicted_mean"],.25)
        self.assertAlmostEqual(tds["absolute_error"],.75)

    def test_unavailable_actual_is_not_fabricated_as_zero(self):
        args=example()
        args["observed"].pop("interceptions")
        args["observed"]["rushing_yards"]=np.nan
        rows=extract_player_stats(**args)
        self.assertNotIn("interceptions",{x["stat"] for x in rows})
        self.assertNotIn("rushing_yards",{x["stat"] for x in rows})
        self.assertEqual(len(rows),6)

    def test_future_or_target_week_benchmark_is_rejected(self):
        for season,week in ((2026,9),(2027,1)):
            with self.subTest(season=season,week=week):
                args=example()
                args["prior"]["latest_observed_season"]=season
                args["prior"]["latest_observed_week"]=week
                with self.assertRaisesRegex(ValueError,"target game"):
                    extract_player_stats(**args)

    def test_invalid_draws_cannot_silently_pass(self):
        args=example()
        args["simulated_stats"]["pass_attempts"]=np.array([33,35,39])
        with self.assertRaisesRegex(ValueError,"Invalid pass_attempts"):
            extract_player_stats(**args)
        args=example()
        args["simulated_stats"]["carries"]=np.array([-1,3,4,5])
        with self.assertRaisesRegex(ValueError,"Negative simulated"):
            extract_player_stats(**args)

    def test_metrics_use_same_player_game_for_baseline_and_simulation(self):
        rows=extract_player_stats(**example())
        attempts=pd.DataFrame(x for x in rows if x["stat"]=="pass_attempts")
        m=stat_summary(attempts)
        self.assertEqual(m["n"],1)
        self.assertEqual(m["sim_mae"],1)
        self.assertEqual(m["history_mae"],1)
        self.assertEqual(m["sim_mae_improvement_pct"],0)
        self.assertEqual(m["below_p90_pct"],100)

    def test_statistics_and_prioritized_dk_are_explicit_diagnostics_only(self):
        rows=extract_player_stats(**example())
        results=opportunity_report(rows)
        self.assertEqual(results["distinct_player_games"],1)
        self.assertEqual(results["stat_observations"],8)
        self.assertFalse(results["production_approved"])
        self.assertTrue(results["research_only"])
        m=results["summary_by_position_and_stat"]["QB.passing_tds"]
        self.assertEqual(m["direct_DK_points_per_unit"],4)
        self.assertIn("not additive causal",str(results["limitations"]).lower())

    def test_output_with_missing_actuals_has_no_fake_td_observations(self):
        args=example()
        args["observed"]["passing_tds"]=np.nan
        args["observed"]["rushing_tds"]=np.nan
        rows=extract_player_stats(**args)
        self.assertFalse(any(x["category"]=="touchdown" for x in rows))


if __name__=="__main__":
    unittest.main()
