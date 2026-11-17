"""Regression tests for neutral-vs-experimental opponent A/B score comparisons."""
import unittest

import pandas as pd

from scripts.backtest_nfl_matchup_simulator import match_results, paired_metrics


def frame():
    return pd.DataFrame([
        dict(season=2026,week=4,game="AAA@BBB",player="One",team="AAA",
             position="QB",actual_partial_dk=20.,
             history_mean_partial_dk=14.,simulated_mean_dk=12.,
             simulated_p10_dk=5.,simulated_p90_dk=28.),
        dict(season=2026,week=4,game="AAA@BBB",player="Two",team="BBB",
             position="WR",actual_partial_dk=10.,
             history_mean_partial_dk=12.,simulated_mean_dk=15.,
             simulated_p10_dk=1.,simulated_p90_dk=25.),
    ])


class PairedOpponentModelTests(unittest.TestCase):
    def test_pairing_rejects_missing_player(self):
        with self.assertRaisesRegex(ValueError,"different player pools"):
            match_results(frame(),frame().iloc[:1])

    def test_pairing_rejects_changed_actual_labels(self):
        candidate=frame()
        candidate.loc[0,"actual_partial_dk"]=100.
        with self.assertRaisesRegex(ValueError,"changed target labels"):
            match_results(frame(),candidate)

    def test_pairing_and_accuracy_measure_actual_model_improvement(self):
        candidate=frame()
        candidate.loc[0,"simulated_mean_dk"]=19.
        candidate.loc[1,"simulated_mean_dk"]=11.
        comparison=paired_metrics(match_results(frame(),candidate))
        self.assertEqual(comparison["baseline_simulator_mae"],6.5)
        self.assertEqual(comparison["matchup_simulator_mae"],1.)
        self.assertAlmostEqual(comparison["matchup_mae_improvement_pct"],84.615,places=3)
        self.assertEqual(comparison["actual_below_p90_pct"],100.)

    def test_no_change_reported_as_zero_lift(self):
        comparison=paired_metrics(match_results(frame(),frame()))
        self.assertEqual(comparison["matchup_mae_improvement_pct"],0.)


if __name__=="__main__":
    unittest.main()
