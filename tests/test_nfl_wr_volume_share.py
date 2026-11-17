"""Unit checks ensure exact arithmetic and no future results enter simulated volume."""
import unittest
import pandas as pd
import numpy as np
from ai2_nfl_pass_volume_share_audit import team_pass_attempt_row,decompose_receiver_errors,summary

class VolumeShareAuditTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(season=2026,week=9,game="AAA@BBB",team="AAA",
            player="First WR",role="WR1",opponent="BBB",
            actual_targets=10.,predicted_targets=6.,prior_targets_pg=9.)
        self.pass_row=dict(season=2026,week=9,game="AAA@BBB",team="AAA",
            actual_pass_attempts=40.,model_pass_attempts=30.,
            historical_team_pass_baseline=32.,passing_qb="A QB")

    def test_exact_error_identity_and_oracle(self):
        f=decompose_receiver_errors([self.row],[self.pass_row])
        self.assertAlmostEqual(float(f.iloc[0]["volume_error_component"]),-2.)
        self.assertAlmostEqual(float(f.iloc[0]["allocation_error_component"]),-2.)
        self.assertAlmostEqual(float(f.iloc[0]["total_target_error"]),-4.)
        self.assertAlmostEqual(float(f.iloc[0]["oracle_actual_volume_prediction"]),8.)
        out=summary(f,pd.DataFrame([self.pass_row]))
        self.assertEqual(out["WR1"]["target_bias"],-4.)
        self.assertEqual(out["WR1"]["oracle_actual_volume_target_mae"],2.)

    def test_team_pass_only_simulated_from_pregame_engine(self):
        players=pd.DataFrame([dict(Name="A QB",TeamAbbrev="AAA",Position="QB")])
        draws={"A QB":{"pass_attempts":np.array([28,32,31])},
            "__TEAM_PASS_BASELINE__|AAA":{"team_pass_attempts_baseline":32.}}
        actual=pd.DataFrame([dict(recent_team="AAA",position="QB",attempts=32),
                             dict(recent_team="AAA",position="WR",attempts=2)])
        a=team_pass_attempt_row(season=2026,week=9,game="AAA@BBB",team="AAA",
               players=players,stat_draws=draws,actual_week=actual,n_sims=3)
        self.assertEqual(a["actual_pass_attempts"],34)
        self.assertAlmostEqual(a["model_pass_attempts"],91/3)
        modified=actual.copy();modified.at[0,"attempts"]=40
        b=team_pass_attempt_row(season=2026,week=9,game="AAA@BBB",team="AAA",
               players=players,stat_draws=draws,actual_week=modified,n_sims=3)
        self.assertEqual(b["model_pass_attempts"],a["model_pass_attempts"])
        self.assertNotEqual(b["actual_pass_attempts"],a["actual_pass_attempts"])

    def test_missing_team_denied(self):
        with self.assertRaisesRegex(ValueError,"Missing team pass"):
            decompose_receiver_errors([self.row],[dict(self.pass_row,team="BBB")])

    def test_invalid_or_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            decompose_receiver_errors([self.row,self.row],[self.pass_row])
        with self.assertRaisesRegex(ValueError,"exceed"):
            decompose_receiver_errors([dict(self.row,actual_targets=41)],[self.pass_row])
        with self.assertRaisesRegex(ValueError,"share exceeds"):
            decompose_receiver_errors([dict(self.row,predicted_targets=33)],[self.pass_row])
if __name__=="__main__":
    unittest.main()
