"""Precommitted Bentley evidence-weighted baseline tests; no software release."""
import unittest

import numpy as np
import pandas as pd

from ai2_nfl_evidence_baseline import evidence_blend,MODES
from scripts.backtest_nfl_evidence_baseline import grade,stop_rule


def fixture():
    roster=[
        {"season":2025,"week":9,"game":"AAA@BBB",
         "player":"Test WR1","team":"AAA","position":"WR"},
        {"season":2025,"week":9,"game":"AAA@BBB",
         "player":"Test RB1","team":"BBB","position":"RB"},
    ]
    forecast=pd.DataFrame([
        {**roster[0],"actual_partial_dk":30.,
         "history_mean_partial_dk":20., "simulated_mean_dk":25.,
         "simulated_p10_dk":10.,"simulated_p90_dk":42.},
        {**roster[1],"actual_partial_dk":5.,
         "history_mean_partial_dk":10., "simulated_mean_dk":8.,
         "simulated_p10_dk":1.,"simulated_p90_dk":17.},
    ])
    evidence=pd.DataFrame([
        {**roster[0],"prior_player_games":8,
         "last_player_season":2025,"last_player_week":8,
         "has_four_plus_snap_games":True,"snap_games":7,
         "routes_verified":False,"game_day_active_verified":False},
        {**roster[1],"prior_player_games":2,
         "last_player_season":2025,"last_player_week":6,
         "has_four_plus_snap_games":False,"snap_games":0,
         "routes_verified":False,"game_day_active_verified":False},
    ])
    return forecast,evidence


class EvidenceBaselineTests(unittest.TestCase):
    def setUp(self):
        self.forecast,self.evidence=fixture()

    def test_dated_reliable_player_gets_more_historical_trust(self):
        f=evidence_blend(self.forecast,self.evidence,"evidence")
        wr=f.loc[f.position.eq("WR")].iloc[0]
        rb=f.loc[f.position.eq("RB")].iloc[0]
        self.assertGreater(wr.historical_weight,rb.historical_weight)
        self.assertAlmostEqual(wr.blended_mean_partial_dk,
                               wr.historical_weight*20+
                               (1-wr.historical_weight)*25)
        self.assertFalse(wr.current_week_active_verified)
        self.assertTrue(wr.p90_not_recalibrated)

    def test_mode_validation_and_equal_weight(self):
        for mode in MODES:
            x=evidence_blend(self.forecast,self.evidence,mode)
            self.assertEqual(len(x),2)
            self.assertTrue(x.blended_mean_partial_dk.notna().all())
        f=evidence_blend(self.forecast,self.evidence,"equal")
        self.assertAlmostEqual(f.iloc[0].blended_mean_partial_dk,22.5)
        with self.assertRaisesRegex(ValueError,"Unknown precommitted"):
            evidence_blend(self.forecast,self.evidence,"overfit_future")

    def test_changing_actuals_cannot_change_pregame_blended_prediction(self):
        a=evidence_blend(self.forecast,self.evidence,"guardrail")
        fake=self.forecast.copy()
        fake["actual_partial_dk"]=[999.,-1000.]
        b=evidence_blend(fake,self.evidence,"guardrail")
        np.testing.assert_array_equal(
            a.blended_mean_partial_dk,b.blended_mean_partial_dk)
        np.testing.assert_array_equal(a.historical_weight,b.historical_weight)

    def test_no_future_player_stats_and_no_hindsight_roster(self):
        invalid=self.evidence.copy()
        invalid.loc[0,"last_player_week"]=9
        with self.assertRaisesRegex(ValueError,"Future player-stat"):
            evidence_blend(self.forecast,invalid,"evidence")
        missing=self.evidence.iloc[[0]]
        with self.assertRaisesRegex(ValueError,"cohort"):
            evidence_blend(self.forecast,missing,"equal")
        extra=self.evidence.iloc[[0]].copy()
        dup=pd.concat([self.evidence,extra],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            evidence_blend(self.forecast,dup,"equal")

    def test_missing_official_active_is_never_assumed_verified(self):
        corrupt=self.evidence.copy()
        corrupt.loc[0,"game_day_active_verified"]=True
        with self.assertRaisesRegex(ValueError,"cannot treat"):
            evidence_blend(self.forecast,corrupt,"guardrail")
        corrupt=self.evidence.copy()
        corrupt.loc[0,"has_four_plus_snap_games"]=True
        corrupt.loc[0,"snap_games"]=1
        with self.assertRaisesRegex(ValueError,"Insufficient snap"):
            evidence_blend(self.forecast,corrupt,"evidence")

    def test_no_estimated_p90_created_from_mean_blend(self):
        f=evidence_blend(self.forecast,self.evidence,"guardrail")
        self.assertFalse(any(c.startswith("blended_p90") for c in f.columns))
        np.testing.assert_allclose(
            f["simulated_p90_dk"],self.forecast["simulated_p90_dk"])

    def test_paired_grade_and_hard_stop(self):
        f=evidence_blend(self.forecast,self.evidence,"equal")
        result=grade(f,"blended_mean_partial_dk","simulated_mean_dk")
        self.assertEqual(result["n"],2)
        self.assertFalse(stop_rule(result))
        self.assertFalse(result["candidate_p90_calibrated"])
        self.assertIn("WR",result["by_position"])


if __name__=="__main__":
    unittest.main()
