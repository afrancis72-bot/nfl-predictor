"""Pregame PFR snap evidence is real historical participation, not active status."""
import unittest

import numpy as np
import pandas as pd

from ai2_nfl_snap_roles import predict_snap_roles
from scripts.backtest_nfl_snap_blend import paired_role_grade, evaluate_weight


def fixtures():
    history=[];snaps=[]
    for week in range(1,8):
        for player,pos,targets,snap_pct in (
            ("Wide A","WR",11 if week>=4 else 6,85 if week>=4 else 65),
            ("Wide B","WR",6,70),
            ("Wide C","WR",4,45),
            ("Starting RB","RB",3,75),
        ):
            history.append({
                "season":2026,"week":week,"season_type":"REG",
                "player_name":player,"recent_team":"AAA","position":pos,
                "targets":targets,"carries":12 if pos=="RB" else 0,
            })
            snaps.append({
                "season":2026,"week":week,"player":player,
                "team":"AAA","position":pos,
                "offense_snaps":int(65*snap_pct/100),
                "offense_pct":snap_pct,
            })
    return pd.DataFrame(history),pd.DataFrame(snaps)


class ParticipationAuditTests(unittest.TestCase):
    def setUp(self):
        self.history,self.snaps=fixtures()

    def test_pregame_snap_strength_changes_player_target_estimate(self):
        e0=predict_snap_roles(self.history,self.snaps,2026,7,weight=0.)
        e7=predict_snap_roles(self.history,self.snaps,2026,7,weight=.7)
        key=("wide a","AAA","WR")
        self.assertEqual(e0[key]["snap_games"],6)
        self.assertEqual(e7[key]["snap_games"],6)
        self.assertEqual(e7[key]["last_observed_week"],6)
        self.assertNotEqual(e0[key]["blend_weight"],e7[key]["blend_weight"])
        self.assertTrue(e7[key]["routes_verified"] is False)
        self.assertGreater(e7[key]["expected_targets"],0.)

    def test_target_week_snap_statistics_cannot_change_forecast(self):
        before=predict_snap_roles(self.history,self.snaps,2026,7,weight=.35)
        history=self.history.copy()
        snaps=self.snaps.copy()
        future=history[history.week.eq(6)].copy()
        future["week"]=7
        future["targets"]=45
        future_snaps=snaps[snaps.week.eq(6)].copy()
        future_snaps["week"]=7
        future_snaps["offense_pct"]=100
        future_snaps["offense_snaps"]=85
        after=predict_snap_roles(
            pd.concat([history,future],ignore_index=True),
            pd.concat([snaps,future_snaps],ignore_index=True),
            2026,7,weight=.35)
        self.assertEqual(before,after)

    def test_low_evidence_disables_participation_change(self):
        short=self.snaps[self.snaps.week.ge(5)].copy()
        report=predict_snap_roles(self.history,short,2026,7,weight=.35)
        self.assertNotIn(("wide a","AAA","WR"),report)

    def test_same_cohort_role_pairing_compares_both_individual_errors(self):
        old=pd.DataFrame([{
            "season":2026,"week":6,"game":"AAA@BBB",
            "team":"AAA","player":"Wide A","opponent":"BBB",
            "role":"WR1","actual_targets":11.,
            "prior_targets_pg":9.,
            "predicted_targets":8.,
            "predicted_p10":4.,"predicted_p90":13.,
        }])
        new=old.copy()
        new.loc[0,"predicted_targets"]=10.
        roles,paired=paired_role_grade(old,new)
        self.assertEqual(roles["WR1"]["n"],1)
        self.assertEqual(roles["WR1"]["baseline_mae"],3.)
        self.assertEqual(roles["WR1"]["candidate_mae"],1.)
        self.assertEqual(roles["WR1_WR2_COMBINED"]["mae_improvement_pct"],66.6667)
        self.assertEqual(len(paired),1)


if __name__=="__main__":
    unittest.main()
