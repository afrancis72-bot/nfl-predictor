"""No-network pregame role, missing WR3 coverage and target-bucket regression."""
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from ai2_nfl_wr_role_audit import (
    pregame_receiver_ranks, receiver_row, team_other_row,
    receiver_role_report,
)
from scripts.audit_nfl_wr_target_roles import _paired_neutral_possession


def roles():
    return pd.DataFrame([
        dict(player="First WR",team="AAA",position="WR",games_played=5,
             latest_observed_season=2026,latest_observed_week=8,targets_pg=9.),
        dict(player="Second WR",team="AAA",position="WR",games_played=5,
             latest_observed_season=2026,latest_observed_week=8,targets_pg=6.),
        dict(player="Third WR",team="AAA",position="WR",games_played=4,
             latest_observed_season=2026,latest_observed_week=8,targets_pg=3.),
        dict(player="Fourth WR",team="AAA",position="WR",games_played=3,
             latest_observed_season=2026,latest_observed_week=8,targets_pg=1.),
        dict(player="Old WR",team="AAA",position="WR",games_played=7,
             latest_observed_season=2026,latest_observed_week=4,targets_pg=12.),
        dict(player="LEAKED WR",team="AAA",position="WR",games_played=4,
             latest_observed_season=2026,latest_observed_week=9,targets_pg=25.),
    ])


def pool():
    return pd.DataFrame([
        dict(Name="First WR",TeamAbbrev="AAA",Position="WR"),
        dict(Name="Second WR",TeamAbbrev="AAA",Position="WR"),
        dict(Name="Team RB",TeamAbbrev="AAA",Position="RB"),
    ])


class ReceiverRoleAudits(unittest.TestCase):
    def test_rank_not_leaking_and_wr3_not_silently_eliminated(self):
        ranks,teams=pregame_receiver_ranks(
            roles(),pool(),2026,9,("AAA","BBB"))
        r=teams["AAA"]
        self.assertEqual([x["role"] for x in r],
                         ["WR1","WR2","WR3","WR4+"])
        self.assertEqual([x["modeled"] for x in r],
                         [True,True,False,False])
        self.assertNotIn(("AAA","leaked wr"),ranks)
        self.assertNotIn(("AAA","old wr"),ranks)
        more=roles().copy()
        extra=more.iloc[-1:].copy()
        extra["player"]="Another future WR"
        extra["targets_pg"]=100
        new,_=pregame_receiver_ranks(pd.concat([more,extra]),
                                      pool(),2026,9,("AAA","BBB"))
        self.assertEqual(ranks,new)

    def test_receiver_rows_use_pregame_roles_not_actual_target_rank(self):
        ranks,_=pregame_receiver_ranks(
            roles(),pool(),2026,9,("AAA","BBB"))
        pre=dict(targets_pg=9.,
                 latest_observed_season=2026,
                 latest_observed_week=8)
        res=receiver_row(
            player="First WR",team="AAA",opponent="BBB",
            season=2026,week=9,game="AAA@BBB",
            observed=dict(targets=2),prior=pre,
            simulated_stats=dict(targets=np.array([7,9,11,13])),
            role_info=ranks[("AAA","first wr")],n_sims=4)
        self.assertEqual(res["role"],"WR1")
        self.assertEqual(res["actual_targets"],2)
        self.assertEqual(res["predicted_targets"],10)
        self.assertEqual(res["predicted_p90"],12.4)
        with self.assertRaisesRegex(ValueError,"Target-week"):
            bad=pre.copy();bad["latest_observed_week"]=9
            receiver_row(
                player="First WR",team="AAA",opponent="BBB",
                season=2026,week=9,game="AAA@BBB",
                observed=dict(targets=2),prior=bad,
                simulated_stats=dict(targets=np.array([7,9,11,13])),
                role_info=ranks[("AAA","first wr")],n_sims=4)

    def test_team_OTHER_not_implicitly_equated_with_real_nonmodeled_targets(self):
        draw={
            "__OPPORTUNITY__|AAA":{
                "other_target_allocation_budget_pg":6.,
                "unmodeled_teammate_target_budget_pg":4.,
                "out_context_method":"PREGAME_ABSENCE_CONDITIONED"},
            "__OTHER_RECEIVER__|AAA":{
                "targets":np.array([6,8,10])},
            "First WR":{"targets":np.array([8,10,12])},
            "Second WR":{"targets":np.array([4,5,6])},
            "Team RB":{"targets":np.array([2,3,4])},
        }
        observed=pd.DataFrame([
            dict(player_name="First WR",recent_team="AAA",position="WR",targets=9),
            dict(player_name="Second WR",recent_team="AAA",position="WR",targets=6),
            dict(player_name="Third WR",recent_team="AAA",position="WR",targets=4),
            dict(player_name="Team RB",recent_team="AAA",position="RB",targets=3),
            dict(player_name="Starting QB",recent_team="AAA",position="QB",
                 targets=0,attempts=36),
        ])
        roles_by_team=pregame_receiver_ranks(
            roles(),pool(),2026,9,("AAA","BBB"))[1]
        r=team_other_row(
            team="AAA",opponent="BBB",season=2026,week=9,
            game="AAA@BBB",pool=pool(),observations=observed,
            stat_draws=draw,history_ranked=roles_by_team["AAA"])
        self.assertEqual(r["historical_top3_wr_not_modeled"],1)
        self.assertEqual(r["actual_nonmodeled_rb_wr_te_targets"],4)
        self.assertEqual(r["actual_team_rb_wr_te_targets_known"],22)
        self.assertEqual(r["actual_qb_pass_attempts_observed"],36)
        self.assertEqual(r["actual_attempts_minus_identified_rb_wr_te_targets"],14)
        self.assertFalse(r["actual_attempt_target_counts_inconsistent"])
        self.assertTrue(r["pass_attempt_target_difference_is_not_certified_throwaways"])
        self.assertAlmostEqual(r["simulated_other_targets_mean"],8)
        self.assertGreater(r["simulated_other_target_fraction"],.20)
        report=receiver_role_report([{
            "season":2026,"week":9,"game":"AAA@BBB",
            "team":"AAA","player":"First WR","role":"WR1",
            "opponent":"BBB","actual_targets":9,
            "predicted_targets":10,"prior_targets_pg":9,
            "predicted_p10":5,"predicted_p90":15,
        }],[r])
        self.assertEqual(report["historical_top3_wr_missing_model_spots"],1)
        self.assertTrue(report["actual_and_sim_other_shares_not_directly_equivalent"])
        self.assertFalse(report["production_approved"])

    def test_paired_role_cohort_cannot_change(self):
        a=pd.DataFrame([dict(
            season=2026,week=9,game="AAA@BBB",team="AAA",
            player="First WR",role="WR1",opponent="BBB",
            actual_targets=10,prior_targets_pg=9,
            predicted_targets=8)])
        b=a.copy();b["predicted_targets"]=9
        x,merged=_paired_neutral_possession(a,b)
        self.assertEqual(x["WR1"]["n"],1)
        self.assertEqual(x["WR1"]["neutral_mae"],2)
        b.loc[0,"role"]="WR2"
        with self.assertRaisesRegex(ValueError,"Different pregame"):
            _paired_neutral_possession(a,b)


if __name__=="__main__":
    unittest.main()
