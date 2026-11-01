"""NFL lineups never export unauditable high projected fantasy scores."""
import unittest
import numpy as np
import pandas as pd
from ai2_nfl_player_audit import (
    summarize_simulated_players,require_reconciled_player_audit)
from ai2_nfl_projection_review import projection_review_flags


class ReconciledScoringTests(unittest.TestCase):
    def fixture(self):
        n=120
        players=pd.DataFrame([
            dict(ID="puka",Name="Puka Nacua",TeamAbbrev="LAR",
                 Position="WR",observed_games=7,proj=20.2,
                 projection_source="nflverse prior game history"),
            dict(ID="bijan",Name="Bijan Robinson",TeamAbbrev="ATL",
                 Position="RB",observed_games=7,proj=19.,
                 projection_source="nflverse prior game history"),
        ])
        recs=np.full(n,13)
        recyards=np.full(n,170.)
        rectds=np.ones(n)
        no=np.zeros(n)
        bijan_recs=np.full(n,5)
        bijan_yards=np.full(n,40.)
        bijan_rush=np.full(n,105.)
        bijan_carries=np.full(n,19)
        bijan_td=np.ones(n)
        points=np.column_stack([
            recs+rec_yards(recyards)+6*rectds+3,
            bijan_recs+rec_yards(bijan_yards)+rec_yards(bijan_rush)+6*bijan_td+3,
        ])
        detail={
            "Puka Nacua":dict(targets=np.full(n,17),receptions=recs,
                receiving_yards=recyards,receiving_tds=rectds,
                carries=no,rushing_yards=no,rushing_tds=no),
            "Bijan Robinson":dict(targets=np.full(n,6),receptions=bijan_recs,
                receiving_yards=bijan_yards,receiving_tds=no,
                carries=bijan_carries,rushing_yards=bijan_rush,
                rushing_tds=bijan_td),
            "__OPPORTUNITY__|LAR":dict(
                active_target_rates_pg={"Puka Nacua":17.},
                target_budget_after=39.),
            "__TEAM_PASS_BASELINE__|LAR":dict(team_pass_attempts_baseline=40.),
            "__TEAM_SCORING__|LAR":dict(expected_offensive_tds=2.6),
        }
        return players,points,detail

    def test_complete_football_worlds_prove_same_scenario_points(self):
        players,points,details=self.fixture()
        audits=summarize_simulated_players(players,points,details)
        self.assertTrue(require_reconciled_player_audit(players,points,audits))
        warnings=projection_review_flags(audits,players)
        self.assertEqual({w["Player"] for w in warnings},
                         {"Puka Nacua","Bijan Robinson"})
        puka=next(w for w in warnings if w["Player"]=="Puka Nacua")
        self.assertIn("receiving yards",puka["Fantasy point breakdown"])
        self.assertIn("Mean targets: 17.0",puka["Simulated opportunities"])
        self.assertIn("Team pass attempts prior",puka["Simulated opportunities"])
        self.assertIn("nflverse",puka["Projection source"])

    def test_missing_stat_report_fails_closed(self):
        players,points,details=self.fixture()
        reports=summarize_simulated_players(players,points,details)
        with self.assertRaisesRegex(ValueError,"Unreconciled NFL"):
            require_reconciled_player_audit(players,points,reports[:1])

    def test_fantasy_points_unbacked_by_real_stats_fail(self):
        players,points,details=self.fixture()
        details["Puka Nacua"]["receiving_yards"]=np.zeros(120)
        with self.assertRaisesRegex(ValueError,"reconciliation"):
            summarize_simulated_players(players,points,details)

    def test_ambiguous_name_blocks_eligible_lineups(self):
        players,points,details=self.fixture()
        players.loc[1,"Name"]="Puka Nacua"
        reports=summarize_simulated_players(players,points,details)
        with self.assertRaisesRegex(ValueError,"Unreconciled NFL"):
            require_reconciled_player_audit(players,points,reports)

    def test_mismatched_identity_or_mean_fails(self):
        players,points,details=self.fixture()
        audits=summarize_simulated_players(players,points,details)
        audits[0]["Rebuilt mean DK"]+=1.0
        with self.assertRaisesRegex(ValueError,"disagree"):
            require_reconciled_player_audit(players,points,audits)
        audits=summarize_simulated_players(players,points,details)
        audits[0]["Team"]="NYG"
        with self.assertRaisesRegex(ValueError,"Misidentified"):
            require_reconciled_player_audit(players,points,audits)


def rec_yards(array):
    return .1*array


if __name__=="__main__":
    unittest.main()
