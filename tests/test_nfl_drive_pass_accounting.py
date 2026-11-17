"""Conservation tests for NFL plays -> dropbacks -> sacks -> attempts.

Postgame observed PBP cannot affect the simulated draws or historical priors.
"""
from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from ai2_nfl_drive_pass_accounting import (
    observed_pbp_team_games,component_row,report_metrics,
)

class TestDrivePassAccounting(unittest.TestCase):
    def setUp(self):
        self.n=3
        self.players=pd.DataFrame([
            {"Name":"A QB","TeamAbbrev":"AAA","Position":"QB"},
            {"Name":"First WR","TeamAbbrev":"AAA","Position":"WR"},
            {"Name":"A RB","TeamAbbrev":"AAA","Position":"RB"},
            {"Name":"A TE","TeamAbbrev":"AAA","Position":"TE"},
        ])
        v=lambda n:np.full(3,n,dtype=np.int32)
        self.draws={
            "__TEAM_PLAYS__|AAA":{
                "simulated_plays":v(20),"simulated_dropbacks":v(12),
                "simulated_rushes":v(8),"simulated_pass_attempts":v(10),
                "simulated_sacks":v(2),
                "pregame_expected_dropbacks":12,"pregame_pass_fraction":.6,
                "pregame_sack_rate":.08,"pregame_pass_attempts":10,
                "expected_plays":20
            },
            "__TEAM_ENVIRONMENT__|AAA":{"team_pass_attempts_prior":12.},
            "__TEAM_PASS_BASELINE__|AAA":{"team_pass_attempts_baseline":12.},
            "__OPPORTUNITY__|AAA":{"other_target_allocation_budget_pg":3.},
            "__OTHER_RECEIVER__|AAA":{"targets":v(4)},
            "First WR":{"targets":v(4)},
            "A RB":{"targets":v(1)},
            "A TE":{"targets":v(1)},
        }
        self.truth=dict(season=2026,week=9,game="AAA@BBB",team="AAA",
            actual_pass_attempts=15,actual_sacks=1,actual_rushes=8,
            actual_dropbacks=16,actual_plays=24,eligible_observed_drives=5)
        self.roles=[dict(player_key="first wr",role="WR1")]
        self.actual_players=pd.DataFrame([
            dict(recent_team="AAA",position="WR",player_name="First WR",targets=10),
            dict(recent_team="AAA",position="TE",player_name="A TE",targets=3)
        ])

    def call(self,truth=None,engine="neutral",draws=None):
        return component_row(season=2026,week=9,game="AAA@BBB",team="AAA",
            players=self.players,stat_draws=draws or self.draws,
            actual=truth or self.truth,observations=self.actual_players,
            pregame_roles=self.roles,n_sims=3,engine_name=engine)

    def test_exact_three_way_pass_attempt_decomposition(self):
        r=self.call()
        self.assertAlmostEqual(r["plays_error_component"],-4*16/24)
        self.assertAlmostEqual(r["pass_run_error_component"],20*(12/20-16/24))
        self.assertAlmostEqual(r["sacks_error_component"],-1)
        self.assertAlmostEqual(r["total_attempts_error"],-5)
        self.assertAlmostEqual(sum(r["simulated_target_allocation"].values()),10)
        self.assertEqual(r["simulated_target_allocation"]["WR1"],4)
        self.assertEqual(r["actual_known_position_and_rank_targets"]["WR1"],10)

    def test_postgame_changes_are_only_truth_not_simulated(self):
        a=self.call()
        altered=dict(self.truth,actual_pass_attempts=17,actual_dropbacks=18,
                     actual_plays=26)
        b=self.call(altered)
        for key in ("sim_plays","sim_dropbacks","sim_pass_attempts","sim_sacks",
                    "pregame_pass_attempts","pregame_pass_fraction"):
            self.assertEqual(a[key],b[key])
        self.assertNotEqual(a["total_attempts_error"],b["total_attempts_error"])

    def test_failed_plays_and_receiver_conservation_rejected(self):
        a=dict(self.draws)
        a["__TEAM_PLAYS__|AAA"]=dict(a["__TEAM_PLAYS__|AAA"],
                                   simulated_pass_attempts=np.array([11,11,11]))
        with self.assertRaisesRegex(ValueError,"conservation"):
            self.call(draws=a)
        b=dict(self.draws)
        b["__OTHER_RECEIVER__|AAA"]={"targets":np.array([5,5,5])}
        with self.assertRaisesRegex(ValueError,"targets plus OTHER"):
            self.call(draws=b)

    def test_possession_requires_drive_world(self):
        with self.assertRaisesRegex(ValueError,"did not produce"):
            self.call(engine="possession")
        a=dict(self.draws)
        a["__POSSESSION_TEAM__|AAA"]={"simulated_drives":np.array([4,4,4])}
        r=self.call(engine="possession",draws=a)
        self.assertEqual(r["sim_drives"],4.)
        self.assertEqual(r["sim_plays_per_drive"],5.)

    def test_pbp_observations_are_separated_from_future_predictions(self):
        sample=[
            {"season":2026,"week":9,"season_type":"REG","game_id":"g1",
             "play_id":i,"posteam":"AAA",
             "pass_attempt":int(i in (1,2)),"sack":int(i==3),
             "rush_attempt":int(i==4)}
            for i in (1,2,3,4)
        ]
        game=pd.DataFrame([dict(season=2026,week=9,game_type="REG",
            home_score=20,away_score=17,game_id="g1",
            away_team="AAA",home_team="BBB")])
        drives=pd.DataFrame([{"season":2026,"week":9,"game_id":"g1","offense":"AAA"}
                             for _ in range(2)])
        observed=observed_pbp_team_games(pd.DataFrame(sample),drives,game)
        self.assertEqual(observed[(2026,9,"AAA@BBB","AAA")]["actual_plays"],4)
        self.assertEqual(observed[(2026,9,"AAA@BBB","AAA")]["actual_dropbacks"],3)
        # nflverse pass_attempt flags include sacks: do not double count.
        overlapping_sack=pd.DataFrame(sample)
        overlapping_sack.loc[0,"sack"]=1
        counted=observed_pbp_team_games(overlapping_sack,drives,game)
        adjusted=counted[(2026,9,"AAA@BBB","AAA")]
        self.assertEqual(adjusted["actual_pass_attempts"],1)
        self.assertEqual(adjusted["actual_sacks"],2)
        self.assertEqual(adjusted["actual_plays"],4)
        # Rush+sack remains invalid and must be rejected.
        contradictory=pd.DataFrame(sample)
        contradictory.loc[0,"rush_attempt"]=1
        with self.assertRaisesRegex(ValueError,"ambiguous rush/pass/sack"):
            observed_pbp_team_games(contradictory,drives,game)

    def test_report_metric_rejects_duplicate(self):
        x=self.call()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            report_metrics(pd.DataFrame([x,x]))

if __name__=="__main__":
    unittest.main()
