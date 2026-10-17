"""Deterministic smoke tests for the research-only snap simulator."""
import unittest
import pandas as pd
import numpy as np
from snap_level_research import simulate_snaps

class SnapEngineTests(unittest.TestCase):
    def setUp(self):
        rows=[]
        for t in ("SEA","DEN"):
            for name,pos,rush,target in (("QB","QB",2,0),("RB","RB",12,3),("WR","WR",0,9),("TE","TE",0,5)):
                rows.append(dict(Name=t+" "+name,TeamAbbrev=t,Position=pos,
                                 **{"FLEX Salary":5000},usage_verified=True,
                                 rush_weight=rush,target_weight=target,carries_per_game=rush,
                                 attempts_per_game=32 if pos=="QB" else 0,
                                 qb_completion_rate=.65 if pos=="QB" else 0,
                                 qb_yards_per_attempt=7 if pos=="QB" else 0))
        self.players=pd.DataFrame(rows)

    def test_reproducible(self):
        a,_=simulate_snaps(self.players,["SEA","DEN"],n_games=20,seed=8)
        b,_=simulate_snaps(self.players,["SEA","DEN"],n_games=20,seed=8)
        np.testing.assert_array_equal(a,b)

    def test_defensive_profile_changes_results(self):
        _,baseline=simulate_snaps(self.players,["SEA","DEN"],n_games=600,seed=10)
        _,tough=simulate_snaps(self.players,["SEA","DEN"],n_games=600,seed=10,
            defense_profiles={"DEN":{"pass_yards_factor":.70,"completion_delta":-.10}})
        def sea_pass_yards(frame):
            return frame.loc[frame.name=="SEA QB","pass_yd"].mean()
        self.assertLess(sea_pass_yards(tough),sea_pass_yards(baseline))

    def test_sacks_do_not_count_as_pass_attempts(self):
        _,base=simulate_snaps(self.players,["SEA","DEN"],n_games=50,seed=71,
            defense_profiles={"DEN":{"pressure_rate":.05,"sack_given_pressure":.05}})
        _,pressure=simulate_snaps(self.players,["SEA","DEN"],n_games=50,seed=71,
            defense_profiles={"DEN":{"pressure_rate":.55,"sack_given_pressure":.55}})
        qb=pressure[pressure.name=="SEA QB"]
        self.assertGreater(qb.sacks.sum(),0)
        self.assertTrue((qb.pass_cmp<=qb.pass_att).all())
        self.assertTrue((qb.pass_att>=0).all())
        self.assertGreater(qb.sacks.sum(),base[base.name=="SEA QB"].sacks.sum())

    def test_receiving_yards_equal_completed_pass_yards(self):
        _,stats=simulate_snaps(self.players,["SEA","DEN"],n_games=40,seed=23)
        for game in stats.game.unique():
            for team in ("SEA","DEN"):
                rows=stats[(stats.game==game)&(stats.team==team)]
                self.assertAlmostEqual(float(rows.pass_yd.sum()),float(rows.rec_yd.sum()),places=2)

    def test_player_points_are_finite(self):
        points,stats=simulate_snaps(self.players,["SEA","DEN"],n_games=10,seed=9)
        self.assertTrue(np.isfinite(points).all())
        self.assertEqual(len(stats),80)

if __name__=="__main__":
    unittest.main()
