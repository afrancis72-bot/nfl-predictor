"""Research-only separation of football-player universe and DK optimizer pool."""
import unittest

import numpy as np
import pandas as pd

from ai2_nfl_full_player_universe import (
    build_football_only_players,validate_football_only_players,
)
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


def fixture():
    pool,rates=sim_fixture()
    r=rates[(rates["team"]=="AAA")&(rates["position"]=="WR")].iloc[0].to_dict()
    ghost=dict(r,player="Missing Salary WR",targets_pg=5.,receptions_pg=3.1,
               receiving_yards_pg=44.,games_played=6,
               latest_observed_season=2026,latest_observed_week=6)
    other=dict(r,player="Minor Team WR",targets_pg=1.2,receptions_pg=.7,
               receiving_yards_pg=8.,games_played=5,
               latest_observed_season=2026,latest_observed_week=6)
    rates=pd.concat([rates,pd.DataFrame([ghost,other])],ignore_index=True)
    return pool,rates


class FootballUniverseResearchTests(unittest.TestCase):
    def setUp(self):
        self.pool,self.rates=fixture()
        self.ghosts=build_football_only_players(self.pool,self.rates,
                  season=2026,week=7)

    def test_history_ghosts_are_not_draftkings_eligible(self):
        self.assertSetEqual(set(self.ghosts["Name"]),
                            {"Missing Salary WR","Minor Team WR"})
        self.assertTrue((self.ghosts["Salary"]==0).all())
        self.assertFalse(self.ghosts["optimizer_eligible"].any())
        self.assertTrue(self.ghosts["football_only"].all())
        self.assertTrue(self.ghosts["role_evidence_status"].str.contains(
                             "GAME_DAY_STATUS_UNVERIFIED").all())
        validate_football_only_players(self.ghosts,self.pool,self.rates)

    def test_salary_listed_but_ineligible_is_still_a_football_player(self):
        candidate=self.pool.iloc[0].to_dict()
        candidate.update(Name="Missing Salary WR",ID="NONSELECTABLE-WR",
                         Position="WR",optimizer_eligible=False)
        extended=pd.concat([self.pool,pd.DataFrame([candidate])],ignore_index=True)
        ghost=build_football_only_players(extended,self.rates,
                                          season=2026,week=7)
        self.assertIn("Missing Salary WR",ghost["Name"].tolist())

    def test_known_out_is_excluded_before_simulation(self):
        excluded=build_football_only_players(self.pool,self.rates,
                  season=2026,week=7,
                  known_out=[dict(player="Missing Salary WR",team="AAA")])
        self.assertEqual(excluded["Name"].tolist(),["Minor Team WR"])

    def test_future_usage_must_not_be_used_as_pregame_ghost(self):
        f=self.rates.copy()
        f.loc[f["player"].eq("Missing Salary WR"),"latest_observed_week"]=7
        with self.assertRaisesRegex(ValueError,"Future/target-week"):
            build_football_only_players(self.pool,f,season=2026,week=7)

    def test_engine_rejects_smuggled_optimizer_eligible_ghost(self):
        corrupt=self.ghosts.copy()
        corrupt.loc[0,"optimizer_eligible"]=True
        with self.assertRaisesRegex(ValueError,"Invalid/duplicate"):
            self.simulate(football_only_players=corrupt)

    def test_engine_refuses_missing_date_or_future_date(self):
        with self.assertRaisesRegex(ValueError,"Dated pregame"):
            self.simulate(football_only_players=self.ghosts)
        with self.assertRaisesRegex(ValueError,"Target-week or future"):
            self.simulate(football_only_players=self.ghosts,
                          football_universe_target_season=2026,
                          football_universe_target_week=6)

    def simulate(self,**extra):
        return simulate_stat_driven_dfs(
            self.pool,self.rates,n_sims=280,seed=18888,
            allow_role_proxies=False,wr_role_team_rng_isolated=True,
            **extra)

    def test_extended_universe_is_unselectable_but_gets_real_targets(self):
        oldpool,oldscore,oldstat=self.simulate()
        newpool,newscore,newstat=self.simulate(
            football_only_players=self.ghosts,
            football_universe_target_season=2026,
            football_universe_target_week=7)
        self.assertEqual(oldpool["ID"].tolist(),newpool["ID"].tolist())
        self.assertEqual(oldpool.columns.tolist(),newpool.columns.tolist())
        self.assertNotIn("football_only",oldpool.columns)
        self.assertNotIn("football_only",newpool.columns)
        self.assertEqual(oldscore.shape,newscore.shape)
        self.assertFalse(newpool["ID"].astype(str).str.startswith("FOOTBALL_ONLY").any())
        self.assertFalse(newpool["Name"].isin(self.ghosts["Name"]).any())
        self.assertEqual(newstat["__FOOTBALL_PLAYER_UNIVERSE__"][
            "historical_football_only_players"],2)
        self.assertEqual(newstat["__FOOTBALL_PLAYER_UNIVERSE__"][
            "football_only_players_exported_to_optimizer"],0)
        for name in self.ghosts["Name"]:
            self.assertGreater(newstat[name]["targets"].mean(),0)
            self.assertGreater(newstat[name]["receptions"].mean(),0)
        self.assertNotEqual(oldstat["__OTHER_RECEIVER__|AAA"]["targets"].mean(),
                            newstat["__OTHER_RECEIVER__|AAA"]["targets"].mean())

    def test_simulated_team_targets_equal_attempts_with_ghost_receivers(self):
        _,_,stat=self.simulate(football_only_players=self.ghosts,
                        football_universe_target_season=2026,
                        football_universe_target_week=7)
        for team in ("AAA","BBB"):
            attempts=stat["__TEAM_PLAYS__|"+team]["simulated_pass_attempts"]
            player_names=self.pool[
                (self.pool["TeamAbbrev"]==team) &
                self.pool["Position"].isin(["WR","RB","TE"])]["Name"].tolist()
            if team=="AAA":
                player_names+=self.ghosts["Name"].tolist()
            total=stat["__OTHER_RECEIVER__|"+team]["targets"].copy()
            for name in player_names:
                total=total+stat[name]["targets"]
            np.testing.assert_array_equal(attempts,total)

    def test_missing_or_ambiguous_history_fails_closed(self):
        doubled=pd.concat([self.rates,self.rates.iloc[[-2]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            build_football_only_players(self.pool,doubled,season=2026,week=7)
        bad=self.ghosts.copy()
        bad.loc[0,"Name"]="WAAA"
        with self.assertRaisesRegex(ValueError,"Invalid/duplicate"):
            validate_football_only_players(bad,self.pool,self.rates)

if __name__=="__main__":
    unittest.main()
