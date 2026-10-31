"""Receiving TD probability must derive from observed catches/TDs only."""
import unittest
import numpy as np
from ai2_nfl_receiving_tds import receiving_td_factors
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


class ObservedRecTDTests(unittest.TestCase):
    def test_empirical_bayes_up_down_and_missing_are_neutral(self):
        records=[
            {"games_played":7,"receptions_pg":6.,"receiving_tds_pg":1.2},
            {"games_played":7,"receptions_pg":6.,"receiving_tds_pg":0.0},
            {"games_played":7,"receptions_pg":6.,"receiving_tds_pg":float("nan")},
            None,
        ]
        x=receiving_td_factors(records)
        self.assertGreater(x[0],1.)
        self.assertLess(x[1],1.)
        self.assertEqual(x[2],1.)
        self.assertEqual(x[3],1.)

    def test_strength_zero_does_not_change_existing_scoring(self):
        pool,rates=sim_fixture()
        a,base,base_stats=simulate_stat_driven_dfs(
            pool,rates,n_sims=140,seed=643,allow_role_proxies=False)
        b,also,other=simulate_stat_driven_dfs(
            pool,rates,n_sims=140,seed=643,allow_role_proxies=False,
            receiving_td_weight=0.)
        np.testing.assert_array_equal(base,also)

    def test_nonzero_strength_changes_players_not_team_td_budget(self):
        pool,rates=sim_fixture()
        rates.loc[(rates["team"]=="AAA")&(rates["position"]=="WR"),"receiving_tds_pg"]=1.0
        rates.loc[(rates["team"]=="AAA")&(rates["position"]=="TE"),"receiving_tds_pg"]=0.0
        a,base,baseline_stats=simulate_stat_driven_dfs(
            pool,rates,n_sims=220,seed=44,allow_role_proxies=False)
        b,scored,stats=simulate_stat_driven_dfs(
            pool,rates,n_sims=220,seed=44,allow_role_proxies=False,
            receiving_td_weight=1.)
        self.assertFalse(np.array_equal(base,scored))
        factors=stats["__RECEIVING_TD_EFFICIENCY__|AAA"]
        self.assertTrue(factors["changes_live_scoring"])
        self.assertGreater(factors["actual_pre_game_catch_td_factors"]["WAAA"],1.)
        self.assertLess(factors["actual_pre_game_catch_td_factors"]["TAAA"],1.)
        np.testing.assert_array_equal(
            stats["__TEAM_SCORING__|AAA"]["simulated_offensive_tds"],
            baseline_stats["__TEAM_SCORING__|AAA"]["simulated_offensive_tds"])
        # This changes which receiver gets a passing TD, not team touchdowns.

    def test_invalid_strength_rejected(self):
        pool,rates=sim_fixture()
        with self.assertRaisesRegex(ValueError,"strength"):
            simulate_stat_driven_dfs(pool,rates,n_sims=10,
                allow_role_proxies=False,receiving_td_weight=2.)


if __name__=="__main__":
    unittest.main()
