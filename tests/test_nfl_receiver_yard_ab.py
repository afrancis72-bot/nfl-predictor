"""Protect paired receiver-yard experiment against changed opportunity or actuals."""
import unittest
import numpy as np
import pandas as pd
from scripts.audit_nfl_receiver_yard_ab import paired_results
from ai2_nfl_other_yard_reconcile import reconcile_other_yards
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture

class ReceivingYardResearchTests(unittest.TestCase):
    def test_other_first_respects_catch_bound_and_no_new_yards_without_catch(self):
        y=np.array([9.,9.,0.])
        c=np.array([1,0,2])
        generated=np.array([120.,120.,120.])
        desired=np.array([160.,160.,160.])
        oth,tot,shift=reconcile_other_yards(y,c,generated,desired,strength=.7)
        np.testing.assert_allclose(oth,[22,9,30])
        np.testing.assert_allclose(shift,[13,0,30])
        np.testing.assert_allclose(tot-generated,shift)

    def test_zero_is_baseline_default(self):
        o=np.array([5.,12.])
        c=np.array([1,2])
        t=np.array([150.,210.])
        d=np.array([180.,160.])
        a=reconcile_other_yards(o,c,t,d)
        b=reconcile_other_yards(o,c,t,d,strength=0.)
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x,y)

    def test_reconciler_does_not_modify_simulated_target_catch_td(self):
        pool,rates=sim_fixture()
        p0,s0,stat0=simulate_stat_driven_dfs(pool,rates,n_sims=200,seed=132,
                                              allow_role_proxies=False,other_yard_strength=0.)
        p1,s1,stat1=simulate_stat_driven_dfs(pool,rates,n_sims=200,seed=132,
                                              allow_role_proxies=False,other_yard_strength=.7)
        self.assertEqual(list(p0.ID),list(p1.ID))
        for p in pool[pool.Position.isin(["RB","WR","TE"])].to_dict("records"):
            name=p["Name"]
            for key in ["targets","receptions","receiving_tds",
                        "carries","rushing_tds","rushing_yards"]:
                np.testing.assert_array_equal(stat0[name][key],stat1[name][key])
        for team in ("AAA","BBB"):
            for key in ("simulated_pass_attempts","simulated_dropbacks","simulated_plays"):
                np.testing.assert_array_equal(stat0["__TEAM_PLAYS__|"+team][key],
                                              stat1["__TEAM_PLAYS__|"+team][key])

    def test_paired_refuses_missing_stats_and_changed_ground_truth(self):
        key=dict(season=2026,week=9,game="AAA@BBB",team="AAA",
                 player="WR A",position="WR")
        player=[dict(key,actual_partial_dk=11.,history_mean_partial_dk=10.,
                     opponent="BBB",simulated_mean_dk=9.,simulated_p90_dk=20.)]
        basic=dict(key,stat="receiving_yards",actual=75.,prior_baseline=63.,
                   category="efficiency_output",direct_dk_weight=.1,
                   predicted_mean=60.,predicted_p90=94.)
        stat=[basic,dict(key,stat="receptions",actual=5.,
             prior_baseline=5.,category="opportunity",direct_dk_weight=1.,
             predicted_mean=4.,predicted_p90=8.)]
        with self.assertRaisesRegex(ValueError,"different"):
            paired_results(player,[dict(player[0],player="WR B")],stat,stat)
        with self.assertRaisesRegex(ValueError,"mismatch"):
            paired_results(player,player,stat,
              [dict(x,actual=99.) if x["stat"]=="receiving_yards" else x for x in stat])
        with self.assertRaisesRegex(ValueError,"different"):
            paired_results(player,player,stat,stat[:1])

if __name__=="__main__":
    unittest.main()
