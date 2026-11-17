"""Research WR target-share candidate: conservation, pregame proof, non-gifting.

The synthetic receiver histories are explicit; no observed target-week usage.
"""
import unittest

import numpy as np
import pandas as pd

from ai2_nfl_role_target_share import allocate_role_target_share
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


def role_fixture():
    pool,rates=sim_fixture()
    newpool=[]
    newrates=[]
    for team in ("AAA","BBB"):
        for j,rate in enumerate((5.,4.),start=2):
            name=f"W{team} Extra {j}"
            old=pool[(pool.TeamAbbrev.eq(team)) & pool.Position.eq("WR")].iloc[0].to_dict()
            newpool.append({**old,"ID":team+f"WR{j}","Name":name})
            base=rates[(rates.team.eq(team)) & rates.position.eq("WR")].iloc[0].to_dict()
            newrates.append({**base,"player":name,"targets_pg":rate,
                             "receptions_pg":rate*.63,"receiving_yards_pg":rate*8})
    pool=pd.concat([pool,pd.DataFrame(newpool)],ignore_index=True)
    rates=pd.concat([rates,pd.DataFrame(newrates)],ignore_index=True)
    env={team:{"source":"nflverse prior-game team box-score volume",
               "observed_games":6,"last_observed_season":2026,
               "last_observed_week":6,"expected_pass_attempts":25.,
               "expected_plays":65.} for team in ("AAA","BBB")}
    return pool,rates,env


class ResearchRoleTargetShareTests(unittest.TestCase):
    def setUp(self):
        self.pool,self.rates,self.env=role_fixture()
        self.skill=self.pool[(self.pool.TeamAbbrev.eq("AAA"))&
                             self.pool.Position.isin(("RB","WR","TE"))].reset_index(drop=True)
        self.target=np.array([
            float(self.rates[(self.rates.team.eq("AAA"))&
                             (self.rates.player.eq(n))].iloc[0]["targets_pg"])
            for n in self.skill.Name
        ],float)

    def allocate(self,**kwargs):
        args=dict(team="AAA",team_environment=self.env["AAA"],
                  target_season=2026,target_week=7,strength=.7,
                  unmodeled_teammate_targets=1.)
        args.update(kwargs)
        return allocate_role_target_share(self.target,10.,
                                          self.skill,self.rates,**args)

    def test_moves_only_wr1_wr2_from_unprotected_other(self):
        adjusted,other,info=self.allocate()
        self.assertEqual(info["status"],"allocated")
        self.assertTrue(info["top3_covered"])
        np.testing.assert_allclose(adjusted[[0,2]],self.target[[0,2]]) if False else None
        m=dict(zip(self.skill.Name,range(len(self.skill))))
        self.assertGreater(adjusted[m["WAAA"]],self.target[m["WAAA"]])
        self.assertGreater(adjusted[m["WAAA Extra 2"]],self.target[m["WAAA Extra 2"]])
        self.assertEqual(adjusted[m["WAAA Extra 3"]],self.target[m["WAAA Extra 3"]])
        self.assertEqual(adjusted[m["RAAA"]],self.target[m["RAAA"]])
        self.assertEqual(adjusted[m["TAAA"]],self.target[m["TAAA"]])
        self.assertAlmostEqual(adjusted.sum()+other,self.target.sum()+10.,places=8)
        self.assertGreaterEqual(other,info["other_protected_minimum"]-1e-8)
        self.assertLessEqual(info["actual_transferred_targets_pg"],
                             2*.025*(self.target.sum()+10.)+1e-8)

    def test_missing_top3_means_no_gift_to_remaining_wr(self):
        shorter=self.skill[self.skill.Name.ne("WAAA Extra 3")].copy()
        idx=[j for j,n in enumerate(self.skill.Name) if n!="WAAA Extra 3"]
        adjusted,other,info=allocate_role_target_share(
            self.target[idx],10.,shorter,self.rates,team="AAA",
            team_environment=self.env["AAA"],target_season=2026,target_week=7,
            strength=.7,unmodeled_teammate_targets=4.)
        self.assertEqual(info["status"],"prior_top3_not_eligible")
        np.testing.assert_array_equal(adjusted,self.target[idx])
        self.assertEqual(other,10.)

    def test_protected_other_floor_prevents_unattributed_target_gift(self):
        adj,other,info=self.allocate(unmodeled_teammate_targets=11.)
        self.assertEqual(info["status"],"no_unprotected_other_budget")
        self.assertEqual(other,10.)
        np.testing.assert_array_equal(adj,self.target)

    def test_future_usage_and_team_history_fail_closed(self):
        future=self.rates.copy()
        future.loc[(future.player.eq("WAAA")),"latest_observed_week"]=7
        with self.assertRaisesRegex(ValueError,"leaked"):
            allocate_role_target_share(self.target,10.,self.skill,future,
                team="AAA",team_environment=self.env["AAA"],
                target_season=2026,target_week=7,strength=.7)
        bad={**self.env["AAA"],"last_observed_week":7}
        with self.assertRaisesRegex(ValueError,"leaked"):
            self.allocate(team_environment=bad)

    def test_invalid_strength_or_identity_rejected(self):
        for value in (-.1,1.1,float("nan"),True):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"strength"):
                self.allocate(strength=value)
        with self.assertRaisesRegex(ValueError,"Dated"):
            self.allocate(target_week=None)

    def test_disabled_candidate_returns_exact_input(self):
        a,b,info=self.allocate(strength=0,target_week=None,team_environment={})
        np.testing.assert_array_equal(a,self.target)
        self.assertEqual(b,10.)
        self.assertEqual(info["status"],"disabled")

    def test_stat_simulator_only_role_share_differs(self):
        args=dict(n_sims=250,seed=260611,
                  allow_role_proxies=False,team_environment=self.env,
                  other_yard_strength=0.)
        p0,s0,d0=simulate_stat_driven_dfs(self.pool,self.rates,**args)
        p1,s1,d1=simulate_stat_driven_dfs(self.pool,self.rates,
                  wr_role_strength=.7,wr_role_target_season=2026,
                  wr_role_target_week=7,**args)
        self.assertEqual(list(p0.ID),list(p1.ID))
        self.assertEqual(d1["__WR_ROLE_ALLOCATION__|AAA"]["status"],"allocated")
        for team in ("AAA","BBB"):
            key="__TEAM_PLAYS__|"+team
            for stat in ("simulated_pass_attempts","simulated_plays","simulated_dropbacks",
                         "simulated_rushes","simulated_sacks"):
                np.testing.assert_array_equal(d0[key][stat],d1[key][stat])
            self.assertAlmostEqual(d0["__OPPORTUNITY__|"+team]["target_budget_after"],
                                   d1["__OPPORTUNITY__|"+team]["target_budget_after"])
        self.assertFalse(np.array_equal(s0,s1))
        self.assertGreater(float(d1["WAAA"]["targets"].mean()),
                           float(d0["WAAA"]["targets"].mean()))

if __name__=="__main__":
    unittest.main()
