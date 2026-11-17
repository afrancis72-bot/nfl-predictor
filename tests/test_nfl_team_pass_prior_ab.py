"""Research-only pregame team-volume blend tests.

These tests protect .5 default and confirm outcomes cannot enter the model.
"""
import unittest
import numpy as np
import pandas as pd

from dfs_stats_engine import simulate_stat_driven_dfs
from ai2_nfl_team_environment import predict_team_environment
from scripts.audit_nfl_team_pass_prior_ab import exact_paired,errors
from test_ai2_nfl_team_pace_scoring import sim_fixture,weekly

class TeamPassWeightResearchTests(unittest.TestCase):
    def setUp(self):
        self.pool,self.rates=sim_fixture()
        self.env=predict_team_environment(weekly(),2026,7)

    def draw(self,**kwargs):
        return simulate_stat_driven_dfs(self.pool,self.rates,n_sims=180,seed=71,
             allow_role_proxies=False,team_environment=self.env,**kwargs)

    def test_default_equal_explicit_half_bit_exact(self):
        a=self.draw()
        b=self.draw(team_pass_env_weight=.5)
        self.assertEqual(a[0].Name.tolist(),b[0].Name.tolist())
        self.assertTrue(np.array_equal(a[1],b[1]))
        for team in ("AAA","BBB"):
            k="__TEAM_PLAYS__|"+team
            self.assertTrue(np.array_equal(a[2][k]["simulated_pass_attempts"],
                                           b[2][k]["simulated_pass_attempts"]))

    def test_candidate_uses_only_team_history(self):
        base=self.draw(team_pass_env_weight=.5)
        cand=self.draw(team_pass_env_weight=1.)
        for team in ("AAA","BBB"):
            r=cand[2]["__TEAM_PASS_BASELINE__|"+team]
            self.assertEqual(r["team_env_weight_effective"],1.)
            self.assertAlmostEqual(r["pregame_pass_attempt_forecast"],
                 self.env[team]["expected_pass_attempts"])
        self.assertFalse(np.array_equal(base[1],cand[1]))

    def test_missing_team_evidence_preserves_qb_fallback(self):
        a=simulate_stat_driven_dfs(self.pool,self.rates,n_sims=180,seed=71,
                                   allow_role_proxies=False,team_environment={})
        b=simulate_stat_driven_dfs(self.pool,self.rates,n_sims=180,seed=71,
                                   allow_role_proxies=False,team_environment={},
                                   team_pass_env_weight=1.)
        self.assertTrue(np.array_equal(a[1],b[1]))
        for team in ("AAA","BBB"):
            self.assertEqual(b[2]["__TEAM_PASS_BASELINE__|"+team]["team_env_weight_effective"],0.)

    def test_invalid_weights_fail_before_simulating(self):
        for w in (-0.01,1.01,float("nan"),float("inf"),True,"1"):
            with self.subTest(weight=w),self.assertRaisesRegex(ValueError,"weight"):
                self.draw(team_pass_env_weight=w)

    def test_actual_future_results_do_not_change_prior(self):
        hist=weekly()
        result=predict_team_environment(hist,2026,7)
        future=hist.copy()
        future["week"]=8
        future["attempts"]=65
        extended=pd.concat([hist,future],ignore_index=True)
        self.assertEqual(result,predict_team_environment(extended,2026,7))

    def test_paired_audit_rejects_wrong_sample_and_truth(self):
        a=[dict(season=2026,week=8,game="AAA@BBB",team="AAA",
                player="AAA QB",position="QB",actual=8.,old=7.,new=9.)]
        b=[dict(a[0])]
        p=exact_paired(a,b,["season","week","game","team","player","position"],["actual"])
        self.assertEqual(errors(p,"actual_base","old_base","old_candidate")["baseline_mae"],1.)
        with self.assertRaisesRegex(ValueError,"truth"):
            exact_paired(a,[dict(a[0],actual=9.)],
                ["season","week","game","team","player","position"],["actual"])
        with self.assertRaisesRegex(ValueError,"different"):
            exact_paired(a,[dict(a[0],player="Changed Name")],
                ["season","week","game","team","player","position"],["actual"])

if __name__=="__main__":
    unittest.main()
