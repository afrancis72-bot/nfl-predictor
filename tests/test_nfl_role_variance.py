"""The optional variance mechanism conserves actual football events."""
import unittest
import numpy as np
import pandas as pd

from dfs_stats_engine import _multinomial_rows,simulate_stat_driven_dfs
from scripts.backtest_nfl_role_variance import p90_pinball,quality
from tests.test_ai2_nfl_team_pace_scoring import sim_fixture


class NflRoleVarianceTests(unittest.TestCase):
    def test_none_preserves_unmodified_sampling_reproducibly(self):
        a=np.random.default_rng(15)
        b=np.random.default_rng(15)
        counts=np.array([4,18,9,0,32])
        shares=np.array([.1,.5,.4])
        got=_multinomial_rows(a,counts,shares)
        old=np.array([b.multinomial(int(n),shares) for n in counts])
        np.testing.assert_array_equal(got,old)

    def test_new_latent_shares_conserve_touches_without_creating_players(self):
        rng=np.random.default_rng(2026)
        counts=np.full(5000,35)
        shares=np.array([.65,.30,.05,0])
        fixed=_multinomial_rows(np.random.default_rng(2026),counts,shares)
        variable=_multinomial_rows(rng,counts,shares,concentration=30.)
        np.testing.assert_array_equal(variable.sum(axis=1),counts)
        self.assertTrue(np.all(variable[:,-1]==0))
        self.assertGreater(variable[:,0].std(),fixed[:,0].std())
        self.assertLess(abs(variable[:,0].mean()-fixed[:,0].mean()),1.)

    def test_scored_fantasy_worlds_reconcile_with_team_touch_budgets(self):
        pool,rates=sim_fixture()
        _,base,base_stats=simulate_stat_driven_dfs(
            pool,rates,n_sims=180,seed=78,allow_role_proxies=False)
        _,changed,stats=simulate_stat_driven_dfs(
            pool,rates,n_sims=180,seed=78,allow_role_proxies=False,
            role_share_concentration=50.)
        self.assertFalse(np.array_equal(base,changed))
        self.assertEqual(base_stats["__ROLE_SHARE_VARIANCE__"]["mode"],
                         "EXISTING_FIXED_HISTORICAL_SHARES")
        self.assertEqual(stats["__ROLE_SHARE_VARIANCE__"]["mode"],
                         "EXPERIMENTAL_MEASURED")
        for team in ("AAA","BBB"):
            self.assertAlmostEqual(
                stats[f"__OPPORTUNITY__|{team}"]["target_budget_before"],
                stats[f"__OPPORTUNITY__|{team}"]["target_budget_after"])
            self.assertAlmostEqual(
                stats[f"__OPPORTUNITY__|{team}"]["carry_budget_before"],
                stats[f"__OPPORTUNITY__|{team}"]["carry_budget_after"])

    def test_p90_pinball_penalizes_both_under_and_over_coverage(self):
        perfect=pd.DataFrame({
            "actual_partial_dk":[10.,15.],
            "simulated_p90_dk":[10.,15.],
            "history_mean_partial_dk":[10.,15.],
            "simulated_mean_dk":[10.,15.]})
        bad=perfect.copy()
        bad.loc[0,"simulated_p90_dk"]=4.
        self.assertLess(p90_pinball(perfect),p90_pinball(bad))

    def test_bad_concentration_fails_fast(self):
        with self.assertRaisesRegex(ValueError,"concentration"):
            _multinomial_rows(np.random.default_rng(8),
                              np.array([2]),np.array([.5,.5]),
                              concentration=-1.)


if __name__=="__main__":
    unittest.main()
