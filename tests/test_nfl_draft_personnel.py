import unittest
import numpy as np
from ai2_nfl_draft_personnel import shrink_efficiency,personnel_factors,availability_masks,scenario_allocation
from ai2_nfl_role_evidence import player_opportunity_evidence

class DraftPersonnel(unittest.TestCase):
    def test_small_sample_ypc(self):
        self.assertGreater(shrink_efficiency(47,5.46,5,4.3),4.3)
        self.assertLess(shrink_efficiency(47,5.46,5,4.3),8.64)
    def test_efficiency_loss_depends_on_talent(self):
        low=[(4,2,20,.1,8)]
        high=[(8,6,90,.7,8)]
        y,t=personnel_factors(low,high)
        self.assertLess(y,1)
        self.assertLess(t,1)
        self.assertEqual(personnel_factors(low,[]),(1.,1.))
        self.assertGreater(personnel_factors(high,low)[0],1.)
    def test_availability_conserves_targets(self):
        r=np.random.default_rng(42)
        mask=availability_masks([True,False],100,r)
        def allocator(rng,totals,p,concentration=None):
            return np.array([rng.multinomial(int(n),p) for n in totals])
        out=scenario_allocation(r,np.full(100,30),[.4,.4,.2],mask,allocator)
        self.assertEqual(mask[:,0].sum(),50)
        self.assertTrue((out[~mask[:,0],0]==0).all())
        self.assertTrue((out.sum(axis=1)==30).all())
    def test_questionable_not_candidate(self):
        self.assertFalse(player_opportunity_evidence(dict(Position="WR",Salary=4400,draft_role_first=True,dk_status="Q",opportunity_verified=True,observed_games=8,observed_targets_pg=6,projected_targets=6))[0])

    def test_real_engine_questionable_worlds_and_conservation(self):
        from dfs_stats_engine import simulate_stat_driven_dfs
        from test_ai2_nfl_team_pace_scoring import sim_fixture
        pool,rates=sim_fixture()
        pool["dk_status"]=""
        pool.loc[pool.Name.eq("WAAA"),"dk_status"]="Q"
        _,scores,stats=simulate_stat_driven_dfs(pool,rates,n_sims=300,seed=53,
            allow_role_proxies=False,draft_personnel=True)
        self.assertTrue(np.isfinite(scores).all())
        ledger=stats["__DRAFT_PERSONNEL__|AAA"]
        self.assertEqual(ledger["questionable_worlds"][0]["active_world_fraction"],.5)
        q=ledger["questionable_worlds"][0]
        self.assertEqual(q["inactive"]["named_target_means"]["WAAA"],0.)
        self.assertGreater(q["active"]["named_target_means"]["WAAA"],0.)
        self.assertGreater(q["inactive"]["named_target_means"]["TAAA"],q["active"]["named_target_means"]["TAAA"])
        for field in ("receiving_yards","receiving_tds"):
            pass_field={"receiving_yards":"passing_yards","receiving_tds":"passing_tds"}[field]
            total=sum(stats[n][field] for n in ("WAAA","RAAA","TAAA"))+stats["__OTHER_RECEIVER__|AAA"][field]
            np.testing.assert_allclose(total,stats["QAAA"][pass_field],atol=.001)
