"""All NFL yardage/TD accounting remains physically tied to real catches."""
import unittest
import numpy as np
from ai2_nfl_other_yard_reconcile import reconcile_other_yards
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


class OtherReceiverYardsTests(unittest.TestCase):
    def test_only_existing_other_receptions_can_absorb_yards(self):
        other=np.array([9.,0.,50.,18.])
        catches=np.array([1,0,2,2])
        team=np.array([165.,85.,200.,260.])
        desired=np.array([230.,200.,260.,230.])
        amended,total,change=reconcile_other_yards(
            other,catches,team,desired,strength=1.)
        self.assertEqual(amended[0],22.)
        self.assertEqual(amended[1],0.)
        self.assertEqual(amended[2],50.)  # already beyond max other/YPR
        self.assertEqual(amended[3],6.)  # lower bound: 3 ypr
        np.testing.assert_allclose(total,team+change)
        self.assertTrue((amended>=0).all())

    def test_zero_strength_preserves_exact_original_arrays(self):
        other=np.array([10.,0.,24.])
        receives=np.array([1,0,4])
        whole=np.array([220.,0.,190.])
        desired=np.array([300.,0.,210.])
        observed,generated,shift=reconcile_other_yards(
            other,receives,whole,desired)
        np.testing.assert_array_equal(observed,other)
        np.testing.assert_array_equal(generated,whole)
        self.assertTrue(np.all(shift==0))

    def test_full_simulation_reconciles_qb_to_receivers_with_original_team_plays(self):
        pool,rates=sim_fixture()
        _,old,old_detail=simulate_stat_driven_dfs(
            pool,rates,n_sims=140,seed=35,allow_role_proxies=False)
        _,base,base_detail=simulate_stat_driven_dfs(
            pool,rates,n_sims=140,seed=35,allow_role_proxies=False,
            other_yard_strength=0.)
        np.testing.assert_array_equal(old,base)
        _,new,scored=simulate_stat_driven_dfs(
            pool,rates,n_sims=140,seed=35,allow_role_proxies=False,
            other_yard_strength=1.)
        self.assertFalse(np.array_equal(new,old))
        for team in ("AAA","BBB"):
            q=scored["Q"+team]["passing_yards"]
            receiver_yards=sum(scored[n+team]["receiving_yards"]
                               for n in ("R","W","T"))
            receiver_yards+=scored[f"__OTHER_RECEIVER__|{team}"]["receiving_yards"]
            np.testing.assert_allclose(q,receiver_yards,atol=.001)
            np.testing.assert_array_equal(
                scored[f"__TEAM_PLAYS__|{team}"]["simulated_plays"],
                old_detail[f"__TEAM_PLAYS__|{team}"]["simulated_plays"])
            np.testing.assert_array_equal(
                scored[f"__TEAM_SCORING__|{team}"]["simulated_offensive_tds"],
                old_detail[f"__TEAM_SCORING__|{team}"]["simulated_offensive_tds"])
            self.assertTrue(np.isfinite(
                scored[f"__YARD_RECONCILIATION__|{team}"]["mean_receiver_yard_scale_factor"]))

    def test_invalid_strength_fails(self):
        with self.assertRaisesRegex(ValueError,"strength"):
            reconcile_other_yards(np.ones(2),np.ones(2),
                                  np.ones(2),np.ones(2),strength=1.5)


if __name__=="__main__":
    unittest.main()
