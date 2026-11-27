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
