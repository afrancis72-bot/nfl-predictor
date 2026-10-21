"""Regression checks for conservation of injury opportunity budgets.

These tests run a minimal synthetic two-team simulation without live feeds.
"""
import unittest
import numpy as np
from injury_opportunity import redistribute

class RedistributionConservationTests(unittest.TestCase):
    def test_target_budget_conserved(self):
        rates=np.array([7.,4.,2.],dtype=float)
        other=3.
        adjusted,remainder=redistribute(rates.copy(),other,missing_volume=5.,
                                         role_weights=np.array([1.,.8,.5]))
        self.assertAlmostEqual(float(adjusted.sum()+remainder),float(rates.sum()+other+5.))
        self.assertTrue(np.all(adjusted>=0))
        self.assertGreaterEqual(remainder,0)
    def test_carry_budget_conserved(self):
        rates=np.array([13.,3.,0.],dtype=float)
        other=4.
        adjusted,remainder=redistribute(rates.copy(),other,missing_volume=7.,
                                         role_weights=np.array([1.,.1,.05]))
        self.assertAlmostEqual(float(adjusted.sum()+remainder),float(rates.sum()+other+7.))
        self.assertTrue(np.all(adjusted>=0))
    def test_zero_missing_volume_no_change(self):
        rates=np.array([5.,3.],dtype=float)
        adjusted,remainder=redistribute(rates.copy(),2.,missing_volume=0.,
                                         role_weights=np.array([1.,1.]))
        np.testing.assert_allclose(adjusted,rates)
        self.assertAlmostEqual(float(remainder),2.)
if __name__=="__main__":
    unittest.main()
