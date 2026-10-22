import unittest
import numpy as np
import pandas as pd
from defensive_receiver_efficiency import adjust_receiving_efficiency,apply_verified_receiver_factor

class ReceiverEfficiencyTests(unittest.TestCase):
    def test_neutral_exact_baseline(self):
        x=adjust_receiving_efficiency(np.array([0,4,10]),.65,12.,1.)
        np.testing.assert_array_equal(x["targets"],[0,4,10])
        np.testing.assert_allclose(x["expected_receiving_yards"],np.array([0,4,10])*.65*12.)
    def test_weakness_improves_efficiency_without_creating_targets(self):
        t=np.array([2,5,9])
        x=adjust_receiving_efficiency(t,.65,12.,1.08)
        np.testing.assert_array_equal(x["targets"],t)
        self.assertGreater(x["catch_probability"],.65)
        self.assertGreater(x["yards_per_reception"],12.)
    def test_unsupported_receiver_stays_neutral(self):
        x=apply_verified_receiver_factor(np.array([4]),.65,12.,"wr2",pd.DataFrame([
            dict(player_id="wr2",supported=False,efficiency_factor=1.10)]))
        self.assertEqual(x["catch_probability"],.65)
    def test_duplicate_identity_fails_neutral(self):
        evidence=pd.DataFrame([dict(player_id="wr2",supported=True,efficiency_factor=1.10)]*2)
        x=apply_verified_receiver_factor(np.array([4]),.65,12.,"wr2",evidence)
        self.assertEqual(x["catch_probability"],.65)
    def test_unsupported_large_adjustment_rejected(self):
        with self.assertRaises(ValueError):
            adjust_receiving_efficiency(np.array([4]),.65,12.,1.5)
if __name__=="__main__":
    unittest.main()
