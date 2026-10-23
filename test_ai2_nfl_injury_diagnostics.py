"""Regression checks for NFL injury opportunity allocation."""
import unittest
import pandas as pd
from ai2_nfl_injury_diagnostics import opportunity_redistribution

class InjuryOpportunityTests(unittest.TestCase):
    def test_out_receiver_frees_targets_without_certification(self):
        rows = pd.DataFrame([
            {"Team":"JAC","Position":"WR","DK Status":"OUT","targets_pg":10.,
             "carries_pg":0., "match_status":"matched","weekly_projection_valid":False,
             "opportunity_verified":False},
            {"Team":"JAX","Position":"WR","DK Status":"","targets_pg":5.,
             "carries_pg":0., "match_status":"matched","weekly_projection_valid":False,
             "opportunity_verified":False},
            {"Team":"JAX","Position":"TE","DK Status":"","targets_pg":3.,
             "carries_pg":0., "match_status":"matched","weekly_projection_valid":False,
             "opportunity_verified":False},
        ])
        out, audit = opportunity_redistribution(rows)
        self.assertEqual(out.iloc[0]["adjusted_targets_pg"], 0.)
        self.assertGreater(out.iloc[1]["adjusted_targets_pg"], 5.)
        self.assertTrue((audit["conservation_delta"].abs() < 1e-8).all())
        self.assertFalse(out["weekly_projection_valid"].any())
        self.assertFalse(out["opportunity_verified"].any())

    def test_unmatched_injury_volume_not_invented(self):
        rows = pd.DataFrame([
            {"Team":"NYJ","Position":"WR","DK Status":"OUT","targets_pg":float("nan"),
             "carries_pg":float("nan"),"match_status":"no unique historical match"},
            {"Team":"NYJ","Position":"WR","DK Status":"","targets_pg":5.,
             "carries_pg":0.,"match_status":"matched"},
        ])
        out, audit = opportunity_redistribution(rows)
        self.assertAlmostEqual(out.iloc[1]["adjusted_targets_pg"], 5.)
        self.assertTrue((audit["removed_observed_volume"] == 0).all())

if __name__ == "__main__":
    unittest.main()
