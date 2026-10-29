"""Offline acceptance tests for the unified NASCAR one-click path."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from ai2_nascar_oneclick import (
    load_dk_slate, practice_updates, build_input_features, run_oneclick,
    validate_portfolio, dk_roster_csv, track_catalog,
)
from nascar_predictor_pro import Config, simulate, optimize_lineups

class NascarUnifiedTests(unittest.TestCase):
    def slate(self,n=24):
        return pd.DataFrame({
            "Name":[f"Test Driver {i:02d}" for i in range(1,n+1)],
            "ID":[str(110000+i) for i in range(1,n+1)],
            "Salary":[6000+(i*175)%5000 for i in range(n)],
            "AvgPointsPerGame":[30.0+i*.73 for i in range(n)],
            "Starting":list(range(1,n+1)),
        })

    def test_track_catalog_and_post_qualifying_slate(self):
        self.assertIn("Phoenix Raceway",set(track_catalog()["track"]))
        self.assertEqual(len(load_dk_slate(self.slate())),24)

    def test_start_grid_is_never_fabricated(self):
        dk=self.slate()
        for bad in [
            dk.drop(columns=["Starting"]),
            dk.assign(Starting=np.nan),
            dk.assign(Starting=[1]*24),
            dk.assign(Starting=list(range(1,24))+[25]),
        ]:
            with self.assertRaises(ValueError):
                load_dk_slate(bad)
        with self.assertRaises(ValueError):
            load_dk_slate(dk.assign(ID=["123"]*24))

    def test_optional_practice_speeds_convert_to_true_ranks(self):
        dk=load_dk_slate(self.slate())
        practice=pd.DataFrame({
            "Driver":dk["Name"].tolist()[:3],
            "avg10_speed":[179.,181.,180.],
            "avg20_rank":[2,1,3],
        })
        updates,count=practice_updates(practice,dk)
        self.assertEqual(count,3)
        self.assertEqual(updates["qualifying_position"].tolist(),dk["Starting"].tolist())
        self.assertEqual(updates["avg10_rank"].iloc[:3].tolist(),[3,1,2])

    def test_simulated_driver_column_alignment_and_projection_seed(self):
        dk=load_dk_slate(self.slate())
        features,meta=build_input_features(dk,"Phoenix Raceway",312)
        self.assertEqual(meta["practice_drivers"],0)
        self.assertEqual(meta["reference_drivers"],0)
        p1,m1=simulate(features,Config(sims=1000,seed=13,laps=312))
        p2,m2=simulate(features,Config(sims=1000,seed=13,laps=312))
        p3,m3=simulate(features,Config(sims=1000,seed=21,laps=312))
        self.assertEqual(m1.shape,(1000,24))
        np.testing.assert_array_equal(m1,m2)
        self.assertFalse(np.array_equal(m1,m3))
        np.testing.assert_allclose(m1.mean(0),p1["proj_dk"].to_numpy(),atol=.003)
        np.testing.assert_array_equal(p1["ID"].to_numpy(),p2["ID"].to_numpy())

    def test_lineup_and_exposure_audit_blocks_illegal_outputs(self):
        dk=load_dk_slate(self.slate())
        features,_=build_input_features(dk,"Phoenix Raceway",312)
        projections,_=simulate(features,Config(sims=1000,seed=42,laps=312))
        # Create two low-salary legal 6-driver groups with unique IDs.
        ids=projections["ID"].astype(str).tolist()
        groups=[ids[:6],ids[6:12]]
        rows=[]
        for ix, group in enumerate(groups,1):
            actual=projections.set_index("ID").loc[group]
            rows.append({"Lineup":ix,"IDs":" | ".join(group),
                         "Salary":int(actual["Salary"].sum())})
        lineup=pd.DataFrame(rows)
        self.assertTrue(validate_portfolio(lineup,projections,2,.5,4,floor=0))
        self.assertIn("D,D,D,D,D,D",dk_roster_csv(lineup).decode("utf-8-sig"))
        with self.assertRaises(ValueError):
            validate_portfolio(lineup,projections,2,.35,4,floor=0)
        with self.assertRaises(ValueError):
            validate_portfolio(lineup.head(1),projections,2,.5,4,floor=0)

    def test_one_button_runs_actual_engine_with_real_grid_and_mocked_candidate_search(self):
        dk=load_dk_slate(self.slate(24))
        def fast_portfolio(projected,sim_matrix,**kwargs):
            return optimize_lineups(projected,n_lineups=2,
                salary_cap=50000,roster_size=6,max_overlap=4,max_exposure=.5)
        with patch("ai2_nascar_oneclick.optimize_scenario_portfolio",side_effect=fast_portfolio):
            result=run_oneclick(dk,"Phoenix Raceway",312,
                sims=1000,seed=55,portfolio_size=2,
                max_exposure=.5,max_overlap=4,salary_floor=0)
        self.assertEqual(result["metadata"]["seed"],55)
        self.assertEqual(result["metadata"]["laps"],312)
        self.assertEqual(len(result["lineups"]),2)
        self.assertEqual(len(result["projections"]),24)
        self.assertTrue(result["roster_csv"].startswith(b"\xef\xbb\xbf"))

if __name__=="__main__":
    unittest.main()
