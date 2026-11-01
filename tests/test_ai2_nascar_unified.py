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
from nascar_reference_dna import build_reference_dna, reference_track_family, parse_profile_text

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

    def test_martinsville_reference_uses_short_track_not_intermediate(self):
        self.assertEqual(reference_track_family("Martinsville Speedway"),"short_track")
        self.assertEqual(reference_track_family("Atlanta Motor Speedway"),"superspeedway")
        def profile(intermediate, short_track, short_elo):
            return ("<html><body><div>Performance Profile</div>"
                    "<div>14.0</div><div>Superspeedway</div>"
                    f"<div>{intermediate}</div><div>Intermediate</div>"
                    f"<div>{short_track}</div><div>Short Track</div>"
                    "<div>18.0</div><div>Road Course</div>"
                    "<div>2026 Cup Season</div><div>25</div><div>Races</div>"
                    "<div>10.0</div><div>Avg Finish</div>"
                    "<div>Best Tracks</div>"
                    "<div>NR-Rating Elo Ratings</div>"
                    "<div>1450</div><div>Superspeedway</div>"
                    "<div>1680</div><div>Intermediate</div>"
                    f"<div>{short_elo}</div><div>Short Track</div>"
                    "<div>1400</div><div>Road Course</div>"
                    "<div>Performance Quality Score</div></body></html>")
        one=profile(21.0,9.0,1620)
        two=profile(7.0,23.0,1410)
        self.assertEqual(parse_profile_text(one)["short_track_avg_finish"],9.)
        self.assertEqual(parse_profile_text(one)["short_track_elo"],1620.)
        dk=pd.DataFrame({"Name":["Driver A","Driver B"],
                         "Salary":[9000,8000],"AvgPointsPerGame":[40.,35.]})
        with patch("nascar_reference_dna.driver_page",side_effect=lambda name:one if name=="Driver A" else two):
            ref=build_reference_dna(dk,"Martinsville Speedway")
        data=ref.set_index("Name")
        self.assertEqual(float(data.loc["Driver A","comparable_track_avg_finish"]),9.)
        self.assertEqual(float(data.loc["Driver B","comparable_track_avg_finish"]),23.)
        self.assertEqual(float(data.loc["Driver A","track_type_elo"]),1620.)
        self.assertGreater(float(data.loc["Driver A","comparable_track_score"]),
                           float(data.loc["Driver B","comparable_track_score"]))
        self.assertGreater(float(data.loc["Driver A","track_type_elo_score"]),
                           float(data.loc["Driver B","track_type_elo_score"]))
        self.assertTrue((ref["reference_track_family"]=="short_track").all())

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

    def test_extra_practice_name_is_ignored_and_reported(self):
        dk=load_dk_slate(self.slate())
        uploaded=pd.DataFrame({
            "Name":[dk["Name"].iloc[0],dk["Name"].iloc[1],dk["Name"].iloc[2],
                    "Driver Outside DK Slate"],
            "avg10_rank":[1,2,3,4], "avg20_rank":[2,1,3,4],
        })
        updates,n,report=practice_updates(uploaded,dk,return_report=True)
        self.assertEqual(n,3)
        self.assertEqual(report["unmatched_practice_names"],["Driver Outside DK Slate"])
        self.assertEqual(report["practice_rows"],4)
        self.assertEqual(len(updates),len(dk))
        self.assertTrue(updates.loc[3:,"avg10_rank"].isna().all())
        self.assertEqual(updates.loc[:2,"avg10_rank"].tolist(),[1,2,3])

    def test_known_driver_aliases_match_without_guessing(self):
        dk=load_dk_slate(self.slate())
        dk.loc[:2,"Name"]=["John H. Nemechek","Ricky Stenhouse","Chris Bell"]
        practice=pd.DataFrame({
            "Name":["John Hunter Nemechek","Ricky Stenhouse Jr.","Christopher Bell"],
            "avg20_rank":[3,2,1],
        })
        updates,n,report=practice_updates(practice,dk,return_report=True)
        self.assertEqual(n,3)
        self.assertFalse(report["unmatched_practice_names"])
        self.assertEqual(updates.loc[:2,"avg20_rank"].tolist(),[3,2,1])

    def test_wrong_race_practice_is_blocked_and_duplicates_do_not_map_twice(self):
        dk=load_dk_slate(self.slate())
        mismatch=pd.DataFrame({
            "Name":[dk["Name"].iloc[0],"Unrelated 1","Unrelated 2","Unrelated 3"],
            "avg20_rank":[1,2,3,4],
        })
        with self.assertRaisesRegex(ValueError,"SAME race and series"):
            practice_updates(mismatch,dk)
        duplicate=pd.DataFrame({
            "Name":[dk["Name"].iloc[0],dk["Name"].iloc[0],dk["Name"].iloc[1]],
            "avg10_rank":[1,2,3],
        })
        with self.assertRaisesRegex(ValueError,"Duplicate practice drivers"):
            practice_updates(duplicate,dk)

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
        ids=projections.sort_values("Salary")["ID"].astype(str).tolist()
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

    def test_driver_veto_excludes_portfolio_but_not_race_field(self):
        dk=load_dk_slate(self.slate(24))
        veto=str(dk["Name"].iloc[0])
        def optimized(eligible,sim_matrix,**kwargs):
            self.assertEqual(len(eligible),23)
            self.assertNotIn(veto,eligible["Name"].tolist())
            self.assertEqual(sim_matrix.shape,(1000,23))
            return optimize_lineups(eligible,n_lineups=2,salary_cap=50000,
                roster_size=6,max_overlap=4,max_exposure=.5)
        with patch("ai2_nascar_oneclick.optimize_scenario_portfolio",side_effect=optimized):
            result=run_oneclick(dk,"Phoenix Raceway",312,
                sims=1000,seed=13,portfolio_size=2,max_exposure=.5,
                max_overlap=4,salary_floor=0,excluded_drivers=[veto])
        self.assertEqual(len(result["projections"]),24)
        self.assertIn(veto,result["metadata"]["excluded"])
        self.assertFalse(result["lineups"]["Drivers"].str.contains(veto,regex=False).any())

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
