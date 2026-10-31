"""Regression tests for the real Team AI² Pro PGA salary-tier review."""
import unittest
import numpy as np
import pandas as pd
from ai2_pga_salary_review import build_review,attach_comparison,rebuild_reviewed

def golfers(n=30):
    idx=np.arange(n)
    salary=6000+((idx*457)%5400)
    mean=65+idx*.57
    p90=mean+25+np.sin(idx)*5
    return pd.DataFrame({
        "player":[f"Golfer {i:02d}" for i in idx],
        "salary":salary,
        "dk_id":[str(1000+i) for i in idx],
        "v11_mean_dk":mean, "v11_p75_dk":mean+15,
        "v11_p90_dk":p90, "v11_salary_value":mean/(salary/1000),
        "dk_points_proxy":mean,"dk_value_per_1000":mean/(salary/1000),
        "make_cut_pct":(.51+.012*idx).clip(0,.9),
        "top10_pct":np.full(n,.12),"win_pct":np.full(n,.01),
        "sg_approach":np.cos(idx/4),"sg_ott":np.sin(idx/4),
        "sg_putting":np.sin(idx/3),"birdie_rate":.22+.03*np.cos(idx/2),
        "measured_rounds":np.full(n,40)
    })

class PGAReviewTests(unittest.TestCase):
    def test_ranking_has_five_salary_bands_and_grounded_evidence(self):
        d=golfers()
        out=build_review(d,{"independent_sg_match":len(d)})
        self.assertEqual(set(out.tier),{"$6K","$7K","$8K","$9K","$10K+"})
        self.assertTrue(out.why.str.contains("Simulated").all())
        self.assertTrue(out.why.str.contains("measured").any())
        self.assertTrue((out.groupby("tier").tier_rank.min()==1).all())

    def test_market_baseline_is_not_mislabeled_as_golf_skill(self):
        d=golfers().drop(columns=["sg_approach","sg_ott","sg_putting","birdie_rate"])
        out=build_review(d,{"independent_sg_match":0,"otis_match":0})
        self.assertTrue(out.concerns.str.contains("Salary-only baseline").all())
        self.assertTrue(out.evidence.eq("Market only").all())

    def test_comparison_does_not_change_independent_projections(self):
        d=golfers()
        base=build_review(d,{"independent_sg_match":30})
        other=pd.DataFrame({"player":d.player,"otis_rank":np.arange(1,31)})
        rrg=pd.DataFrame({"player":d.player,"rrg_fit_rank_audit":np.arange(30,0,-1)})
        compared=attach_comparison(base,other,rrg)
        self.assertIn("RRG course-fit rank",compared)
        np.testing.assert_array_equal(compared.mean_dk,base.mean_dk)

    def test_vetoed_golfers_never_appear_and_originals_preserved(self):
        d=golfers()
        rng=np.random.default_rng(3)
        draws=rng.normal(d["v11_mean_dk"].to_numpy(),
                         20,size=(400,len(d)))
        d["v11_mean_dk"]=draws.mean(axis=0)
        d["dk_points_proxy"]=d["v11_mean_dk"]
        result={"projections":d,"raw_draws":draws,
                "metadata":{"seed":43}}
        original=d.copy(deep=True)
        out=rebuild_reviewed(result,vetoes=["Golfer 29","Golfer 28"],
                  lineup_count=3,max_exposure=.67,salary_floor=39000,
                  min_unique=1,seed=43)
        self.assertTrue(set(out["lineups"].player).isdisjoint({"Golfer 29","Golfer 28"}))
        self.assertEqual(len(out["summary"]),3)
        self.assertTrue((out["exposure"].exposure_pct<=67.00001).all())
        self.assertIn("sim_lineup_p90",out["summary"])
        pd.testing.assert_frame_equal(original,result["projections"])

    def test_veto_lock_collision_is_blocked(self):
        d=golfers()
        draws=np.tile(d.v11_mean_dk.to_numpy(),(10,1))
        with self.assertRaises(ValueError):
            rebuild_reviewed({"projections":d,"raw_draws":draws,
                         "metadata":{"seed":43}},vetoes=["Golfer 01"],locks=["Golfer 01"],
                         lineup_count=2)

if __name__=="__main__":
    unittest.main()
