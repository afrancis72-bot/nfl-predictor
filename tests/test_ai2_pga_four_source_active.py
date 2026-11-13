"""Verify the *active unified app* PGA weights really change V11 scoring.

Synthetic acceptance data only. Never confuse this with a calibrated result.
"""
import unittest
import numpy as np
import pandas as pd
from ai2_pga_oneclick import run_tournament, parse_dk
from four_source_blend import SOURCE_DEFAULTS

class UnifiedPgaWeightsAcceptance(unittest.TestCase):
    def setUp(self):
        n=24
        self.dk=pd.DataFrame({
            "Name":[f"Golfer {i:02d}" for i in range(n)],
            "ID":[str(90000+i) for i in range(n)],
            "Salary":[7600+(i%8)*200 for i in range(n)],
        })
        names=self.dk.Name
        x=np.linspace(-1.5,1.5,n)
        self.independent=pd.DataFrame({
            "Name":names,"SG: OTT":np.sin(x*2)*.8,
            "SG: APP":x*.55,"SG: ARG":np.cos(x*2.3)*.9,
            "SG: PUTT":np.sin(x*3+.6)*.5,"SG: Total":x*.1})
        self.weighted=pd.DataFrame({
            "Name":names,"SG OTT (w)":np.cos(x*2.1)*.6,
            "SG APP (w)":np.sin(x*2.2)*.7,
            "SG ARG (w)":np.sin(x*2.1+.2)*.8,
            "SG PUTT (w)":np.cos(x*1.5)*.55})
        self.otis=pd.DataFrame({
            "Player":names,"True Skill":np.linspace(35,70,n),
            "Course Fit":np.sin(x*2.7)*16+50,"Form":np.cos(x*1.3)*10+52,
            "Fit: APP":x*15+50})
        self.defaults={"weighted_sg":20,"standard_sg":20,
                      "otis_course_fit":30,"rrg_course_fit":0,"advanced_stats":0}
        self.kwargs=dict(tournament="Synthetic PGA Tour Acceptance",course_par=72,
            no_cut=False,cut_size=15,sims=1000,seed=610,
            lineup_count=2,max_exposure=.5,salary_floor=46000,min_unique=2,
            strategy="Balanced / Single Entry",
            otis_raw=self.otis,independent_raw=self.independent,
            weighted_sg_raw=self.weighted,four_source_mode=True,
            source_weights=self.defaults)

    def test_new_source_weights_flow_through_actual_unified_simulation(self):
        a=run_tournament(self.dk,category_weights={
            "sg_ott":25,"sg_approach":25,"sg_arg":25,"sg_putting":25},**self.kwargs)
        b=run_tournament(self.dk,category_weights={
            "sg_ott":22.67,"sg_approach":22.67,"sg_arg":32,"sg_putting":22.66},**self.kwargs)
        self.assertTrue(a["metadata"]["four_source_enabled"])
        self.assertEqual(len(a["summary"]),2)
        self.assertTrue((a["exposure"]["exposure_pct"]<=50.+1e-6).all())
        self.assertTrue((a["summary"]["salary"].between(46000,50000)).all())
        self.assertEqual(a["four_source"]["blend"]["eligible_golfers"],len(self.dk))
        self.assertAlmostEqual(a["four_source"]["blend"]["weights"]["otis_course_fit"],30/70)
        self.assertGreater(a["four_source"]["blend"]["max_abs_delta_strokes"],.01)
        self.assertFalse(np.allclose(a["projections"]["expected_round"],
                                     b["projections"]["expected_round"]))
        self.assertFalse(np.allclose(a["projections"]["v11_mean_dk"],
                                     b["projections"]["v11_mean_dk"]))
        self.assertIn("sim_portfolio_at_least_one_6of6_pct",a["portfolio_audit"])

    def test_missing_required_weighted_export_fails(self):
        with self.assertRaisesRegex(ValueError,"weighted SG CSV is required"):
            run_tournament(self.dk,weighted_sg_raw=None,
                category_weights={"sg_ott":25,"sg_approach":25,"sg_arg":25,"sg_putting":25},
                **{k:v for k,v in self.kwargs.items() if k!="weighted_sg_raw"})

    def test_existing_unified_mode_remains_available(self):
        a=run_tournament(self.dk,tournament="Synthetic Legacy PGA Mode",
            course_par=72,no_cut=False,cut_size=15,sims=1000,seed=610,
            lineup_count=2,max_exposure=.5,salary_floor=46000,min_unique=2,
            independent_raw=self.independent,otis_raw=self.otis,
            four_source_mode=False)
        self.assertFalse(a["metadata"]["four_source_enabled"])
        self.assertEqual(len(a["summary"]),2)

if __name__=="__main__": unittest.main()
