"""Regression checks for PGA source weights in the ACTIVE unified sports preview.

Real Austin data column *shapes*, no private golfer data committed. These
tests protect against the prior NFL-style failure mode where a slider moves
a label but not the actual V11 source projections.
"""
import unittest
import numpy as np
import pandas as pd
from four_source_blend import blend_four_sources, normalize_weighted_sg
from pga_official_season_stats import (
    retrieve_official_season_stats, normalize_stat_table, fill_missing_measured_stats,
)
from ai2_pga_oneclick import run_tournament
from golf_first_v11 import simulate_golf_first

class StubTour:
    def __init__(self,names): self.names=names
    def pga_stats(self,stat_id,year,tour):
        return pd.DataFrame({"stat_id":[stat_id]*len(self.names),
            "player_name":self.names,"avg":np.linspace(68.4,73.1,len(self.names)),
            "rank":range(1,len(self.names)+1)})

class FourSourceActivePreviewTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(391)
        self.names=[f"Golfer {i}" for i in range(12)]
        self.weights=pd.DataFrame({"Name":self.names,
            "SG OTT (w)":rng.normal(0,1,12),
            "SG APP (w)":rng.normal(0,1,12),
            "SG ARG (w)":rng.normal(0,1,12),
            "SG PUTT (w)":rng.normal(0,1,12)})
        self.standard=pd.DataFrame({"Name":self.names,
            "SG OTT":rng.normal(0,1,12),"SG APP":rng.normal(0,1,12),
            "SG ARG":rng.normal(0,1,12),"SG PUTT":rng.normal(0,1,12)})
        self.otis=pd.DataFrame({"Player":self.names,
            "True Skill":rng.uniform(35,85,12),"Course Fit":rng.uniform(35,85,12),
            "Form":rng.uniform(35,85,12)})
        self.dk=pd.DataFrame({"player":self.names,"salary":[8000]*12,
                              "dk_id":[str(i+100) for i in range(12)]})
        self.sw={"weighted_sg":20,"standard_sg":20,
                 "otis_course_fit":30,"rrg_course_fit":0,"advanced_stats":0}

    def test_real_rungood_column_layout(self):
        cleaned=normalize_weighted_sg(self.weights)
        self.assertEqual(len(cleaned),12)
        self.assertEqual(set(cleaned.columns),
          {"player","w_sg_ott","w_sg_approach","w_sg_arg","w_sg_putting"})

    def test_run_unified_model_uses_custom_weights_before_72_holes(self):
        kw=dict(tournament="2026 RSM Test",course_par=72,no_cut=False,
            cut_size=8,otis_raw=self.otis,independent_raw=self.standard,
            weighted_sg_raw=self.weights,source_weights=self.sw,
            four_source_mode=True,sims=1000,seed=610,
            lineup_count=1,max_exposure=1.0,salary_floor=46500,
            min_unique=3,strategy="Balanced / Single Entry")
        arg25=run_tournament(self.dk,category_weights={
            "sg_ott":25,"sg_approach":25,"sg_arg":25,"sg_putting":25},**kw)
        arg32=run_tournament(self.dk,category_weights={
            "sg_ott":22.67,"sg_approach":22.67,"sg_arg":32,"sg_putting":22.66},**kw)
        self.assertTrue(arg25["metadata"]["four_source_enabled"])
        self.assertEqual(arg25["four_source"]["blend"]["eligible_golfers"],12)
        self.assertAlmostEqual(arg25["four_source"]["blend"]["weights"]["weighted_sg"],20/70)
        self.assertEqual(len(arg25["summary"]),1)
        self.assertEqual(len(arg32["summary"]),1)
        self.assertFalse(np.allclose(
            arg25["projections"]["v11_mean_dk"],
            arg32["projections"]["v11_mean_dk"]))
        self.assertFalse(np.allclose(
            arg25["projections"]["expected_round"],
            arg32["projections"]["expected_round"]))
        self.assertTrue(arg25["summary"].salary.between(46500,50000).all())
        self.assertTrue(arg32["summary"].salary.between(46500,50000).all())

    def test_tour_before_cut_and_no_future_leakage(self):
        p=StubTour(self.names)
        model,audit=retrieve_official_season_stats(
            pd.DataFrame({"player":self.names}),"2026-11-19",
            provider=p,selected_stats=["scoring_before_cut"],
            today="2026-11-12")
        self.assertEqual(audit.loc[0,"matched_golfers"],12)
        self.assertAlmostEqual(model.loc[0,"scoring_before_cut"],68.4)
        with self.assertRaisesRegex(ValueError,"past slate"):
            retrieve_official_season_stats(pd.DataFrame({"player":self.names}),
                                           "2026-10-01",provider=p,today="2026-11-12")
        invalid=pd.DataFrame({"stat_id":["116"],"player_name":["Golfer 0"],"rank":[1]})
        normalized,status=normalize_stat_table(invalid,"scoring_before_cut")
        self.assertTrue(normalized.empty)
        self.assertIn("header",status)

    def test_upload_precedence(self):
        original=pd.DataFrame({"player":self.names,"sg_ott":[1.5]+[np.nan]*11})
        tour=pd.DataFrame({"player":self.names,"sg_ott":[.1]*12})
        new,audit=fill_missing_measured_stats(original,tour)
        self.assertEqual(new.loc[0,"sg_ott"],1.5)
        self.assertEqual(new.loc[5,"sg_ott"],.1)
        self.assertEqual(audit["sg_ott"]["official_filled"],11)

if __name__=="__main__":unittest.main()
