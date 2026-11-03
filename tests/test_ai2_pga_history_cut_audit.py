"""Regression guards for the actual deployed unified PGA app (synthetic fixtures)."""
import unittest

import numpy as np
import pandas as pd

from ai2_pga_historical_evidence import name_key,parse_finish,extract_finish_history
from ai2_pga_oneclick import (
    parse_dk,normalize_otis,normalize_independent_stats,
    normalize_rrg_course_fit,run_tournament,
)
from portfolio_optimizer import add_objectives


class LivePGAHistoryAndCutAuditTests(unittest.TestCase):
    def slate(self,n=24):
        names=[f"First{i} Last{i}" for i in range(n)]
        return pd.DataFrame({
            "Name":names,"ID":[str(67000+i) for i in range(n)],
            "Salary":[7800+(i%8)*200 for i in range(n)],
            "Status":["Active"]*n})

    def measured_stats(self):
        d=self.slate()
        raw=pd.DataFrame({
            "Name":d.Name,
            "Rounds":[50]*len(d),
            "SG TOT":np.linspace(-.7,1.3,len(d)),
            "SG APP":np.linspace(-.2,.8,len(d)),
            "SG OTT":np.linspace(.4,-.2,len(d)),
            "SG PUTT":np.linspace(-.2,.4,len(d)),
        })
        for number in range(1,9):
            raw[f"Recent -{number}S"]=["CUT" if i==0 else
                                        "T12" if i==1 else
                                        "T37" for i in range(len(raw))]
        raw["Hist 2025"]=["CUT","T9"]+["T35"]*(len(raw)-2)
        raw["Hist 2026"]=["W"]*len(raw) # Current year: must never leak.
        return raw

    def test_last_first_and_tournament_source_names(self):
        self.assertEqual(name_key("Last, First"),name_key("First Last"))
        self.assertEqual(parse_finish("T14"),("finish",14))
        self.assertEqual(parse_finish("W"),("finish",1))
        self.assertEqual(parse_finish("CUT"),("cut",None))
        self.assertEqual(parse_finish("WD"),("withdrawal",None))
        self.assertIsNone(parse_finish("L"))
        d=parse_dk(self.slate())
        names=d.player.str.split(" ",n=1)
        reversed_names=names.map(lambda x:f"{x[1]}, {x[0]}")
        stats=self.measured_stats()
        stats["Name"]=reversed_names
        _,matched=normalize_independent_stats(stats,d)
        self.assertEqual(matched,len(d))
        otis=pd.DataFrame({"Player":reversed_names,
                           "True Skill":np.full(len(d),55),
                           "Course Fit":np.full(len(d),50),
                           "Form":np.full(len(d),53)})
        _,count=normalize_otis(otis,d)
        self.assertEqual(count,len(d))
        fit=pd.DataFrame({"Player":reversed_names,
                           "Approach":np.linspace(0,1,len(d))})
        _,counts=normalize_rrg_course_fit(fit,d)
        self.assertEqual(counts["matched"],len(d))

    def test_recent_finish_columns_and_historical_year_audit(self):
        dk=parse_dk(self.slate())
        source=self.measured_stats()
        source.loc[2,"Recent -1S"]="WD"
        source.loc[2,"Recent -2S"]="CUT"
        data,meta=extract_finish_history(source,dk,event_year=2026)
        self.assertEqual(meta["recent_matched"],len(dk))
        self.assertEqual(meta["historical_event_matched"],len(dk))
        self.assertEqual(meta["event_history_columns"],["2025"])
        self.assertEqual(meta["recent_finish_columns"],
                         [str(n) for n in range(1,9)])
        a=data.set_index("player")
        self.assertEqual(a.loc["First0 Last0","recent_cuts_made"],0)
        self.assertEqual(a.loc["First1 Last1","recent_top25"],8)
        self.assertEqual(a.loc["First2 Last2","recent_withdrawals"],1)
        self.assertEqual(a.loc["First2 Last2","recent_missed_cuts"],1)
        self.assertGreater(a.loc["First0 Last0","recent_form_adjustment_strokes"],0)
        self.assertLess(a.loc["First1 Last1","recent_form_adjustment_strokes"],0)
        self.assertTrue(data.recent_form_adjustment_strokes.abs().le(.12).all())

    def test_cut_equity_uses_cut_probability_not_just_p75(self):
        d=pd.DataFrame({
            "player":["A","B","C","D"],
            "salary":[8000]*4,
            "make_cut_pct":[.85,.80,.35,.30],
            "win_pct":[.04,.04,.04,.04],
            "top10_pct":[.20]*4,
            "v11_mean_dk":[80,82,84,86],
            "v11_p75_dk":[100,102,104,106],
            "v11_p90_dk":[113,114,116,119],
            "dk_points_proxy":[80,82,84,86],
            "dk_value_per_1000":[10,10.25,10.5,10.75],
        })
        scored=add_objectives(d)
        self.assertGreater(scored.loc[0,"obj_cut"],scored.loc[3,"obj_cut"])
        self.assertGreater(scored.loc[3,"obj_gpp"],scored.loc[0,"obj_gpp"])
        self.assertFalse(np.allclose(scored.obj_cut,scored.obj_balanced))

    def test_live_oneclick_uses_recent_history_and_exports_joint_cut_worlds(self):
        source=self.measured_stats()
        args=dict(tournament="Synthetic Historical Acceptance",course_par=72,
            no_cut=False,cut_size=12,sims=1000,seed=19,lineup_count=2,
            max_exposure=.5,salary_floor=46000,min_unique=2,
            event_year=2026,strategy="Cut Equity")
        with_history=run_tournament(self.slate(),independent_raw=source,**args)
        without_history=run_tournament(
            self.slate(),independent_raw=source.drop(
                columns=[x for x in source.columns
                         if x.startswith("Recent ") or x.startswith("Hist ")]),**args)
        self.assertEqual(with_history["metadata"]["recent_form_matched"],24)
        self.assertEqual(with_history["metadata"]["historical_event_matched"],24)
        self.assertEqual(with_history["raw_cuts"].shape,
                         with_history["raw_draws"].shape)
        self.assertIn("sim_lineup_6of6_pct",with_history["summary"])
        self.assertIn("sim_lineup_p99",with_history["summary"])
        self.assertTrue(with_history["summary"]["sim_lineup_6of6_pct"].between(0,100).all())
        self.assertTrue((with_history["summary"]["sim_lineup_p99"] >=
                         with_history["summary"]["sim_lineup_p95"]).all())
        p=with_history["projections"].set_index("player")
        self.assertGreater(float(p.loc["First0 Last0","recent_form_adjustment_strokes"]),0)
        self.assertLess(float(p.loc["First1 Last1","recent_form_adjustment_strokes"]),0)
        self.assertFalse(np.allclose(
            with_history["projections"]["v11_mean_dk"],
            without_history["projections"]["v11_mean_dk"]))
        # No double-claim of calibrated tour/field history.
        self.assertFalse(with_history["metadata"]["recent_form_is_calibrated"])

    def test_no_cut_six_of_six_is_always_full(self):
        result=run_tournament(self.slate(),
            tournament="Synthetic No Cut Audit",course_par=72,
            no_cut=True,sims=1000,seed=8,lineup_count=2,
            max_exposure=.5,salary_floor=46000,min_unique=2)
        self.assertTrue(result["summary"]["sim_lineup_6of6_pct"].eq(100.).all())


if __name__=="__main__":
    unittest.main()
