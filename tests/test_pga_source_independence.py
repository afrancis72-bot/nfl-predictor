"""PGA unified app source independence and contest-allocation regression tests."""
import ast
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sports" / "pga" / "legacy"))

from rrg_course_fit import normalize_rrg_course_fit, attach_source_comparison
from featured_entry import choose_featured_lineup

sys.path.insert(0, str(ROOT))
from sports.pga.dk_entries import fill_entries_template


class SourceIndependenceTests(unittest.TestCase):
    def test_rickrungood_alias_match_and_missing_values(self):
        dk = pd.DataFrame({"player": [
            "Cameron Davis", "Rico Hoey", "Zachary Bauchou",
            "Jordan L. Smith", "Missing Player"
        ]})
        rrg = pd.DataFrame({
            "Player": ["Cam Davis", "Richard Hoey", "Zach Bauchou", "Jordan Smith"],
            "FIT": [0.21, 0.4, 0.3, 0.5],
            "Approach": [0.07, 0.09, 0.1, 0.11],
        })
        normalized, report = normalize_rrg_course_fit(rrg, dk)
        self.assertEqual(report["matched"], 4)
        self.assertEqual(report["total"], 5)
        self.assertTrue(np.isnan(normalized.loc[normalized.player.eq("Missing Player"),"rrg_fit"].iat[0]))
        self.assertEqual(
            float(normalized.loc[normalized.player.eq("Cameron Davis"),"rrg_fit"].iat[0]), 0.21
        )

    def test_source_comparison_never_changes_predictions(self):
        base = pd.DataFrame({
            "player":["A","B"],"model_rank":[1,2],
            "win_pct":[0.10,0.05],"expected_round":[69.0,69.9]
        })
        rrg = pd.DataFrame({
            "player":["A","B"],"rrg_fit":[0.2,0.5],"rrg_rank":[2,1]
        })
        otis = pd.DataFrame({
            "player":["A","B"],"otis_rank":[2,1],
            "otis_true_skill":[65,80]
        })
        result = attach_source_comparison(base, rrg, otis)
        pd.testing.assert_series_equal(base.win_pct, result.win_pct)
        pd.testing.assert_series_equal(base.expected_round, result.expected_round)
        self.assertEqual(result["model_vs_rrg_rank_delta"].tolist(), [-1,1])

    def test_missing_otis_stays_missing(self):
        p = pd.DataFrame({"player":["A"],"model_rank":[1]})
        r = attach_source_comparison(p,pd.DataFrame(),pd.DataFrame())
        self.assertTrue(pd.isna(r.otis_rank.iat[0]))
        self.assertTrue(pd.isna(r.rrg_rank.iat[0]))

    def test_rejects_incorrect_rrg_schema(self):
        with self.assertRaises(ValueError):
            normalize_rrg_course_fit(pd.DataFrame({"Name":["A"],"Value":[10]}),
                                     pd.DataFrame({"player":["A"]}))


class FeaturedContestTests(unittest.TestCase):
    def sample_portfolio(self, count=11):
        golfers = []
        roster = []
        for i in range(1, count+1):
            for g in range(6):
                name=f"Golfer {i}-{g}"
                golfers.append({"player":name,"ID":str(400000+i*10+g),"salary":7000})
                roster.append({"lineup":i,"player":name,"salary":7000})
        return pd.DataFrame(roster), pd.DataFrame(golfers)

    def test_featured_lineup_selected_by_model_metrics(self):
        portfolio, golfers=self.sample_portfolio(count=3)
        all_predictions=golfers[["player"]].assign(v11_p90_dk=100)
        summary=pd.DataFrame({
            "lineup":[1,2,3],
            "sum_dk_proxy":[420,485,430],
            "avg_make_cut_pct":[0.55,0.80,0.55],
            "sum_top10_pct":[0.5,1.8,0.7]
        })
        updated, winner=choose_featured_lineup(summary,portfolio,all_predictions)
        self.assertEqual(winner,2)
        self.assertEqual(updated.loc[updated.lineup.eq(2),"contest_allocation"].iat[0],"$5 SINGLE ENTRY")
        self.assertEqual(updated.contest_allocation.eq("$5 SINGLE ENTRY").sum(),1)

    def test_actual_dk_fees_direct_feature_to_five_dollar_row(self):
        roster, golfers=self.sample_portfolio()
        # DraftKings exports use six repeated G roster columns; pandas suffixes them.
        d={"Entry ID":[str(10000+i) for i in range(11)],
           "Contest Name":["PGA Contest"]*11,
           "Entry Fee":["$0.50"]*11}
        d["Entry Fee"][8]="$5.00"
        for g in range(6):
            d["G" if g==0 else f"G.{g}"]=[""]*11
        template=pd.DataFrame(d)
        result=fill_entries_template(roster,golfers,template,featured_lineup=7)
        self.assertEqual(result.loc[8,"Entry ID"],"10008")
        self.assertEqual(result.iat[8,3],"Golfer 7-0 (400070)")
        self.assertEqual(result.loc[8,"Entry Fee"],"$5.00")
        self.assertEqual(result.loc[0,"Entry ID"],"10000")

    def test_invalid_fee_allocation_fails_closed(self):
        roster,golfers=self.sample_portfolio()
        d={"Entry ID":[str(10000+i) for i in range(11)],"Entry Fee":["$0.50"]*11}
        for g in range(6):
            d["G" if g==0 else f"G.{g}"]=[""]*11
        with self.assertRaises(ValueError):
            fill_entries_template(roster,golfers,pd.DataFrame(d),featured_lineup=7)

    def test_app_sources_parse(self):
        paths=[
            ROOT/"sports/pga/legacy/app.py",
            ROOT/"sports/pga/legacy/pga_predictor_pro.py",
            ROOT/"sports/pga/legacy/rrg_course_fit.py",
            ROOT/"sports/pga/legacy/featured_entry.py",
            ROOT/"sports/pga/dk_entries.py",
            ROOT/"pages/3_PGA_Entries.py"
        ]
        for path in paths:
            with self.subTest(path=str(path)):
                ast.parse(path.read_text(encoding="utf-8"),filename=str(path))


if __name__=="__main__":
    unittest.main()
