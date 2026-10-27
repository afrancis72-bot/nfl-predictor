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
from rrg_strokes_gained import normalize_rrg_strokes_gained
from featured_entry import choose_featured_lineup
from mexico_course_evidence import mexico_course_evidence, mexico_course_thesis
from portfolio_optimizer import optimize_portfolio, PortfolioSettings

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



class ObservedSgTests(unittest.TestCase):
    def setUp(self):
        self.dk = pd.DataFrame({
            "player": ["Cameron Davis", "Rico Hoey", "Zachary Bauchou", "Jordan L. Smith", "No Data"],
            "salary": [8800, 8200, 7000, 6800, 6000],
        })
        self.csv = pd.DataFrame({
            "Name": ["Cam Davis", "Richard Hoey", "Zach Bauchou", "Jordan Smith", "No Data"],
            "Salary": [8800, 8200, 7000, 6800, 6000],
            "Rounds": [50, 20, 4, 2, 0],
            "SG OTT": [0.5, 0.3, 0.8, 0.4, np.nan],
            "SG APP": [0.7, 0.2, 0.5, -0.2, np.nan],
            "SG ARG": [0.1, 0.4, 0.2, -0.3, np.nan],
            "SG PUTT": [-0.1, 0.1, 0.2, 0.8, np.nan],
            "SG TOT": [1.2, 1.0, 1.7, 0.7, np.nan],
            "Driving Distance": [300, 310, 320, 295, np.nan],
            "Driving Accuracy %": [64, 58, 61, 72, np.nan],
            "GIR %": [69, 68, 71, 62, np.nan],
            "Recent -1S": ["T10", "CUT", "2", "T4", ""],
        })

    def test_complete_field_and_named_aliases(self):
        result, audit = normalize_rrg_strokes_gained(self.csv, self.dk)
        self.assertEqual(audit["matched"], 5)
        self.assertEqual(audit["full_sg_components"], 4)
        self.assertEqual(audit["sg_total_available"], 4)
        self.assertEqual(
            float(result.loc[result.player.eq("Cameron Davis"),"rrg_raw_sg_approach"].iat[0]), 0.7
        )
        self.assertEqual(
            float(result.loc[result.player.eq("Rico Hoey"),"rrg_raw_sg_total"].iat[0]), 1.0
        )
        self.assertEqual(len(result), len(self.dk))

    def test_small_sample_shrinkage_and_missing_data(self):
        result, _ = normalize_rrg_strokes_gained(self.csv, self.dk, prior_rounds=12.0)
        long = result.loc[result.player.eq("Cameron Davis")].iloc[0]
        thin = result.loc[result.player.eq("Zachary Bauchou")].iloc[0]
        missing = result.loc[result.player.eq("No Data")].iloc[0]
        self.assertAlmostEqual(long["sg_approach"], 0.7 * 50 / 62)
        self.assertAlmostEqual(thin["sg_approach"], 0.5 * 4 / 16)
        self.assertLess(thin["model_data_confidence"], long["model_data_confidence"])
        self.assertTrue(pd.isna(missing["sg_approach"]))
        self.assertEqual(float(missing["model_data_confidence"]), 0.0)
        self.assertEqual(long["driving_distance"], 300.0)  # Not shrunk as if SG.
        self.assertIn("T10",long["rrg_recent_finishes"])

    def test_bad_salary_and_collision_fail_closed(self):
        invalid = self.csv.copy()
        invalid.loc[0,"Salary"] = 9000
        with self.assertRaises(ValueError):
            normalize_rrg_strokes_gained(invalid, self.dk)
        repeated = pd.concat([self.csv,self.csv.iloc[[0]]],ignore_index=True)
        with self.assertRaises(ValueError):
            normalize_rrg_strokes_gained(repeated, self.dk)

    def test_reject_missing_sg_columns(self):
        with self.assertRaises(ValueError):
            normalize_rrg_strokes_gained(self.csv.drop(columns=["SG APP"]),self.dk)



class MexicoCourseEvidenceTests(unittest.TestCase):
    def test_preserves_chart_values_not_prediction_weights(self):
        evidence=mexico_course_evidence()
        values=evidence.set_index("Statistic")["Chart association"]
        self.assertAlmostEqual(float(values["SG: Total"]), 0.437)
        self.assertAlmostEqual(float(values["SG: Approach"]), 0.354)
        self.assertAlmostEqual(float(values["Approach: 175–200 yards"]), 0.277)
        self.assertAlmostEqual(float(values["Approach: 200–225 yards"]), 0.240)
        self.assertAlmostEqual(float(values["Driving Accuracy %"]), 0.005)
        self.assertAlmostEqual(float(values["SG: Around-the-Green"]), 0.063)
        self.assertEqual(
            evidence.loc[evidence["Statistic"].eq("Approach: 175–200 yards"),
                         "Displayed comparison rank"].iat[0],
            "#1/6",
        )

    def test_unavailable_distance_bucket_is_labeled(self):
        evidence=mexico_course_evidence()
        long_iron=evidence[evidence["Statistic"].str.contains("175–200|200–225",regex=True)]
        self.assertEqual(len(long_iron),2)
        self.assertTrue(long_iron["L50 golfer data"].eq("Not present in L50").all())
        thesis=mexico_course_thesis()
        self.assertIn("not direct weight percentages",thesis["guardrail"])

    def test_unified_course_dna_page_displays_evidence(self):
        app=(ROOT/"sports/pga/legacy/app.py").read_text(encoding="utf-8")
        self.assertIn("mexico_course_evidence()",app)
        self.assertIn("pga_mexico_2026_course_evidence.csv",app)
        self.assertIn("not an automated",app)


class MinimumMeasuredRoundsTests(unittest.TestCase):
    def test_lock_with_missing_sg_is_blocked(self):
        rows = []
        for i in range(12):
            rows.append({
                "player": f"Golfer {i}", "salary": 7500,
                "win_pct": 0.01+(i/1000), "top10_pct": 0.07+(i/1000),
                "make_cut_pct": 0.55, "dk_points_proxy": 60+i,
                "v11_mean_dk": 60+i, "v11_p75_dk": 80+i,
                "v11_p90_dk": 100+i, "rrg_sg_rounds": 0 if i==0 else 50,
                "model_data_confidence": 0 if i==0 else 50/62,
            })
        with self.assertRaisesRegex(ValueError, "lack adequate current SG data"):
            optimize_portfolio(pd.DataFrame(rows),
                PortfolioSettings(lineup_count=1,min_measured_sg_rounds=12),
                locks=["Golfer 0"])

    def test_pga_setup_has_native_sg_input(self):
        app=(ROOT/"sports/pga/legacy/app.py").read_text(encoding="utf-8")
        self.assertIn("normalize_rrg_strokes_gained(pd.read_csv(rrg_sg_up),players)",app)
        self.assertIn("sg_stats.copy()",app)
        self.assertIn("rrg_fit",app)
        self.assertIn("attach_source_comparison(pred,rrg_fit,otis_fit)",app)


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


    def test_feature_veto_excludes_finau_only_from_five_dollar_lineup(self):
        portfolio, golfers = self.sample_portfolio(count=3)
        # Add Finau to the strongest model lineup; other lines remain valid.
        portfolio.loc[portfolio["lineup"].eq(2) & portfolio["player"].eq("Golfer 2-0"),"player"]="Tony Finau"
        projections = golfers[["player"]].assign(v11_p90_dk=100)
        projections = pd.concat([
            projections, pd.DataFrame([{"player":"Tony Finau","v11_p90_dk":100}])
        ],ignore_index=True)
        summary = pd.DataFrame({
            "lineup":[1,2,3],
            "sum_dk_proxy":[420,485,430],
            "avg_make_cut_pct":[0.55,0.80,0.55],
            "sum_top10_pct":[0.5,1.8,0.7],
        })
        # Unrestricted model chooses the highest-scoring Finau lineup.
        unfiltered, winner = choose_featured_lineup(summary,portfolio,projections)
        self.assertEqual(winner,2)
        # Restricted $5 assignment chooses the best remaining lineup instead.
        filtered, winner = choose_featured_lineup(
            summary,portfolio,projections,feature_exclusions=["Tony Finau"]
        )
        self.assertNotEqual(winner,2)
        self.assertFalse(bool(filtered.loc[filtered.lineup.eq(2),"feature_eligible"].iat[0]))
        self.assertEqual(
            filtered.loc[filtered.lineup.eq(2),"contest_allocation"].iat[0],"$0.50 GPP"
        )
        self.assertIn("Tony Finau",portfolio["player"].tolist())
        self.assertEqual(filtered.contest_allocation.eq("$5 SINGLE ENTRY").sum(),1)

    def test_every_feature_candidate_vetoed_fails_closed(self):
        portfolio, golfers=self.sample_portfolio(count=3)
        predictions=golfers[["player"]].assign(v11_p90_dk=100)
        summary=pd.DataFrame({
            "lineup":[1,2,3],
            "sum_dk_proxy":[420,485,430],
            "avg_make_cut_pct":[0.55,0.80,0.55],
            "sum_top10_pct":[0.5,1.8,0.7]
        })
        # All twelve+ players vetoed; no silent override permitted.
        with self.assertRaisesRegex(ValueError,"No eligible lineup for the"):
            choose_featured_lineup(
                summary,portfolio,predictions,feature_exclusions=golfers.player.tolist()
            )

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

    def test_mexico_routing_and_assisted_export_warning(self):
        app=(ROOT/"sports/pga/legacy/app.py").read_text(encoding="utf-8")
        self.assertIn(
            "official_mexico_pars=[4,3,5,4,3,5,3,5,4,4,4,5,4,4,3,4,3,4]",
            app,
        )
        self.assertIn("hole_pars=official_mexico_pars",app)
        self.assertIn("OTIS-ASSISTED RUN",app)
        self.assertIn("pga_otis_assisted_comparison_NOT_INDEPENDENT.csv",app)

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
