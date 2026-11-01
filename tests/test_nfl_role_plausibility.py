"""Review observed usage divergence without vetoing user-selected NFL players."""
import unittest
import pandas as pd
from ai2_nfl_role_plausibility import compare_simulated_role_to_history


class RolePlausibilityTests(unittest.TestCase):
    def fixtures(self):
        history=pd.DataFrame([
            dict(player="Puka Nacua",team="LAR",position="WR",games_played=7,
                 source="nflverse weekly stats",targets_pg=9.,carries_pg=.1,
                 receiving_tds_pg=.5,rushing_tds_pg=0),
            dict(player="Bijan Robinson",team="ATL",position="RB",games_played=7,
                 source="nflverse weekly stats",targets_pg=5.,carries_pg=15.,
                 receiving_tds_pg=.1,rushing_tds_pg=.5),
            dict(player="Well Matched Player",team="NYG",position="WR",
                 games_played=7,source="nflverse weekly stats",
                 targets_pg=7.,carries_pg=0,receiving_tds_pg=.4,
                 rushing_tds_pg=0),
        ])
        audit=[
            dict(Player="Puka Nacua",Team="LAR",Position="WR",
                 **{"Audit status":"Verified: per-simulation DK points match components",
                    "Mean targets":17., "Mean carries":0.,
                    "Mean receiving TD":1.0,"Mean rushing TD":0}),
            dict(Player="Bijan Robinson",Team="ATL",Position="RB",
                 **{"Audit status":"Verified: per-simulation DK points match components",
                    "Mean targets":5.4,"Mean carries":22.,
                    "Mean receiving TD":0.1,"Mean rushing TD":1.1}),
            dict(Player="Well Matched Player",Team="NYG",Position="WR",
                 **{"Audit status":"Verified: per-simulation DK points match components",
                    "Mean targets":8.,"Mean carries":0.,
                    "Mean receiving TD":.5,"Mean rushing TD":0}),
        ]
        injury=[
            {"Team":"LAR","Unavailable player":"Davante Adams",
             "Additional redistribution %":20.,"Already reflected share":.8},
            {"Team":"ATL","Unavailable player":"Brian Robinson Jr.",
             "Additional redistribution %":15.,"Already reflected share":.85},
        ]
        return history,audit,injury

    def test_puka_bijan_flags_with_actual_observed_comparisons(self):
        history,audit,injury=self.fixtures()
        rows=compare_simulated_role_to_history(audit,history,injury)
        names={x["Player"] for x in rows}
        self.assertEqual(names,{"Puka Nacua","Bijan Robinson"})
        self.assertIn("Targets significantly",next(x["Concern"] for x in rows
                        if x["Player"]=="Puka Nacua" and x["Concern"].startswith("Targets")))
        self.assertTrue(any(x["Concern"].startswith("Carries")
                            for x in rows if x["Player"]=="Bijan Robinson"))
        self.assertTrue(any("touchdowns" in x["Concern"]
                            for x in rows if x["Player"]=="Bijan Robinson"))
        puka=next(x for x in rows if x["Player"]=="Puka Nacua")
        self.assertIn("Davante Adams",puka["Current-week injury context"])
        self.assertIn("20.0%",puka["Current-week injury context"])
        self.assertTrue(all("not automatically" in x["Required review"] or
                            "don't assume" in x["Required review"] for x in rows))

    def test_normal_unchanged_roles_not_flagged(self):
        history,audit,injury=self.fixtures()
        rows=compare_simulated_role_to_history(audit,history,injury)
        self.assertNotIn("Well Matched Player",{x["Player"] for x in rows})

    def test_missing_ambiguous_or_unverified_history_cannot_create_flag(self):
        history,audit,injury=self.fixtures()
        bad=history.copy()
        bad.loc[bad["player"]=="Puka Nacua","source"]="unknown model"
        rows=compare_simulated_role_to_history(audit,bad,injury)
        self.assertNotIn("Puka Nacua",{x["Player"] for x in rows})
        duplicate=pd.concat([history,history.iloc[[0]]],ignore_index=True)
        rows=compare_simulated_role_to_history(audit,duplicate,injury)
        self.assertNotIn("Puka Nacua",{x["Player"] for x in rows})
        with self.assertRaisesRegex(ValueError,"provenance"):
            compare_simulated_role_to_history(
                audit,history.drop(columns=["source"]),injury)

    def test_receiving_and_rushing_yard_efficiency_flag_point_inflation(self):
        history, audits, injury = self.fixtures()
        history.loc[history["player"].eq("Puka Nacua"), "receiving_yards_pg"] = 90.
        history.loc[history["player"].eq("Bijan Robinson"), "rushing_yards_pg"] = 75.
        # Fixtures model an exaggerated per-opportunity result. These are NOT
        # actual current-week projections or claims about either player.
        audits[0]["Mean targets"] = 11.
        audits[0]["Mean receiving yards"] = 170.
        audits[1]["Mean carries"] = 20.
        audits[1]["Mean rushing yards"] = 150.
        rows = compare_simulated_role_to_history(audits, history, injury)
        puka = [r for r in rows if r["Player"] == "Puka Nacua"
                and r["Concern"].startswith("Receiving yards per target")]
        bijan = [r for r in rows if r["Player"] == "Bijan Robinson"
                 and r["Concern"].startswith("Rushing yards per carry")]
        self.assertEqual(len(puka), 1)
        self.assertEqual(len(bijan), 1)
        self.assertEqual(puka[0]["Observed yards/opportunity"], 10.)
        self.assertGreater(puka[0]["Simulated yards/opportunity"], 15.)
        self.assertEqual(bijan[0]["Observed yards/opportunity"], 5.)
        self.assertIn("do not automatically", bijan[0]["Required review"])

    def test_efficiency_without_valid_observed_yards_does_not_invent_baseline(self):
        history, audits, injury = self.fixtures()
        audits[0]["Mean receiving yards"] = 200.
        audits[1]["Mean rushing yards"] = 160.
        rows = compare_simulated_role_to_history(audits, history, injury)
        self.assertFalse(any("yards per" in row["Concern"].lower() for row in rows))
        # A four-game sample must not be treated as sufficiently established
        # efficiency, even with high simulated per-target yardage.
        history.loc[history["player"].eq("Puka Nacua"), "receiving_yards_pg"] = 50.
        history.loc[history["player"].eq("Puka Nacua"), "games_played"] = 4
        rows = compare_simulated_role_to_history(audits, history, injury)
        self.assertFalse(any(row["Player"] == "Puka Nacua"
                             and "yards per" in row["Concern"].lower() for row in rows))

    def test_role_report_does_not_mutate_predictions_or_selection(self):
        history,audit,injury=self.fixtures()
        before=history.copy(deep=True)
        points_before=[dict(row) for row in audit]
        compare_simulated_role_to_history(audit,history,injury)
        pd.testing.assert_frame_equal(history,before)
        self.assertEqual(audit,points_before)


if __name__=="__main__":
    unittest.main()
