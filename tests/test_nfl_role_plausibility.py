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

    def test_role_report_does_not_mutate_predictions_or_selection(self):
        history,audit,injury=self.fixtures()
        before=history.copy(deep=True)
        points_before=[dict(row) for row in audit]
        compare_simulated_role_to_history(audit,history,injury)
        pd.testing.assert_frame_equal(history,before)
        self.assertEqual(audit,points_before)


if __name__=="__main__":
    unittest.main()
