"""DraftKings matchup and verified-opponent regression tests."""
import unittest
from ai2_nfl_matchups import opponent_from_game_info
from ai2_nfl_inputs import inspect_saved_nfl_inputs
from ai2_nfl_eligibility import eligible_nfl_rows

def saved(text):
    return ("fixture.csv", text.encode("utf-8"), "2026-10-22T20:00:00+00:00")

class TestMatchups(unittest.TestCase):
    def test_away_and_home(self):
        self.assertEqual(opponent_from_game_info("BUF", "BUF@MIA 1:00PM ET"), "MIA")
        self.assertEqual(opponent_from_game_info("MIA", "BUF@MIA 1:00PM ET"), "BUF")

    def test_bad_matchup_fails(self):
        self.assertIsNone(opponent_from_game_info("NYJ", "BUF@MIA 1:00PM ET"))
        self.assertIsNone(opponent_from_game_info("BUF", "TBD"))
        self.assertIsNone(opponent_from_game_info("BUF", "BUF@BUF 1:00PM ET"))

    def test_ingestion_overrides_projection_opponent(self):
        salaries = saved("ID,Name,Position,TeamAbbrev,Salary,Status,Game Info\n1,Quarterback,QB,BUF,7000,,BUF@MIA 1:00PM ET\n")
        projections = saved("ID,proj,projection_source,weekly_projection_valid,opportunity_verified,opponent\n1,20,weekly,true,true,NYJ\n")
        rows, issues = inspect_saved_nfl_inputs(salaries, projections)
        self.assertEqual(issues, [])
        self.assertEqual(rows[0]["opponent"], "MIA")
        self.assertEqual(len(eligible_nfl_rows(rows)), 1)

    def test_unknown_qb_opponent_excluded(self):
        salaries = saved("ID,Name,Position,TeamAbbrev,Salary,Status,Game Info\n1,Quarterback,QB,BUF,7000,,UNKNOWN\n")
        projections = saved("ID,proj,projection_source,weekly_projection_valid,opportunity_verified\n1,20,weekly,true,true\n")
        rows, issues = inspect_saved_nfl_inputs(salaries, projections)
        self.assertEqual(issues, [])
        self.assertEqual(eligible_nfl_rows(rows), [])

if __name__ == "__main__":
    unittest.main()
