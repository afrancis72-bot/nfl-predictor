"""Unit tests for NFL read-only ingestion and fail-closed preflight.

Run: python -m unittest discover -s tests -p 'test_ai2_nfl*.py'
"""
import unittest
from ai2_nfl_inputs import inspect_saved_nfl_inputs, parse_csv_payload
from ai2_nfl_preflight import validate_pool


def saved(csv_text):
    return ("sample.csv", csv_text.encode("utf-8"), "2026-10-22T20:00:00+00:00")


class TestNFLInputs(unittest.TestCase):
    def test_missing_files(self):
        rows, issues = inspect_saved_nfl_inputs(None, None)
        self.assertEqual(rows, [])
        self.assertEqual(len(issues), 2)

    def test_csv_rejects_duplicate_headers(self):
        with self.assertRaises(ValueError):
            parse_csv_payload(b"ID,ID\n1,2\n")

    def test_id_join_uses_salary_authority(self):
        salaries = saved("ID,Name,Position,TeamAbbrev,Salary,Status\n1,Actual,QB,ABC,7000,\n")
        projections = saved("ID,Name,proj,projection_source,weekly_projection_valid,opportunity_verified\n1,Wrong,20,verified,true,true\n")
        rows, issues = inspect_saved_nfl_inputs(salaries, projections)
        self.assertFalse(issues)
        self.assertEqual(rows[0]["Name"], "Actual")
        self.assertEqual(rows[0]["Salary"], "7000")

    def test_game_comes_from_authoritative_salary_matchup(self):
        salaries = saved("ID,Name,Position,TeamAbbrev,Salary,Status,Game Info\n1,Actual,QB,NYJ,7000,,NYJ@MIA 1:00PM ET\n")
        projections = saved("ID,proj,game,projection_source,weekly_projection_valid,opportunity_verified\n1,20,BUF@NE,verified,true,true\n")
        rows, issues = inspect_saved_nfl_inputs(salaries, projections)
        self.assertFalse(issues)
        self.assertEqual(rows[0]["game"], "NYJ@MIA")
        self.assertEqual(rows[0]["opponent"], "MIA")

    def test_mismatched_ids_block(self):
        salaries = saved("ID,Name\n1,A\n")
        projections = saved("ID,proj\n2,20\n")
        rows, issues = inspect_saved_nfl_inputs(salaries, projections)
        self.assertEqual(rows, [])
        self.assertTrue(any("absent" in item for item in issues))


class TestNFLPreflight(unittest.TestCase):
    def test_no_validation_never_ready(self):
        result = validate_pool([], source_fresh=False, historical_validation_passed=False)
        self.assertFalse(result.ready)

    def test_ppg_fallback_never_eligible(self):
        row = {"Name": "A", "Position": "QB", "TeamAbbrev": "ABC",
               "Salary": 6000, "proj": 22, "projection_source": "DK PPG fallback",
               "weekly_projection_valid": True, "opportunity_verified": True}
        result = validate_pool([row] * 9, source_fresh=True,
                               historical_validation_passed=True)
        self.assertEqual(result.eligible_count, 0)
        self.assertFalse(result.ready)

    def test_string_boolean_does_not_certify(self):
        row = {"Name": "A", "Position": "QB", "TeamAbbrev": "ABC",
               "Salary": 6000, "proj": 22, "projection_source": "weekly",
               "weekly_projection_valid": "true", "opportunity_verified": "true"}
        result = validate_pool([row] * 9, source_fresh=True,
                               historical_validation_passed=True)
        self.assertFalse(result.ready)


if __name__ == "__main__":
    unittest.main()
