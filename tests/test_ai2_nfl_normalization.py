"""Integration tests for normalized NFL CSV verification flags."""
import unittest
from ai2_nfl_inputs import inspect_saved_nfl_inputs
from ai2_nfl_preflight import validate_pool

def saved(text):
    return ("fixture.csv", text.encode("utf-8"), "2026-10-22T20:00:00+00:00")

class TestNFLNormalization(unittest.TestCase):
    def test_true_flags_normalized(self):
        salaries = saved("ID,Name,Position,TeamAbbrev,Salary,Status\n1,Player,QB,AAA,7000,\n")
        projections = saved("ID,proj,projection_source,weekly_projection_valid,opportunity_verified\n1,20,Weekly model,true,TRUE\n")
        rows, issues = inspect_saved_nfl_inputs(salaries, projections)
        self.assertEqual(issues, [])
        self.assertIs(rows[0]["weekly_projection_valid"], True)
        self.assertIs(rows[0]["opportunity_verified"], True)

    def test_ambiguous_flag_not_verified(self):
        salaries = saved("ID,Name,Position,TeamAbbrev,Salary,Status\n1,Player,QB,AAA,7000,\n")
        projections = saved("ID,proj,projection_source,weekly_projection_valid,opportunity_verified\n1,20,Weekly model,yes,1\n")
        rows, issues = inspect_saved_nfl_inputs(salaries, projections)
        self.assertEqual(issues, [])
        self.assertIs(rows[0]["weekly_projection_valid"], False)
        self.assertIs(rows[0]["opportunity_verified"], False)
        check = validate_pool(rows, source_fresh=True, historical_validation_passed=True)
        self.assertFalse(check.ready)
        self.assertEqual(check.eligible_count, 0)

if __name__ == "__main__":
    unittest.main()
