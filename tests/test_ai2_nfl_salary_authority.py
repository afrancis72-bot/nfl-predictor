"""Regression tests for salary ID uniqueness and defense normalization."""
import unittest
from ai2_nfl_inputs import inspect_saved_nfl_inputs

def saved(text):
    return ("fixture.csv", text.encode("utf-8"), "2026-10-22T20:00:00+00:00")

class TestSalaryAuthority(unittest.TestCase):
    def test_duplicate_salary_ids_block(self):
        salaries = saved("ID,Name,Position,TeamAbbrev,Salary,Status\n1,A,QB,BUF,7000,\n1,B,QB,MIA,6000,\n")
        projections = saved("ID,proj,projection_source,weekly_projection_valid,opportunity_verified\n1,20,weekly,true,true\n")
        rows, issues = inspect_saved_nfl_inputs(salaries, projections)
        self.assertEqual(rows, [])
        self.assertTrue(any("duplicated" in reason for reason in issues))

    def test_defense_label_normalized(self):
        salaries = saved("ID,Name,Position,TeamAbbrev,Salary,Status,Game Info\n1,Defense,D/ST,BUF,3000,,BUF@MIA 1:00PM ET\n")
        projections = saved("ID,proj,projection_source,weekly_projection_valid,opportunity_verified\n1,8,weekly,true,true\n")
        rows, issues = inspect_saved_nfl_inputs(salaries, projections)
        self.assertEqual(issues, [])
        self.assertEqual(rows[0]["Position"], "DST")

if __name__ == "__main__":
    unittest.main()
