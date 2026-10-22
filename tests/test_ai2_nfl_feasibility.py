"""Feasibility search regression tests."""
import unittest
from ai2_nfl_feasibility import find_classic_lineup

def pool():
    positions = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "DST", "WR")
    return [{"ID": str(i), "Name": str(i), "Position": p,
             "TeamAbbrev": "T" + str(i), "Salary": 5500, "opponent": "OTHER"}
            for i, p in enumerate(positions)]

class TestFeasibility(unittest.TestCase):
    def test_finds_legal_lineup(self):
        result = find_classic_lineup(pool())
        self.assertIs(result.feasible, True)
        self.assertEqual(len(result.lineup), 9)

    def test_missing_position(self):
        rows = pool()
        rows[0]["Position"] = "WR"
        result = find_classic_lineup(rows)
        self.assertIs(result.feasible, False)

    def test_salary_infeasible(self):
        rows = pool()
        for row in rows:
            row["Salary"] = 3000
        result = find_classic_lineup(rows)
        self.assertIs(result.feasible, False)

    def test_budget_exhaustion_not_false(self):
        rows = pool()
        for row in rows:
            row["Salary"] = 3000
        result = find_classic_lineup(rows, max_checks=1)
        self.assertIsNone(result.feasible)

    def test_same_team_rb_rejected(self):
        rows = pool()
        rows[2]["TeamAbbrev"] = rows[1]["TeamAbbrev"]
        result = find_classic_lineup(rows)
        self.assertIs(result.feasible, False)

if __name__ == "__main__":
    unittest.main()
