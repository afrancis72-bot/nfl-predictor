"""End-to-end NFL Classic ingestion-to-feasibility regression (no exports)."""
import unittest
from ai2_nfl_inputs import inspect_saved_nfl_inputs
from ai2_nfl_eligibility import eligible_nfl_rows
from ai2_nfl_feasibility import find_classic_lineup
from ai2_nfl_preflight import validate_pool

def saved(content):
    return ("test.csv", content.encode("utf-8"), "2026-10-22T20:00:00+00:00")

def fixtures():
    positions = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "D/ST", "WR"]
    teams = ["BUF", "KC", "MIA", "NYJ", "LAR", "SF", "DET", "TEN", "CHI"]
    salary = ["ID,Name,Position,TeamAbbrev,Salary,Status,Game Info"]
    projections = ["ID,proj,projection_source,weekly_projection_valid,opportunity_verified"]
    for i, (position, team) in enumerate(zip(positions, teams), 1):
        game = "BUF@MIA 1:00PM ET" if team in {"BUF", "MIA"} else team + "@DAL 1:00PM ET"
        salary.append(f"{i},Player{i},{position},{team},5500,,{game}")
        projections.append(f"{i},12,weekly,true,true")
    return saved("\n".join(salary)+"\n"), saved("\n".join(projections)+"\n")

class TestEndToEnd(unittest.TestCase):
    def test_ingest_to_diagnostic_only(self):
        salaries, projections = fixtures()
        rows, issues = inspect_saved_nfl_inputs(salaries, projections)
        self.assertEqual(issues, [])
        eligible = eligible_nfl_rows(rows)
        self.assertEqual(len(eligible), 9)
        self.assertEqual(rows[7]["Position"], "DST")
        self.assertEqual(rows[0]["opponent"], "MIA")
        feasibility = find_classic_lineup(eligible)
        self.assertIs(feasibility.feasible, True)
        self.assertEqual(len(feasibility.lineup), 9)
        preflight = validate_pool(rows, source_fresh=False,
                                  historical_validation_passed=False,
                                  lineup_feasibility_passed=feasibility.feasible is True)
        self.assertFalse(preflight.ready)
        self.assertTrue(any("freshness" in r for r in preflight.reasons))
        self.assertTrue(any("Historical" in r for r in preflight.reasons))

    def test_opposing_defense_is_not_legal(self):
        salaries, projections = fixtures()
        altered = salaries[1].decode().replace("8,Player8,D/ST,TEN", "8,Player8,D/ST,MIA")
        rows, issues = inspect_saved_nfl_inputs(saved(altered), projections)
        self.assertEqual(issues, [])
        self.assertIs(find_classic_lineup(eligible_nfl_rows(rows)).feasible, False)

if __name__ == "__main__":
    unittest.main()
