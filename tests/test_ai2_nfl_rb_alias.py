"""Prevent team aliases from evading the RB uniqueness rule."""
import unittest
from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_feasibility import find_classic_lineup

def sample():
    positions = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "DST", "WR")
    teams = ("BUF", "JAC", "JAX", "MIA", "NYJ", "LAR", "KC", "TEN", "SF")
    return [{"ID": str(i), "Position": p, "TeamAbbrev": t,
             "Salary": 5500, "opponent": "OTHER"} for i, (p, t) in enumerate(zip(positions, teams))]

class TestTeamAliasConstraints(unittest.TestCase):
    def test_same_team_rb_alias_rejected_by_validator(self):
        issues = check_classic_lineup(sample())
        self.assertTrue(any("running backs" in issue for issue in issues))

    def test_same_team_rb_alias_rejected_by_search(self):
        self.assertIs(find_classic_lineup(sample()).feasible, False)

if __name__ == "__main__":
    unittest.main()
