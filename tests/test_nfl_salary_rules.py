import unittest
from ai2_nfl_eligibility import eligible_nfl_rows

class EligibilityTests(unittest.TestCase):
    def test_minimums_and_roles(self):
        def row(identifier, pos, salary, role=True):
            return dict(ID=identifier, Position=pos, Salary=salary, proj=10,
                TeamAbbrev="DAL", opponent="ARI", dk_status="",
                projection_source="verified", weekly_projection_valid=True,
                opportunity_verified=role, projected_targets=5, projected_touches=12)
        rows=[row("a","WR",3900),row("b","WR",4000),
              row("c","RB",4400),row("d","RB",4500),
              row("e","WR",4800,False)]
        self.assertEqual({p["ID"] for p in eligible_nfl_rows(rows)},{"b","d"})

if __name__=="__main__":
    unittest.main()
