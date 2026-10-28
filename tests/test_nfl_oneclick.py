import unittest
from ai2_nfl_oneclick import build_oneclick_portfolio

class OneClickTest(unittest.TestCase):
    def test_missing_salary_and_projection_does_not_fabricate(self):
        salary_csv=b"ID,Name,Position,Salary,Game Info,TeamAbbrev\n1,Example,QB,6500,DAL@ARI 11/01/2026 01:00PM ET,DAL\n"
        saved=("salary.csv",salary_csv,"2026-10-27T22:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "automated VERIFIED Week 8"):
            build_oneclick_portfolio(saved,None,seed=42)
    def test_seed_validation(self):
        with self.assertRaisesRegex(ValueError, "Seed"):
            build_oneclick_portfolio(None,None,seed=-1)

if __name__=="__main__":
    unittest.main()
