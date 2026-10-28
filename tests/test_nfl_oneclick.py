import unittest
from unittest.mock import patch
from ai2_nfl_oneclick import build_oneclick_portfolio, _slate_week

class OneClickTests(unittest.TestCase):
    def setUp(self):
        csv = b"ID,Name,Position,Salary,Game Info,TeamAbbrev\\n1,Example,QB,6500,DAL@ARI 11/01/2026 01:00PM ET,DAL\\n"
        self.saved = ("salary.csv", csv.replace(b"\\n", b"\n"), "2026-10-27T22:00:00+00:00")

    def test_week_from_dk_slate(self):
        self.assertEqual(_slate_week(self.saved), (2026, 8))

    @patch("ai2_nfl_auto_intelligence.collect_automatic_baselines")
    def test_auto_research_not_manual_upload(self, fetch):
        fetch.return_value = {"matched_count": 100, "missing_count": 10,
                              "blockers": ["Injuries unverified", "DST unverified"]}
        with self.assertRaisesRegex(ValueError, "Automatic NFL research completed"):
            build_oneclick_portfolio(self.saved, None, seed=42)
        fetch.assert_called_once()

    def test_invalid_seed(self):
        with self.assertRaisesRegex(ValueError, "Seed"):
            build_oneclick_portfolio(self.saved, None, seed=-1)

if __name__ == "__main__":
    unittest.main()
