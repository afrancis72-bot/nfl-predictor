import unittest
from unittest.mock import patch
from ai2_nfl_oneclick import build_oneclick_portfolio, _slate_week

class OneClickTests(unittest.TestCase):
    def setUp(self):
        csv = ("ID,Name,Position,Salary,Game Info,TeamAbbrev\\n"
               "1,Example,QB,6500,DAL@ARI 11/01/2026 01:00PM ET,DAL\\n")
        self.saved = ("salary.csv", csv.replace("\\n", "\n").encode(),
                      "2026-10-27T22:00:00+00:00")

    def test_week_from_dk_slate(self):
        self.assertEqual(_slate_week(self.saved), (2026, 8))

    @patch("ai2_nfl_oneclick.create_observed_model_pool")
    @patch("ai2_nfl_oneclick.collect_automatic_baselines")
    def test_auto_research_without_projection_upload(self, research, pool):
        research.return_value = {"blockers": [], "injury_meta": {"source": "test"}}
        pool.side_effect = ValueError("Not enough observed players")
        with self.assertRaisesRegex(ValueError, "Not enough observed players"):
            build_oneclick_portfolio(self.saved, None, seed=42)
        research.assert_called_once_with(self.saved, 2026, 8)
        pool.assert_called_once()

    def test_seed_bounds(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            build_oneclick_portfolio(self.saved, None, seed=-1)

    def test_seed_string_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            build_oneclick_portfolio(self.saved, None, seed="42")

if __name__ == "__main__":
    unittest.main()
