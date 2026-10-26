"""Pure NBA minutes-based prop baseline checks; no live data needed."""
import unittest
from sports.nba.prop_model import PlayerRates, project_props


class NBAFoundationTests(unittest.TestCase):
    def setUp(self):
        self.sample = PlayerRates(
            player="Test Player", team="LAL", minutes_per_game=30,
            points_per_36=27, rebounds_per_36=9,
            assists_per_36=6, threes_per_36=3, sample_games=12
        )

    def test_minute_scaling(self):
        result = project_props(self.sample, 24)
        self.assertEqual(result["points"], 18.0)
        self.assertEqual(result["rebounds"], 6.0)
        self.assertEqual(result["assists"], 4.0)
        self.assertEqual(result["threes"], 2.0)
        self.assertEqual(result["points_rebounds_assists"], 28.0)

    def test_does_not_use_book_line(self):
        result = project_props(self.sample, 36)
        self.assertEqual(result["points"], 27)
        self.assertIn("Baseline only", result["limitations"])

    def test_rejects_impossible_minutes(self):
        with self.assertRaises(ValueError):
            project_props(self.sample, 50)


if __name__ == "__main__":
    unittest.main()
