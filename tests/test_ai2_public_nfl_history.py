"""Regression checks for public NFL history and pregame leakage."""
import unittest
import pandas as pd
from props_data import make_rates, fetch_weekly


class PublicHistoryTests(unittest.TestCase):
    def test_target_week_never_enters_rates(self):
        raw = pd.DataFrame([
            dict(season=2026, week=5, season_type="REG", player_name="Example Back",
                 recent_team="AAA", position="RB", carries=10, targets=2),
            dict(season=2026, week=6, season_type="REG", player_name="Example Back",
                 recent_team="AAA", position="RB", carries=14, targets=4),
            dict(season=2026, week=7, season_type="REG", player_name="Example Back",
                 recent_team="AAA", position="RB", carries=99, targets=99),
        ])
        result = make_rates(raw, 2026, 7)
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["games_played"]), 2)
        self.assertEqual(int(result.iloc[0]["latest_observed_week"]), 6)
        self.assertLess(float(result.iloc[0]["carries_pg"]), 15)

    def test_missing_player_ids_are_not_discarded(self):
        raw = pd.DataFrame([
            dict(season=2026, week=5, season_type="REG", player_id="gsis-1",
                 player_name="Player One", recent_team="AAA", position="RB",
                 carries=10, targets=2),
            dict(season=2026, week=5, season_type="REG", player_id=None,
                 player_name="Player Two", recent_team="BBB", position="WR",
                 carries=0, targets=5),
        ])
        rates = make_rates(raw, 2026, 7)
        self.assertEqual(set(rates["player"]), {"Player One", "Player Two"})

    def test_same_name_without_ids_does_not_merge_teams(self):
        raw = pd.DataFrame([
            dict(season=2026, week=5, season_type="REG", player_name="Same Name",
                 recent_team="AAA", position="RB", carries=10, targets=1),
            dict(season=2026, week=5, season_type="REG", player_name="Same Name",
                 recent_team="BBB", position="RB", carries=15, targets=2),
        ])
        rates = make_rates(raw, 2026, 7)
        self.assertEqual(set(rates["team"]), {"AAA", "BBB"})

    def test_empty_release_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Empty or invalid"):
            fetch_weekly([2026], reader=lambda _: pd.DataFrame())

    def test_missing_week_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing week"):
            fetch_weekly([2026], reader=lambda _: pd.DataFrame({"player": ["A"]}))

    def test_reader_frame_is_not_mutated(self):
        source = pd.DataFrame({"week": [1]})
        fetch_weekly([2026], reader=lambda _: source)
        self.assertNotIn("season", source.columns)

    def test_public_fetch_uses_requested_seasons(self):
        seen = []
        def reader(url):
            seen.append(url)
            return pd.DataFrame({"week": [1]})
        result = fetch_weekly([2026, 2025, 2026], reader=reader)
        self.assertEqual(len(seen), 2)
        self.assertTrue(any("2025" in url for url in seen))
        self.assertTrue(any("2026" in url for url in seen))
        self.assertEqual(set(result["season"]), {2025, 2026})


if __name__ == "__main__":
    unittest.main()
