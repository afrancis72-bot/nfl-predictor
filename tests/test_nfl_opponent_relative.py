"""Chronological and talent-adjusted opponent quality regression tests."""
import unittest
import pandas as pd

from ai2_nfl_opponent_relative import (
    SOURCE, opponent_relative_profiles, prior_player_baselines_for_grading)


def fixture():
    rows = []
    players = [("Elite", "AAA", 10, 100, "DAL"),
               ("Regular", "BBB", 5, 50, "DAL"),
               ("Underrated", "CCC", 5, 50, "NYJ"),
               ("Emerging", "DDD", 5, 50, "NYJ")]
    for week in (1, 2, 3, 4):
        for name, team, catches, yards, opponent in players:
            receiving = yards
            receptions = catches
            if week == 4 and opponent == "NYJ":
                receiving = 110
                receptions = 8
            rows.append({
                "season": 2025, "week": week, "season_type": "REG",
                "player_display_name": name, "player_id": name,
                "recent_team": team, "position": "WR",
                "opponent_team": ("CHI" if week < 4 else opponent),
                "rushing_yards": 0, "rushing_tds": 0,
                "receiving_yards": receiving, "receptions": receptions,
                "receiving_tds": 0,
            })
    return pd.DataFrame(rows)


class RelativeOpponentTests(unittest.TestCase):
    def test_elite_receiver_does_not_automatically_make_defense_weak(self):
        h = fixture()
        p = opponent_relative_profiles(
            h, 2025, 5, min_observations=2, defense_window=10)
        dal = p[(p.defense == "DAL") & p.position.eq("WR")].iloc[0]
        nyj = p[(p.defense == "NYJ") & p.position.eq("WR")].iloc[0]
        # Even though elite receiver had a much larger absolute DK score,
        # Dallas allowed no EXCESS versus either player's normal production.
        self.assertAlmostEqual(float(dal["raw_excess_dk"]), 0.0)
        self.assertGreater(float(nyj["raw_excess_dk"]), 5.0)
        self.assertEqual(dal["source"], SOURCE)
        self.assertGreater(dal["mean_actual_dk"], 10.0)
        self.assertLess(float(nyj["shrunk_excess_dk"]),
                        float(nyj["raw_excess_dk"]))

    def test_no_current_or_future_week_data_enters_pregame_defense_profile(self):
        h = fixture()
        first = opponent_relative_profiles(
            h, 2025, 4, min_observations=2)
        altered = h.copy()
        altered.loc[altered.week.eq(4), "receiving_yards"] = 1000
        later = pd.concat([altered, altered[altered.week.eq(4)].assign(
            week=5, receiving_yards=2000)], ignore_index=True)
        second = opponent_relative_profiles(
            later, 2025, 4, min_observations=2)
        pd.testing.assert_frame_equal(first, second)
        # The November 2025 result is not known to a pregame 2025 week 4 profile.
        self.assertFalse((first.defense == "DAL").any())

    def test_week_four_baseline_excludes_week_four_box_score(self):
        h = fixture()
        scored = prior_player_baselines_for_grading(h, 2025, 4)
        regular = scored[scored.player.eq("Regular")].iloc[0]
        self.assertAlmostEqual(regular["excess_dk"], 0.0)
        better = scored[scored.player.eq("Underrated")].iloc[0]
        self.assertGreater(better["excess_dk"], 5.0)
        # A repeat score of 110 yards in future game can't revise the past.
        h.loc[h.week.eq(4), "receiving_yards"] = 1000
        scored_again = prior_player_baselines_for_grading(h, 2025, 4)
        p = scored_again[scored_again.player.eq("Regular")].iloc[0]
        self.assertAlmostEqual(
            p["pregame_player_baseline_dk"],
            regular["pregame_player_baseline_dk"])

    def test_too_little_history_neutral_and_no_fabricated_stats(self):
        h = fixture()
        # All players have exactly two prior game observations at week 3.
        self.assertTrue(prior_player_baselines_for_grading(h, 2025, 3).empty)
        p = opponent_relative_profiles(
            h, 2025, 5, min_observations=500)
        self.assertTrue(p.empty)

    def test_duplicate_player_week_and_missing_opponent_fail_closed(self):
        h = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate player"):
            opponent_relative_profiles(pd.concat([h, h.iloc[[0]]]), 2025, 5)
        with self.assertRaisesRegex(ValueError, "opponent_team"):
            opponent_relative_profiles(h.drop(columns="opponent_team"), 2025, 5)


if __name__ == "__main__":
    unittest.main()
