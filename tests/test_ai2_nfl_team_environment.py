"""No-lookahead checks for reconstructed NFL team-level game volume."""
import unittest
import pandas as pd
from scripts.backtest_nfl_team_environment import observed_team_volumes, _estimate


def history():
    rows = []
    for w in range(1, 7):
        rows.append({"season": 2026, "week": w, "season_type": "REG",
                     "player_name": "Test Quarterback", "recent_team": "MIA",
                     "opponent_team": "NYJ", "position": "QB",
                     "attempts": 30 + w, "sacks_suffered": 2, "carries": 3})
        rows.append({"season": 2026, "week": w, "season_type": "REG",
                     "player_name": "Test Runningback", "recent_team": "MIA",
                     "opponent_team": "NYJ", "position": "RB",
                     "carries": 20, "targets": 3})
    return pd.DataFrame(rows)


class TeamEnvironmentTests(unittest.TestCase):
    def test_reconstructs_scrimmage_events_without_double_counting_passes(self):
        observed = observed_team_volumes(history())
        self.assertEqual(len(observed), 6)
        self.assertEqual(observed.iloc[0]["pass_attempts"], 31.)
        self.assertEqual(observed.iloc[0]["rush_attempts"], 23.)
        self.assertEqual(observed.iloc[0]["plays_proxy"], 56.)

    def test_next_week_not_used_in_prior(self):
        base = observed_team_volumes(history())
        p1 = _estimate(base, 2026, 6, "MIA", "plays_proxy", 1.0)
        altered = base.copy()
        altered.loc[altered.week.eq(6), "plays_proxy"] = 300.
        p2 = _estimate(altered, 2026, 6, "MIA", "plays_proxy", 1.0)
        self.assertAlmostEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
