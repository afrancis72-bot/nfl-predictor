"""Synthetic leakage and matchup adjustment regression checks."""
import unittest
import numpy as np
import pandas as pd
from ai2_nfl_matchup_diagnostics import defensive_allowance, apply_matchup_diagnostics

def player_week(week, yards, opponent="MIA"):
    return {"season": 2026, "week": week, "season_type": "REG",
            "player_name": "Test QB", "recent_team": "NYJ", "opponent_team": opponent,
            "position": "QB", "passing_yards": yards, "passing_tds": 1,
            "passing_interceptions": 0, "rushing_yards": 10, "rushing_tds": 0}

class MatchupTests(unittest.TestCase):
    def test_future_week_does_not_change_pregame_factors(self):
        pre = pd.DataFrame([player_week(4, 200), player_week(5, 300)])
        with_future = pd.concat([pre, pd.DataFrame([player_week(7, 900)])], ignore_index=True)
        a = defensive_allowance(pre, 2026, 7)
        b = defensive_allowance(with_future, 2026, 7)
        self.assertAlmostEqual(float(a.iloc[0]["multiplier"]), float(b.iloc[0]["multiplier"]))

    def test_missing_opponent_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "opponent_team"):
            defensive_allowance(pd.DataFrame([player_week(4, 200)]).drop(columns=["opponent_team"]), 2026, 7)

    def test_provisional_matchup_does_not_certify_player(self):
        baselines = pd.DataFrame([{"opponent": "MIA", "Position": "QB",
                                  "historical_partial_dk": 20.0,
                                  "weekly_projection_valid": False,
                                  "opportunity_verified": False}])
        allowances = pd.DataFrame([{"defense": "MIA", "position": "QB", "multiplier": 1.1}])
        out = apply_matchup_diagnostics(baselines, allowances)
        self.assertAlmostEqual(float(out.iloc[0]["matchup_adjusted_partial_dk"]), 22)
        self.assertFalse(bool(out.iloc[0]["weekly_projection_valid"]))
        self.assertFalse(bool(out.iloc[0]["opportunity_verified"]))

if __name__ == "__main__":
    unittest.main()
