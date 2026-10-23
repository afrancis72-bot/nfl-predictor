"""DST historical indicator tests: no future-week leakage or fake verification."""
import unittest
import pandas as pd
from ai2_nfl_dst_diagnostics import historical_dst_indicators, attach_dst_indicators

def row(week, pos, picks=0, pass_tds=0, rush_tds=0):
    return {"season":2026, "week":week, "season_type":"REG",
            "player_name":"Example "+pos, "recent_team":"NYJ",
            "opponent_team":"MIA", "position":pos,
            "passing_interceptions":picks, "passing_tds":pass_tds,
            "rushing_tds":rush_tds}

class DstTests(unittest.TestCase):
    def test_future_week_excluded(self):
        past = pd.DataFrame([row(5,"QB",1,2),row(5,"RB",0,0,1)])
        future = pd.concat([past,pd.DataFrame([row(7,"QB",9,9)])],ignore_index=True)
        a = historical_dst_indicators(past,2026,7)
        b = historical_dst_indicators(future,2026,7)
        self.assertAlmostEqual(float(a.iloc[0]["opposing_interceptions_pg"]),
                               float(b.iloc[0]["opposing_interceptions_pg"]))
        self.assertFalse(bool(a.iloc[0]["complete_dst_projection"]))

    def test_attach_does_not_mark_dst_verified(self):
        hist = pd.DataFrame([row(5,"QB",1,2)])
        indicator = historical_dst_indicators(hist,2026,7)
        slate = pd.DataFrame([{"Team":"MIA","Position":"DST",
                               "weekly_projection_valid":False,
                               "opportunity_verified":False}])
        result = attach_dst_indicators(slate,indicator)
        self.assertEqual(result.iloc[0]["dst_games_observed"],1)
        self.assertFalse(bool(result.iloc[0]["weekly_projection_valid"]))

    def test_missing_opponent_fails_closed(self):
        hist = pd.DataFrame([row(5,"QB",1,2)]).drop(columns=["opponent_team"])
        with self.assertRaisesRegex(ValueError,"opponent_team"):
            historical_dst_indicators(hist,2026,7)

if __name__ == "__main__":
    unittest.main()
