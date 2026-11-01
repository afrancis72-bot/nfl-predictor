"""Status matching integration test with unrelated public-data providers isolated.

The goal is official OUT and DK Q eligibility, not team-pace or historical
schema normalization. Those inputs have their own dedicated validation tests.
"""
import unittest
from unittest.mock import patch
import pandas as pd
from ai2_nfl_auto_intelligence import collect_automatic_baselines


class ResearchTest(unittest.TestCase):
    @patch("ai2_nfl_auto_intelligence.make_rates")
    @patch("ai2_nfl_auto_intelligence.build_historical_baselines")
    def test_injury_status(self, mocked_baseline, mocked_rates):
        mocked_rates.return_value = pd.DataFrame([
            {"player": name, "team": "T", "position": "RB",
             "carries_pg": 5.0, "targets_pg": 1.0}
            for name in ("A", "B", "C")
        ])
        mocked_baseline.return_value = pd.DataFrame([
            {"ID": "1", "Name": "A", "Team": "T", "Position": "RB",
             "DK Status": "", "match_status": "matched"},
            {"ID": "2", "Name": "B", "Team": "T", "Position": "RB",
             "DK Status": "", "match_status": "matched"},
            {"ID": "3", "Name": "C", "Team": "T", "Position": "RB",
             "DK Status": "Q", "match_status": "matched"},
        ])
        saved = (
            "dk.csv",
            b"ID,Name,TeamAbbrev,Position\n1,A,T,RB\n2,B,T,RB\n3,C,T,RB\n",
            "2026-10-31T20:00:00-04:00",
        )
        # The prior test passed a fabricated one-column nflverse frame. That
        # broke when genuine team-pace and absence readers were added. Mock
        # those providers so this test exclusively covers injury eligibility.
        with (
            patch("ai2_nfl_auto_intelligence.predict_team_environment",
                  return_value={}),
            patch("ai2_nfl_auto_intelligence.predict_snap_roles",
                  return_value={}),
            patch("ai2_nfl_auto_intelligence.player_role_trends",
                  return_value=[]),
            patch("ai2_nfl_auto_intelligence.estimate_injury_presence",
                  return_value=({}, [])),
            patch("ai2_nfl_auto_intelligence.defensive_allowance",
                  return_value=pd.DataFrame()),
            patch("ai2_nfl_auto_intelligence.reserve_players_for_slate",
                  return_value=([], [], [])),
        ):
            result = collect_automatic_baselines(
                saved, 2026, 8, history=pd.DataFrame({"isolated": [1]}),
                snap_history=pd.DataFrame(),
                roster_history=pd.DataFrame([{"season": 2026, "week": 8}]),
                injury_fetcher=lambda: ({"b": "OUT"}, {"source": "mock NFL report"}),
            )

        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["excluded_count"], 1)
        self.assertEqual(result["questionable_count"], 1)
        self.assertFalse(result["release_ready"])
        status_by_id = dict(zip(result["players"]["ID"],result["players"]["current_status"]))
        self.assertEqual(status_by_id, {"1": "", "2": "OUT", "3": "Q"})


if __name__ == "__main__":
    unittest.main()
