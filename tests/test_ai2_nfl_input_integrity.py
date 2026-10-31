"""Tests of NFL pregame feature lineage and role trend guardrails."""
import unittest
import pandas as pd
from ai2_nfl_input_integrity import player_role_trends, input_lineage_rows, lineup_role_warnings


def weekly_example(extra=None):
    rows = []
    for week, targets in [(1, 10), (2, 10), (3, 10), (4, 5), (5, 5)]:
        rows.append(dict(season=2026, week=week, season_type="REG", player_id="wr-1",
                         player_name="Example Receiver", player_display_name="Example Receiver",
                         recent_team="MIA", position="WR", targets=targets))
    rows.extend(extra or [])
    return pd.DataFrame(rows)


SALARY = [{"Name": "Example Receiver", "TeamAbbrev": "MIA", "Position": "WR"}]


class NFLIntegrityTests(unittest.TestCase):
    def test_declining_pregame_role_flagged(self):
        rows = player_role_trends(weekly_example(), SALARY, 2026, 6)
        self.assertEqual(rows[0]["Role trend"], "DECLINING")
        self.assertAlmostEqual(rows[0]["Recent 3 per game"], 6.67)
        self.assertAlmostEqual(rows[0]["Earlier up-to-5 per game"], 10.)
        self.assertEqual(rows[0]["Change %"], -33.3)

    def test_no_future_week_leak(self):
        later = dict(season=2026, week=6, season_type="REG", player_id="wr-1",
                     player_name="Example Receiver", recent_team="MIA", position="WR", targets=99)
        pre = player_role_trends(weekly_example(), SALARY, 2026, 6)[0]
        contaminated = player_role_trends(weekly_example([later]), SALARY, 2026, 6)[0]
        self.assertEqual(pre, contaminated)

    def test_same_team_only_and_no_imputed_zero(self):
        moved = weekly_example([dict(season=2026, week=5, season_type="REG", player_id="wr-1",
                         player_name="Example Receiver", recent_team="NYJ", position="WR", targets=100)])
        rows = player_role_trends(moved, SALARY, 2026, 6)
        self.assertEqual(rows[0]["Change %"], -33.3)
        short = player_role_trends(weekly_example().iloc[:4], SALARY, 2026, 6)
        self.assertEqual(short[0]["Role trend"], "INSUFFICIENT_HISTORY")
        missing = weekly_example()
        missing.loc[4, "targets"] = None
        uncertain = player_role_trends(missing, SALARY, 2026, 6)
        self.assertEqual(uncertain[0]["Role trend"], "INCOMPLETE_STATS")

    def test_research_only_signals_honestly_labeled(self):
        rows = input_lineage_rows({
            "opponent_allowances": pd.DataFrame([{"defense": "NYJ"}]),
            "injury_meta": {"source": "official-report"},
            "role_trends": [{"Player": "Somebody"}]})
        statuses = {r["Information"]: r["Data use"] for r in rows}
        self.assertEqual(statuses["Opponent fantasy points allowed by position"], "RESEARCH_ONLY")
        self.assertEqual(statuses["Opposing defender absences and coverage questions"], "RESEARCH_ONLY")
        self.assertEqual(statuses["Last-3 versus earlier same-team workload"], "RESEARCH_ONLY")
        self.assertEqual(statuses["Weather and wind effects"], "NOT_CONNECTED")

    def test_flags_only_selected_players(self):
        trend = player_role_trends(weekly_example(), SALARY, 2026, 6)
        lineups = [{"candidate": {"players": [{"Name": "Example Receiver", "TeamAbbrev": "MIA"}]}}]
        flagged = lineup_role_warnings(lineups, trend)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["Lineup"], 1)
        self.assertEqual(flagged[0]["Role trend"], "DECLINING")


if __name__ == "__main__":
    unittest.main()
