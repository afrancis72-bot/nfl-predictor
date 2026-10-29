"""Opponent-injury evidence gates and leak-free defensive matchup hypotheses."""
import unittest
from bs4 import BeautifulSoup
import pandas as pd
from ai2_nfl_official_injuries import _table_team
from ai2_nfl_opponent_intelligence import make_opponent_report, report_summary


class OpponentIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.slate = [
            {"Name": "Example Receiver", "Position": "WR",
             "TeamAbbrev": "NYJ", "Game Info": "LV@NYJ 11/01/2026 01:00PM ET"},
            {"Name": "Example Raider", "Position": "WR",
             "TeamAbbrev": "LV", "Game Info": "LV@NYJ 11/01/2026 01:00PM ET"}]
        self.allowances = pd.DataFrame([
            {"defense": "LV", "position": "WR", "multiplier": 1.12,
             "player_game_observations": 40},
            {"defense": "NYJ", "position": "WR", "multiplier": .88,
             "player_game_observations": 30},
            {"defense": "LV", "position": "RB", "multiplier": 1.18,
             "player_game_observations": 5}])

    def test_away_defender_absence_is_not_a_free_wr_projection_boost(self):
        meta = {"source": "https://www.nfl.com/injuries/",
                "checked_utc": "2026-10-28T23:00:00+00:00",
                "practice_rows": [
                    {"team": "LV", "name": "Raiders Corner",
                     "position": "CB", "game_status": "OUT",
                     "practice_status": "Did Not Participate"},
                    {"team": "LV", "name": "Raiders Safety",
                     "position": "S", "game_status": "",
                     "practice_status": "Limited Participation"},
                    {"team": "NYJ", "name": "Jets Defensive Tackle",
                     "position": "DT", "game_status": "QUESTIONABLE",
                     "practice_status": "Did Not Participate"}]}
        data = make_opponent_report(self.slate, self.allowances, meta)
        jets_wr = next(x for x in data if x["Offense"] == "NYJ" and x["Position"] == "WR")
        self.assertEqual(jets_wr["Opposing defense"], "LV")
        self.assertEqual(jets_wr["Historical allowance index"], 1.12)
        self.assertIn("Raiders Corner", jets_wr["Defenders ruled out"])
        self.assertIn("Raiders Safety", jets_wr["Defenders to monitor"])
        self.assertEqual(jets_wr["Applied simulation boost"], 1.0)
        self.assertEqual(jets_wr["Coverage assignment"], "Unknown — no verified CB/receiver route map")
        jets_rb = next(x for x in data if x["Offense"] == "NYJ" and x["Position"] == "RB")
        self.assertIsNone(jets_rb["Historical allowance index"]) # sparse defense evidence
        raiders_rb = next(x for x in data if x["Offense"] == "LV" and x["Position"] == "RB")
        self.assertEqual(raiders_rb["Defenders ruled out"], "None verified")
        self.assertIn("Jets Defensive Tackle", raiders_rb["Defenders to monitor"])
        summary = report_summary(data)
        self.assertEqual(summary["offensive_teams"], 2)
        self.assertEqual(summary["opponent_injury_flags"], 7)

    def test_unknown_defensive_team_never_attributed_to_slate_opponent(self):
        meta = {"source": "official", "practice_rows": [
            {"team": "", "name": "Unknown Corner", "position": "CB",
             "game_status": "OUT", "practice_status": "DNP"}]}
        data = make_opponent_report(self.slate, self.allowances, meta)
        self.assertTrue(all(x["Defenders ruled out"] == "None verified" for x in data))

    def test_official_table_nearby_exact_team_label_only(self):
        html = """
          <div>Raiders</div><table>
          <thead><tr><th>Player</th><th>Position</th><th>Game Status</th></tr></thead>
          <tbody><tr><td>Corner</td><td>CB</td><td>Out</td></tr></tbody></table>
          <div>Jets</div><table>
          <thead><tr><th>Player</th><th>Position</th><th>Game Status</th></tr></thead>
          <tbody><tr><td>Linebacker</td><td>LB</td><td>Questionable</td></tr></tbody></table>
        """
        tables = BeautifulSoup(html, "html.parser").select("table")
        self.assertEqual(_table_team(tables[0]), "LV")
        self.assertEqual(_table_team(tables[1]), "NYJ")


if __name__ == "__main__":
    unittest.main()
