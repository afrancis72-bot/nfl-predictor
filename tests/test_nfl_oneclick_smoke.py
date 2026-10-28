"""Offline full-path integration: one upload -> observed rates -> sim -> 20 DK lineups.

The fixtures are fabricated for software testing only. They are NEVER used
as actual NFL projections or exposed in the production build interface.
"""
import csv
import io
import unittest
from collections import Counter
from unittest.mock import patch

import pandas as pd
from ai2_nfl_oneclick import build_oneclick_portfolio
from ai2_nfl_lineup import check_classic_lineup

class OneClickSmokeTests(unittest.TestCase):
    def test_complete_20_lineup_portfolio_with_one_confirmed_out(self):
        matchups = [("ARI", "DAL"), ("LV", "NYJ"), ("KC", "DEN")]
        salary, observations = [], []
        index = 0
        for away, home in matchups:
            game = f"{away}@{home} 11/01/2026 01:00PM ET"
            for team in (away, home):
                roster = [("QB", 1, 7000), ("RB", 3, 6500),
                          ("WR", 3, 5500), ("TE", 2, 4500), ("DST", 1, 3200)]
                for pos, quantity, price in roster:
                    for k in range(quantity):
                        index += 1
                        name = f"{team} {pos} {k}"
                        status = "OUT" if team == "DAL" and pos == "RB" and k == 0 else ""
                        salary.append({"ID":str(index), "Name":name, "Position":pos,
                            "Salary":price, "TeamAbbrev":team, "Status":status, "Game Info":game})
                        if pos == "DST":
                            continue
                        for week in range(1,8):
                            stat = {"player_display_name":name,"player_name":name,
                                    "player_id":str(index),"recent_team":team,
                                    "position":pos, "season":2026,"week":week,"season_type":"REG",
                                    "attempts":34 if pos=="QB" else 0,
                                    "completions":23 if pos=="QB" else 0,
                                    "passing_yards":240 if pos=="QB" else 0,
                                    "passing_tds":1.5 if pos=="QB" else 0,
                                    "passing_interceptions":0.7 if pos=="QB" else 0,
                                    "sacks_suffered":2 if pos=="QB" else 0,
                                    "carries":3 if pos=="QB" else 15 if pos=="RB" else 0,
                                    "rushing_yards":18 if pos=="QB" else 65 if pos=="RB" else 0,
                                    "rushing_tds":0.2 if pos=="QB" else 0.5 if pos=="RB" else 0,
                                    "targets":3 if pos=="RB" else 7 if pos=="WR" else 4 if pos=="TE" else 0,
                                    "receptions":2 if pos=="RB" else 5 if pos=="WR" else 3 if pos=="TE" else 0,
                                    "receiving_yards":20 if pos=="RB" else 70 if pos=="WR" else 37 if pos=="TE" else 0,
                                    "receiving_tds":0.1 if pos=="RB" else 0.35 if pos=="WR" else 0.2 if pos=="TE" else 0}
                            observations.append(stat)
        writer_buffer = io.StringIO()
        writer = csv.DictWriter(writer_buffer,fieldnames=list(salary[0]))
        writer.writeheader()
        writer.writerows(salary)
        saved = ("test_slate.csv",writer_buffer.getvalue().encode(),"2026-10-28T22:00:00+00:00")
        history = pd.DataFrame(observations)
        with patch("ai2_nfl_auto_intelligence.fetch_weekly",return_value=history), patch(
             "ai2_nfl_auto_intelligence.fetch_official_injury_statuses",
             return_value=({},{"source":"test fixture"})):
            result = build_oneclick_portfolio(saved,seed=42)
        self.assertEqual(len(result["summary"]),20)
        self.assertEqual(len(list(csv.reader(io.StringIO(result["csv"])))),21)
        exposure = Counter()
        for choice in result["ranked"]:
            roster = choice["candidate"]["players"]
            self.assertEqual(check_classic_lineup(roster,salary_floor=49600),[])
            self.assertNotIn("DAL RB 0",[p["Name"] for p in roster])
            exposure.update(p["ID"] for p in roster)
        self.assertLessEqual(max(exposure.values()),7)

if __name__=="__main__":
    unittest.main()
