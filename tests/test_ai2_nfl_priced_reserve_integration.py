"""Integration: source-confirmed DK-pricelist IR excludes athlete from real NFL pool."""
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import pandas as pd

from ai2_nfl_auto_intelligence import collect_automatic_baselines


class ReserveOneClickIntegrationTests(unittest.TestCase):
    def inputs(self):
        csv=("ID,Name,Position,Salary,Game Info,TeamAbbrev,Status\n"
             "1,Davante Adams,WR,7100,LAR@ATL 11/01/2026 01:00PM ET,LAR,\n"
             "2,Puka Nacua,WR,8500,LAR@ATL 11/01/2026 01:00PM ET,LAR,\n"
             "3,Bijan Robinson,RB,8300,LAR@ATL 11/01/2026 01:00PM ET,ATL,\n")
        saved=("DKSalaries.csv",csv.encode(),"2026-10-31T13:00:00+00:00")
        baseline=pd.DataFrame([
            {"ID":"1","Name":"Davante Adams","Team":"LAR","Position":"WR",
             "DK Status":"","match_status":"matched"},
            {"ID":"2","Name":"Puka Nacua","Team":"LAR","Position":"WR",
             "DK Status":"","match_status":"matched"},
            {"ID":"3","Name":"Bijan Robinson","Team":"ATL","Position":"RB",
             "DK Status":"","match_status":"matched"},
        ])
        rate=pd.DataFrame([
            dict(player="Davante Adams",team="LAR",position="WR"),
            dict(player="Puka Nacua",team="LAR",position="WR"),
            dict(player="Bijan Robinson",team="ATL",position="RB"),
        ])
        history=pd.DataFrame([
            dict(season=2026,week=week,season_type="REG",recent_team=team,
                 player_name=name,position=pos,targets=targets,carries=carries)
            for week in range(1,9)
            for name,team,pos,targets,carries in (
               ("Davante Adams","LAR","WR",7,0),
               ("Puka Nacua","LAR","WR",12,0),
               ("Bijan Robinson","ATL","RB",5,19))
        ])
        return saved,baseline,rate,history

    def evaluate(self,roster):
        saved,baseline,rate,history=self.inputs()
        calls=[]
        def redistribute(rates,statuses):
            calls.append(statuses.copy())
            return rates.copy(),[]
        with ExitStack() as stack:
            for target,result in [
                ("ai2_nfl_auto_intelligence.build_historical_baselines",baseline),
                ("ai2_nfl_auto_intelligence.make_rates",rate),
                ("ai2_nfl_auto_intelligence.predict_team_environment",{}),
                ("ai2_nfl_auto_intelligence.predict_snap_roles",{}),
                ("ai2_nfl_auto_intelligence.player_role_trends",[]),
                ("ai2_nfl_auto_intelligence.match_injury_statuses",{}),
                ("ai2_nfl_auto_intelligence.defensive_allowance",
                 pd.DataFrame(columns=["defense","position","multiplier","player_game_observations"]))
            ]:
                stack.enter_context(patch(target,return_value=result))
            stack.enter_context(patch(
                "ai2_nfl_auto_intelligence.redistribute_opportunities",
                side_effect=redistribute))
            output=collect_automatic_baselines(
                saved,2026,9,history=history,snap_history=pd.DataFrame(),
                roster_history=roster,
                injury_fetcher=lambda:([],{"source":"NFL official injury report",
                                           "practice_rows":[],"checked_utc":"2026-10-31"}))
        return output,calls

    def test_priced_IR_blocks_player_and_shares_one_injury_context(self):
        roster=pd.DataFrame([
            dict(season=2026,week=9,game_type="REG",team="LAR",
                 full_name="Davante Adams",position="WR",status="RES"),
            dict(season=2026,week=9,game_type="REG",team="LAR",
                 full_name="Puka Nacua",position="WR",status="ACT"),
        ])
        output,calls=self.evaluate(roster)
        players=output["players"].set_index("Name")
        self.assertEqual(players.loc["Davante Adams","current_status"],"IR")
        self.assertFalse(bool(players.loc["Davante Adams","model_candidate"]))
        self.assertTrue(bool(players.loc["Davante Adams","excluded_for_status"]))
        self.assertTrue(bool(players.loc["Puka Nacua","model_candidate"]))
        self.assertEqual(output["excluded_count"],1)
        self.assertEqual(output["candidate_count"],2)
        self.assertEqual(len(output["confirmed_priced_reserve_out"]),1)
        self.assertEqual(len(output["confirmed_off_slate_out"]),0)
        self.assertIn(("davante adams","LAR"),output["injury_presence"])
        self.assertEqual(calls[-1][("Davante Adams","LAR")],"IR")

    def test_future_reserve_not_applied_to_prior_slate(self):
        roster=pd.DataFrame([
            dict(season=2026,week=10,game_type="REG",team="LAR",
                 full_name="Davante Adams",position="WR",status="RES"),
        ])
        output,calls=self.evaluate(roster)
        self.assertEqual(output["candidate_count"],3)
        self.assertEqual(output["confirmed_priced_reserve_out"],[])
        self.assertIn("not published",output["reserve_status_issue"])

    def test_missing_weekly_roster_does_not_invent_injury(self):
        roster=pd.DataFrame([
            dict(season=2026,week=8,game_type="REG",team="LAR",
                 full_name="Davante Adams",position="WR",status="RES"),
        ])
        output,calls=self.evaluate(roster)
        self.assertEqual(output["candidate_count"],3)
        self.assertIn("latest W8",output["reserve_status_issue"])
        self.assertEqual(output["confirmed_priced_reserve_out"],[])


if __name__=="__main__":
    unittest.main()
