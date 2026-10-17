"""Unit tests for pregame defense priors and held-out evaluation."""
import unittest
import numpy as np
import pandas as pd
from defensive_priors_research import build_defense_profiles
from backtest_research import evaluate_predictions
from historical_replay_research import replay_game

class DefensePriorTests(unittest.TestCase):
    def test_future_week_cannot_change_profile(self):
        rows=[]
        for week in (1,2,3,6):
            for defense in ("SEA","DEN"):
                for play in range(220):
                    kind="pass" if play%2 else "run"
                    yards=(2 if defense=="DEN" else 9) if kind=="pass" else 4
                    if week==6: yards=1000
                    rows.append(dict(season=2026,week=week,season_type="REG",
                        defteam=defense,posteam="DEN" if defense=="SEA" else "SEA",
                        play_type=kind,yards_gained=yards,complete_pass=1 if kind=="pass" else 0))
        frame=pd.DataFrame(rows)
        a,_=build_defense_profiles(frame,2026,6)
        b,_=build_defense_profiles(frame[frame.week<6],2026,6)
        self.assertEqual(a,b)
        self.assertLess(a["DEN"]["pass_yards_factor"],a["SEA"]["pass_yards_factor"])

    def test_no_pregame_data_rejected(self):
        data=pd.DataFrame([dict(season=2026,week=6,defteam="SEA",
            posteam="DEN",play_type="pass",yards_gained=5)])
        with self.assertRaises(ValueError):
            build_defense_profiles(data,2026,6)

class BacktestTests(unittest.TestCase):
    def test_keyed_quantiles_and_metrics(self):
        simulations=[]
        for name,vals in (("A",[10,20,30,40]),("B",[100,120,140,160])):
            for sim,val in enumerate(vals):
                simulations.append(dict(matchup_id="g1",player=name,stat="pass_yd",
                    simulation=sim,value=val))
        actuals=pd.DataFrame([dict(matchup_id="g1",player="B",stat="pass_yd",actual=130),
                              dict(matchup_id="g1",player="A",stat="pass_yd",actual=25)])
        joined,metrics=evaluate_predictions(pd.DataFrame(simulations),actuals)
        self.assertEqual(len(joined),2)
        self.assertAlmostEqual(joined.set_index("player").loc["A","median"],25)
        self.assertAlmostEqual(joined.set_index("player").loc["B","median"],130)
        self.assertEqual(metrics.loc[0,"observations"],2)

class ReplayIntegrationTests(unittest.TestCase):
    def test_end_to_end_replay_with_pregame_data(self):
        players=[]
        for team in ("SEA","DEN"):
            for role,rush,target in (("QB",2,0),("RB",12,2),("WR",0,8),("TE",0,4)):
                players.append(dict(Name=team+" "+role,TeamAbbrev=team,
                    Position=role,usage_verified=True,
                    rush_weight=rush,target_weight=target,
                    carries_per_game=rush,
                    **{"FLEX Salary":5000}))
        pbp=[]
        for week in (1,2,3):
            for team,opp in (("SEA","DEN"),("DEN","SEA")):
                for play in range(220):
                    pbp.append(dict(season=2025,week=week,season_type="REG",
                        defteam=team,posteam=opp,play_type="pass" if play%2 else "run",
                        yards_gained=6 if play%2 else 4,complete_pass=1 if play%3 else 0))
        actual=pd.DataFrame([
            dict(player="SEA QB",stat="pass_yd",actual=210,baseline=200),
            dict(player="DEN QB",stat="pass_yd",actual=190,baseline=200)])
        result=replay_game("2025_04_SEA_DEN",2025,4,["SEA","DEN"],
            pd.DataFrame(players),pd.DataFrame(pbp),actual,n_games=6,seed=17)
        self.assertEqual(len(result["details"]),2)
        self.assertIn("baseline_mae",result["metrics"].columns)
        self.assertEqual(result["predictions"].simulation.nunique(),6)

if __name__=="__main__":
    unittest.main()
