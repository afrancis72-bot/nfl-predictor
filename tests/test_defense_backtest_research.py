"""Unit tests for pregame defense priors and held-out evaluation."""
import unittest
import numpy as np
import pandas as pd
from defensive_priors_research import build_defense_profiles
from backtest_research import evaluate_predictions

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

if __name__=="__main__":
    unittest.main()
