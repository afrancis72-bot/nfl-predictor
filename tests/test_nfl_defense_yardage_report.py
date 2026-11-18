"""Direct yardage descriptive report is not a speculative scoring adjustment."""
import unittest
import numpy as np
import pandas as pd
from ai2_nfl_defense_yardage_report import summarize_defense_yards


def fixture():
    data=[]
    # Against same defense: 4 out of 5 role-QB or WR opponents beat
    # their OWN pregame yardage expectation; other role is below.
    for w,(base,yards) in enumerate(
        [(70,95),(90,100),(100,124),(80,62),(110,150)],1):
        data.append({
            "season":2026,"week":w,"defense":"AAA",
            "role":"WR1","yardage_type":"receiving_yards","player":f"WR {w}",
            "own_pregame_mean_yards":base,"actual_yards":yards,
            "difference_yards":yards-base,
            "above_own_baseline":yards>base})
        data.append({
            "season":2026,"week":w,"defense":"AAA",
            "role":"WR2","yardage_type":"receiving_yards","player":f"WR2 {w}",
            "own_pregame_mean_yards":60.,"actual_yards":40.,
            "difference_yards":-20.,"above_own_baseline":False})
    return pd.DataFrame(data)


class DefenseYardMatrixTests(unittest.TestCase):
    def test_consistent_role_is_flagged_but_not_entire_defense(self):
        result=summarize_defense_yards(fixture())
        wr1=result[result.role.eq("WR1")].iloc[0]
        wr2=result[result.role.eq("WR2")].iloc[0]
        self.assertEqual(wr1["player_games"],5)
        self.assertEqual(wr1["above_baseline_pct"],80.)
        self.assertEqual(wr1["average_player_pregame_yards"],90.)
        self.assertEqual(wr1["average_allowed_yards"],106.2)
        self.assertTrue(wr1["repeated_above_baseline"])
        self.assertFalse(wr2["repeated_above_baseline"])
        self.assertTrue(wr1["description_only_no_prediction_claim"])

    def test_one_outlier_is_not_consistency(self):
        x=fixture().query("role == 'WR1'").copy()
        x.loc[x.index[1:],"actual_yards"]=12.
        x["difference_yards"]=x.actual_yards-x.own_pregame_mean_yards
        x["above_own_baseline"]=x.actual_yards>x.own_pregame_mean_yards
        z=summarize_defense_yards(x).iloc[0]
        self.assertFalse(z["repeated_above_baseline"])
        self.assertEqual(z["above_baseline_games"],1)

    def test_small_samples_are_not_signals(self):
        z=summarize_defense_yards(fixture().head(2))
        self.assertFalse(z.iloc[0]["enough_games"])
        self.assertFalse(z.iloc[0]["repeated_above_baseline"])

    def test_duplicate_observation_fails_closed(self):
        x=fixture()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            summarize_defense_yards(pd.concat([x,x.iloc[[0]]]))

    def test_missing_yards_fails_closed(self):
        x=fixture()
        x.loc[x.index[0],"actual_yards"]=np.nan
        with self.assertRaisesRegex(ValueError,"Missing"):
            summarize_defense_yards(x)


if __name__=="__main__":
    unittest.main()
