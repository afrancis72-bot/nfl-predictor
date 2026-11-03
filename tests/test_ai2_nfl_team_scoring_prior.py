"""Team scoring evidence must come from past games, not salary or future scores."""
import unittest
import pandas as pd
from ai2_nfl_team_environment import measured_team_games, predict_team_environment


def history():
    rows=[]
    for week in range(1,8):
        for team,passing,rushing in (("DET",3,1),("DAL",2,1),("LV",1,0)):
            rows.append({
                "season":2026,"week":week,"season_type":"REG",
                "player_name":f"QB {team}","recent_team":team,
                "position":"QB","attempts":35,"sacks_suffered":2,
                "carries":3,"passing_tds":passing,"rushing_tds":0})
            rows.append({
                "season":2026,"week":week,"season_type":"REG",
                "player_name":f"RB {team}","recent_team":team,
                "position":"RB","attempts":0,"carries":24,
                "rushing_tds":rushing})
    return pd.DataFrame(rows)


class PregameTeamScoring(unittest.TestCase):
    def test_actual_scoring_offenses_have_finite_ordered_prior(self):
        result=predict_team_environment(history(),2026,8)
        self.assertEqual(set(result),{"DET","DAL","LV"})
        self.assertGreater(result["DET"]["expected_offensive_tds"],
                           result["DAL"]["expected_offensive_tds"])
        self.assertGreater(result["DAL"]["expected_offensive_tds"],
                           result["LV"]["expected_offensive_tds"])
        for row in result.values():
            self.assertTrue(row["team_scoring_source"].startswith("nflverse"))
            self.assertFalse("calibrated" in row["team_scoring_source"].lower()
                             and "not" not in row["team_scoring_source"].lower())
            self.assertEqual(row["team_scoring_observed_games"],6)
            self.assertAlmostEqual(
                min(4.8,row["expected_passing_tds"]+row["expected_rushing_tds"]),
                row["expected_offensive_tds"],delta=.001)

    def test_no_future_week_can_change_pregame_scoring_prior(self):
        first=predict_team_environment(history(),2026,7)
        altered=history()
        altered.loc[altered.week.ge(7),"passing_tds"]=20
        altered.loc[altered.week.ge(7),"rushing_tds"]=20
        second=predict_team_environment(altered,2026,7)
        self.assertEqual(first,second)

    def test_missing_td_fields_do_not_create_fictitious_scoring(self):
        original=history().drop(columns=["passing_tds","rushing_tds"])
        result=predict_team_environment(original,2026,8)
        self.assertTrue(result)
        self.assertTrue(all("expected_offensive_tds" not in row for row in result.values()))

    def test_actual_offensive_touchdown_count_avoids_double_counted_receiver_td(self):
        raw=history()
        raw["receiving_tds"]=50
        games=measured_team_games(raw,2026,8)
        self.assertEqual(games.loc[(games.team=="DET")&(games.week==1),"passing_tds"].iloc[0],3.)
        self.assertEqual(games.loc[(games.team=="DET")&(games.week==1),"rushing_tds"].iloc[0],1.)


if __name__=="__main__":
    unittest.main()
