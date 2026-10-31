"""Real weekly reserve status means off-slate players can be included safely."""
import unittest
import pandas as pd
from ai2_nfl_roster_status import reserve_players_missing_from_slate


class RosterReserveTests(unittest.TestCase):
    def setUp(self):
        self.salary=[{"Name":"Puka Nacua","TeamAbbrev":"LAR"},
                     {"Name":"Bijan Robinson","TeamAbbrev":"ATL"}]
        self.rates=pd.DataFrame([
            {"player":"Davante Adams","team":"LAR","position":"WR"},
            {"player":"Brian Robinson Jr.","team":"ATL","position":"RB"}])
        self.roster=pd.DataFrame([
            {"season":2026,"week":9,"game_type":"REG","team":"LAR",
             "full_name":"Davante Adams","position":"WR","status":"RES"},
            {"season":2026,"week":9,"game_type":"REG","team":"ATL",
             "full_name":"Brian Robinson Jr.","position":"RB","status":"RES"}])
    def test_target_week_reserve_source_includes_both_off_salary(self):
        result=reserve_players_missing_from_slate(
            self.roster,self.salary,self.rates,2026,9)
        self.assertEqual({r["player"] for r in result},
                         {"Davante Adams","Brian Robinson Jr."})
        self.assertTrue(all(r["week"]==9 for r in result))
    def test_stale_snapshot_or_future_status_not_applied(self):
        self.assertEqual(reserve_players_missing_from_slate(
            self.roster,self.salary,self.rates,2026,8),[])
        self.assertEqual(reserve_players_missing_from_slate(
            self.roster,self.salary,self.rates,2026,10),[])
    def test_active_player_and_teams_not_on_slate_never_excluded(self):
        r=self.roster.copy()
        r.loc[0,"status"]="ACT"
        r.loc[1,"team"]="NYG"
        self.assertEqual(reserve_players_missing_from_slate(
            r,self.salary,self.rates,2026,9),[])
    def test_player_present_on_salary_is_not_double_counted(self):
        salaries=self.salary+[{"Name":"Davante Adams","TeamAbbrev":"LAR"}]
        result=reserve_players_missing_from_slate(
            self.roster,salaries,self.rates,2026,9)
        self.assertEqual([r["player"] for r in result],["Brian Robinson Jr."])
    def test_rejects_missing_observation_provenance(self):
        with self.assertRaisesRegex(ValueError,"explicit"):
            reserve_players_missing_from_slate(
                self.roster.drop(columns=["status"]),self.salary,
                self.rates,2026,9)


if __name__=="__main__":
    unittest.main()
