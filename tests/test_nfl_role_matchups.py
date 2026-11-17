"""Every matchup must use only past player statistics and past role ranks."""
import unittest
import pandas as pd
from ai2_nfl_role_matchups import (
    defense_role_profiles, role_observations, slate_role_report)


def example():
    rows=[]
    people=[
        ("Alpha","AAA","WR",10,12,120),
        ("Beta","AAA","WR",5,6,40),
        ("Gamma","AAA","WR",3,4,20),
        ("Quarterback","AAA","QB",32,0,0),
        ("Runner1","AAA","RB",18,3,30),
        ("Runner2","AAA","RB",9,2,20),
        ("Tight","AAA","TE",4,4,35),
        ("BackupTight","AAA","TE",2,3,20),
        ("OtherAlpha","BBB","WR",7,8,70),
    ]
    for week in range(1,6):
        for name,team,pos,volume,receptions,receiving_yards in people:
            r=dict(
                season=2025,week=week,season_type="REG",
                player_display_name=name,player_id=name,
                recent_team=team,position=pos,
                opponent_team=("MIA" if team=="AAA" else "DAL") if week==4 else "NYJ",
                passing_yards=245 if pos=="QB" else 0,
                passing_tds=2 if pos=="QB" else 0,
                passing_interceptions=0,
                attempts=volume if pos=="QB" else 0,
                carries=volume if pos=="RB" else 0,
                rushing_yards=volume*4 if pos=="RB" else 0,
                rushing_tds=0,
                targets=volume if pos in ("WR","TE") else
                        receptions if pos=="RB" else 0,
                receptions=receptions if pos!="QB" else 0,
                receiving_yards=receiving_yards if pos!="QB" else 0,
                receiving_tds=0,
            )
            if week==4 and team=="AAA":
                if name=="Alpha":
                    r["receiving_yards"]=190
                if name=="Beta":
                    r["receiving_yards"]=10
                if name=="Tight":
                    r["receiving_yards"]=95
            rows.append(r)
    return pd.DataFrame(rows)


class RoleOpponentMatchupTests(unittest.TestCase):
    def test_previous_weeks_calculate_wr1_wr2_and_te1_relative_to_self(self):
        h=example()
        p,details=defense_role_profiles(
            h,2025,5,min_observations=1)
        mia=p[p.defense.eq("MIA")].set_index("role")
        self.assertAlmostEqual(float(mia.loc["WR1","raw_excess_dk"]),7.)
        self.assertAlmostEqual(float(mia.loc["WR2","raw_excess_dk"]),-3.)
        self.assertAlmostEqual(float(mia.loc["TE1","raw_excess_dk"]),6.)
        self.assertGreater(int(mia.loc["QB1","n"]),0)
        self.assertGreater(int(mia.loc["RB1","n"]),0)
        alpha=details[(details.player.eq("Alpha")) &
                      details.defense.eq("MIA")].iloc[0]
        beta=details[(details.player.eq("Beta")) &
                      details.defense.eq("MIA")].iloc[0]
        self.assertEqual(alpha["role"],"WR1")
        self.assertEqual(beta["role"],"WR2")
        self.assertEqual(alpha["prior_games"],3)
        self.assertEqual(beta["prior_games"],3)

    def test_looking_at_week_four_cannot_use_week_four_matchup(self):
        h=example()
        before,_=defense_role_profiles(h,2025,4,min_observations=1)
        tampered=h.copy()
        tampered.loc[tampered.week.eq(4),"receiving_yards"]=1000
        tampered.loc[tampered.week.eq(5),"receiving_yards"]=5000
        after,_=defense_role_profiles(tampered,2025,4,min_observations=1)
        pd.testing.assert_frame_equal(before,after)
        self.assertFalse((before.defense=="MIA").any())

    def test_pregame_rank_not_based_on_current_week_target_spike(self):
        h=example()
        h.loc[(h.week.eq(4)) & h.player_display_name.eq("Beta"),"targets"]=30
        observations=role_observations(h,2025,5)
        beta=observations[(observations.week==4)&
                          (observations.player=="Beta")].iloc[0]
        self.assertEqual(beta["role"],"WR2")

    def test_slate_rows_flag_insufficient_history_and_show_raw_delta(self):
        h=example()
        p,_=defense_role_profiles(h,2025,5,min_observations=1)
        slate=[
            {"ID":"1","Name":"Alpha","Position":"WR",
             "TeamAbbrev":"AAA","Salary":"8000","Status":"",
             "Game Info":"AAA@MIA 1:00PM ET"},
            {"ID":"2","Name":"Beta","Position":"WR",
             "TeamAbbrev":"AAA","Salary":"6000","Status":"",
             "Game Info":"AAA@MIA 1:00PM ET"},
            {"ID":"3","Name":"Out","Position":"WR",
             "TeamAbbrev":"AAA","Salary":"5000","Status":"OUT",
             "Game Info":"AAA@MIA 1:00PM ET"},
            {"ID":"4","Name":"NoData","Position":"WR",
             "TeamAbbrev":"AAA","Salary":"4000","Status":"",
             "Game Info":"AAA@MIA 1:00PM ET"},
        ]
        rows=slate_role_report(slate,h,2025,5,p)
        self.assertEqual(len(rows),4)
        by_name={x["Player"]:x for x in rows}
        self.assertEqual(by_name["Alpha"]["Role"],"WR1")
        self.assertEqual(by_name["Beta"]["Role"],"WR2")
        self.assertEqual(by_name["Alpha"]["Defense versus role (raw DK above own average)"],7.)
        self.assertEqual(by_name["Beta"]["Defense versus role (raw DK above own average)"],-3.)
        self.assertIsNone(by_name["Out"]["Defense versus role (shrunk DK)"])
        self.assertEqual(by_name["NoData"]["Role"],"")

    def test_official_out_excludes_player_from_current_role_rank(self):
        h=example()
        p,_=defense_role_profiles(h,2025,5,min_observations=1)
        slate=[
            {"ID":"1","Name":"Alpha","Position":"WR","TeamAbbrev":"AAA",
             "Salary":"8000","Status":"","Game Info":"AAA@MIA 1:00PM ET"},
            {"ID":"2","Name":"Beta","Position":"WR","TeamAbbrev":"AAA",
             "Salary":"6000","Status":"","Game Info":"AAA@MIA 1:00PM ET"},
        ]
        rows=slate_role_report(slate,h,2025,5,p,
                               official_status_by_id={"1":"OUT"})
        by_name={r["Player"]:r for r in rows}
        self.assertEqual(by_name["Alpha"]["Role"],"")
        self.assertEqual(by_name["Alpha"]["Slate status"],"OUT")
        self.assertEqual(by_name["Beta"]["Role"],"WR1")

    def test_last_season_only_is_not_current_wr1_evidence(self):
        h=example()
        rows=slate_role_report([
            {"ID":"1","Name":"Alpha","Position":"WR","TeamAbbrev":"AAA",
             "Salary":"8000","Status":"",
             "Game Info":"AAA@MIA 11/22/2026 1:00PM ET"}],
            h,2026,11,pd.DataFrame())
        self.assertEqual(rows[0]["Role"],"")
        self.assertIsNone(rows[0]["Player pregame DK baseline"])

    def test_missing_opponent_rejected_and_duplicate_rejected(self):
        h=example()
        with self.assertRaisesRegex(ValueError,"opponent_team"):
            defense_role_profiles(h.drop(columns="opponent_team"),2025,5)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            defense_role_profiles(pd.concat([h,h.iloc[[0]]]),2025,5)


if __name__=="__main__":
    unittest.main()
