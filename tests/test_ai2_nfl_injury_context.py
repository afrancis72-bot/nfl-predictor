"""Regression checks for pregame absence-conditioned NFL simulation."""
import unittest
import numpy as np
import pandas as pd

from ai2_nfl_injury_context import estimate_injury_presence, official_off_slate_out
from dfs_stats_engine import simulate_stat_driven_dfs


def make_weekly():
    rows=[]
    for week in range(1,9):
        for team in ("AAA","BBB"):
            rows.extend([
                dict(season=2026,week=week,season_type="REG",recent_team=team,
                     player_name="Primary QB "+team,position="QB",attempts=33,
                     carries=3,targets=0),
                dict(season=2026,week=week,season_type="REG",recent_team=team,
                     player_name="Star Runner "+team,position="RB",carries=19,
                     targets=5),
                dict(season=2026,week=week,season_type="REG",recent_team=team,
                     player_name="Star Receiver "+team,position="WR",
                     carries=0,targets=11),
            ])
            if week <= 5:
                rows.extend([
                    dict(season=2026,week=week,season_type="REG",recent_team=team,
                         player_name="Missing Runner "+team,position="RB",
                         carries=10,targets=2),
                    dict(season=2026,week=week,season_type="REG",recent_team=team,
                         player_name="Missing Receiver "+team,position="WR",
                         carries=0,targets=8),
                ])
    return pd.DataFrame(rows)


def sim_fixture():
    pool=[];rates=[]
    for team in ("AAA","BBB"):
        for position,name,salary in (("QB","Primary QB "+team,6200),
                                     ("RB","Star Runner "+team,5900),
                                     ("WR","Star Receiver "+team,6500),
                                     ("TE","Target TE "+team,3800),
                                     ("DST","Defense "+team,2900)):
            pool.append(dict(ID=team+position,Name=name,Position=position,
                             TeamAbbrev=team,Salary=salary,game="AAA@BBB",
                             proj=18.,optimizer_eligible=True))
            if position=="DST":
                continue
            rate=dict(player=name,team=team,position=position,games_played=6,
                      source="nflverse weekly stats",pass_attempts_pg=0.,
                      completions_pg=0.,pass_yards_pg=0.,passing_tds_pg=0.,
                      interceptions_pg=0.,sacks_suffered_pg=0.,carries_pg=0.,
                      rushing_yards_pg=0.,rushing_tds_pg=0.,targets_pg=0.,
                      receptions_pg=0.,receiving_yards_pg=0.,receiving_tds_pg=0.)
            if position=="QB":
                rate.update(pass_attempts_pg=32.,completions_pg=20.,
                            pass_yards_pg=225.,passing_tds_pg=1.45,
                            interceptions_pg=.75,sacks_suffered_pg=2.,
                            carries_pg=3.,rushing_yards_pg=14.)
            elif position=="RB":
                rate.update(carries_pg=17.,rushing_yards_pg=72.,rushing_tds_pg=.5,
                            targets_pg=5.,receptions_pg=4.,receiving_yards_pg=35.)
            elif position=="WR":
                rate.update(targets_pg=10.,receptions_pg=7.,receiving_yards_pg=92.,
                            receiving_tds_pg=.65)
            elif position=="TE":
                rate.update(targets_pg=4.,receptions_pg=3.,
                            receiving_yards_pg=33.,receiving_tds_pg=.25)
            rates.append(rate)
        for position,name,car,targ in (("RB","Missing Runner "+team,10.,2.),
                                       ("WR","Missing Receiver "+team,0.,8.)):
            rates.append(dict(player=name,team=team,position=position,
                              games_played=5,source="nflverse weekly stats",
                              pass_attempts_pg=0.,carries_pg=car,targets_pg=targ))
    return pd.DataFrame(pool),pd.DataFrame(rates)


class InjuryContextTests(unittest.TestCase):
    def test_absence_already_in_last_three_games_reduces_reallocation(self):
        out=[dict(player="Missing Receiver AAA",team="AAA"),
             dict(player="Missing Runner AAA",team="AAA")]
        context,report=estimate_injury_presence(make_weekly(),out,2026,9)
        for key in (("missing receiver aaa","AAA"),("missing runner aaa","AAA")):
            self.assertLess(context[key]["share"],.70)
            self.assertGreater(context[key]["share"],0.)
            self.assertEqual(context[key]["status"],"ALREADY_ABSENT_IN_RECENT_GAMES")
        self.assertEqual(len(report),2)
        self.assertEqual(report[0]["Last team game week"],8)

    def test_does_not_use_upcoming_week_or_next_season(self):
        out=[dict(player="Missing Receiver AAA",team="AAA")]
        base=estimate_injury_presence(make_weekly(),out,2026,8)
        later=pd.concat([make_weekly(),pd.DataFrame([
            dict(season=2027,week=1,player_name="Missing Receiver AAA",
                 recent_team="AAA",position="WR",targets=30),
            dict(season=2026,week=9,player_name="Missing Receiver AAA",
                 recent_team="AAA",position="WR",targets=30)])],ignore_index=True)
        self.assertEqual(base,estimate_injury_presence(later,out,2026,8))

    def test_missing_all_prior_weeks_means_no_incremental_injury_boost(self):
        h=make_weekly()
        h=h[~h["player_name"].eq("Missing Receiver AAA")]
        p,_=estimate_injury_presence(
            h,[{"player":"Missing Receiver AAA","team":"AAA"}],2026,9)
        self.assertEqual(p[("missing receiver aaa","AAA")]["share"],0.)

    def test_no_guess_out_from_missing_salary(self):
        rates=pd.DataFrame([
            dict(player="Missing Receiver AAA",team="AAA",position="WR")])
        salaries=[{"Name":"Star Receiver AAA","TeamAbbrev":"AAA"}]
        self.assertEqual(official_off_slate_out([],salaries,rates),[])
        self.assertEqual(official_off_slate_out(
            [dict(name="Missing Receiver AAA",team="AAA",position="WR",
                  game_status="QUESTIONABLE")],salaries,rates),[])
        actual=official_off_slate_out(
            [dict(name="Missing Receiver AAA",team="AAA",position="WR",
                  game_status="OUT")],salaries,rates)
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]["team"],"AAA")

    def test_injury_budget_scored_once_and_conserved(self):
        pool,rates=sim_fixture()
        outs=[dict(player="Missing Receiver AAA",team="AAA"),
              dict(player="Missing Runner AAA",team="AAA")]
        historical,_=estimate_injury_presence(make_weekly(),outs,2026,9)
        _,baseline,_=simulate_stat_driven_dfs(
            pool,rates,n_sims=300,seed=142,allow_role_proxies=False,
            out_players=outs,injury_presence={})
        _,conditioned,details=simulate_stat_driven_dfs(
            pool,rates,n_sims=300,seed=142,allow_role_proxies=False,
            out_players=outs,injury_presence=historical)
        _,legacy,legacy_details=simulate_stat_driven_dfs(
            pool,rates,n_sims=300,seed=142,allow_role_proxies=False,
            out_players=outs)
        self.assertFalse(np.array_equal(conditioned,legacy))
        self.assertFalse(np.array_equal(conditioned,baseline))
        injury=details["__OPPORTUNITY__|AAA"]
        self.assertEqual(injury["out_context_method"],"PREGAME_ABSENCE_CONDITIONED")
        self.assertLess(injury["removed_targets_pg"],
                        injury["historical_out_targets_if_always_active_pg"])
        self.assertLess(injury["removed_carries_pg"],
                        injury["historical_out_carries_if_always_active_pg"])
        self.assertGreater(injury["removed_targets_pg"],0.)
        self.assertAlmostEqual(injury["target_budget_before"],injury["target_budget_after"])
        self.assertAlmostEqual(injury["carry_budget_before"],injury["carry_budget_after"])
        self.assertGreater(legacy_details["__OPPORTUNITY__|AAA"]["removed_targets_pg"],
                           injury["removed_targets_pg"])


if __name__=="__main__":
    unittest.main()
