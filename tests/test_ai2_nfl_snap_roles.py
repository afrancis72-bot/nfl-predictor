"""True pregame offensive-snap scoring and lineage regression tests."""
import unittest
import numpy as np
import pandas as pd
from ai2_nfl_snap_roles import (fetch_snap_counts, normalize_snap_counts,
                                 predict_snap_roles, apply_snap_roles_to_rates)
from dfs_stats_engine import simulate_stat_driven_dfs
from ai2_nfl_input_integrity import input_lineage_rows


def weekly_data():
    rows = []
    for wk in range(1, 7):
        for team in ("AAA", "BBB"):
            rows.append(dict(season=2026, week=wk, season_type="REG",
                             player_name="Runner "+team, position="RB",
                             recent_team=team, carries=12+wk,
                             targets=4+wk%2, receptions=3, rushing_yards=55,
                             receiving_yards=30))
            rows.append(dict(season=2026, week=wk, season_type="REG",
                             player_name="Wide "+team, position="WR",
                             recent_team=team, targets=5+wk,
                             carries=0, receptions=4, receiving_yards=70))
    return pd.DataFrame(rows)


def snap_data():
    rows = []
    for wk in range(1, 7):
        for team in ("AAA", "BBB"):
            for name, position, pct in (("Runner "+team, "RB", 35+wk*6),
                                         ("Wide "+team, "WR", 75+wk*2)):
                rows.append(dict(season=2026, week=wk, game_type="REG",
                                 player=name, team=team, position=position,
                                 offense_snaps=int(pct*.65), offense_pct=pct))
    return pd.DataFrame(rows)


def sim_fixture():
    rows, rates = [], []
    for team in ("AAA", "BBB"):
        for position, stem, salary in (("QB", "Q", 6000), ("RB", "R", 5500),
                                       ("WR", "W", 5200), ("TE", "T", 4300),
                                       ("DST", "D", 3000)):
            name=stem+team
            rows.append(dict(ID=team+position,Name=name,Position=position,
                             TeamAbbrev=team,Salary=salary,game="AAA@BBB",
                             proj=15.,optimizer_eligible=True))
            if position == "DST":
                continue
            r=dict(player=name,team=team,position=position,games_played=6,
                   source="nflverse weekly stats",
                   pass_attempts_pg=0.,completions_pg=0.,pass_yards_pg=0.,
                   passing_tds_pg=0.,interceptions_pg=0.,sacks_suffered_pg=0.,
                   carries_pg=0.,rushing_yards_pg=0.,rushing_tds_pg=0.,
                   targets_pg=0.,receptions_pg=0.,receiving_yards_pg=0.,
                   receiving_tds_pg=0.)
            if position == "QB":
                r.update(pass_attempts_pg=32.,completions_pg=21.,pass_yards_pg=235.,
                         passing_tds_pg=1.4,interceptions_pg=.7,sacks_suffered_pg=2.,
                         carries_pg=3.,rushing_yards_pg=15.)
            elif position == "RB":
                r.update(carries_pg=13.,rushing_yards_pg=50.,
                         rushing_tds_pg=.4,targets_pg=3.,receptions_pg=2.1,
                         receiving_yards_pg=18.)
            elif position == "WR":
                r.update(targets_pg=7.,receptions_pg=4.4,
                         receiving_yards_pg=62.,receiving_tds_pg=.4)
            else:
                r.update(targets_pg=4.,receptions_pg=3.,
                         receiving_yards_pg=32.,receiving_tds_pg=.25)
            rates.append(r)
    return pd.DataFrame(rows),pd.DataFrame(rates)


class SnapRolesTests(unittest.TestCase):
    def test_only_prior_games_no_target_week_contamination(self):
        current = predict_snap_roles(weekly_data(), snap_data(), 2026, 6)
        stats = weekly_data()
        snaps = snap_data()
        added = dict(season=2026,week=6,season_type="REG",
                     player_name="Runner AAA",position="RB",
                     recent_team="AAA",carries=100,targets=99)
        contaminated = predict_snap_roles(pd.concat([stats,pd.DataFrame([added])],
                                  ignore_index=True),snaps,2026,6)
        self.assertEqual(current, contaminated)

    def test_fewer_than_four_observed_snap_games_excluded(self):
        self.assertEqual(predict_snap_roles(weekly_data(),snap_data().query("week <= 3"),2026,7),{})

    def test_recent_snap_share_and_historical_snap_targets_used(self):
        profiles=predict_snap_roles(weekly_data(),snap_data(),2026,7)
        p=profiles[("runner aaa","AAA","RB")]
        self.assertEqual(p["snap_games"],6)
        self.assertGreater(p["recent_snap_pct"],p["earlier_snap_pct"])
        self.assertGreater(p["expected_carries"],0)
        self.assertFalse(p["routes_verified"])

    def test_snap_adjustment_changes_actual_game_simulations(self):
        pool,rates=sim_fixture()
        players,a,astats=simulate_stat_driven_dfs(pool,rates,n_sims=200,seed=19,
                                                   allow_role_proxies=False)
        profile={("waaa","AAA","WR"):{
            "expected_targets":12.,"expected_carries":0.,
            "recent_snap_pct":92.,"earlier_snap_pct":70.,
            "snap_games":5,"blend_weight":.6}}
        players,b,bstats=simulate_stat_driven_dfs(
            pool,rates,n_sims=200,seed=19,allow_role_proxies=False,
            snap_profiles=profile)
        self.assertFalse(np.array_equal(a,b))
        self.assertGreater(bstats["__SNAP_OPPORTUNITY__|AAA"]["scored_player_count"],0)
        self.assertEqual(astats["__SNAP_OPPORTUNITY__|AAA"]["scored_player_count"],0)
        changes=bstats["__SNAP_OPPORTUNITY__|AAA"]["observed_role_adjustments"]
        self.assertGreater(changes[0]["target_after"],changes[0]["target_before"])

    def test_no_unsubstantiated_snap_or_route_use(self):
        res=input_lineage_rows({"snap_profiles":{}})
        labels={x["Information"]:x["Data use"] for x in res}
        self.assertEqual(labels["Previous games' observed offensive snap counts and player opportunity"],"UNAVAILABLE")
        self.assertEqual(labels["Verified routes run, targets per route, and current-week depth-chart roles"],"NOT_CONNECTED")

    def test_normalizes_fractional_percentages(self):
        snap=snap_data()
        snap["offense_pct"]=snap["offense_pct"]/100.
        s=normalize_snap_counts(snap,2026,7)
        self.assertGreater(s["offense_pct"].median(),30.)

    def test_same_team_identity_required(self):
        s=snap_data()
        s.loc[s["team"].eq("AAA"),"team"]="CCC"
        result=predict_snap_roles(weekly_data(),s,2026,7)
        self.assertNotIn(("runner aaa","AAA","RB"),result)


if __name__=="__main__":
    unittest.main()
