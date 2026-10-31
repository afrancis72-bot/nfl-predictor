"""Validate team-pace data truly changes NFL scoring while no-lookahead holds."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from ai2_nfl_team_environment import measured_team_games, predict_team_environment
from ai2_nfl_simulation_adapter import run_verified_stat_simulation
from dfs_stats_engine import simulate_stat_driven_dfs


def weekly():
    rows = []
    for wk in range(1, 7):
        for team, passes, rush in (("AAA", 42, 20), ("BBB", 25, 30)):
            rows.append(dict(season=2026, week=wk, season_type="REG",
                             player_name="Q"+team, recent_team=team, position="QB",
                             attempts=passes, completions=int(passes*.65),
                             passing_yards=passes*7, passing_tds=2, passing_interceptions=1,
                             sacks_suffered=2, carries=3, rushing_yards=15,
                             rushing_tds=0, targets=0))
            rows.append(dict(season=2026, week=wk, season_type="REG",
                             player_name="R"+team, recent_team=team, position="RB",
                             carries=rush, rushing_yards=rush*4,
                             rushing_tds=1, targets=4, receptions=3,
                             receiving_yards=25, receiving_tds=0))
    return pd.DataFrame(rows)


def sim_fixture():
    pool, rates = [], []
    for team in ("AAA", "BBB"):
        for pos, prefix, salary in (("QB", "Q", 6000), ("RB", "R", 5500),
                                    ("WR", "W", 5500), ("TE", "T", 4500),
                                    ("DST", "D", 3000)):
            name=prefix+team
            pool.append(dict(ID=team+pos, Name=name, Position=pos,
                             TeamAbbrev=team, Salary=salary, game="AAA@BBB",
                             proj=15., optimizer_eligible=True,
                             weekly_projection_valid=True, opportunity_verified=True))
            if pos=="DST":
                continue
            r=dict(player=name, team=team, position=pos, games_played=6,
                   latest_observed_season=2026, latest_observed_week=6,
                   source="nflverse weekly stats", pass_attempts_pg=0.,
                   completions_pg=0., pass_yards_pg=0., passing_tds_pg=0.,
                   interceptions_pg=0., sacks_suffered_pg=0.,
                   carries_pg=0., rushing_yards_pg=0., rushing_tds_pg=0.,
                   targets_pg=0., receptions_pg=0., receiving_yards_pg=0.,
                   receiving_tds_pg=0.)
            if pos=="QB":
                r.update(pass_attempts_pg=32., completions_pg=21., pass_yards_pg=230.,
                         passing_tds_pg=1.5, interceptions_pg=.7,
                         sacks_suffered_pg=2., carries_pg=3., rushing_yards_pg=15.)
            elif pos=="RB":
                r.update(carries_pg=13., rushing_yards_pg=53.,
                         rushing_tds_pg=.5, targets_pg=3., receptions_pg=2.4,
                         receiving_yards_pg=18.)
            elif pos=="WR":
                r.update(targets_pg=7., receptions_pg=4.4,
                         receiving_yards_pg=62., receiving_tds_pg=.4,
                         carries_pg=0.)
            elif pos=="TE":
                r.update(targets_pg=4., receptions_pg=3.,
                         receiving_yards_pg=32., receiving_tds_pg=.25,
                         carries_pg=0.)
            rates.append(r)
    return pd.DataFrame(pool), pd.DataFrame(rates)


class TeamPaceScoringTests(unittest.TestCase):
    def test_no_target_week_leak(self):
        history=weekly()
        before=predict_team_environment(history,2026,6)
        later=pd.concat([history, pd.DataFrame([
            {**history.iloc[0].to_dict(),"week":6,"attempts":55,"carries":45}
        ])],ignore_index=True)
        self.assertEqual(before, predict_team_environment(later,2026,6))
        self.assertLess(measured_team_games(history,2026,6)["week"].max(),6)

    def test_team_volume_requires_actual_history(self):
        insufficient=weekly().query("week <= 2")
        self.assertEqual(predict_team_environment(insufficient,2026,3),{})
        profiles=predict_team_environment(weekly(),2026,7)
        self.assertIn("AAA",profiles)
        self.assertIn("BBB",profiles)
        self.assertGreater(profiles["AAA"]["expected_pass_attempts"],
                           profiles["BBB"]["expected_pass_attempts"])

    def test_adapter_rejects_unsourced_pace(self):
        pool,rates=sim_fixture()
        with self.assertRaisesRegex(ValueError,"source"):
            run_verified_stat_simulation(pool,rates,n_sims=100,
                 team_environment={"AAA":{"expected_plays":70.,
                  "expected_pass_attempts":33.,"observed_games":6}})

    def test_scored_team_pace_changes_football_outcomes(self):
        pool,rates=sim_fixture()
        baseline_players,baseline_scores,baseline_stats=simulate_stat_driven_dfs(
            pool,rates,n_sims=240,seed=113,allow_role_proxies=False)
        experiment={"AAA":{"expected_plays":75.,"expected_pass_attempts":41.,
                           "observed_games":6,"source":"nflverse measured team pregame"},
                    "BBB":{"expected_plays":53.,"expected_pass_attempts":24.,
                           "observed_games":6,"source":"nflverse measured team pregame"}}
        changed_players,changed_scores,changed_stats=simulate_stat_driven_dfs(
            pool,rates,n_sims=240,seed=113,allow_role_proxies=False,
            team_environment=experiment)
        self.assertFalse(np.array_equal(baseline_scores,changed_scores))
        self.assertFalse(baseline_stats["__TEAM_ENVIRONMENT__|AAA"]["scored_team_history"])
        self.assertTrue(changed_stats["__TEAM_ENVIRONMENT__|AAA"]["scored_team_history"])
        self.assertGreater(changed_stats["__TEAM_PLAYS__|AAA"]["expected_plays"],
                           baseline_stats["__TEAM_PLAYS__|AAA"]["expected_plays"])


if __name__=="__main__":
    unittest.main()
