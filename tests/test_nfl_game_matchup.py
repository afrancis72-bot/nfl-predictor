"""Team opponent TD/play conditioning must be measured pregame and conserved."""
import unittest
import numpy as np
import pandas as pd

from ai2_nfl_game_matchup import (
    SOURCE, _past_team_games, team_defense_game_profiles,
    validated_game_profile_lookup,
)
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


def historical_games():
    records=[]
    for week in range(1, 11):
        for team,opp in (("AAA","BBB"),("BBB","AAA")):
            pass_tds=(1 if week<=3 else 3) if team=="AAA" else 2
            for pos,prefix in (("QB","Q"),("RB","R")):
                records.append(dict(
                    season=2025,week=week,season_type="REG",
                    player_name=prefix+team,recent_team=team,
                    opponent_team=opp,position=pos,
                    attempts=(32 if pos=="QB" else 0),
                    sacks_suffered=(2 if pos=="QB" else 0),
                    passing_tds=(pass_tds if pos=="QB" else 0),
                    carries=(3 if pos=="QB" else 26),
                    rushing_tds=(0 if pos=="QB" else 1),
                ))
    return pd.DataFrame(records)


def effect(defense,td_delta,plays_delta):
    return dict(
        defense=defense,source=SOURCE,td_delta=td_delta,
        plays_delta=plays_delta,sample_games=8,
        last_observed_season=2026,last_observed_week=8,
        target_season=2026,target_week=9)


class OpponentGameEffectTests(unittest.TestCase):
    def test_future_target_and_later_weeks_not_in_pregame_profiles(self):
        h=historical_games()
        earlier=team_defense_game_profiles(h,2025,10)
        future=h.copy()
        extra=future.query("week == 10").copy()
        extra["week"]=11
        extra["passing_tds"]=9
        self.assertEqual(earlier,team_defense_game_profiles(
            pd.concat([h,extra],ignore_index=True),2025,10))
        self.assertLess(_past_team_games(h,2025,10)["week"].max(),10)

    def test_prior_opponent_own_baseline_produces_shrunk_profiles(self):
        profiles=team_defense_game_profiles(historical_games(),2025,10)
        self.assertIn("BBB",profiles)
        self.assertIn("AAA",profiles)
        self.assertEqual(profiles["BBB"]["source"],SOURCE)
        self.assertGreater(profiles["BBB"]["td_delta"],0.)
        self.assertGreaterEqual(profiles["BBB"]["sample_games"],5)
        self.assertEqual(profiles["BBB"]["last_observed_week"],9)
        self.assertLessEqual(abs(profiles["BBB"]["td_delta"]),.7)

    def test_guard_rejects_unsourced_future_and_out_of_bounds_effects(self):
        valid={"BBB":effect("BBB",-.45,-5.)}
        self.assertEqual(validated_game_profile_lookup(
            valid,season=2026,week=9,strength=.5)["BBB"],(-.225,-2.5))
        for edit,wrong in (("source","MOCK"),("target_week",10),
                           ("sample_games",1),("td_delta",5.),
                           ("last_observed_week",9)):
            row=effect("BBB",-.45,-5.)
            row[edit]=wrong
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validated_game_profile_lookup(
                    {"BBB":row},season=2026,week=9,strength=.5)

    def test_zero_strength_preserves_exact_original_simulation(self):
        pool,rates=sim_fixture()
        x,base,raw=simulate_stat_driven_dfs(
            pool,rates,seed=50,n_sims=180,allow_role_proxies=False)
        x,new,newraw=simulate_stat_driven_dfs(
            pool,rates,seed=50,n_sims=180,allow_role_proxies=False,
            game_matchup_profiles={"BBB":effect("BBB",-.45,-5.)},
            game_matchup_strength=0.)
        np.testing.assert_array_equal(base,new)
        self.assertNotIn("__OPPONENT_GAME_EFFECT__|AAA",newraw)

    def test_opponent_changes_team_tds_plays_script_but_conserves_all_events(self):
        pool,rates=sim_fixture()
        env={
            "AAA":dict(expected_plays=67.,expected_pass_attempts=34.,
                        expected_offensive_tds=2.6,
                        expected_passing_tds=1.75,expected_rushing_tds=.85,
                        observed_games=6,source="nflverse pregame"),
            "BBB":dict(expected_plays=67.,expected_pass_attempts=34.,
                        expected_offensive_tds=1.9,
                        expected_passing_tds=1.2,expected_rushing_tds=.7,
                        observed_games=6,source="nflverse pregame"),
        }
        _,base,raw=simulate_stat_driven_dfs(
            pool,rates,seed=90,n_sims=400,allow_role_proxies=False,
            team_environment=env)
        _,new,world=simulate_stat_driven_dfs(
            pool,rates,seed=90,n_sims=400,allow_role_proxies=False,
            team_environment=env,
            game_matchup_profiles={"BBB":effect("BBB",-.45,-5.),
                                   "AAA":effect("AAA",.25,3.)},
            game_matchup_strength=1.,
            game_matchup_target_season=2026,game_matchup_target_week=9)
        self.assertFalse(np.array_equal(base,new))
        self.assertLess(world["__TEAM_SCORING__|AAA"]["expected_offensive_tds"],
                        raw["__TEAM_SCORING__|AAA"]["expected_offensive_tds"])
        self.assertLess(world["__TEAM_PLAYS__|AAA"]["expected_plays"],
                        raw["__TEAM_PLAYS__|AAA"]["expected_plays"])
        self.assertGreater(world["__OPPONENT_GAME_EFFECT__|AAA"][
            "script_center_trailing_positive"],0.)
        for team in ("AAA","BBB"):
            v=world["__TEAM_PLAYS__|"+team]
            np.testing.assert_array_equal(v["simulated_plays"],
                                          v["simulated_dropbacks"]+v["simulated_rushes"])
            s=world["__TEAM_SCORING__|"+team]
            np.testing.assert_array_equal(s["simulated_offensive_points"],
                  7*s["simulated_offensive_tds"]+3*s["simulated_field_goals"])
            q=world["Q"+team]["passing_yards"]
            receiving=sum(world[p+team]["receiving_yards"]
                          for p in ("R","W","T"))
            receiving+=world["__OTHER_RECEIVER__|"+team]["receiving_yards"]
            np.testing.assert_allclose(q,receiving,atol=.002)


if __name__ == "__main__":
    unittest.main()
