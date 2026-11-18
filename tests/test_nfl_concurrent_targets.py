"""No-leakage, concurrent-game receiver shares and 2-team scoring conservation."""
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from ai2_nfl_concurrent_targets import (
    pregame_concurrent_target_prior,allocate_concurrent_target_budget)
from ai2_nfl_game_conservation import audit_team_game_conservation
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


def recent_games():
    rows=[]
    for week in range(1,8):
        for team in ("AAA","BBB"):
            # Week 7 must NEVER be visible to target-week 7's pregame prior.
            shock=3 if week==7 else 1
            for pos,name,attempts,targets in (
                ("QB","Q"+team,32,0),
                ("RB","R"+team,0,3*shock),
                ("WR","W"+team,0,8*shock),
                ("TE","T"+team,0,5*shock),
                ("WR",f"Extra {team} {week}",0,4),
            ):
                rows.append({
                    "season":2026,"week":week,"season_type":"REG",
                    "recent_team":team,"position":pos,"player_name":name,
                    "attempts":attempts,"targets":targets,
                })
    return pd.DataFrame(rows)


class ConcurrentGameTargetTests(unittest.TestCase):
    def setUp(self):
        self.pool,self.rates=sim_fixture()
        self.weekly=recent_games()

    def profile(self, history=None, season=2026, week=7):
        return pregame_concurrent_target_prior(
            self.weekly if history is None else history,
            team="AAA",roster=self.pool,season=season,week=week)

    def test_concurrent_shares_observed_same_game_not_sum_of_role_averages(self):
        p=self.profile()
        self.assertEqual(p["observed_games"],6)
        self.assertEqual(p["last_observed_week"],6)
        self.assertAlmostEqual(p["modeled_share"],16/32)
        self.assertAlmostEqual(p["unmodeled_share"],4/32)
        self.assertAlmostEqual(p["no_identified_target_share"],12/32)
        self.assertFalse(p["game_day_active_verified"])
        rates=np.array([3.,8.,5.])
        modeled,other,unassigned=allocate_concurrent_target_budget(rates,32.,p)
        np.testing.assert_allclose(modeled,rates)
        self.assertAlmostEqual(other,4.)
        self.assertAlmostEqual(unassigned,12.)
        self.assertAlmostEqual(modeled.sum()+other+unassigned,32.)

    def test_future_outcomes_are_not_used_in_prior(self):
        a=self.profile()
        b=self.profile(history=self.weekly[self.weekly["week"].lt(7)])
        self.assertEqual(a,b)
        # An exact target-week restriction is enforced even if week 7 exists.
        self.assertLess(max(g["week"] for g in a["historical_team_games"]),7)

    def test_fails_closed_on_corrupt_prior(self):
        p=self.profile()
        with self.assertRaisesRegex(ValueError,"Invalid prior receiver"):
            allocate_concurrent_target_budget(
                np.array([7.,4.,3.]),32.,{**p,"unmodeled_share":2.})
        with self.assertRaisesRegex(ValueError,"Missing positive"):
            allocate_concurrent_target_budget(np.array([0.,0.]),32.,p)

    def test_unavailable_history_is_not_invented(self):
        self.assertIsNone(self.profile(week=3))

    def test_two_team_worlds_have_real_no_identified_target_bucket(self):
        priors={team:pregame_concurrent_target_prior(
                    self.weekly,team=team,roster=self.pool,
                    season=2026,week=7) for team in ("AAA","BBB")}
        original,base_score,base=simulate_stat_driven_dfs(
            self.pool,self.rates,n_sims=300,seed=24687,
            allow_role_proxies=False)
        players,new_score,stat=simulate_stat_driven_dfs(
            self.pool,self.rates,n_sims=300,seed=24687,
            allow_role_proxies=False,concurrent_target_profiles=priors)
        self.assertEqual(players["ID"].tolist(),original["ID"].tolist())
        self.assertEqual(new_score.shape,base_score.shape)
        self.assertFalse(np.array_equal(new_score,base_score))
        for team in ("AAA","BBB"):
            audit=audit_team_game_conservation(stat,players,team)
            self.assertEqual(audit["worlds_verified"],300)
            self.assertTrue(audit["modeled_no_identified_target_attempts_separately"])
            self.assertIsNotNone(audit["mean_unmodeled_receiver_attempt_slots"])
            self.assertGreater(audit["mean_no_identified_target_attempt_slots"],1.)
            self.assertTrue((stat[
                f"__UNIDENTIFIED_TARGET_ATTEMPTS__|{team}"
            ]["receptions"]==0).all())
            meta=stat[f"__CONCURRENT_TARGET_SPLIT__|{team}"]
            self.assertFalse(meta["production_enabled"])
            self.assertFalse(meta["game_day_active_verified"])
            self.assertAlmostEqual(
                meta["modeled_target_weight_pg"]+
                meta["unmodeled_receiver_target_weight_pg"]+
                meta["unidentified_attempt_weight_pg"],
                meta["pregame_pass_attempt_budget_pg"])
            self.assertTrue(np.isfinite(new_score).all())
        self.assertNotIn("__UNIDENTIFIED_TARGET_ATTEMPTS__|AAA",base)
        self.assertFalse(stat["__FOOTBALL_PLAYER_UNIVERSE__"][
            "verified_game_day_active"])

    def test_no_prior_does_not_change_legacy_simulation(self):
        _,old,_=simulate_stat_driven_dfs(
            self.pool,self.rates,n_sims=130,seed=77,
            allow_role_proxies=False)
        _,same,stat=simulate_stat_driven_dfs(
            self.pool,self.rates,n_sims=130,seed=77,
            allow_role_proxies=False,
            concurrent_target_profiles={"AAA":None,"BBB":None})
        np.testing.assert_array_equal(old,same)
        self.assertEqual(
            stat["__CONCURRENT_TARGET_SPLIT__|AAA"]["status"],
            "INSUFFICIENT_CONCURRENT_PREGAME_TEAM_GAMES")


if __name__=="__main__":
    unittest.main()
