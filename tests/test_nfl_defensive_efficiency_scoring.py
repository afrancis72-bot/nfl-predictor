"""Defense-adjusted yards must conserve real passes and game event counts."""
import unittest
import numpy as np
import pandas as pd
from ai2_nfl_defensive_efficiency import build_defense_profiles,matchup_yard_multiplier
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


class OpposingDefenseTests(unittest.TestCase):
    def observed(self):
        return pd.DataFrame([
            dict(defense="BBB",position="WR",multiplier=1.23,
                 player_game_observations=65,
                 source="nflverse observed pregame opponent_team, player-week partial DK"),
            dict(defense="BBB",position="RB",multiplier=.82,
                 player_game_observations=60,
                 source="nflverse observed pregame opponent_team, player-week partial DK"),
        ])
    def test_requires_real_observed_history_and_minimum_sample(self):
        obs=self.observed()
        out=build_defense_profiles(obs,2026,9)
        self.assertEqual(out[("BBB","WR")]["week"],9)
        self.assertAlmostEqual(matchup_yard_multiplier(out,"BBB","WR",strength=1.),1.12)
        self.assertAlmostEqual(matchup_yard_multiplier(out,"BBB","RB",strength=1.),.88)
        self.assertEqual(matchup_yard_multiplier(out,"AAA","WR",strength=1.),1.)
        wrong=obs.copy()
        wrong["source"]="MOCK PREDICTIONS"
        self.assertEqual(build_defense_profiles(wrong,2026,9),{})
        wrong=obs.copy()
        wrong["player_game_observations"]=4
        self.assertEqual(build_defense_profiles(wrong,2026,9),{})

    def test_football_stat_draws_maintain_team_passing_yards(self):
        pool,rates=sim_fixture()
        profiles=build_defense_profiles(self.observed(),2026,9)
        players,base,baseline=simulate_stat_driven_dfs(
            pool,rates,n_sims=200,seed=77,allow_role_proxies=False)
        players,adjusted,scored=simulate_stat_driven_dfs(
            pool,rates,n_sims=200,seed=77,allow_role_proxies=False,
            defense_profiles=profiles,defense_weight=1.)
        self.assertFalse(np.array_equal(base,adjusted))
        self.assertEqual(scored["__OPPONENT_YARD_EFFICIENCY__|AAA"]["opponent"],"BBB")
        self.assertTrue(scored["__OPPONENT_YARD_EFFICIENCY__|AAA"]["scored_players"]>=2)
        for team in ("AAA","BBB"):
            world=scored[f"__TEAM_PLAYS__|{team}"]
            np.testing.assert_array_equal(world["simulated_plays"],
                                          world["simulated_dropbacks"]+world["simulated_rushes"])
            # The QB's passing yards must equal the same receiving yards
            # allocated to skill players plus verified OTHER receiver.
            qb_name="Q"+team
            q=scored[qb_name]["passing_yards"]
            skill_names=["R"+team,"W"+team,"T"+team]
            generated=sum(scored[n]["receiving_yards"] for n in skill_names)
            generated+=scored[f"__OTHER_RECEIVER__|{team}"]["receiving_yards"]
            np.testing.assert_allclose(q,generated,atol=.001)
            np.testing.assert_array_equal(
                baseline[f"__TEAM_SCORING__|{team}"]["simulated_offensive_tds"],
                scored[f"__TEAM_SCORING__|{team}"]["simulated_offensive_tds"])

    def test_zero_does_not_change_live_sim(self):
        pool,rates=sim_fixture()
        p=build_defense_profiles(self.observed(),2026,9)
        _,base,_=simulate_stat_driven_dfs(pool,rates,n_sims=170,seed=66,
                                          allow_role_proxies=False)
        _,unchanged,_=simulate_stat_driven_dfs(pool,rates,n_sims=170,seed=66,
                                          allow_role_proxies=False,defense_profiles=p)
        np.testing.assert_array_equal(base,unchanged)
    def test_malicious_unsourced_injected_factor_rejected(self):
        with self.assertRaisesRegex(ValueError,"unsourced"):
            matchup_yard_multiplier({("BBB","WR"):
                {"source":"unverified","factor":1.25,"sample":200}},
                "BBB","WR",strength=.5)


if __name__=="__main__":
    unittest.main()
