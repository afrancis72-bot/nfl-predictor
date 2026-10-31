"""Pre-kickoff market totals adjust only scored football priors, not outcomes."""
import unittest
import numpy as np
import pandas as pd
from ai2_nfl_game_market import market_totals_for_slate,market_tds_prior
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


class MarketTotalsTests(unittest.TestCase):
    def test_only_same_week_before_kickoff_total_used(self):
        df=pd.DataFrame([
            {"season":2026,"week":8,"game_type":"REG",
             "home_team":"AAA","away_team":"BBB",
             "total_line":54.,"home_score":999,"away_score":999},
            {"season":2026,"week":9,"game_type":"REG",
             "home_team":"CCC","away_team":"DDD",
             "total_line":45.,"home_score":-9,"away_score":99},
        ])
        result=market_totals_for_slate(df,2026,8)
        self.assertEqual(result[("AAA","BBB")]["total_points"],54.)
        self.assertEqual(len(result),1)
        self.assertNotIn("home_score",str(result))

    def test_no_invented_zero_missing_or_ambiguous_total(self):
        df=pd.DataFrame([
            {"season":2026,"week":8,"game_type":"REG",
             "home_team":"AAA","away_team":"BBB","total_line":np.nan},
        ])
        self.assertEqual(market_totals_for_slate(df,2026,8),{})
        d=pd.concat([df.assign(total_line=50),df.assign(total_line=55)])
        self.assertEqual(market_totals_for_slate(d,2026,8),{})

    def test_different_market_totals_change_sourced_scoring_but_not_volume(self):
        pool,rates=sim_fixture()
        _,a,base=simulate_stat_driven_dfs(
            pool,rates,n_sims=180,seed=312,allow_role_proxies=False)
        market={("AAA","BBB"):{
            "total_points":62.,
            "source":"nflverse schedules sourced pregame total_line"}}
        _,b,detail=simulate_stat_driven_dfs(
            pool,rates,n_sims=180,seed=312,allow_role_proxies=False,
            market_game_totals=market,market_weight=.5)
        self.assertFalse(np.array_equal(a,b))
        for team in ("AAA","BBB"):
            self.assertTrue(detail[f"__TEAM_SCORING__|{team}"]["market_scored"])
            self.assertEqual(detail[f"__TEAM_SCORING__|{team}"]["market_total_line"],62.)
            np.testing.assert_array_equal(
                detail[f"__TEAM_PLAYS__|{team}"]["simulated_plays"],
                detail[f"__TEAM_PLAYS__|{team}"]["simulated_dropbacks"]+
                detail[f"__TEAM_PLAYS__|{team}"]["simulated_rushes"])

    def test_no_unsourced_market_adjustment(self):
        pool,rates=sim_fixture()
        with self.assertRaisesRegex(ValueError,"requires actual"):
            simulate_stat_driven_dfs(
                pool,rates,n_sims=30,seed=1,allow_role_proxies=False,
                market_game_totals={("AAA","BBB"):{"total_points":55.}},
                market_weight=.5)

    def test_market_mix_zero_matches_historical_prior(self):
        for total in (29.,40.,58.):
            self.assertEqual(market_tds_prior(2.5,1.65,total,weight=0.),2.5)
        self.assertLess(market_tds_prior(2.5,1.65,32.,weight=.5),2.5)
        self.assertGreater(market_tds_prior(2.5,1.65,58.,weight=.5),2.5)


if __name__=="__main__":
    unittest.main()
