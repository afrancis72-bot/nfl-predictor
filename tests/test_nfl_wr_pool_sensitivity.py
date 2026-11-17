"""The 3.5 default stays fixed, 2.0 can add a historical WR, and paired truth stays identical."""
import unittest
import pandas as pd
from scripts.backtest_nfl_player_dk_means import pregame_game_pool
from scripts.audit_nfl_wr_pool_sensitivity import pair_roles

class WrPoolSensitivityTests(unittest.TestCase):
    def test_default_is_preserved(self):
        def p(team,name,pos,targets=0,attempts=0):
            return dict(team=team,player=name,position=pos,games_played=3,
               latest_observed_season=2026,latest_observed_week=8,
               targets_pg=targets,pass_attempts_pg=attempts,carries_pg=0)
        rates=pd.DataFrame([p("AAA","A QB","QB",attempts=33),
           p("BBB","B QB","QB",attempts=33),
           p("AAA","Star","WR",targets=8),
           p("AAA","Third","WR",targets=2.5),
           p("BBB","Star B","WR",targets=7)])
        base=pregame_game_pool(rates,"AAA","BBB",2026,9)
        explicit=pregame_game_pool(rates,"AAA","BBB",2026,9,wr_min_targets_pg=3.5)
        relaxed=pregame_game_pool(rates,"AAA","BBB",2026,9,wr_min_targets_pg=2.0)
        self.assertEqual(set(base.Name),set(explicit.Name))
        self.assertEqual(set(relaxed.Name)-set(base.Name),{"Third"})

    def test_pairing_rejects_role_and_truth_changes(self):
        item=dict(season=2026,week=9,game="AAA@BBB",team="AAA",
            player="A WR",opponent="BBB",role="WR1",
            actual_targets=7,prior_targets_pg=8,predicted_targets=6)
        m,_,_=pair_roles([item],[dict(item,predicted_targets=7)])
        self.assertEqual(m["common_role_errors"]["WR1"]["base_mae"],1)
        with self.assertRaisesRegex(ValueError,"role"):
            pair_roles([item],[dict(item,role="WR2")])
        with self.assertRaisesRegex(ValueError,"actual"):
            pair_roles([item],[dict(item,actual_targets=25)])
if __name__=="__main__":unittest.main()
