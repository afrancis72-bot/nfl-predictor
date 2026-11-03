"""Regression for real observed high-volume RB reentry watch."""
import unittest
import pandas as pd
from ai2_nfl_role_intelligence import returning_rb_competition
from ai2_nfl_portfolio_risk import maximum_player_appearances


class RBReturnWatch(unittest.TestCase):
    def base(self, veteran_status="QUESTIONABLE", veteran_latest=3):
        baseline = pd.DataFrame([
            {"ID":"hall","Name":"Breece Hall","Team":"NYJ","Position":"RB",
             "current_status":veteran_status},
            {"ID":"allen","Name":"Braelon Allen","Team":"NYJ","Position":"RB",
             "current_status":""},
            {"ID":"other","Name":"Other RB","Team":"KC","Position":"RB",
             "current_status":""},
        ])
        rates = pd.DataFrame([
            {"player":"Breece Hall","team":"NYJ","position":"RB","games_played":3,
             "carries_pg":16.,"latest_observed_season":2026,
             "latest_observed_week":veteran_latest},
            {"player":"Braelon Allen","team":"NYJ","position":"RB","games_played":7,
             "carries_pg":18.,"latest_observed_season":2026,
             "latest_observed_week":8},
            {"player":"Other RB","team":"KC","position":"RB","games_played":8,
             "carries_pg":15.,"latest_observed_season":2026,
             "latest_observed_week":8},
        ])
        return baseline,rates

    def test_hall_questionable_caps_affected_backup_not_entire_team(self):
        baselines,rates=self.base()
        risk, watch=returning_rb_competition(baselines,rates,season=2026,week=9)
        self.assertEqual([x["returning_player"] for x in watch],["Breece Hall"])
        self.assertEqual(list(risk),["allen"])
        rb={"ID":"allen","Position":"RB",
            "return_competition_reason":"Breece Hall return unconfirmed"}
        self.assertEqual(maximum_player_appearances(rb,50,.28),4)

    def test_out_on_undated_injury_page_is_a_watch_not_clearance(self):
        baselines,rates=self.base(veteran_status="OUT")
        risk,watch=returning_rb_competition(baselines,rates,season=2026,week=9)
        self.assertEqual(len(watch),1)
        self.assertIn("not independently verified",watch[0]["status_note"])
        self.assertIn("allen",risk)

    def test_ir_status_does_not_create_reentry_watch(self):
        baselines,rates=self.base(veteran_status="IR")
        risk,watch=returning_rb_competition(baselines,rates,season=2026,week=9)
        self.assertEqual((risk,watch),({},[]))

    def test_recently_active_starter_does_not_trigger_return_watch(self):
        baselines,rates=self.base(veteran_latest=8)
        risk,watch=returning_rb_competition(baselines,rates,season=2026,week=9)
        self.assertEqual((risk,watch),({},[]))


if __name__=="__main__":
    unittest.main()
