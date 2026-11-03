"""Safety regressions for role-aware NFL Classic exposure and overlap."""
import unittest

from ai2_nfl_portfolio_risk import (
    uncertain_running_back_reason, maximum_player_appearances,
    overlap_conflicts, overlap_groups, MAX_SHARED_PLAYERS,
)


class PortfolioRiskRegression(unittest.TestCase):
    def test_singletary_like_four_game_role_is_not_treated_as_featured_back(self):
        rb = {
            "ID":"backup", "Name":"Backup RB", "Position":"RB",
            "observed_games":4, "observed_carries_pg":10.0,
            "observed_targets_pg":1.0,"observed_rushing_yards_pg":45,
            "observed_recent_four_games":4,
            "observed_recent_four_rush_yd_pg":20,
            "observed_recent_four_carries_pg":8,
            "observed_recent_four_targets_pg":1,
            "projected_touches":15,
        }
        self.assertIsNotNone(uncertain_running_back_reason(rb))
        self.assertEqual(maximum_player_appearances(rb, 50, .28), 6)

    def test_unverified_role_does_not_invent_rushing_history(self):
        rb={"ID":"rookie", "Position":"RB", "Salary":6000}
        self.assertIsNone(uncertain_running_back_reason(rb))
        self.assertEqual(maximum_player_appearances(rb, 50, .28), 14)

    def test_strict_base_cap(self):
        wr={"ID":"wr","Position":"WR"}
        self.assertEqual(maximum_player_appearances(wr, 50, .28), 14)
        self.assertEqual(maximum_player_appearances(wr, 20, .28), 5)

    def test_seven_shared_players_is_a_conflict(self):
        a=[{"ID":str(i)} for i in range(9)]
        b=[{"ID":str(i)} for i in list(range(7))+["9","10"]]
        c=[{"ID":str(i)} for i in list(range(6))+["9","10","11"]]
        self.assertEqual(MAX_SHARED_PLAYERS,6)
        self.assertEqual(overlap_conflicts(b, [[str(i) for i in range(9)]]),1)
        self.assertEqual(overlap_conflicts(c, [[str(i) for i in range(9)]]),0)
        groups=overlap_groups([
            {"candidate":{"players":a}},
            {"candidate":{"players":b}},
            {"candidate":{"players":c}},
        ])
        self.assertTrue(any(set(g)=={0,1} for g in groups))
        self.assertFalse(any(set(g)=={0,2} for g in groups))


if __name__=="__main__":
    unittest.main()
