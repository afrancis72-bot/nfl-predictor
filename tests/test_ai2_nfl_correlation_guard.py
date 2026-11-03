"""Live unified-app regression tests for opposing DST/pass-catcher correlation."""
import unittest
from ai2_nfl_sim_portfolio import _solve_joint_portfolio
from ai2_nfl_correlation_guard import (
    opposing_pass_catcher_dst_pairs, opposing_dst_lineup_budget,
    correlation_audit_rows,
)


class CorrelationGuardTests(unittest.TestCase):
    def test_draftkings_game_info_fallback_and_team_alias(self):
        roster = [
            {"ID": "jsn", "Name": "JSN", "Position": "WR",
             "TeamAbbrev": "SEA", "Game Info": "ARZ@SEA 4:25PM ET"},
            {"ID": "ari", "Name": "Cardinals", "Position": "DST",
             "TeamAbbrev": "ARI", "opponent": "SEA"},
        ]
        pairs = opposing_pass_catcher_dst_pairs(roster)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["player_id"], "jsn")
        self.assertEqual(pairs[0]["defense_team"], "ARI")

    def test_unrelated_players_not_conflicts(self):
        roster = [
            {"ID": "jsn", "Position": "WR", "TeamAbbrev": "SEA", "opponent": "ARI"},
            {"ID": "dst", "Position": "DST", "TeamAbbrev": "HOU", "opponent": "JAX"},
        ]
        self.assertEqual(opposing_pass_catcher_dst_pairs(roster), [])

    def test_budget_is_floor_not_round_up(self):
        self.assertEqual(opposing_dst_lineup_budget(15), 0)
        self.assertEqual(opposing_dst_lineup_budget(20), 0)
        self.assertEqual(opposing_dst_lineup_budget(50), 2)
        self.assertEqual(opposing_dst_lineup_budget(150), 6)

    def test_joint_solver_cannot_select_correlated_receivers(self):
        def candidate(label, mean, bad):
            defense_team = "ARI" if bad else "LAC"
            return {
                "candidate": {"players": [
                    {"ID": label + "-qb", "Position": "QB",
                     "TeamAbbrev": "DET", "opponent": "MIA", "Salary": 6500},
                    {"ID": label + "-wr", "Position": "WR",
                     "TeamAbbrev": "SEA", "opponent": "ARI", "Salary": 5500},
                    {"ID": label + "-dst", "Position": "DST",
                     "TeamAbbrev": defense_team, "opponent": "SEA" if bad else "KC",
                     "Salary": 3500}], "salary": 49800},
                "simulated_mean": mean,
                "simulated_p90": mean + 20,
                "simulated_p99": mean + 40,
            }
        pool = [candidate("bad", 400, True),
                candidate("good1", 200, False),
                candidate("good2", 180, False)]
        chosen, _ = _solve_joint_portfolio(
            pool, count=2, premium_count=0, max_exposure=1.0,
            upside_share=0.25, value_salary_ceiling=5000,
            value_max_appearances=2, premium_value_max_appearances=1)
        self.assertEqual(len(chosen), 2)
        self.assertTrue(all(not opposing_pass_catcher_dst_pairs(x["candidate"]["players"])
                            for x in chosen))

    def test_audit_exposes_exact_pair(self):
        ranked = [{"candidate": {"players": [
            {"ID": "jsn", "Name": "JSN", "Position": "WR",
             "TeamAbbrev": "SEA", "opponent": "ARI"},
            {"ID": "ari", "Name": "Arizona DST", "Position": "DST",
             "TeamAbbrev": "ARI", "opponent": "SEA"},
        ]}}]
        rows = correlation_audit_rows(ranked)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["Conflicted"])
        self.assertIn("JSN", rows[0]["Players facing rostered DST"])


if __name__ == "__main__":
    unittest.main()
