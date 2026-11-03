"""Live unified-app regression tests for opposing DST/pass-catcher correlation."""
import unittest
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
