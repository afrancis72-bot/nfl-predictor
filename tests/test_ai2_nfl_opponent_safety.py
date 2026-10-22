"""Regression tests for QB opponent requirements and team aliases."""
import unittest
from ai2_nfl_matchups import opponent_from_game_info
from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_feasibility import find_classic_lineup

def roster():
    positions = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "DST", "WR"]
    return [{"ID": str(i), "Position": p, "TeamAbbrev": "T"+str(i),
             "Salary": 5500, "opponent": "OTHER"} for i, p in enumerate(positions)]

class TestOpponentSafety(unittest.TestCase):
    def test_alias_matchup(self):
        self.assertEqual(opponent_from_game_info("JAC", "JAC@TEN 1:00PM ET"), "TEN")
        self.assertEqual(opponent_from_game_info("TEN", "JAC@TEN 1:00PM ET"), "JAX")

    def test_missing_qb_opponent_rejected(self):
        rows = roster()
        rows[0].pop("opponent")
        self.assertTrue(any("opponent is missing" in x for x in check_classic_lineup(rows)))
        self.assertIs(find_classic_lineup(rows).feasible, False)

    def test_alias_dst_conflict_rejected(self):
        rows = roster()
        rows[0]["opponent"] = "JAC"
        rows[7]["TeamAbbrev"] = "JAX"
        self.assertTrue(any("opposing defense" in x for x in check_classic_lineup(rows)))
        self.assertIs(find_classic_lineup(rows).feasible, False)

if __name__ == "__main__":
    unittest.main()
