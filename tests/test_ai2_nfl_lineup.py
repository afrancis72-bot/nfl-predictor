"""Tests for NFL Classic roster structure."""
import unittest
from ai2_nfl_lineup import check_classic_lineup

def sample():
    positions = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "DST", "WR"]
    teams = ["A", "B", "C", "A", "B", "C", "D", "E", "F"]
    return [{"ID": str(i), "Position": p, "TeamAbbrev": t, "Salary": 5500,
             "opponent": "Z"} for i, (p, t) in enumerate(zip(positions, teams))]

class TestClassicLineup(unittest.TestCase):
    def test_valid_structure(self):
        self.assertEqual(check_classic_lineup(sample()), [])

    def test_duplicate_id(self):
        players = sample()
        players[1]["ID"] = players[0]["ID"]
        self.assertTrue(any("unique" in x for x in check_classic_lineup(players)))

    def test_same_team_running_backs(self):
        players = sample()
        players[2]["TeamAbbrev"] = players[1]["TeamAbbrev"]
        self.assertTrue(any("running backs" in x for x in check_classic_lineup(players)))

    def test_qb_against_defense(self):
        players = sample()
        players[0]["opponent"] = players[7]["TeamAbbrev"]
        self.assertTrue(any("opposing defense" in x for x in check_classic_lineup(players)))

    def test_every_offensive_position_against_dst(self):
        for i in (0,1,2,3,4,5,6,8):
            players=sample()
            players[i]["opponent"]=players[7]["TeamAbbrev"]
            self.assertTrue(any("opposing defense" in x for x in check_classic_lineup(players)))

    def test_same_team_rb_dst_allowed(self):
        players=sample()
        players[1]["TeamAbbrev"]=players[7]["TeamAbbrev"]
        self.assertEqual(check_classic_lineup(players),[])

    def test_dst_opponent_reverse_lookup(self):
        players=sample()
        players[7]["opponent"]=players[1]["TeamAbbrev"]
        self.assertTrue(any("RB" in x and "opposing defense" in x for x in check_classic_lineup(players)))

    def test_export_rejects_conflict(self):
        from ai2_nfl_export import export_classic_candidates
        players=sample()
        players[1]["opponent"]=players[7]["TeamAbbrev"]
        with self.assertRaisesRegex(ValueError,"opposing defense"):
            export_classic_candidates([{"players":players}])

    def test_salary_floor(self):
        players = sample()
        players[0]["Salary"] = 1000
        self.assertTrue(any("Salary" in x for x in check_classic_lineup(players)))

    def test_missing_position(self):
        players = sample()
        players[0]["Position"] = "WR"
        self.assertTrue(any("required" in x for x in check_classic_lineup(players)))

if __name__ == "__main__":
    unittest.main()
