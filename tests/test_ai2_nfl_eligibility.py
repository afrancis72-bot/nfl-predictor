"""Tests that feasibility only sees conservatively eligible players."""
import unittest
from ai2_nfl_eligibility import eligible_nfl_rows
from ai2_nfl_feasibility import find_classic_lineup

def row(i, pos):
    return {"ID": str(i), "Name": str(i), "Position": pos, "TeamAbbrev": "T"+str(i),
            "Salary": 5500, "proj": 12, "projection_source": "weekly",
            "weekly_projection_valid": True, "opportunity_verified": True,
            "dk_status": "", "opponent": "OTHER"}

class TestEligibilityFeasibility(unittest.TestCase):
    def test_out_player_not_used(self):
        positions = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "DST", "WR")
        rows = [row(i, pos) for i, pos in enumerate(positions)]
        rows[0]["dk_status"] = "OUT"
        eligible = eligible_nfl_rows(rows)
        self.assertEqual(len(eligible), 8)
        self.assertIs(find_classic_lineup(eligible).feasible, False)

    def test_fallback_and_unverified_not_used(self):
        rows = [row(0, "QB"), row(1, "RB"), row(2, "WR")]
        rows[0]["projection_source"] = "DK PPG fallback"
        rows[1]["opportunity_verified"] = False
        self.assertEqual([r["ID"] for r in eligible_nfl_rows(rows)], ["2"])

    def test_duplicate_id_rejected(self):
        rows = [row(1, "RB"), row(1, "RB")]
        self.assertEqual(len(eligible_nfl_rows(rows)), 1)

if __name__ == "__main__":
    unittest.main()
