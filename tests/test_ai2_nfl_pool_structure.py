"""Regression checks for NFL pool positional readiness."""
import unittest
from ai2_nfl_preflight import validate_pool

def player(i, pos):
    return {"ID": str(i), "Name": "Player " + str(i),
            "Position": pos, "TeamAbbrev": "T" + str(i),
            "Salary": 5500, "proj": 15, "projection_source": "weekly",
            "weekly_projection_valid": True, "opportunity_verified": True}

class TestPreflightStructure(unittest.TestCase):
    def test_nine_qbs_cannot_pass(self):
        rows = [player(i, "QB") for i in range(9)]
        result = validate_pool(rows, source_fresh=True,
                               historical_validation_passed=True,
                               lineup_feasibility_passed=True)
        self.assertFalse(result.ready)
        self.assertTrue(any("RB" in reason for reason in result.reasons))

    def test_duplicate_ids_not_counted(self):
        rows = [player(1, "QB") for _ in range(9)]
        result = validate_pool(rows, source_fresh=True,
                               historical_validation_passed=True,
                               lineup_feasibility_passed=True)
        self.assertEqual(result.eligible_count, 1)
        self.assertFalse(result.ready)

    def test_valid_position_mix_still_requires_feasibility_proof(self):
        positions = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "DST", "WR"]
        rows = [player(i, p) for i, p in enumerate(positions)]
        result = validate_pool(rows, source_fresh=True,
                               historical_validation_passed=True)
        self.assertFalse(result.ready)
        self.assertTrue(any("feasibility" in reason for reason in result.reasons))

if __name__ == "__main__":
    unittest.main()
