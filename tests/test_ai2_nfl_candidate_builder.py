"""NFL candidate builder safety checks, using independent fixture projections."""
import unittest
from ai2_nfl_candidate_builder import build_classic_candidates
from ai2_nfl_lineup import check_classic_lineup

class CandidateBuilderTests(unittest.TestCase):
    def test_legal_lineups_only(self):
        rows = []
        for pos, count, salary in (("QB", 3, 6500), ("RB", 8, 6500),
                                   ("WR", 12, 5500), ("TE", 4, 5000),
                                   ("DST", 4, 3500)):
            for i in range(count):
                rows.append({"ID": pos + str(i), "Name": pos + str(i),
                             "Position": pos, "TeamAbbrev": "T" + str(i),
                             "opponent": "OPP" + str(i), "Salary": salary,
                             "proj": 10 + i, "opportunity_verified": True})
        lineups = build_classic_candidates(rows, count=3, attempts=4000)
        self.assertTrue(lineups)
        for entry in lineups:
            self.assertEqual(check_classic_lineup(entry["players"]), [])
            self.assertGreaterEqual(entry["salary"], 47500)
            self.assertLessEqual(entry["salary"], 50000)
    def test_empty_pool_fails_closed(self):
        self.assertEqual(build_classic_candidates([]), [])
if __name__ == "__main__":
    unittest.main()
