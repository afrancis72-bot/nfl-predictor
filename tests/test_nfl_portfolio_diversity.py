import unittest
from ai2_nfl_portfolio_diversity import portfolio_overlap_audit


class PortfolioDiversityTests(unittest.TestCase):
    def entry(self, number, name, contest="$0.50"):
        positions = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR", "DST"]
        players = []
        for i, position in enumerate(positions):
            ident = str(i) if number == 1 or i <= 6 else f"variation{number}-{i}"
            players.append({
                "ID": ident, "Name": name if position == "QB" else f"Player {ident}",
                "Position": position,
                "TeamAbbrev": "DAL" if position in {"QB", "WR"} else "ARI",
            })
        return {"candidate": {"players": players, "salary": 49800},
                "contest_tier": contest, "simulated_mean": 150,
                "simulated_p99": 220}

    def test_near_duplicates_flagged_but_not_excluded(self):
        entries = [self.entry(1, "Dak Prescott", "$3"),
                   self.entry(2, "Dak Prescott"),
                   self.entry(3, "Dak Prescott")]
        report = portfolio_overlap_audit(entries)
        self.assertEqual(report["summary"]["pairs"], 3)
        self.assertEqual(report["summary"]["pairs_at_least_alert"], 3)
        self.assertEqual(report["summary"]["distinct_qbs"], 1)
        self.assertEqual(len(report["lineup_scripts"]), 3)
        self.assertTrue(all(item["Shared players"] == 7
                            for item in report["near_duplicate_pairs"]))
        self.assertIn("Preserve strong repeated cores",
                      report["summary"]["concentration_policy"])

    def test_invalid_roster_refused(self):
        with self.assertRaisesRegex(ValueError, "non-unique"):
            portfolio_overlap_audit([{
                "candidate": {"players": [
                    {"ID": "1", "Position": "QB"} for _ in range(9)]},
                "contest_tier": "$3"}])


if __name__ == "__main__":
    unittest.main()
