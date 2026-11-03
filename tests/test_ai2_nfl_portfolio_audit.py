"""Audit downloads and build fingerprint have no temporary-file dependencies."""
import csv
import io
import unittest
from ai2_nfl_portfolio_audit import portfolio_audit_csv
from ai2_build_identity import deployed_build_identity


class UnifiedAuditRegression(unittest.TestCase):
    def test_csv_bytes_survive_without_local_file(self):
        roster = [
            {"ID": "qb", "Name": "QB", "Position": "QB", "TeamAbbrev": "DET"},
            {"ID": "wr", "Name": "WR", "Position": "WR", "TeamAbbrev": "DET"},
            {"ID": "dst", "Name": "DST", "Position": "DST", "TeamAbbrev": "SEA",
             "opponent": "ARI"},
        ]
        entry = {"candidate": {"players": roster, "salary": 49900},
                 "simulated_mean": 135.21, "simulated_p90": 177.34,
                 "simulated_p99": 205.78}
        content = portfolio_audit_csv([entry], seed=101010)
        self.assertIsInstance(content, bytes)
        report = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["Seed"], "101010")
        self.assertEqual(report[0]["QB passing teammates"], "WR")

    def test_build_fingerprint_includes_correct_live_branch(self):
        identity = deployed_build_identity()
        self.assertEqual(identity["branch"], "feature/unified-ai2-platform")
        self.assertEqual(identity["entrypoint"], "unified_app.py")
        self.assertEqual(len(identity["fingerprint"]), 16)


if __name__ == "__main__":
    unittest.main()
