"""Current-week DK availability warnings are never participation certification."""
import unittest
import pandas as pd
from ai2_nfl_availability import availability_report


class AvailabilityTests(unittest.TestCase):
    def test_out_is_blocked_and_questionable_is_review(self):
        rows = pd.DataFrame([
            {"Name": "A", "TeamAbbrev": "AAA", "Position": "RB", "Status": "OUT"},
            {"Name": "B", "TeamAbbrev": "BBB", "Position": "WR", "Status": "Q"},
            {"Name": "C", "TeamAbbrev": "CCC", "Position": "QB", "Status": ""},
        ])
        result = availability_report(rows)
        self.assertEqual(result["availability_gate"].tolist(), ["BLOCK", "REVIEW", "REVIEW"])

    def test_no_status_never_passes(self):
        rows = pd.DataFrame([{"Name": "A", "TeamAbbrev": "AAA", "Position": "RB"}])
        result = availability_report(rows)
        self.assertEqual(result.iloc[0]["availability_gate"], "REVIEW")

    def test_healthy_label_not_official_certification(self):
        rows = pd.DataFrame([{"Name": "A", "TeamAbbrev": "AAA", "Position": "RB", "Status": "ACTIVE"}])
        self.assertEqual(availability_report(rows).iloc[0]["availability_gate"], "REVIEW")


if __name__ == "__main__":
    unittest.main()
