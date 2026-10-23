"""NFL Classic candidate roster export tests."""
import csv
import io
import unittest
from ai2_nfl_export import export_classic_candidates, slot_classic_lineup

class ExportTests(unittest.TestCase):
    def test_roster_slots_and_ids(self):
        positions = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB", "DST"]
        salaries = [6500, 6000, 6000, 5500, 5500, 5500, 4500, 5500, 3000]
        players = [{"ID": str(i + 1), "Position": p, "Salary": salary,
                    "TeamAbbrev": "T" + str(i), "opponent": "OPP" + str(i)}
                   for i, (p, salary) in enumerate(zip(positions, salaries))]
        csv_text = export_classic_candidates([{"players": players}])
        rows = list(csv.reader(io.StringIO(csv_text)))
        self.assertEqual(rows[0], ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"])
        self.assertEqual(len(rows[1]), 9)
        self.assertEqual(set(rows[1]), {str(i) for i in range(1, 10)})
    def test_illegal_roster_rejected(self):
        with self.assertRaises(ValueError):
            slot_classic_lineup([])
if __name__ == "__main__":
    unittest.main()
