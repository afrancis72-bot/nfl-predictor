"""Test PGA DraftKings Entries import/export using a realistic embedded salary table."""
import io
import unittest

import pandas as pd

from sports.pga.dk_entries import fill_entries_template
from sports.pga.dk_template import parse_dk_entries


def example_csv():
    return (
        "Entry ID,Contest Name,Contest ID,Entry Fee,G,G,G,G,G,G,,Instructions\n"
        "995011,PGA Mini,456,$0.50,,,,,,,,1. Contest entries\n"
        "995012,PGA Mini,456,$0.50,,,,,,,,2. Current lineup\n"
        ",,,,,,,,,,,\n"
        ",,,,,,,,,,,Position,Name + ID,Name,ID,Salary\n"
        ",,,,,,,,,,,G,Golfer 1 (10101),Golfer 1,10101,7800\n"
    ).encode("utf-8")


def fixtures():
    golfers = pd.DataFrame({
        "player": [f"Golfer {n}" for n in range(1, 13)],
        "ID": [str(10100 + n) for n in range(1, 13)],
        "salary": [7800] * 12,
    })
    portfolio = pd.DataFrame({
        "lineup": [1] * 6 + [2] * 6,
        "player": golfers.player.tolist(),
        "salary": [7800] * 12,
    })
    return portfolio, golfers


class PGARealTemplateTests(unittest.TestCase):
    def test_parser_ignores_embedded_dk_salary_table(self):
        template = parse_dk_entries(example_csv())
        self.assertEqual(len(template), 2)
        self.assertEqual(template["Entry ID"].tolist(), ["995011", "995012"])
        self.assertEqual(template.columns.tolist().count("G"), 1)
        self.assertEqual(
            [c for c in template.columns if c == "G" or c.startswith("G.")],
            ["G", "G.1", "G.2", "G.3", "G.4", "G.5"]
        )

    def test_fills_native_template_and_preserves_contest(self):
        portfolio, golfers = fixtures()
        entries = parse_dk_entries(example_csv())
        completed = fill_entries_template(portfolio, golfers, entries)
        self.assertEqual(completed["Entry ID"].tolist(), ["995011", "995012"])
        self.assertEqual(completed["Contest ID"].tolist(), ["456", "456"])
        self.assertEqual(completed.iloc[0, 4], "Golfer 1 (10101)")
        self.assertEqual(completed.iloc[1, 9], "Golfer 12 (10112)")
        self.assertEqual(completed.columns.tolist().count("G"), 6)
        # Output is a standard CSV with six repeated G headers, not G.1 etc.
        roundtrip = pd.read_csv(io.StringIO(completed.to_csv(index=False)), dtype=str)
        self.assertEqual(roundtrip["G.5"].iloc[1], "Golfer 12 (10112)")

    def test_rejects_stale_salary(self):
        portfolio, golfers = fixtures()
        golfers.loc[0, "salary"] = 7900
        with self.assertRaisesRegex(ValueError, "salary differs"):
            fill_entries_template(portfolio, golfers, parse_dk_entries(example_csv()))

    def test_rejects_extra_lineups_without_entries(self):
        portfolio, golfers = fixtures()
        with self.assertRaisesRegex(ValueError, "more lineups"):
            fill_entries_template(
                portfolio, golfers,
                parse_dk_entries(example_csv()).iloc[:1].copy()
            )

    def test_rejects_wrong_sport(self):
        with self.assertRaisesRegex(ValueError, "six G"):
            parse_dk_entries(example_csv().replace(b"G,G,G,G,G,G", b"D,D,D,D,D,D"))

    def test_rejects_duplicate_entry_ids(self):
        template = example_csv().replace(b"995012", b"995011")
        with self.assertRaisesRegex(ValueError, "Duplicate Entry IDs"):
            parse_dk_entries(template)


if __name__ == "__main__":
    unittest.main(verbosity=2)
