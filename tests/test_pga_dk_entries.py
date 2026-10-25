"""PGA DK template exporter validates entry IDs, salaries and six golfer slots."""
import io
import unittest
import pandas as pd

from sports.pga.dk_entries import fill_entries_template


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
    csv_text = (
        "Entry ID,Contest Name,Contest ID,Entry Fee,G,G,G,G,G,G,,Instructions\n"
        "995011,PGA Test,456,$0.50,,,,,,,,1. First entry\n"
        "995012,PGA Test,456,$0.50,,,,,,,,2. Second entry\n"
    )
    template = pd.read_csv(io.StringIO(csv_text), dtype=str)
    return portfolio, golfers, template


class PGAEntriesTests(unittest.TestCase):
    def test_fills_entries_and_preserves_contest_metadata(self):
        portfolio, golfers, template = fixtures()
        output = fill_entries_template(portfolio, golfers, template)
        self.assertEqual(output.shape, template.shape)
        self.assertEqual(output["Entry ID"].tolist(), template["Entry ID"].tolist())
        self.assertEqual(output["Contest ID"].tolist(), template["Contest ID"].tolist())
        self.assertEqual([c for c in output.columns if c == "G"], ["G"] * 6)
        self.assertEqual(output.iloc[0, 4], "Golfer 1 (10101)")
        self.assertEqual(output.iloc[1, 9], "Golfer 12 (10112)")
        self.assertTrue(all(" (" in str(v) for v in output.iloc[:, 4:10].to_numpy().ravel()))

    def test_rejects_missing_dk_ids(self):
        portfolio, golfers, template = fixtures()
        golfers.loc[0, "ID"] = None
        with self.assertRaisesRegex(ValueError, "missing or malformed"):
            fill_entries_template(portfolio, golfers, template)

    def test_rejects_over_cap_lineup(self):
        portfolio, golfers, template = fixtures()
        portfolio.loc[:5, "salary"] = 8400
        with self.assertRaisesRegex(ValueError, "salary cap"):
            fill_entries_template(portfolio, golfers, template)

    def test_rejects_template_without_six_slots(self):
        portfolio, golfers, template = fixtures()
        template = template.drop(columns=["G.5"])
        with self.assertRaisesRegex(ValueError, "exactly six golfer"):
            fill_entries_template(portfolio, golfers, template)

    def test_rejects_unmapped_golfer(self):
        portfolio, golfers, template = fixtures()
        portfolio.loc[0, "player"] = "Unlisted Golfer"
        with self.assertRaisesRegex(ValueError, "missing from current DraftKings"):
            fill_entries_template(portfolio, golfers, template)


if __name__ == "__main__":
    unittest.main(verbosity=2)
