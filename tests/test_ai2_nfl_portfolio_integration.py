"""Synthetic integration smoke test: selection -> legality -> DraftKings CSV.

This is NOT a historical or current-slate performance validation.
"""
import csv
import io
import random
import unittest
from collections import Counter

import numpy as np
import pandas as pd

from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_sim_portfolio import select_simulated_portfolio
from ai2_nfl_export import export_classic_candidates


class PortfolioIntegrationTests(unittest.TestCase):
    def test_twenty_legal_entries_and_exposure_in_export(self):
        rng = random.Random(17)
        pool = []
        for position, n in (("QB", 12), ("RB", 28), ("WR", 40),
                            ("TE", 16), ("DST", 12)):
            for j in range(n):
                pool.append({"ID": str(len(pool)+1), "Name": f"{position}{j}",
                             "Position": position, "TeamAbbrev": f"T{j%12:02d}",
                             "opponent": "ZZZ", "Salary": 5500,
                             "proj": rng.uniform(5, 25)})
        by_position = {p: [row for row in pool if row["Position"] == p]
                       for p in ("QB", "RB", "WR", "TE", "DST")}
        candidates, seen = [], set()
        for _ in range(10000):
            roster = (rng.sample(by_position["QB"], 1) +
                      rng.sample(by_position["RB"], 2) +
                      rng.sample(by_position["WR"], 4) +
                      rng.sample(by_position["TE"], 1) +
                      rng.sample(by_position["DST"], 1))
            if check_classic_lineup(roster):
                continue
            signature = tuple(sorted(p["ID"] for p in roster))
            if signature not in seen:
                seen.add(signature)
                candidates.append({"players": roster,
                                   "projected_points": sum(p["proj"] for p in roster)})
        candidates.sort(key=lambda c: c["projected_points"], reverse=True)
        candidates = candidates[:120]
        draws = np.random.default_rng(5).normal(15, 8, (1000, len(pool))).clip(0)
        selected = select_simulated_portfolio(candidates, pd.DataFrame(pool),
                                               draws, count=20, max_exposure=0.35)
        self.assertEqual(len(selected), 20)
        self.assertEqual(Counter(x["portfolio_role"] for x in selected),
                         {"balanced": 15, "upside": 5})
        exposures = Counter(p["ID"] for x in selected for p in x["candidate"]["players"])
        self.assertLessEqual(max(exposures.values()), 7)
        self.assertTrue(all(not check_classic_lineup(x["candidate"]["players"])
                            for x in selected))
        exported = export_classic_candidates([x["candidate"] for x in selected])
        rows = list(csv.reader(io.StringIO(exported)))
        self.assertEqual(rows[0], ["QB", "RB", "RB", "WR", "WR", "WR",
                                   "TE", "FLEX", "DST"])
        self.assertEqual(len(rows), 21)
        self.assertTrue(all(len(row) == len(set(row)) == 9 for row in rows[1:]))


if __name__ == "__main__":
    unittest.main()
