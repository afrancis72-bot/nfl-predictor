"""Simulation portfolio tests with strict full-roster validation."""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from ai2_nfl_sim_portfolio import select_simulated_portfolio


class SimPortfolioTests(unittest.TestCase):
    def test_missing_dst_draws_fail_closed(self):
        roster = [{"ID": str(i), "Position": p, "Salary": 5500}
                  for i, p in enumerate(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR", "DST"])]
        players = pd.DataFrame({"ID": [str(i) for i in range(8)]})
        with patch("ai2_nfl_sim_portfolio.check_classic_lineup", return_value=[]):
            with self.assertRaisesRegex(ValueError, "including DST"):
                select_simulated_portfolio([{"players": roster}], players,
                                           np.ones((100, 8)))

    def test_scores_shared_scenarios_and_exports_candidate(self):
        roster = [{"ID": str(i), "Position": p, "Salary": 5500}
                  for i, p in enumerate(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR", "DST"])]
        players = pd.DataFrame({"ID": [str(i) for i in range(9)]})
        draws = np.tile(np.arange(100, dtype=float)[:, None], (1, 9))
        with patch("ai2_nfl_sim_portfolio.check_classic_lineup", return_value=[]):
            result = select_simulated_portfolio([{"players": roster}], players, draws,
                                                count=1, max_exposure=1)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["simulated_mean"], 445.5)

    def test_invalid_draws_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid simulation"):
            select_simulated_portfolio([], pd.DataFrame({"ID": ["1"]}),
                                       np.full((100, 1), np.nan))


if __name__ == "__main__":
    unittest.main()
