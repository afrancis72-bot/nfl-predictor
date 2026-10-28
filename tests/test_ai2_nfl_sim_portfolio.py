"""Simulation portfolio tests with strict full-roster validation."""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from ai2_nfl_sim_portfolio import select_simulated_portfolio, _solve_joint_portfolio


class SimPortfolioTests(unittest.TestCase):
    def test_missing_dst_draws_fail_closed(self):
        roster = [{"ID": str(i), "Position": p, "Salary": 5500}
                  for i, p in enumerate(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR", "DST"])]
        players = pd.DataFrame({"ID": [str(i) for i in range(8)]})
        with patch("ai2_nfl_sim_portfolio.check_classic_lineup", return_value=[]):
            with self.assertRaisesRegex(ValueError, "including DST"):
                select_simulated_portfolio([{"players": roster, "salary": 49500}], players,
                                           np.ones((100, 8)), premium_count=0, require_opportunity=False)

    def test_scores_shared_scenarios_and_exports_candidate(self):
        roster = [{"ID": str(i), "Position": p, "Salary": 5500}
                  for i, p in enumerate(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR", "DST"])]
        players = pd.DataFrame({"ID": [str(i) for i in range(9)]})
        draws = np.tile(np.arange(100, dtype=float)[:, None], (1, 9))
        with patch("ai2_nfl_sim_portfolio.check_classic_lineup", return_value=[]):
            result = select_simulated_portfolio([{"players": roster, "salary": 49500}], players, draws,
                                                count=1, max_exposure=1, premium_count=0, require_opportunity=False)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["simulated_mean"], 445.5)

    def test_partial_portfolio_is_rejected(self):
        roster = [{"ID": str(i), "Position": p, "Salary": 5500}
                  for i, p in enumerate(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR", "DST"])]
        players = pd.DataFrame({"ID": [str(i) for i in range(9)]})
        with patch("ai2_nfl_sim_portfolio.check_classic_lineup", return_value=[]):
            with self.assertRaisesRegex(ValueError, "no partial portfolio"):
                select_simulated_portfolio([{"players": roster, "salary": 49500}], players,
                                           np.ones((100, 9)), count=2,
                                           max_exposure=0.5, premium_count=0, require_opportunity=False)

    def test_premium_only_exclusion_keeps_gray_eligible_for_regular(self):
        # The user's Week 8 decision blocks Gray from premium entries,
        # not from all 20 or from any future slate.
        def scored(ident, team, ceiling):
            return {"candidate": {"players": [
                {"ID": ident, "Name": "Noah Gray" if ident == "gray" else "Other TE",
                 "Position": "TE", "TeamAbbrev": team, "Salary": 2900}],
                "salary": 49800},
                "simulated_mean": ceiling / 2,
                "simulated_p90": ceiling * 0.85, "simulated_p99": ceiling}
        chosen, premium_ids = _solve_joint_portfolio(
            [scored("gray", "KC", 17), scored("other", "BUF", 15)],
            count=2, premium_count=1, max_exposure=1.0,
            upside_share=0.5, value_salary_ceiling=5000,
            value_max_appearances=2, premium_value_max_appearances=1,
            premium_excluded_ids={"gray"})
        self.assertEqual(len(chosen), 2)
        self.assertIn("gray", [str(p["ID"]) for x in chosen
                              if x["contest_tier"] == "$0.50"
                              for p in x["candidate"]["players"]])
        self.assertEqual([str(p["ID"]) for x in chosen
                          if x["contest_tier"] == "$3"
                          for p in x["candidate"]["players"]], ["other"])

    def test_invalid_draws_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid simulation"):
            select_simulated_portfolio([], pd.DataFrame({"ID": ["1"]}),
                                       np.full((100, 1), np.nan))


if __name__ == "__main__":
    unittest.main()
