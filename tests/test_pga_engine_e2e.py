"""End-to-end synthetic PGA engine tests for the unified development branch."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "sports" / "pga" / "legacy"
sys.path.insert(0, str(ROOT))

from pga_predictor_pro import Config, predict_from_dataframes
from golf_first_v11 import simulate_golf_first
from portfolio_optimizer import PortfolioSettings, optimize_portfolio


def synthetic_field():
    n = 24
    return pd.DataFrame({
        "player": [f"Fixture Golfer {i:02d}" for i in range(n)],
        "salary": [6900 + 200 * (i % 15) for i in range(n)],
        "owgr": list(range(1, n + 1)),
        "sg_total": np.linspace(-.8, 1.4, n),
        "sg_approach": np.linspace(-.4, 1.2, n),
        "sg_ott": np.linspace(-.2, .5, n),
        "sg_arg": np.linspace(-.1, .6, n),
        "sg_putting": np.linspace(-.4, .8, n),
    })


def run_engine():
    classic = predict_from_dataframes(
        Config(tournament="SyntheticFixture", sims=100, seed=207),
        synthetic_field(),
    )
    result, draws, diagnostic = simulate_golf_first(
        classic, sims=100, seed=207, course_par=72, no_cut=False, cut_size=12
    )
    result["dk_points_proxy"] = result.v11_mean_dk
    result["make_cut_pct"] = result.v11_make_cut_pct / 100
    result["top10_pct"] = result.v11_top10_pct / 100
    result["win_pct"] = result.v11_win_pct / 100
    return result, draws, diagnostic


class PGAEngineEndToEnd(unittest.TestCase):
    def test_determinism_and_sanity(self):
        pred, draws, diag = run_engine()
        _, other_draws, _ = run_engine()
        np.testing.assert_array_equal(draws, other_draws)
        self.assertEqual(draws.shape, (100, 24))
        self.assertTrue(np.isfinite(draws).all())
        # Negative outcomes are possible under DraftKings golf scoring.
        self.assertGreater(float(draws.mean()), 0)
        self.assertFalse(diag["no_cut"])
        np.testing.assert_allclose(pred.v11_mean_dk, draws.mean(axis=0), atol=1e-5)
        for col in ("v11_make_cut_pct", "v11_win_pct", "v11_top10_pct"):
            self.assertTrue(pred[col].between(0, 100).all(), col)
        self.assertTrue((pred.v11_p90_dk >= pred.v11_median_dk).all())

    def test_portfolio_legality(self):
        pred, _, _ = run_engine()
        params = PortfolioSettings(
            lineup_count=2, salary_cap=50000, salary_floor=45000,
            min_player_salary=6500, roster_size=6, max_exposure=.50,
            min_unique=3, seed=207, candidate_samples=9000, target_candidates=2000
        )
        portfolio, summary, exposure = optimize_portfolio(pred, params)
        self.assertEqual(len(summary), 2)
        self.assertTrue(summary.salary.between(45000, 50000).all())
        self.assertTrue(portfolio.groupby("lineup").player.nunique().eq(6).all())
        self.assertTrue(exposure.lineups.le(1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
