"""P90 correction must have pregame provenance, correct splits, and no draw changes."""
import unittest
import pandas as pd
import numpy as np
from ai2_nfl_p90_calibration import (
    fit_p90_margins,apply_p90_margin,measure,pinball)


def training(n=150):
    rng=np.random.default_rng(602)
    positions=np.array(["QB","RB","WR","TE"])
    p=positions[np.arange(n)%len(positions)]
    modeled=np.tile(np.array([22.,19.,17.,13.]),int(np.ceil(n/4)))[:n]
    # This test deliberately simulates undercovered forecast quantiles.
    residual=rng.normal(loc=0.,scale=8.,size=n)
    actual=np.maximum(0.,modeled+residual)
    return pd.DataFrame({
        "season":2025,"week":np.arange(n)%10+5,
        "position":p,"simulated_p90_dk":modeled,
        "actual_partial_dk":actual
    })


class P90CalTests(unittest.TestCase):
    def test_training_cannot_see_2026_or_future_2025(self):
        x=training()
        for y,w in [(2026,4),(2025,16),(2025,3)]:
            copy=x.copy()
            copy.loc[0,"season"]=y
            copy.loc[0,"week"]=w
            with self.assertRaisesRegex(ValueError,"2025 weeks 5-14"):
                fit_p90_margins(copy)
    def test_bounded_observation_count_and_invalid_arrays(self):
        with self.assertRaisesRegex(ValueError,"Too few"):
            fit_p90_margins(training(42))
        with self.assertRaises(ValueError):
            pinball([float("nan")],[5.])
    def test_role_correction_positive_only_with_explicit_fallback(self):
        fit=fit_p90_margins(training())
        self.assertTrue(fit["training_rows"]>=150)
        for role in ("QB","RB","WR","TE","UNKNOWN"):
            result=apply_p90_margin(24.,role,fit,mode="role")
            self.assertGreaterEqual(result,24.)
        with self.assertRaisesRegex(ValueError,"Unsupported"):
            apply_p90_margin(20.,"RB",fit,mode="linear")
        with self.assertRaisesRegex(ValueError,"prior-season"):
            apply_p90_margin(20.,"RB",dict(training_season=2026),mode="pooled")
    def test_calibration_does_not_change_simulated_mean_or_game_draws(self):
        train=training()
        fit=fit_p90_margins(train)
        n=60
        holdout=train.iloc[:n].copy()
        before=holdout.to_numpy().copy()
        original_mean=holdout["simulated_p90_dk"].mean()
        report=measure(holdout,fit,mode="pooled")
        self.assertGreaterEqual(report["reported_p90_mean_dk"],original_mean)
        self.assertTrue(np.array_equal(before,holdout.to_numpy()))


if __name__=="__main__":
    unittest.main()
