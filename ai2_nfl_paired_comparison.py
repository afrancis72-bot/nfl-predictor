"""Paired evaluation of correlated NFL simulation against empirical benchmark.

Requires exactly the same observed player-games in both sources. This prevents
the more complex model from appearing superior merely by excluding hard cases.
The comparison is descriptive; it is not a significance test or profitability claim.
"""
import numpy as np
import pandas as pd


def compare_matched_predictions(simulated, empirical):
    """Return player-level paired errors and model-level coverage metrics.

    simulated: slate_id, ID, actual_dk, sim_mean, covered_50, covered_80
    empirical: slate_id, ID, actual_dk, baseline_mean, covered_50, covered_80
    Both inputs must represent the same full set of player-game identifiers.
    """
    keys = ["slate_id", "ID"]
    required_sim = set(keys + ["actual_dk", "sim_mean", "covered_50", "covered_80"])
    required_emp = set(keys + ["actual_dk", "baseline_mean", "covered_50", "covered_80"])
    if not isinstance(simulated, pd.DataFrame) or not isinstance(empirical, pd.DataFrame):
        raise ValueError("Both audit inputs must be DataFrames")
    if not required_sim.issubset(simulated) or not required_emp.issubset(empirical):
        raise ValueError("Missing required paired-audit columns")
    if simulated.empty or empirical.empty:
        raise ValueError("Cannot compare empty historical audits")
    left, right = simulated.copy(), empirical.copy()
    for frame in (left, right):
        for key in keys:
            frame[key] = frame[key].astype("string").str.strip()
            if frame[key].isna().any() or frame[key].eq("").any():
                raise ValueError("Missing historical slate or player ID")
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate historical player-game keys")
    paired = left.merge(right, on=keys, how="outer", suffixes=("_sim", "_emp"),
                        indicator=True, validate="one_to_one")
    if not paired["_merge"].eq("both").all():
        raise ValueError("Models must cover exactly the same player-games")
    for field in ("actual_dk_sim", "actual_dk_emp", "sim_mean", "baseline_mean"):
        paired[field] = pd.to_numeric(paired[field], errors="coerce")
        if not np.isfinite(paired[field].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite historical values: {field}")
    if not np.allclose(paired["actual_dk_sim"], paired["actual_dk_emp"], rtol=0, atol=1e-6):
        raise ValueError("Models disagree on observed fantasy points")
    paired["sim_absolute_error"] = (paired.sim_mean-paired.actual_dk_sim).abs()
    paired["baseline_absolute_error"] = (paired.baseline_mean-paired.actual_dk_sim).abs()
    paired["sim_improvement"] = paired.baseline_absolute_error-paired.sim_absolute_error
    for field in ("covered_50_sim", "covered_80_sim", "covered_50_emp", "covered_80_emp"):
        if paired[field].isna().any() or not paired[field].isin([True, False]).all():
            raise ValueError(f"Invalid interval coverage: {field}")
    summary = pd.DataFrame([
        {"model": "correlated_simulation",
         "player_games": len(paired),
         "mean_absolute_error": paired.sim_absolute_error.mean(),
         "central_50_coverage": paired.covered_50_sim.mean(),
         "central_80_coverage": paired.covered_80_sim.mean()},
        {"model": "empirical_baseline",
         "player_games": len(paired),
         "mean_absolute_error": paired.baseline_absolute_error.mean(),
         "central_50_coverage": paired.covered_50_emp.mean(),
         "central_80_coverage": paired.covered_80_emp.mean()},
    ])
    return paired.drop(columns=["_merge"]), summary
