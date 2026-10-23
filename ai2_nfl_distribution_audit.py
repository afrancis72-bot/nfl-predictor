"""Out-of-sample distribution diagnostics for NFL simulated fantasy points.

This does not establish profitability or calibration on its own. An observed
score must come from the same game/slate and must not be used to fit the draws.
"""
import numpy as np
import pandas as pd


def distribution_diagnostics(players, draws, actuals):
    """Compare player predictive distributions against observed DK points.

    actuals requires ID and actual_dk columns. Duplicate IDs, nonfinite values
    and unmatched rows fail closed. Coverage uses central 50% and 80% intervals.
    """
    if not isinstance(players, pd.DataFrame) or not isinstance(actuals, pd.DataFrame):
        raise ValueError("Players and actuals must be DataFrames")
    if not {"ID", "Name", "Position"}.issubset(players.columns):
        raise ValueError("Simulated players need ID, Name and Position")
    if not {"ID", "actual_dk"}.issubset(actuals.columns):
        raise ValueError("Observed results need ID and actual_dk")
    x = np.asarray(draws, dtype=float)
    if x.ndim != 2 or x.shape[0] < 100 or x.shape[1] != len(players):
        raise ValueError("Draw matrix dimensions do not match the player pool")
    if not np.isfinite(x).all():
        raise ValueError("Simulation draws must be finite")
    ids = players["ID"].astype("string").str.strip()
    actual_ids = actuals["ID"].astype("string").str.strip()
    if ids.isna().any() or actual_ids.isna().any() or ids.duplicated().any() or actual_ids.duplicated().any():
        raise ValueError("Player IDs must be present and unique")
    truth = pd.to_numeric(actuals["actual_dk"], errors="coerce")
    if not np.isfinite(truth.to_numpy(dtype=float)).all():
        raise ValueError("Observed fantasy points must be finite")
    # Compare the same complete game/slate, not a hand-picked subset of winners
    # or healthy players. Extra result IDs are permitted (non-simulated athletes).
    # The caller must still attest the input slate was frozen before kickoff.
    if "game" in players.columns and "game" in actuals.columns:
        expected_games = players["game"].astype("string").str.strip()
        observed_games = actuals["game"].astype("string").str.strip()
        if expected_games.isna().any() or expected_games.eq("").any():
            raise ValueError("Simulated matchup labels are missing")
        if observed_games.isna().any() or observed_games.eq("").any():
            raise ValueError("Observed matchup labels are missing")
        game_lookup = dict(zip(actual_ids, observed_games))
        if any(i in game_lookup and game_lookup[i] != g
               for i, g in zip(ids, expected_games)):
            raise ValueError("Observed and simulated matchup labels disagree")
    # Exact historical comparison must never include an athlete from a
    # different contest date. A single matchup label cannot establish date.
    if "slate_date" in players.columns or "slate_date" in actuals.columns:
        if "slate_date" not in players.columns or "slate_date" not in actuals.columns:
            raise ValueError("Both sources must provide slate_date when either does")
        predicted_dates = players["slate_date"].astype("string").str.strip()
        observed_dates = actuals["slate_date"].astype("string").str.strip()
        if predicted_dates.isna().any() or predicted_dates.eq("").any() or observed_dates.isna().any() or observed_dates.eq("").any():
            raise ValueError("Slate dates must be present")
        date_lookup = dict(zip(actual_ids, observed_dates))
        if any(i in date_lookup and date_lookup[i] != d for i, d in zip(ids, predicted_dates)):
            raise ValueError("Observed and simulated slate dates disagree")
    lookup = dict(zip(actual_ids, truth))
    if not ids.isin(lookup).all():
        raise ValueError("Missing observed scores for one or more simulated players")
    y = np.asarray([lookup[i] for i in ids], dtype=float)
    mean = x.mean(axis=0)
    q10, q25, q50, q75, q90 = np.percentile(x, [10, 25, 50, 75, 90], axis=0)
    result = pd.DataFrame({
        "ID": ids.to_numpy(), "Name": players["Name"].astype(str).to_numpy(),
        "Position": players["Position"].astype(str).to_numpy(),
        "actual_dk": y, "sim_mean": mean, "sim_median": q50,
        "sim_p10": q10, "sim_p25": q25, "sim_p75": q75, "sim_p90": q90,
        "absolute_error": np.abs(mean-y),
        "covered_50": (y >= q25) & (y <= q75),
        "covered_80": (y >= q10) & (y <= q90),
        "pit": (x < y[None, :]).mean(axis=0)
               + 0.5*(x == y[None, :]).mean(axis=0),
    })
    return result


def distribution_summary(diagnostics):
    """Aggregated accuracy, coverage and probability integral transform."""
    if diagnostics.empty:
        raise ValueError("No player results to summarize")
    return {
        "players": int(len(diagnostics)),
        "mean_absolute_error": float(diagnostics["absolute_error"].mean()),
        "central_50_coverage": float(diagnostics["covered_50"].mean()),
        "central_80_coverage": float(diagnostics["covered_80"].mean()),
        "mean_pit": float(diagnostics["pit"].mean()),
    }
