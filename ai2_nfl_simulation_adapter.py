"""Fail-closed adapter for the legacy stat-driven NFL simulation engine.

This adapter deliberately does not synthesize missing player rates, certify user
flags, or use sportsbook fields. It is not a complete weekly data pipeline.
"""
import pandas as pd
from dfs_stats_engine import simulate_stat_driven_dfs

REQUIRED_POOL = {"ID", "Name", "Position", "TeamAbbrev", "Salary", "game", "proj", "optimizer_eligible"}
REQUIRED_RATES = {"player", "team", "position", "games_played", "pass_attempts_pg",
                  "carries_pg", "targets_pg"}


def run_verified_stat_simulation(pool, rates, *, n_sims=10000, seed=20261022, out_players=None):
    """Run the existing game-level engine only on explicitly verified inputs.

    Returns the engine's (players, points matrix, stat draws) without marking
    any lineup or prediction as calibrated, profitable, or release-ready.
    """
    if not isinstance(pool, pd.DataFrame) or not isinstance(rates, pd.DataFrame):
        raise ValueError("Pool and historical rates must be DataFrames")
    missing_pool = REQUIRED_POOL - set(pool.columns)
    missing_rates = REQUIRED_RATES - set(rates.columns)
    if missing_pool or missing_rates:
        raise ValueError(f"Missing pool columns: {sorted(missing_pool)}; "
                         f"missing rate columns: {sorted(missing_rates)}")
    if not 100 <= int(n_sims) <= 100000:
        raise ValueError("Simulation count must be 100–100,000")
    if pool.empty or rates.empty:
        raise ValueError("Verified slate and historical rates must be nonempty")
    if not {"weekly_projection_valid", "opportunity_verified"}.issubset(pool.columns):
        raise ValueError("Independent projection and opportunity verification flags are required")
    active = pool[pool["optimizer_eligible"].eq(True)].copy()
    if active.empty or not (active["weekly_projection_valid"].eq(True)
                           & active["opportunity_verified"].eq(True)).all():
        raise ValueError("Every eligible player must have independently verified weekly inputs")
    if active["ID"].astype(str).duplicated().any():
        raise ValueError("Duplicate DraftKings player IDs")
    if active["game"].isna().any() or not active["game"].astype(str).str.contains("@", regex=False).all():
        raise ValueError("Every eligible player needs an explicit scheduled matchup")
    # A role proxy is not observed player history; reject rather than invent
    # enough opportunity to make a depth player appear playable.
    if "source" not in rates.columns:
        raise ValueError("Historical rate provenance/source is required")
    if rates["source"].astype(str).str.contains("ROLE_PROXY", case=False).any():
        raise ValueError("Unverified role-proxy rates are not eligible for official simulation")
    return simulate_stat_driven_dfs(pool, rates, n_sims=int(n_sims),
                                    seed=int(seed), out_players=out_players,
                                    allow_role_proxies=False)
