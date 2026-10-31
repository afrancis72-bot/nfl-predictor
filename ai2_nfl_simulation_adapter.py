"""Fail-closed adapter for the legacy stat-driven NFL simulation engine.

This adapter deliberately does not synthesize missing player rates, certify user
flags, or use sportsbook fields. It is not a complete weekly data pipeline.
"""
import pandas as pd
import numpy as np
from dfs_stats_engine import simulate_stat_driven_dfs

REQUIRED_POOL = {"ID", "Name", "Position", "TeamAbbrev", "Salary", "game", "proj", "optimizer_eligible"}
REQUIRED_RATES = {"player", "team", "position", "games_played", "pass_attempts_pg",
                  "carries_pg", "targets_pg"}


def run_verified_stat_simulation(pool, rates, *, n_sims=10000, seed=20261022, out_players=None,
                                 team_environment=None):
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
    if isinstance(n_sims, bool) or not isinstance(n_sims, int):
        raise ValueError("Simulation count must be an integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Random seed must be an integer")
    if not 100 <= n_sims <= 100000:
        raise ValueError("Simulation count must be 100–100,000")
    if pool.empty or rates.empty:
        raise ValueError("Verified slate and historical rates must be nonempty")
    if not {"weekly_projection_valid", "opportunity_verified"}.issubset(pool.columns):
        raise ValueError("Independent projection and opportunity verification flags are required")
    active = pool[pool["optimizer_eligible"].eq(True)].copy()
    if active.empty or not (active["weekly_projection_valid"].eq(True)
                           & active["opportunity_verified"].eq(True)).all():
        raise ValueError("Every eligible player must have independently verified weekly inputs")
    # A malformed slate must never reach the statistical engine. Its QB
    # selection and lineup ordering depend on finite, positive projections.
    salary = pd.to_numeric(active["Salary"], errors="coerce")
    projection = pd.to_numeric(active["proj"], errors="coerce")
    if (not np.isfinite(salary.to_numpy(dtype=float)).all()
            or not salary.gt(0).all()):
        raise ValueError("Every eligible player requires a positive finite DraftKings salary")
    if (not np.isfinite(projection.to_numpy(dtype=float)).all()
            or not projection.gt(0).all()):
        raise ValueError("Every eligible player requires a positive finite independent projection")
    if not active["Position"].astype(str).str.upper().isin(
            ("QB", "RB", "WR", "TE", "DST")).all():
        raise ValueError("Unsupported NFL roster position")
    if active["ID"].isna().any() or active["ID"].astype(str).str.strip().eq("").any():
        raise ValueError("Every eligible player needs a nonempty DraftKings ID")
    if active["Name"].isna().any() or active["Name"].astype(str).str.strip().eq("").any():
        raise ValueError("Every eligible player needs a nonempty name")
    if active["TeamAbbrev"].isna().any() or active["TeamAbbrev"].astype(str).str.strip().eq("").any():
        raise ValueError("Every eligible player needs a team")
    if active["ID"].astype(str).duplicated().any():
        raise ValueError("Duplicate DraftKings player IDs")
    if active["game"].isna().any() or not active["game"].astype(str).str.contains("@", regex=False).all():
        raise ValueError("Every eligible player needs an explicit scheduled matchup")
    # The legacy engine silently skips games unless both teams are represented.
    # Reject such a slate before a partial zero-point matrix reaches the UI.
    for matchup, group in active.groupby("game"):
        sides = str(matchup).split()[0].split("@")
        if len(sides) != 2 or not all(sides):
            raise ValueError(f"Invalid matchup format: {matchup}")
        from dfs_stats_engine import _team
        expected = {_team(side) for side in sides}
        observed = {_team(team) for team in group["TeamAbbrev"].astype(str)}
        if observed != expected:
            raise ValueError(f"Incomplete matchup {matchup}: expected {sorted(expected)}, "
                             f"found {sorted(observed)}")
    # A role proxy is not observed player history; reject rather than invent
    # enough opportunity to make a depth player appear playable.
    if "source" not in rates.columns:
        raise ValueError("Historical rate provenance/source is required")
    # Only explicitly observed histories can be treated as verified inputs.
    # Arbitrary labels or missing observation counts are not proof of history.
    source = rates["source"].astype("string").str.strip()
    if source.isna().any() or source.eq("").any():
        raise ValueError("Historical source provenance must be nonempty")
    games = pd.to_numeric(rates["games_played"], errors="coerce")
    if games.isna().any() or not games.between(1, 25).all():
        raise ValueError("Historical games_played must be observed and between 1 and 25")
    # nflverse leaves inapplicable opportunity columns null (for example,
    # pass attempts for wide receivers). Do not reject legitimate observations
    # or silently convert missing *applicable* opportunity into zero.
    required_by_position = {
        "QB": ("pass_attempts_pg",),
        "RB": ("carries_pg", "targets_pg"),
        "WR": ("targets_pg",),
        "TE": ("targets_pg",),
    }
    positions = rates["position"].astype(str).str.upper().str.strip()
    for field in ("pass_attempts_pg", "carries_pg", "targets_pg"):
        values = pd.to_numeric(rates[field], errors="coerce")
        applicable = positions.map(lambda pos: field in required_by_position.get(pos, ()))
        if ((applicable & values.isna()) | (values.notna() & ~values.between(0, 100))).any():
            raise ValueError(f"Invalid historical opportunity rate: {field}")
    if rates["source"].astype(str).str.contains("ROLE_PROXY", case=False).any():
        raise ValueError("Unverified role-proxy rates are not eligible for official simulation")
    if team_environment is not None:
        if not isinstance(team_environment, dict):
            raise ValueError("Team environment must be an observed source dictionary")
        for team, profile in team_environment.items():
            if not isinstance(profile, dict) or not str(profile.get("source") or "").startswith("nflverse"):
                raise ValueError("Team environment requires actual nflverse source evidence")
            plays = float(profile.get("expected_plays", float("nan")))
            attempts = float(profile.get("expected_pass_attempts", float("nan")))
            games = int(profile.get("observed_games", 0))
            if games < 3 or not (np.isfinite(plays) and 48 <= plays <= 80
                                  and np.isfinite(attempts) and 18 <= attempts <= 50):
                raise ValueError("Unverified or implausible team game environment")
    return simulate_stat_driven_dfs(active, rates, n_sims=int(n_sims),
                                    seed=int(seed), out_players=out_players,
                                    allow_role_proxies=False, team_environment=team_environment)
