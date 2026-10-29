"""Transparent, deliberately uncalibrated NBA player-prop research simulations."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = {"player", "team", "opponent", "market", "line", "projection", "stdev", "side"}

def validate_props(data: pd.DataFrame) -> pd.DataFrame:
    x = data.copy()
    x.columns = x.columns.astype(str).str.strip().str.lower()
    missing = REQUIRED.difference(x.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    for col in ("player", "team", "opponent", "market", "side"):
        x[col] = x[col].fillna("").astype(str).str.strip()
    x["side"] = x["side"].str.title()
    if (~x["side"].isin(["Over", "Under"])).any():
        raise ValueError("side must be Over or Under for each prop.")
    for col in ("line", "projection", "stdev"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if x[["line", "projection", "stdev"]].isna().any().any():
        raise ValueError("line, projection, and stdev must be numeric.")
    if (x["stdev"] <= 0).any() or (x["projection"] < 0).any():
        raise ValueError("stdev must be positive and projection nonnegative.")
    if (x[["player", "team", "opponent", "market"]] == "").any().any():
        raise ValueError("Player, team, opponent, and market are required.")
    if len(x) > 150:
        raise ValueError("Limit prototype to 150 prop lines per run.")
    x["game"] = x.apply(lambda r: "-".join(sorted([r["team"].upper(), r["opponent"].upper()])), axis=1)
    x["label"] = [
        f"{i + 1}. {r.player} · {r.market} {r.side} {r.line:g}"
        for i, r in enumerate(x.itertuples(index=False))
    ]
    return x.reset_index(drop=True)

def simulate(data: pd.DataFrame, trials: int = 10000, seed: int = 42):
    """Correlated Gaussian factor prototype. Factors are hypotheses, NOT fitted statistics.

    A game-wide, team-wide and player-wide shock enter each player's sampled stat.
    Standard-deviation contribution is normalized to 1, independently of the number
    of props per player. Scores are floored at 0. No unsupported injury adjustments.
    """
    x = validate_props(data)
    if trials < 1000 or trials > 100000:
        raise ValueError("trials must be between 1,000 and 100,000")
    rng = np.random.default_rng(seed)
    games = {g: rng.standard_normal(trials) for g in x["game"].unique()}
    teams = {t: rng.standard_normal(trials) for t in x["team"].str.upper().unique()}
    players = {(r.game, r.team.upper(), r.player): rng.standard_normal(trials) for r in x.itertuples()}
    game_weight, team_weight, player_weight = 0.22, 0.22, 0.34
    residual_weight = np.sqrt(1 - game_weight**2 - team_weight**2 - player_weight**2)
    hits = np.empty((trials, len(x)), dtype=bool)
    sampled = np.empty((trials, len(x)), dtype=np.float32)
    for j, r in enumerate(x.itertuples(index=False)):
        z = (game_weight * games[r.game]
             + team_weight * teams[r.team.upper()]
             + player_weight * players[(r.game, r.team.upper(), r.player)]
             + residual_weight * rng.standard_normal(trials))
        draws = np.maximum(0, r.projection + r.stdev * z)
        sampled[:, j] = draws
        hits[:, j] = draws > r.line if r.side == "Over" else draws < r.line
    summary = x[["label", "player", "team", "market", "side", "line", "projection", "stdev"]].copy()
    summary["sim_mean"] = sampled.mean(axis=0)
    summary["hit_rate"] = hits.mean(axis=0)
    summary["p90"] = np.quantile(sampled, 0.90, axis=0)
    return summary, hits

def parlay_probability(hits: np.ndarray, indices) -> float:
    idx = list(indices)
    if not idx:
        raise ValueError("Select at least one prop for a parlay.")
    return float(hits[:, idx].all(axis=1).mean())

def expected_net_roi(probability: float, american_odds: int) -> float:
    if american_odds == 0 or -100 < american_odds < 100:
        raise ValueError("Enter standard American odds such as +450 or -110.")
    profit = american_odds / 100 if american_odds > 0 else 100 / abs(american_odds)
    return probability * profit - (1 - probability)
