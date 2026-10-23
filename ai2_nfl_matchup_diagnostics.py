"""Pregame opponent-adjustment diagnostics derived only from observed NFL player stats.

This is an exploratory matchup model, NOT verified player opportunity or a
complete DST projection. No target/future-week statistics are permitted.
"""
import numpy as np
import pandas as pd
from props_data import clean_weekly, included_history, dk_classic_points
from ai2_nfl_matchups import canonical_team

def defensive_allowance(history, season, week, *, prior_seasons=1, shrink_games=8):
    """Aggregate opponent offense by position and defending team.

    nflverse player-week records include opponent_team in standard releases.
    If absent, fail explicitly instead of inventing defensive matchups.
    Return position-level observed DK partial points per opposing player-game,
    shrunk toward league position averages for sparse defensive samples.
    """
    if not isinstance(shrink_games, (int, float)) or shrink_games <= 0:
        raise ValueError("shrink_games must be positive")
    h = included_history(history, season, week, prior_seasons=prior_seasons)
    if "opponent_team" not in h.columns:
        raise ValueError("nflverse player-week feed lacks opponent_team; matchup adjustments unavailable")
    if h.empty:
        raise ValueError("No observed pregame opponent history")
    h = h.copy()
    h["defense"] = h["opponent_team"].map(canonical_team)
    h["position"] = h["position"].astype(str).str.upper()
    h = h[h["defense"].str.len().between(2, 3)]
    if h.empty:
        raise ValueError("No valid defending teams in observed history")
    scores = []
    for _, r in h.iterrows():
        score = dk_classic_points(r.to_dict(), r["position"])
        if np.isfinite(score):
            scores.append((r["defense"], r["position"], float(score)))
    if not scores:
        raise ValueError("No complete historical scoring observations for matchup adjustment")
    d = pd.DataFrame(scores, columns=["defense", "position", "partial_dk"])
    league = d.groupby("position")["partial_dk"].mean().to_dict()
    grouped = d.groupby(["defense", "position"])["partial_dk"].agg(["mean", "count"]).reset_index()
    grouped["league_mean"] = grouped["position"].map(league)
    grouped["shrunk_allowed"] = ((grouped["count"] * grouped["mean"]
                                  + shrink_games * grouped["league_mean"])
                                 / (grouped["count"] + shrink_games))
    grouped["multiplier"] = (grouped["shrunk_allowed"] /
                              grouped["league_mean"].replace(0, np.nan)).clip(0.75, 1.25)
    grouped["source"] = "nflverse observed pregame opponent_team, player-week partial DK"
    return grouped.rename(columns={"count": "player_game_observations",
                                   "mean": "observed_allowed_per_player_game"})

def apply_matchup_diagnostics(baselines, allowances):
    """Add provisional matchup-adjusted score; never change verification flags."""
    out = baselines.copy()
    lookups = allowances.set_index(["defense", "position"])["multiplier"].to_dict()
    out["matchup_multiplier"] = [lookups.get((canonical_team(opp), pos), np.nan)
                                 for opp, pos in zip(out["opponent"], out["Position"])]
    out["matchup_adjusted_partial_dk"] = pd.to_numeric(
        out["historical_partial_dk"], errors="coerce") * out["matchup_multiplier"]
    out["matchup_model_note"] = "exploratory defense allowance; no snap, injury, or role certification"
    return out
