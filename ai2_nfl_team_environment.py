"""Measured team offensive volume for the unified NFL Classic simulation.

Runs only on nflverse box scores from games BEFORE the slate week.
Results are historical role/game-environment forecasts, not certified snaps.
Weights were chosen in the separate 2025-train/2026-holdout exercise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ai2_nfl_matchups import canonical_team
from props_data import clean_weekly


def measured_team_games(history, season, week):
    """Count offensive passing attempts + QB sacks + all rostered carries.

    These are an observable *box-score proxy* for team scrimmage plays.
    Penalty no-plays and some stat-recording edge cases are absent.
    """
    frame = clean_weekly(history)
    frame = frame[((frame["season"] < int(season)) |
                  ((frame["season"] == int(season)) & (frame["week"] < int(week))))].copy()
    frame["team_norm"] = frame["recent_team"].map(canonical_team)
    if frame.empty:
        return pd.DataFrame(columns=["season", "week", "team", "passes", "plays", "sacks"])
    output = []
    for (yr, wk, team), group in frame.groupby(["season", "week", "team_norm"], sort=True):
        if not team:
            continue
        qbs = group[group["position"].astype(str).str.upper().eq("QB")]
        if qbs.empty:
            continue
        passes = pd.to_numeric(qbs["attempts"], errors="coerce")
        sacks = pd.to_numeric(qbs["sacks_suffered"], errors="coerce")
        carries = pd.to_numeric(group["carries"], errors="coerce")
        if not passes.notna().any() or not sacks.notna().any() or not carries.notna().any():
            continue
        pass_count = float(passes.sum())
        sack_count = float(sacks.sum())
        rush_count = float(carries.sum())
        plays = pass_count + sack_count + rush_count
        if (not np.isfinite([pass_count, sack_count, plays]).all()
                or not (10 <= pass_count <= 65 and 35 <= plays <= 100)):
            continue
        output.append({"season": int(yr), "week": int(wk), "team": team,
                       "passes": pass_count, "sacks": sack_count, "plays": plays})
    return pd.DataFrame(output, columns=["season", "week", "team", "passes", "sacks", "plays"])


def predict_team_environment(history, season, week, *, min_games=3):
    """Observed team-specific pace and passing opportunity for each current-slate team.

    Last six games, 0.85 recency decay, 4-game reliability prior.
    Empirically tested shrinkage: 0.5 for plays, 1.0 for pass attempts.
    Missing/inadequate history is explicit and does NOT fabricate verified pace.
    """
    games = measured_team_games(history, season, week)
    if games.empty:
        return {}
    league = {field: float(games[field].mean()) for field in ("passes", "plays")}
    output = {}
    for team, g in games.groupby("team"):
        recent = g.sort_values(["season", "week"]).tail(6)
        if len(recent) < min_games:
            continue
        elapsed = np.arange(len(recent)-1, -1, -1)
        weights = np.power(.85, elapsed)
        reliability = len(recent)/(len(recent)+4.)
        estimate = {}
        for metric, strength in (("plays", .5), ("passes", 1.)):
            own = float(np.average(recent[metric].to_numpy(float), weights=weights))
            estimate[metric] = round(float(league[metric] +
                strength * reliability * (own - league[metric])), 4)
        if not (48 <= estimate["plays"] <= 80 and 18 <= estimate["passes"] <= 50):
            continue
        output[team] = {"expected_plays": estimate["plays"],
                        "expected_pass_attempts": estimate["passes"],
                        "observed_games": len(recent),
                        "last_observed_season": int(recent.iloc[-1]["season"]),
                        "last_observed_week": int(recent.iloc[-1]["week"]),
                        "source": "nflverse prior-game team box-score volume; 2025-tuned shrinkage"}
    return output
