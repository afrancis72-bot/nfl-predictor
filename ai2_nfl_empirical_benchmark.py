"""Leakage-safe empirical NFL player-score distribution baseline.

This is a *benchmark*, not the correlated game simulator. Uses only player
games preceding the target week and compares predictive intervals to outcomes.
"""
import numpy as np
import pandas as pd
from props_data import clean_weekly, dk_classic_points


def walkforward_empirical_distribution(history, season, first_week=4, last_week=18,
                                       window=8, min_history=3):
    h = clean_weekly(history).copy()
    h["_partial_dk"] = h.apply(lambda row: dk_classic_points(row, row.position), axis=1)
    h = h[np.isfinite(h["_partial_dk"])].copy()
    if h.empty:
        return pd.DataFrame()
    # Use stable player IDs where present. Without an ID, match name+team
    # conservatively rather than merge unrelated athletes.
    if "player_id" in h.columns:
        ids = h["player_id"].astype("string").str.strip()
        h["_identity"] = "NAME:" + h.player_name.astype(str).str.casefold() + "|TEAM:" + h.recent_team.astype(str)
        good = ids.notna() & ids.ne("")
        h.loc[good, "_identity"] = "ID:" + ids.loc[good]
    else:
        h["_identity"] = "NAME:" + h.player_name.astype(str).str.casefold() + "|TEAM:" + h.recent_team.astype(str)
    out = []
    season = int(season)
    for week in range(int(first_week), int(last_week)+1):
        target = h[(h.season == season) & (h.week == week)]
        previous = h[((h.season < season) | ((h.season == season) & (h.week < week)))
                     & (h.season >= season-1)]
        if target.empty or previous.empty:
            continue
        for _, row in target.iterrows():
            past = previous[previous["_identity"] == row["_identity"]].sort_values(["season", "week"]).tail(int(window))
            if len(past) < int(min_history):
                continue
            scores = past["_partial_dk"].to_numpy(dtype=float)
            q10, q25, q50, q75, q90 = np.percentile(scores, [10, 25, 50, 75, 90])
            observed = float(row["_partial_dk"])
            out.append({
                "season": season, "week": week, "player": row.player_name,
                "team": row.recent_team, "position": row.position,
                "history_games": len(scores), "actual_partial_dk": observed,
                "baseline_mean": float(scores.mean()),
                "absolute_error": abs(float(scores.mean())-observed),
                "covered_50": bool(q25 <= observed <= q75),
                "covered_80": bool(q10 <= observed <= q90),
                "p10": q10, "p25": q25, "median": q50, "p75": q75, "p90": q90,
            })
    return pd.DataFrame(out)
