"""Historical opponent offense indicators for DST, not a complete DST projection.

Player-week data cannot independently establish DST sacks, fumble recoveries,
return touchdowns or official points allowed. Never certify this as DK DST score.
"""
import numpy as np
import pandas as pd
from props_data import included_history
from ai2_nfl_matchups import canonical_team

def historical_dst_indicators(history, season, week, *, shrink_games=6):
    """Pregame defending-team game averages from opposing QB/RB production.

    Only uses observed weeks preceding the requested target. Turnovers here are
    opposing QB passing interceptions; opponent offensive TDs are partial
    indicators, NOT official points allowed or DST fantasy points.
    """
    if shrink_games <= 0:
        raise ValueError("shrink_games must be positive")
    h = included_history(history, season, week)
    required = {"opponent_team", "recent_team", "position", "passing_interceptions",
                "passing_tds", "rushing_tds", "season", "week"}
    missing = required - set(h.columns)
    if missing:
        raise ValueError("Missing historical DST indicator fields: " + ", ".join(sorted(missing)))
    if h.empty:
        raise ValueError("No observed pregame DST history")
    h = h[h["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    h["defense"] = h["opponent_team"].map(canonical_team)
    h["offense"] = h["recent_team"].map(canonical_team)
    h = h[(h["defense"] != h["offense"]) & h["defense"].str.len().between(2, 3)]
    for stat in ("passing_interceptions", "passing_tds", "rushing_tds"):
        h[stat] = pd.to_numeric(h[stat], errors="coerce")
    # Missing stats are unknown, not zero. Group only complete observed rows.
    h = h.dropna(subset=["passing_interceptions", "passing_tds", "rushing_tds"])
    if h.empty:
        raise ValueError("No complete opponent offensive indicators")
    # Aggregate player rows to a unique defending-team/game/offense record.
    games = h.groupby(["season", "week", "defense", "offense"], as_index=False)[
        ["passing_interceptions", "passing_tds", "rushing_tds"]].sum()
    games = games.rename(columns={"passing_interceptions": "opponent_pass_interceptions",
                                   "passing_tds": "opponent_pass_tds",
                                   "rushing_tds": "opponent_rush_tds"})
    games["opponent_offensive_tds"] = games["opponent_pass_tds"] + games["opponent_rush_tds"]
    averages = games.groupby("defense").agg(
        games_observed=("week", "size"),
        opposing_interceptions_pg=("opponent_pass_interceptions", "mean"),
        opposing_offensive_tds_pg=("opponent_offensive_tds", "mean")).reset_index()
    for col in ("opposing_interceptions_pg", "opposing_offensive_tds_pg"):
        prior = float(games["opponent_pass_interceptions" if col.startswith("opposing_interceptions")
                            else "opponent_offensive_tds"].mean())
        averages[col] = ((averages["games_observed"] * averages[col] + shrink_games * prior)
                         / (averages["games_observed"] + shrink_games))
    averages["complete_dst_projection"] = False
    averages["missing_dst_inputs"] = "sacks, fumble recoveries, return TDs, official points allowed"
    averages["source"] = "nflverse observed opposing offensive player-week records"
    return averages

def attach_dst_indicators(baselines, indicators):
    out = baselines.copy()
    index = indicators.set_index("defense")
    for field in ("games_observed", "opposing_interceptions_pg", "opposing_offensive_tds_pg"):
        out["dst_" + field] = out["Team"].map(
            lambda t: index[field].get(canonical_team(t), np.nan))
        out.loc[out["Position"] != "DST", "dst_" + field] = np.nan
    return out
