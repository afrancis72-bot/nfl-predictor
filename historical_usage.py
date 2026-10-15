"""Historical player usage priors from nflverse weekly player statistics.

Load a local, explicitly dated CSV/parquet snapshot. Never download implicitly
during a contest run. Only games before the target matchup are eligible.
"""
import numpy as np
import pandas as pd

def build_usage_priors(weekly, as_of_week, season=2026, recent_games=5):
    required={"season","week","player_name","recent_team","position","carries","targets"}
    missing=required-set(weekly.columns)
    if missing: raise ValueError("Missing weekly fields: "+", ".join(sorted(missing)))
    d=weekly.copy()
    d=d[(pd.to_numeric(d.season,errors="coerce")==season)&
        (pd.to_numeric(d.week,errors="coerce")<as_of_week)]
    d=d.sort_values("week").groupby(["player_name","recent_team"],group_keys=False).tail(recent_games)
    if d.empty: raise ValueError("No pregame historical data")
    for c in ["carries","targets","rushing_yards","receiving_yards","receptions"]:
        if c not in d: d[c]=0
        d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0).clip(lower=0)
    out=d.groupby(["player_name","recent_team"],as_index=False).agg(
        games=("week","nunique"), carries=("carries","sum"),
        targets=("targets","sum"), rushing_yards=("rushing_yards","sum"),
        receiving_yards=("receiving_yards","sum"),receptions=("receptions","sum"))
    for c in ["carries","targets","rushing_yards","receiving_yards","receptions"]:
        out[c+"_per_game"]=out[c]/out.games.clip(lower=1)
    return out

def apply_usage_priors(players, priors, min_games=2):
    """Strict name/team join; fail closed if a player lacks historical coverage."""
    p=players.copy()
    needed={"Name","TeamAbbrev"}
    if not needed.issubset(p): raise ValueError("Missing salary-file identifiers")
    merged=p.merge(priors,left_on=["Name","TeamAbbrev"],
                   right_on=["player_name","recent_team"],how="left",validate="many_to_one")
    merged["usage_verified"]=merged["games"].fillna(0)>=min_games
    merged["rush_weight"]=np.where(merged.usage_verified,merged["carries_per_game"].fillna(0),0.)
    merged["target_weight"]=np.where(merged.usage_verified,merged["targets_per_game"].fillna(0),0.)
    # No salary-based fallback: missing history requires explicit manual review.
    return merged
