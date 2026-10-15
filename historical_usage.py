"""Pregame nflverse usage ingestion for research simulator.

Source: https://github.com/nflverse/nflverse-data/releases/tag/stats_player
Fetch the current season snapshot explicitly; never silently update contest inputs.
"""
from __future__ import annotations
from pathlib import Path
import io
import urllib.request
import re
import unicodedata
import numpy as np
import pandas as pd

def download_weekly_stats(season: int, destination: str):
    """Download an auditable raw snapshot; network access is explicit."""
    if not 1999 <= int(season) <= 2100:
        raise ValueError("Invalid season")
    url=(f"https://github.com/nflverse/nflverse-data/releases/download/"
         f"stats_player/stats_player_week_{int(season)}.csv")
    req=urllib.request.Request(url,headers={"User-Agent":"NFL-research/1.0"})
    with urllib.request.urlopen(req,timeout=45) as response:
        raw=response.read(30_000_001)
    if len(raw)>30_000_000: raise ValueError("Stats file exceeds safety limit")
    frame=pd.read_csv(io.BytesIO(raw),low_memory=False)
    if frame.empty: raise ValueError("Downloaded empty weekly stats")
    target=Path(destination)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(raw)
    return frame

def build_usage_priors(weekly, as_of_week, season=2026, recent_games=5):
    """Never use a game from or after the target week; no look-ahead leakage."""
    d=weekly.copy()
    # The current stats_player release uses team; legacy files use recent_team.
    if "recent_team" not in d and "team" in d:
        d=d.rename(columns={"team":"recent_team"})
    if "player_display_name" in d:
        d["player_name"]=d["player_display_name"]
    required={"season","week","player_name","recent_team","position","carries","targets"}
    missing=required-set(d.columns)
    if missing: raise ValueError("Missing weekly fields: "+", ".join(sorted(missing)))
    d=d[(pd.to_numeric(d.season,errors="coerce")==int(season))&
        (pd.to_numeric(d.week,errors="coerce")<int(as_of_week))]
    if "season_type" in d: d=d[d.season_type.astype(str).str.upper()=="REG"]
    if d.empty: raise ValueError("No pregame historical data")
    # Prefer immutable player IDs where available to avoid name collisions.
    id_col="player_id" if "player_id" in d else "player_name"
    d=d.sort_values("week").groupby([id_col,"recent_team"],group_keys=False).tail(recent_games)
    for c in ("attempts","completions","passing_yards","carries","targets","rushing_yards","receiving_yards","receptions"):
        if c not in d and c in ("attempts","completions","passing_yards"):
            d[c]=0
        if c not in d: raise ValueError("Missing statistical column: "+c)
        d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0).clip(lower=0)
    grouping=[id_col,"recent_team","player_name","position"]
    out=d.groupby(grouping,as_index=False).agg(
        games=("week","nunique"),attempts=("attempts","sum"),completions=("completions","sum"),passing_yards=("passing_yards","sum"),carries=("carries","sum"),targets=("targets","sum"),
        rushing_yards=("rushing_yards","sum"),receiving_yards=("receiving_yards","sum"),
        receptions=("receptions","sum"))
    for c in ("attempts","completions","passing_yards","carries","targets","rushing_yards","receiving_yards","receptions"):
        out[c+"_per_game"]=out[c]/out.games.clip(lower=1)
    out["receiver_catch_rate"]=np.where(out["targets"]>0,out["receptions"]/out["targets"].clip(lower=1),0.)
    out["receiver_yards_per_catch"]=np.where(out["receptions"]>0,out["receiving_yards"]/out["receptions"].clip(lower=1),0.)
    out["rusher_yards_per_carry"]=np.where(out["carries"]>0,out["rushing_yards"]/out["carries"].clip(lower=1),0.)
    out["qb_yards_per_attempt"]=np.where(out["attempts"]>0,out["passing_yards"]/out["attempts"].clip(lower=1),0.)
    out["qb_completion_rate"]=np.where(out["attempts"]>0,out["completions"]/out["attempts"].clip(lower=1),0.)
    return out

def _canonical_team(team):
    value=str(team).upper().strip()
    return {"LA":"LAR","JAC":"JAX","WSH":"WAS"}.get(value,value)

def _canonical_name(name):
    value=unicodedata.normalize("NFKD",str(name)).encode("ascii","ignore").decode("ascii")
    value=re.sub(r"[^a-zA-Z0-9 ]"," ",value).lower()
    value=re.sub(r"\b(jr|sr|ii|iii|iv|v)\b","",value)
    return " ".join(value.split())

def apply_usage_priors(players, priors, min_games=2):
    """Strict identity and team matching; unknowns are NOT guessed."""
    p=players.copy()
    if not {"Name","TeamAbbrev"}.issubset(p):
        raise ValueError("Salary file needs Name and TeamAbbrev")
    left=p.copy()
    right=priors.copy()
    left["_team_key"]=left["TeamAbbrev"].map(_canonical_team)
    left["_name_key"]=left["Name"].map(_canonical_name)
    right["_team_key"]=right["recent_team"].map(_canonical_team)
    right["_name_key"]=right["player_name"].map(_canonical_name)
    if right.duplicated(["_name_key","_team_key"]).any():
        raise ValueError("Ambiguous normalized player-name/team matches; resolve identities")
    merged=left.merge(right,on=["_name_key","_team_key"],how="left",validate="many_to_one")
    merged=merged.drop(columns=["_name_key","_team_key"])
    merged["usage_verified"]=merged["games"].fillna(0)>=min_games
    merged["rush_weight"]=np.where(merged.usage_verified,merged["carries_per_game"].fillna(0),0.)
    merged["target_weight"]=np.where(merged.usage_verified,merged["targets_per_game"].fillna(0),0.)
    return merged

def usage_audit(merged):
    columns=["Name","TeamAbbrev","Position","games","usage_verified",
             "carries_per_game","targets_per_game","attempts_per_game","passing_yards_per_game","qb_yards_per_attempt","qb_completion_rate","receiver_catch_rate","receiver_yards_per_catch","rusher_yards_per_carry","rush_weight","target_weight"]
    return merged[[c for c in columns if c in merged]].copy()
