"""Pregame-only opponent defensive priors from nflverse play-by-play.

The input must contain play-level records and game identifiers. No betting markets,
fantasy prices, or data from the target week may influence these estimates.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED={"season","week","defteam","posteam","play_type","yards_gained"}

def build_defense_profiles(pbp: pd.DataFrame, season: int, as_of_week: int,
                           min_plays: int=80, shrinkage_plays: int=150):
    """Return team profiles using exclusively regular-season plays before kickoff.

    Profiles describe opponent-relative passing/rushing yardage and completion
    tendencies. They are descriptive priors, NOT opponent-adjusted power ratings.
    """
    missing=REQUIRED-set(pbp.columns)
    if missing: raise ValueError("Missing play-by-play columns: "+", ".join(sorted(missing)))
    d=pbp.copy()
    d=d[(pd.to_numeric(d["season"],errors="coerce")==int(season)) &
        (pd.to_numeric(d["week"],errors="coerce")<int(as_of_week))]
    if "season_type" in d.columns:
        d=d[d["season_type"].astype(str).str.upper()=="REG"]
    d=d[d["play_type"].isin(["pass","run"]) & d["defteam"].notna() & d["posteam"].notna()]
    d=d[d["defteam"]!=d["posteam"]].copy()
    if d.empty: raise ValueError("No pregame defensive play data")
    d["yards_gained"]=pd.to_numeric(d["yards_gained"],errors="coerce")
    d=d[d["yards_gained"].notna()]
    if d.empty: raise ValueError("No valid yardage observations")
    # Scrambles and sacks are not treated as completed passes.
    if "complete_pass" in d.columns:
        d["completion"]=pd.to_numeric(d["complete_pass"],errors="coerce").fillna(0).clip(0,1)
    else:
        d["completion"]=np.nan
    league_pass=d[d.play_type=="pass"]
    league_run=d[d.play_type=="run"]
    if league_pass.empty or league_run.empty:
        raise ValueError("Need both passing and rushing plays")
    lp=float(league_pass.yards_gained.mean())
    lr=float(league_run.yards_gained.mean())
    lc=float(league_pass.completion.mean()) if league_pass.completion.notna().any() else np.nan
    if lp<=0 or lr<=0: raise ValueError("Invalid league yardage baseline")
    profiles={}
    audit=[]
    for team,plays in d.groupby("defteam"):
        passes=plays[plays.play_type=="pass"]
        runs=plays[plays.play_type=="run"]
        pn,rn=len(passes),len(runs)
        pass_factor=(float(passes.yards_gained.mean())/lp) if pn else 1.
        run_factor=(float(runs.yards_gained.mean())/lr) if rn else 1.
        pw=pn/(pn+shrinkage_plays) if pn>=min_plays else 0.
        rw=rn/(rn+shrinkage_plays) if rn>=min_plays else 0.
        pass_factor=1+pw*(pass_factor-1)
        run_factor=1+rw*(run_factor-1)
        comp_delta=0.
        if pn>=min_plays and np.isfinite(lc) and passes.completion.notna().any():
            comp_delta=pw*(float(passes.completion.mean())-lc)
        profiles[str(team)]={
            "pass_yards_factor":float(np.clip(pass_factor,.65,1.35)),
            "run_yards_factor":float(np.clip(run_factor,.65,1.35)),
            "completion_delta":float(np.clip(comp_delta,-.15,.15))}
        audit.append({"Team":team,"Pass plays faced":pn,"Rush plays faced":rn,
                      "Pass yards/play allowed":round(float(passes.yards_gained.mean()),2) if pn else np.nan,
                      "Run yards/play allowed":round(float(runs.yards_gained.mean()),2) if rn else np.nan,
                      "Pass factor":round(profiles[str(team)]["pass_yards_factor"],3),
                      "Run factor":round(profiles[str(team)]["run_yards_factor"],3),
                      "Completion delta":round(profiles[str(team)]["completion_delta"],3),
                      "Pass prior sufficient":pn>=min_plays,"Run prior sufficient":rn>=min_plays})
    return profiles,pd.DataFrame(audit)
