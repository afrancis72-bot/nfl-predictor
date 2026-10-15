"""Experimental NFL classic slate input preparation. No production changes.

This prepares DK classic pools for matchup-level research simulations.
It does NOT claim to provide a calibrated classic lineup optimizer.
"""
from __future__ import annotations
import re
import pandas as pd

REQUIRED = {"Name", "ID", "Position", "Roster Position", "Salary", "Game Info", "TeamAbbrev"}
MATCHUP = re.compile(r"^([A-Z]{2,3})@([A-Z]{2,3})\\s")

def prepare_classic_slate(salaries: pd.DataFrame, excluded_names=()):
    """Validate DraftKings classic export and return eligible pool plus matchup audit.

    Preserve K/DST for a future separate model; snap simulation uses offensive
    players only. OUT overrides are explicit and logged, never inferred from DK.
    """
    missing = REQUIRED - set(salaries.columns)
    if missing:
        raise ValueError("Missing DK columns: " + ", ".join(sorted(missing)))
    p = salaries.copy()
    p["Name"] = p["Name"].astype(str).str.strip()
    p["TeamAbbrev"] = p["TeamAbbrev"].astype(str).str.upper().str.strip()
    p["Position"] = p["Position"].astype(str).str.upper().str.strip()
    p["Salary"] = pd.to_numeric(p["Salary"], errors="raise")
    if (p["Salary"] < 0).any() or p["ID"].isna().any():
        raise ValueError("Invalid salary or missing DraftKings ID")
    if p["ID"].duplicated().any():
        raise ValueError("Duplicate DK player IDs")
    p["matchup"] = p["Game Info"].astype(str).str.extract(MATCHUP.pattern, expand=True).apply(
        lambda r: "@".join(r.dropna().astype(str)), axis=1)
    if (p["matchup"] == "").any():
        raise ValueError("Unrecognized Game Info matchup")
    for game, group in p.groupby("matchup"):
        if not set(group["TeamAbbrev"]).issubset(set(game.split("@"))):
            raise ValueError(f"Team mismatch for {game}")
    excluded = set(excluded_names)
    p["manual_exclusion"] = p["Name"].isin(excluded)
    eligible = p.loc[~p["manual_exclusion"]].copy()
    offensive = eligible[eligible["Position"].isin(["QB", "RB", "WR", "TE"])].copy()
    offensive["FLEX Salary"] = offensive["Salary"]
    audits = []
    for game, group in eligible.groupby("matchup", sort=True):
        teams = game.split("@")
        audits.append({"matchup": game, "teams": ", ".join(teams),
                       "players": len(group),
                       "offensive_players": int(group["Position"].isin(["QB","RB","WR","TE"]).sum()),
                       "qbs": int((group["Position"]=="QB").sum()),
                       "k_dst_players": int(group["Position"].isin(["K","DST","D"]).sum())})
    return eligible, offensive, pd.DataFrame(audits), p.loc[p["manual_exclusion"]].copy()

def matchup_pool(offensive: pd.DataFrame, matchup: str) -> pd.DataFrame:
    """Produce the exact four required columns for one-game snap lab upload."""
    g = offensive.loc[offensive["matchup"] == matchup]
    if g.empty:
        raise ValueError("No eligible offensive players for matchup")
    return g[["Name", "TeamAbbrev", "Position", "FLEX Salary"]].copy()
