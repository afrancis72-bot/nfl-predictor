"""Source-grounded NFL pregame market TOTALS for scenario testing only.

nflverse schedules include pre-kickoff closing total lines (historical)
and sometimes updated available totals for future games. Closing lines
are used only as pregame historical benchmarks; never use any result fields.
SPREAD NOT interpreted until its provider sign convention is confirmed.
"""
from __future__ import annotations
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from ai2_nfl_matchups import canonical_team

SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def fetch_schedule_lines(*, reader=None):
    reader=reader or pd.read_csv
    schedule=reader(SCHEDULE_URL)
    if not isinstance(schedule,pd.DataFrame) or schedule.empty:
        raise ValueError("No sourced nflverse schedule market data")
    return schedule


def market_totals_for_slate(schedule,season,week):
    required={"season","week","game_type","home_team","away_team","total_line"}
    if not isinstance(schedule,pd.DataFrame) or not required.issubset(schedule):
        raise ValueError("Schedule does not contain verified team, game, week and total_line fields")
    s=schedule.copy()
    s["season"]=pd.to_numeric(s["season"],errors="coerce")
    s["week"]=pd.to_numeric(s["week"],errors="coerce")
    s=s[(s["season"].eq(int(season)))&(s["week"].eq(int(week)))&
        (s["game_type"].astype(str).str.upper().eq("REG"))].copy()
    s["total_line"]=pd.to_numeric(s["total_line"],errors="coerce")
    s["home"]=s["home_team"].map(canonical_team)
    s["away"]=s["away_team"].map(canonical_team)
    s=s[s["total_line"].between(27.,68.) &
        (s["home"].ne(""))&(s["away"].ne(""))&(s["home"].ne(s["away"]))].copy()
    items={}
    for _,r in s.iterrows():
        key=tuple(sorted([str(r["home"]),str(r["away"])]))
        if key in items:
            # Ambiguous same-week game? Not scored, never guess.
            items[key]=None
            continue
        items[key]={
            "total_points":float(r["total_line"]),
            "teams":key,
            "source":"nflverse schedules sourced pregame total_line",
            "season":int(season),"week":int(week),
            "spread_used":False,
            "caution":"Closing total for completed games; may change before upcoming kickoff"
        }
    return {k:v for k,v in items.items() if v is not None}


def market_tds_prior(historical_expected_tds,estimated_field_goals,game_total,
                     *,weight=0.):
    """Symmetric implied team points -> team offensive TD adjustment.

    This intentionally ignores point spreads to avoid an undocumented sign
    convention. Total is shared by two teams, and we never use team results
    to form market expectation.
    """
    historical=float(historical_expected_tds)
    if game_total is None or weight==0.:
        return historical
    line=float(game_total)
    if not np.isfinite(line) or not 27.<=line<=68.:
        raise ValueError("Invalid pregame market total")
    if not np.isfinite(weight) or not 0<=weight<=1:
        raise ValueError("Invalid market mixture weight")
    implied_offense_tds=float(np.clip(
        (line/2.-3.*float(estimated_field_goals))/7.,.35,4.8))
    return float(np.clip((1.-weight)*historical+weight*implied_offense_tds,.35,4.8))
