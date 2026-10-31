"""Published nflverse weekly roster status for off-slate injured/reserve players.

Unlike game injury practice tables, this can include players on reserve/IR
who disappeared from DraftKings salary files. Only explicit SAME target-week
reserve status can mark a player unavailable. No guessed OUT from absence.
"""
from __future__ import annotations
import pandas as pd
from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team

ROSTER_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
              "weekly_rosters/roster_weekly_{season}.csv")
RESERVE_STATUSES = frozenset({"RES", "IR", "PUP", "NFI"})
ELIGIBLE_POS = frozenset({"QB", "RB", "WR", "TE"})


def fetch_weekly_roster(season, *, reader=None):
    reader = reader or pd.read_csv
    frame = reader(ROSTER_URL.format(season=int(season)))
    if not isinstance(frame,pd.DataFrame) or frame.empty:
        raise ValueError("No nflverse weekly roster records available")
    return frame


def reserve_players_missing_from_slate(roster, salary_rows, rates, season, week):
    required={"season","week","team","full_name","position","status"}
    if not isinstance(roster,pd.DataFrame) or not required.issubset(roster.columns):
        raise ValueError("Weekly roster source lacks explicit season/week/player/status fields")
    records=roster.copy()
    records["season"]=pd.to_numeric(records["season"],errors="coerce")
    records["week"]=pd.to_numeric(records["week"],errors="coerce")
    records=records[(records["season"].eq(int(season))) &
                    (records["week"].eq(int(week)))].copy()
    if "game_type" in records:
        records=records[records["game_type"].astype(str).str.upper().eq("REG")]
    records["status"]=records["status"].astype(str).str.strip().str.upper()
    records["name"]=records["full_name"].map(_norm_name)
    records["team_norm"]=records["team"].map(canonical_team)
    records["position"]=records["position"].astype(str).str.strip().str.upper()
    records=records[(records["status"].isin(RESERVE_STATUSES)) &
                    (records["position"].isin(ELIGIBLE_POS)) &
                    (records["name"].ne("")) & (records["team_norm"].ne(""))]
    slate_names={(_norm_name(s.get("Name","")),canonical_team(s.get("TeamAbbrev","")))
                 for s in salary_rows}
    slate_teams={canonical_team(s.get("TeamAbbrev","")) for s in salary_rows}
    rated={(_norm_name(r["player"]),canonical_team(r["team"]),
            str(r["position"]).upper())
           for r in rates.to_dict("records")}
    candidates=[]
    unique = records.groupby(["name","team_norm","position"],dropna=False)
    for (name,team,pos),g in unique:
        key=(name,team)
        if (len(g)!=1 or team not in slate_teams or key in slate_names
                or (name,team,pos) not in rated):
            continue
        row=g.iloc[0]
        candidates.append({"player":str(row["full_name"]),"team":team,
                           "position":pos,"roster_status":str(row["status"]),
                           "season":int(season),"week":int(week),
                           "source":"nflverse explicit same-week reserve status; missing DK salary"})
    return candidates
