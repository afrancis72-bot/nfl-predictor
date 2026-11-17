"""Cross-check an official NFL article capture against independently listed kickoff.

The source is nflverse's published schedules (games.csv). It supplies a
second check; it is not claimed to be the NFL's own official game-day feed.
The schedule gametime column is always in US Eastern time, including games
played in Europe. If the schedule is missing or appears stale, FAIL CLOSED.

Never silently allow a caller-provided future kickoff to masquerade as the
real kickoff for a completed game.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ai2_nfl_matchups import canonical_team
from ai2_nfl_pregame_availability import _utc

GAMES_SOURCE="https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def verified_match_kickoff(schedule,*,season,week,game):
    if not isinstance(schedule,pd.DataFrame):
        raise ValueError("Independently published NFL schedule is required")
    needed={"season","week","game_type","away_team","home_team","gameday","gametime"}
    if not needed.issubset(schedule.columns):
        raise ValueError("Independent schedule missing game/time fields")
    teams=str(game).split("@")
    if len(teams)!=2 or len(set(teams))!=2 or not all(teams):
        raise ValueError("Expected AWAY@HOME matchup identity")
    a,h=[canonical_team(t) for t in teams]
    subset=schedule[
        pd.to_numeric(schedule["season"],errors="coerce").eq(int(season)) &
        pd.to_numeric(schedule["week"],errors="coerce").eq(int(week)) &
        schedule["game_type"].astype(str).eq("REG") &
        schedule["away_team"].map(canonical_team).eq(a) &
        schedule["home_team"].map(canonical_team).eq(h)]
    if len(subset)!=1:
        raise ValueError(f"Expected exactly one published regular-season matchup {game}: {len(subset)}")
    row=subset.iloc[0]
    day=str(row["gameday"])
    clock=str(row["gametime"])
    try:
        time=datetime.fromisoformat(f"{day}T{clock}")
        if time.tzinfo is not None:
            raise ValueError("nflverse schedule gametime must be local Eastern, not timezone tagged")
        local=time.replace(tzinfo=ZoneInfo("America/New_York"))
    except (TypeError,ValueError) as e:
        raise ValueError("Invalid independent schedule gameday/ET kickoff") from e
    utc=local.astimezone(ZoneInfo("UTC"))
    return utc.isoformat()


def require_schedule_kickoff(schedule,*,season,week,game,kickoff_utc):
    expected=_utc(verified_match_kickoff(
        schedule,season=season,week=week,game=game),"verified scheduled kickoff")
    claimed=_utc(kickoff_utc,"kickoff_utc")
    if abs((expected-claimed).total_seconds())>60:
        raise ValueError(f"Claimed kickoff does not match independently published game schedule: {expected.isoformat()}")
    return expected.isoformat()


def fetch_and_verify_kickoff(*,season,week,game,kickoff_utc,reader=None):
    reader=reader or pd.read_csv
    frame=reader(GAMES_SOURCE)
    return require_schedule_kickoff(
        frame,season=season,week=week,game=game,kickoff_utc=kickoff_utc)
