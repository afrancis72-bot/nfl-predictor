"""Research: construct a football-player universe independently from DK eligibility.

A salary/optimizer slate is NOT an active NFL roster. Start with that roster
and add historically used pregame teammates (RB/WR/TE) as football-only rows.
They are not DraftKings eligible, never exported to the optimizer, and never
asserted to be confirmed game-day ACTIVE without separate status evidence.

This is a strictly dated prior-week opportunity universe, NOT live roster
or injury verification. Out-player filtering is explicit, and incomplete,
ambiguous or future-dated data fails closed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team


def build_football_only_players(
    dk_pool, pregame_rates, *,
    season, week, min_games=2, max_weeks_since=3,
    min_touches_pg=0.5, known_out=None,
):
    """Return non-DK offensive players eligible for internal football simulation.

    'eligibility' here means sufficiently observed historical opportunity, NOT
    today's verified active status. The candidate deliberately retains a field
    explaining that uncertainty instead of treating missing DK data as inactive.
    """
    if not isinstance(dk_pool,pd.DataFrame) or not isinstance(pregame_rates,pd.DataFrame):
        raise ValueError("Historical rates and slate selection pool must be DataFrames")
    required_pool={"Name","TeamAbbrev","Position","game","optimizer_eligible"}
    required_rates={"player","team","position","games_played","targets_pg",
                    "carries_pg","latest_observed_season","latest_observed_week"}
    if not required_pool.issubset(dk_pool):
        raise ValueError("Incomplete DK selection pool for football universe")
    if not required_rates.issubset(pregame_rates):
        raise ValueError("Incomplete pregame rates for football universe")
    if not (2020<=int(season)<=2100 and 1<=int(week)<=22):
        raise ValueError("Expected target season/week required")
    if min_games<1 or max_weeks_since<1 or min_touches_pg<0:
        raise ValueError("Invalid historical player universe thresholds")
    excluded=set()
    for item in known_out or []:
        if not isinstance(item,dict) or "team" not in item or "player" not in item:
            raise ValueError("Known-OUT exclusions must provide team/player")
        excluded.add((canonical_team(item["team"]),_norm_name(item["player"])))
    games={}
    roster={}
    for row in dk_pool.to_dict("records"):
        team=canonical_team(row["TeamAbbrev"])
        name=_norm_name(row["Name"])
        game=str(row["game"])
        if team not in games:
            games[team]=game
        elif games[team]!=game:
            raise ValueError("Ambiguous game identity for modeled team")
        roster[(team,name)]=True
    for team,game in games.items():
        if "@" not in game:
            raise ValueError("Game key must identify both teams")
        opponents={canonical_team(side) for side in game.split("@")}
        if len(opponents)!=2 or team not in opponents:
            raise ValueError("Team not in game identity")
    seen=set()
    additions=[]
    for row in pregame_rates.to_dict("records"):
        team=canonical_team(row["team"])
        name=_norm_name(row["player"])
        pos=str(row["position"]).upper()
        if team not in games or pos not in ("RB","WR","TE") or not name:
            continue
        identity=(team,name)
        if identity in seen:
            raise ValueError("Duplicate same-team player history in input")
        seen.add(identity)
        if identity in roster or identity in excluded:
            continue
        last=(int(row.get("latest_observed_season") or 0),
              int(row.get("latest_observed_week") or 0))
        if last>=(int(season),int(week)):
            raise ValueError("Future/target-week player usage would leak into universe")
        if last[0]!=int(season) or not (0<int(week)-last[1]<=max_weeks_since):
            continue
        games_played=int(row.get("games_played") or 0)
        if games_played<min_games:
            continue
        try:
            targets=float(row["targets_pg"])
            carries=float(row["carries_pg"])
        except (ValueError,TypeError):
            continue
        if not np.isfinite([targets,carries]).all() or min(targets,carries)<0:
            raise ValueError("Negative/nonfinite observed teammate workload")
        if targets+carries<min_touches_pg:
            continue
        additions.append({
            "ID":f"FOOTBALL_ONLY|{team}|{pos}|{name}",
            "Name":str(row["player"]),
            "TeamAbbrev":team,
            "Position":pos,
            "game":games[team],
            "Salary":0,
            "proj":0.,
            "optimizer_eligible":False,
            "football_only":True,
            "role_evidence_status":"PREGAME_HISTORICAL_ROLE_GAME_DAY_STATUS_UNVERIFIED",
            "latest_observed_season":last[0],
            "latest_observed_week":last[1],
            "observed_games":games_played,
            "observed_targets_pg":targets,
            "observed_carries_pg":carries,
            "source":"nflverse pregame player weekly use, not DK eligibility or active list",
        })
    columns=["ID","Name","TeamAbbrev","Position","game","Salary","proj",
             "optimizer_eligible","football_only","role_evidence_status",
             "latest_observed_season","latest_observed_week",
             "observed_games","observed_targets_pg","observed_carries_pg",
             "source"]
    return pd.DataFrame(additions,columns=columns)


def validate_football_only_players(football_only, selected, rates):
    """Fail closed if a research-only unselected player is smuggled into DK pool.

    The synthetic football slate must not replace or duplicate a selected
    player's identity or invent a team outside the original slate.
    """
    if not isinstance(football_only,pd.DataFrame):
        raise ValueError("Football-only input must be a DataFrame")
    if football_only.empty:
        return football_only.copy()
    needed={"ID","Name","TeamAbbrev","Position","game","optimizer_eligible",
            "football_only","role_evidence_status","Salary","proj",
            "latest_observed_season","latest_observed_week"}
    if not needed.issubset(football_only):
        raise ValueError("Football-only players lack provenance fields")
    original={(canonical_team(r["TeamAbbrev"]),_norm_name(r["Name"]))
              for r in selected.to_dict("records")}
    slate={(canonical_team(r["TeamAbbrev"]),str(r["game"]))
           for r in selected.to_dict("records")}
    expected={(canonical_team(r["team"]),_norm_name(r["player"]),str(r["position"]).upper())
              for r in rates.to_dict("records")}
    found=set()
    for row in football_only.to_dict("records"):
        team=canonical_team(row["TeamAbbrev"])
        name=_norm_name(row["Name"])
        position=str(row["Position"]).upper()
        key=(team,name,position)
        if (not isinstance(row["football_only"],(bool,np.bool_))
            or not row["football_only"]
            or not isinstance(row["optimizer_eligible"],(bool,np.bool_))
            or row["optimizer_eligible"]
            or str(row["role_evidence_status"])!=
                "PREGAME_HISTORICAL_ROLE_GAME_DAY_STATUS_UNVERIFIED"
            or float(row["Salary"])!=0 or
                (team,name) in original or key in found or key not in expected
            or (team,str(row["game"])) not in slate
            or position not in {"WR","RB","TE"}):
            raise ValueError("Invalid/duplicate/nonhistorical football-only candidate")
        found.add(key)
    return football_only.copy().reset_index(drop=True)
