"""Pregame NBA rotation candidates from each team's recent *actual game*.

An NBA game has 240 regulation player-minutes. Season historical eligibility
alone can include 16-18 players who are not simultaneously in the rotation.
Use observed game participation to avoid combining incompatible rotations.

Questionable/Probable players can rejoin the pool conditionally even after
missing recent games; this does NOT claim they will be active at tip-off.
"""
from __future__ import annotations
from datetime import date
import pandas as pd

from nba_engine.game_conservation import observed_complete_team_games
from nba_engine.injury_report import normalize_person


def select_plausible_rotation(
        projections:pd.DataFrame, logs:pd.DataFrame,when:date,
        *,uncertain_status=None,latest_games=2):
    if projections.empty:
        raise ValueError("Cannot establish NBA rotation with no players")
    if latest_games<1:
        raise ValueError("NBA rotation requires recent completed team games")
    games=observed_complete_team_games(logs,when)
    chosen=[]
    audits=[]
    uncertain_status=uncertain_status or {}
    for team,rows in projections.groupby("team",sort=True):
        recent=games.loc[games["team"].eq(team)].tail(max(1,latest_games))
        if len(recent)<1:
            raise ValueError(f"{team}: missing verified recent NBA rotation games")
        dates=set(recent["date"])
        played=logs.loc[logs["team"].eq(team)&
                        logs["date"].isin(dates),"player_id"].astype(str)
        observed=set(played)
        ids=rows["player_id"].astype(str)
        medical=rows.apply(
            lambda row:(str(row["team"]),normalize_person(row["player"])) in
            uncertain_status,axis=1)
        is_recent=ids.isin(observed)
        included=is_recent|medical
        selected=rows.loc[included].copy()
        rejected=rows.loc[~included].copy()
        if len(selected)<7 or float(selected["minutes"].sum())<195:
            raise ValueError(
                f"{team}: recent official player-game history does not "
                "verify seven players and 195 regulation minutes. "
                "Cannot fill the rotation with older season averages.")
        chosen.append(selected)
        for row in rejected.itertuples(index=False):
            audits.append({
                "team":team,"player":row.player,
                "player_id":str(row.player_id),
                "status":"excluded from projected rotation",
                "reason":f"no appearance in last {len(recent)} observed team games "
                         "and no current Questionable/Probable official report",
            })
        for row in selected.itertuples(index=False):
            key=(team,normalize_person(row.player))
            if key in uncertain_status and str(row.player_id) not in observed:
                audits.append({
                    "team":team,"player":row.player,
                    "player_id":str(row.player_id),
                    "status":"returning player modeled conditionally",
                    "reason":"official "+uncertain_status[key]+
                             "; no recent game appearance; active and minutes unverified",
                })
    p=pd.concat(chosen,ignore_index=True)
    return p,pd.DataFrame(
        audits,columns=["team","player","player_id","status","reason"])
