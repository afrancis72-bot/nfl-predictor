"""Leakage-free, role-specific defensive STAT-channel evidence for NFL Classic.

This deliberately does NOT add fantasy points to a player. Opposing defenders
are evaluated against previous WR1/WR2/WR3, RB1/RB2, TE1/TE2 and QB1 roles
based on those opponents' *pregame* target/carry/attempt rankings. For each
historical game, the player's opportunity/efficiency is compared with their
OWN prior eight-game baseline, NOT the league-average player at the position.

The output is a conservative, sourced, dated set of multipliers to input
player stat-rate fields. This is experimental football intelligence; the
separate Bentley release gate blocks consumer lineup builds until blind
out-of-sample validation establishes forecasting value.
"""
from __future__ import annotations

from collections import defaultdict, deque
import numpy as np
import pandas as pd

from props_data import clean_weekly
from ai2_nfl_role_matchups import (
    _name, _usage, role_observations, ROLE_LIMIT, INVALID_STATUS,
)
from ai2_nfl_matchups import canonical_team
from ai2_nfl_opponent_relative import _identity

SOURCE = "nflverse own-player pregame stat channels by pregame role versus opponent"
# (numerator, denominator, minimum own pregame denominator,
#  minimum result-game denominator). Missing denominators represent volume / game.
CHANNELS = {
    "QB": {"pass_attempts": ("attempts", None, 0., 0.),
           "pass_ypa": ("passing_yards", "attempts", 60., 12.)},
    "RB": {"carries": ("carries", None, 0., 0.),
           "targets": ("targets", None, 0., 0.),
           "rush_ypc": ("rushing_yards", "carries", 22., 5.),
           "receive_ypt": ("receiving_yards", "targets", 12., 2.)},
    "WR": {"targets": ("targets", None, 0., 0.),
           "catch_rate": ("receptions", "targets", 18., 3.),
           "receive_ypt": ("receiving_yards", "targets", 18., 3.)},
    "TE": {"targets": ("targets", None, 0., 0.),
           "catch_rate": ("receptions", "targets", 12., 2.),
           "receive_ypt": ("receiving_yards", "targets", 12., 2.)},
}
COLUMNS = ("defense","role","position","channel","observations",
           "defense_games","opportunities","raw_relative_pct",
           "shrunk_relative_pct","source","target_season","target_week")
MAX_ABS_RESIDUAL = .75
MAX_ABS_EFFECT = .18


def _numeric(row, key):
    try:
        x=float(row.get(key,float("nan")))
        return x if np.isfinite(x) and x>=0 else float("nan")
    except (TypeError,ValueError):
        return float("nan")


def role_channel_observations(history, season, week):
    """Strictly prior-week opponent-game rows with per-player frozen baselines."""
    yr,wk=int(season),int(week)
    if not 1<=wk<=23:
        raise ValueError("Invalid target week")
    if "opponent_team" not in history.columns:
        raise ValueError("Missing source opponent_team, cannot infer defensive matchups")
    roles=role_observations(history,yr,wk)
    if roles.empty:
        return pd.DataFrame()
    role_lookup={}
    for x in roles.to_dict("records"):
        k=(int(x["season"]),int(x["week"]),x["offense"],
           x["position"],_name(x["player"]))
        if k in role_lookup:
            raise ValueError("Duplicate role-labeled historical opponent game")
        role_lookup[k]=x

    h=clean_weekly(history)
    h=h[(h["season"].ge(yr-1))&
        ((h["season"].lt(yr))|
         (h["season"].eq(yr)&h["week"].lt(wk)))].copy()
    h=h.sort_values(["season","week","recent_team","player_name"],kind="stable")
    previous=defaultdict(lambda: deque(maxlen=8))
    output=[]
    seen=set()
    for (sy,sw),frame in h.groupby(["season","week"],sort=True):
        pending=[]
        for row in frame.to_dict("records"):
            ident=_identity(row)
            pos=ident[2]
            if pos not in CHANNELS:
                continue
            stamp=(int(sy),int(sw),ident)
            if stamp in seen:
                raise ValueError("Duplicate player-week in defensive stat history")
            seen.add(stamp)
            role=role_lookup.get((int(sy),int(sw),ident[1],pos,
                                  _name(row["player_name"])))
            games=previous[ident]
            if role and len(games)>=3:
                for channel,(num,den,min_prior,min_game) in CHANNELS[pos].items():
                    old=[(_numeric(q,num),_numeric(q,den) if den else 1.)
                         for q in games]
                    old=[(a,b) for a,b in old if np.isfinite(a) and
                         np.isfinite(b) and b>=0]
                    if len(old)<3:
                        continue
                    base_num=sum(a for a,b in old)
                    base_den=sum(b for a,b in old)
                    if base_den<=0 or base_den<min_prior:
                        continue
                    # Per-game opportunity baseline; per-opportunity conversion
                    # baseline. The TARGET game cannot affect either.
                    baseline=(base_num/len(old) if den is None
                              else base_num/base_den)
                    actual_num=_numeric(row,num)
                    actual_den=_numeric(row,den) if den else 1.
                    if (not np.isfinite(actual_num) or
                        not np.isfinite(actual_den) or
                        actual_den<min_game or baseline<=0):
                        continue
                    current=(actual_num if den is None
                             else actual_num/max(actual_den,1.))
                    relative=(current/baseline)-1.
                    output.append({
                        "season":int(sy),"week":int(sw),
                        "defense":role["defense"],"offense":ident[1],
                        "player":str(row["player_name"]),
                        "position":pos,"role":role["role"],"channel":channel,
                        "baseline":baseline,"actual":current,
                        "result_opportunities":actual_den if den else max(1.,baseline),
                        "relative_pct":float(np.clip(
                            relative,-MAX_ABS_RESIDUAL,MAX_ABS_RESIDUAL)),
                        "prior_games":len(old),
                        "source":SOURCE,
                    })
            pending.append((ident,row))
        # Entire gameweek is written only AFTER observations have been scored.
        for ident,row in pending:
            previous[ident].append(row)
    return pd.DataFrame(output)


def role_channel_profiles(history, season, week, *, defense_window=10,
                          min_observations=5, min_defense_games=3,
                          min_opportunities=18., shrink_games=10.,
                          shrink_opportunities=90.):
    """Defense x role x STAT channel, bounded and shrunk to no effect."""
    yr,wk=int(season),int(week)
    if (defense_window<1 or min_observations<2 or min_defense_games<2
        or min_opportunities<=0 or shrink_games<=0 or
        shrink_opportunities<=0):
        raise ValueError("Invalid conservative defensive role evidence rules")
    observations=role_channel_observations(history,yr,wk)
    if observations.empty:
        return pd.DataFrame(columns=COLUMNS),observations
    profiles=[]
    for defense,team in observations.groupby("defense"):
        periods=sorted(set(zip(team["season"].astype(int),
                               team["week"].astype(int))))
        recent=set(periods[-int(defense_window):])
        active=team[[ (int(z["season"]),int(z["week"])) in recent
                      for z in team.to_dict("records")]]
        for (role,channel),group in active.groupby(["role","channel"]):
            n=int(len(group))
            g=int(group[["season","week"]].drop_duplicates().shape[0])
            opps=float(group["result_opportunities"].sum())
            if n<min_observations or g<min_defense_games or opps<min_opportunities:
                continue
            # One player-game is never an automatic 30-point WR3 projection.
            # Clip each excess, shrink by independent games AND opportunity
            # volume, then cap the resulting rate change conservatively.
            raw=float(group["relative_pct"].mean())
            reliability=(n/(n+shrink_games))*(opps/(opps+shrink_opportunities))
            estimate=float(np.clip(raw*reliability,
                                   -MAX_ABS_EFFECT,MAX_ABS_EFFECT))
            profiles.append({
                "defense":defense,"role":role,
                "position":role[:2],"channel":channel,
                "observations":n,"defense_games":g,
                "opportunities":round(opps,3),
                "raw_relative_pct":round(raw,6),
                "shrunk_relative_pct":round(estimate,6),
                "source":SOURCE,"target_season":yr,"target_week":wk,
            })
    return pd.DataFrame(profiles,columns=COLUMNS),observations


def validated_role_lookup(profiles, *, season, week, strength):
    """Fail closed on unknown or future/undated role matchup inputs."""
    if isinstance(strength,bool) or not isinstance(strength,(float,int,np.floating)):
        raise ValueError("Role defensive strength must be numeric")
    if not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError("Role defensive strength must be in [0,1]")
    if strength==0.:
        return {}
    if not isinstance(profiles,pd.DataFrame):
        raise ValueError("Active role defense requires source-backed profiles")
    if profiles.empty:
        return {}
    missing=set(COLUMNS)-set(profiles.columns)
    if missing:
        raise ValueError("Incomplete role defensive stat profiles: "+str(sorted(missing)))
    result={}
    for row in profiles.to_dict("records"):
        if (row["source"]!=SOURCE or
            (int(row["target_season"]),int(row["target_week"])) !=
            (int(season),int(week))):
            raise ValueError("Unverified or wrong-week role defensive stat profile")
        role=str(row["role"]); channel=str(row["channel"])
        if role not in {f"{pos}{i}" for pos,n in ROLE_LIMIT.items()
                        for i in range(1,n+1)}:
            raise ValueError("Unknown defensive role")
        if channel not in CHANNELS[role[:2]]:
            raise ValueError("Unknown role stat channel")
        if (int(row["observations"])<5 or
            int(row["defense_games"])<3 or
            float(row["opportunities"])<18):
            raise ValueError("Insufficient defensive role sample")
        estimate=float(row["shrunk_relative_pct"])
        if not np.isfinite(estimate) or abs(estimate)>MAX_ABS_EFFECT:
            raise ValueError("Unbounded role defensive adjustment")
        key=(canonical_team(row["defense"]),role,channel)
        if key in result:
            raise ValueError("Duplicate defensive role channel")
        result[key]=1.+strength*estimate
    return result


def apply_role_defense_to_rates(rates,slate_roles,profiles,*,season,week,
                                strength):
    """Apply validated PREgame role/opponent effects to real player stat rates.

    Relative target/carry changes affect both opportunities and aggregate
    outcome baselines, preserving historical per-target/carry conversion.
    Catch-rate and per-target efficiency changes adjust their respective
    per-opportunity fields; no direct fantasy-point awards are made.

    Returns a fresh DataFrame and an inspectable individual-player ledger.
    """
    factors=validated_role_lookup(profiles,season=season,week=week,
                                   strength=strength)
    if not isinstance(rates,pd.DataFrame):
        raise ValueError("Missing pregame stat-rate inputs")
    revised=rates.copy(deep=True)
    ledger=[]
    if not factors:
        return revised,ledger
    # pandas 2.x+ enforces int64 assignment types: source rates may contain
    # integer carries/targets but a role adjustment must be fractional.
    stat_fields={field for channels in CHANNELS.values()
                 for channel in channels for field in {
                    "targets":("targets_pg","receptions_pg","receiving_yards_pg","receiving_tds_pg"),
                    "carries":("carries_pg","rushing_yards_pg","rushing_tds_pg"),
                    "pass_attempts":("pass_attempts_pg","completions_pg","pass_yards_pg",
                                     "passing_tds_pg","interceptions_pg"),
                    "catch_rate":("receptions_pg",),
                    "receive_ypt":("receiving_yards_pg",),
                    "rush_ypc":("rushing_yards_pg",),
                    "pass_ypa":("pass_yards_pg",),
                 }[channel]}
    for field in stat_fields.intersection(revised.columns):
        revised[field]=pd.to_numeric(revised[field],errors="coerce").astype(float)
    if not isinstance(slate_roles,list):
        raise ValueError("Source-backed slate player role report is required")
    indexed={}
    for k,row in revised.iterrows():
        key=(_name(row.get("player","")),canonical_team(row.get("team","")),
             str(row.get("position","")).upper())
        if key in indexed:
            raise ValueError("Duplicate player/team position in stat rates")
        indexed[key]=k
    seen=set()
    for row in slate_roles:
        role=str(row.get("Role",""))
        if not role or role not in {f"{p}{i}" for p,n in ROLE_LIMIT.items()
                                 for i in range(1,n+1)}:
            continue
        status=str(row.get("Slate status","")).upper().strip()
        if status in INVALID_STATUS or status in {"Q","QUESTIONABLE","GTD","UNKNOWN"}:
            # Unresolved availability may change the current target hierarchy.
            # Do not extrapolate a historic WR3 into a verified current role.
            continue
        if int(row.get("Player history games",0) or 0)<3:
            continue
        if "Pregame within-team usage rank" not in str(
                row.get("Role evidence","")):
            continue
        name=str(row.get("Player",""))
        team=canonical_team(row.get("Team",""))
        opponent=canonical_team(row.get("Opponent",""))
        position=str(row.get("Position","")).upper()
        key=(_name(name),team,position)
        if key in seen:
            raise ValueError("Duplicate current-week player role")
        seen.add(key)
        if key not in indexed or not opponent or position!=role[:2]:
            continue
        i=indexed[key]
        applied={}
        for channel in CHANNELS[position]:
            factor=factors.get((opponent,role,channel),1.)
            if abs(factor-1.)<1e-12:
                continue
            applied[channel]=round(float(factor),6)
            fields={
                "targets":("targets_pg","receptions_pg","receiving_yards_pg",
                           "receiving_tds_pg"),
                "carries":("carries_pg","rushing_yards_pg","rushing_tds_pg"),
                "pass_attempts":("pass_attempts_pg","completions_pg",
                                 "pass_yards_pg","passing_tds_pg",
                                 "interceptions_pg"),
                "catch_rate":("receptions_pg",),
                "receive_ypt":("receiving_yards_pg",),
                "rush_ypc":("rushing_yards_pg",),
                "pass_ypa":("pass_yards_pg",),
            }[channel]
            for col in fields:
                if col not in revised.columns:
                    continue
                old=pd.to_numeric(revised.at[i,col],errors="coerce")
                if pd.notna(old) and np.isfinite(old) and old>=0:
                    revised.at[i,col]=float(old)*factor
        if applied:
            ledger.append({
                "player":name,"team":team,"position":position,
                "opponent":opponent,"pregame_role":role,
                "factor_by_stat_channel":applied,
                "source":SOURCE,
                "target_season":int(season),"target_week":int(week),
                "strength":float(strength),
                "opportunities_not_automatically_invented":True,
            })
    return revised,ledger
