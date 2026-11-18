"""Straightforward *observed* NFL defense vs pregame positional-role YARDS.

This is a descriptive research report, not a scoring multiplier or a claim
that defenses cause all observed differences. Each opponent's WR1/2/3,
RB1/2, TE1/2 and QB1 role is assigned BEFORE that matchup using only
historical teammate usage. Compare own pregame trailing 8-game yard mean
with the player's actual same-game yardage; report per-defense per-role
count, above-baseline rate, signed and absolute yard difference and
game-level details. No spread/odds, optimizer, simulation or live status
is required. No target-game future leakage into baseline or role labels.
"""
from __future__ import annotations

from collections import defaultdict,deque
import numpy as np
import pandas as pd

from props_data import clean_weekly
from ai2_nfl_matchups import canonical_team
from ai2_nfl_role_matchups import _name,role_observations
from ai2_nfl_opponent_relative import _identity

ROLE_CHANNELS={
    "QB1":("pass_yards","passing_yards"),
    "RB1":("rush_yards","rushing_yards"),
    "RB2":("rush_yards","rushing_yards"),
    "RB1_RECEIVING":("receiving_yards","receiving_yards"),
    "RB2_RECEIVING":("receiving_yards","receiving_yards"),
    "WR1":("receiving_yards","receiving_yards"),
    "WR2":("receiving_yards","receiving_yards"),
    "WR3":("receiving_yards","receiving_yards"),
    "TE1":("receiving_yards","receiving_yards"),
    "TE2":("receiving_yards","receiving_yards"),
}
MIN_BASELINES={"pass_yards":100.,"rush_yards":12.,"receiving_yards":10.}
MIN_PREGAME_GAMES=3
REQUIRED_ROLES={"QB1","RB1","RB2","WR1","WR2","WR3","TE1","TE2"}


def observed_defense_yards(history,season,target_week):
    """One row for each valid actual player-game role/yardage channel."""
    season,week=int(season),int(target_week)
    if season<2020 or not 1<=week<=23:
        raise ValueError("Invalid target season and week")
    roles=role_observations(history,season,week)
    if roles.empty:
        raise ValueError("No source-backed pregame ranked opponent roles")
    role_lookup={}
    for row in roles.to_dict("records"):
        k=(int(row["season"]),int(row["week"]),canonical_team(row["offense"]),
           row["position"],_name(row["player"]))
        if k in role_lookup:
            raise ValueError("Ambiguous labeled role/player/game")
        role_lookup[k]=row
    h=clean_weekly(history)
    h=h[(h["season"].ge(season-1))&
        ((h["season"].lt(season))|
         ((h["season"].eq(season))&(h["week"].lt(week))))].copy()
    if "opponent_team" not in h:
        raise ValueError("Historical defense report requires opponent_team")
    h=h.sort_values(["season","week","recent_team","player_name"],kind="stable")
    past=defaultdict(lambda:deque(maxlen=8))
    rows=[];seen=set()
    for (yr,wk),g in h.groupby(["season","week"],sort=True):
        updates=[]
        for player in g.to_dict("records"):
            ident=_identity(player)
            key=(int(yr),int(wk),ident)
            if key in seen:
                raise ValueError("Duplicate player in week")
            seen.add(key)
            prior=past[ident]
            role=role_lookup.get((int(yr),int(wk),ident[1],ident[2],
                                  _name(player["player_name"])))
            if yr==season and role is not None and len(prior)>=MIN_PREGAME_GAMES:
                role_name=str(role["role"])
                if role_name not in REQUIRED_ROLES:
                    raise ValueError("Unsupported role label")
                channels=[(role_name,*ROLE_CHANNELS[role_name])]
                if role_name in ("RB1","RB2"):
                    ch=role_name+"_RECEIVING"
                    channels.append((ch,*ROLE_CHANNELS[ch]))
                for displayed_role,metric,field in channels:
                    previous=pd.to_numeric(
                        pd.Series([x.get(field,np.nan) for x in prior]),
                        errors="coerce").to_numpy(dtype=float)
                    current=pd.to_numeric(pd.Series([player.get(field,np.nan)]),
                                           errors="coerce").iloc[0]
                    if (np.isfinite(previous).sum()<MIN_PREGAME_GAMES or
                        not np.isfinite(current)):
                        continue
                    baseline=float(np.mean(previous[np.isfinite(previous)]))
                    if baseline<MIN_BASELINES[metric]:
                        # Player with 0-10 previous yards is a volatile sample,
                        # not evidence of a historically weak defense.
                        continue
                    opponent=canonical_team(player.get("opponent_team",""))
                    if not opponent or opponent!=canonical_team(role["defense"]):
                        raise ValueError("Defense opponent identity disagreement")
                    actual=float(current)
                    rows.append({
                        "season":int(yr),"week":int(wk),
                        "defense":opponent,"offense":ident[1],
                        "player":str(player["player_name"]),
                        "role":displayed_role,"yardage_type":metric,
                        "own_pregame_mean_yards":round(baseline,4),
                        "actual_yards":round(actual,4),
                        "difference_yards":round(actual-baseline,4),
                        "ratio_to_own_baseline":round(actual/baseline,5),
                        "above_own_baseline":bool(actual>baseline),
                        "prior_player_games":int(np.isfinite(previous).sum()),
                        "role_is_pregame_usage_proxy":True,
                        "verified_depth_chart_or_coverage":False,
                    })
            updates.append((ident,player))
        # Finish the ENTIRE week before adding it to historical baselines.
        for ident,player in updates:
            past[ident].append(player)
    if not rows:
        raise ValueError("No valid historical same-player defensive yard observations")
    return pd.DataFrame(rows)


def summarize_defense_yards(observations,*,min_sample=4,
                            min_consistency_rate=.67,
                            min_excess_pct=.10):
    """Descriptive consistency flag, not a statistical-significance claim."""
    needed={"season","week","defense","role","yardage_type",
            "own_pregame_mean_yards","actual_yards","difference_yards",
            "above_own_baseline","player"}
    if not isinstance(observations,pd.DataFrame) or observations.empty or (
        needed-set(observations.columns)):
        raise ValueError("Missing game-by-game defense vs role yard comparisons")
    if not 2<=min_sample<=20 or not .5<=min_consistency_rate<=1 or min_excess_pct<0:
        raise ValueError("Invalid consistency parameters")
    keys=["season","week","defense","role","yardage_type","player"]
    if observations.duplicated(keys).any():
        raise ValueError("Duplicate defensive role player result")
    cols=["own_pregame_mean_yards","actual_yards","difference_yards"]
    if not np.isfinite(observations[cols].to_numpy(float)).all():
        raise ValueError("Missing measured own-player baseline or actual yards")
    results=[]
    for (defense,role,metric),g in observations.groupby(
            ["defense","role","yardage_type"],sort=True):
        n=len(g);baseline=float(g["own_pregame_mean_yards"].mean())
        yards=float(g["actual_yards"].mean())
        above=int(g["above_own_baseline"].sum())
        pct=100*(yards-baseline)/baseline if baseline else np.nan
        results.append({
            "defense":defense,"role":role,"yardage_type":metric,
            "player_games":n,
            "defense_games":int(g[["season","week"]].drop_duplicates().shape[0]),
            "average_player_pregame_yards":round(baseline,2),
            "average_allowed_yards":round(yards,2),
            "extra_yards_vs_player_average":round(yards-baseline,2),
            "extra_yards_pct":round(float(pct),1),
            "median_excess_yards":round(float(g["difference_yards"].median()),2),
            "above_baseline_games":above,
            "above_baseline_pct":round(100*above/n,1),
            "enough_games":bool(n>=min_sample),
            "repeated_above_baseline":bool(
                n>=min_sample and above/n>=min_consistency_rate and
                pct>=min_excess_pct*100),
            "description_only_no_prediction_claim":True,
        })
    return pd.DataFrame(results).sort_values(
        ["role","extra_yards_vs_player_average"],ascending=[True,False]).reset_index(drop=True)
