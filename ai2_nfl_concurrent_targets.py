"""Pregame, *concurrent team-game* receiver-opportunity research.

Historical player rates are conditioned on different injury/depth-chart weeks
and cannot simply be summed into the current team's pass-attempt budget.
This module examines complete SAME-TEAM SAME-GAME box-score target ledgers,
strictly BEFORE the target week, and estimates three attempt-slot fractions:
modeled RB/WR/TE receivers, unmodeled RB/WR/TE receivers, and attempts lacking
an identifiable RB/WR/TE target in the weekly dataset. The latter is NOT a
certified play-by-play throwaway count (spikes/trick-play oddities also exist).

All outputs are retrospective research priors, NOT verified active rosters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from props_data import clean_weekly
from ai2_nfl_matchups import canonical_team
from ai2_nfl_auto_baselines import _norm_name


def pregame_concurrent_target_prior(
    history, *, team, roster, season, week, lookback=6,
    min_games=3, decay=.85
):
    """Return dated target fractions; None for insufficient historical coverage.

    Shares are ratios of *weighted same-game counts*, never the sum of
    independently fitted player averages. No target-week or future outcomes.
    Roster must be the actual pregame modeled player pool (NOT a hindsight list).
    """
    if not isinstance(roster,pd.DataFrame):
        raise ValueError("Concurrent target study requires pregame modeled roster")
    required={"Name","TeamAbbrev","Position"}
    if not required.issubset(roster.columns):
        raise ValueError("Concurrent target roster missing identity columns")
    if int(lookback)<int(min_games) or int(min_games)<2 or not (0<decay<=1):
        raise ValueError("Invalid concurrent target window settings")
    season=int(season);week=int(week);team=canonical_team(team)
    if not (2020<=season<=2100 and 1<=week<=22 and team):
        raise ValueError("Invalid target-game identity")
    modeled={
        (_norm_name(row["Name"]),str(row["Position"]).upper())
        for row in roster.to_dict("records")
        if canonical_team(row["TeamAbbrev"])==team and
        str(row["Position"]).upper() in ("RB","WR","TE")
    }
    if not modeled:
        raise ValueError("No eligible modeled receiver identities in team game")
    f=clean_weekly(history)
    f=f[(f["season"].eq(season))&(f["week"].lt(week))&
        f["recent_team"].map(canonical_team).eq(team)].copy()
    if f.empty:
        return None
    snapshots=[]
    for (yr,w),game in f.groupby(["season","week"],sort=True):
        qb=game[game["position"].astype(str).str.upper().eq("QB")]
        skill=game[game["position"].astype(str).str.upper().isin(("RB","WR","TE"))]
        attempts=pd.to_numeric(qb["attempts"],errors="coerce")
        targets=pd.to_numeric(skill["targets"],errors="coerce")
        if (qb.empty or skill.empty or attempts.isna().any() or
                targets.isna().any() or (attempts<0).any() or
                (targets<0).any()):
            continue
        a=float(attempts.sum());t=float(targets.sum())
        if not (15<=a<=65) or t>a+1e-8:
            continue
        if skill.duplicated(["player_name","position"]).any():
            raise ValueError(f"Ambiguous same-game receiver identities {team} {yr} W{w}")
        identified=0.
        for player,target in zip(skill.to_dict("records"),targets):
            key=(_norm_name(player["player_name"]),str(player["position"]).upper())
            if key in modeled:
                identified+=float(target)
        other=max(0.,t-identified)
        unassigned=max(0.,a-t)
        if abs((identified+other+unassigned)-a)>1e-6:
            raise ValueError("Historical team target accounting does not reconcile")
        snapshots.append(dict(season=int(yr),week=int(w),attempts=a,
                              modeled=identified,unmodeled=other,
                              no_identified_target=unassigned))
    recent=snapshots[-int(lookback):]
    if len(recent)<min_games:
        return None
    weights=np.power(float(decay),np.arange(len(recent)-1,-1,-1))
    attempts=float(np.dot(weights,[x["attempts"] for x in recent]))
    if attempts<=0:
        return None
    fractions={
        "modeled_share":float(np.dot(weights,[x["modeled"] for x in recent])/attempts),
        "unmodeled_share":float(np.dot(weights,[x["unmodeled"] for x in recent])/attempts),
        "no_identified_target_share":float(np.dot(
            weights,[x["no_identified_target"] for x in recent])/attempts)
    }
    if any(not np.isfinite(x) or x<0 for x in fractions.values()):
        raise ValueError("Invalid concurrent target fractions")
    if not np.isclose(sum(fractions.values()),1.,rtol=0,atol=1e-8):
        raise ValueError("Concurrent same-game opportunity shares fail conservation")
    return {
        "team":team,"target_season":season,"target_week":week,
        "source":"nflverse dated same-team-game QB attempts and RB/WR/TE targets",
        "observed_games":len(recent),
        "last_observed_season":recent[-1]["season"],
        "last_observed_week":recent[-1]["week"],
        "selected_receivers_in_roster":len(modeled),
        "historical_attempts_weighted":round(attempts,5),
        "historical_team_games":recent,
        "game_day_active_verified":False,
        "research_only":True,
        **fractions,
    }


def allocate_concurrent_target_budget(target_rates, pass_attempt_prior, profile):
    """Reconcile modeled + OTHER + unidentified attempt slots to QB attempts.

    Current player *relative* rates determine within-modeled-player rank.
    Same-game historical allocation decides the team's modeled-versus-OTHER
    versus no-identifiable-target fractions. Never create extra attempts.
    """
    r=np.asarray(target_rates,dtype=float)
    a=float(pass_attempt_prior)
    if r.ndim!=1 or not len(r) or (r<0).any() or not np.isfinite(r).all():
        raise ValueError("Invalid pregame modeled target rates")
    if not np.isfinite(a) or a<=0 or r.sum()<=0:
        raise ValueError("Missing positive QB attempts or modeled receiver history")
    if not isinstance(profile,dict) or not profile.get("research_only"):
        raise ValueError("Concurrent prior requires a dated research source")
    if not str(profile.get("source","")).startswith("nflverse dated same-team-game"):
        raise ValueError("Unsupported concurrent target evidence source")
    shares=np.array([profile.get(k,float("nan")) for k in (
        "modeled_share","unmodeled_share","no_identified_target_share")],float)
    if (not np.isfinite(shares).all() or (shares<0).any() or
            not np.isclose(shares.sum(),1.,rtol=0,atol=1e-8)):
        raise ValueError("Invalid prior receiver attempt-slot shares")
    # No guessed minimum OTHER: different historical teammates are not all
    # assumed active together. Zero modeled share is a hard no-data blocker.
    if shares[0]<=0:
        raise ValueError("No observed current-pool receivers in concurrent history")
    modeled=r/r.sum()*(a*shares[0])
    other=float(a*shares[1])
    no_target=float(a*shares[2])
    if not np.isclose(modeled.sum()+other+no_target,a,rtol=0,atol=1e-6):
        raise ValueError("Concurrent target allocation changed QB pass budget")
    return modeled,other,no_target
