"""Research-only role-aware PREKICKOFF target-share allocation for WR1/WR2.

Only a documented historical target-share gap may move targets away from the
unmodeled OTHER bucket. No new pass attempts, players, targets, TDs or routes
are fabricated. OTHER is partially protected for real unmodeled receivers
and throwaways. Missing/incomplete pregame evidence means NO adjustment.

A normalized role is an *observed target-rate rank*, not an official WR depth
chart designation. This is a candidate, NOT a calibrated live scorer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team

MAX_ROLE_SHARE_ADD = .025
MIN_OTHER_SHARE = .10


def allocate_role_target_share(target_rates,other_budget,skill,rates,*,team,
                               team_environment,target_season,target_week,
                               strength=.0,unmodeled_teammate_targets=0.):
    """Return adjusted active targets + OTHER and transparent diagnostics.

    The team pass-attempt forecast and each receiver's prior target history
    derive only from games preceding the requested target week. The modeled
    eligible WR1/WR2/WR3 must cover the top three recent WR histories, so the
    candidate cannot take targets from a missing high-volume WR3. Only the
    unprotected portion of OTHER can be a donor; RB/TE/WR3 rates never decrease.
    """
    if isinstance(strength,bool) or not isinstance(strength,(int,float,np.floating)) or (
        not np.isfinite(strength)) or not 0<=strength<=1:
        raise ValueError("Role target-share strength must be finite [0,1]")
    active=np.asarray(target_rates,dtype=float).copy()
    leftover=float(other_budget)
    if active.ndim!=1 or len(active)!=len(skill) or (
        not np.isfinite(active).all()) or (active<0).any() or (
        not np.isfinite(leftover)) or leftover<0:
        raise ValueError("Invalid prior role target allocation")
    original_budget=float(active.sum()+leftover)
    if original_budget<=0:
        raise ValueError("No receiving target share budget")
    info={
        "source":"strictly pregame nflverse WR target/game, team pass attempts and roster evidence",
        "status":"disabled" if strength==0 else "no_adjustment",
        "strength":float(strength),
        "total_budget_before":original_budget,
        "total_budget_after":original_budget,
        "other_budget_before":leftover,
        "other_budget_after":leftover,
        "other_protected_minimum":None,
        "role_proposals":[],
        "actual_transferred_targets_pg":0.,
        "top3_covered":False,
    }
    if strength==0:
        return active,leftover,info
    if target_season is None or target_week is None:
        raise ValueError("Dated pregame target week required for role candidate")
    season,week=int(target_season),int(target_week)
    if not (2020<=season<=2100 and 1<=week<=22):
        raise ValueError("Invalid role target-week identity")
    if not isinstance(team_environment,dict):
        raise ValueError("Verified pregame team volume is required")
    env=team_environment
    if not str(env.get("source","")).startswith("nflverse"):
        raise ValueError("Role share candidate requires observed NFL team pass volume")
    if int(env.get("observed_games",0))<3:
        info["status"]="insufficient_team_games"
        return active,leftover,info
    env_last=(int(env.get("last_observed_season",0)),
              int(env.get("last_observed_week",0)))
    if env_last >= (season,week):
        raise ValueError("Target-week team passing observation leaked into role model")
    passes=float(env.get("expected_pass_attempts",float("nan")))
    if not np.isfinite(passes) or not 18<=passes<=50:
        raise ValueError("Invalid measured pregame passing attempt prior")
    if not isinstance(rates,pd.DataFrame):
        raise ValueError("Prior-week observed player rates required")
    cols={"player","team","position","games_played","targets_pg",
          "latest_observed_season","latest_observed_week"}
    if not cols.issubset(rates.columns):
        raise ValueError("Observed WR ranks missing pregame identity columns")
    team=canonical_team(team)
    # Strict verified current-season prior recency. Excluding historical
    # absences is not a claim that a player is active today.
    observed=[]
    for row in rates.to_dict("records"):
        if canonical_team(row.get("team"))!=team or str(row.get("position")).upper()!="WR":
            continue
        date=(int(row.get("latest_observed_season") or 0),
              int(row.get("latest_observed_week") or 0))
        if date>=(season,week):
            raise ValueError("Target/future WR usage leaked into target-share candidate")
        if date[0]!=season or not (0<week-date[1]<=3):
            continue
        games=int(row.get("games_played") or 0)
        if games<3:
            continue
        try:
            t=float(row.get("targets_pg"))
        except (TypeError,ValueError):
            continue
        if not np.isfinite(t) or t<=0:
            continue
        identity=_norm_name(row["player"])
        if identity:
            observed.append((identity,t,games))
    names=[n for n,_,_ in observed]
    if len(set(names))!=len(names):
        raise ValueError("Ambiguous same-team WR history prevents role assignment")
    ranked=sorted(observed,key=lambda x:(-x[1],x[0]))
    if len(ranked)<3:
        info["status"]="too_few_prior_ranked_wrs"
        return active,leftover,info
    skill_names={
        _norm_name(p["Name"]):j for j,p in enumerate(skill.to_dict("records"))
        if str(p["Position"]).upper()=="WR"
    }
    # No transfer when a prior top-three player is missing from the eligible
    # slate. Their recent volume belongs in OTHER until availability verified.
    if not all(n in skill_names for n,_,_ in ranked[:3]):
        info["status"]="prior_top3_not_eligible"
        return active,leftover,info
    info["top3_covered"]=True
    protected=max(MIN_OTHER_SHARE*original_budget,
                  max(0.,float(unmodeled_teammate_targets)))
    info["other_protected_minimum"]=float(protected)
    available=max(0.,leftover-protected)
    if available<=1e-9:
        info["status"]="no_unprotected_other_budget"
        return active,leftover,info

    proposals=[]
    for idx,(identity,prior_targets,games) in enumerate(ranked[:2],start=1):
        j=skill_names[identity]
        current_share=float(active[j]/original_budget)
        prior_share=float(prior_targets/passes)
        gap=max(0.,prior_share-current_share)
        desired=min(MAX_ROLE_SHARE_ADD, float(strength)*gap)*original_budget
        proposals.append(dict(index=j,role=f"WR{idx}",player_key=identity,
                              prior_games=int(games),prior_targets_pg=prior_targets,
                              prior_target_share=prior_share,
                              current_share=current_share,
                              proposed_add_targets_pg=desired))
    wish=sum(x["proposed_add_targets_pg"] for x in proposals)
    if wish<=1e-9:
        info["status"]="no_positive_prior_share_gap"
        info["role_proposals"]=proposals
        return active,leftover,info
    factor=min(1.,available/wish)
    for proposal in proposals:
        transfer=proposal["proposed_add_targets_pg"]*factor
        active[proposal["index"]]+=transfer
        leftover-=transfer
        proposal["transferred_targets_pg"]=float(transfer)
    if (leftover+1e-8<protected or (active<0).any()
        or not np.isclose(active.sum()+leftover,original_budget,
                          rtol=0,atol=1e-8)):
        raise ValueError("Role target-share accounting violated OTHER protection or conservation")
    info.update(status="allocated",total_budget_after=float(active.sum()+leftover),
                other_budget_after=float(leftover),
                role_proposals=proposals,
                actual_transferred_targets_pg=float(wish*factor))
    return active,leftover,info
