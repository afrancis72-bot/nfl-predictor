"""Audit actual scored NFL player stat opportunities against pregame observed rates.

Flags are evidence for a player's final human review. This does not change
scored football outcomes, auto-veto a candidate, claim live usage, or infer
injured teammates. It distinguishes *simulated* fantasy events from observed
nflverse rates and calls out unusually aggressive extrapolation.
"""
from __future__ import annotations
import math
import pandas as pd
from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team

SOURCE="nflverse weekly stats"


def _number(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError):
        return None


def compare_simulated_role_to_history(audits, historical_rates, injury_context=None):
    """Use scored actual football worlds and historically observed player games.

    Thresholds are review flags, NOT fitted predictive cutoffs or projections.
    A justified new role may appropriately exceed them. Never auto-cap points.
    """
    if not isinstance(historical_rates,pd.DataFrame):
        raise ValueError("Historical player evidence missing; role review unavailable")
    required={"player","team","position","games_played","source"}
    if not required.issubset(historical_rates.columns):
        raise ValueError("Role review missing observed stat provenance")
    history={}
    for r in historical_rates.to_dict("records"):
        key=(_norm_name(r["player"]),canonical_team(r["team"]),
             str(r["position"]).upper())
        if key in history:
            history[key]=None
        else:
            history[key]=r
    prior_absences={}
    for x in injury_context or []:
        team=canonical_team(x.get("Team") or "")
        prior_absences.setdefault(team,[]).append({
            "player":x.get("Unavailable player"),
            "already_reflected":x.get("Already reflected share"),
            "incremental_pct":x.get("Additional redistribution %")})
    results=[]
    for audit in audits or []:
        if not str(audit.get("Audit status") or "").startswith(
                "Verified: per-simulation DK points match components"):
            continue
        player=str(audit.get("Player") or "")
        team=canonical_team(audit.get("Team") or "")
        position=str(audit.get("Position") or "").upper()
        raw=history.get((_norm_name(player),team,position))
        if raw is None or not str(raw.get("source") or "").startswith(SOURCE):
            continue
        n=int(_number(raw.get("games_played")) or 0)
        if n<3:
            continue
        tests=[
            ("Targets", "Mean targets", "targets_pg",.35,3.),
            ("Carries", "Mean carries", "carries_pg",.35,4.),
        ]
        if position in {"WR","TE","RB"}:
            for label,pred_key,obs_key,percent,min_extra in tests:
                sim=_number(audit.get(pred_key))
                observed=_number(raw.get(obs_key))
                if sim is None or observed is None or observed<2.:
                    continue
                if sim < observed*(1.+percent) or sim-observed<min_extra:
                    continue
                results.append({
                    "Player":player,"Team":team,"Position":position,
                    "Concern":label+" significantly above observed role",
                    "Historically observed per game":round(observed,2),
                    "Simulated per game":round(sim,2),
                    "Increase %":round(100*(sim/observed-1.),1),
                    "Prior player-games":n,
                    "Current-week injury context":", ".join(
                        str(x["player"])+" ("+str(x["incremental_pct"])+"% newly freed)"
                        for x in prior_absences.get(team,[])) or
                        "No sourced OUT teammate in injury adjustment",
                    "Required review":"Verify recent actual workload and team rotation; "
                                      "do not automatically cap or exclude",
                    "Evidence":"nflverse weighted pregame rates vs same-world NFL simulation",
                })
        # A high share of team touchdowns is a consequential difference from
        # observed player scoring, even when targets/carries are unchanged.
        rec=_number(audit.get("Mean receiving TD")) or 0.
        rush=_number(audit.get("Mean rushing TD")) or 0.
        observed_rec=_number(raw.get("receiving_tds_pg")) or 0.
        observed_rush=_number(raw.get("rushing_tds_pg")) or 0.
        if n>=5 and rec+rush>=.85 and rec+rush-(observed_rec+observed_rush)>=.35:
            results.append({
                "Player":player,"Team":team,"Position":position,
                "Concern":"Simulated touchdowns substantially exceed observed rate",
                "Historically observed per game":round(observed_rec+observed_rush,3),
                "Simulated per game":round(rec+rush,3),
                "Increase %":(round(100*((rec+rush)/(observed_rec+observed_rush)-1),1)
                              if observed_rec+observed_rush>.05 else None),
                "Prior player-games":n,
                "Current-week injury context":", ".join(
                    str(x["player"])+" ("+str(x["incremental_pct"])+"% newly freed)"
                    for x in prior_absences.get(team,[])) or
                    "No sourced OUT teammate in injury adjustment",
                "Required review":"Check team TD budget and who is actually receiving goal-line "
                                  "or end-zone opportunities; don't assume every high projection is wrong",
                "Evidence":"nflverse weighted pregame TD rates vs same-world NFL simulation",
            })
    return sorted(results,key=lambda r:(r["Team"],r["Player"],r["Concern"]))
