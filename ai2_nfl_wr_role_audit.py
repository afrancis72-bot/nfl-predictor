"""Research-only pregame WR1 / WR2 / WR3 target and OTHER bucket audit.

Rank ALL recently observed same-team receivers using their PRE-game nflverse
target averages, not their target-week performance, projections, or future
availability. Keep their simulated eligibility separately: a ranked WR3 can
be absent from the model because the current historical pool requires >=3.5
prior-week targets/game. This is a usage-rank proxy, never an official depth
chart, slot/outside assignment or injury verification.

Actual target-week outcomes are used ONLY after ranks are fixed, for grading.
The model's OTHER target bucket includes unmodeled receivers, throwaways and
other unattributed attempts. It is not directly equivalent to the actual
team's unmodeled-player receiving targets.

Does not modify any production scoring or roster eligibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team


def _finite(value):
    try:
        v=float(value)
    except (ValueError, TypeError):
        return None
    return v if np.isfinite(v) else None


def pregame_receiver_ranks(rates, pool, season, week, teams):
    """All prior-active WRs ranked before kickoff, modeled eligibility separate."""
    if not isinstance(rates,pd.DataFrame) or not isinstance(pool,pd.DataFrame):
        raise ValueError("Historical rates and pregame eligible pool required")
    eligible = {
        (canonical_team(row["TeamAbbrev"]),_norm_name(row["Name"]))
        for row in pool.to_dict("records")
        if str(row["Position"]).upper()=="WR"
    }
    teams={canonical_team(t) for t in teams}
    candidates={t:[] for t in teams}
    for row in rates.to_dict("records"):
        if str(row.get("position","")).upper()!="WR":
            continue
        t=canonical_team(row.get("team",""))
        if t not in teams:
            continue
        year=int(row.get("latest_observed_season") or 0)
        last=int(row.get("latest_observed_week") or 0)
        games=int(row.get("games_played") or 0)
        targets=_finite(row.get("targets_pg"))
        if (year!=int(season) or last>=int(week)
            or int(week)-last>3 or games<2 or targets is None or targets<=0):
            continue
        name=_norm_name(row["player"])
        if not name:
            continue
        candidates[t].append((name, targets, games))
    labels={}
    by_team={}
    for team,items in candidates.items():
        names=[n for n,_,_ in items]
        if len(names)!=len(set(names)):
            raise ValueError("Ambiguous same-team WR history; cannot assign role")
        ranked=sorted(items,key=lambda z:(-z[1],z[0]))
        by_team[team]=[]
        for idx,(name,targets,games) in enumerate(ranked,1):
            role=f"WR{idx}" if idx<=3 else "WR4+"
            row={
                "team":team,"player_key":name,"prior_rank":idx,"role":role,
                "prior_targets_pg":targets,"prior_games":games,
                "modeled":(team,name) in eligible,
            }
            labels[(team,name)]=row
            by_team[team].append(row)
    return labels, by_team


def receiver_row(*, player,team,opponent,season,week,game,observed,prior,
                 simulated_stats,role_info,n_sims,model="neutral"):
    """One matched modeled receiver with a pregame-only role label."""
    if role_info is None:
        raise ValueError("Model WR lacks pregame role evidence")
    if role_info["team"]!=canonical_team(team):
        raise ValueError("Receiver role assigned to another team")
    if not role_info["modeled"]:
        raise ValueError("Receiver missing from pregame eligible player pool")
    old=_finite(prior.get("targets_pg"))
    actual=_finite(observed.get("targets"))
    if old is None or actual is None:
        return None
    if (int(prior.get("latest_observed_season") or 0)!=int(season)
        or int(prior.get("latest_observed_week") or 0)>=int(week)):
        raise ValueError("Target-week observations in WR pregame role baseline")
    if not isinstance(simulated_stats,dict) or "targets" not in simulated_stats:
        raise ValueError("Missing modeled receiver target distribution")
    draws=np.asarray(simulated_stats["targets"],dtype=float)
    if draws.ndim!=1 or len(draws)!=n_sims or not np.isfinite(draws).all() or (draws<0).any():
        raise ValueError("Invalid simulated receiving target distribution")
    return {
        "season":int(season),"week":int(week),"game":str(game),
        "player":str(player),"team":str(team),"opponent":str(opponent),
        "role":role_info["role"],"rank":role_info["prior_rank"],
        "prior_targets_pg":float(old),"role_prior_targets_pg":float(role_info["prior_targets_pg"]),
        "prior_games":role_info["prior_games"],"actual_targets":float(actual),
        "predicted_targets":float(draws.mean()),
        "predicted_p10":float(np.percentile(draws,10)),
        "predicted_p90":float(np.percentile(draws,90)),
        "n_sims":int(n_sims),"model":str(model),
    }


def team_other_row(*, team, opponent,season,week,game,
                   pool,observations,stat_draws,history_ranked):
    """Compare team budgets and actual modeled/nonmodeled target coverage.

    Model OTHER is not the same statistical population as actual unmodeled
    receiving targets; it additionally includes unattached attempts. Display
    both side by side as diagnostics, NEVER treat their difference as bias.
    """
    label=str(team)
    diag=stat_draws.get(f"__OPPORTUNITY__|{label}")
    other=stat_draws.get(f"__OTHER_RECEIVER__|{label}")
    if not isinstance(diag,dict) or not isinstance(other,dict):
        raise ValueError(f"Missing simulation target conservation for {label}")
    model_skill=pool[
        (pool["TeamAbbrev"].eq(team)) &
        (pool["Position"].isin(("RB","WR","TE")))
    ]
    modeled_keys={
        (_norm_name(z["Name"]),str(z["Position"]).upper())
        for z in model_skill.to_dict("records")
    }
    if not modeled_keys:
        raise ValueError(f"No modeled receiving eligibility for {label}")
    played=observations[
        (observations["recent_team"].map(canonical_team).eq(team)) &
        (observations["position"].astype(str).str.upper().isin(("RB","WR","TE")))
    ].copy()
    played["targets"] = pd.to_numeric(played["targets"],errors="coerce")
    known=played[played["targets"].notna()]
    actual_total=float(known["targets"].sum())
    modeled_truth=sum(float(z["targets"]) for z in known.to_dict("records")
                      if (_norm_name(z["player_name"]),str(z["position"]).upper()) in modeled_keys)
    actual_nonmodeled=actual_total-modeled_truth
    other_draw=np.asarray(other["targets"],dtype=float)
    if not np.isfinite(other_draw).all() or (other_draw<0).any():
        raise ValueError("Invalid OTHER target distribution")
    total=other_draw.copy()
    for player in model_skill.to_dict("records"):
        s=stat_draws.get(str(player["Name"]))
        if not isinstance(s,dict) or "targets" not in s:
            raise ValueError("No modeled receiving targets for eligible player")
        arr=np.asarray(s["targets"],dtype=float)
        if arr.shape!=total.shape or (arr<0).any():
            raise ValueError("Incompatible target sample dimensions")
        total+=arr
    if not (total>0).any():
        raise ValueError("No simulator team targets")
    modeled_receivers=(
        model_skill["Position"].eq("WR").sum()
    )
    rank3=history_ranked[:3]
    excluded_top=[x for x in rank3 if not x["modeled"]]
    return {
        "season":int(season),"week":int(week),"game":str(game),
        "team":str(team),"opponent":str(opponent),
        "historically_rankable_wr":int(len(history_ranked)),
        "eligible_modeled_wr":int(modeled_receivers),
        "historical_top3_wr_not_modeled":int(len(excluded_top)),
        "top3_omitted_roles":",".join(x["role"] for x in excluded_top),
        "actual_team_rb_wr_te_targets_known":actual_total,
        "actual_targets_to_modeled_players":modeled_truth,
        "actual_nonmodeled_rb_wr_te_targets":float(actual_nonmodeled),
        "actual_team_skill_player_rows":int(len(played)),
        "actual_skill_player_missing_targets":int(played["targets"].isna().sum()),
        "actual_nonmodeled_target_fraction":(
            float(actual_nonmodeled/actual_total) if actual_total>0 else None),
        "simulated_other_targets_mean":float(other_draw.mean()),
        "simulated_total_allocated_targets_mean":float(total.mean()),
        "simulated_other_target_fraction":float(
            other_draw.sum()/total.sum()),
        "other_budget_pg_from_model":float(diag["other_target_allocation_budget_pg"]),
        "unmodeled_history_target_budget_pg":float(
            diag["unmodeled_teammate_target_budget_pg"]),
        "model_source":str(diag.get("out_context_method","")),
    }


def _metrics(frame):
    if frame.empty:
        return {"n":0}
    y=frame["actual_targets"].to_numpy(float)
    pred=frame["predicted_targets"].to_numpy(float)
    base=frame["prior_targets_pg"].to_numpy(float)
    p10=frame["predicted_p10"].to_numpy(float)
    p90=frame["predicted_p90"].to_numpy(float)
    if not all(np.isfinite(v).all() for v in (y,pred,base,p10,p90)):
        raise ValueError("Nonfinite role forecast data")
    ae=np.abs(pred-y)
    be=np.abs(base-y)
    a=float(ae.mean()); b=float(be.mean())
    return {
        "n":int(len(frame)),
        "mean_actual_targets":round(float(y.mean()),3),
        "mean_predicted_targets":round(float(pred.mean()),3),
        "mean_historical_targets":round(float(base.mean()),3),
        "sim_mae_targets":round(a,4),
        "history_mae_targets":round(b,4),
        "sim_improvement_vs_history_pct":round(
            100*(b-a)/b,3) if b>0 else None,
        "sim_bias_targets":round(float((pred-y).mean()),3),
        "p90_actual_at_or_below_pct":round(
            100*float(np.mean(y<=p90)),2),
        "p10_p90_actual_within_pct":round(
            100*float(np.mean((y>=p10)&(y<=p90))),2),
    }


def receiver_role_report(rows,team_rows):
    if not rows or not team_rows:
        raise ValueError("No modeled WRs and team target accounting")
    f=pd.DataFrame(rows)
    t=pd.DataFrame(team_rows)
    if f.duplicated(["season","week","game","player","team"]).any():
        raise ValueError("Duplicate player-game role label")
    if t.duplicated(["season","week","game","team"]).any():
        raise ValueError("Duplicate team-game target budget")
    roles={
        name:_metrics(f[f["role"].eq(name)])
        for name in ("WR1","WR2","WR3","WR4+")
    }
    by_opponent=[]
    for (role,defense),group in f.groupby(["role","opponent"]):
        if len(group)<5:
            continue
        by_opponent.append({
            "role":role,"defense":defense,**_metrics(group)
        })
    team_has_missing=int(t["actual_skill_player_missing_targets"].sum())
    actual_total=t["actual_team_rb_wr_te_targets_known"].sum()
    actual_nonmodeled=t["actual_nonmodeled_rb_wr_te_targets"].sum()
    sim_total=t["simulated_total_allocated_targets_mean"].sum()
    sim_other=t["simulated_other_targets_mean"].sum()
    return {
        "all_matched_wr":_metrics(f),
        "roles":roles,
        "sample_team_games":int(len(t)),
        "rank3_omitted_team_games":int((t["historical_top3_wr_not_modeled"]>0).sum()),
        "historical_top3_wr_missing_model_spots":int(t["historical_top3_wr_not_modeled"].sum()),
        "eligible_wr_per_team_game_mean":round(float(t["eligible_modeled_wr"].mean()),3),
        "teams_with_3_or_more_modeled_wr":int((t["eligible_modeled_wr"]>=3).sum()),
        "actual_team_skill_player_missing_targets":team_has_missing,
        "actual_nonmodeled_rb_wr_te_target_share_pct":(
            round(100*float(actual_nonmodeled/actual_total),2)
            if actual_total>0 and team_has_missing==0 else None),
        "simulated_other_target_bucket_share_pct":(
            round(100*float(sim_other/sim_total),2) if sim_total>0 else None),
        "actual_and_sim_other_shares_not_directly_equivalent":True,
        "per_opponent_role_min_n5":by_opponent,
        "research_only":True,
        "production_approved":False,
        "caveats":[
            "WR1/2/3 proxy from pregame prior target ranking, not official starter status or DB coverage",
            "Historical role rank includes receivers with prior observed usage who may have been inactive on game day",
            "Observed actual nonmodeled targets omit unattributed attempts and are NOT the same as simulated OTHER targets",
            "Reconstructed pool excludes historically sparse receivers; inactive status not frozen",
            "Role sample sizes vary; small opponent-role combinations are not actionable",
            "All 2025/2026 periods have been previously inspected; not independent validation",
        ],
    }
