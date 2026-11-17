"""Role-aware WR1/WR2 target-share research A/B on existing NFL simulator.

A single optional allocation parameter changes only share distribution from
modeled OTHER to *verified historically top-ranked, eligible* WR1/WR2.
Total team pass attempts, team-target budgets, and untouched RB/TE/WR3
pregame allocations are not manually increased.

All outcomes here were previously inspected, hence NOT an independent holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate
from scripts.audit_nfl_team_pass_prior_ab import (
    WINDOWS,PLAYER_KEYS,WR_KEYS,TEAM_KEYS,exact_paired,errors,by_groups,
    bootstrap_game_ci,
)

STAT_KEYS=PLAYER_KEYS+["stat"]

def compare(scores,roles,teams,stats,diag):
    players=exact_paired(scores["base"],scores["candidate"],PLAYER_KEYS,
            ["actual_partial_dk","history_mean_partial_dk","opponent"])
    wr=exact_paired(roles["base"],roles["candidate"],WR_KEYS,
            ["actual_targets","role","rank","prior_targets_pg","opponent"])
    team=exact_paired(teams["base"],teams["candidate"],TEAM_KEYS,
            ["actual_pass_attempts","historical_team_pass_baseline","passing_qb"])
    stat=exact_paired(stats["base"],stats["candidate"],STAT_KEYS,
            ["actual","prior_baseline","category","direct_dk_weight"])
    role_diag=pd.DataFrame(diag)
    if role_diag.empty or role_diag.duplicated(TEAM_KEYS).any():
        raise ValueError("No unique team-game role audit")
    if len(role_diag)!=len(team):
        raise ValueError("Incomplete candidate team coverage")
    for _,x in role_diag.iterrows():
        if x["actual_transferred_targets_pg"]<-1e-9 or (
            x["other_budget_after"]>x["other_budget_before"]+1e-8):
            raise ValueError("Invalid recipient/OTHER target conservation")
        if x["status"]=="allocated":
            if not x["top3_covered"] or x["other_budget_after"]+1e-8<x["other_protected_minimum"]:
                raise ValueError("Ineligible WR3 or real OTHER teammate was robbed")
    out={
        "player_partial_dk":errors(players,"actual_partial_dk_base",
                                  "simulated_mean_dk_base","simulated_mean_dk_candidate"),
        "by_position":by_groups(players,"position","actual_partial_dk_base",
                                  "simulated_mean_dk_base","simulated_mean_dk_candidate"),
        "game_cluster_player_lift_ci":bootstrap_game_ci(
            players,"actual_partial_dk_base",
            "simulated_mean_dk_base","simulated_mean_dk_candidate"),
        "WR_targets":errors(wr,"actual_targets_base",
                           "predicted_targets_base","predicted_targets_candidate"),
        "WR_targets_by_pregame_role":by_groups(
            wr,"role_base","actual_targets_base",
            "predicted_targets_base","predicted_targets_candidate"),
        "game_cluster_WR_target_lift_ci":bootstrap_game_ci(
            wr,"actual_targets_base","predicted_targets_base",
            "predicted_targets_candidate"),
        "team_pass_attempts":errors(team,"actual_pass_attempts_base",
                                   "model_pass_attempts_base","model_pass_attempts_candidate"),
        "stat_by_position":{},
        "team_role_coverage":{
            "total_team_games":len(role_diag),
            "allocated_team_games":int(role_diag.status.eq("allocated").sum()),
            "top3_all_eligible_team_games":int(role_diag.top3_covered.sum()),
            "total_targets_per_game_shift_from_other":round(float(
                   role_diag.actual_transferred_targets_pg.sum()/len(role_diag)),4),
            "mean_transfer_when_activated":round(float(
                  role_diag.loc[role_diag.status.eq("allocated"),
                                "actual_transferred_targets_pg"].mean()),4)
                  if role_diag.status.eq("allocated").any() else 0.,
            "no_adjustment_reason_counts":dict(Counter(role_diag.status)),
        },
        "p90_actual_below":{
            "player_base_pct":round(100*float(
                (players.actual_partial_dk_base<=players.simulated_p90_dk_base).mean()),2),
            "player_candidate_pct":round(100*float(
                (players.actual_partial_dk_base<=players.simulated_p90_dk_candidate).mean()),2),
            "WR_target_base_pct":round(100*float(
                (wr.actual_targets_base<=wr.predicted_p90_base).mean()),2),
            "WR_target_candidate_pct":round(100*float(
                (wr.actual_targets_base<=wr.predicted_p90_candidate).mean()),2),
        },
        "count":{"players":len(players),"WR":len(wr),"teams":len(team),"stats":len(stat)},
    }
    for (position,name),grp in stat.groupby(["position","stat"]):
        if name not in {"targets","receptions","receiving_yards","receiving_tds",
                        "passing_yards","pass_attempts","rushing_yards","rushing_tds"}:
            continue
        out["stat_by_position"][str(position)+"."+str(name)]=errors(
            grp,"actual_base","predicted_mean_base","predicted_mean_candidate")
    return out,players,wr,team,stat,role_diag


def replay(weekly,schedules,snaps,*,windows=WINDOWS,n_sims=220):
    result={
        "title":"NFL role-anchored WR target share controlled retrospective",
        "strength":{"baseline":0.0,"candidate":0.7},
        "historical_role_gate":{
            "target_history_min_games":3,
            "all_top_three_eligible":True,
            "MAX_ROLE_SHARE_ADD":0.025,
            "MIN_OTHER_SHARE":0.10,
            "protect_prior_unmodeled_player_targets":True,
        },
        "other_yard_strength":0.0,
        "team_pass_env_weight":0.5,
        "production_approved":False,
        "one_click_modified":False,
        "independent_holdout":False,
        "candidate_chosen_after_2025_2026_outcomes_inspected":True,
        "simulations_per_game":n_sims,
        "windows":{},
        "limitations":[
            "2025 early, 2025 reference and 2026 retrospective already inspected: NOT blinded",
            "No dated real historical DK salaries or verified game-day active WR roster",
            "No verified routes-run; observed WR target ranks are NOT official depth-chart labels",
            "OTHER includes unmodeled receivers and throwaways; conservative floor is only a proxy",
            "The candidate only transfers targets when WR1/WR2 plus WR3 have prior eligible usage",
            "P90 coverage with 220 scenarios is descriptive, not production 10000 calibration",
            "Random scenario paths may diverge after changed target multinomial draw",
            "No projected return/fumble scoring in partial DK history",
        ]
    }
    output={}
    for label,(season,weeks) in windows.items():
        run={}
        for model,strength in (("base",0.),("candidate",.7)):
            role=[];team=[];stat=[];role_diag=[]
            score,issues=evaluate(weekly,schedules,snaps,season,weeks,
                games_per_week=None,n_sims=n_sims,team_pass_env_weight=.5,
                other_yard_strength=0.,wr_role_strength=strength,
                wr_role_rows=role,team_pass_rows=team,
                role_allocation_rows=role_diag,stat_audit_rows=stat,
                audit_model_name=model)
            if issues or score.empty:
                raise ValueError(f"{label}/{model} failed historical games: {issues[:8]}")
            run[model]={
                "scores":score,"roles":pd.DataFrame(role),"teams":pd.DataFrame(team),
                "stats":pd.DataFrame(stat),"diag":role_diag
            }
        metrics,pc,wc,tc,st,rd=compare(
            {x:r["scores"] for x,r in run.items()},
            {x:r["roles"] for x,r in run.items()},
            {x:r["teams"] for x,r in run.items()},
            {x:r["stats"] for x,r in run.items()},run["candidate"]["diag"])
        if len(rd)<40 or len(wc)<80:
            raise ValueError("Incomplete season target-role sample")
        result["windows"][label]=metrics
        output[label+"_players"]=pc
        output[label+"_wr"]=wc
        output[label+"_team"]=tc
        output[label+"_football_stats"]=st
        rd=rd.copy()
        rd["role_proposals"]=rd.role_proposals.map(json.dumps)
        output[label+"_role_transfer_ledger"]=rd
        print(label,json.dumps(metrics,sort_keys=True),flush=True)
    return result,output


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-sims",type=int,default=220)
    ap.add_argument("--out-dir",default="nfl_role_target_share_ab")
    a=ap.parse_args(argv)
    if not 100<=a.n_sims<=10000:
        ap.error("Scenario count must be 100..10000")
    history=fetch_weekly([2024,2025,2026])
    snaps=fetch_snap_counts([2025,2026])
    sched=pd.read_csv(GAMES_SOURCE)
    report,files=replay(history,sched,snaps,n_sims=a.n_sims)
    out=Path(a.out_dir);out.mkdir(exist_ok=True,parents=True)
    for key,frame in files.items():
        frame.to_csv(out/(key+".csv"),index=False)
    (out/"validation.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print("Completed research only: no production change or predictive acceptance",flush=True)

if __name__=="__main__":
    main()
