"""Participation-aware NFL target-rate forecast A/B, offline only.

The nflverse/PFR snap-count feed is an actual observed historical participation
signal, NOT verified target-week ACTIVE status or route counts. Only season/week
before kickoff is permitted. Training 2025 W9-12 selects an alternative snap
blend if it clears predefined WR1/WR2 target MAE and DK error thresholds.
2026 W4-10 is a RETROSPECTIVE confirmation already studied previously, NOT
an untouched future holdout. The baseline default remains 0.35.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate
from scripts.backtest_nfl_matchup_simulator import match_results,paired_metrics
from scripts.audit_nfl_wr_target_roles import _paired_neutral_possession
from ai2_nfl_wr_role_audit import _metrics

BASELINE_WEIGHT=.35
CANDIDATE_WEIGHTS=(0.,.15,.5,.7)
KEYS=["season","week","game","team","player"]


def run_forecast(weekly,schedules,snaps,season,weeks,n_sims,weight,games_per_week):
    target_rows=[];participation=[]
    predictions,skips=evaluate(
        weekly,schedules,snaps,season,weeks,
        games_per_week=games_per_week,n_sims=n_sims,
        isolate_team_rng=True,wr_role_rows=target_rows,
        snap_participation_rows=participation,
        snap_role_blend_weight=weight,
        audit_model_name=f"snap_{weight}")
    if predictions.empty or not target_rows or not participation:
        raise ValueError(f"No real historical participation grade {season} {weight}: {skips[:5]}")
    return predictions,pd.DataFrame(target_rows),pd.DataFrame(participation),skips


def paired_role_grade(base_wr,alt_wr):
    paired=_paired_neutral_possession(base_wr,alt_wr)[1]
    out={}
    for role in ("WR1","WR2","WR3","WR4+"):
        a=paired[paired["role_neutral"].eq(role)]
        if a.empty:
            out[role]={"n":0}
            continue
        actual=a["actual_targets_neutral"].to_numpy(float)
        pred0=a["predicted_targets_neutral"].to_numpy(float)
        pred1=a["predicted_targets_possession"].to_numpy(float)
        prior=a["prior_targets_pg_neutral"].to_numpy(float)
        out[role]={
            "n":len(a),
            "actual_targets_mean":round(float(actual.mean()),4),
            "baseline_mean_targets":round(float(pred0.mean()),4),
            "candidate_mean_targets":round(float(pred1.mean()),4),
            "historical_prior_mae":round(float(abs(actual-prior).mean()),4),
            "baseline_mae":round(float(abs(actual-pred0).mean()),4),
            "candidate_mae":round(float(abs(actual-pred1).mean()),4),
            "baseline_bias":round(float((pred0-actual).mean()),4),
            "candidate_bias":round(float((pred1-actual).mean()),4),
        }
    combined=paired[paired["role_neutral"].isin(("WR1","WR2"))]
    y=combined["actual_targets_neutral"].to_numpy(float)
    p0=combined["predicted_targets_neutral"].to_numpy(float)
    p1=combined["predicted_targets_possession"].to_numpy(float)
    old=float(abs(p0-y).mean())
    new=float(abs(p1-y).mean())
    out["WR1_WR2_COMBINED"]={
        "n":int(len(combined)),
        "baseline_mae":round(old,5),
        "candidate_mae":round(new,5),
        "mae_improvement_pct":round(100*(old-new)/old,4) if old else None,
        "baseline_bias":round(float((p0-y).mean()),4),
        "candidate_bias":round(float((p1-y).mean()),4),
    }
    return out,paired


def evaluate_weight(base,alternate,year,weight):
    p0,w0,e0,fail0=base
    p1,w1,e1,fail1=alternate
    if fail0!=fail1:
        raise ValueError(f"Different skipped games for {year} and {weight}: {fail0[:4]} / {fail1[:4]}")
    paired=match_results(p0,p1)
    role,pair_wr=paired_role_grade(w0,w1)
    joined=e0.merge(e1,on=["season","week","game","team","player","position"],
                    how="outer",validate="one_to_one",
                    suffixes=("_baseline","_candidate"),indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Different historical snap participant cohorts")
    if not joined["has_4plus_matched_pregame_snap_games_baseline"].eq(
            joined["has_4plus_matched_pregame_snap_games_candidate"]).all():
        raise ValueError("Candidate changed historical snap evidence coverage")
    if not joined["last_observed_snap_week_baseline"].fillna(-1).eq(
            joined["last_observed_snap_week_candidate"].fillna(-1)).all():
        raise ValueError("Different pregame snaps in same period")
    wr=joined[joined.position.eq("WR")]
    team=wr.groupby(["season","week","game","team"],sort=True)
    complete_three=int((team["has_4plus_matched_pregame_snap_games_baseline"]
                        .sum()>=3).sum())
    result={
        "period_year":year,"candidate_weight":float(weight),
        "dk":paired_metrics(paired),
        "WR_targets":role,
        "sampled_player_games":len(joined),
        "sampled_WR_player_games":len(wr),
        "WR_snap_evidence_count":int(wr[
            "has_4plus_matched_pregame_snap_games_baseline"].sum()),
        "WR_snap_evidence_coverage_pct":round(
            float(100*wr["has_4plus_matched_pregame_snap_games_baseline"].mean()),3),
        "team_games_with_3_or_more_WR_snap_records":complete_three,
        "team_games_total":int(wr.groupby(
            ["season","week","game","team"]).ngroups),
        "missing_snap_rows_are_not_inactive":True,
        "current_week_status_proven":False,
        "routes_verified":False,
        "skipped_games":fail0[:30],
        "skipped_count":len(fail0),
        "production_approved":False,
    }
    return result,paired,pair_wr,joined


def run_comparison(weekly,schedules,snaps,*,n_sims=180,games_per_week=2):
    train=(2025,(9,10,11,12))
    test=(2026,(4,5,6,7,8,9,10))
    btrain=run_forecast(weekly,schedules,snaps,*train,n_sims,
                        BASELINE_WEIGHT,games_per_week)
    results={};frames={}
    for weight in CANDIDATE_WEIGHTS:
        candidate=run_forecast(weekly,schedules,snaps,*train,n_sims,
                               weight,games_per_week)
        report,paired,wr,coverage=evaluate_weight(
            btrain,candidate,2025,weight)
        results[str(weight)]=report
        frames[f"2025_snap_{weight}_paired_wr"]=wr
        print("2025",weight,"WR1/2 MAE lift",
              report["WR_targets"]["WR1_WR2_COMBINED"]["mae_improvement_pct"],
              "DK MAE lift",report["dk"]["matchup_mae_improvement_pct"],
              flush=True)
    # Training-only release-safe decision: no actuation if role target gains
    # do not exceed 1%, are driven by one role regressing >2%, or if DK
    # accuracy worsens >0.5%. This threshold is specified before 2026 grade.
    qualified=[]
    for w in CANDIDATE_WEIGHTS:
        x=results[str(w)]
        target=x["WR_targets"]
        dk=x["dk"]["matchup_mae_improvement_pct"]
        if (target["WR1_WR2_COMBINED"]["mae_improvement_pct"]>=1.
            and dk is not None and dk>=-.5
            and all(target[k]["candidate_mae"]<=1.02*target[k]["baseline_mae"]
                    for k in ("WR1","WR2"))):
            qualified.append(w)
    selected=(max(qualified,key=lambda w:(
        results[str(w)]["WR_targets"]["WR1_WR2_COMBINED"]["mae_improvement_pct"],
        -w)) if qualified else BASELINE_WEIGHT)
    btest=run_forecast(weekly,schedules,snaps,*test,n_sims,
                       BASELINE_WEIGHT,games_per_week)
    if selected==BASELINE_WEIGHT:
        candidate=btest
    else:
        candidate=run_forecast(weekly,schedules,snaps,*test,n_sims,
                               selected,games_per_week)
    testmetrics,paired,wr,coverage=evaluate_weight(btest,candidate,2026,selected)
    frames["2026_paired_dk"]=paired
    frames["2026_paired_wr"]=wr
    frames["2026_pregame_snap_coverage"]=coverage
    return {
        "study":"Observed offensive participation role-weight research",
        "historical_snap_source":"nflverse PFR prior-game offensive snaps",
        "routes_and_live_active_unverified":True,
        "train":"2025 W9-12 already researched",
        "test":"2026 W4-10 previously researched, NOT blind",
        "baseline_snap_weight":BASELINE_WEIGHT,
        "training_2025":results,
        "selected_weight_from_2025_only":selected,
        "selected_nondefault":selected!=BASELINE_WEIGHT,
        "retrospective_2026":testmetrics,
        "production_approved":False,
        "forward_holdout_unused":True,
        "n_sims":n_sims,
        "games_per_week":games_per_week,
    },frames


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sims",type=int,default=180)
    p.add_argument("--games-per-week",type=int,default=2)
    p.add_argument("--out-dir",default="nfl_snap_participation_ab")
    a=p.parse_args(argv)
    if not 100<=a.n_sims<=10000 or a.games_per_week<0:
        p.error("Invalid study simulation count/game limit")
    history=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report,frames=run_comparison(
        history,schedules,snaps,n_sims=a.n_sims,
        games_per_week=a.games_per_week or None)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    for name,df in frames.items():
        df.to_csv(out/(name+".csv"),index=False)
    (out/"snap_participation_results.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    x=report["retrospective_2026"]
    print(json.dumps({
        "selected_weight_from_2025_only":report["selected_weight_from_2025_only"],
        "nondefault_selected":report["selected_nondefault"],
        "2026_matched":x["dk"]["n"],
        "2026_wr_snap_coverage_pct":x["WR_snap_evidence_coverage_pct"],
        "2026_WR1":x["WR_targets"]["WR1"],
        "2026_WR2":x["WR_targets"]["WR2"],
        "2026_DK":x["dk"],
        "production_approved":False,"retrospective_not_blind":True,
    },indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
