"""Paired historical PBP player-red-zone allocation vs unchanged NFL scoring.

Every candidate is scored against the exact same player-game set and actual
partial DK points as an opponent-neutral baseline. This tests player-specific
red-zone TARGET and goal-line CARRY shares, not fantasy-point additives.

Chronological selection from 2025 Weeks 9-12, 2026 Weeks 4-10 retrospective
diagnostics only. 2026 results were used in earlier AI2 development, so NOT an
independent unseen test. No production or lineup changes are made.

The red-zone factors only change the recipient of a passing/rushing TD, never
the total team touchdowns, passing attempts, targets, catches or carries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ai2_nfl_pbp_drives import PBP_RELEASE
from ai2_nfl_redzone_usage import FIELDS, pregame_redzone_profiles
from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE, evaluate
from scripts.backtest_nfl_matchup_simulator import match_results, paired_metrics

STRENGTHS=(0.,.5,1.)


def paired_report(neutral, candidate):
    pair=match_results(neutral,candidate)
    return {
        "overall":paired_metrics(pair),
        "by_position":{
            pos:paired_metrics(group)
            for pos,group in pair.groupby("position")},
        "by_week":{
            str(w):paired_metrics(group)
            for w,group in pair.groupby("week")},
        "by_opponent":{
            opp:paired_metrics(group)
            for opp,group in pair.groupby("opponent_neutral")},
    },pair


def run_audit(weekly, schedules, snaps, pbp, *, n_sims=300,
              training_weeks=(9,10,11,12),
              retrospective_weeks=(4,5,6,7,8,9,10)):
    def eval_period(year,weeks,weight):
        return evaluate(weekly,schedules,snaps,year,weeks,
                        games_per_week=None,n_sims=n_sims,
                        redzone_pbp=pbp,redzone_strength=weight)
    train_base,err0=eval_period(2025,training_weeks,0.)
    if train_base.empty:
        raise ValueError("No matched 2025 pregame forecast games")
    train={}
    for w in STRENGTHS:
        if w==0:
            candidates,err=train_base,err0
        else:
            candidates,err=eval_period(2025,training_weeks,w)
        result,_=paired_report(train_base,candidates)
        train[str(w)]={
            "summary":result["overall"],
            "by_position":result["by_position"],
            "skipped_games":err[:20],
            "skipped_count":len(err),
        }
    # Select a nonzero coefficient ONLY when the 2025 training period shows
    # >=1% MAE improvement AND the game-bootstrap lower 95% bound is positive.
    qualifies=[]
    for weight in STRENGTHS[1:]:
        scores=train[str(weight)]["summary"]
        ci=scores["paired_game_cluster_lift_ci"]["lift_pct_95_ci"]
        if (scores["matchup_mae_improvement_pct"]>=1.0 and
                ci is not None and ci[0]>0 and
                train[str(weight)]["skipped_count"]==len(err0)):
            qualifies.append(weight)
    selected=min(qualifies,key=lambda v:(
        train[str(v)]["summary"]["matchup_simulator_mae"],v)
    ) if qualifies else 0.

    base,err_base=eval_period(2026,retrospective_weeks,0.)
    selected_rows,err_selected=(
        eval_period(2026,retrospective_weeks,selected)
        if selected else (base,err_base)
    )
    compared,rows=paired_report(base,selected_rows)
    # Always show one fixed positive-strength diagnostic, even if it didn't
    # earn selection from the earlier 2025 period. Never use it for release.
    probe=.5
    probe_rows,err_probe=(
        (selected_rows,err_selected) if selected==probe else
        eval_period(2026,retrospective_weeks,probe)
    )
    probe_scores,probe_comparison=paired_report(base,probe_rows)
    test=compared["overall"]
    ci=test["paired_game_cluster_lift_ci"]["lift_pct_95_ci"]
    checks={
        "2025_selected_positive_role_strength":selected>0,
        "no_missing_matched_player_games":
            err_selected==err_base and err_probe==err_base,
        "2026_mae_improvement_ge_3pct":(
            test["matchup_mae_improvement_pct"] is not None and
            test["matchup_mae_improvement_pct"]>=3),
        "game_cluster_95pct_lower_lift_above_zero":
            ci is not None and ci[0]>0,
        "2026_p90_coverage_85_to_95pct":
            85<=test["actual_below_p90_pct"]<=95,
        "no_position_mae_regression_over_3pct":all(
            x["matchup_mae_improvement_pct"] is not None and
            x["matchup_mae_improvement_pct"]>=-3
            for x in compared["by_position"].values()),
        "at_least_500_matched_player_games":test["n"]>=500,
    }
    return {
        "name":"NFL GSIS player-specific red-zone/goal-line TD allocation A/B",
        "training":"2025 weeks 9-12; only data before each week",
        "retrospective":"2026 weeks 4-10; previously inspected (not untouched)",
        "n_sims_per_game":n_sims,
        "training_results":train,
        "selected_from_2025":selected,
        "retrospective_selected":compared,
        "retrospective_fixed_0_5_experiment":{
            **probe_scores,
            "not_eligible_for_release":True,
        },
        "diagnostic_failures":{
            "training_neutral":err0[:20],
            "2026_neutral":err_base[:20],
            "2026_selected":err_selected[:20],
            "2026_fixed_probe":err_probe[:20]},
        "gates":checks,
        "research_pass":all(checks.values()),
        "production_approved":False,
        "limitations":[
            "Player pool reconstructed from prior-week rates, not dated DK salaries",
            "Partial DK fantasy scores exclude some events",
            "No historical pre-kickoff inactive roster snapshots; matched players only",
            "TD allocation does not alter team TD budgets or yardage volume",
            "Red-zone roles are last-eight-team-game observed GSIS opportunities, not certified depth-chart routes",
            "Previously studied 2026 period cannot be called new holdout",
            "Cannot establish contest profitability or justify larger entries",
        ],
    },rows,probe_comparison


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sims",type=int,default=300)
    p.add_argument("--out-dir",default="nfl_redzone_audit")
    args=p.parse_args(argv)
    if not 100<=args.n_sims<=10000:
        p.error("--n-sims must be between 100 and 10000")
    print("Fetching pregame play-by-play GSIS usage data",flush=True)
    weekly=fetch_weekly([2024,2025,2026])
    pbp=pd.concat([
        pd.read_parquet(PBP_RELEASE.format(season=yr),columns=sorted(FIELDS))
        for yr in (2024,2025,2026)],ignore_index=True)
    profiles=pregame_redzone_profiles(pbp,weekly,2026,9)
    count=len(profiles["players"])
    receivers=sum(k[2]=="receiving" for k in profiles["players"])
    rushers=sum(k[2]=="rushing" for k in profiles["players"])
    print(f"Real GSIS matched player-role records: {count},"
          f" receivers: {receivers}, runners: {rushers}",flush=True)
    print("Representative pregame role factors:",[
        (key,info["factor"]) for key,info in list(profiles["players"].items())[:12]
    ],flush=True)
    if count<50 or receivers<20 or rushers<20:
        raise ValueError("Real nflverse GSIS matching insufficient for player red-zone experiment")
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report,selected,probe=run_audit(
        weekly,schedules,snaps,pbp,n_sims=args.n_sims)
    output=Path(args.out_dir)
    output.mkdir(parents=True,exist_ok=True)
    selected.to_csv(output/"selected_model_paired_players.csv",index=False)
    probe.to_csv(output/"fixed_half_strength_paired_players.csv",index=False)
    (output/"results.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        "training_selected_strength":report["selected_from_2025"],
        "2026_selected":report["retrospective_selected"]["overall"],
        "2026_fixed_half_strength":report[
            "retrospective_fixed_0_5_experiment"]["overall"],
        "gates":report["gates"],
        "research_pass":report["research_pass"],
        "production_approved":False,
        "artifacts":str(output)
    },indent=2),flush=True)


if __name__=="__main__":
    main()
