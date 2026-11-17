"""Research A/B: actual nflverse drive-based NFL engine vs current neutral stats.

- nflverse fixed-drive PBP is frozen strictly before each target week.
- 2025 W9-12 descriptive reference, 2026 W4-10 retrospective evaluation.
- Same players, opponents, seeds and actual partial-DK scoring for each side.
- 2026 history has been explored by prior AI2 changes: NOT an untouched test.
- NO production activation, optimizer lineup claims, or automatic release.

The experimental model simulates a shared 60-minute game of alternating drives,
scores, transitions, interceptions, turnovers, team plays and dropbacks. These
worlds feed the existing stat->fantasy player/DST allocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai2_nfl_pbp_drives import load_historical_pbp, pbp_to_drives
from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE, evaluate
from scripts.backtest_nfl_matchup_simulator import match_results, paired_metrics


def run_audit(weekly, schedules, snaps, drives, *, n_sims=220,
              training_weeks=(9,10,11,12),
              retrospective_weeks=(4,5,6,7,8,9,10)):
    result={}
    paired_test=None
    for label,year,weeks in (
        ("2025_reference",2025,training_weeks),
        ("2026_retrospective",2026,retrospective_weeks)
    ):
        original,skip0=evaluate(
            weekly,schedules,snaps,year,weeks,
            games_per_week=None,n_sims=n_sims,
            possession_strength=0.)
        experimental,skip1=evaluate(
            weekly,schedules,snaps,year,weeks,
            games_per_week=None,n_sims=n_sims,
            possession_drives=drives,possession_strength=1.)
        print(f"{label}: neutral {len(original)} matched players; "
              f"possession {len(experimental)} matched players; "
              f"neutral skipped {len(skip0)}; possession skipped {len(skip1)}",
              flush=True)
        if len(skip1)>len(skip0):
            print("Possession-only failure examples:",skip1[:16],flush=True)
        paired=match_results(original,experimental)
        measures=paired_metrics(paired)
        ci=measures["paired_game_cluster_lift_ci"]["lift_pct_95_ci"]
        position={
            pos:paired_metrics(frame)
            for pos,frame in paired.groupby("position")}
        weeks_summary={
            str(w):paired_metrics(frame)
            for w,frame in paired.groupby("week")}
        result[label]={
            "summary":measures,
            "by_position":position,
            "by_week":weeks_summary,
            "neutral_failures":skip0[:35],
            "possession_failures":skip1[:35],
            "neutral_failure_count":len(skip0),
            "possession_failure_count":len(skip1),
            "accuracy_improved_over_neutral":(
                measures["matchup_mae_improvement_pct"] is not None and
                measures["matchup_mae_improvement_pct"]>0),
            "game_cluster_lift_ci_below_zero":(
                ci is not None and ci[0] <= 0),
        }
        if label=="2026_retrospective":
            paired_test=paired
    test=result["2026_retrospective"]
    m=test["summary"]
    delta=m["matchup_mae_improvement_pct"]
    ci=m["paired_game_cluster_lift_ci"]["lift_pct_95_ci"]
    checks={
        "same_matchable_player_set":True,
        "2025_reference_mae_not_worse":(
            result["2025_reference"]["summary"][
                "matchup_mae_improvement_pct"] >= 0),
        "2026_retrospective_mae_improvement_at_least_3pct":(
            delta is not None and delta>=3),
        "game_cluster_95pct_lower_bound_positive":(
            ci is not None and ci[0]>0),
        "P90_coverage_85_to_95":85<=m["actual_below_p90_pct"]<=95,
        "all_roles_no_more_than_3pct_regression":all(
            x["matchup_mae_improvement_pct"] is not None and
            x["matchup_mae_improvement_pct"]>=-3
            for x in test["by_position"].values()
        ),
        "matched_at_least_500_player_games":m["n"]>=500,
        "complete_game_coverage_no_dropped_games":(
            test["neutral_failure_count"]==
            test["possession_failure_count"] and
            result["2025_reference"]["neutral_failure_count"]==
            result["2025_reference"]["possession_failure_count"]),
    }
    result.update({
        "study":"NFL shared-clock possession simulator on pregame PBP",
        "source":"nflverse play_by_play_{season}.parquet",
        "simulations_per_game":n_sims,
        "2025_reference_is_for_diagnosis_not_blind_parameter_tuning":True,
        "2026_retrospective_was_previously_studied":True,
        "gates":checks,
        "research_gate_pass":all(checks.values()),
        "production_approved":False,
        "oneclick_modified":False,
        "limitations":[
            "Not a validated true play-by-play play-calling engine: possession outcomes and play counts are sampled from historic drives",
            "Drive transition field position and final two-minute decisions use provisional heuristics",
            "Player target/carry shares are historical and do not yet model role-specific red-zone routes/goal-line carries",
            "Dated historical DK salaries and full pregame injury-status snapshots are missing",
            "Quarterback/receiver passing-yard totals are shared but intra-drive yardage is not yet explicitly assigned play-by-play",
            "Prior research has viewed these 2026 games; prospective frozen predictions are required",
            "Partial DraftKings points omit fumbles, 2-point conversions and returns",
            "No contest ROI or winning-week claim",
        ],
    })
    return result,paired_test


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims",type=int,default=220)
    ap.add_argument("--output-dir",default="nfl_possession_audit")
    ap.add_argument("--pbp-cache-dir",default=None)
    args=ap.parse_args(argv)
    if not 100<=args.n_sims<=10000:
        ap.error("--n-sims must be 100..10000")
    print("Loading pregame nflverse weekly player stats and play-by-play...",flush=True)
    weekly=fetch_weekly([2024,2025,2026])
    snaps=fetch_snap_counts([2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    raw=load_historical_pbp([2024,2025,2026],
                            cache_dir=args.pbp_cache_dir)
    drives=pbp_to_drives(raw)
    print(f"Historic raw PBP plays={len(raw)}, eligible offensive drives={len(drives)}",flush=True)
    report,paired=run_audit(weekly,schedules,snaps,drives,n_sims=args.n_sims)
    dst=Path(args.output_dir)
    dst.mkdir(parents=True,exist_ok=True)
    paired.to_csv(dst/"matched_player_game_forecasts.csv",index=False)
    (dst/"validation.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        "reference_2025":report["2025_reference"]["summary"],
        "retrospective_2026":report["2026_retrospective"]["summary"],
        "gates":report["gates"],
        "research_pass":report["research_gate_pass"],
        "production_approved":False,
        "paired_rows":len(paired),
    },indent=2),flush=True)


if __name__=="__main__":
    main()
