"""Paired test of opponent-conditioned team TD/plays/game script versus neutral.

Only 2025 Weeks 9-12 choose the coefficient. Score 2026 Weeks 4-10 once
against the unchanged seeded production-stat model, and disclose an additional
experimental sensitivity at strength 0.5 even if training selects neutral.
2026 has ALREADY been used in earlier AI2 work: this is a retrospective
research diagnostic, not an untouched test or a paid DFS release.

Outputs JSON + rows of paired forecasts. The release gate FAILS rather than
auto-enabling a weak or undercalibrated adjustment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE, evaluate
from scripts.backtest_nfl_matchup_simulator import match_results, paired_metrics


STRENGTHS = (0., .25, .5, 1.)


def run_comparison(history, schedules, snaps, *,
                   train_weeks=(9,10,11,12),
                   test_weeks=(4,5,6,7,8,9,10),
                   n_sims=400):
    def evaluate_period(season, weeks, strength):
        return evaluate(history, schedules, snaps, season, weeks,
                        games_per_week=None, n_sims=n_sims,
                        game_matchup_strength=strength)

    train_neutral, training_errors=evaluate_period(2025, train_weeks, 0.)
    if train_neutral.empty:
        raise ValueError("Training games were not simulated: " + str(training_errors[:5]))
    training={}
    for w in STRENGTHS:
        if w==0:
            candidates,issues=train_neutral,training_errors
        else:
            candidates,issues=evaluate_period(2025,train_weeks,w)
        scores=paired_metrics(match_results(train_neutral,candidates))
        training[str(w)]={"metrics": scores,"skipped_games":issues[:12]}
    baseline_mae=training["0.0"]["metrics"]["baseline_simulator_mae"]
    eligible=[]
    for w in STRENGTHS[1:]:
        m=training[str(w)]["metrics"]
        if (m["matchup_mae_improvement_pct"] is not None and
                m["matchup_mae_improvement_pct"]>=1. and
                m["paired_game_cluster_lift_ci"]["lift_pct_95_ci"] is not None and
                m["paired_game_cluster_lift_ci"]["lift_pct_95_ci"][0]>0):
            eligible.append(w)
    selected=min(eligible,key=lambda w: (
        training[str(w)]["metrics"]["matchup_simulator_mae"],w)
    ) if eligible else 0.

    old,skipped0=evaluate_period(2026,test_weeks,0.)
    selected_rows,skipped1=(
        evaluate_period(2026,test_weeks,selected)
        if selected else (old,skipped0))
    pair=match_results(old,selected_rows)
    scored=paired_metrics(pair)
    by_position={
        p:paired_metrics(g) for p,g in pair.groupby("position")
    }
    by_week={
        str(w):paired_metrics(g) for w,g in pair.groupby("week")
    }
    ci=scored["paired_game_cluster_lift_ci"]["lift_pct_95_ci"]
    criteria={
        "training_selected_nonzero_strength": selected>0,
        "2026_retrospective_MAE_improves": (
            scored["matchup_mae_improvement_pct"] is not None and
            scored["matchup_mae_improvement_pct"]>0),
        "game_cluster_95pct_lower_lift_positive": (
            ci is not None and ci[0]>0),
        "P90_coverage_85_to_95pct":(
            85<=scored["actual_below_p90_pct"]<=95),
        "each_position_no_greater_than_3pct_regression": all(
            v["matchup_mae_improvement_pct"] is not None and
            v["matchup_mae_improvement_pct"]>=-3
            for v in by_position.values()),
    }
    # Always score a positive experimental setting as an additional diagnostic.
    # It can never override a training-based selection or unlock production.
    probe=.5
    probed,probe_errors=(
        selected_rows,skipped1 if selected==probe else skipped0
    ) if selected==probe else evaluate_period(2026,test_weeks,probe)
    experimental_probe=paired_metrics(match_results(old,probed))
    out={
        "name":"NFL opponent team scoring/pace/game-script exploratory A/B",
        "2025_training":training,
        "2025_selected_strength":selected,
        "2026_reference":"Weeks 4-10, previously seen research weeks, not untouched",
        "2026_selected_model":scored,
        "2026_by_position":by_position,
        "2026_by_week":by_week,
        "2026_strength_0_5_probe_regardless_of_training_selection":{
            "metrics":experimental_probe,"skipped_games":probe_errors[:12],
            "not_eligible_for_selection":True,
        },
        "checks":criteria,
        "research_diagnostic_pass":all(criteria.values()),
        "production_approved":False,
        "no_automatic_scoring_changes":True,
        "skipped_neutral":skipped0[:40],
        "skipped_selected":skipped1[:40],
        "limitations":[
            "2026 results were studied in earlier model work; not fresh holdout",
            "Only historically active matched players; missing pregame inactive snapshots",
            "Partial DraftKings scoring, no replayed live DK salaries or contests",
            "Team plays are box-score proxies, not calibrated drive-by-drive possessions",
            "2025 four-week training may be insufficient for a reliable effect",
            "Pace and TD opponent profiles do not model defensive player injuries",
        ]
    }
    return out,pair


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",default="nfl_game_matchup_audit")
    p.add_argument("--n-sims",type=int,default=400)
    args=p.parse_args(argv)
    if not 100<=args.n_sims<=10000:
        p.error("n-sims must be between 100 and 10000")
    hist=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report,paired=run_comparison(hist,schedules,snaps,n_sims=args.n_sims)
    dest=Path(args.out_dir)
    dest.mkdir(parents=True,exist_ok=True)
    paired.to_csv(dest/"paired_player_forecasts.csv",index=False)
    (dest/"results.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        "train_selected_strength":report["2025_selected_strength"],
        "2026_selected":report["2026_selected_model"],
        "2026_experimental_half_strength":report[
            "2026_strength_0_5_probe_regardless_of_training_selection"
        ]["metrics"],
        "research_diagnostic_pass":report["research_diagnostic_pass"],
        "production_approved":False,
        "artifacts":str(dest)
    },indent=2))


if __name__=="__main__":
    main()
