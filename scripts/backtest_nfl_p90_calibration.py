"""Expanded rolling-period NFL P90 calibration with two untouched time splits.

2025 W5-14: fit on only completed pregame predictions.
2025 W15-18: validate global vs position-specific correction.
2026 W4-8: strictly held-out independent confirmation.
Two to four full Sunday games per week; cannot cherry-pick known winners.
The extra P90 shift changes *quantile reporting only*, never modifies the
game-level scenarios, projected means, lineup scoring, or approved entries.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate,summary
from ai2_nfl_p90_calibration import fit_p90_margins,measure

FIT_WEEKS=list(range(5,15))
VALIDATION_WEEKS=list(range(15,19))
TEST_WEEKS=list(range(4,9))


def run_backtest(history,schedules,snaps,*,games_per_week=3,n_sims=220):
    observations={}
    problems={}
    for split,season,weeks in (
        ("2025_fit",2025,FIT_WEEKS),
        ("2025_validation",2025,VALIDATION_WEEKS),
        ("2026_holdout",2026,TEST_WEEKS)
    ):
        df,issues=evaluate(history,schedules,snaps,season,weeks,
                           games_per_week=games_per_week,n_sims=n_sims)
        if df.empty:
            raise ValueError(f"Historical {split} evaluation failed: {issues[:3]}")
        observations[split]=df
        problems[split]=issues
    training=observations["2025_fit"]
    val=observations["2025_validation"]
    test=observations["2026_holdout"]
    calibration=fit_p90_margins(training)
    v={
        mode: measure(val,calibration if mode!="uncalibrated" else None,
                      mode=mode if mode!="uncalibrated" else "pooled")
        for mode in ("uncalibrated","pooled","role")
    }
    # Optimize P90 pinball on 2025 validation ONLY, prefer no correction
    # when a seemingly improved version has worse interval coverage.
    baseline=v["uncalibrated"]
    choices=["uncalibrated","pooled","role"]
    valid=[mode for mode in choices if
           (mode=="uncalibrated" or
            (v[mode]["p90_pinball"]<=baseline["p90_pinball"] and
             abs(v[mode]["actual_below_reported_p90_pct"]-90)<
             abs(baseline["actual_below_reported_p90_pct"]-90)))]
    chosen=min(valid,key=lambda mode:(v[mode]["p90_pinball"],
                                       0 if mode=="uncalibrated" else 1))
    h={
        mode: measure(test,calibration if mode!="uncalibrated" else None,
                      mode=mode if mode!="uncalibrated" else "pooled")
        for mode in ("uncalibrated","pooled","role")
    }
    selected=h[chosen]
    base=h["uncalibrated"]
    return {
        "periods":{
            "fitting":"2025 weeks 5-14", "validation":"2025 weeks 15-18",
            "independent_holdout":"2026 weeks 4-8",
            "games_per_week":games_per_week,
            "simulations_per_game":n_sims},
        "total_matched_player_games":sum(len(x) for x in observations.values()),
        "fit_stats":summary(training),
        "validation_stats":summary(val),
        "holdout_stats":summary(test),
        "estimated_from_2025_only":calibration,
        "2025_validation":v,
        "2026_holdout":h,
        "selected_without_2026_evidence":chosen,
        "holdout_p90_error_improved":selected["p90_pinball"]<base["p90_pinball"],
        "holdout_coverage_closer_90":abs(selected["actual_below_reported_p90_pct"]-90)<
                                     abs(base["actual_below_reported_p90_pct"]-90),
        "skipped_games":{k:{"count":len(v),"examples":v[:4]} for k,v in problems.items()},
        "release_caveat":"Do not inject a corrected P90 into football-stat worlds. Player appearance selection and missing prior-day inactive records bias coverage. Small samples mean confirmation is required."
    }


def run():
    history=fetch_weekly([2024,2025,2026])
    schedule=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    return run_backtest(history,schedule,snaps)


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
