"""Check if using *observed* receiving touchdown per catch improves fantasy means.

This changes who receives a team passing TD, not team touchdown count.
Choose amount from 2025 training games, then report independent 2026 games.
A positive result is still not a contest-ROI guarantee.
"""
from __future__ import annotations
import json
import pandas as pd
from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate,summary

WEIGHTS=(0.,.5,1.)


def run():
    history=fetch_weekly([2024,2025,2026])
    schedule=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    results={}
    for strength in WEIGHTS:
        train,train_fail=evaluate(history,schedule,snaps,2025,[10,11,12],
                                  receiving_td_weight=strength)
        test,test_fail=evaluate(history,schedule,snaps,2026,[4,5,6,7,8],
                                receiving_td_weight=strength)
        if train.empty or test.empty:
            raise ValueError(f"Insufficient pregame TD scoring holdout at weight {strength}: {train_fail[:2]}, {test_fail[:2]}")
        results[str(strength)]={
            "train":summary(train),"holdout":summary(test),
            "holdout_by_position":{p:summary(g) for p,g in test.groupby("position")},
            "train_issues":train_fail[:3],"heldout_issues":test_fail[:3]}
    choice=min(WEIGHTS,key=lambda strength:(
        results[str(strength)]["train"]["sim_mean_mae"],strength))
    a=results["0.0"]["holdout"]
    b=results[str(choice)]["holdout"]
    return {
        "chosen_weight_on_2025_only":choice,
        "training":"2025 weeks 10-12",
        "holdout":"2026 weeks 4-8",
        "variants":results,
        "holdout_mean_mae_improved":b["sim_mean_mae"]<a["sim_mean_mae"],
        "holdout_p90_coverage_closer_90":abs(b["simulated_p90_actual_coverage_pct"]-90)<
                                          abs(a["simulated_p90_actual_coverage_pct"]-90),
        "status":"EXPERIMENTAL, NOT enabled in production",
        "caveat":"Player partial DK, pregame active roles inferred only from prior-week data, no contest ROI."
    }


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
