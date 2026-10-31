"""Try a football-consistent, scenario-level solution to underdispersed NFL P90.

Role-share uncertainty uses a Dirichlet-multinomial draw per team per game,
conserving passes/carries and preserving the fantasy-stat accounting. Select
the concentration on 2025 training ONLY; report 2026 independent holdout.
Never put adjustments into user one-click until the heldout is defensible.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE, evaluate, summary

TRAIN_WEEKS=[10,11,12]
TEST_WEEKS=[4,5,6,7,8]
CONCENTRATIONS=[None, 30.0, 75.0, 150.0]


def p90_pinball(rows):
    actual=rows["actual_partial_dk"].to_numpy(float)
    quantile=rows["simulated_p90_dk"].to_numpy(float)
    delta=actual-quantile
    return float(np.mean(np.maximum(.9*delta,-.1*delta)))


def quality(rows):
    if rows.empty:
        return {"matched":0,"p90_pinball":None}
    result=summary(rows)
    result["p90_pinball"]=round(p90_pinball(rows),4)
    result["over_p90_count"]=int((rows["actual_partial_dk"] > rows["simulated_p90_dk"]).sum())
    return result


def compare(historical,schedules,snaps,*,concentrations=CONCENTRATIONS,
            games_per_week=2,n_sims=320):
    results={}
    for concentration in concentrations:
        label="production_fixed_share" if concentration is None else f"role_concentration_{concentration:g}"
        train,train_failed=evaluate(historical,schedules,snaps,2025,TRAIN_WEEKS,
                                    games_per_week=games_per_week,n_sims=n_sims,
                                    role_share_concentration=concentration)
        test,test_failed=evaluate(historical,schedules,snaps,2026,TEST_WEEKS,
                                  games_per_week=games_per_week,n_sims=n_sims,
                                  role_share_concentration=concentration)
        if train.empty or test.empty:
            raise ValueError(f"Missing actual-player holdout for {label}: {train_failed[:2]}, {test_failed[:2]}")
        results[label]={
            "concentration":concentration,
            "training":quality(train),
            "holdout":quality(test),
            "by_position":{p:quality(g) for p,g in test.groupby("position")},
            "train_issues":train_failed[:3],
            "holdout_issues":test_failed[:3],
        }
    choice=min(results, key=lambda k:(
        results[k]["training"]["p90_pinball"],
        0 if k=="production_fixed_share" else 1))
    base=results["production_fixed_share"]["holdout"]
    picked=results[choice]["holdout"]
    return {
        "training":"2025 weeks 10-12, 2 Sunday games/week; independent role-share selection",
        "heldout":"2026 weeks 4-8, 2 Sunday games/week; not involved in selection",
        "variants":results,
        "train_selected_variant":choice,
        "heldout_p90_pinball_improved":picked["p90_pinball"] < base["p90_pinball"],
        "heldout_p90_coverage_closer_to_90":abs(picked["simulated_p90_actual_coverage_pct"]-90)<
                                             abs(base["simulated_p90_actual_coverage_pct"]-90),
        "release_recommendation":"BENCHMARK ONLY: do not activate on small/uncalibrated sample",
        "caveats":"Partial DK points, target-week inactives not frozen, players who actually appeared only; not a test of full-field contest ROI.",
    }


def run():
    history=fetch_weekly([2024,2025,2026])
    schedule=pd.read_csv(GAMES_SOURCE)
    snap=fetch_snap_counts([2025,2026])
    return compare(history,schedule,snap)


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
