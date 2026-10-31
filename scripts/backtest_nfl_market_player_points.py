"""Actual player DK holdout for market-derived game touchdown priors.

The training selection uses ONLY 2025 completed Sunday games; evaluated
independently on 2026 weeks 4-8. A market prior is NOT enabled in live NFL
until the player forecast and lineup-level impact are tested.
"""
from __future__ import annotations
import json
import pandas as pd
from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate,summary

CANDIDATES=(0.,.5,1.)


def compare(history,schedules,snap,*,candidates=CANDIDATES,games_per_week=2,n_sims=320):
    outcomes={}
    for weight in candidates:
        train,failed_train=evaluate(history,schedules,snap,2025,[10,11,12],
                                    games_per_week=games_per_week,n_sims=n_sims,
                                    market_weight=weight)
        test,failed_test=evaluate(history,schedules,snap,2026,[4,5,6,7,8],
                                  games_per_week=games_per_week,n_sims=n_sims,
                                  market_weight=weight)
        if train.empty or test.empty:
            raise ValueError(f"Missing role-scored market holdout weight {weight}: {failed_train[:3]}, {failed_test[:3]}")
        outcomes[str(weight)]={
            "weight":weight,"train":summary(train),"holdout":summary(test),
            "holdout_by_position":{p:summary(group) for p,group in test.groupby("position")},
            "skipped_train":failed_train[:4],"skipped_holdout":failed_test[:4],
        }
    chosen=min(outcomes.values(),key=lambda d:(d["train"]["sim_mean_mae"],d["weight"]))
    base=outcomes["0.0"]["holdout"]
    selected=chosen["holdout"]
    return {
        "train_period":"2025 weeks 10-12",
        "independent_holdout_period":"2026 weeks 4-8",
        "training_selected_weight":chosen["weight"],
        "outcomes":outcomes,
        "selected_holdout_mean_improved":selected["sim_mean_mae"]<base["sim_mean_mae"],
        "selected_holdout_p90_coverage_nearer_90":abs(selected["simulated_p90_actual_coverage_pct"]-90)<abs(base["simulated_p90_actual_coverage_pct"]-90),
        "production_market_adjustment":False,
        "caveats":"Partial DK player stats excluding special teams/fumbles; prior-game eligible and observed in actual contest; odds are closing totals not frozen early-week quotes, no cash ROI."
    }


def run():
    history=fetch_weekly([2024,2025,2026])
    schedule=pd.read_csv(GAMES_SOURCE)
    snap=fetch_snap_counts([2025,2026])
    return compare(history,schedule,snap)


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
