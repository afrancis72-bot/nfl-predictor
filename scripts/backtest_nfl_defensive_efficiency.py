"""Pregame defense allowed vs actual NFL player scores after target-week game.

Defense input only prior opponent_team games and position partial-DK
rates; one conservative yardage multiplier, independent team touches.
Tune using 2025 training games, evaluate on independent 2026 games.
"""
import json
import pandas as pd
from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate,summary

WEIGHTS=(0.,.5,1.)


def compare(history,schedules,snaps,*,weights=WEIGHTS,games_per_week=2,n_sims=260):
    result={}
    for strength in weights:
        train,skip_train=evaluate(
            history,schedules,snaps,2025,[9,10,11,12],
            games_per_week=games_per_week,n_sims=n_sims,defense_weight=strength)
        test,skip_test=evaluate(
            history,schedules,snaps,2026,[4,5,6,7,8],
            games_per_week=games_per_week,n_sims=n_sims,defense_weight=strength)
        if train.empty or test.empty:
            raise ValueError(f"Insufficient defender history for weight {strength}: {skip_train[:4]}; {skip_test[:4]}")
        result[str(strength)]={
            "training":summary(train),
            "holdout":summary(test),
            "by_position":{p:summary(g) for p,g in test.groupby("position")},
            "skipped_train":skip_train[:4],
            "skipped_holdout":skip_test[:4],
        }
    selected=min(weights,key=lambda z:(result[str(z)]["training"]["sim_mean_mae"],z))
    baseline=result["0.0"]["holdout"]
    tested=result[str(selected)]["holdout"]
    return {
        "training_period":"2025 weeks 9-12",
        "heldout_period":"2026 weeks 4-8; heldout from weight selection",
        "training_selected_strength":selected,
        "variants":result,
        "holdout_player_mean_error_improved":tested["sim_mean_mae"]<baseline["sim_mean_mae"],
        "holdout_player_P90_coverage_nearer_90":abs(tested["simulated_p90_actual_coverage_pct"]-90)<
                                                   abs(baseline["simulated_p90_actual_coverage_pct"]-90),
        "note":"A research test, not confirmed profitable lineup scoring; source was only pregame nflverse opponent_team histories. Player appears in actual game; no official historical inactive snapshots."
    }


def run():
    history=fetch_weekly([2024,2025,2026])
    schedule=pd.read_csv(GAMES_SOURCE)
    snap=fetch_snap_counts([2025,2026])
    return compare(history,schedule,snap)


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
