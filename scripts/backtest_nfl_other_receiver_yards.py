"""Determine whether missing teammates' QB yards are incorrectly given to stars.

Tests only actual game statistics available after kickoff; forecasts are
produced only from prior player games/snap roles. TRAIN: 2025; HOLDOUT: 2026.
Do not enable on a prettier mean or one individual's favorable outcome.
"""
from __future__ import annotations
import json
import pandas as pd
from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate,summary

STRENGTHS=(0.,.5,1.)


def run():
    history=fetch_weekly([2024,2025,2026])
    schedule=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    stats={}
    for strength in STRENGTHS:
        train,train_errors=evaluate(history,schedule,snaps,2025,[9,10,11,12],
                                    games_per_week=2,n_sims=260,
                                    other_yard_strength=strength)
        heldout,heldout_errors=evaluate(history,schedule,snaps,2026,[4,5,6,7,8],
                                        games_per_week=2,n_sims=260,
                                        other_yard_strength=strength)
        if train.empty or heldout.empty:
            raise ValueError(f"Missing complete games for OTHER-receiving yards study: {train_errors[:2]} | {heldout_errors[:2]}")
        stats[str(strength)]={
            "training":summary(train),
            "heldout":summary(heldout),
            "heldout_by_position":{
                key:summary(rows) for key,rows in heldout.groupby("position")},
            "train_failures":train_errors[:3],"holdout_failures":heldout_errors[:3],
        }
    selected=min(STRENGTHS,key=lambda z:(stats[str(z)]["training"]["sim_mean_mae"],z))
    baseline=stats["0.0"]["heldout"]
    chosen=stats[str(selected)]["heldout"]
    return {
        "training_period":"2025 weeks 9-12",
        "independent_heldout":"2026 weeks 4-8",
        "train_selected_strength":selected,
        "variants":stats,
        "heldout_mean_mae_improved":chosen["sim_mean_mae"]<baseline["sim_mean_mae"],
        "heldout_p90_coverage_nearer90":abs(chosen["simulated_p90_actual_coverage_pct"]-90)<
                                     abs(baseline["simulated_p90_actual_coverage_pct"]-90),
        "note":"Shadow-only experiment. Other-bucket capacity enforced by caught passes, not actual other receiver identities. Prior-player appearance selection; not full DFS profitability."
    }


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
