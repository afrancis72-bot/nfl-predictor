"""Research integration check: pregame status gate cannot fabricate 2026 availability.

When the source snapshot dictionary is empty, results MUST equal the
original salary-pool-only game on the same pregame historical inputs.
Actual target-week stats only grade this check; they never label ACTIVE.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import evaluate,GAMES_SOURCE
from scripts.audit_nfl_team_pass_prior_ab import exact_paired,PLAYER_KEYS,WR_KEYS


def run():
    weekly=fetch_weekly([2025,2026])
    snaps=fetch_snap_counts([2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    kwargs=dict(season=2026,weeks=[8],games_per_week=2,n_sims=180,
                full_universe_fix_team_aliases=True,
                wr_role_team_rng_isolated=True)
    original_wr=[]
    original,issues0=evaluate(weekly,schedules,snaps,
                    wr_role_rows=original_wr,**kwargs)
    gated_wr=[];history=[]
    gated,issues1=evaluate(weekly,schedules,snaps,
                   wr_role_rows=gated_wr,
                   availability_gate=True,
                   availability_snapshots={},availability_gate_rows=history,
                   full_player_universe=True,**kwargs)
    if issues0 or issues1:
        raise ValueError("Historical games incomplete: "+repr((issues0,issues1)))
    players=exact_paired(original,gated,PLAYER_KEYS,
        ["actual_partial_dk","history_mean_partial_dk","opponent"])
    wr=exact_paired(original_wr,gated_wr,WR_KEYS,
        ["actual_targets","role","rank","prior_targets_pg","opponent"])
    for col in ("simulated_mean_dk","simulated_p90_dk"):
        if not np.array_equal(players[col+"_base"],players[col+"_candidate"]):
            raise ValueError("Missing pregame availability changed fantasy projection "+col)
    for col in ("predicted_targets","predicted_p90"):
        if not np.array_equal(wr[col+"_base"],wr[col+"_candidate"]):
            raise ValueError("Missing availability evidence changed WR target "+col)
    if not history or any(r["source_snapshot_present"] or r["internal_players_admitted"]
                           for r in history):
        raise ValueError("No-evidence gate incorrectly admitted historical teammates")
    return {
        "window":"2026 week 8, 2 Sunday games, 180 simulations/game",
        "independent_holdout":False,
        "predicted_player_games":len(players),
        "predicted_wr_games":len(wr),
        "source_snapshots_used":0,
        "historical_target_game_status_inferred":False,
        "unverified_teammates_simulated":0,
        "same_player_dk_means_and_p90_bitwise":True,
        "same_receiver_target_means_and_p90_bitwise":True,
        "production_approved":False,
        "snapshots_needed_for_actual_new_opportunity_inferences":True,
        "coverage_ledger":history,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="nfl_pregame_availability_gate/historical_no_source_control.json")
    args=ap.parse_args()
    r=run()
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(r,indent=2,allow_nan=False))
    print(json.dumps(r,sort_keys=True),flush=True)
    print("No-source historical control passed; NOT actual pregame status validation")


if __name__=="__main__":
    main()
