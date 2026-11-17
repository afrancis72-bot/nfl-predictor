"""Research A/B: internally simulate additional observed pregame offensive teammates.

Both modes return the IDENTICAL DK-eligible roster. Only candidate models
additional RB/WR/TE as separate football players and audits their modeled
stats; never promotes them to the optimizer or claims verified active status.

All inspected 2025/26 games are exploratory, not unseen deployment holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate
from scripts.audit_nfl_team_pass_prior_ab import (
    WINDOWS,PLAYER_KEYS,WR_KEYS,TEAM_KEYS,exact_paired,errors,
    by_groups,bootstrap_game_ci,
)

def run(weekly,schedules,snaps,*,windows=WINDOWS,n_sims=220):
    result={
        "name":"Football player universe independent from DK player pool, first research integration",
        "candidate":"Model historically involved pregame RB/WR/TE omitted by DK selection; project only DK eligible players",
        "control":"Existing DK-player-pool-only team world",
        "team_pass_env_weight":0.5,
        "other_yard_strength":0.0,
        "wr_role_strength":0.0,
        "common_team_random_streams_in_both_modes":True,
        "production_approved":False,
        "one_click_modified":False,
        "independent_holdout":False,
        "status":"EXPLORATORY",
        "windows":{},
        "limitations":[
            "Game-day ACTIVE/OUT status of internal historical players is NOT verified",
            "Missing salary/depth players may have been INACTIVE historically",
            "Only same-season prior 3 weeks, >=2 observed games, 0.5 touches/game included",
            "2025 and 2026 inspected retrospective windows are not untouched evaluation",
            "Historical DK sample generated from past roles, not actual dated salary files",
            "Only RB/WR/TE extra teammate roles are modeled this phase",
            "Research-only common per-team RNG schedule differs from production",
            "Partial DK score excludes fumbles/returns and lacks historical injury availability",
            "220 per-game scenarios; full 10k-sim and future validation required",
        ]
    }
    files={}
    for label,(season,weeks) in windows.items():
        game_data={}
        for key,full in (("control",False),("candidate",True)):
            wr=[];team=[];ghost=[]
            scored,failures=evaluate(weekly,schedules,snaps,season,weeks,
                games_per_week=None,n_sims=n_sims,
                full_player_universe=full,
                full_universe_rows=ghost,
                team_pass_env_weight=.5,
                wr_role_strength=0.,
                other_yard_strength=0.,
                wr_role_team_rng_isolated=True,
                wr_role_rows=wr,team_pass_rows=team)
            if failures or scored.empty:
                raise ValueError(f"{label}/{key} failed {failures[:6]}")
            game_data[key]={
                "scores":scored,
                "wr":pd.DataFrame(wr),
                "team":pd.DataFrame(team),
                "ghost":pd.DataFrame(ghost),
            }
        a=game_data["control"];b=game_data["candidate"]
        p=exact_paired(a["scores"],b["scores"],PLAYER_KEYS,
            ["actual_partial_dk","history_mean_partial_dk","opponent"])
        w=exact_paired(a["wr"],b["wr"],WR_KEYS,
            ["actual_targets","role","rank","prior_targets_pg","opponent"])
        t=exact_paired(a["team"],b["team"],TEAM_KEYS,
            ["actual_pass_attempts","historical_team_pass_baseline","passing_qb"])
        # The SAME team-level simulation draw path was used; allocation cannot
        # change total attempts, play volume, or induce targets from nowhere.
        if not np.allclose(t.model_pass_attempts_base,
                           t.model_pass_attempts_candidate,rtol=0,atol=1e-8):
            mismatch=t.loc[~np.isclose(t.model_pass_attempts_base,
                t.model_pass_attempts_candidate,rtol=0,atol=1e-8)].copy()
            mismatch["delta"]=mismatch.model_pass_attempts_candidate-mismatch.model_pass_attempts_base
            cols=["season","week","game","team","passing_qb_base",
                  "passing_qb_candidate","model_pass_attempts_base",
                  "model_pass_attempts_candidate",
                  "sim_dropbacks_mean_base","sim_dropbacks_mean_candidate",
                  "sim_sacks_mean_base","sim_sacks_mean_candidate",
                  "sim_plays_mean_base","sim_plays_mean_candidate",
                  "pregame_expected_dropbacks_base","pregame_expected_dropbacks_candidate",
                  "pregame_sack_rate_base","pregame_sack_rate_candidate",
                  "pregame_expected_plays_base","pregame_expected_plays_candidate","delta"]
            raise ValueError("Full player-universe candidate changed team pass-attempt totals: "+
                             f"n={len(mismatch)} max_delta={float(mismatch.delta.abs().max()):.4f} "+
                             f"examples={mismatch[cols].head(8).to_dict('records')}")
        ghosts=b["ghost"]
        if ghosts.empty or ghosts["eligible_in_original_DK_pool"].any():
            raise ValueError("Missing or DK-selectable internal historical teammates")
        counts={
            "player_games":len(p),"WR_player_games":len(w),
            "team_games":len(t),"non_DK_internal_player_games":len(ghosts),
            "non_DK_WR_games":int(ghosts.position.eq("WR").sum()),
            "non_DK_RB_games":int(ghosts.position.eq("RB").sum()),
            "non_DK_TE_games":int(ghosts.position.eq("TE").sum()),
            "internal_players_with_actual_boxscore":int(ghosts.actual_boxscore_present.sum()),
            "internal_players_without_actual_boxscore":int((~ghosts.actual_boxscore_present).sum()),
            "internal_WR3_and_other_targets_are_not_proven_active":True,
        }
        metrics={
            "n":counts,
            "all_eligible_DK_players_unchanged":True,
            "passing_attempts_unchanged":True,
            "player_partial_DK":errors(p,"actual_partial_dk_base",
                                         "simulated_mean_dk_base","simulated_mean_dk_candidate"),
            "player_DK_by_position":by_groups(p,"position","actual_partial_dk_base",
                                         "simulated_mean_dk_base","simulated_mean_dk_candidate"),
            "WR_targets":errors(w,"actual_targets_base",
                                  "predicted_targets_base","predicted_targets_candidate"),
            "WR_targets_by_pregame_role":by_groups(w,"role_base","actual_targets_base",
                                  "predicted_targets_base","predicted_targets_candidate"),
            "player_MAE_lift_95ci":bootstrap_game_ci(p,"actual_partial_dk_base",
                                  "simulated_mean_dk_base","simulated_mean_dk_candidate"),
            "WR_target_MAE_lift_95ci":bootstrap_game_ci(w,"actual_targets_base",
                                  "predicted_targets_base","predicted_targets_candidate"),
            "P90_coverage_pct":{
                "DK_base":round(float((p.actual_partial_dk_base<=p.simulated_p90_dk_base).mean()*100),2),
                "DK_candidate":round(float((p.actual_partial_dk_base<=p.simulated_p90_dk_candidate).mean()*100),2),
                "WR_target_base":round(float((w.actual_targets_base<=w.predicted_p90_base).mean()*100),2),
                "WR_target_candidate":round(float((w.actual_targets_base<=w.predicted_p90_candidate).mean()*100),2),
            },
        }
        if len(t)<40 or len(p)<150:
            raise ValueError("Incomplete season retrospective")
        result["windows"][label]=metrics
        files[label+"_dk_players"]=p
        files[label+"_WR_targets"]=w
        files[label+"_team_attempts"]=t
        files[label+"_internally_simulated_non_DK_players"]=ghosts
        print(label,json.dumps(metrics,sort_keys=True),flush=True)
    return result,files

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--n-sims",type=int,default=220)
    p.add_argument("--out-dir",default="nfl_full_player_universe")
    a=p.parse_args(argv)
    if not 100<=a.n_sims<=10000:
        p.error("n-sims must be 100..10000")
    weekly=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report,files=run(weekly,schedules,snaps,n_sims=a.n_sims)
    output=Path(a.out_dir)
    output.mkdir(parents=True,exist_ok=True)
    for key,df in files.items():
        df.to_csv(output/(key+".csv"),index=False)
    (output/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print("Research full-universe A/B complete; live DK application unchanged",flush=True)

if __name__=="__main__":
    main()
